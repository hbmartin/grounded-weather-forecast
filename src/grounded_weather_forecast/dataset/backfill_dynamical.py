"""dynamical.org synthetic backfill: real sub-24 h leads, at last.

The Previous Runs backfill quantizes leads to whole days, so the 0-24 h
buckets — where the products actually live and anchoring earns its keep —
were entirely unevaluated on synthetic data. dynamical.org publishes free,
keyless, analysis-ready Zarr archives of full forecast cycles at native
3-6-hourly steps (GEFS back to 2020; AIFS-ENS since mid-2025), which fills
exactly that gap. The ensemble mean becomes the source value.

Publication-lag honesty is load-bearing: a cycle initialized at 00Z was not
*available* at 00Z. ``fetched_at = init_time + publication_lag`` (default
6 h), so leads are computed against availability, steps earlier than the lag
are dropped, and synthetic short-lead skill is never inflated by pretending
the archive saw a cycle before it existed.

Heavy dependencies (dynamical-catalog, xarray, zarr/icechunk) live in the
optional ``backfill`` extra and are imported lazily, mirroring the
lightgbm pattern.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from importlib import import_module
from importlib.util import find_spec
from typing import Any

import numpy as np
import polars as pl

from grounded_weather_forecast.config import Config
from grounded_weather_forecast.contracts import SourceKind
from grounded_weather_forecast.dataset.providers import HOURLY_COLUMN_MAP

type DatasetOpener = Callable[[str], Any]

_MAGNUS_A = 17.625
_MAGNUS_B = 243.04
_SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True, slots=True)
class DynamicalDataset:
    source: str
    catalog_id: str
    has_humidity: bool


DYNAMICAL_DATASETS: Mapping[str, DynamicalDataset] = {
    "gefs": DynamicalDataset(
        source="dynamical_gefs",
        catalog_id="noaa-gefs-forecast-35-day",
        has_humidity=True,
    ),
    "aifs_ens": DynamicalDataset(
        source="dynamical_aifs_ens",
        catalog_id="ecmwf-aifs-ens-forecast",
        has_humidity=False,
    ),
}

HAVE_DYNAMICAL = find_spec("dynamical_catalog") is not None


class DynamicalBackfillError(RuntimeError):
    """The dynamical.org dataset was unavailable or malformed."""


def _dynamical_error_types() -> tuple[type[Exception], ...]:
    """The optional catalog's public error base, when the extra is installed."""
    if not HAVE_DYNAMICAL:
        return ()
    module = import_module("dynamical_catalog")
    error_type = getattr(module, "DynamicalCatalogError", None)
    return (error_type,) if isinstance(error_type, type) else ()


def open_catalog_dataset(catalog_id: str) -> Any:  # pragma: no cover - network
    if not HAVE_DYNAMICAL:
        msg = (
            "the dynamical backfill needs the optional dependencies: "
            "uv sync --extra backfill"
        )
        raise DynamicalBackfillError(msg)
    return import_module("dynamical_catalog").open(catalog_id)


def _dew_point_c(temp_c: np.ndarray, humidity_pct: np.ndarray) -> np.ndarray:
    """Magnus-formula dew point from temperature and relative humidity."""
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.log(np.clip(humidity_pct, 1e-3, 100.0) / 100.0) + (
            _MAGNUS_A * temp_c
        ) / (_MAGNUS_B + temp_c)
        return _MAGNUS_B * gamma / (_MAGNUS_A - gamma)


def _point_selection(dataset: Any, latitude: float, longitude: float) -> Any:
    """Nearest grid cell, tolerating 0..360 longitude conventions."""
    lon_values = np.asarray(dataset["longitude"].values, dtype=np.float64)
    request = longitude % 360.0 if float(lon_values.max()) > 180.0 else longitude
    return dataset.sel(latitude=latitude, longitude=request, method="nearest")


def _member_mean(window: Any) -> Any:
    if "ensemble_member" in window.dims:
        return window.mean("ensemble_member", skipna=True)
    return window


def _canonical_columns(
    point: Any, n_init: int, n_lead: int, *, has_humidity: bool
) -> dict[str, np.ndarray]:
    """Flattened (init x lead) canonical variable arrays from a point dataset."""
    shape = (n_init * n_lead,)

    def values(name: str) -> np.ndarray:
        if name not in point:
            return np.full(shape, np.nan)
        return np.asarray(point[name].values, dtype=np.float64).reshape(shape)

    temp = values("temperature_2m")
    humidity = (
        values("relative_humidity_2m") if has_humidity else np.full(shape, np.nan)
    )
    wind_u, wind_v = values("wind_u_10m"), values("wind_v_10m")
    columns = {
        "temp_c": temp,
        "humidity_pct": humidity,
        "dew_point_c": _dew_point_c(temp, humidity)
        if has_humidity
        else np.full(shape, np.nan),
        "wind_speed_ms": np.hypot(wind_u, wind_v),
        "pressure_sea_hpa": values("pressure_reduced_to_mean_sea_level") / 100.0,
    }
    # Deliberately absent: gusts (not in these stores), precipitation (step
    # accumulations would double-count in hourly aggregation), and PoP.
    missing = [
        canonical
        for canonical in HOURLY_COLUMN_MAP.values()
        if canonical not in columns
    ]
    columns |= {name: np.full(shape, np.nan) for name in missing}
    return columns


def _long_frame(
    point: Any, spec: DynamicalDataset, publication_lag: timedelta
) -> pl.DataFrame:
    init_times = np.asarray(point["init_time"].values)
    lead_times = np.asarray(point["lead_time"].values)
    n_init, n_lead = init_times.shape[0], lead_times.shape[0]
    init_grid = np.repeat(init_times, n_lead)
    lead_grid = np.tile(lead_times, n_init)
    valid_times = init_grid + lead_grid
    fetched = init_grid + np.timedelta64(int(publication_lag.total_seconds()), "s")
    columns = _canonical_columns(point, n_init, n_lead, has_humidity=spec.has_humidity)
    frame = pl.DataFrame(
        {
            "fetched_at": fetched,
            "valid_time": valid_times,
            **dict(columns),
        }
    ).with_columns(
        pl.col("fetched_at").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
        pl.col("valid_time").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
        *(pl.col(name).cast(pl.Float64).fill_nan(None) for name in columns),
    )
    return (
        frame.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("run_id"),
            pl.lit(spec.source).alias("source"),
            pl.lit(SourceKind.SYNTHETIC.value).alias("source_kind"),
            (
                (pl.col("valid_time") - pl.col("fetched_at")).dt.total_seconds()
                / _SECONDS_PER_HOUR
            ).alias("lead_hours"),
        )
        .filter(pl.col("lead_hours") >= 0.0)
        .sort("source", "fetched_at", "valid_time")
        .select(
            "run_id",
            "source",
            "source_kind",
            "fetched_at",
            "valid_time",
            "lead_hours",
            *HOURLY_COLUMN_MAP.values(),
        )
    )


def _backfill_failure(
    model: str, spec: DynamicalDataset, exc: Exception
) -> DynamicalBackfillError:
    msg = (
        f"dynamical backfill failed for {model!r} "
        f"({spec.catalog_id}): {type(exc).__name__}: {exc}"
    )
    return DynamicalBackfillError(msg)


def backfill_dynamical_long(
    config: Config,
    start: date,
    end: date,
    models: Sequence[str] | None = None,
    opener: DatasetOpener = open_catalog_dataset,
) -> pl.DataFrame:
    """Extract full forecast cycles at the station point as a synthetic long frame."""
    selected = tuple(models) if models else config.backfill.dynamical_models
    unknown = sorted(set(selected) - set(DYNAMICAL_DATASETS))
    if unknown:
        msg = f"unknown dynamical models {unknown}; known: {sorted(DYNAMICAL_DATASETS)}"
        raise DynamicalBackfillError(msg)
    if not selected:
        msg = "set [backfill.dynamical].models to backfill"
        raise DynamicalBackfillError(msg)
    if end < start:
        msg = f"backfill end {end} precedes start {start}"
        raise DynamicalBackfillError(msg)
    lag = timedelta(hours=config.backfill.dynamical_publication_lag_hours)
    max_lead = timedelta(hours=config.backfill.dynamical_max_lead_hours) + lag
    start_instant = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    end_instant = datetime(end.year, end.month, end.day, 23, 59, tzinfo=timezone.utc)
    frames: list[pl.DataFrame] = []
    catalog_errors = _dynamical_error_types()
    for model in selected:
        spec = DYNAMICAL_DATASETS[model]
        try:
            dataset = opener(spec.catalog_id)
        except (DynamicalBackfillError,):  # noqa: B013 - project tuple convention
            raise
        # These built-ins come directly from the optional catalog boundary.
        # Do not catch them around our extraction helpers: the same classes are
        # also raised by ordinary programming errors there.
        except (ValueError, LookupError, TypeError, OSError, *catalog_errors) as exc:
            raise _backfill_failure(model, spec, exc) from exc
        try:
            point = _point_selection(
                dataset, config.station.latitude, config.station.longitude
            )
            window = point.sel(
                init_time=slice(
                    np.datetime64(start_instant.replace(tzinfo=None)),
                    np.datetime64(end_instant.replace(tzinfo=None)),
                ),
                lead_time=slice(np.timedelta64(0, "s"), max_lead),
            )
            if window["init_time"].shape[0] == 0:
                continue
            frames.append(_long_frame(_member_mean(window), spec, lag))
        except (DynamicalBackfillError,):  # noqa: B013 - project tuple convention
            raise
        # Lazy reads can raise the catalog's public error hierarchy or storage
        # errors after open(). Normalize only those unambiguous dependency
        # failures; built-in TypeError/ValueError/LookupError must stay visible.
        except (OSError, *catalog_errors) as exc:
            raise _backfill_failure(model, spec, exc) from exc
    if not frames:
        msg = "no dynamical cycles found in the requested window"
        raise DynamicalBackfillError(msg)
    return pl.concat(frames).sort("source", "fetched_at", "valid_time")
