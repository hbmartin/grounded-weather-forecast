"""Open-Meteo Ensemble API ingestion: honest spread as feature columns.

Provider columns are structurally under-dispersed — most repackage the same
global parents — so their cross-source spread is not an uncertainty signal.
Real NWP and ML ensembles are. This module polls the Ensemble API for the
configured models (e.g. WeatherNext 2, AIFS-ENS, AIGEFS, GEFS), reduces the
members to per-(model, valid_time, variable) statistics, and appends them to
a dedicated parquet store. The matrix build as-of joins the statistics into
``ens__{model}__{variable}__{stat}`` feature columns.

Ensembles are **features, not sources**: they inform dispersion (EMOS spread
links, GBM features, interval width) without entering the grounding/weighting
source set, so the effective-ensemble-size accounting of the blend stays
honest.

Open-Meteo retains only the latest run's individual members, so the ingest
cron must run every model cycle; only reduced statistics are stored, never
raw members. The stored ``fetched_at`` is the retrieval instant — the same
as-of eligibility semantics as the provider archive.
"""

import urllib.parse
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from grounded_weather_forecast.config import Config
from grounded_weather_forecast.dataset.backfill import (
    BACKFILL_VARIABLES,
    MAX_PREVIOUS_DAYS,
    PREVIOUS_RUNS_URL,
    Fetcher,
    _chunks,
    http_fetcher,
)
from grounded_weather_forecast.dataset.snapshots import snapshot_long
from grounded_weather_forecast.storage import atomic_write_parquet, locked_path

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
# Open-Meteo archives per-run ensemble mean and spread (std dev of members)
# as pseudo-models on the Previous Runs API, e.g. `ncep_gefs025_ensemble_mean`
# with `temperature_2m_previous_day2` / `temperature_2m_spread_previous_day2`
# fields — verified live 2026-08-04. Members themselves are retained only a
# few days; the statistics reach back to ~March 2026.
ENSEMBLE_MEAN_SUFFIX = "_ensemble_mean"
FEATURE_PREFIX = "ens__"
STAT_COLUMNS: tuple[str, ...] = ("mean", "sd", "p10", "p25", "p50", "p75", "p90")
_QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
_OPEN_METEO_NAME = {canonical: name for name, canonical in BACKFILL_VARIABLES.items()}
_FORECAST_DAYS = 16
_KEY = ("model", "fetched_at", "valid_time", "variable")

_SCHEMA: dict[str, pl.DataType] = {
    "model": pl.String(),
    "fetched_at": pl.Datetime("us", "UTC"),
    "valid_time": pl.Datetime("us", "UTC"),
    "variable": pl.String(),
    **{stat: pl.Float64() for stat in STAT_COLUMNS},
    "n_members": pl.Int32(),
}


class EnsembleError(RuntimeError):
    """The Ensemble API response was missing or malformed."""


def ensembles_path(config: Config) -> Path:
    return config.dataset.dir / "ensembles.parquet"


def build_ensemble_url(config: Config, model: str) -> str:
    """Ensemble API request for one model and the configured variables."""
    fields = [_OPEN_METEO_NAME[v] for v in config.ensembles.variables]
    query = urllib.parse.urlencode(
        {
            "latitude": config.station.latitude,
            "longitude": config.station.longitude,
            "hourly": ",".join(fields),
            "models": model,
            "forecast_days": _FORECAST_DAYS,
            "timezone": "UTC",
            "wind_speed_unit": "ms",
        }
    )
    return f"{ENSEMBLE_URL}?{query}"


def _member_matrix(
    hourly: dict[str, object], field: str, n_times: int
) -> np.ndarray | None:
    """(n_times, n_members) float matrix for one variable; None if absent."""
    columns = [
        values
        for key, values in hourly.items()
        if (key == field or key.startswith(f"{field}_member"))
        and isinstance(values, list)
        and len(values) == n_times
    ]
    if not columns:
        return None
    return np.asarray(
        [
            [float(v) if isinstance(v, (int, float)) else np.nan for v in column]
            for column in columns
        ],
        dtype=np.float64,
    ).T


def _stat_row(members: np.ndarray) -> dict[str, float | int | None]:
    finite = members[np.isfinite(members)]
    if finite.size == 0:
        return {**dict.fromkeys(STAT_COLUMNS), "n_members": 0}
    quantiles = np.quantile(finite, _QUANTILES)
    return {
        "mean": float(finite.mean()),
        "sd": float(finite.std(ddof=1)) if finite.size > 1 else None,
        **{
            f"p{int(level * 100)}": float(q)
            for level, q in zip(_QUANTILES, quantiles, strict=True)
        },
        "n_members": int(finite.size),
    }


def parse_ensemble(
    payload: dict[str, object],
    model: str,
    fetched_at: datetime,
    variables: tuple[str, ...],
) -> pl.DataFrame:
    """Payload -> long statistics frame, one row per (valid_time, variable)."""
    raw_hourly = payload.get("hourly")
    raw_times = raw_hourly.get("time") if isinstance(raw_hourly, dict) else None
    if not isinstance(raw_hourly, dict) or not isinstance(raw_times, list):
        msg = f"ensemble payload for {model!r} has no hourly.time block"
        raise EnsembleError(msg)
    hourly: dict[str, object] = {str(key): value for key, value in raw_hourly.items()}
    times = [
        datetime.fromisoformat(t).replace(tzinfo=UTC)
        for t in raw_times
        if isinstance(t, str)
    ]
    rows: list[dict[str, object]] = []
    for canonical in variables:
        matrix = _member_matrix(hourly, _OPEN_METEO_NAME[canonical], len(times))
        if matrix is None:
            continue
        rows.extend(
            {
                "model": model,
                "fetched_at": fetched_at,
                "valid_time": valid,
                "variable": canonical,
                **_stat_row(matrix[i]),
            }
            for i, valid in enumerate(times)
        )
    if not rows:
        msg = f"ensemble payload for {model!r} contained no requested variables"
        raise EnsembleError(msg)
    return pl.DataFrame(rows, schema=_SCHEMA).sort(list(_KEY))


def ingest_ensembles(
    config: Config,
    fetcher: Fetcher = http_fetcher,
    now: datetime | None = None,
) -> pl.DataFrame:
    """Fetch every configured model once; returns the combined long frame."""
    if not config.ensembles.models:
        msg = "set [ensembles].models to ingest ensemble statistics"
        raise EnsembleError(msg)
    frames: list[pl.DataFrame] = []
    for model in config.ensembles.models:
        payload = dict(fetcher(build_ensemble_url(config, model)))
        fetched_at = (now or datetime.now(tz=UTC)).replace(microsecond=0)
        frames.append(
            parse_ensemble(
                payload,
                model,
                fetched_at,
                config.ensembles.variables,
            )
        )
    return pl.concat(frames).sort(list(_KEY))


def ensemble_mean_model(model: str) -> str:
    """The archived-statistics pseudo-model for an ensemble model slug.

    Slugs that already end in ``_ensemble`` (e.g. ``ecmwf_aifs025_ensemble``)
    replace that suffix rather than doubling it: the API knows
    ``ecmwf_aifs025_ensemble_mean``, not ``…_ensemble_ensemble_mean``.
    """
    return model.removesuffix("_ensemble") + ENSEMBLE_MEAN_SUFFIX


def build_ensemble_backfill_url(
    config: Config,
    model: str,
    start: date,
    end: date,
    previous_days: int = MAX_PREVIOUS_DAYS,
) -> str:
    """Previous Runs request for one ensemble-mean pseudo-model."""
    fields: list[str] = []
    for canonical in config.ensembles.variables:
        name = _OPEN_METEO_NAME[canonical]
        for day in range(1, previous_days + 1):
            fields.append(f"{name}_previous_day{day}")
            fields.append(f"{name}_spread_previous_day{day}")
    query = urllib.parse.urlencode(
        {
            "latitude": config.station.latitude,
            "longitude": config.station.longitude,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(fields),
            "models": ensemble_mean_model(model),
            "timezone": "UTC",
            "wind_speed_unit": "ms",
        }
    )
    return f"{PREVIOUS_RUNS_URL}?{query}"


def parse_ensemble_mean_runs(
    payload: dict[str, object],
    model: str,
    variables: tuple[str, ...],
    previous_days: int = MAX_PREVIOUS_DAYS,
) -> pl.DataFrame:
    """Previous Runs payload -> the same long statistics frame the poller
    writes: each day-offset is a run vintage with ``fetched_at`` set
    ``day`` days before the valid hour, so the as-of join treats a
    backfilled statistic exactly like a poll made at that instant. Only
    ``mean`` and ``sd`` are archived; the percentile columns stay null
    rather than being faked from a normal assumption.
    """
    raw_hourly = payload.get("hourly")
    raw_times = raw_hourly.get("time") if isinstance(raw_hourly, dict) else None
    if not isinstance(raw_hourly, dict) or not isinstance(raw_times, list):
        msg = f"ensemble-mean payload for {model!r} has no hourly.time block"
        raise EnsembleError(msg)
    times = [
        datetime.fromisoformat(t).replace(tzinfo=UTC)
        for t in raw_times
        if isinstance(t, str)
    ]
    rows: list[dict[str, object]] = []
    for canonical in variables:
        name = _OPEN_METEO_NAME[canonical]
        for day in range(1, previous_days + 1):
            means = raw_hourly.get(f"{name}_previous_day{day}")
            spreads = raw_hourly.get(f"{name}_spread_previous_day{day}")
            if not (isinstance(means, list) and len(means) == len(times)):
                continue
            if not (isinstance(spreads, list) and len(spreads) == len(times)):
                spreads = [None] * len(times)
            offset = timedelta(days=day)
            rows.extend(
                {
                    "model": model,
                    "fetched_at": valid - offset,
                    "valid_time": valid,
                    "variable": canonical,
                    **dict.fromkeys(STAT_COLUMNS),
                    "mean": float(mean) if isinstance(mean, (int, float)) else None,
                    "sd": float(spread) if isinstance(spread, (int, float)) else None,
                    "n_members": None,
                }
                for valid, mean, spread in zip(times, means, spreads, strict=True)
                if isinstance(mean, (int, float))
            )
    if not rows:
        msg = f"ensemble-mean payload for {model!r} contained no usable offsets"
        raise EnsembleError(msg)
    return pl.DataFrame(rows, schema=_SCHEMA).sort(list(_KEY))


def backfill_ensembles(
    config: Config,
    start: date,
    end: date,
    fetcher: Fetcher = http_fetcher,
    chunk_days: int = 90,
    models: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Archived ensemble mean/spread over a valid-date range, all models."""
    selected = models or config.ensembles.models
    if not selected:
        msg = "set [ensembles].models to backfill ensemble statistics"
        raise EnsembleError(msg)
    if end < start:
        msg = f"ensemble backfill end {end} precedes start {start}"
        raise EnsembleError(msg)
    frames: list[pl.DataFrame] = []
    for model in selected:
        for span_start, span_end in _chunks(start, end, chunk_days):
            url = build_ensemble_backfill_url(config, model, span_start, span_end)
            frames.append(
                parse_ensemble_mean_runs(
                    dict(fetcher(url)), model, config.ensembles.variables
                )
            )
    return pl.concat(frames).sort(list(_KEY))


def trim_backfill_to_live(frame: pl.DataFrame, existing: pl.DataFrame) -> pl.DataFrame:
    """Drop backfilled vintages that would shadow live-polled statistics.

    The as-of join keeps ONE fetched_at per (issue, model): a later-but-sparse
    archived vintage (one valid hour per day offset) would shadow an earlier
    live poll carrying the full horizon. Backfilled rows (``n_members`` null)
    therefore never cross into a model's live-polled era.
    """
    if existing.is_empty():
        return frame
    live_starts = (
        existing.filter(pl.col("n_members").is_not_null())
        .group_by("model")
        .agg(pl.col("fetched_at").min().alias("live_start"))
    )
    if live_starts.is_empty():
        return frame
    return (
        frame.join(live_starts, on="model", how="left")
        .filter(
            pl.col("live_start").is_null()
            | (pl.col("fetched_at") < pl.col("live_start"))
        )
        .drop("live_start")
    )


def load_ensembles(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema=_SCHEMA)
    return pl.read_parquet(path)


def append_ensembles(path: Path, fresh: pl.DataFrame) -> tuple[int, int]:
    """Append-dedupe onto the parquet store; returns (new_rows, total_rows)."""
    with locked_path(path):
        existing = load_ensembles(path)
        combined = (
            pl.concat([existing, fresh.select(existing.columns)], how="vertical")
            .unique(subset=list(_KEY), keep="first")
            .sort(list(_KEY))
        )
        atomic_write_parquet(combined, path)
        return combined.height - existing.height, combined.height


def ensemble_features(
    ensembles: pl.DataFrame,
    snapshots: pl.DataFrame,
    max_age_hours: float,
) -> pl.DataFrame:
    """As-of ensemble statistics, wide: one ``ens__…`` column per stat.

    Reuses the source as-of machinery with the model slug standing in for the
    source, so a stale ensemble run ages out exactly like a stale provider.
    """
    if ensembles.is_empty() or snapshots.is_empty():
        return pl.DataFrame(
            schema={
                "issue_time": pl.Datetime("us", "UTC"),
                "valid_time": pl.Datetime("us", "UTC"),
            }
        )
    snap = snapshot_long(
        ensembles.rename({"model": "source"}), snapshots, max_age_hours
    )
    if snap.is_empty():
        return pl.DataFrame(
            schema={
                "issue_time": pl.Datetime("us", "UTC"),
                "valid_time": pl.Datetime("us", "UTC"),
            }
        )
    long = snap.unpivot(
        index=["issue_time", "valid_time", "source", "variable"],
        on=list(STAT_COLUMNS),
        variable_name="stat",
    ).with_columns(
        pl.format(
            f"{FEATURE_PREFIX}{{}}__{{}}__{{}}",
            pl.col("source"),
            pl.col("variable"),
            pl.col("stat"),
        ).alias("feature")
    )
    wide = long.pivot(
        on="feature",
        index=["issue_time", "valid_time"],
        values="value",
        aggregate_function="last",
    )
    # Sorted columns keep parquet bytes and the dataset fingerprint
    # deterministic across rebuilds, mirroring the fx pivot's stable sort.
    ordered = sorted(c for c in wide.columns if c.startswith(FEATURE_PREFIX))
    return wide.sort("issue_time", "valid_time").select(
        "issue_time", "valid_time", *ordered
    )
