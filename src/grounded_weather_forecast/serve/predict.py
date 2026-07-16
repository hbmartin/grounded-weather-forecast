"""Emit the three products for right now.

The as-of snapshot machinery is the same code the dataset build uses, with a
single snapshot at ``now``: each source contributes its latest forecast fetched
at or before now and not older than the staleness cap. Blenders are fitted on
the live matrix's scoreable history and applied to the future rows of that
snapshot, so serving and backtesting run through one code path.

The minutely product is the anchored nowcast: the hourly path interpolated to
minutes, plus the current observation's residual against the blend, decayed
with lead. Providers' native minutely precipitation is blended separately,
because it is the only genuinely minute-resolution signal they publish.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from grounded_weather_forecast.backtest.splits import (
    daily_truth_known_at,
    hourly_truth_known_at,
)
from grounded_weather_forecast.blenders.registry import get_factory
from grounded_weather_forecast.config import Config
from grounded_weather_forecast.contracts import (
    DAILY_VARIABLES,
    HOURLY_VARIABLES,
    BlendResult,
    ForecastMatrix,
    TruthSemantics,
    VariableSpec,
    hourly_variable,
)
from grounded_weather_forecast.dataset.matrix import (
    build_daily_matrix,
    build_hourly_matrix,
    build_observation_features,
    build_truth,
    matrix_path,
    matrix_sources,
    to_forecast_matrix,
    to_supervised_slice,
)
from grounded_weather_forecast.dataset.providers import read_forecast_archive
from grounded_weather_forecast.evaluation import dataset_fingerprint
from grounded_weather_forecast.leads import daily_bucket, hourly_bucket
from grounded_weather_forecast.serve.history import load_archived_forecast
from grounded_weather_forecast.serve.schema import (
    SCHEMA_VERSION,
    DailyPoint,
    Forecast,
    HourlyPoint,
    MinutelyPoint,
)
from grounded_weather_forecast.serve.selection import (
    Selection,
    SelectionMap,
    method_for,
)

MINUTELY_HORIZON_MINUTES = 60
HOURLY_HORIZON_HOURS = 48
DAILY_HORIZON_DAYS = 10
_OBS_STALENESS = timedelta(minutes=30)
_MINUTELY_VARIABLES = (
    "temp_c",
    "humidity_pct",
    "dew_point_c",
    "wind_speed_ms",
)


class NoForecastDataError(RuntimeError):
    """No provider forecast is fresh enough to serve from."""


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Everything needed to serve one issue time."""

    issue_time: datetime
    hourly: pl.DataFrame
    daily: pl.DataFrame
    minutely: pl.DataFrame
    observation: dict[str, float]
    observation_at: datetime | None


@dataclass(frozen=True, slots=True)
class VariableBlend:
    point: np.ndarray
    methods: list[str]
    reasons: list[str]
    release_ids: list[str | None]
    quantiles: list[dict[str, float]]


def _latest_observation(
    minute: pl.DataFrame, issue_time: datetime
) -> tuple[dict[str, float], datetime | None]:
    if minute.is_empty():
        return {}, None
    recent = minute.filter(
        (pl.col("ts") <= issue_time) & (pl.col("ts") > issue_time - _OBS_STALENESS)
    ).sort("ts")
    if recent.is_empty():
        return {}, None
    row = recent.row(recent.height - 1, named=True)
    values = {
        key: float(value)
        for key, value in row.items()
        if key != "ts" and isinstance(value, float) and not math.isnan(value)
    }
    return values, row["ts"]


def build_snapshot(config: Config, issue_time: datetime) -> Snapshot:
    """The as-of view of every source at ``issue_time``, plus current truth."""
    _, hourly_truth, daily_truth = build_truth(config)
    causal_minute = build_observation_features(config)
    archive = read_forecast_archive(config.forecasts)
    snapshots = pl.DataFrame(
        {"issue_time": [issue_time]},
        schema={"issue_time": pl.Datetime("us", "UTC")},
    )
    hourly = build_hourly_matrix(
        archive.hourly, snapshots, hourly_truth, causal_minute, config
    ).filter(pl.col("lead_hours") <= HOURLY_HORIZON_HOURS)
    daily = build_daily_matrix(
        archive.daily, snapshots, hourly, daily_truth, config
    ).filter(pl.col("lead_days") < DAILY_HORIZON_DAYS)
    minutely = archive.minutely.filter(
        (pl.col("fetched_at") <= issue_time)
        & (
            pl.col("fetched_at")
            > issue_time - timedelta(hours=config.forecasts.max_forecast_age_hours)
        )
        & (pl.col("valid_time") > issue_time)
        & (
            pl.col("valid_time")
            <= issue_time + timedelta(minutes=MINUTELY_HORIZON_MINUTES)
        )
    )
    observation, observation_at = _latest_observation(causal_minute, issue_time)
    if hourly.is_empty() and daily.is_empty():
        msg = (
            f"no provider forecast within {config.forecasts.max_forecast_age_hours}h "
            f"of {issue_time.isoformat()}"
        )
        raise NoForecastDataError(msg)
    return Snapshot(
        issue_time=issue_time,
        hourly=hourly,
        daily=daily,
        minutely=minutely,
        observation=observation,
        observation_at=observation_at,
    )


def _fit_methods(
    train: pl.DataFrame,
    predict_frame: pl.DataFrame,
    variable: VariableSpec,
    method_ids: set[str],
    *,
    daily: bool,
    semantics: TruthSemantics,
) -> tuple[dict[str, BlendResult], ForecastMatrix] | None:
    """Fit each needed method on history and predict the snapshot's rows."""
    truth_sources = matrix_sources(train)
    if not truth_sources:
        return None
    slice_ = to_supervised_slice(train, variable, daily=daily, semantics=semantics)
    if slice_.x.n_rows == 0:
        return None
    x = to_forecast_matrix(predict_frame, variable, daily=daily, sources=truth_sources)
    results: dict[str, BlendResult] = {}
    for method_id in sorted(method_ids):
        blender = get_factory(method_id)().fit(slice_)
        results[method_id] = blender.predict(x)
    return results, x


def _cold_start_equal_weight(
    predict_frame: pl.DataFrame,
    variable: VariableSpec,
    *,
    daily: bool,
) -> np.ndarray:
    sources = matrix_sources(predict_frame)
    if not sources:
        return np.full(predict_frame.height, np.nan)
    matrix = to_forecast_matrix(predict_frame, variable, daily=daily, sources=sources)
    with np.errstate(invalid="ignore"):
        counts = matrix.availability.sum(axis=1)
        totals = np.where(matrix.availability, matrix.values, 0.0).sum(axis=1)
        return np.where(counts > 0, totals / counts, np.nan)


def _blend_variable(
    train: pl.DataFrame,
    predict_frame: pl.DataFrame,
    variable: VariableSpec,
    selections: SelectionMap,
    config: Config,
    *,
    daily: bool,
    semantics: TruthSemantics,
    force_method: str | None = None,
) -> VariableBlend | None:
    """Per row: the prediction of the method selected for that row's bucket."""
    product = "daily" if daily else "hourly"
    buckets: list[str | None] = [
        daily_bucket(lead) if daily else hourly_bucket(lead)
        for lead in (
            predict_frame["lead_days"] if daily else predict_frame["lead_hours"]
        ).to_list()
    ]
    chosen: list[Selection] = [
        Selection(force_method, reason="forced by --method")
        if force_method
        else method_for(selections, product, variable.name, bucket, config)
        for bucket in buckets
    ]
    fitted = _fit_methods(
        train,
        predict_frame,
        variable,
        {c.method_id for c in chosen},
        daily=daily,
        semantics=semantics,
    )
    if fitted is None:
        if {selection.method_id for selection in chosen} != {"equal_weight"}:
            return None
        point = _cold_start_equal_weight(predict_frame, variable, daily=daily)
        return VariableBlend(
            point=point,
            methods=["equal_weight"] * predict_frame.height,
            reasons=["degraded cold start: no scoreable training truth"]
            * predict_frame.height,
            release_ids=[None] * predict_frame.height,
            quantiles=[{} for _ in range(predict_frame.height)],
        )
    results, _ = fitted
    point = np.full(predict_frame.height, np.nan)
    quantiles: list[dict[str, float]] = []
    for row, selection in enumerate(chosen):
        result = results[selection.method_id]
        point[row] = result.point[row]
        quantiles.append(
            {
                str(level): float(result.quantiles[row, index])
                for index, level in enumerate(result.quantile_levels)
            }
            if result.quantiles is not None
            else {}
        )
    return VariableBlend(
        point=point,
        methods=[selection.method_id for selection in chosen],
        reasons=[selection.reason for selection in chosen],
        release_ids=[selection.release_id for selection in chosen],
        quantiles=quantiles,
    )


def _finite(value: float, variable: VariableSpec | None = None) -> float | None:
    if not math.isfinite(value):
        return None
    bounded = float(value)
    if variable is not None:
        if variable.minimum is not None:
            bounded = max(bounded, variable.minimum)
        if variable.maximum is not None:
            bounded = min(bounded, variable.maximum)
    return round(bounded, 3)


def _cohere_hourly(points: list[HourlyPoint]) -> None:
    """Enforce cross-variable physical relationships after marginal clipping."""
    for point in points:
        values = point.values
        temperature = values.get("temp_c")
        dew_point = values.get("dew_point_c")
        if temperature is not None and dew_point is not None:
            values["dew_point_c"] = min(dew_point, temperature)
        speed = values.get("wind_speed_ms")
        gust = values.get("wind_gust_ms")
        if speed is not None and gust is not None:
            values["wind_gust_ms"] = max(gust, speed)


def _cohere_daily(points: list[DailyPoint]) -> None:
    for point in points:
        high = point.values.get("temp_max_c")
        low = point.values.get("temp_min_c")
        if high is not None and low is not None and high < low:
            point.values["temp_max_c"], point.values["temp_min_c"] = low, high


def hourly_product(
    snapshot: Snapshot,
    train: pl.DataFrame,
    selections: SelectionMap,
    config: Config,
    semantics: TruthSemantics | Mapping[str, TruthSemantics],
    force_method: str | None = None,
) -> tuple[list[HourlyPoint], dict[str, np.ndarray]]:
    """Blended hourly path, plus the raw per-variable arrays for anchoring."""
    frame = snapshot.hourly.sort("valid_time")
    blended: dict[str, VariableBlend] = {}
    for variable in HOURLY_VARIABLES:
        variable_semantics = (
            semantics.get(variable.name, TruthSemantics.INSTANTANEOUS)
            if isinstance(semantics, Mapping)
            else semantics
        )
        result = _blend_variable(
            train,
            frame,
            variable,
            selections,
            config,
            daily=False,
            semantics=variable_semantics,
            force_method=force_method,
        )
        if result is None:
            continue
        blended[variable.name] = result
    variables = {variable.name: variable for variable in HOURLY_VARIABLES}
    points = [
        HourlyPoint(
            valid_time=row["valid_time"].isoformat(),
            lead_hours=round(float(row["lead_hours"]), 2),
            lead_bucket=row["lead_bucket"],
            values={
                name: _finite(result.point[index], variables[name])
                for name, result in blended.items()
            },
            methods={name: result.methods[index] for name, result in blended.items()},
            quantiles={
                name: {
                    level: value
                    for level, raw in result.quantiles[index].items()
                    if (value := _finite(raw, variables[name])) is not None
                }
                for name, result in blended.items()
                if result.quantiles[index]
            },
            selection_reasons={
                name: result.reasons[index] for name, result in blended.items()
            },
        )
        for index, row in enumerate(frame.iter_rows(named=True))
    ]
    _cohere_hourly(points)
    return points, {name: result.point for name, result in blended.items()}


def daily_product(
    snapshot: Snapshot,
    train: pl.DataFrame,
    selections: SelectionMap,
    config: Config,
    force_method: str | None = None,
) -> list[DailyPoint]:
    """Daily hi/lo and PoP as their own supervised targets (decision 8).

    The hourly-blend aggregates ride along as ``ewagg__*`` features inside the
    matrix, so a method that can use them (the GBM) does, and one that cannot
    still sees every provider's native daily value.
    """
    frame = snapshot.daily.sort("forecast_date")
    if frame.is_empty():
        return []
    blended: dict[str, VariableBlend] = {}
    for variable in DAILY_VARIABLES:
        result = _blend_variable(
            train,
            frame,
            variable,
            selections,
            config,
            daily=True,
            semantics=TruthSemantics.INSTANTANEOUS,
            force_method=force_method,
        )
        if result is None:
            continue
        blended[variable.name] = result
    variables = {variable.name: variable for variable in DAILY_VARIABLES}
    points = [
        DailyPoint(
            date_local=row["forecast_date"].isoformat(),
            lead_days=int(row["lead_days"]),
            values={
                name: _finite(result.point[index], variables[name])
                for name, result in blended.items()
            },
            methods={name: result.methods[index] for name, result in blended.items()},
            quantiles={
                name: {
                    level: value
                    for level, raw in result.quantiles[index].items()
                    if (value := _finite(raw, variables[name])) is not None
                }
                for name, result in blended.items()
                if result.quantiles[index]
            },
            selection_reasons={
                name: result.reasons[index] for name, result in blended.items()
            },
        )
        for index, row in enumerate(frame.iter_rows(named=True))
    ]
    _cohere_daily(points)
    return points


def _anchor_weight(minutes_ahead: int, tau_hours: float) -> float:
    weight = math.exp(-(minutes_ahead / 60.0) / tau_hours)
    return 0.0 if weight < 0.05 else weight


def _minutely_precip(snapshot: Snapshot) -> dict[datetime, tuple[float, float]]:
    """Availability-weighted mean of providers' native minutely precipitation."""
    if snapshot.minutely.is_empty():
        return {}
    latest = (
        snapshot.minutely.sort("fetched_at")
        .group_by("source", "valid_time")
        .last()
        .group_by("valid_time")
        .agg(
            pl.col("precip_intensity_mmh").mean().alias("intensity"),
            pl.col("pop").mean().alias("pop"),
        )
    )
    return {
        row["valid_time"]: (row["intensity"], row["pop"])
        for row in latest.iter_rows(named=True)
    }


def minutely_product(
    snapshot: Snapshot,
    hourly_blend: dict[str, np.ndarray],
    config: Config,
) -> list[MinutelyPoint]:
    """The anchored nowcast: hourly path interpolated to minutes + decayed residual."""
    frame = snapshot.hourly.sort("valid_time")
    if frame.is_empty():
        return []
    leads = frame["lead_hours"].to_numpy().astype(float)
    precip = _minutely_precip(snapshot)
    points: list[MinutelyPoint] = []
    for minute in range(1, MINUTELY_HORIZON_MINUTES + 1):
        valid = snapshot.issue_time + timedelta(minutes=minute)
        lead = minute / 60.0
        values: dict[str, float | None] = {}
        for name in _MINUTELY_VARIABLES:
            path = hourly_blend.get(name)
            if path is None or not np.isfinite(path).any():
                continue
            usable = np.isfinite(path)
            interpolated = float(np.interp(lead, leads[usable], path[usable]))
            observed = snapshot.observation.get(name)
            if observed is not None:
                now_blend = float(np.interp(0.0, leads[usable], path[usable]))
                residual = observed - now_blend
                interpolated += (
                    _anchor_weight(minute, config.predict.minutely_tau_hours) * residual
                )
            values[name] = _finite(interpolated, hourly_variable(name))
        intensity, pop = precip.get(valid, (None, None))
        finite_intensity = (
            _finite(
                float(intensity),
                VariableSpec(
                    "precip_intensity_mmh",
                    HOURLY_VARIABLES[0].kind,
                    "mm/h",
                    minimum=0.0,
                ),
            )
            if intensity is not None
            else None
        )
        finite_pop = (
            _finite(float(pop), hourly_variable("pop")) if pop is not None else None
        )
        points.append(
            MinutelyPoint(
                valid_time=valid.isoformat(),
                minutes_ahead=minute,
                temp_c=values.get("temp_c"),
                humidity_pct=values.get("humidity_pct"),
                dew_point_c=values.get("dew_point_c"),
                wind_speed_ms=values.get("wind_speed_ms"),
                precip_intensity_mmh=finite_intensity,
                pop=finite_pop,
                methods={
                    **dict.fromkeys(values, "anchored_hourly_blend"),
                    **(
                        {
                            "precip_intensity_mmh": "native_equal_weight",
                            "pop": "native_equal_weight",
                        }
                        if intensity is not None or pop is not None
                        else {}
                    ),
                },
            )
        )
    return points


def _training_matrix(
    config: Config, product: str, issue_time: datetime
) -> pl.DataFrame:
    path = matrix_path(config.dataset.dir, product, "live")
    if not path.exists():
        return pl.DataFrame()
    frame = pl.read_parquet(path)
    if frame.is_empty():
        return frame
    known_at = (
        daily_truth_known_at(frame, config.station.timezone)
        if product == "daily"
        else hourly_truth_known_at(frame)
    )
    return frame.with_columns(known_at).filter(
        (pl.col("issue_time") <= issue_time) & (pl.col("truth_known_at") <= issue_time)
    )


def predict(
    config: Config,
    selections: SelectionMap,
    now: datetime | None = None,
    semantics: TruthSemantics | Mapping[str, TruthSemantics] = (
        TruthSemantics.INSTANTANEOUS
    ),
    force_method: str | None = None,
) -> Forecast:
    """Assemble the whole forecast document for one issue time."""
    issue_time = (now or datetime.now(tz=UTC)).replace(second=0, microsecond=0)
    if now is not None:
        archived = load_archived_forecast(
            config.predict.history_path, issue_time.isoformat()
        )
        if archived is not None:
            return archived
    snapshot = build_snapshot(config, issue_time)
    hourly_train = _training_matrix(config, "hourly", issue_time)
    daily_train = _training_matrix(config, "daily", issue_time)
    hourly, hourly_blend = hourly_product(
        snapshot, hourly_train, selections, config, semantics, force_method
    )
    daily = (
        daily_product(snapshot, daily_train, selections, config, force_method)
        if not snapshot.daily.is_empty()
        else []
    )
    release_ids = sorted(
        {
            selection.release_id
            for selection in selections.values()
            if selection.release_id is not None
        }
    )
    degraded = not release_ids and force_method is None
    return Forecast(
        schema_version=SCHEMA_VERSION,
        issued_at=issue_time.isoformat(),
        latitude=config.station.latitude,
        longitude=config.station.longitude,
        dataset_fingerprint=dataset_fingerprint(config),
        sources=list(matrix_sources(snapshot.hourly)),
        observation_at=snapshot.observation_at.isoformat()
        if snapshot.observation_at
        else None,
        minutely=minutely_product(snapshot, hourly_blend, config),
        hourly=hourly,
        daily=daily,
        timezone=config.station.timezone,
        status="degraded" if degraded else "ready",
        release_ids=release_ids,
    )
