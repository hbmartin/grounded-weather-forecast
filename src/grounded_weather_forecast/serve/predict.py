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
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from grounded_weather_forecast.backtest.splits import (
    daily_truth_known_at,
    hourly_truth_known_at,
)
from grounded_weather_forecast.blenders.registry import (
    UnknownMethodError,
    get_factory,
    supports_product,
)
from grounded_weather_forecast.config import Config
from grounded_weather_forecast.contracts import (
    DAILY_VARIABLES,
    HOURLY_VARIABLES,
    Blender,
    BlendResult,
    ForecastMatrix,
    Product,
    SupervisedSlice,
    TruthSemantics,
    VariableSpec,
    daily_variable,
    hourly_variable,
)
from grounded_weather_forecast.dataset.ensembles import (
    ensembles_path,
    load_ensembles,
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
from grounded_weather_forecast.serve.observability import snapshot_observability
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
    no_evidence_reason,
)

MINUTELY_HORIZON_MINUTES = 60
HOURLY_HORIZON_HOURS = 48
DAILY_HORIZON_DAYS = 10
OBS_STALENESS = timedelta(minutes=30)
_MINUTELY_VARIABLES = (
    "temp_c",
    "humidity_pct",
    "dew_point_c",
    "wind_speed_ms",
)


class NoForecastDataError(RuntimeError):
    """No provider forecast is fresh enough to serve from."""


class UnsupportedMethodError(ValueError):
    """An explicit serving method violates the product/variable contract."""


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
        (pl.col("ts") <= issue_time) & (pl.col("ts") > issue_time - OBS_STALENESS)
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
        archive.hourly,
        snapshots,
        hourly_truth,
        causal_minute,
        config,
        ensembles=load_ensembles(ensembles_path(config)),
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


_STATEFUL_METHODS = frozenset({"ewa", "boa"})


def _stateful_blender(
    method_id: str,
    slice_: SupervisedSlice,
    config: Config,
    product: str,
    variable_name: str,
) -> Blender:
    """Warm-started online experts: validate state, advance its cursors, save.

    The latest state may come from an older dataset fingerprint: its schema,
    source set, and processed-prefix digests decide compatibility. Any
    mismatch falls back to a full replay.
    """
    from grounded_weather_forecast.artifacts import (  # noqa: PLC0415
        ArtifactError,
        ArtifactStore,
    )
    from grounded_weather_forecast.blenders.experts import (  # noqa: PLC0415
        OnlineExperts,
    )

    store = ArtifactStore(config.artifacts_dir / "state")
    fingerprint = dataset_fingerprint(config)
    blender = None
    try:
        _, state = store.load_latest_state(
            method_id=method_id,
            product=product,
            variable=variable_name,
        )
        if state.get("sources") == list(slice_.x.sources):
            blender = OnlineExperts.from_state(state, method_id).advance(slice_)
    except (ArtifactError, ValueError, OSError):
        blender = None
    if blender is None:
        fitted = get_factory(method_id)().fit(slice_)
        if not isinstance(fitted, OnlineExperts):  # pragma: no cover - registry gap
            return fitted
        blender = fitted
    with suppress(OSError):
        store.save(
            fingerprint=fingerprint,
            method_id=method_id,
            product=product,
            variable=variable_name,
            state=blender.to_state(),
        )
    return blender


def _fit_methods(
    train: pl.DataFrame,
    predict_frame: pl.DataFrame,
    variable: VariableSpec,
    method_ids: set[str],
    *,
    daily: bool,
    semantics: TruthSemantics,
    config: Config | None = None,
    issue_time: datetime | None = None,
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
    product = "daily" if daily else "hourly"
    for method_id in sorted(method_ids):
        if config is not None and method_id in _STATEFUL_METHODS:
            blender = _stateful_blender(
                method_id, slice_, config, product, variable.name
            )
        else:
            blender = get_factory(method_id)().fit(slice_)
        if config is not None and issue_time is not None:
            snapshot_observability(
                blender,
                method_id=method_id,
                product=product,
                variable=variable.name,
                config=config,
                issue_time=issue_time,
            )
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


def _validated_selection(
    selection: Selection,
    product: Product,
    variable: VariableSpec,
    *,
    explicit: bool,
) -> Selection:
    """Validate registration and compatibility before fitting a method."""
    try:
        get_factory(selection.method_id)
    except UnknownMethodError as exc:
        if explicit:
            raise UnsupportedMethodError(str(exc.args[0])) from exc
        return Selection(
            "equal_weight",
            reason=f"degraded stale selection: unknown method {selection.method_id}",
        )
    if supports_product(selection.method_id, product, variable):
        return selection
    if explicit:
        msg = (
            f"method {selection.method_id!r} does not support "
            f"{product.value}.{variable.name}; choose a compatible method or use auto"
        )
        raise UnsupportedMethodError(msg)
    return Selection(
        "equal_weight",
        reason=(
            f"degraded stale selection: {selection.method_id} does not "
            f"support {product.value}.{variable.name}"
        ),
    )


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
    issue_time: datetime | None = None,
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
    product_kind = Product.DAILY if daily else Product.HOURLY
    chosen = [
        _validated_selection(
            selection,
            product_kind,
            variable,
            explicit=force_method is not None or selection.reason == "pinned in config",
        )
        for selection in chosen
    ]
    fitted = _fit_methods(
        train,
        predict_frame,
        variable,
        {c.method_id for c in chosen},
        daily=daily,
        semantics=semantics,
        config=config,
        issue_time=issue_time,
    )
    if fitted is None:
        if {selection.method_id for selection in chosen} != {"equal_weight"}:
            return None
        point = _cold_start_equal_weight(predict_frame, variable, daily=daily)
        return VariableBlend(
            point=point,
            methods=["equal_weight"] * predict_frame.height,
            reasons=[
                selection.reason
                if selection.reason.startswith("degraded stale selection:")
                else "degraded cold start: no scoreable training truth"
                for selection in chosen
            ],
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


def _quantile_grid(raw: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    pairs = sorted((float(level), value) for level, value in raw.items())
    if not pairs:
        return np.empty(0), np.empty(0)
    return (
        np.asarray([level for level, _ in pairs], dtype=np.float64),
        np.asarray([value for _, value in pairs], dtype=np.float64),
    )


def _curve_on(
    raw: dict[str, float], point: float | None, levels: np.ndarray
) -> np.ndarray:
    source_levels, values = _quantile_grid(raw)
    if source_levels.size:
        return np.interp(levels, source_levels, values)
    fallback = np.nan if point is None else point
    return np.full(levels.shape[0], fallback, dtype=np.float64)


def _write_curve(
    raw: dict[str, float], union_levels: np.ndarray, curve: np.ndarray
) -> None:
    source_levels, _ = _quantile_grid(raw)
    if not source_levels.size:
        return
    coherent = np.maximum.accumulate(curve)
    mapped = np.interp(source_levels, union_levels, coherent)
    for key in tuple(raw):
        raw[key] = float(np.interp(float(key), source_levels, mapped))


def _enforce_mapped_pair(
    lower_q: dict[str, float],
    upper_q: dict[str, float],
    lower_point: float | None,
    upper_point: float | None,
    union: np.ndarray,
    *,
    adjust: str,
) -> None:
    """Conservatively preserve coherence after mapping to unequal knot grids."""
    if adjust in {"lower", "both"} and lower_q:
        lower_levels, lower_values = _quantile_grid(lower_q)
        previous = float(union[0])
        for index, level in enumerate(lower_levels):
            bound = float(_curve_on(upper_q, upper_point, np.array([previous]))[0])
            lower_values[index] = min(lower_values[index], bound)
            previous = float(level)
        lower_values = np.maximum.accumulate(lower_values)
        for key, value in zip(sorted(lower_q, key=float), lower_values, strict=True):
            lower_q[key] = float(value)
    if adjust in {"upper", "both"} and upper_q:
        upper_levels, upper_values = _quantile_grid(upper_q)
        for index, _level in enumerate(upper_levels):
            next_level = (
                float(upper_levels[index + 1])
                if index + 1 < upper_levels.shape[0]
                else float(union[-1])
            )
            bound = float(_curve_on(lower_q, lower_point, np.array([next_level]))[0])
            upper_values[index] = max(upper_values[index], bound)
        upper_values = np.maximum.accumulate(upper_values)
        for key, value in zip(sorted(upper_q, key=float), upper_values, strict=True):
            upper_q[key] = float(value)


def _cohere_pair(
    values: dict[str, float | None],
    quantiles: dict[str, dict[str, float]],
    lower_name: str,
    upper_name: str,
    *,
    adjust: str,
) -> None:
    """Enforce ``lower <= upper`` on points and every advertised quantile."""
    lower_q = quantiles.get(lower_name, {})
    upper_q = quantiles.get(upper_name, {})
    lower_levels, _ = _quantile_grid(lower_q)
    upper_levels, _ = _quantile_grid(upper_q)
    union = np.unique(np.concatenate((lower_levels, upper_levels)))
    if union.size:
        lower_curve = _curve_on(lower_q, values.get(lower_name), union)
        upper_curve = _curve_on(upper_q, values.get(upper_name), union)
        paired = np.isfinite(lower_curve) & np.isfinite(upper_curve)
        match adjust:
            case "lower":
                lower_curve[paired] = np.minimum(
                    lower_curve[paired], upper_curve[paired]
                )
            case "upper":
                upper_curve[paired] = np.maximum(
                    upper_curve[paired], lower_curve[paired]
                )
            case "both":
                original_lower = lower_curve.copy()
                lower_curve[paired] = np.minimum(
                    original_lower[paired], upper_curve[paired]
                )
                upper_curve[paired] = np.maximum(
                    original_lower[paired], upper_curve[paired]
                )
            case _:  # pragma: no cover - private invariant
                msg = f"unknown coherence adjustment: {adjust}"
                raise ValueError(msg)
        _write_curve(lower_q, union, lower_curve)
        _write_curve(upper_q, union, upper_curve)
        _enforce_mapped_pair(
            lower_q,
            upper_q,
            values.get(lower_name),
            values.get(upper_name),
            union,
            adjust=adjust,
        )
    lower = values.get(lower_name)
    upper = values.get(upper_name)
    if lower is not None and upper is not None:
        match adjust:
            case "lower":
                values[lower_name] = min(lower, upper)
            case "upper":
                values[upper_name] = max(upper, lower)
            case "both":
                values[lower_name], values[upper_name] = (
                    min(lower, upper),
                    max(lower, upper),
                )


def _keep_points_inside_quantiles(
    values: dict[str, float | None], quantiles: dict[str, dict[str, float]]
) -> None:
    for name, raw in quantiles.items():
        point = values.get(name)
        finite = [value for value in raw.values() if math.isfinite(value)]
        if point is not None and finite:
            values[name] = min(max(point, min(finite)), max(finite))


def _cohere_hourly(points: list[HourlyPoint]) -> None:
    """Enforce cross-variable relationships on points and distributions."""
    for point in points:
        _cohere_pair(
            point.values,
            point.quantiles,
            "dew_point_c",
            "temp_c",
            adjust="lower",
        )
        _cohere_pair(
            point.values,
            point.quantiles,
            "wind_speed_ms",
            "wind_gust_ms",
            adjust="upper",
        )
        _keep_points_inside_quantiles(point.values, point.quantiles)
        temperature = point.values.get("temp_c")
        dew_point = point.values.get("dew_point_c")
        if temperature is not None and dew_point is not None:
            point.values["dew_point_c"] = min(dew_point, temperature)
        speed = point.values.get("wind_speed_ms")
        gust = point.values.get("wind_gust_ms")
        if speed is not None and gust is not None:
            point.values["wind_gust_ms"] = max(gust, speed)


def _cohere_daily(points: list[DailyPoint]) -> None:
    for point in points:
        _cohere_pair(
            point.values,
            point.quantiles,
            "temp_min_c",
            "temp_max_c",
            adjust="both",
        )
        _keep_points_inside_quantiles(point.values, point.quantiles)
        low = point.values.get("temp_min_c")
        high = point.values.get("temp_max_c")
        if low is not None and high is not None and low > high:
            point.values["temp_min_c"], point.values["temp_max_c"] = high, low


def hourly_product(
    snapshot: Snapshot,
    train: pl.DataFrame,
    selections: SelectionMap,
    config: Config,
    semantics: TruthSemantics | Mapping[str, TruthSemantics],
    force_method: str | None = None,
) -> tuple[list[HourlyPoint], dict[str, VariableBlend]]:
    """Blended hourly path, plus the per-variable blends for the minutely product."""
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
            issue_time=snapshot.issue_time,
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
    return points, blended


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
            issue_time=snapshot.issue_time,
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


def _now_forecast(leads: np.ndarray, path: np.ndarray) -> float:
    """The path extrapolated to lead 0 (``np.interp`` would clamp to the
    first hourly value, ~0.5-1 h out, and misstate the anchor residual)."""
    if leads.shape[0] >= 2:
        slope = (path[1] - path[0]) / (leads[1] - leads[0])
        return float(path[0] - slope * leads[0])
    return float(path[0])


def _lead_zero_path(
    leads: np.ndarray, path: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Prepend the same lead-zero extrapolation used by the anchor residual."""
    order = np.argsort(leads, kind="stable")
    ordered_leads, ordered_path = leads[order], path[order]
    if ordered_leads[0] <= 0.0:
        return ordered_leads, ordered_path
    return (
        np.insert(ordered_leads, 0, 0.0),
        np.insert(ordered_path, 0, _now_forecast(ordered_leads, ordered_path)),
    )


def _minute_path(
    leads: np.ndarray,
    path: np.ndarray,
    methods: list[str],
    lead: float,
) -> tuple[float, float, bool]:
    """Interpolate within one anchoring regime and return its lead-zero value."""
    order = np.argsort(leads, kind="stable")
    ordered_leads = leads[order]
    ordered_path = path[order]
    ordered_methods = [methods[index] for index in order]
    if ordered_leads.shape[0] == 1:
        value = float(ordered_path[0])
        return value, value, ordered_methods[0].startswith("anchored")
    right = int(np.searchsorted(ordered_leads, lead, side="left"))
    left = max(min(right - 1, ordered_leads.shape[0] - 2), 0)
    right = left + 1
    anchored = (
        ordered_methods[left].startswith("anchored"),
        ordered_methods[right].startswith("anchored"),
    )
    if anchored[0] != anchored[1]:
        distances = (
            abs(lead - float(ordered_leads[left])),
            abs(float(ordered_leads[right]) - lead),
        )
        chosen = right if distances[1] < distances[0] else left
        value = float(ordered_path[chosen])
        return value, value, anchored[chosen]
    segment_leads, segment_path = _lead_zero_path(
        ordered_leads[[left, right]],
        ordered_path[[left, right]],
    )
    return (
        float(np.interp(lead, segment_leads, segment_path)),
        float(segment_path[0]),
        anchored[0],
    )


def minutely_product(
    snapshot: Snapshot,
    hourly_blend: dict[str, VariableBlend],
    config: Config,
) -> list[MinutelyPoint]:
    """The anchored nowcast: hourly path interpolated to minutes, anchored ONCE.

    When the selected hourly method is already an ``anchored_*`` blender, its
    fitted correction is baked into the path and the minutely product only
    interpolates — anchoring is applied exactly once, by whichever stage the
    leaderboard promoted. The config decay is the cold-start fallback for
    un-anchored paths.
    """
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
            blend = hourly_blend.get(name)
            if blend is None or not np.isfinite(blend.point).any():
                continue
            usable = np.isfinite(blend.point)
            usable_leads, path = leads[usable], blend.point[usable]
            usable_methods = [
                method
                for method, available in zip(blend.methods, usable, strict=True)
                if available
            ]
            interpolated, now_forecast, already_anchored = _minute_path(
                usable_leads,
                path,
                usable_methods,
                lead,
            )
            observed = snapshot.observation.get(name)
            if observed is not None and not already_anchored:
                residual = observed - now_forecast
                interpolated += (
                    _anchor_weight(minute, config.predict.minutely_tau_hours) * residual
                )
            values[name] = _finite(interpolated, hourly_variable(name))
        temperature = values.get("temp_c")
        dew_point = values.get("dew_point_c")
        if temperature is not None and dew_point is not None:
            values["dew_point_c"] = min(dew_point, temperature)
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


def _supported_release_ids(selections: SelectionMap) -> list[str]:
    release_ids: set[str] = set()
    for (product_name, variable_name, _), selection in selections.items():
        try:
            product = Product(product_name)
            variable = (
                daily_variable(variable_name)
                if product is Product.DAILY
                else hourly_variable(variable_name)
            )
        except (KeyError, ValueError):
            continue
        try:
            get_factory(selection.method_id)
        except UnknownMethodError:
            continue
        if selection.release_id is not None and supports_product(
            selection.method_id, product, variable
        ):
            release_ids.add(selection.release_id)
    return sorted(release_ids)


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
    release_ids = _supported_release_ids(selections)
    degraded = not release_ids and force_method is None
    status_reason = (
        no_evidence_reason(config, config.dataset.dir / "scores") if degraded else None
    )
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
        status_reason=status_reason,
        release_ids=release_ids,
    )
