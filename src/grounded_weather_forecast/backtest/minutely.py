"""The minutely/short-lead backtest: scoring the nowcast path constructions.

The minutely product DERIVES from the hourly blend — serving interpolates the
promoted hourly path per minute and anchors it to the latest observation with
a hand-set exponential decay (``[predict].minutely_tau_hours``). No blender
registry method has sub-hourly semantics, so a minutely backtest is not the
39-blender engine with a third product: the candidates are the PATH
CONSTRUCTIONS themselves — anchor decay rates, the full shift, a fitted
per-bucket response (the RAFT design at minute resolution), a persist-then-
ramp weight, and the two references (the un-anchored interpolation the
product would be without its anchor, and flat observation persistence, the
nowcast floor).

Each tau is its own method id: the leaderboard, gates, and selection ARE the
grid search, with per-bucket resolution and statistical protection for free,
and a promoted id maps back to serving parameters with no stored state. The
proxy hourly path is a fold-fitted grounded equal weight — the same base the
anchored hourly family uses — trained under the HOURLY truth clock (2 h,
stricter than the minutely one: using the minute clock for the grounding fit
would leak). Minute truth comes from the shared minute grid; the anchor
residual uses the matrix ``obs__{var}`` issue-time observations, mirroring
serving's staleness semantics.

Scores land in the standard schema with ``product="minutely"`` and the
sub-hourly lead buckets; everything downstream (report loop, leaderboard,
promotion, selection keys, self-verification joins) absorbs the new product
by construction.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol, Self

import numpy as np
import polars as pl

from grounded_weather_forecast.backtest.scores import (
    SCORES_SCHEMA,
    empty_scores,
    stamped_scores,
)
from grounded_weather_forecast.backtest.splits import (
    WindowMode,
    fold_plans,
    hourly_truth_known_at,
    minutely_truth_known_at,
)
from grounded_weather_forecast.blenders.combine import GroundedEqualWeight
from grounded_weather_forecast.config import Config
from grounded_weather_forecast.contracts import (
    FloatArray,
    TruthSemantics,
    VariableSpec,
    hourly_variable,
)
from grounded_weather_forecast.dataset.matrix import (
    assert_single_kind,
    matrix_feature_columns,
    matrix_sources,
    to_forecast_matrix,
    to_supervised_slice,
)
from grounded_weather_forecast.evaluation import EvaluationRun
from grounded_weather_forecast.leads import MINUTELY_BUCKETS, minutely_bucket_expr

MINUTELY_SCORE_VARIABLES: tuple[str, ...] = (
    "temp_c",
    "humidity_pct",
    "dew_point_c",
    "wind_speed_ms",
)
# Inside a 60-minute horizon the weights for tau = 3/6/12 h are 0.72/0.85/
# 0.92 at minute 60 — indistinguishable; anchor_full covers the flat end.
MINUTELY_TAU_GRID_HOURS: tuple[float, ...] = (0.25, 0.5, 1.0, 3.0)
_PATH_MAX_LEAD_HOURS = 3.0
_ANCHOR_WEIGHT_FLOOR = 0.05  # mirrors serving and anchoring._WEIGHT_FLOOR
_HORIZON_MINUTES = 60
_MIN_BUCKET_FIT_ROWS = 60
_SLOPE_CLIP = (-0.5, 1.5)  # the RAFT clip
_RAMP_CLIP = (0.0, 1.0)


def now_forecast(leads: FloatArray, path: FloatArray) -> float:
    """The path extrapolated to lead 0 (``np.interp`` would clamp to the
    first hourly value, ~0.5-1 h out, and misstate the anchor residual)."""
    if leads.shape[0] >= 2:
        slope = (path[1] - path[0]) / (leads[1] - leads[0])
        return float(path[0] - slope * leads[0])
    return float(path[0])


def lead_zero_path(
    leads: FloatArray, path: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Prepend the same lead-zero extrapolation used by the anchor residual."""
    order = np.argsort(leads, kind="stable")
    ordered_leads, ordered_path = leads[order], path[order]
    if ordered_leads[0] <= 0.0:
        return ordered_leads, ordered_path
    return (
        np.insert(ordered_leads, 0, 0.0),
        np.insert(ordered_path, 0, now_forecast(ordered_leads, ordered_path)),
    )


def anchor_weight(lead_hours: FloatArray, tau_hours: float) -> FloatArray:
    """Exponential anchor decay with serving's hard floor at 0.05."""
    weight = np.exp(-lead_hours / tau_hours)
    return np.where(weight < _ANCHOR_WEIGHT_FLOOR, 0.0, weight)


@dataclass(frozen=True)
class MinuteFrame:
    """Aligned per-(snapshot, minute) arrays for one variable."""

    issue_time: pl.Series
    valid_time: pl.Series
    lead_hours: FloatArray
    base: FloatArray  # interpolated proxy path at the minute
    observation: FloatArray  # obs at issue, NaN when absent (per snapshot)
    residual: FloatArray  # observation - lead-zero now-forecast, per snapshot
    y_true: FloatArray  # minute-grid truth, NaN when absent

    @property
    def height(self) -> int:
        return int(self.lead_hours.shape[0])


class MinutelyPathMethod(Protocol):
    method_id: str

    def fit(self, train: MinuteFrame) -> Self: ...

    def predict(self, test: MinuteFrame) -> FloatArray: ...


@dataclass
class InterpPath:
    """The product minus its anchor: pure interpolation. Reference."""

    method_id: str = "minutely_interp"

    def fit(self, train: MinuteFrame) -> Self:  # noqa: ARG002 - closed form
        return self

    def predict(self, test: MinuteFrame) -> FloatArray:
        return test.base.copy()


@dataclass
class ObsPersistence:
    """Flat last observation: the nowcast floor. Reference."""

    method_id: str = "minutely_persistence"

    def fit(self, train: MinuteFrame) -> Self:  # noqa: ARG002 - closed form
        return self

    def predict(self, test: MinuteFrame) -> FloatArray:
        # Rows with no observation at issue stay NaN and drop from this
        # method's scores only (the leaderboard coverage gate absorbs ~2%).
        return test.observation.copy()


@dataclass
class AnchorTau:
    """path + exp(-lead/tau) * residual, serving's construction measured."""

    tau_hours: float
    method_id: str = ""

    def __post_init__(self) -> None:
        if not self.method_id:
            self.method_id = f"minutely_anchor_tau_{self.tau_hours:g}h"

    def fit(self, train: MinuteFrame) -> Self:  # noqa: ARG002 - closed form
        return self

    def predict(self, test: MinuteFrame) -> FloatArray:
        correction = np.nan_to_num(test.residual, nan=0.0)
        return test.base + anchor_weight(test.lead_hours, self.tau_hours) * correction


@dataclass
class AnchorFull:
    """path + residual: the tau -> infinity parallel shift."""

    method_id: str = "minutely_anchor_full"

    def fit(self, train: MinuteFrame) -> Self:  # noqa: ARG002 - closed form
        return self

    def predict(self, test: MinuteFrame) -> FloatArray:
        return test.base + np.nan_to_num(test.residual, nan=0.0)


def _bucket_indices(lead_hours: FloatArray) -> np.ndarray:
    """Index into MINUTELY_BUCKETS per row; -1 when out of range."""
    indices = np.full(lead_hours.shape[0], -1, dtype=np.int64)
    for index, bucket in enumerate(MINUTELY_BUCKETS):
        indices[(lead_hours >= bucket.lo) & (lead_hours < bucket.hi)] = index
    return indices


@dataclass
class _FittedResponse:
    """Per-bucket through-origin response of realized error to the residual.

    The shared machinery of the RAFT-style slope and the persist-then-ramp
    weight; they differ only in the clip and the monotone constraint.
    """

    method_id: str
    clip: tuple[float, float]
    monotone_non_increasing: bool = False
    _weights: FloatArray = field(default_factory=lambda: np.array([]))

    def fit(self, train: MinuteFrame) -> Self:
        buckets = _bucket_indices(train.lead_hours)
        error = train.y_true - train.base
        usable = np.isfinite(error) & np.isfinite(train.residual) & (buckets >= 0)
        weights = np.full(len(MINUTELY_BUCKETS), np.nan)
        residual_all = train.residual[usable]
        error_all = error[usable]
        denominator_all = float(np.sum(residual_all * residual_all))
        global_weight = (
            float(np.sum(error_all * residual_all) / denominator_all)
            if denominator_all > 0.0
            else 0.0
        )
        for index in range(len(MINUTELY_BUCKETS)):
            mask = usable & (buckets == index)
            if int(mask.sum()) < _MIN_BUCKET_FIT_ROWS:
                weights[index] = global_weight
                continue
            denominator = float(np.sum(train.residual[mask] ** 2))
            weights[index] = (
                float(np.sum(error[mask] * train.residual[mask]) / denominator)
                if denominator > 0.0
                else global_weight
            )
        weights = np.clip(weights, *self.clip)
        if self.monotone_non_increasing:
            weights = np.minimum.accumulate(weights)
        self._weights = weights
        return self

    def predict(self, test: MinuteFrame) -> FloatArray:
        if self._weights.shape[0] == 0:
            return test.base.copy()
        buckets = _bucket_indices(test.lead_hours)
        weight = np.where(buckets >= 0, self._weights[np.clip(buckets, 0, None)], 0.0)
        return test.base + weight * np.nan_to_num(test.residual, nan=0.0)


def minutely_methods() -> tuple[MinutelyPathMethod, ...]:
    """The candidate set; order fixes the report's method universe."""
    return (
        InterpPath(),
        ObsPersistence(),
        *(AnchorTau(tau) for tau in MINUTELY_TAU_GRID_HOURS),
        AnchorFull(),
        _FittedResponse(
            method_id="minutely_ramp",
            clip=_RAMP_CLIP,
            monotone_non_increasing=True,
        ),
        _FittedResponse(method_id="minutely_fitted_slope", clip=_SLOPE_CLIP),
    )


@dataclass(frozen=True, slots=True)
class MinutelyRequest:
    variables: tuple[str, ...] = MINUTELY_SCORE_VARIABLES
    window: WindowMode = "expanding"
    minute_step: int = 1  # volume escape hatch; 1 = score every minute


def minute_frame(
    path_rows: pl.DataFrame,
    truth_grid: pl.DataFrame,
    variable: str,
    base_point: FloatArray,
    minute_step: int,
) -> MinuteFrame:
    """One interpolated proxy path per snapshot, expanded to minutes 1..60."""
    minutes = np.arange(1, _HORIZON_MINUTES + 1, minute_step, dtype=np.int64)
    lead_grid = minutes / 60.0
    issue_times: list[datetime] = []
    valid_times: list[datetime] = []
    leads: list[FloatArray] = []
    bases: list[FloatArray] = []
    observations: list[FloatArray] = []
    residuals: list[FloatArray] = []
    observation_column = f"obs__{variable}"
    with_base = path_rows.with_columns(
        pl.Series("__base", base_point, dtype=pl.Float64)
    )
    for issue, group in with_base.group_by("issue_time", maintain_order=True):
        issue_time = issue[0]
        raw_leads = group["lead_hours"].to_numpy().astype(np.float64)
        raw_path = group["__base"].to_numpy().astype(np.float64)
        finite = np.isfinite(raw_leads) & np.isfinite(raw_path)
        if int(finite.sum()) == 0:
            continue
        segment_leads, segment_path = lead_zero_path(
            raw_leads[finite], raw_path[finite]
        )
        interpolated = np.interp(lead_grid, segment_leads, segment_path)
        observation = (
            float(group[observation_column][0])
            if observation_column in group.columns
            and group[observation_column][0] is not None
            else float("nan")
        )
        residual = observation - float(segment_path[0])
        minute_zero = issue_time.replace(second=0, microsecond=0)
        issue_times.extend([issue_time] * minutes.shape[0])
        valid_times.extend(
            minute_zero + timedelta(minutes=int(minute)) for minute in minutes
        )
        leads.append(lead_grid)
        bases.append(interpolated)
        observations.append(np.full(minutes.shape[0], observation))
        residuals.append(np.full(minutes.shape[0], residual))
    if not issue_times:
        empty = np.empty(0)
        return MinuteFrame(
            issue_time=pl.Series("issue_time", [], dtype=pl.Datetime("us", "UTC")),
            valid_time=pl.Series("valid_time", [], dtype=pl.Datetime("us", "UTC")),
            lead_hours=empty,
            base=empty,
            observation=empty,
            residual=empty,
            y_true=empty,
        )
    valid_series = pl.Series("valid_time", valid_times, dtype=pl.Datetime("us", "UTC"))
    truth = (
        pl.DataFrame({"valid_time": valid_series})
        .join(
            truth_grid.select("valid_time", variable),
            on="valid_time",
            how="left",
        )[variable]
        .to_numpy()
        .astype(np.float64)
    )
    return MinuteFrame(
        issue_time=pl.Series("issue_time", issue_times, dtype=pl.Datetime("us", "UTC")),
        valid_time=valid_series,
        lead_hours=np.concatenate(leads),
        base=np.concatenate(bases),
        observation=np.concatenate(observations),
        residual=np.concatenate(residuals),
        y_true=truth,
    )


def _snapshot_clock(path_rows: pl.DataFrame) -> pl.DataFrame:
    """One row per snapshot with the minute product's truth clock."""
    return (
        path_rows.group_by("issue_time")
        .agg(pl.len())
        .sort("issue_time")
        .with_columns(
            (
                pl.col("issue_time").dt.truncate("1m")
                + pl.duration(minutes=_HORIZON_MINUTES)
            ).alias("valid_time")
        )
    )


def run_minutely_backtest(
    matrix: pl.DataFrame,
    truth_grid: pl.DataFrame,
    request: MinutelyRequest,
    config: Config,
) -> pl.DataFrame:
    """Rolling-origin evaluation of the path constructions; standard scores."""
    if matrix.is_empty() or truth_grid.is_empty():
        return empty_scores()
    kind = assert_single_kind(matrix)
    methods = minutely_methods()
    evaluation = EvaluationRun.create(
        config,
        source_kind=kind,
        source_set=matrix_sources(matrix),
        feature_set=matrix_feature_columns(matrix),
        product="minutely",
        window=request.window,
        semantics=dict.fromkeys(request.variables, TruthSemantics.INSTANTANEOUS),
        methods=tuple(method.method_id for method in methods),
    )
    path_rows = matrix.filter(pl.col("lead_hours") <= _PATH_MAX_LEAD_HOURS).sort(
        "issue_time", "lead_hours"
    )
    if path_rows.is_empty():
        return empty_scores()
    snapshots = _snapshot_clock(path_rows)
    folds = fold_plans(
        snapshots["issue_time"],
        minutely_truth_known_at(snapshots),
        config.backtest,
        request.window,
    )
    hourly_known = hourly_truth_known_at(matrix)
    results: list[pl.DataFrame] = []
    for fold in folds:
        test_issues = snapshots["issue_time"][fold.test_rows]
        train_issues = snapshots["issue_time"][fold.train_rows]
        if test_issues.is_empty():
            continue
        # The grounding proxy trains under the stricter HOURLY truth clock;
        # the minute clock would leak hourly truth into the fit.
        train_matrix = matrix.filter(pl.Series(hourly_known) <= fold.origin)
        for variable_name in request.variables:
            variable = hourly_variable(variable_name)
            truth_column = f"t__{variable_name}__inst"
            train_scored = (
                train_matrix.filter(pl.col(truth_column).is_not_null())
                if truth_column in train_matrix.columns
                else train_matrix.head(0)
            )
            if train_scored.is_empty():
                continue
            proxy = GroundedEqualWeight().fit(
                to_supervised_slice(
                    train_scored,
                    variable,
                    daily=False,
                    semantics=TruthSemantics.INSTANTANEOUS,
                )
            )
            trained_sources = matrix_sources(train_scored)

            def frame_for(
                issues: pl.Series,
                *,
                name: str = variable_name,
                spec: VariableSpec = variable,
                fitted: GroundedEqualWeight = proxy,
                sources: tuple[str, ...] = trained_sources,
            ) -> MinuteFrame:
                rows = path_rows.filter(pl.col("issue_time").is_in(issues))
                if rows.is_empty():
                    return minute_frame(rows, truth_grid, name, np.empty(0), 1)
                base = fitted.predict(
                    to_forecast_matrix(rows, spec, sources=sources)
                ).point
                return minute_frame(rows, truth_grid, name, base, request.minute_step)

            test_frame = frame_for(test_issues)
            if test_frame.height == 0:
                continue
            train_frame = frame_for(train_issues)
            keys = (
                pl.DataFrame(
                    {
                        "issue_time": test_frame.issue_time,
                        "valid_time": test_frame.valid_time,
                    }
                )
                .with_columns(
                    pl.Series("lead_hours", test_frame.lead_hours, dtype=pl.Float64)
                )
                .with_columns(
                    minutely_bucket_expr(pl.col("lead_hours")).alias("lead_bucket")
                )
            )
            for method in minutely_methods():
                fitted = method.fit(train_frame)
                prediction = fitted.predict(test_frame)
                results.append(
                    stamped_scores(
                        keys,
                        evaluation,
                        method_id=method.method_id,
                        variable=variable_name,
                        product="minutely",
                        source_kind=kind,
                        semantics=TruthSemantics.INSTANTANEOUS.value,
                        window=request.window,
                        fold_origin=fold.origin,
                        y_pred=prediction,
                        y_true=test_frame.y_true,
                    )
                )
    if not results:
        return empty_scores()
    return pl.concat(results).cast(SCORES_SCHEMA)
