"""IDR/EasyUQ: the zero-tuning distributional benchmark, in three cuts.

Isotonic Distributional Regression (Henzi, Ziegel & Gneiting 2021) learns the
full conditional CDF given ONE covariate — here the grounded blend's point —
under the sole assumption that larger forecasts mean stochastically larger
outcomes. No parametric family, no hyperparameters, provably calibrated
in-sample: the universal benchmark any fancier distributional method must
beat (the EasyUQ recommendation).

Three registered variants, board-arbitrated:

- ``idr`` — the original global fit, kept as the control. Conditioning only
  on forecast VALUE pools every lead bucket into one map; IDR under
  stochastic order cannot represent distributions that agree in location
  but differ in spread, so sharp 0-1h and wide 240h+ errors blur (measured:
  pooled coverage80 = 0.52 against the 0.80 target).
- ``idr_bucket`` — per-lead-bucket fits (published practice: EasyUQ and
  EUPPBench train per lead) with subagging (JRSS-B: subsample halves
  without replacement, average the CDFs) smoothing the step-function tails.
  No coverage guarantee; the smoothing arm of the A/B.
- ``idr_bucket_dcp`` — the same per-bucket fits wrapped in distributional
  conformal prediction (Chernozhukov, Wüthrich & Zhu 2021): a chronological
  calibration split, PIT-rank conformity, and finite-sample-adjusted
  quantile levels. The only variant with a marginal coverage guarantee
  regardless of IDR misspecification; the guarantee arm.

Implementation: for each threshold ``z`` on a grid of training-outcome
quantiles, the conditional exceedance ``P(Y > z | X = x)`` must be isotonic
in ``x``; fit it with pool-adjacent-violators over the sorted covariate, and
read predictive quantiles off the resulting CDF stack. Prediction uses the
step function of the sorted training covariate (no extrapolation beyond the
observed range — a known, documented IDR limitation).
"""

import math
from dataclasses import dataclass, field
from importlib import import_module
from typing import Self

import numpy as np

from grounded_weather_forecast.blenders.combine import GroundedEqualWeight
from grounded_weather_forecast.blenders.protocol import (
    finalize_point,
    finalize_quantiles,
)
from grounded_weather_forecast.blenders.registry import BlenderFactory, register
from grounded_weather_forecast.contracts import (
    Blender,
    BlendResult,
    FloatArray,
    ForecastMatrix,
    SupervisedSlice,
    TargetKind,
    VariableSpec,
)
from grounded_weather_forecast.leads import LeadBucket, buckets_for_product

QUANTILE_LEVELS: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))
_THRESHOLD_LEVELS = np.linspace(0.02, 0.98, 49)
_MIN_FIT_ROWS = 100
_SUBAGG_SAMPLES = 50
_MIN_CALIBRATION_ROWS = 25
_CALIBRATION_FRACTION = 0.25


def pava_isotonic(values: FloatArray, weights: FloatArray | None = None) -> FloatArray:
    """Pool-adjacent-violators: the L2 isotonic (non-decreasing) regression.

    Delegates to scipy's compiled PAVA — this is the single hottest call in a
    backtest (tens of thousands of fits per ``idr_bucket`` instance), and the
    weighted L2 isotonic solution is unique, so the values returned match the
    former pure-Python stack loop exactly. Inputs are finite by construction
    (grouped exceedance means with positive counts).
    """
    optimize = import_module("scipy.optimize")  # heavy import; load on first use
    w = None if weights is None else weights.astype(np.float64)
    result = optimize.isotonic_regression(
        values.astype(np.float64), weights=w, increasing=True
    )
    return np.asarray(result.x, dtype=np.float64)


@dataclass(frozen=True)
class IdrState:
    """One fitted IDR distribution: the CDF stack over the covariate grid."""

    sorted_x: FloatArray
    cdf_stack: FloatArray  # (n_unique_x, n_thresholds)
    thresholds: FloatArray


def fit_idr_state(
    x: FloatArray,
    y: FloatArray,
    *,
    thresholds: FloatArray | None = None,
    min_rows: int = _MIN_FIT_ROWS,
) -> IdrState | None:
    """Fit one IDR distribution on finite (x, y) pairs; None below the floor.

    ``thresholds`` pins the outcome grid — subagging fits every subsample on
    the full sample's grid so their CDFs average cell for cell.
    """
    scored = np.isfinite(x) & np.isfinite(y)
    if int(scored.sum()) < min_rows:
        return None
    order = np.argsort(x[scored], kind="stable")
    x_sorted, y_sorted = x[scored][order], y[scored][order]
    unique_x, first, counts = np.unique(x_sorted, return_index=True, return_counts=True)
    if thresholds is None:
        thresholds = np.unique(np.quantile(y_sorted, _THRESHOLD_LEVELS))
    # P(Y <= z | x) must be non-increasing in x, so fit the exceedance
    # indicator isotonically and read the CDF as its complement. Equal
    # covariates are pooled first so their fitted distribution cannot
    # depend on stable-sort input order.
    cdf = np.empty((unique_x.shape[0], thresholds.shape[0]))
    for column, z in enumerate(thresholds):
        raw_exceeds = (y_sorted > z).astype(np.float64)
        grouped_exceeds = np.add.reduceat(raw_exceeds, first) / counts
        cdf[:, column] = 1.0 - pava_isotonic(grouped_exceeds, counts.astype(np.float64))
    # enforce monotonicity across thresholds too (finite-sample wiggles)
    return IdrState(
        sorted_x=unique_x,
        cdf_stack=np.maximum.accumulate(cdf, axis=1),
        thresholds=thresholds,
    )


def _grid_positions(state: IdrState, points: FloatArray) -> np.ndarray:
    return np.clip(
        np.searchsorted(state.sorted_x, points, side="right") - 1,
        0,
        state.sorted_x.shape[0] - 1,
    )


def subagged_idr_state(x: FloatArray, y: FloatArray) -> IdrState | None:
    """Per-bucket IDR with subsample aggregation (JRSS-B scheme).

    Halves without replacement, CDFs averaged on the full sample's covariate
    grid and outcome thresholds — linear aggregation preserves isotonicity
    and smooths the step-function tails that bias q10/q90 inward. Seeded by
    the sample size so refits on identical data are identical (backtest
    reproducibility; no wall-clock or global randomness).
    """
    full = fit_idr_state(x, y)
    if full is None:
        return None
    scored = np.isfinite(x) & np.isfinite(y)
    x_finite, y_finite = x[scored], y[scored]
    n = x_finite.shape[0]
    half = n // 2
    if half < _MIN_FIT_ROWS // 2:
        return full
    rng = np.random.default_rng(n)
    accumulated = np.zeros_like(full.cdf_stack)
    used = 0
    for _ in range(_SUBAGG_SAMPLES):
        chosen = rng.choice(n, size=half, replace=False)
        state = fit_idr_state(
            x_finite[chosen],
            y_finite[chosen],
            thresholds=full.thresholds,
            min_rows=_MIN_FIT_ROWS // 2,
        )
        if state is None:
            continue
        accumulated += state.cdf_stack[_grid_positions(state, full.sorted_x)]
        used += 1
    if not used:
        return full
    return IdrState(
        sorted_x=full.sorted_x,
        cdf_stack=np.maximum.accumulate(accumulated / used, axis=1),
        thresholds=full.thresholds,
    )


def quantile_rows_from_state(
    state: IdrState,
    base_point: FloatArray,
    levels: FloatArray,
) -> FloatArray:
    """Predictive quantiles at ``levels`` for each row's covariate value."""
    positions = _grid_positions(state, base_point)
    rows = np.empty((base_point.shape[0], levels.shape[0]))
    # Quantiles depend only on the grid position, so interpolate once per
    # distinct position instead of once per row.
    for position in np.unique(positions):
        cdf = state.cdf_stack[position]
        rows[positions == position] = np.interp(
            levels,
            cdf,
            state.thresholds,
            left=state.thresholds[0],
            right=state.thresholds[-1],
        )
    return rows


def pit_values(state: IdrState, x: FloatArray, y: FloatArray) -> FloatArray:
    """F̂(y | x) for calibration rows, interpolated on the threshold grid."""
    positions = _grid_positions(state, x)
    pits = np.empty(x.shape[0])
    # np.interp vectorizes over the query points, so batch rows sharing a
    # grid position rather than interpolating row by row.
    for position in np.unique(positions):
        rows = positions == position
        pits[rows] = np.interp(
            y[rows],
            state.thresholds,
            state.cdf_stack[position],
            left=0.0,
            right=1.0,
        )
    return pits


def dcp_adjusted_levels(pits: FloatArray, levels: FloatArray) -> FloatArray:
    """Finite-sample-adjusted quantile levels from calibration PIT ranks.

    Split distributional conformal prediction (Chernozhukov-Wüthrich-Zhu):
    reading the fitted CDF at the ceil((n+1)·tau)-th order statistic of the
    calibration PITs gives marginal coverage tau regardless of how
    misspecified the fitted distribution is.
    """
    ordered = np.sort(pits)
    n = ordered.shape[0]
    adjusted = np.empty(levels.shape[0])
    for index, level in enumerate(levels):
        rank = min(max(math.ceil((n + 1) * float(level)), 1), n)
        adjusted[index] = ordered[rank - 1]
    return np.clip(adjusted, 1e-6, 1.0 - 1e-6)


@dataclass
class Idr:
    """Isotonic distributional regression on the base blend's point."""

    base_factory: BlenderFactory
    method_id: str = "idr"
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _state: IdrState | None = None

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._base = self.base_factory().fit(train)
        base_point = self._base.predict(train.x).point
        self._state = fit_idr_state(base_point, train.y)
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        base_point = self._base.predict(x).point
        if self._state is None:
            return BlendResult(
                point=finalize_point(base_point, self._kind, self._variable)
            )
        quantiles = finalize_quantiles(
            quantile_rows_from_state(
                self._state,
                np.nan_to_num(base_point, nan=0.0),
                np.asarray(QUANTILE_LEVELS),
            ),
            self._kind,
            self._variable,
        )
        missing = ~np.isfinite(base_point)
        quantiles[missing] = np.nan
        point = np.where(
            ~missing,
            quantiles[:, len(QUANTILE_LEVELS) // 2],
            np.nan,
        )
        return BlendResult(
            point=finalize_point(point, self._kind, self._variable),
            quantiles=quantiles,
            quantile_levels=QUANTILE_LEVELS,
        )


@dataclass
class _BucketedIdr:
    """Shared per-bucket machinery for the two idr_bucket variants.

    Buckets below the row floor fall back to the global fit (a cliff, not
    shrinkage: two step-function CDFs on different covariate grids have no
    principled linear blend — a CDF-mixture rung is the known refinement).
    """

    base_factory: BlenderFactory
    method_id: str
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _buckets: tuple[LeadBucket, ...] = ()
    _states: dict[str, IdrState] = field(default_factory=dict)
    _global: IdrState | None = None

    def _fit_bucket(
        self, x: FloatArray, y: FloatArray, rows: np.ndarray, label: str | None
    ) -> IdrState | None:
        raise NotImplementedError

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._base = self.base_factory().fit(train)
        base_point = self._base.predict(train.x).point
        lead = train.x.lead_hours
        self._buckets = buckets_for_product(train.x.product)
        self._states = {}
        all_rows = np.arange(lead.shape[0])
        self._global = self._fit_bucket(base_point, train.y, all_rows, None)
        for bucket in self._buckets:
            rows = all_rows[(lead >= bucket.lo) & (lead < bucket.hi)]
            if rows.shape[0] < _MIN_FIT_ROWS:
                continue
            state = self._fit_bucket(base_point, train.y, rows, bucket.label)
            if state is not None:
                self._states[bucket.label] = state
        return self

    def _levels_for(self, label: str | None) -> FloatArray:  # noqa: ARG002
        return np.asarray(QUANTILE_LEVELS)

    def _state_for(self, lead: float) -> tuple[IdrState | None, str | None]:
        for bucket in self._buckets:
            if bucket.contains(lead):
                state = self._states.get(bucket.label)
                if state is not None:
                    return state, bucket.label
                return self._global, None
        return self._global, None

    def _rows_by_state(
        self, lead_hours: FloatArray
    ) -> list[tuple[IdrState, str | None, np.ndarray]]:
        """Row indices grouped by the (state, label) that serves them.

        The batched equivalent of calling ``_state_for`` per row: first
        matching bucket wins, bucket rows without a fitted state fall to the
        global fit under the default levels, and rows outside every bucket
        fall to the global fit too.
        """
        remaining = np.ones(lead_hours.shape[0], dtype=bool)
        grouped: list[tuple[IdrState, str | None, np.ndarray]] = []
        global_rows: list[np.ndarray] = []
        for bucket in self._buckets:
            mask = remaining & (lead_hours >= bucket.lo) & (lead_hours < bucket.hi)
            if not mask.any():
                continue
            remaining &= ~mask
            state = self._states.get(bucket.label)
            if state is not None:
                grouped.append((state, bucket.label, np.nonzero(mask)[0]))
            else:
                global_rows.append(np.nonzero(mask)[0])
        if remaining.any():
            global_rows.append(np.nonzero(remaining)[0])
        if self._global is not None and global_rows:
            grouped.append((self._global, None, np.concatenate(global_rows)))
        return grouped

    def predict(self, x: ForecastMatrix) -> BlendResult:
        base_point = self._base.predict(x).point
        if self._global is None and not self._states:
            return BlendResult(
                point=finalize_point(base_point, self._kind, self._variable)
            )
        safe_point = np.nan_to_num(base_point, nan=0.0)
        rows = np.full((x.n_rows, len(QUANTILE_LEVELS)), np.nan)
        for state, label, indices in self._rows_by_state(x.lead_hours):
            rows[indices] = quantile_rows_from_state(
                state,
                safe_point[indices],
                self._levels_for(label),
            )
        quantiles = finalize_quantiles(rows, self._kind, self._variable)
        missing = ~np.isfinite(base_point)
        quantiles[missing] = np.nan
        point = np.where(~missing, quantiles[:, len(QUANTILE_LEVELS) // 2], np.nan)
        return BlendResult(
            point=finalize_point(point, self._kind, self._variable),
            quantiles=quantiles,
            quantile_levels=QUANTILE_LEVELS,
        )


@dataclass
class IdrBucket(_BucketedIdr):
    """Per-lead-bucket IDR with subagging: the smoothing arm of the A/B."""

    method_id: str = "idr_bucket"

    def _fit_bucket(
        self,
        x: FloatArray,
        y: FloatArray,
        rows: np.ndarray,
        label: str | None,  # noqa: ARG002
    ) -> IdrState | None:
        return subagged_idr_state(x[rows], y[rows])


@dataclass
class IdrBucketDcp(_BucketedIdr):
    """Per-bucket IDR under distributional conformal prediction: the
    guarantee arm — finite-sample marginal coverage from a chronological
    calibration split, however misspecified the isotonic fit is."""

    method_id: str = "idr_bucket_dcp"
    _adjusted: dict[str, FloatArray] = field(default_factory=dict)
    _global_adjusted: FloatArray | None = None

    def fit(self, train: SupervisedSlice) -> Self:
        self._adjusted = {}
        self._global_adjusted = None
        return super().fit(train)

    def _fit_bucket(
        self, x: FloatArray, y: FloatArray, rows: np.ndarray, label: str | None
    ) -> IdrState | None:
        # Rows arrive in matrix order, which the dataset builds
        # chronologically — the tail is the most recent evidence, held out
        # so calibration never sees its own training residuals.
        split = rows.shape[0] - max(
            int(rows.shape[0] * _CALIBRATION_FRACTION), _MIN_CALIBRATION_ROWS
        )
        if split < _MIN_FIT_ROWS // 2:
            return None
        fit_rows, calibration_rows = rows[:split], rows[split:]
        state = fit_idr_state(x[fit_rows], y[fit_rows], min_rows=_MIN_FIT_ROWS // 2)
        if state is None:
            return None
        finite = np.isfinite(x[calibration_rows]) & np.isfinite(y[calibration_rows])
        if int(finite.sum()) < _MIN_CALIBRATION_ROWS:
            return None
        pits = pit_values(
            state, x[calibration_rows][finite], y[calibration_rows][finite]
        )
        levels = dcp_adjusted_levels(pits, np.asarray(QUANTILE_LEVELS))
        if label is None:
            self._global_adjusted = levels
        else:
            self._adjusted[label] = levels
        return state

    def _levels_for(self, label: str | None) -> FloatArray:
        if label is not None and label in self._adjusted:
            return self._adjusted[label]
        if self._global_adjusted is not None:
            return self._global_adjusted
        return np.asarray(QUANTILE_LEVELS)


def _idr() -> Blender:
    return Idr(GroundedEqualWeight, "idr")


def _idr_bucket() -> Blender:
    return IdrBucket(GroundedEqualWeight)


def _idr_bucket_dcp() -> Blender:
    return IdrBucketDcp(GroundedEqualWeight)


register("idr", _idr)
register("idr_bucket", _idr_bucket)
register("idr_bucket_dcp", _idr_bucket_dcp)
