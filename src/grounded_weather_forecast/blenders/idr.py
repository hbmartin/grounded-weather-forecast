"""IDR/EasyUQ: the zero-tuning distributional benchmark.

Isotonic Distributional Regression (Henzi, Ziegel & Gneiting 2021) learns the
full conditional CDF given ONE covariate — here the grounded blend's point —
under the sole assumption that larger forecasts mean stochastically larger
outcomes. No parametric family, no hyperparameters, provably calibrated
in-sample: the universal benchmark any fancier distributional method must
beat (the EasyUQ recommendation).

Implementation: for each threshold ``z`` on a grid of training-outcome
quantiles, the conditional exceedance ``P(Y > z | X = x)`` must be isotonic
in ``x``; fit it with pool-adjacent-violators over the sorted covariate, and
read predictive quantiles off the resulting CDF stack. Prediction uses the
step function of the sorted training covariate (no extrapolation beyond the
observed range — a known, documented IDR limitation).
"""

from dataclasses import dataclass
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

QUANTILE_LEVELS: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))
_THRESHOLD_LEVELS = np.linspace(0.02, 0.98, 49)
_MIN_FIT_ROWS = 100


def pava_isotonic(values: FloatArray, weights: FloatArray | None = None) -> FloatArray:
    """Pool-adjacent-violators: the L2 isotonic (non-decreasing) regression."""
    n = values.shape[0]
    w = np.ones(n) if weights is None else weights.astype(np.float64).copy()
    level_values = values.astype(np.float64).copy()
    level_weights = w
    # blocks[i] = index of the last element in the block starting at i
    starts: list[int] = []
    means: list[float] = []
    sizes: list[float] = []
    counts: list[int] = []
    for index in range(n):
        starts.append(index)
        means.append(float(level_values[index]))
        sizes.append(float(level_weights[index]))
        counts.append(1)
        while len(means) > 1 and means[-2] >= means[-1]:
            total = sizes[-2] + sizes[-1]
            means[-2] = (means[-2] * sizes[-2] + means[-1] * sizes[-1]) / total
            sizes[-2] = total
            counts[-2] += counts[-1]
            starts.pop()
            means.pop()
            sizes.pop()
            counts.pop()
    result = np.empty(n)
    cursor = 0
    for mean, count in zip(means, counts, strict=True):
        result[cursor : cursor + count] = mean
        cursor += count
    return result


@dataclass
class Idr:
    """Isotonic distributional regression on the base blend's point."""

    base_factory: BlenderFactory
    method_id: str = "idr"
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _sorted_x: FloatArray | None = None
    _cdf_stack: FloatArray | None = None  # (n_train, n_thresholds)
    _thresholds: FloatArray | None = None

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._base = self.base_factory().fit(train)
        base_point = self._base.predict(train.x).point
        scored = np.isfinite(base_point) & np.isfinite(train.y)
        if int(scored.sum()) < _MIN_FIT_ROWS:
            return self
        x = base_point[scored]
        y = train.y[scored]
        order = np.argsort(x, kind="stable")
        x_sorted, y_sorted = x[order], y[order]
        unique_x, first, counts = np.unique(
            x_sorted, return_index=True, return_counts=True
        )
        thresholds = np.unique(np.quantile(y_sorted, _THRESHOLD_LEVELS))
        # P(Y <= z | x) must be non-increasing in x, so fit the exceedance
        # indicator isotonically and read the CDF as its complement. Equal
        # covariates are pooled first so their fitted distribution cannot
        # depend on stable-sort input order.
        cdf = np.empty((unique_x.shape[0], thresholds.shape[0]))
        for column, z in enumerate(thresholds):
            raw_exceeds = (y_sorted > z).astype(np.float64)
            grouped_exceeds = np.add.reduceat(raw_exceeds, first) / counts
            cdf[:, column] = 1.0 - pava_isotonic(
                grouped_exceeds, counts.astype(np.float64)
            )
        # enforce monotonicity across thresholds too (finite-sample wiggles)
        self._cdf_stack = np.maximum.accumulate(cdf, axis=1)
        self._sorted_x = unique_x
        self._thresholds = thresholds
        return self

    def _quantile_rows(self, base_point: FloatArray) -> FloatArray:
        if (
            self._sorted_x is None
            or self._cdf_stack is None
            or self._thresholds is None
        ):
            msg = "IDR distribution missing; fit before requesting quantiles"
            raise RuntimeError(msg)
        positions = np.clip(
            np.searchsorted(self._sorted_x, base_point, side="right") - 1,
            0,
            self._sorted_x.shape[0] - 1,
        )
        levels = np.asarray(QUANTILE_LEVELS)
        rows = np.empty((base_point.shape[0], levels.shape[0]))
        for index, position in enumerate(positions):
            cdf = self._cdf_stack[position]
            rows[index] = np.interp(
                levels,
                cdf,
                self._thresholds,
                left=self._thresholds[0],
                right=self._thresholds[-1],
            )
        return rows

    def predict(self, x: ForecastMatrix) -> BlendResult:
        base_point = self._base.predict(x).point
        if self._sorted_x is None:
            return BlendResult(
                point=finalize_point(base_point, self._kind, self._variable)
            )
        quantiles = finalize_quantiles(
            self._quantile_rows(np.nan_to_num(base_point, nan=0.0)),
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


def _idr() -> Blender:
    return Idr(GroundedEqualWeight, "idr")


register("idr", _idr)
