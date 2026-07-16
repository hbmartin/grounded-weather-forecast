"""Combining blenders on top of grounding: the floor pipeline.

- ``grounded_equal_weight``: affine-ground each source, then equal-weight the
  available corrected values. The bar every fancier method must beat.
- ``inverse_mse``: affine-ground, then weight sources per lead bucket by the
  inverse of their corrected training MSE (Bates-Granger), renormalized over
  availability per row.
"""

from dataclasses import dataclass, field
from typing import Self

import numpy as np

from grounded_weather_forecast.blenders.grounding import (
    BIAS_ONLY,
    FREE_SLOPE,
    AffineGrounding,
)
from grounded_weather_forecast.blenders.protocol import (
    FittedBuckets,
    PerBucketFitter,
    finalize_point,
    masked_average,
    renormalize_weights,
)
from grounded_weather_forecast.blenders.registry import register
from grounded_weather_forecast.contracts import (
    BlendResult,
    ForecastMatrix,
    SupervisedSlice,
    TargetKind,
)
from grounded_weather_forecast.leads import buckets_for_product

_MIN_MSE = 1e-6
_MIN_ROWS_PER_SOURCE = 12


@dataclass
class GroundedEqualWeight:
    method_id: str = "grounded_equal_weight"
    slope_shrinkage: float = BIAS_ONLY
    _kind: TargetKind = TargetKind.CONTINUOUS
    _grounding: AffineGrounding = field(default_factory=AffineGrounding)

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._grounding = AffineGrounding(self.slope_shrinkage).fit(train)
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        corrected = self._grounding.transform(x)
        point = masked_average(corrected, x.availability)
        return BlendResult(point=finalize_point(point, self._kind))


@dataclass
class InverseMseWeights:
    method_id: str = "inverse_mse"
    slope_shrinkage: float = BIAS_ONLY
    _kind: TargetKind = TargetKind.CONTINUOUS
    _grounding: AffineGrounding = field(default_factory=AffineGrounding)

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._grounding = AffineGrounding(self.slope_shrinkage).fit(train)
        corrected = self._grounding.transform(train.x)
        errors = (corrected - train.y[:, np.newaxis]) ** 2
        availability = train.x.availability

        def fit_weights(rows: np.ndarray) -> np.ndarray:
            counts = availability[rows].sum(axis=0)
            sums = np.where(availability[rows], errors[rows], 0.0).sum(axis=0)
            mse = np.where(
                counts >= _MIN_ROWS_PER_SOURCE,
                sums / np.maximum(counts, 1),
                np.nan,
            )
            fallback = float(np.nanmax(mse)) if np.isfinite(mse).any() else 1.0
            mse = np.where(np.isnan(mse), fallback, mse)
            return 1.0 / np.maximum(mse, _MIN_MSE)

        fitter = PerBucketFitter[np.ndarray](
            buckets=buckets_for_product(train.x.product), fit_one=fit_weights
        )
        self._fitted: FittedBuckets[np.ndarray] = fitter.fit(train.x.lead_hours)
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        corrected = self._grounding.transform(x)
        point = np.full(x.n_rows, np.nan)

        def use(weights: np.ndarray, rows: np.ndarray) -> None:
            row_weights = renormalize_weights(weights, x.availability[rows])
            filled = np.where(x.availability[rows], corrected[rows], 0.0)
            blended = (filled * row_weights).sum(axis=1)
            covered = x.availability[rows].any(axis=1)
            point[rows] = np.where(covered, blended, np.nan)

        self._fitted.apply(x.lead_hours, use)
        return BlendResult(point=finalize_point(point, self._kind))


def _affine_equal_weight() -> GroundedEqualWeight:
    """Free-slope grounding, kept on the leaderboard beside the bias-only
    default so the archive can say which correction it actually supports."""
    return GroundedEqualWeight(
        method_id="affine_equal_weight", slope_shrinkage=FREE_SLOPE
    )


register("grounded_equal_weight", GroundedEqualWeight)
register("affine_equal_weight", _affine_equal_weight)
register("inverse_mse", InverseMseWeights)
