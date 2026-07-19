"""Combining blenders on top of grounding: the floor pipeline.

- ``grounded_equal_weight``: affine-ground each source, then equal-weight the
  available corrected values. The bar every fancier method must beat.
- ``grounded_median_equal_weight``: the same pipeline with a median intercept.
  Promotion is on MAE, whose optimal constant offset is the median residual,
  not the mean; the two are registered side by side so the data decides.
- ``inverse_mse`` / ``inverse_mae``: affine-ground, then weight sources per
  lead bucket by the inverse of their corrected training error (Bates-Granger),
  renormalized over availability per row. The L1 variant tracks the metric the
  leaderboard promotes on, and is what the NBM uses operationally.
"""

from dataclasses import dataclass, field
from typing import Self

import numpy as np

from grounded_weather_forecast.blenders.grounding import (
    BIAS_ONLY,
    FREE_SLOPE,
    AffineGrounding,
    InterceptEstimator,
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
    VariableSpec,
)
from grounded_weather_forecast.leads import buckets_for_product

_MIN_LOSS = 1e-6
_MIN_ROWS_PER_SOURCE = 12


@dataclass
class GroundedEqualWeight:
    method_id: str = "grounded_equal_weight"
    slope_shrinkage: float = BIAS_ONLY
    intercept: InterceptEstimator = "mean"
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _grounding: AffineGrounding = field(default_factory=AffineGrounding)

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._grounding = AffineGrounding(
            self.slope_shrinkage, intercept=self.intercept
        ).fit(train)
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        corrected = self._grounding.transform(x)
        point = masked_average(corrected, x.availability)
        return BlendResult(point=finalize_point(point, self._kind, self._variable))

    def to_state(self) -> dict[str, object]:
        """Compact observability state: the fitted grounding coefficients."""
        return {"method_id": self.method_id, "grounding": self._grounding.to_state()}


@dataclass
class InverseErrorWeights:
    """Grounded sources weighted by inverse training error per lead bucket.

    ``loss_power`` selects the tracked loss: 2 is the classic inverse-MSE
    Bates-Granger weighting, 1 the MAE-consistent variant.
    """

    method_id: str = "inverse_mse"
    slope_shrinkage: float = BIAS_ONLY
    loss_power: float = 2.0
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _grounding: AffineGrounding = field(default_factory=AffineGrounding)

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._grounding = AffineGrounding(self.slope_shrinkage).fit(train)
        corrected = self._grounding.transform(train.x)
        errors = np.abs(corrected - train.y[:, np.newaxis]) ** self.loss_power
        availability = train.x.availability

        def fit_weights(rows: np.ndarray) -> np.ndarray:
            counts = availability[rows].sum(axis=0)
            sums = np.where(availability[rows], errors[rows], 0.0).sum(axis=0)
            mean_loss = np.where(
                counts >= _MIN_ROWS_PER_SOURCE,
                sums / np.maximum(counts, 1),
                np.nan,
            )
            fallback = (
                float(np.nanmax(mean_loss)) if np.isfinite(mean_loss).any() else 1.0
            )
            mean_loss = np.where(np.isnan(mean_loss), fallback, mean_loss)
            return 1.0 / np.maximum(mean_loss, _MIN_LOSS)

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
        return BlendResult(point=finalize_point(point, self._kind, self._variable))

    def to_state(self) -> dict[str, object]:
        """Compact observability state: grounding plus per-bucket weights."""
        return {
            "method_id": self.method_id,
            "grounding": self._grounding.to_state(),
            "weights": {
                "global": self._fitted.global_state.tolist(),
                "buckets": {
                    label: weights.tolist()
                    for label, weights in self._fitted.states.items()
                },
            },
        }


def _affine_equal_weight() -> GroundedEqualWeight:
    """Free-slope grounding, kept on the leaderboard beside the bias-only
    default so the archive can say which correction it actually supports."""
    return GroundedEqualWeight(
        method_id="affine_equal_weight", slope_shrinkage=FREE_SLOPE
    )


def _grounded_median_equal_weight() -> GroundedEqualWeight:
    return GroundedEqualWeight(
        method_id="grounded_median_equal_weight", intercept="median"
    )


def _inverse_mae() -> InverseErrorWeights:
    return InverseErrorWeights(method_id="inverse_mae", loss_power=1.0)


register("grounded_equal_weight", GroundedEqualWeight)
register("affine_equal_weight", _affine_equal_weight)
register("grounded_median_equal_weight", _grounded_median_equal_weight)
register("inverse_mse", InverseErrorWeights)
register("inverse_mae", _inverse_mae)
