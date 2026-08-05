"""Climatology-damped blending: regression toward climatology at long leads.

Deep into week two the providers that still reach a lead bucket are few and
correlated, and their shared bias goes uncorrected; the harmonic climatology
is unbiased by construction but ignores the weather. The classical remedy is
lead-dependent damping — serve ``alpha * blend + (1 - alpha) * climatology``
with alpha fitted per lead bucket — so the product glides from provider
signal at short leads to seasonal structure where provider skill dies,
instead of serving provider noise at full weight (Monhart et al. 2018 on
lead-dependent station corrections; the harmonic method's standing 240h+
leaderboard win is the local evidence). Alpha is a direct MAE grid search
per bucket with empirical-Bayes shrinkage toward the global fit, and the
plain blend is recovered exactly at the ``alpha = 1`` boundary.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np

from grounded_weather_forecast.blenders.baselines import HarmonicClimatology
from grounded_weather_forecast.blenders.combine import GroundedEqualWeight
from grounded_weather_forecast.blenders.protocol import (
    FittedBuckets,
    PerBucketFitter,
    finalize_point,
)
from grounded_weather_forecast.blenders.registry import BlenderFactory, register
from grounded_weather_forecast.contracts import (
    Blender,
    BlendResult,
    ForecastMatrix,
    SupervisedSlice,
    TargetKind,
    VariableSpec,
)
from grounded_weather_forecast.leads import buckets_for_product

# Descending, so argmin ties resolve toward the undamped blend.
_ALPHA_GRID = np.linspace(1.0, 0.0, 21)


def _blend_alpha(local: float, global_state: float, weight: float) -> float:
    return weight * local + (1.0 - weight) * global_state


@dataclass
class DampedBlend:
    """``alpha * base + (1 - alpha) * climatology`` with per-bucket alpha."""

    base_factory: BlenderFactory
    method_id: str = "damped_grounded_equal_weight"
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._base = self.base_factory().fit(train)
        self._climatology = HarmonicClimatology().fit(train)
        base_point = self._base.predict(train.x).point
        climatology_point = self._climatology.predict(train.x).point
        y = train.y

        def fit_alpha(rows: np.ndarray) -> float:
            usable = np.isfinite(base_point[rows]) & np.isfinite(
                climatology_point[rows]
            )
            if not usable.any():
                return 0.0  # no base signal in this bucket: pure climatology
            base_residual = (base_point[rows] - y[rows])[usable]
            climatology_residual = (climatology_point[rows] - y[rows])[usable]
            losses = [
                float(
                    np.mean(
                        np.abs(
                            alpha * base_residual + (1.0 - alpha) * climatology_residual
                        )
                    )
                )
                for alpha in _ALPHA_GRID
            ]
            return float(_ALPHA_GRID[int(np.argmin(losses))])

        fitter = PerBucketFitter[float](
            buckets=buckets_for_product(train.x.product),
            fit_one=fit_alpha,
            blend=_blend_alpha,
        )
        self._fitted: FittedBuckets[float] = fitter.fit(train.x.lead_hours)
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        base_point = self._base.predict(x).point
        climatology_point = self._climatology.predict(x).point
        point = np.full(x.n_rows, np.nan)

        def use(alpha: float, rows: np.ndarray) -> None:
            damped = alpha * base_point[rows] + (1.0 - alpha) * climatology_point[rows]
            # A row without base signal degrades to pure climatology — the
            # method's own alpha -> 0 limit — rather than "no opinion".
            point[rows] = np.where(np.isfinite(damped), damped, climatology_point[rows])

        self._fitted.apply(x.lead_hours, use)
        return BlendResult(point=finalize_point(point, self._kind, self._variable))

    def to_state(self) -> dict[str, object]:
        """Per-bucket damping weights; ``alpha = 1`` is the undamped blend."""
        return {
            "method_id": self.method_id,
            "alpha": {
                "global": self._fitted.global_state,
                "buckets": dict(self._fitted.states),
            },
        }


def _damped_grounded_equal_weight() -> Blender:
    return DampedBlend(GroundedEqualWeight, "damped_grounded_equal_weight")


register("damped_grounded_equal_weight", _damped_grounded_equal_weight)

_PROTOCOL_CHECK: tuple[type[Blender], ...] = (DampedBlend,)
