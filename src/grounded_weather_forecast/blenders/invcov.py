"""Inverse-covariance (GLS) weights: the honest test of diagonal-only.

Timmermann's advice — covariances are too hard to estimate to be worth
estimating — is a load-bearing assumption of every ``inverse_*`` combiner
here, while the measured error-correlation matrix (nws/pirate_weather/
visual_crossing at rho 0.96-0.97, k_eff ~1.8) argues the off-diagonals are
exactly where the information is. This method runs the counterfactual:
minimum-variance weights ``w proportional to Sigma^-1 1`` under a
Ledoit-Wolf-shrunk error covariance, with deviations from 1/k capped so an
ill-conditioned estimate cannot short-sell a provider into oblivion.

The literature expects it to lose at this archive size — that is the
point: future-work #12 wants the loss measured, not assumed.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np

from grounded_weather_forecast.blenders.protocol import (
    finalize_point,
    masked_average,
)
from grounded_weather_forecast.blenders.registry import register
from grounded_weather_forecast.contracts import (
    Blender,
    BlendResult,
    FloatArray,
    ForecastMatrix,
    SupervisedSlice,
    TargetKind,
    VariableSpec,
)

_MIN_COMPLETE_ROWS = 30


def ledoit_wolf_covariance(errors: FloatArray) -> FloatArray:
    """Sample covariance shrunk toward the scaled identity (Ledoit-Wolf 2004)."""
    n, k = errors.shape
    centered = errors - errors.mean(axis=0, keepdims=True)
    sample = centered.T @ centered / n
    mu = float(np.trace(sample)) / k
    target = mu * np.eye(k)
    distance = float(((sample - target) ** 2).sum())
    if distance <= 0.0:
        return target
    beta_sum = 0.0
    for row in range(n):
        outer = np.outer(centered[row], centered[row])
        beta_sum += float(((outer - sample) ** 2).sum())
    shrinkage = min(1.0, (beta_sum / n**2) / distance)
    return shrinkage * target + (1.0 - shrinkage) * sample


def gls_weights(errors: FloatArray) -> FloatArray:
    """Capped minimum-variance weights from the shrunk error covariance."""
    k = errors.shape[1]
    covariance = ledoit_wolf_covariance(errors)
    try:
        raw = np.linalg.solve(covariance, np.ones(k))
    except np.linalg.LinAlgError:
        return np.full(k, 1.0 / k)
    total = float(raw.sum())
    if total <= 0.0:
        return np.full(k, 1.0 / k)
    weights = raw / total
    return _capped_simplex(np.maximum(weights, 0.0), cap=2.0 / k)


def _capped_simplex(weights: FloatArray, cap: float) -> FloatArray:
    """Project onto the simplex with a per-weight ceiling.

    A plain clip-then-renormalize lets the renormalization push a capped
    weight straight back over the ceiling; fixing capped entries and
    redistributing the remainder among the free ones terminates in at most
    k passes.
    """
    k = weights.shape[0]
    if k * cap <= 1.0:
        return np.full(k, 1.0 / k)
    capped = np.zeros(k, dtype=bool)
    result = weights.copy()
    for _ in range(k):
        free = ~capped
        free_total = float(result[free].sum())
        remaining = 1.0 - cap * float(capped.sum())
        if free_total <= 0.0:
            result[free] = remaining / max(int(free.sum()), 1)
        else:
            result[free] = result[free] * remaining / free_total
        over = free & (result > cap)
        if not over.any():
            break
        result[over] = cap
        capped |= over
    return result


@dataclass
class InverseCovariance:
    """Equal weight's GLS counterfactual over the shrunk error covariance."""

    method_id: str = "inverse_covariance"
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _weights_by_source: dict[str, float] | None = None

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        # Never-observed columns carry no covariance information and must not
        # shrink the complete-case set (the all-NaN-source invariance).
        observed = np.isfinite(train.x.values).any(axis=0)
        errors = train.x.values[:, observed] - train.y[:, np.newaxis]
        complete = np.isfinite(errors).all(axis=1)
        if int(complete.sum()) < _MIN_COMPLETE_ROWS or int(observed.sum()) < 2:
            # Pairwise-incomplete archives cannot support a joint covariance;
            # serving 1/k here would just duplicate equal_weight, so abstain.
            self._weights_by_source = None
            return self
        weights = gls_weights(errors[complete])
        names = [
            source
            for source, seen in zip(train.x.sources, observed, strict=True)
            if seen
        ]
        self._weights_by_source = {
            source: float(weight) for source, weight in zip(names, weights, strict=True)
        }
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        if self._weights_by_source is None:
            point = np.full(x.n_rows, np.nan)
        else:
            weights = np.asarray(
                [self._weights_by_source.get(source, 0.0) for source in x.sources]
            )
            if float(weights.sum()) <= 0.0:
                point = np.full(x.n_rows, np.nan)
            else:
                point = masked_average(x.values, x.availability, weights=weights)
        return BlendResult(point=finalize_point(point, self._kind, self._variable))

    def to_state(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "method_id": self.method_id,
            "weights": self._weights_by_source,
        }


def _inverse_covariance() -> Blender:
    return InverseCovariance()


register("inverse_covariance", _inverse_covariance)
