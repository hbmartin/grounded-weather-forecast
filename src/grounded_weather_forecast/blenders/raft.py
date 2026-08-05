"""RAFT-style anchoring: a freely fitted response to the verified residual.

Schuhen, Thorarinsdottir & Lenkoski (2020): when an hour verifies, the
remaining hours of the trajectory are updated by a *regression* of future
error on the just-observed error — the lead-pair error correlation decides
how much of the residual persists, not an assumed exponential form. The
measured 0-1h gap (persistence 1.129 °C vs 1.46-1.48 for every anchored
method) is exactly what a mis-specified decay shape looks like.

Within the batch blender protocol the freshest verified signal is the
issue-time residual the anchored family already extracts
(``anchoring.issue_residuals``); what changes is the response: a per-bucket
through-origin regression slope, empirical-Bayes shrunk, that may exceed 1,
flip sign, or die instantly — none of which ``exp(-lead/tau)`` can express.
"""

from dataclasses import dataclass, field
from typing import Self

import numpy as np

from grounded_weather_forecast.blenders.anchoring import issue_residuals
from grounded_weather_forecast.blenders.combine import GroundedEqualWeight
from grounded_weather_forecast.blenders.protocol import (
    FittedBuckets,
    PerBucketFitter,
    finalize_point,
)
from grounded_weather_forecast.blenders.registry import register
from grounded_weather_forecast.contracts import (
    Blender,
    BlendResult,
    ForecastMatrix,
    SupervisedSlice,
    TargetKind,
    VariableSpec,
    obs_col,
)
from grounded_weather_forecast.leads import buckets_for_product

_SLOPE_MIN = -0.5
_SLOPE_MAX = 1.5


@dataclass
class RaftGrounded:
    """Grounded equal weight plus a per-bucket fitted residual response."""

    method_id: str = "raft_grounded"
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _fitted: FittedBuckets[float] | None = field(default=None, repr=False)

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._observation_column = obs_col(train.variable.name)
        self._base = GroundedEqualWeight().fit(train)
        base_point = self._base.predict(train.x).point
        residuals = issue_residuals(train.x, base_point, self._observation_column)
        error = train.y - base_point
        usable = np.isfinite(error) & np.isfinite(residuals) & (np.abs(residuals) > 0.0)
        if not usable.any():
            self._fitted = None
            return self

        def fit_one(rows: np.ndarray) -> float:
            selected = rows[usable[rows]]
            if selected.shape[0] == 0:
                return 0.0
            r0 = residuals[selected]
            denominator = float((r0 * r0).sum())
            if denominator <= 0.0:
                return 0.0
            slope = float((error[selected] * r0).sum()) / denominator
            return float(np.clip(slope, _SLOPE_MIN, _SLOPE_MAX))

        fitter = PerBucketFitter(
            buckets=buckets_for_product(train.x.product),
            fit_one=fit_one,
            blend=lambda local, glob, w: w * local + (1.0 - w) * glob,
        )
        self._fitted = fitter.fit(train.x.lead_hours)
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        base_point = self._base.predict(x).point
        fitted = self._fitted
        if fitted is None:
            return BlendResult(
                point=finalize_point(base_point, self._kind, self._variable)
            )
        residuals = issue_residuals(x, base_point, self._observation_column)
        correction = np.nan_to_num(residuals, nan=0.0)
        slopes = np.zeros(x.n_rows)

        def use(slope: float, rows: np.ndarray) -> None:
            slopes[rows] = slope

        fitted.apply(x.lead_hours, use)
        return BlendResult(
            point=finalize_point(
                base_point + slopes * correction, self._kind, self._variable
            )
        )

    def to_state(self) -> dict[str, object]:
        fitted = self._fitted
        return {
            "schema_version": 1,
            "method_id": self.method_id,
            "slopes": None
            if fitted is None
            else {
                "global": float(fitted.global_state),
                "buckets": {
                    label: float(state)
                    for label, state in sorted(fitted.states.items())
                },
            },
        }


def _raft_grounded() -> Blender:
    return RaftGrounded()


register("raft_grounded", _raft_grounded)
