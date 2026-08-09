"""PoP recalibration heads: online-Platt vs beta calibration, the wet-season A/B.

Vendor probability-of-precipitation is a probability-shaped score, not a
calibrated probability — the 2026-07 monsoon burst verified at PoP 1.0 while
no provider exceeded 0.3. Two mutually exclusive recalibration maps exist
and both are registered so the Brier column arbitrates as wet-season labels
arrive (the compare-and-contrast principle):

- ``pop_platt`` (Gupta & Ramdas, calibeating): two-parameter logistic on the
  log-odds, ``sigmoid(a + b * logit(p))``.
- ``pop_beta`` (Kull, Silva Filho & Flach): three-parameter logistic on
  ``(ln p, -ln(1-p))`` — fixes Platt's mis-shape when the input is itself a
  bounded, skewed probability, which is precisely what vendor PoP is.

Both consume the equal-weight mean of the provider PoP columns (a dry season
gives no evidence to pick a best PoP provider), fit per lead bucket with the
shared empirical-Bayes shrinkage, and are point-only: quantiles make no
sense for a Bernoulli parameter. The separation guard matters more than the
optimizer here — a rainless training window has one class, and the identity
map must serve until the sky provides labels.
"""

from dataclasses import dataclass, field
from typing import Self

import numpy as np

from grounded_weather_forecast.blenders.protocol import (
    FittedBuckets,
    finalize_point,
    fit_shrunk_buckets,
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

_EPS = 1e-4
_MAX_COEF = 8.0  # separation guard: a one-class window cannot run away
_MIN_EACH_CLASS = 5
_NEWTON_STEPS = 25
_RIDGE = 1e-4
_WET_THRESHOLD = 0.5  # PoP truth is binary occurrence


def _clip_probability(p: FloatArray) -> FloatArray:
    return np.clip(p, _EPS, 1.0 - _EPS)


def _platt_features(p: FloatArray) -> FloatArray:
    clipped = _clip_probability(p)
    return np.column_stack([np.ones_like(clipped), np.log(clipped / (1.0 - clipped))])


def _beta_features(p: FloatArray) -> FloatArray:
    clipped = _clip_probability(p)
    return np.column_stack(
        [np.ones_like(clipped), np.log(clipped), -np.log(1.0 - clipped)]
    )


_IDENTITY = {
    "pop_platt": np.array([0.0, 1.0]),
    # a=0, b=c=1: sigmoid(ln p - ln(1-p)) = p
    "pop_beta": np.array([0.0, 1.0, 1.0]),
}
_FEATURES = {"pop_platt": _platt_features, "pop_beta": _beta_features}


def _penalized_log_loss(
    features: FloatArray, y: FloatArray, coefficients: FloatArray
) -> float:
    logits = np.clip(features @ coefficients, -30.0, 30.0)
    predicted = np.clip(1.0 / (1.0 + np.exp(-logits)), 1e-12, 1.0 - 1e-12)
    penalty = 0.5 * _RIDGE * float(coefficients @ coefficients)
    return float(
        -(y * np.log(predicted) + (1.0 - y) * np.log(1.0 - predicted)).sum() + penalty
    )


def _fit_logistic(
    features: FloatArray, y: FloatArray, identity: FloatArray
) -> FloatArray:
    """Damped Newton on ridge log-loss with an identity fallback.

    The backtracking line search is load-bearing: a full Newton step from
    the identity start overshoots on realistic bucket sizes and the raw
    iteration oscillates to +-1e5 instead of converging.
    """
    wet = int((y > _WET_THRESHOLD).sum())
    dry = int(y.shape[0] - wet)
    if wet < _MIN_EACH_CLASS or dry < _MIN_EACH_CLASS:
        return identity.copy()
    coefficients = identity.astype(np.float64).copy()
    for _ in range(_NEWTON_STEPS):
        logits = features @ coefficients
        predicted = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        gradient = features.T @ (predicted - y) + _RIDGE * coefficients
        weights = np.maximum(predicted * (1.0 - predicted), 1e-6)
        hessian = (features * weights[:, np.newaxis]).T @ features
        hessian += _RIDGE * np.eye(features.shape[1])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return identity.copy()
        current = _penalized_log_loss(features, y, coefficients)
        scale = 1.0
        for _halving in range(20):
            trial = coefficients - scale * step
            if _penalized_log_loss(features, y, trial) <= current:
                coefficients = trial
                break
            scale *= 0.5
        else:
            break  # no improving step remains: converged or stuck
        if float(np.max(np.abs(scale * step))) < 1e-8:
            break
    if not np.isfinite(coefficients).all():
        return identity.copy()
    return np.clip(coefficients, -_MAX_COEF, _MAX_COEF)


@dataclass
class PopCalibrator:
    """Per-bucket logistic recalibration of the provider-mean PoP."""

    method_id: str
    _variable: VariableSpec | None = None
    _fitted: FittedBuckets[FloatArray] | None = field(default=None, repr=False)

    def fit(self, train: SupervisedSlice) -> Self:
        self._variable = train.variable
        base = masked_average(train.x.values, train.x.availability)
        y = train.y
        usable = np.isfinite(base) & np.isfinite(y)
        features_fn = _FEATURES[self.method_id]
        identity = _IDENTITY[self.method_id]

        def fit_one(rows: np.ndarray) -> FloatArray:
            selected = rows[usable[rows]]
            if selected.shape[0] == 0:
                return identity.copy()
            return _fit_logistic(
                features_fn(base[selected]),
                (y[selected] > _WET_THRESHOLD).astype(np.float64),
                identity,
            )

        self._fitted = fit_shrunk_buckets(train.x.product, train.x.lead_hours, fit_one)
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        base = masked_average(x.values, x.availability)
        point = np.full(x.n_rows, np.nan)
        features_fn = _FEATURES[self.method_id]
        valid = np.isfinite(base)
        fitted = self._fitted
        if fitted is None or not valid.any():
            return BlendResult(
                point=finalize_point(base, TargetKind.PROBABILITY, self._variable)
            )

        def use(coefficients: FloatArray, rows: np.ndarray) -> None:
            selected = rows[valid[rows]]
            if selected.shape[0] == 0:
                return
            logits = features_fn(base[selected]) @ coefficients
            point[selected] = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

        fitted.apply(x.lead_hours, use)
        return BlendResult(
            point=finalize_point(point, TargetKind.PROBABILITY, self._variable)
        )

    def to_state(self) -> dict[str, object]:
        fitted = self._fitted
        return {
            "schema_version": 1,
            "coefficients": {
                "global": None
                if fitted is None
                else list(np.asarray(fitted.global_state)),
                "buckets": {}
                if fitted is None
                else {
                    label: list(np.asarray(state))
                    for label, state in sorted(fitted.states.items())
                },
            },
        }


def _pop_platt() -> Blender:
    return PopCalibrator("pop_platt")


def _pop_beta() -> Blender:
    return PopCalibrator("pop_beta")


register("pop_platt", _pop_platt, variables=frozenset({"pop"}))
register("pop_beta", _pop_beta, variables=frozenset({"pop"}))
