"""CSGD-EMOS: the first honest quantile emitter for precipitation.

Scheuerer & Hamill (2015): daily/hourly precipitation follows a gamma
distribution shifted left by ``delta < 0`` and censored at zero — the
censored mass IS the dry probability, so occurrence and amount come from
one coherent distribution instead of a Gaussian head clamped at zero
(which is why ``emos``/``idr`` are excluded from precip). The mean links
affinely to the base blend and the spread to log ensemble dispersion,
exactly the ``emos`` scaffold; the fit maximizes the weighted censored
log-likelihood (zero observations contribute ``log F(0)``, wet ones the
shifted-gamma density).

A dry training window cannot identify an amount distribution: below
``_MIN_WET_ROWS`` wet observations the head abstains to the base point,
recorded in ``fit_status`` — the same honesty contract as the GBM floor.
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Self

import numpy as np

from grounded_weather_forecast.blenders.combine import GroundedEqualWeight
from grounded_weather_forecast.blenders.emos import (
    _minimize,
    _recency_weights,
    _spread,
)
from grounded_weather_forecast.blenders.protocol import (
    coefficient_state,
    finalize_point,
    quantile_blend_result,
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
_MIN_FIT_ROWS = 60
_MIN_WET_ROWS = 10
_MIN_SIGMA = 1e-3
_MIN_SHAPE_DISTANCE = 1e-3
_WET_EPS = 1e-9


def _gamma_parameters(
    mu: FloatArray, sigma: FloatArray, delta: float
) -> tuple[FloatArray, FloatArray]:
    """Shape/scale of the shifted gamma with mean ``mu`` and sd ``sigma``."""
    distance = np.maximum(mu - delta, _MIN_SHAPE_DISTANCE)
    sigma = np.maximum(sigma, _MIN_SIGMA)
    shape = (distance / sigma) ** 2
    scale = sigma**2 / distance
    return shape, scale


@dataclass
class CsgdEmos:
    """Censored shifted-gamma head on any base blend."""

    base_factory: BlenderFactory
    method_id: str = "csgd_emos"
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _parameters: FloatArray | None = None
    _fit_status: str = "unfitted"

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._base = self.base_factory().fit(train)
        base_point = self._base.predict(train.x).point
        spread = _spread(train.x, train.variable.name)
        weights = _recency_weights(train.x)
        scored = np.isfinite(base_point) & np.isfinite(train.y)
        if int(scored.sum()) < _MIN_FIT_ROWS:
            self._parameters = None
            self._fit_status = "insufficient_rows"
            return self
        y = train.y[scored]
        wet = y > _WET_EPS
        if int(wet.sum()) < _MIN_WET_ROWS:
            self._parameters = None
            self._fit_status = "insufficient_wet_rows"
            return self
        base = base_point[scored]
        log_spread = np.log(np.maximum(spread[scored], _MIN_SIGMA))
        w = weights[scored] / weights[scored].sum()
        stats = import_module("scipy.stats")
        wet_sd = max(float(np.std(y[wet])), _MIN_SIGMA)
        initial = np.array([0.0, 1.0, np.log(wet_sd), 0.0, -0.1])

        def censored_loss(parameters: FloatArray) -> float:
            a, b, c, d, delta = parameters
            if delta >= 0.0:
                # The shift must sit below zero or the censored (dry) mass
                # vanishes and every dry observation has zero likelihood.
                return float("inf")
            mu = a + b * base
            sigma = np.exp(c + d * log_spread)
            shape, scale = _gamma_parameters(mu, sigma, delta)
            log_likelihood = np.where(
                y > _WET_EPS,
                stats.gamma.logpdf(y - delta, shape, scale=scale),
                stats.gamma.logcdf(-delta, shape, scale=scale),
            )
            if not np.isfinite(log_likelihood).all():
                return float("inf")
            return float(-(w * log_likelihood).sum())

        # Five parameters need a bigger simplex budget than the 4-parameter
        # Gaussian head's default.
        self._parameters = _minimize(censored_loss, initial, max_iterations=1500)
        self._fit_status = "unfitted" if self._parameters is None else "converged"
        return self

    def _distribution(self, x: ForecastMatrix) -> tuple[FloatArray, FloatArray, float]:
        base_point = self._base.predict(x).point
        if self._parameters is None:  # pragma: no cover - guarded by predict
            msg = "CSGD parameters missing; fit before predict"
            raise RuntimeError(msg)
        a, b, c, d, delta = self._parameters
        mu = a + b * base_point
        variable_name = self._variable.name if self._variable else ""
        log_spread = np.log(np.maximum(_spread(x, variable_name), _MIN_SIGMA))
        sigma = np.exp(c + d * log_spread)
        shape, scale = _gamma_parameters(mu, sigma, float(delta))
        return shape, scale, float(delta)

    def predict(self, x: ForecastMatrix) -> BlendResult:
        if self._parameters is None:
            base_point = self._base.predict(x).point
            return BlendResult(
                point=finalize_point(base_point, self._kind, self._variable)
            )
        stats = import_module("scipy.stats")
        shape, scale, delta = self._distribution(x)
        levels = np.asarray(QUANTILE_LEVELS)
        quantiles = np.maximum(
            delta
            + np.column_stack(
                [stats.gamma.ppf(level, shape, scale=scale) for level in levels]
            ),
            0.0,
        )
        return quantile_blend_result(
            quantiles, QUANTILE_LEVELS, self._kind, self._variable
        )

    def to_state(self) -> dict[str, object]:
        return coefficient_state(
            self.method_id,
            self._variable,
            self._fit_status,
            self._parameters,
            (
                "mean_intercept",
                "mean_slope",
                "log_sigma_intercept",
                "log_sigma_slope",
                "shift",
            ),
        )


def _csgd_emos() -> Blender:
    return CsgdEmos(GroundedEqualWeight, "csgd_emos")


register(
    "csgd_emos",
    _csgd_emos,
    variables=frozenset({"precip_mm", "precip_sum_mm"}),
)
