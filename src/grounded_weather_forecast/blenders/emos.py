"""EMOS/NGR: the first calibrated predictive distribution.

Ensemble Model Output Statistics (Gneiting et al. 2005): a Gaussian predictive
distribution whose mean is an affine function of the base blend and whose
spread is linked to ensemble dispersion. Unbounded variables minimize the
closed-form Gaussian CRPS. Bounded variables optimize weighted truncated-normal
likelihood, matching the family used for their emitted quantiles.

The spread predictor prefers real ensemble standard deviations (the ``ens__*``
statistics from the Ensemble API ingest) and falls back to the cross-provider
spread — which is structurally under-dispersed (shared parents), so with the
fallback expect the fitted link to lean on its intercept. Variables bounded
below (wind) use a truncated normal so no probability mass sits on impossible
values.
"""

from dataclasses import dataclass
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

QUANTILE_LEVELS: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))
_RECENCY_SCALE_DAYS = 45.0
_MIN_FIT_ROWS = 60
_MIN_SIGMA = 1e-3
_SQRT_PI = float(np.sqrt(np.pi))


def _normal_pdf(z: FloatArray) -> FloatArray:
    return np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)


def _normal_cdf(z: FloatArray) -> FloatArray:
    special = import_module("scipy.special")
    return 0.5 * (1.0 + special.erf(z / np.sqrt(2.0)))


def gaussian_crps(y: FloatArray, mu: FloatArray, sigma: FloatArray) -> FloatArray:
    """Closed-form CRPS of a normal predictive distribution."""
    z = (y - mu) / sigma
    return sigma * (
        z * (2.0 * _normal_cdf(z) - 1.0) + 2.0 * _normal_pdf(z) - 1.0 / _SQRT_PI
    )


def _spread(x: ForecastMatrix, variable_name: str) -> FloatArray:
    """Per-row dispersion: real ensemble sd where ingested, provider sd else."""
    sd_columns = [
        c
        for c in x.features.columns
        if c.startswith("ens__") and c.endswith(f"__{variable_name}__sd")
    ]
    with np.errstate(invalid="ignore"):
        provider_sd = np.nanstd(np.where(x.availability, x.values, np.nan), axis=1)
    provider_sd = np.nan_to_num(provider_sd, nan=0.0)
    if sd_columns:
        block = (
            x.features.select(sd_columns)
            .cast(dict.fromkeys(sd_columns, float))  # type: ignore[arg-type]
            .to_numpy()
            .astype(np.float64)
        )
        with np.errstate(invalid="ignore"):
            ensemble_sd = np.nanmean(block, axis=1)
        return np.where(np.isfinite(ensemble_sd), ensemble_sd, provider_sd)
    return provider_sd


def _recency_weights(x: ForecastMatrix) -> FloatArray:
    if "issue_time" not in x.features.columns:
        return np.ones(x.n_rows)
    issue = x.features["issue_time"].to_numpy()
    age_days = (issue.max() - issue).astype("timedelta64[s]").astype(
        np.float64
    ) / 86400.0
    return np.exp(-age_days / _RECENCY_SCALE_DAYS)


@dataclass
class Emos:
    """CRPS-fit Gaussian head on any base blend."""

    base_factory: BlenderFactory
    method_id: str = "emos"
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _parameters: FloatArray | None = None
    _fit_family: str = "gaussian"

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
            return self
        y = train.y[scored]
        base = base_point[scored]
        log_spread = np.log(np.maximum(spread[scored], _MIN_SIGMA))
        w = weights[scored] / weights[scored].sum()
        residual_sd = max(float(np.std(y - base)), _MIN_SIGMA)

        minimum = train.variable.minimum
        maximum = train.variable.maximum
        bounded = minimum is not None or maximum is not None
        stats = import_module("scipy.stats") if bounded else None
        self._fit_family = "truncated_normal" if bounded else "gaussian"

        def loss(parameters: FloatArray) -> float:
            a, b, c, d = parameters
            mu = a + b * base
            sigma = np.maximum(np.exp(c + d * log_spread), _MIN_SIGMA)
            if stats is not None:
                lower = -np.inf if minimum is None else (minimum - mu) / sigma
                upper = np.inf if maximum is None else (maximum - mu) / sigma
                log_likelihood = stats.truncnorm.logpdf(
                    y, lower, upper, loc=mu, scale=sigma
                )
                if not np.isfinite(log_likelihood).all():
                    return float("inf")
                return float(-(w * log_likelihood).sum())
            return float((w * gaussian_crps(y, mu, sigma)).sum())

        optimize = import_module("scipy.optimize")
        result = optimize.minimize(
            loss,
            np.array([0.0, 1.0, np.log(residual_sd), 0.0]),
            method="Nelder-Mead",
            options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-6},
        )
        self._parameters = np.asarray(result.x, dtype=np.float64)
        return self

    def _distribution(self, x: ForecastMatrix) -> tuple[FloatArray, FloatArray]:
        base_point = self._base.predict(x).point
        if self._parameters is None:  # pragma: no cover - guarded by predict
            msg = "EMOS parameters missing; fit before predict"
            raise RuntimeError(msg)
        a, b, c, d = self._parameters
        mu = a + b * base_point
        variable_name = self._variable.name if self._variable else ""
        log_spread = np.log(np.maximum(_spread(x, variable_name), _MIN_SIGMA))
        sigma = np.maximum(np.exp(c + d * log_spread), _MIN_SIGMA)
        return mu, sigma

    def _quantiles(self, mu: FloatArray, sigma: FloatArray) -> FloatArray:
        stats = import_module("scipy.stats")
        levels = np.asarray(QUANTILE_LEVELS)
        minimum = self._variable.minimum if self._variable else None
        maximum = self._variable.maximum if self._variable else None
        if minimum is not None or maximum is not None:
            a = -np.inf if minimum is None else (minimum - mu) / sigma
            b = np.inf if maximum is None else (maximum - mu) / sigma
            return np.column_stack(
                [
                    stats.truncnorm.ppf(level, a, b, loc=mu, scale=sigma)
                    for level in levels
                ]
            )
        return mu[:, np.newaxis] + sigma[:, np.newaxis] * stats.norm.ppf(levels)

    def predict(self, x: ForecastMatrix) -> BlendResult:
        if self._parameters is None:
            base_point = self._base.predict(x).point
            return BlendResult(
                point=finalize_point(base_point, self._kind, self._variable)
            )
        mu, sigma = self._distribution(x)
        quantiles = finalize_quantiles(
            self._quantiles(mu, sigma), self._kind, self._variable
        )
        median = quantiles[:, len(QUANTILE_LEVELS) // 2]
        return BlendResult(
            point=finalize_point(median, self._kind, self._variable),
            quantiles=quantiles,
            quantile_levels=QUANTILE_LEVELS,
        )


def _emos() -> Blender:
    return Emos(GroundedEqualWeight, "emos")


register("emos", _emos)
