"""SAMOS-informed grounding: a ridge-fit bias curve in solar/annual phase.

The batch challenger to the adaptive EWMA grounding. Instead of learning one
scalar intercept per (source, lead bucket) — which averages the diurnal cycle
away — the intercept becomes a smooth curve over deterministic phase
features: sine of solar elevation (approximately season-invariant, the
standardized-anomaly trick from the mountain-postprocessing literature) plus
annual and semiannual day-of-year harmonics for the residual seasonal drift.
The slope stays fixed at 1 (bias-only, per ADR 0004): the *intercept* becomes
phase-dependent, nothing buys the slope back.

The ridge (intercept unpenalized) is what makes a short archive safe: over a
three-month arc the annual harmonics are nearly collinear and unpenalized
least squares would fit huge cancelling coefficients — the exact failure the
climatology baseline exhibited before its ridge (Limitations §4.3).
"""

from dataclasses import dataclass, field
from typing import Self

import numpy as np

from grounded_weather_forecast.blenders.protocol import (
    FittedBuckets,
    PerBucketFitter,
    finalize_point,
    masked_average,
)
from grounded_weather_forecast.blenders.registry import register
from grounded_weather_forecast.contracts import (
    BlendResult,
    FloatArray,
    ForecastMatrix,
    SupervisedSlice,
    TargetKind,
    VariableSpec,
)
from grounded_weather_forecast.leads import buckets_for_product

_RIDGE = 1.0
_MIN_FIT_ROWS = 48


def _phase_design(x: ForecastMatrix) -> tuple[FloatArray, tuple[str, ...]]:
    """Design matrix and its semantic column identity.

    Degrades gracefully: without any context columns it is a lone intercept,
    i.e. exactly the scalar bias-only correction.
    """
    columns: list[FloatArray] = [np.ones(x.n_rows)]
    names = ["intercept"]
    features = x.features
    if "solar_elevation_deg" in features.columns:
        sin_elevation = np.sin(
            np.deg2rad(features["solar_elevation_deg"].cast(float).to_numpy())
        )
        columns += [sin_elevation, sin_elevation**2]
        names += ["sin_solar_elevation", "sin_solar_elevation_squared"]
    elif {"hour_sin", "hour_cos"} <= set(features.columns):
        columns += [
            features["hour_sin"].cast(float).to_numpy(),
            features["hour_cos"].cast(float).to_numpy(),
        ]
        names += ["hour_sin", "hour_cos"]
    if {"doy_sin", "doy_cos"} <= set(features.columns):
        doy_sin = features["doy_sin"].cast(float).to_numpy()
        doy_cos = features["doy_cos"].cast(float).to_numpy()
        columns += [
            doy_sin,
            doy_cos,
            2.0 * doy_sin * doy_cos,  # semiannual sin
            doy_cos**2 - doy_sin**2,  # semiannual cos
        ]
        names += ["doy_sin", "doy_cos", "semiannual_sin", "semiannual_cos"]
    return np.column_stack(columns), tuple(names)


def _ridge_fit(design: FloatArray, residual: FloatArray) -> FloatArray:
    """Ridge with an unpenalized intercept, via the augmented system."""
    penalty = np.eye(design.shape[1]) * np.sqrt(_RIDGE)
    penalty[0, 0] = 0.0
    augmented = np.vstack([design, penalty])
    target = np.concatenate([residual, np.zeros(design.shape[1])])
    coefficients, *_ = np.linalg.lstsq(augmented, target, rcond=None)
    return coefficients


@dataclass
class HarmonicGrounding:
    """Per-(source, lead bucket) phase-dependent bias curves."""

    _by_source: dict[str, FittedBuckets[FloatArray]] = field(default_factory=dict)
    _feature_names: tuple[str, ...] = ("intercept",)

    def fit(self, train: SupervisedSlice) -> Self:
        design, self._feature_names = _phase_design(train.x)
        buckets = buckets_for_product(train.x.product)
        residuals = train.y[:, np.newaxis] - train.x.values
        for index, source in enumerate(train.x.sources):

            def fit_one(rows: np.ndarray, index: int = index) -> FloatArray:
                available = rows[train.x.availability[rows, index]]
                if available.shape[0] < _MIN_FIT_ROWS:
                    return np.zeros(design.shape[1])
                return _ridge_fit(design[available], residuals[available, index])

            fitter = PerBucketFitter[FloatArray](
                buckets=buckets, fit_one=fit_one, min_rows=_MIN_FIT_ROWS
            )
            self._by_source[source] = fitter.fit(train.x.lead_hours)
        return self

    def transform(self, x: ForecastMatrix) -> FloatArray:
        corrected = x.values.copy()
        design, feature_names = _phase_design(x)
        if feature_names != self._feature_names:
            return corrected
        for index, source in enumerate(x.sources):
            fitted = self._by_source.get(source)
            if fitted is None:
                continue

            def use(
                coefficients: FloatArray, rows: np.ndarray, index: int = index
            ) -> None:
                if coefficients.shape[0] != design.shape[1]:
                    return  # feature set changed between fit and predict
                corrected[rows, index] = x.values[rows, index] + (
                    design[rows] @ coefficients
                )

            fitted.apply(x.lead_hours, use)
        return corrected

    def to_state(self) -> dict[str, object]:
        return {
            source: {
                "global": fitted.global_state.tolist(),
                "buckets": {
                    label: state.tolist() for label, state in fitted.states.items()
                },
            }
            for source, fitted in self._by_source.items()
        }


@dataclass
class HarmonicGroundedEqualWeight:
    method_id: str = "harmonic_grounded_equal_weight"
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _grounding: HarmonicGrounding = field(default_factory=HarmonicGrounding)

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._grounding = HarmonicGrounding().fit(train)
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        corrected = self._grounding.transform(x)
        point = masked_average(corrected, x.availability)
        return BlendResult(point=finalize_point(point, self._kind, self._variable))


register("harmonic_grounded_equal_weight", HarmonicGroundedEqualWeight)
