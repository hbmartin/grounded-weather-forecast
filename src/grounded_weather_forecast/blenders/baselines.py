"""Baseline blenders: the bars every real method must clear.

- ``persistence``: the station reading at issue time, unchanged.
- ``climatology``: harmonic (Fourier month + hour) regression on truth alone.
- ``best_provider``: the single best source per lead bucket on training MAE.
- ``equal_weight``: raw availability-renormalized mean across sources.

A NaN prediction means "this method has nothing to say for this row" (e.g.
persistence without an issue-time observation); the engine stores it as null
and promotion compares methods on one shared common-case mask with coverage shown.
"""

import math
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
    Blender,
    BlendResult,
    ForecastMatrix,
    SupervisedSlice,
    TargetKind,
    VariableSpec,
    obs_col,
)
from grounded_weather_forecast.leads import buckets_for_product


@dataclass
class Persistence:
    method_id: str = "persistence"
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._obs_column = obs_col(train.variable.name)
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        if self._obs_column in x.features.columns:
            point = (
                x.features[self._obs_column]
                .cast(float)
                .fill_nan(None)
                .to_numpy()
                .astype(np.float64)
            )
        else:
            point = np.full(x.n_rows, np.nan)
        return BlendResult(point=finalize_point(point, self._kind, self._variable))


def _harmonic_design(
    hour: np.ndarray | None, month: np.ndarray | None, n: int
) -> np.ndarray:
    columns = [np.ones(n)]
    if month is not None:
        angle = 2.0 * np.pi * (month - 1) / 12.0
        columns += [np.sin(angle), np.cos(angle)]
    if hour is not None:
        angle = 2.0 * np.pi * hour / 24.0
        columns += [np.sin(angle), np.cos(angle), np.sin(2 * angle), np.cos(2 * angle)]
    return np.column_stack(columns)


def _feature_or_none(x: ForecastMatrix, name: str) -> np.ndarray | None:
    if name not in x.features.columns:
        return None
    return x.features[name].to_numpy().astype(np.float64)


@dataclass
class HarmonicClimatology:
    """Ridge-regularized Fourier regression on month-of-year and hour-of-day.

    The ridge is not decoration. A training window shorter than a year covers
    only an arc of the annual cycle, which leaves the annual sine and cosine
    nearly collinear; unpenalized least squares then fits huge, cancelling
    coefficients that explode the moment they are extrapolated into a season the
    window never saw. The penalty shrinks the seasonal terms toward zero, so a
    short archive degrades gracefully to "mean plus diurnal cycle" instead of
    producing a baseline so bad it flatters everything measured against it. The
    intercept is never penalized.
    """

    method_id: str = "climatology"
    ridge: float = 1.0
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _coefficients: np.ndarray = field(default_factory=lambda: np.zeros(1))
    _has_hour: bool = False
    _has_month: bool = False

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        hour = _feature_or_none(train.x, "valid_hour_local")
        month = _feature_or_none(train.x, "valid_month")
        self._has_hour = hour is not None
        self._has_month = month is not None
        design = _harmonic_design(hour, month, train.x.n_rows)
        penalty = np.eye(design.shape[1]) * math.sqrt(self.ridge)
        penalty[0, 0] = 0.0  # never shrink the intercept
        augmented = np.vstack([design, penalty])
        target = np.concatenate([train.y, np.zeros(design.shape[1])])
        self._coefficients, *_ = np.linalg.lstsq(augmented, target, rcond=None)
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        hour = _feature_or_none(x, "valid_hour_local") if self._has_hour else None
        month = _feature_or_none(x, "valid_month") if self._has_month else None
        design = _harmonic_design(hour, month, x.n_rows)
        point = design @ self._coefficients
        return BlendResult(point=finalize_point(point, self._kind, self._variable))


@dataclass
class BestProvider:
    """Per lead bucket: rank sources by training MAE, predict from the best
    available source at predict time (falling back down the ranking)."""

    method_id: str = "best_provider"
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _sources: tuple[str, ...] = ()

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._sources = train.x.sources
        values, y = train.x.values, train.y

        def rank_sources(rows: np.ndarray) -> np.ndarray:
            errors = np.abs(values[rows] - y[rows][:, np.newaxis])
            counts = (~np.isnan(errors)).sum(axis=0)
            sums = np.nansum(errors, axis=0)
            mae_per_source = np.where(counts > 0, sums / np.maximum(counts, 1), np.inf)
            return np.argsort(mae_per_source)

        fitter = PerBucketFitter[np.ndarray](
            buckets=buckets_for_product(train.x.product), fit_one=rank_sources
        )
        self._fitted: FittedBuckets[np.ndarray] = fitter.fit(train.x.lead_hours)
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        point = np.full(x.n_rows, np.nan)

        def use(ranking: np.ndarray, rows: np.ndarray) -> None:
            for row in rows:
                for source_index in ranking:
                    if x.availability[row, source_index]:
                        point[row] = x.values[row, source_index]
                        break

        self._fitted.apply(x.lead_hours, use)
        return BlendResult(point=finalize_point(point, self._kind, self._variable))

    def to_state(self) -> dict[str, object]:
        """Per-bucket source rankings by ascending training MAE, as names."""

        def names(ranking: np.ndarray) -> list[str]:
            return [self._sources[int(index)] for index in ranking]

        return {
            "sources": list(self._sources),
            "global": names(self._fitted.global_state),
            "buckets": {
                label: names(ranking) for label, ranking in self._fitted.states.items()
            },
        }


@dataclass
class EqualWeight:
    """Raw availability-renormalized mean across sources."""

    method_id: str = "equal_weight"
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        point = masked_average(x.values, x.availability)
        return BlendResult(point=finalize_point(point, self._kind, self._variable))


def _register_baselines() -> None:
    register("persistence", Persistence)
    register("climatology", HarmonicClimatology)
    register("best_provider", BestProvider)
    register("equal_weight", EqualWeight)


_register_baselines()

_PROTOCOL_CHECK: tuple[type[Blender], ...] = (
    Persistence,
    HarmonicClimatology,
    BestProvider,
    EqualWeight,
)
