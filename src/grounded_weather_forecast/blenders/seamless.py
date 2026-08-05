"""Seamless multimodel regression: the single-model challenger architecture.

Dabernig & Atencia (2024): one per-bucket ridge regression over all source
columns, with two tricks that dissolve this package's separate stages into
coefficients. (a) An expired model's last available lead is carried forward
within the snapshot as a "model persistence" predictor, so a source falling
off its horizon bends its coefficient instead of cliffing out of the blend.
(b) The issue-time station observation is one more predictor whose fitted
coefficient decays naturally with lead — anchoring without an anchoring
stage. Grounding, weighting, and anchoring all emerge from the regression;
this is the honest test of whether the three-stage decomposition earns its
keep.

Win condition (research log 2026-07): beats the online experts at mid leads
or the anchored family at short leads without losing the long ones.
"""

from dataclasses import dataclass, field
from typing import Self

import numpy as np
import polars as pl

from grounded_weather_forecast.blenders.protocol import (
    FittedBuckets,
    PerBucketFitter,
    finalize_point,
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
    obs_col,
)
from grounded_weather_forecast.leads import buckets_for_product

_RIDGE = 10.0
_MIN_FIT_ROWS = 48


def _forward_filled(
    x: ForecastMatrix, sources: tuple[str, ...] | None = None
) -> FloatArray:
    """Per-snapshot model persistence: carry each source's last lead forward."""
    sources = sources if sources is not None else x.sources
    indices = [x.sources.index(source) for source in sources]
    values = x.values[:, indices]
    if "issue_time" not in x.features.columns:
        return values
    frame = pl.DataFrame(
        {
            "issue_time": x.features["issue_time"],
            "lead": pl.Series(x.lead_hours),
            "row": pl.Series(np.arange(x.n_rows)),
        }
    ).with_columns(
        *(
            pl.Series(source, values[:, index]).fill_nan(None)
            for index, source in enumerate(sources)
        )
    )
    filled = (
        frame.sort("issue_time", "lead")
        .with_columns(
            *(pl.col(source).forward_fill().over("issue_time") for source in sources)
        )
        .sort("row")
    )
    return filled.select(list(sources)).to_numpy().astype(np.float64)


def _design(
    x: ForecastMatrix,
    observation_column: str,
    sources: tuple[str, ...] | None = None,
) -> FloatArray:
    """[forward-filled sources, issue-time observation, 1] with mean-filling.

    Sources still missing after the forward fill (never published for this
    snapshot) take the row mean of the available ones, so the ridge sees a
    complete matrix without inventing information the equal-weight fallback
    would not also assume. The observation column falls back to the row mean
    as well — its coefficient then simply cannot help on those rows.
    """
    values = _forward_filled(x, sources)
    with np.errstate(invalid="ignore"):
        row_mean = np.nanmean(values, axis=1)
    filled = np.where(np.isfinite(values), values, row_mean[:, np.newaxis])
    if observation_column in x.features.columns:
        observation = x.features[observation_column].cast(pl.Float64).to_numpy()
        observation = np.where(np.isfinite(observation), observation, row_mean)
    else:
        observation = row_mean
    return np.column_stack([filled, observation, np.ones(x.n_rows)])


@dataclass
class SeamlessRegression:
    """Per-bucket ridge over sources + model persistence + the observation."""

    method_id: str = "seamless_regression"
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _fitted: FittedBuckets[FloatArray] | None = field(default=None, repr=False)

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._observation_column = obs_col(train.variable.name)
        # Never-observed columns would only inject mean-filled noise into the
        # ridge (and break the all-NaN-source invariance); drop them here and
        # select the same named columns at predict.
        observed = np.isfinite(train.x.values).any(axis=0)
        self._sources = tuple(
            source
            for source, seen in zip(train.x.sources, observed, strict=True)
            if seen
        )
        design = _design(train.x, self._observation_column, self._sources)
        y = train.y
        usable = np.isfinite(design).all(axis=1) & np.isfinite(y)
        if int(usable.sum()) < _MIN_FIT_ROWS:
            self._fitted = None
            return self
        columns = design.shape[1]

        def fit_one(rows: np.ndarray) -> FloatArray:
            selected = rows[usable[rows]]
            if selected.shape[0] == 0:
                return np.zeros(columns)
            matrix = design[selected]
            gram = matrix.T @ matrix + _RIDGE * np.eye(columns)
            try:
                return np.linalg.solve(gram, matrix.T @ y[selected])
            except np.linalg.LinAlgError:
                return np.zeros(columns)

        fitter = PerBucketFitter(
            buckets=buckets_for_product(train.x.product),
            fit_one=fit_one,
            min_rows=_MIN_FIT_ROWS,
            blend=lambda local, glob, w: w * local + (1.0 - w) * glob,
        )
        self._fitted = fitter.fit(train.x.lead_hours)
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        point = np.full(x.n_rows, np.nan)
        fitted = self._fitted
        if fitted is None:
            return BlendResult(point=finalize_point(point, self._kind, self._variable))
        if not set(self._sources) <= set(x.sources):
            return BlendResult(point=finalize_point(point, self._kind, self._variable))
        design = _design(x, self._observation_column, self._sources)
        usable = np.isfinite(design).all(axis=1)

        def use(coefficients: FloatArray, rows: np.ndarray) -> None:
            selected = rows[usable[rows]]
            if selected.shape[0] and coefficients.shape[0] == design.shape[1]:
                point[selected] = design[selected] @ coefficients

        fitted.apply(x.lead_hours, use)
        return BlendResult(point=finalize_point(point, self._kind, self._variable))

    def to_state(self) -> dict[str, object]:
        fitted = self._fitted
        return {
            "schema_version": 1,
            "method_id": self.method_id,
            "coefficients": None
            if fitted is None
            else {
                "global": list(np.asarray(fitted.global_state)),
                "buckets": {
                    label: list(np.asarray(state))
                    for label, state in sorted(fitted.states.items())
                },
            },
        }


def _seamless_regression() -> Blender:
    return SeamlessRegression()


register("seamless_regression", _seamless_regression)
