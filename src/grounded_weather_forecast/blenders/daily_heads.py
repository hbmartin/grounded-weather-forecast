"""First-class daily Tmax/Tmin heads: direct marginal vs extreme-of-path.

The daily temperature product blends provider *daily* values; the hourly
path only rides along as equal-weight aggregate features. Two competing
architectures exist for doing better and no published head-to-head at
single-station scale was found (future-work #6) — so both are registered
and the leaderboard arbitrates:

- ``daily_marginal_emos`` (Meng & Taylor 2022): calibrate the daily extreme
  directly — a Gaussian EMOS whose mean links to both the grounded daily
  base blend and the cross-source mean of per-source hourly-path extremes
  (``path__{source}__max/min`` features from the dataset build).
- ``daily_path_extreme`` (Schefzik-flavored): treat each source's own
  hourly-path extreme as an ensemble member, ground each member per lead
  bucket (bias-only, shrunk), and read the daily distribution straight off
  the grounded members — the members' rank structure is the ECC coupling,
  collapsed to the scalar extreme. Zero distributional parameters.

Both abstain (NaN, "no opinion") on rows without path features, so the
coverage gate — not a crash — decides their eligibility.
"""

from dataclasses import dataclass, field
from importlib import import_module
from typing import Self

import numpy as np

from grounded_weather_forecast.blenders.combine import GroundedEqualWeight
from grounded_weather_forecast.blenders.emos import (
    _minimize,
    _recency_weights,
    _spread,
    gaussian_crps,
)
from grounded_weather_forecast.blenders.protocol import (
    FittedBuckets,
    coefficient_state,
    finalize_point,
    finalize_quantiles,
    fit_shrunk_buckets,
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
_MIN_SIGMA = 1e-3
_MIN_MEMBERS = 3

DAILY_HEAD_VARIABLES = frozenset({"temp_max_c", "temp_min_c"})


def _path_bound(variable_name: str) -> str:
    return "max" if variable_name == "temp_max_c" else "min"


def _path_columns(x: ForecastMatrix, variable_name: str) -> list[str]:
    bound = _path_bound(variable_name)
    return sorted(
        column
        for column in x.features.columns
        if column.startswith("path__") and column.endswith(f"__{bound}")
    )


def _path_members(x: ForecastMatrix, variable_name: str) -> FloatArray | None:
    columns = _path_columns(x, variable_name)
    if not columns:
        return None
    return (
        x.features.select(columns)
        .cast(dict.fromkeys(columns, float))  # type: ignore[arg-type]
        .to_numpy()
        .astype(np.float64)
    )


@dataclass
class DailyMarginalEmos:
    """Direct daily-extreme EMOS on the base blend plus the path signal."""

    base_factory: BlenderFactory
    method_id: str = "daily_marginal_emos"
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _parameters: FloatArray | None = None
    _fit_status: str = "unfitted"

    def _predictors(self, x: ForecastMatrix, base_point: FloatArray) -> FloatArray:
        """The path predictor: cross-source mean extreme, base-filled.

        Filling missing path values with the base keeps ``mu`` continuous —
        the path coefficient simply contributes nothing on those rows.
        """
        members = _path_members(x, self._variable.name if self._variable else "")
        if members is None:
            return base_point
        with np.errstate(invalid="ignore"):
            path = np.nanmean(members, axis=1)
        return np.where(np.isfinite(path), path, base_point)

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._base = self.base_factory().fit(train)
        base_point = self._base.predict(train.x).point
        path = self._predictors(train.x, base_point)
        spread = _spread(train.x, train.variable.name)
        weights = _recency_weights(train.x)
        scored = np.isfinite(base_point) & np.isfinite(train.y)
        if int(scored.sum()) < _MIN_FIT_ROWS:
            self._parameters = None
            self._fit_status = "insufficient_rows"
            return self
        y = train.y[scored]
        base = base_point[scored]
        path_scored = path[scored]
        log_spread = np.log(np.maximum(spread[scored], _MIN_SIGMA))
        w = weights[scored] / weights[scored].sum()
        residual_sd = max(float(np.std(y - base)), _MIN_SIGMA)
        initial = np.array([0.0, 1.0, 0.0, np.log(residual_sd), 0.0])

        def loss(parameters: FloatArray) -> float:
            a, b, e, c, d = parameters
            mu = a + b * base + e * path_scored
            sigma = np.maximum(np.exp(c + d * log_spread), _MIN_SIGMA)
            return float((w * gaussian_crps(y, mu, sigma)).sum())

        # Five parameters need a bigger simplex budget than the 4-parameter
        # Gaussian head's default.
        self._parameters = _minimize(loss, initial, max_iterations=1500)
        self._fit_status = "unfitted" if self._parameters is None else "converged"
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        base_point = self._base.predict(x).point
        if self._parameters is None:
            return BlendResult(
                point=finalize_point(base_point, self._kind, self._variable)
            )
        a, b, e, c, d = self._parameters
        path = self._predictors(x, base_point)
        mu = a + b * base_point + e * path
        variable_name = self._variable.name if self._variable else ""
        log_spread = np.log(np.maximum(_spread(x, variable_name), _MIN_SIGMA))
        sigma = np.maximum(np.exp(c + d * log_spread), _MIN_SIGMA)
        stats = import_module("scipy.stats")
        levels = np.asarray(QUANTILE_LEVELS)
        return quantile_blend_result(
            mu[:, np.newaxis] + sigma[:, np.newaxis] * stats.norm.ppf(levels),
            QUANTILE_LEVELS,
            self._kind,
            self._variable,
        )

    def to_state(self) -> dict[str, object]:
        return coefficient_state(
            self.method_id,
            self._variable,
            self._fit_status,
            self._parameters,
            (
                "mean_intercept",
                "mean_base_slope",
                "mean_path_slope",
                "log_sigma_intercept",
                "log_sigma_slope",
            ),
        )


@dataclass
class DailyPathExtreme:
    """Grounded ensemble of per-source hourly-path extremes."""

    method_id: str = "daily_path_extreme"
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _columns: tuple[str, ...] = ()
    _fitted: FittedBuckets[FloatArray] | None = field(default=None, repr=False)

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._columns = tuple(_path_columns(train.x, train.variable.name))
        members = _path_members(train.x, train.variable.name)
        if members is None or train.x.n_rows < _MIN_FIT_ROWS:
            self._fitted = None
            return self
        y = train.y

        def fit_one(rows: np.ndarray) -> FloatArray:
            biases = np.zeros(members.shape[1])
            for column in range(members.shape[1]):
                usable = rows[np.isfinite(members[rows, column]) & np.isfinite(y[rows])]
                if usable.shape[0] > 0:
                    biases[column] = float(np.mean(y[usable] - members[usable, column]))
            return biases

        self._fitted = fit_shrunk_buckets(train.x.product, train.x.lead_hours, fit_one)
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        point = np.full(x.n_rows, np.nan)
        members = _path_members(x, self._variable.name if self._variable else "")
        fitted = self._fitted
        if (
            members is None
            or fitted is None
            or tuple(_path_columns(x, self._variable.name if self._variable else ""))
            != self._columns
        ):
            return BlendResult(point=finalize_point(point, self._kind, self._variable))
        grounded = np.full_like(members, np.nan)

        def use(biases: FloatArray, rows: np.ndarray) -> None:
            grounded[rows] = members[rows] + biases[np.newaxis, :]

        fitted.apply(x.lead_hours, use)
        counts = np.isfinite(grounded).sum(axis=1)
        enough = counts >= _MIN_MEMBERS
        with np.errstate(invalid="ignore", all="ignore"):
            point = np.where(enough, np.nanmedian(grounded, axis=1), np.nan)
        if not enough.any():
            return BlendResult(point=finalize_point(point, self._kind, self._variable))
        levels = np.asarray(QUANTILE_LEVELS)
        quantiles = np.full((x.n_rows, levels.shape[0]), np.nan)
        rows = np.flatnonzero(enough)
        with np.errstate(invalid="ignore", all="ignore"):
            quantiles[rows] = np.nanquantile(grounded[rows], levels, axis=1).T
        quantiles[rows] = finalize_quantiles(
            quantiles[rows], self._kind, self._variable
        )
        return BlendResult(
            point=finalize_point(point, self._kind, self._variable),
            quantiles=quantiles,
            quantile_levels=QUANTILE_LEVELS,
        )

    def to_state(self) -> dict[str, object]:
        fitted = self._fitted
        return {
            "schema_version": 1,
            "method_id": self.method_id,
            "variable": self._variable.name if self._variable else None,
            "members": list(self._columns),
            "fitted": fitted is not None,
            "biases": None
            if fitted is None
            else {
                "global": list(np.asarray(fitted.global_state)),
                "buckets": {
                    label: list(np.asarray(state))
                    for label, state in sorted(fitted.states.items())
                },
            },
        }


def _daily_marginal_emos() -> Blender:
    return DailyMarginalEmos(GroundedEqualWeight)


def _daily_path_extreme() -> Blender:
    return DailyPathExtreme()


register(
    "daily_marginal_emos",
    _daily_marginal_emos,
    variables=DAILY_HEAD_VARIABLES,
)
register(
    "daily_path_extreme",
    _daily_path_extreme,
    variables=DAILY_HEAD_VARIABLES,
)
