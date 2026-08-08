"""LightGBM stacker: the flexible nonlinear ceiling.

One model per (variable, product) mapping [source forecasts + lead + calendar
+ ages + issue-time observations + ensemble spread] to truth. Trees handle
missing sources natively (NaN goes down a learned default branch), so no
imputation or availability special-casing is needed.

lightgbm is imported lazily and the method registers only when the import
succeeds, so the package stays importable where wheels lag (e.g. new CPython).
"""

from dataclasses import dataclass, field
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Self

import numpy as np

from grounded_weather_forecast.blenders.protocol import (
    finalize_point,
    finalize_quantiles,
    masked_average,
)
from grounded_weather_forecast.blenders.registry import register
from grounded_weather_forecast.contracts import (
    CONTEXT_FEATURE_COLUMNS,
    DAILY_VARIABLES,
    HOURLY_VARIABLES,
    BlendResult,
    FloatArray,
    ForecastMatrix,
    SupervisedSlice,
    TargetKind,
    VariableSpec,
)

_PARAMS: dict[str, Any] = {
    # huber, not regression_l1: LightGBM forbids monotone_constraints under
    # leaf-renewing objectives (l1, quantile), and the blend-mean constraint
    # is the point of the containment. Huber keeps l1's outlier robustness.
    "objective": "huber",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "seed": 20260713,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}
_NUM_ROUNDS = 300
# The quantile head trains one booster per level; fewer rounds per booster
# bounds the ~19x fit cost.
_QUANTILE_ROUNDS = 150
QUANTILE_LEVELS: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))
_MEDIAN_INDEX = QUANTILE_LEVELS.index(0.5)


def _numeric_feature_columns(x: ForecastMatrix) -> list[str]:
    return sorted(
        c
        for c in x.features.columns
        if c in ("valid_hour_local", "valid_month", *CONTEXT_FEATURE_COLUMNS)
        or c.startswith(("age__", "obs__", "ewagg__", "ens__"))
    )


def build_features(x: ForecastMatrix) -> tuple[FloatArray, list[str]]:
    """Numeric design matrix: sources, lead, calendar/context, spread, count."""
    columns: list[FloatArray] = [x.values]
    names: list[str] = [f"src__{source}" for source in x.sources]
    columns.append(x.lead_hours[:, np.newaxis])
    names.append("lead_hours")
    feature_names = _numeric_feature_columns(x)
    if feature_names:
        block = (
            x.features.select(feature_names)
            .cast(dict.fromkeys(feature_names, float))  # type: ignore[arg-type]
            .to_numpy()
            .astype(np.float64)
        )
        columns.append(block)
        names.extend(feature_names)
    with np.errstate(invalid="ignore"):
        spread = np.nanstd(x.values, axis=1)
    columns.append(np.nan_to_num(spread, nan=0.0)[:, np.newaxis])
    names.append("source_spread")
    columns.append(x.availability.sum(axis=1).astype(np.float64)[:, np.newaxis])
    names.append("n_available")
    # The equal-weight consensus, appended so the booster can be monotone-
    # constrained on it: a higher blend must never lower the prediction.
    columns.append(masked_average(x.values, x.availability)[:, np.newaxis])
    names.append("blend_mean")
    return np.column_stack(columns), names


def _monotone_constraints(feature_names: list[str]) -> list[int]:
    """+1 on the blend consensus, unconstrained elsewhere.

    Built from the fitted feature order, never hard-coded — predict()
    realigns columns by name when the schema drifts.
    """
    return [1 if name == "blend_mean" else 0 for name in feature_names]


def _variable_spec(name: str | None) -> VariableSpec | None:
    for spec in (*HOURLY_VARIABLES, *DAILY_VARIABLES):
        if spec.name == name:
            return spec
    return None


@dataclass
class GbmStacker:
    method_id: str = "gbm"
    # A 300-round booster on a few hundred rows memorizes noise yet can still
    # sneak through the promotion gate on a lucky fold; below the floor the
    # method abstains entirely (NaN = "no opinion"), so its slice coverage
    # falls under the leaderboard's 0.8 eligibility bar.
    min_fit_rows: int = 500
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _feature_names: list[str] = field(default_factory=list)
    _training_rows: int = 0
    _fit_status: str = "unfitted"

    def _begin_fit(self, train: SupervisedSlice) -> FloatArray | None:
        """Record the slice identity; None (abstention) below the row floor."""
        self._kind = train.variable.kind
        self._variable = train.variable
        self._training_rows = train.x.n_rows
        if train.x.n_rows < self.min_fit_rows:
            self._fit_status = "insufficient_rows"
            return None
        features, self._feature_names = build_features(train.x)
        return features

    def fit(self, train: SupervisedSlice) -> Self:
        features = self._begin_fit(train)
        if features is None:
            return self
        lightgbm = import_module("lightgbm")
        dataset = lightgbm.Dataset(
            features, label=train.y, feature_name=self._feature_names
        )
        params = {
            **_PARAMS,
            "monotone_constraints": _monotone_constraints(self._feature_names),
            "monotone_constraints_method": "advanced",
        }
        self._booster = lightgbm.train(params, dataset, num_boost_round=_NUM_ROUNDS)
        self._fit_status = "fit"
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        if self._fit_status != "fit":
            point = np.full(x.n_rows, np.nan)
            return BlendResult(point=finalize_point(point, self._kind, self._variable))
        features = self._aligned_features(x)
        point = np.asarray(self._booster.predict(features), dtype=np.float64)
        return BlendResult(point=finalize_point(point, self._kind, self._variable))

    def _aligned_features(self, x: ForecastMatrix) -> FloatArray:
        """The fitted feature order, re-aligned by name when the schema drifts."""
        features, names = build_features(x)
        if names == self._feature_names:
            return features
        aligned = np.full((features.shape[0], len(self._feature_names)), np.nan)
        index = {name: i for i, name in enumerate(names)}
        for target_position, name in enumerate(self._feature_names):
            if name in index:
                aligned[:, target_position] = features[:, index[name]]
        return aligned

    def to_state(self) -> dict[str, Any]:
        if self._fit_status != "fit":
            return {
                "fit_status": self._fit_status,
                "training_rows": self._training_rows,
                "min_fit_rows": self.min_fit_rows,
                "kind": self._kind.value,
                "variable": self._variable.name if self._variable else None,
            }
        return {
            "model": self._booster.model_to_string(),
            "feature_names": self._feature_names,
            "kind": self._kind.value,
            "variable": self._variable.name if self._variable else None,
        }

    def observability_state(self) -> dict[str, Any]:
        """Compact glass-box state: importances only, never the booster."""
        if self._fit_status != "fit":
            return {
                "variable": self._variable.name if self._variable else None,
                "kind": self._kind.value,
                "fit_status": self._fit_status,
                "training_rows": self._training_rows,
                "min_fit_rows": self.min_fit_rows,
            }
        gain = self._booster.feature_importance(importance_type="gain")
        split = self._booster.feature_importance(importance_type="split")
        return {
            "variable": self._variable.name if self._variable else None,
            "kind": self._kind.value,
            "fit_status": self._fit_status,
            "training_rows": self._training_rows,
            "num_trees": int(self._booster.num_trees()),
            "feature_names": list(self._feature_names),
            "importance_gain": {
                name: float(value)
                for name, value in zip(self._feature_names, gain, strict=True)
            },
            "importance_split": {
                name: int(value)
                for name, value in zip(self._feature_names, split, strict=True)
            },
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "GbmStacker":
        lightgbm = import_module("lightgbm")
        stacker = cls()
        stacker._kind = TargetKind(state["kind"])
        stacker._variable = _variable_spec(state.get("variable"))
        stacker._feature_names = list(state["feature_names"])
        stacker._booster = lightgbm.Booster(model_str=state["model"])
        stacker._fit_status = "fit"
        return stacker


@dataclass
class GbmQuantile(GbmStacker):
    """Native quantile head: one pinball-loss booster per level.

    The containment counterpart to ``gbm``: instead of a point that gets
    residual-dressed at serve time, every level is learned directly (with the
    same blend-mean monotone constraint), so the leaderboard scores its
    CRPS/pinball columns and arbitrates whether it is ever trusted. Boosters
    at neighboring levels can cross in finite samples; ``finalize_quantiles``
    sorts each row's grid.
    """

    method_id: str = "gbm_quantile"

    def fit(self, train: SupervisedSlice) -> Self:
        features = self._begin_fit(train)
        if features is None:
            return self
        lightgbm = import_module("lightgbm")
        self._level_boosters = []
        for level in QUANTILE_LEVELS:
            dataset = lightgbm.Dataset(
                features, label=train.y, feature_name=self._feature_names
            )
            # No monotone constraint here: LightGBM forbids it under the
            # quantile objective (leaf-renewing). Containment for this head
            # is the fit-rows floor, the sorted grid, and the leaderboard's
            # CRPS/pinball columns.
            params = {**_PARAMS, "objective": "quantile", "alpha": level}
            self._level_boosters.append(
                lightgbm.train(params, dataset, num_boost_round=_QUANTILE_ROUNDS)
            )
        self._fit_status = "fit"
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        if self._fit_status != "fit":
            point = np.full(x.n_rows, np.nan)
            return BlendResult(point=finalize_point(point, self._kind, self._variable))
        features = self._aligned_features(x)
        rows = np.column_stack(
            [
                np.asarray(booster.predict(features), dtype=np.float64)
                for booster in self._level_boosters
            ]
        )
        quantiles = finalize_quantiles(rows, self._kind, self._variable)
        point = quantiles[:, _MEDIAN_INDEX]
        return BlendResult(
            point=finalize_point(point, self._kind, self._variable),
            quantiles=quantiles,
            quantile_levels=QUANTILE_LEVELS,
        )

    def to_state(self) -> dict[str, Any]:
        if self._fit_status != "fit":
            return super().to_state()
        return {
            "models": [booster.model_to_string() for booster in self._level_boosters],
            "quantile_levels": list(QUANTILE_LEVELS),
            "feature_names": self._feature_names,
            "kind": self._kind.value,
            "variable": self._variable.name if self._variable else None,
        }

    def observability_state(self) -> dict[str, Any]:
        if self._fit_status != "fit":
            return super().observability_state()
        median = self._level_boosters[_MEDIAN_INDEX]
        gain = median.feature_importance(importance_type="gain")
        return {
            "variable": self._variable.name if self._variable else None,
            "kind": self._kind.value,
            "fit_status": self._fit_status,
            "training_rows": self._training_rows,
            "num_levels": len(self._level_boosters),
            "num_trees_median": int(median.num_trees()),
            "feature_names": list(self._feature_names),
            "importance_gain_median": {
                name: float(value)
                for name, value in zip(self._feature_names, gain, strict=True)
            },
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "GbmQuantile":
        lightgbm = import_module("lightgbm")
        stacker = cls()
        stacker._kind = TargetKind(state["kind"])
        stacker._variable = _variable_spec(state.get("variable"))
        stacker._feature_names = list(state["feature_names"])
        stacker._level_boosters = [
            lightgbm.Booster(model_str=model) for model in state["models"]
        ]
        stacker._fit_status = "fit"
        return stacker


HAVE_LIGHTGBM = find_spec("lightgbm") is not None

if HAVE_LIGHTGBM:  # pragma: no branch
    register("gbm", GbmStacker)
    register("gbm_quantile", GbmQuantile)
