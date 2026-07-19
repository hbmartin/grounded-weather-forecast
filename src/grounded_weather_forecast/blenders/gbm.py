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

from grounded_weather_forecast.blenders.protocol import finalize_point
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
    "objective": "regression_l1",
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
    return np.column_stack(columns), names


def _variable_spec(name: str | None) -> VariableSpec | None:
    for spec in (*HOURLY_VARIABLES, *DAILY_VARIABLES):
        if spec.name == name:
            return spec
    return None


@dataclass
class GbmStacker:
    method_id: str = "gbm"
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _feature_names: list[str] = field(default_factory=list)

    def fit(self, train: SupervisedSlice) -> Self:
        lightgbm = import_module("lightgbm")
        self._kind = train.variable.kind
        self._variable = train.variable
        features, self._feature_names = build_features(train.x)
        dataset = lightgbm.Dataset(
            features, label=train.y, feature_name=self._feature_names
        )
        self._booster = lightgbm.train(_PARAMS, dataset, num_boost_round=_NUM_ROUNDS)
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        features, names = build_features(x)
        if names != self._feature_names:
            aligned = np.full((features.shape[0], len(self._feature_names)), np.nan)
            index = {name: i for i, name in enumerate(names)}
            for target_position, name in enumerate(self._feature_names):
                if name in index:
                    aligned[:, target_position] = features[:, index[name]]
            features = aligned
        point = np.asarray(self._booster.predict(features), dtype=np.float64)
        return BlendResult(point=finalize_point(point, self._kind, self._variable))

    def to_state(self) -> dict[str, Any]:
        return {
            "model": self._booster.model_to_string(),
            "feature_names": self._feature_names,
            "kind": self._kind.value,
            "variable": self._variable.name if self._variable else None,
        }

    def observability_state(self) -> dict[str, Any]:
        """Compact glass-box state: importances only, never the booster."""
        gain = self._booster.feature_importance(importance_type="gain")
        split = self._booster.feature_importance(importance_type="split")
        return {
            "variable": self._variable.name if self._variable else None,
            "kind": self._kind.value,
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
        return stacker


HAVE_LIGHTGBM = find_spec("lightgbm") is not None

if HAVE_LIGHTGBM:  # pragma: no branch
    register("gbm", GbmStacker)
