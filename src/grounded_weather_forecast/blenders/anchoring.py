"""Anchoring: short-lead correction of a blend toward the latest observation.

The station's unique asset is a live thermometer no provider has. At short
leads, the blend's current residual (observation minus the blend's own
now-forecast) persists; adding it back with an exponential decay in lead
dominates everything else in hour one and fades to nothing by half a day.

``Anchored`` wraps any base blender factory as its own leaderboard-visible
method. The decay timescale is fitted per variable by grid search on the
training slice, and "no anchoring" wins the grid when it is genuinely better.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np
import polars as pl

from grounded_weather_forecast.blenders.combine import (
    GroundedEqualWeight,
    InverseMseWeights,
)
from grounded_weather_forecast.blenders.protocol import finalize_point
from grounded_weather_forecast.blenders.registry import BlenderFactory, register
from grounded_weather_forecast.contracts import (
    Blender,
    BlendResult,
    FloatArray,
    ForecastMatrix,
    SupervisedSlice,
    TargetKind,
    obs_col,
)

TAU_GRID_HOURS: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 6.0, 12.0, 24.0)
_ANCHOR_MAX_LEAD = 3.0
_WEIGHT_FLOOR = 0.05


def issue_residuals(
    x: ForecastMatrix, base_point: FloatArray, observation_column: str
) -> FloatArray:
    """Per-row anchor residual: obs(issue) minus the base blend's now-forecast.

    The now-forecast is the base's prediction on the same snapshot's
    shortest-lead row (must be under 3 h, else the snapshot has no anchor).
    Rows without a usable anchor get NaN (treated as zero correction).
    """
    if (
        observation_column not in x.features.columns
        or "issue_time" not in x.features.columns
    ):
        return np.full(x.n_rows, np.nan)
    frame = pl.DataFrame(
        {
            "issue_time": x.features["issue_time"],
            "obs": x.features[observation_column],
            "lead": pl.Series(x.lead_hours),
            "base": pl.Series(base_point),
            "row": pl.Series(np.arange(x.n_rows)),
        }
    )
    anchors = (
        frame.filter(
            (pl.col("lead") < _ANCHOR_MAX_LEAD)
            & pl.col("base").is_not_nan()
            & pl.col("obs").is_not_null()
        )
        .sort("lead")
        .group_by("issue_time", maintain_order=True)
        .first()
        .select("issue_time", (pl.col("obs") - pl.col("base")).alias("r0"))
    )
    joined = frame.join(anchors, on="issue_time", how="left")
    return joined.sort("row")["r0"].cast(pl.Float64).fill_null(np.nan).to_numpy()


@dataclass
class Anchored:
    """Protocol wrapper: base blend plus lead-decayed anchor residual."""

    base_factory: BlenderFactory
    method_id: str
    _kind: TargetKind = TargetKind.CONTINUOUS
    _tau_hours: float | None = None

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._observation_column = obs_col(train.variable.name)
        self._base = self.base_factory().fit(train)
        base_point = self._base.predict(train.x).point
        residuals = issue_residuals(train.x, base_point, self._observation_column)
        self._tau_hours = self._search_tau(
            train.x.lead_hours, base_point, residuals, train.y
        )
        return self

    @staticmethod
    def _search_tau(
        lead: FloatArray,
        base_point: FloatArray,
        residuals: FloatArray,
        y: FloatArray,
    ) -> float | None:
        scored = ~np.isnan(base_point)
        if not scored.any() or np.isnan(residuals[scored]).all():
            return None
        correction = np.nan_to_num(residuals, nan=0.0)

        def mse(tau: float | None) -> float:
            if tau is None:
                anchored = base_point
            else:
                weight = np.exp(-lead / tau)
                weight = np.where(weight < _WEIGHT_FLOOR, 0.0, weight)
                anchored = base_point + weight * correction
            return float(np.mean((anchored[scored] - y[scored]) ** 2))

        candidates: list[float | None] = [None, *TAU_GRID_HOURS]
        return min(candidates, key=mse)

    def predict(self, x: ForecastMatrix) -> BlendResult:
        base_point = self._base.predict(x).point
        if self._tau_hours is None:
            return BlendResult(point=finalize_point(base_point, self._kind))
        residuals = issue_residuals(x, base_point, self._observation_column)
        correction = np.nan_to_num(residuals, nan=0.0)
        weight = np.exp(-x.lead_hours / self._tau_hours)
        weight = np.where(weight < _WEIGHT_FLOOR, 0.0, weight)
        point = base_point + weight * correction
        return BlendResult(point=finalize_point(point, self._kind))


def _anchored_gew() -> Blender:
    return Anchored(GroundedEqualWeight, "anchored_grounded_equal_weight")


def _anchored_inverse_mse() -> Blender:
    return Anchored(InverseMseWeights, "anchored_inverse_mse")


register("anchored_grounded_equal_weight", _anchored_gew)
register("anchored_inverse_mse", _anchored_inverse_mse)
