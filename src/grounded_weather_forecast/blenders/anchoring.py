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
    InverseErrorWeights,
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
    VariableSpec,
    obs_col,
)

TAU_GRID_HOURS: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 6.0, 12.0, 24.0)
# Anchor-row eligibility, not correction reach: observation information in
# temperature persists to ~12-20 h operationally (LAMP), and 6-hourly
# synthetic cycles have no row under 3 h at all, so 6 h is the honest gate.
_ANCHOR_MAX_LEAD = 6.0
_WEIGHT_FLOOR = 0.05


def issue_residuals(
    x: ForecastMatrix, base_point: FloatArray, observation_column: str
) -> FloatArray:
    """Per-row anchor residual: obs(issue) minus the base blend's now-forecast.

    The now-forecast is the base's prediction on the same snapshot's
    shortest-lead row (must be under 6 h, else the snapshot has no anchor).
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
    _variable: VariableSpec | None = None
    _tau_hours: float | None = None

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
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
            return BlendResult(
                point=finalize_point(base_point, self._kind, self._variable)
            )
        residuals = issue_residuals(x, base_point, self._observation_column)
        correction = np.nan_to_num(residuals, nan=0.0)
        weight = np.exp(-x.lead_hours / self._tau_hours)
        weight = np.where(weight < _WEIGHT_FLOOR, 0.0, weight)
        point = base_point + weight * correction
        return BlendResult(point=finalize_point(point, self._kind, self._variable))

    def to_state(self) -> dict[str, object]:
        """Compact observability state: the fitted decay and its base."""
        return {
            "tau_hours": self._tau_hours,
            "base_method_id": getattr(getattr(self, "_base", None), "method_id", None),
            "tau_grid_hours": list(TAU_GRID_HOURS),
        }


_FIT_BIN_EDGES: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 6.0, 12.0, 24.0)
_MIN_BIN_ROWS = 24
_MAX_TREND_GAIN_HOURS = 3.0


def _bin_centers() -> FloatArray:
    edges = np.asarray(_FIT_BIN_EDGES)
    return (edges[:-1] + edges[1:]) / 2.0


@dataclass
class AnchoredEmpirical:
    """Anchoring with per-lead weights *fitted from data*, LAMP-style.

    If the base blend's error were a stationary AR(1) in lead, the optimal
    anchor would be exactly the exponential ``rho**lead`` — so instead of
    assuming the shape, fit it: per lead bin, regress the base's realized
    error on the issue-time anchor residual (and optionally the observed
    issue-time tendency), clip the residual weight into [0, 1], and enforce
    that it never rises with lead. Persist-then-ramp (INCA) and exponential
    decay both emerge as special cases when the data supports them.

    ``use_trend`` adds the ``obs__{var}__trend15m`` regressor: the observed
    derivative is the one signal a level-only anchor discards, with its gain
    capped at ``_MAX_TREND_GAIN_HOURS`` so a momentary ramp can never
    extrapolate unboundedly.

    Day/night-conditioned weights are deliberately deferred until the live
    short-lead archive can support the split.
    """

    base_factory: BlenderFactory
    method_id: str
    use_trend: bool = False
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _residual_weights: FloatArray | None = None
    _trend_weights: FloatArray | None = None
    _fitted_bins: int | None = None

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._observation_column = obs_col(train.variable.name)
        self._trend_column = f"{self._observation_column}__trend15m"
        self._base = self.base_factory().fit(train)
        base_point = self._base.predict(train.x).point
        residuals = issue_residuals(train.x, base_point, self._observation_column)
        errors = train.y - base_point
        trend = self._trends(train.x)
        self._fit_bins(train.x.lead_hours, residuals, errors, trend)
        return self

    def _trends(self, x: ForecastMatrix) -> FloatArray:
        if not self.use_trend or self._trend_column not in x.features.columns:
            return np.full(x.n_rows, np.nan)
        return (
            x.features[self._trend_column]
            .cast(float)
            .fill_null(float("nan"))
            .to_numpy()
            .astype(np.float64)
        )

    def _fit_bins(
        self,
        lead: FloatArray,
        residuals: FloatArray,
        errors: FloatArray,
        trend: FloatArray,
    ) -> None:
        n_bins = len(_FIT_BIN_EDGES) - 1
        raw_weights = np.full(n_bins, np.nan)
        trend_weights = np.zeros(n_bins)
        bin_rows = np.zeros(n_bins, dtype=np.int64)
        usable = np.isfinite(residuals) & np.isfinite(errors)
        for index in range(n_bins):
            in_bin = (
                usable
                & (lead >= _FIT_BIN_EDGES[index])
                & (lead < _FIT_BIN_EDGES[index + 1])
            )
            bin_rows[index] = int(in_bin.sum())
            if bin_rows[index] < _MIN_BIN_ROWS:
                continue
            r0, e = residuals[in_bin], errors[in_bin]
            denominator = float(r0 @ r0)
            if denominator <= 0.0:
                continue
            raw_weights[index] = np.clip(float(r0 @ e) / denominator, 0.0, 1.0)
            trend_usable = in_bin & np.isfinite(trend)
            if self.use_trend and int(trend_usable.sum()) >= _MIN_BIN_ROWS:
                trend_bin = trend[trend_usable]
                centered = (
                    errors[trend_usable] - raw_weights[index] * residuals[trend_usable]
                )
                trend_denominator = float(trend_bin @ trend_bin)
                if trend_denominator > 0.0:
                    trend_weights[index] = np.clip(
                        float(trend_bin @ centered) / trend_denominator,
                        0.0,
                        _MAX_TREND_GAIN_HOURS,
                    )
        self._fitted_bins = int(np.isfinite(raw_weights).sum())
        if np.isnan(raw_weights).all():
            self._residual_weights = np.zeros(n_bins)
            self._trend_weights = trend_weights
            return
        # unfitted bins borrow from their fitted neighbors (an empty leading
        # bin must not zero the whole curve through the monotone constraint)
        indices = np.arange(n_bins, dtype=np.float64)
        fitted = np.isfinite(raw_weights)
        filled = np.interp(indices, indices[fitted], raw_weights[fitted])
        # A leading unfitted bin must not inherit its long-lead neighbor's
        # weight either (np.interp left-clamps): under the AR(1) residual
        # model the anchor weight rises toward 1 as lead -> 0, so extrapolate
        # the earliest fitted bin's implied decay back toward zero lead —
        # persist, then ramp. This is the measured 0-1 h regime where raw
        # persistence beat every anchored method on the first live folds.
        # Gated on the fitted weight clearing its own estimation noise
        # (~2/sqrt(n)): amplifying a noise-level weight toward 1 would turn a
        # harmless near-zero anchor into an aggressive wrong one.
        first = int(np.flatnonzero(fitted)[0])
        first_weight = float(raw_weights[first])
        credible = first_weight >= max(0.2, 2.0 / np.sqrt(float(bin_rows[first])))
        if first > 0 and credible and first_weight < 1.0:
            centers = _bin_centers()
            rho = first_weight ** (1.0 / centers[first])
            filled[:first] = rho ** centers[:first]
        # observation information can only fade with lead
        self._residual_weights = np.minimum.accumulate(filled)
        self._trend_weights = trend_weights

    @staticmethod
    def _zero_lead_weight(weights: FloatArray) -> float:
        """The residual weight as lead -> 0, by extrapolating the fitted decay.

        At zero lead the anchor observation *is* the target up to instrument
        noise, so a decaying curve is extended toward (0, <=1) in log space.
        A flat fitted curve is respected as-is — the ramp is only ever
        inferred from a measured decay, never invented.
        """
        centers = _bin_centers()
        first = float(weights[0])
        if first <= 0.0:
            return 0.0
        second = float(weights[1]) if len(weights) > 1 else first
        if not 0.0 < second < first:
            return first
        rho = (second / first) ** (1.0 / (centers[1] - centers[0]))
        return float(min(1.0, first * rho ** -centers[0]))

    def _weights_at(
        self, lead: FloatArray, weights: FloatArray, *, ramp_to_zero_lead: bool = False
    ) -> FloatArray:
        centers = np.append(_bin_centers(), _FIT_BIN_EDGES[-1])
        tapered = np.append(weights, 0.0)
        if ramp_to_zero_lead:
            centers = np.concatenate(([0.0], centers))
            tapered = np.concatenate(([self._zero_lead_weight(weights)], tapered))
        return np.interp(lead, centers, tapered, right=0.0)

    def predict(self, x: ForecastMatrix) -> BlendResult:
        base_point = self._base.predict(x).point
        if self._residual_weights is None:
            return BlendResult(
                point=finalize_point(base_point, self._kind, self._variable)
            )
        residuals = issue_residuals(x, base_point, self._observation_column)
        correction = np.nan_to_num(residuals, nan=0.0)
        point = (
            base_point
            + self._weights_at(
                x.lead_hours, self._residual_weights, ramp_to_zero_lead=True
            )
            * correction
        )
        if self.use_trend and self._trend_weights is not None:
            trend = np.nan_to_num(self._trends(x), nan=0.0)
            point = point + self._weights_at(x.lead_hours, self._trend_weights) * trend
        return BlendResult(point=finalize_point(point, self._kind, self._variable))

    def to_state(self) -> dict[str, object]:
        """Compact observability state: the fitted per-lead anchor weights."""
        residual = self._residual_weights
        trend = self._trend_weights
        return {
            "residual_weights": residual.tolist() if residual is not None else None,
            "trend_weights": trend.tolist() if trend is not None else None,
            "zero_lead_weight": (
                self._zero_lead_weight(residual) if residual is not None else None
            ),
            # 0 fitted bins means every weight came from the fallback path —
            # the silent all-NaN degeneration is visible here, not just absent.
            "fitted_bins": getattr(self, "_fitted_bins", None),
            "bin_edges": list(_FIT_BIN_EDGES),
            "use_trend": self.use_trend,
            "base_method_id": getattr(getattr(self, "_base", None), "method_id", None),
        }


def _anchored_gew() -> Blender:
    return Anchored(GroundedEqualWeight, "anchored_grounded_equal_weight")


def _anchored_inverse_mse() -> Blender:
    return Anchored(InverseErrorWeights, "anchored_inverse_mse")


def _anchored_fitted_gew() -> Blender:
    return AnchoredEmpirical(GroundedEqualWeight, "anchored_fitted_grounded")


def _anchored_fitted_ewma() -> Blender:
    from grounded_weather_forecast.blenders.ewma_grounding import (  # noqa: PLC0415
        EwmaGroundedBlend,
    )

    return AnchoredEmpirical(EwmaGroundedBlend, "anchored_fitted_ewma")


def _anchored_trend_gew() -> Blender:
    return AnchoredEmpirical(
        GroundedEqualWeight, "anchored_trend_grounded", use_trend=True
    )


register("anchored_grounded_equal_weight", _anchored_gew)
register("anchored_inverse_mse", _anchored_inverse_mse)
register("anchored_fitted_grounded", _anchored_fitted_gew)
register("anchored_fitted_ewma", _anchored_fitted_ewma)
register("anchored_trend_grounded", _anchored_trend_gew)
