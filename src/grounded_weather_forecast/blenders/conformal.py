"""Adaptive conformal intervals calibrated on later, out-of-sample forecasts.

A compact conformal-PID-style tracker (Angelopoulos, Candès & Tibshirani
2023; SAOCP-adjacent): per (lead bucket x day/night) cell, an online quantile
tracker follows the absolute-residual quantile of the base blend at a step of
0.1x the trailing-window score maximum (the paper's scale heuristic — the P
term), seeded at readiness with the conservative empirical quantile of the
buffered scores so a large systematic offset is covered immediately. A
saturating tangent integrator adds at most ~2.6x the score scale on top when
realized coverage drifts from target, with authority that decays as log t / t
(the I term — recovers coverage through regime shifts without the unbounded
windup a linear integral accumulates during a persistent-miss run). Every
calibration score also feeds a global cell per day/night flag, so buckets
whose calibration split is empty (long leads under an expanding archive) fall
back to pooled radii instead of emitting nothing. The base model is retained
from a chronological proper-training split; only predictions for the later
calibration split update the interval tracker. This avoids the optimistic
in-sample residuals that would result from evaluating a flexible base model
on rows it already fitted.

Intervals are symmetric around the base point — the honest first cut; the
EMOS/IDR heads carry asymmetry where the data supports it, and the
leaderboard arbitrates.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Self

import numpy as np

from grounded_weather_forecast.blenders.combine import GroundedEqualWeight
from grounded_weather_forecast.blenders.ewma_grounding import EwmaGroundedBlend
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
    SourceKind,
    SupervisedSlice,
    TargetKind,
    VariableSpec,
)
from grounded_weather_forecast.leads import LeadBucket, buckets_for_product

# target central coverages and the quantile levels they imply
COVERAGES: tuple[float, ...] = (0.5, 0.8, 0.9)
QUANTILE_LEVELS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.75, 0.9, 0.95)
_STEP = 0.1  # quantile-tracker step, in units of the trailing score maximum
_SCALE_WINDOW = 50  # trailing scores kept for the scale anchor and warm start
_CSAT = 0.5  # integrator threshold scale (sublinear h(t) = t * _CSAT / log t)
_ARG_CLIP = 1.2  # tan argument bound: integrator authority tops out ~2.6x scale
_GLOBAL_LABEL = "__global__"
_MIN_UPDATES = 20
_MIN_PROPER_ROWS = 60
_MIN_CALIBRATION_ROWS = 20
_PROPER_FRACTION = 0.7
_HOURLY_RESOLUTION_DELAY_US = 2 * 3_600_000_000
_DAILY_RESOLUTION_DELAY_US = 25 * 3_600_000_000


@dataclass
class _CellState:
    """Radii for each coverage target plus the trailing score buffer."""

    radii: FloatArray
    error_sums: FloatArray
    recent: deque[float] = field(default_factory=lambda: deque(maxlen=_SCALE_WINDOW))
    updates: int = 0

    def update(self, score: float) -> None:
        self.updates += 1
        self.recent.append(score)
        if self.updates < _MIN_UPDATES:
            return
        if self.updates == _MIN_UPDATES:
            self._warm_start()
            return
        step = _STEP * self.scale_anchor()
        for index, coverage in enumerate(COVERAGES):
            covered = score <= self.effective_radius(index)
            # P: pinball-gradient quantile tracking of the score distribution
            self.radii[index] += step * (coverage if not covered else coverage - 1.0)
            self.radii[index] = max(self.radii[index], 0.0)
            # I: realized-coverage error accumulates without clipping; the
            # saturating tangent in effective_radius bounds its authority
            self.error_sums[index] += coverage - float(covered)

    def _warm_start(self) -> None:
        # Conservative split-conformal quantile of the buffered scores, so a
        # systematic offset is covered at readiness instead of after hundreds
        # of tracker steps.
        ordered = np.sort(np.asarray(self.recent, dtype=float))
        count = ordered.shape[0]
        for index, coverage in enumerate(COVERAGES):
            rank = min(count, math.ceil((count + 1) * coverage))
            self.radii[index] = float(ordered[rank - 1])
        self.error_sums[:] = 0.0

    def scale_anchor(self) -> float:
        if not self.recent:
            return 1e-6
        return max(*self.recent, 1e-6)

    def effective_radius(self, index: int) -> float:
        t = max(self.updates, 2)
        argument = float(
            np.clip(
                float(self.error_sums[index]) * math.log(t) / (t * _CSAT),
                -_ARG_CLIP,
                _ARG_CLIP,
            )
        )
        integrator = self.scale_anchor() * math.tan(argument)
        return max(float(self.radii[index]) + integrator, 0.0)

    def ready(self) -> bool:
        return self.updates >= _MIN_UPDATES


def _fresh_cell() -> _CellState:
    return _CellState(
        radii=np.zeros(len(COVERAGES)), error_sums=np.zeros(len(COVERAGES))
    )


def _bucket_labels(
    lead_hours: FloatArray, buckets: tuple[LeadBucket, ...]
) -> list[str]:
    labels = []
    for lead in lead_hours:
        for bucket in buckets:
            if bucket.contains(float(lead)):
                labels.append(bucket.label)
                break
        else:
            labels.append("__outside__")
    return labels


def _day_flags(x: ForecastMatrix) -> np.ndarray:
    if "solar_elevation_deg" in x.features.columns:
        elevation = (
            x.features["solar_elevation_deg"].cast(float).fill_null(0.0).to_numpy()
        )
        return elevation > 0.0
    return np.zeros(x.n_rows, dtype=bool)


def _time_us(x: ForecastMatrix, column: str) -> np.ndarray | None:
    if column not in x.features.columns:
        return None
    values = x.features[column].to_numpy()
    try:
        return values.astype("datetime64[us]").astype(np.int64)
    except (TypeError, ValueError):
        return np.asarray(
            [np.datetime64(value, "us").astype(np.int64) for value in values],
            dtype=np.int64,
        )


def _resolution_us(x: ForecastMatrix, issue_us: np.ndarray) -> np.ndarray:
    if (known := _time_us(x, "truth_known_at")) is not None:
        return known
    if (valid := _time_us(x, "valid_time")) is not None:
        return valid + _HOURLY_RESOLUTION_DELAY_US
    if (forecast_date := _time_us(x, "forecast_date")) is not None:
        # Persisted daily matrices carry the exact timezone-aware value. This
        # conservative fallback supports manually constructed contracts.
        return forecast_date + _DAILY_RESOLUTION_DELAY_US
    return issue_us + np.rint(x.lead_hours * 3_600_000_000).astype(np.int64)


def _subset(train: SupervisedSlice, rows: np.ndarray) -> SupervisedSlice:
    x = ForecastMatrix.build(
        sources=train.x.sources,
        values=train.x.values[rows],
        lead_hours=train.x.lead_hours[rows],
        features=train.x.features[rows],
        product=train.x.product,
    )
    return SupervisedSlice(
        x=x,
        y=train.y[rows],
        variable=train.variable,
        source_kind=SourceKind(train.source_kind),
    )


@dataclass
class Conformal:
    """Wrapper: any base blend plus tracked symmetric interval radii."""

    base_factory: BlenderFactory
    method_id: str = "conformal_gew"
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _cells: dict[tuple[str, bool], _CellState] = field(default_factory=dict)
    _split_metadata: dict[str, object] = field(default_factory=dict)

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._cells.clear()
        issue = _time_us(train.x, "issue_time")
        if issue is None:
            self._base = self.base_factory().fit(train)
            self._split_metadata = {
                "strategy": "point_only",
                "reason": "issue_time feature unavailable",
                "proper_rows": train.x.n_rows,
                "calibration_rows": 0,
            }
            return self
        unique_issues = np.unique(issue)
        split_index = int(np.floor(unique_issues.shape[0] * _PROPER_FRACTION))
        split_index = min(max(split_index, 1), max(unique_issues.shape[0] - 1, 1))
        if unique_issues.shape[0] < 2:
            self._base = self.base_factory().fit(train)
            self._split_metadata = {
                "strategy": "point_only",
                "reason": "fewer than two issue times",
                "proper_rows": train.x.n_rows,
                "calibration_rows": 0,
            }
            return self
        cutoff = int(unique_issues[split_index])
        resolution = _resolution_us(train.x, issue)
        proper_rows = np.flatnonzero((issue < cutoff) & (resolution <= cutoff))
        calibration_rows = np.flatnonzero(issue >= cutoff)
        if (
            proper_rows.shape[0] < _MIN_PROPER_ROWS
            or calibration_rows.shape[0] < _MIN_CALIBRATION_ROWS
        ):
            self._base = self.base_factory().fit(train)
            self._split_metadata = {
                "strategy": "point_only",
                "reason": "chronological split below minimum row counts",
                "proper_rows": int(proper_rows.shape[0]),
                "calibration_rows": int(calibration_rows.shape[0]),
                "cutoff_issue_us": cutoff,
            }
            return self
        proper = _subset(train, proper_rows)
        self._base = self.base_factory().fit(proper)
        base_point = self._base.predict(train.x).point
        scores = np.abs(train.y - base_point)
        labels = _bucket_labels(
            train.x.lead_hours, buckets_for_product(train.x.product)
        )
        day = _day_flags(train.x)
        order = np.lexsort(
            (
                train.x.lead_hours[calibration_rows],
                issue[calibration_rows],
                resolution[calibration_rows],
            )
        )
        for row in calibration_rows[order]:
            if not np.isfinite(scores[row]):
                continue
            is_day = bool(day[row])
            # every score also feeds the pooled cell, so buckets with an empty
            # calibration split fall back to global radii instead of emitting
            # nothing
            for key in ((labels[row], is_day), (_GLOBAL_LABEL, is_day)):
                self._cells.setdefault(key, _fresh_cell()).update(float(scores[row]))
        self._split_metadata = {
            "strategy": "chronological_70_30",
            "proper_rows": int(proper_rows.shape[0]),
            "calibration_rows": int(calibration_rows.shape[0]),
            "cutoff_issue_us": cutoff,
            "minimum_proper_rows": _MIN_PROPER_ROWS,
            "minimum_calibration_rows": _MIN_CALIBRATION_ROWS,
        }
        return self

    def _cell_for(self, label: str, *, is_day: bool) -> _CellState | None:
        # ladder: bucket day/night, bucket other-phase, then the pooled cells
        for key in (
            (label, is_day),
            (label, not is_day),
            (_GLOBAL_LABEL, is_day),
            (_GLOBAL_LABEL, not is_day),
        ):
            cell = self._cells.get(key)
            if cell is not None and cell.ready():
                return cell
        return None

    def predict(self, x: ForecastMatrix) -> BlendResult:
        base_point = self._base.predict(x).point
        labels = _bucket_labels(x.lead_hours, buckets_for_product(x.product))
        day = _day_flags(x)
        radii = np.full((x.n_rows, len(COVERAGES)), np.nan)
        for row in range(x.n_rows):
            cell = self._cell_for(labels[row], is_day=bool(day[row]))
            if cell is None:
                continue
            radii[row] = [
                cell.effective_radius(index) for index in range(len(COVERAGES))
            ]
        if not np.isfinite(radii).any():
            return BlendResult(
                point=finalize_point(base_point, self._kind, self._variable)
            )
        # symmetric grid: (0.05, 0.1, 0.25) mirror (0.95, 0.9, 0.75)
        half = radii[:, ::-1]  # widest first: 0.9, 0.8, 0.5
        quantiles = np.column_stack(
            [
                base_point - half[:, 0],
                base_point - half[:, 1],
                base_point - half[:, 2],
                base_point + half[:, 2],
                base_point + half[:, 1],
                base_point + half[:, 0],
            ]
        )
        return BlendResult(
            point=finalize_point(base_point, self._kind, self._variable),
            quantiles=finalize_quantiles(quantiles, self._kind, self._variable),
            quantile_levels=QUANTILE_LEVELS,
        )

    def to_state(self) -> dict[str, object]:
        return {
            "schema_version": 3,
            "cells": {
                f"{label}|{'day' if is_day else 'night'}": {
                    "radii": cell.radii.tolist(),
                    "error_sums": cell.error_sums.tolist(),
                    "scale_anchor": cell.scale_anchor(),
                    "recent": list(cell.recent),
                    "updates": cell.updates,
                }
                for (label, is_day), cell in sorted(self._cells.items())
            },
            "coverages": list(COVERAGES),
            "calibration": self._split_metadata,
        }


def _conformal_gew() -> Blender:
    return Conformal(GroundedEqualWeight, "conformal_gew")


def _conformal_ewma() -> Blender:
    return Conformal(EwmaGroundedBlend, "conformal_ewma")


register("conformal_gew", _conformal_gew)
register("conformal_ewma", _conformal_ewma)
