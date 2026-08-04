"""Cluster-pruned equal weight: one vote per information source.

The live archive shows nws, pirate_weather, and visual_crossing forecasting
temperature with error correlations above 0.96 — three copies of one signal
that a plain mean counts three times, while genuinely independent feeds
(NBM) are diluted. Greedy correlation clustering on training errors keeps
the lowest-MAE member of each near-duplicate group and equal-weights the
survivors; the leaderboard arbitrates whether de-duplication beats raw
averaging per slice.
"""

from dataclasses import dataclass, field
from typing import Self

import numpy as np

from grounded_weather_forecast.blenders.protocol import (
    finalize_point,
    masked_average,
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
)

_MIN_OVERLAP = 24


def _error_correlation(a: FloatArray, b: FloatArray) -> float | None:
    """Pearson correlation over jointly finite rows; None when unestimable."""
    overlap = np.isfinite(a) & np.isfinite(b)
    if int(overlap.sum()) < _MIN_OVERLAP:
        return None
    a_overlap, b_overlap = a[overlap], b[overlap]
    if np.std(a_overlap) == 0.0 or np.std(b_overlap) == 0.0:
        return None
    return float(np.corrcoef(a_overlap, b_overlap)[0, 1])


@dataclass
class ClusterEqualWeight:
    method_id: str = "cluster_equal_weight"
    correlation_threshold: float = 0.9
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _sources: tuple[str, ...] = ()
    _keep: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._sources = train.x.sources
        errors = train.x.values - train.y[:, np.newaxis]
        finite = np.isfinite(errors)
        counts = finite.sum(axis=0)
        sums = np.where(finite, np.abs(errors), 0.0).sum(axis=0)
        mae_per_source = np.where(counts > 0, sums / np.maximum(counts, 1), np.inf)
        keep = np.zeros(errors.shape[1], dtype=bool)
        # Best sources claim their cluster first; an unestimable correlation
        # (thin overlap, degenerate errors) is never treated as duplication.
        for candidate in np.argsort(mae_per_source, kind="stable"):
            duplicate = any(
                (rho := _error_correlation(errors[:, candidate], errors[:, kept]))
                is not None
                and abs(rho) >= self.correlation_threshold
                for kept in np.nonzero(keep)[0]
            )
            if not duplicate:
                keep[candidate] = True
        self._keep = keep
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        keep = self._keep
        if keep.shape[0] != x.availability.shape[1]:
            keep = np.ones(x.availability.shape[1], dtype=bool)
        availability = x.availability & keep[np.newaxis, :]
        point = masked_average(x.values, availability)
        # A row whose available sources were all pruned falls back to the
        # full mean rather than losing coverage to de-duplication.
        fallback = masked_average(x.values, x.availability)
        point = np.where(np.isfinite(point), point, fallback)
        return BlendResult(point=finalize_point(point, self._kind, self._variable))

    def to_state(self) -> dict[str, object]:
        """Which sources survived clustering, in matrix order."""
        kept = zip(self._sources, self._keep, strict=True)
        return {
            "method_id": self.method_id,
            "correlation_threshold": self.correlation_threshold,
            "kept": [source for source, keep in kept if keep],
            "pruned": [
                source
                for source, keep in zip(self._sources, self._keep, strict=True)
                if not keep
            ],
        }


register("cluster_equal_weight", ClusterEqualWeight)

_PROTOCOL_CHECK: tuple[type[Blender], ...] = (ClusterEqualWeight,)
