"""Source-count shrinkage for precipitation: trust few providers less.

Past day six only one or two providers still publish daily precipitation, so
the equal-weight blend degenerates to a single provider's monsoon guess served
nearly undiluted (the 2026-08-07 phantom-rain report). ``DampedBlend`` already
regresses toward climatology, but its per-bucket alpha is fitted by MAE search
— exactly where the far buckets are thinnest, the fit is noisiest. This method
imposes the trust schedule structurally instead: a per-row pseudo-count weight
``n / (n + k)`` toward the provider mean, with the harmonic climatology (which
fits ~0 on a dry archive and adapts as wet-season truth accumulates) as the
anchor. No skill parameter is estimated, so the method is well-defined on
slices with almost no evidence — the exact regime it exists for. The A/B
against ``damped_grounded_equal_weight`` is arbitrated by the leaderboard.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np

from grounded_weather_forecast.blenders.baselines import (
    EqualWeight,
    HarmonicClimatology,
)
from grounded_weather_forecast.blenders.protocol import finalize_point
from grounded_weather_forecast.blenders.registry import register
from grounded_weather_forecast.contracts import (
    Blender,
    BlendResult,
    ForecastMatrix,
    SupervisedSlice,
    TargetKind,
    VariableSpec,
)


@dataclass
class SparseShrink:
    """``w * provider_mean + (1 - w) * climatology`` with ``w = n / (n + k)``."""

    method_id: str = "precip_sparse_shrink"
    pseudo_sources: float = 2.0
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._base = EqualWeight().fit(train)
        self._climatology = HarmonicClimatology().fit(train)
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        base = self._base.predict(x).point
        anchor = self._climatology.predict(x).point
        counts = x.availability.sum(axis=1).astype(np.float64)
        weight = counts / (counts + self.pseudo_sources)
        shrunk = weight * base + (1.0 - weight) * anchor
        # A row with no provider signal is the w -> 0 limit: pure climatology.
        point = np.where(np.isfinite(shrunk), shrunk, anchor)
        return BlendResult(point=finalize_point(point, self._kind, self._variable))

    def to_state(self) -> dict[str, object]:
        return {"method_id": self.method_id, "pseudo_sources": self.pseudo_sources}


register("precip_sparse_shrink", SparseShrink, variables=frozenset({"precip_sum_mm"}))

_PROTOCOL_CHECK: tuple[type[Blender], ...] = (SparseShrink,)
