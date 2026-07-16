"""Shared blender scaffolding: weight renormalization, masked averaging,
target clipping, and per-lead-bucket fitting."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from grounded_weather_forecast.contracts import BoolArray, FloatArray, TargetKind
from grounded_weather_forecast.leads import LeadBucket


def renormalize_weights(weights: FloatArray, availability: BoolArray) -> FloatArray:
    """Per-row weights over available sources, renormalized to sum to 1.

    Rows with no available source get all-zero weights.
    """
    masked = np.where(availability, weights[np.newaxis, :], 0.0)
    totals = masked.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(totals > 0.0, masked / totals, 0.0)


def masked_average(
    values: FloatArray, availability: BoolArray, weights: FloatArray | None = None
) -> FloatArray:
    """Weighted average over available sources; NaN where none are available."""
    k = values.shape[1]
    base = np.ones(k, dtype=np.float64) if weights is None else weights
    row_weights = renormalize_weights(base, availability)
    filled = np.where(availability, values, 0.0)
    point = (filled * row_weights).sum(axis=1)
    return np.where(availability.any(axis=1), point, np.nan)


def finalize_point(point: FloatArray, kind: TargetKind) -> FloatArray:
    """Clip probability targets into [0, 1]; pass continuous through."""
    match kind:
        case TargetKind.PROBABILITY:
            return np.clip(point, 0.0, 1.0)
        case _:
            return point


@dataclass
class PerBucketFitter[S]:
    """Fit one state object per lead bucket, with a global-fit fallback.

    ``fit_one`` receives the row subset for a bucket and returns the state;
    buckets with fewer than ``min_rows`` rows fall back to the global state.
    """

    buckets: tuple[LeadBucket, ...]
    fit_one: Callable[[np.ndarray], S]
    min_rows: int = 24

    def fit(self, lead: FloatArray) -> "FittedBuckets[S]":
        all_rows = np.arange(lead.shape[0])
        global_state = self.fit_one(all_rows)
        states: dict[str, S] = {}
        for bucket in self.buckets:
            rows = all_rows[(lead >= bucket.lo) & (lead < bucket.hi)]
            if rows.shape[0] >= self.min_rows:
                states[bucket.label] = self.fit_one(rows)
        return FittedBuckets(
            buckets=self.buckets, states=states, global_state=global_state
        )


@dataclass(frozen=True)
class FittedBuckets[S]:
    buckets: tuple[LeadBucket, ...]
    states: Mapping[str, S]
    global_state: S

    def state_for(self, lead: float) -> S:
        for bucket in self.buckets:
            if bucket.contains(lead):
                return self.states.get(bucket.label, self.global_state)
        return self.global_state

    def apply(self, lead: FloatArray, use: Callable[[S, np.ndarray], None]) -> None:
        """Group rows by bucket state and invoke ``use(state, row_indices)``."""
        all_rows = np.arange(lead.shape[0])
        assigned = np.zeros(lead.shape[0], dtype=bool)
        for bucket in self.buckets:
            mask = (lead >= bucket.lo) & (lead < bucket.hi)
            if mask.any():
                use(self.states.get(bucket.label, self.global_state), all_rows[mask])
                assigned |= mask
        if (~assigned).any():
            use(self.global_state, all_rows[~assigned])
