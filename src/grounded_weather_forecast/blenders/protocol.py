"""Shared blender scaffolding: weight renormalization, masked averaging,
target clipping, and per-lead-bucket fitting."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from grounded_weather_forecast.contracts import (
    BlendResult,
    BoolArray,
    FloatArray,
    Product,
    TargetKind,
    VariableSpec,
)
from grounded_weather_forecast.leads import LeadBucket, buckets_for_product


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


def finalize_point(
    point: FloatArray, kind: TargetKind, variable: VariableSpec | None = None
) -> FloatArray:
    """Clip probability targets into [0, 1]; clamp declared variable bounds.

    The bounds clamp mirrors the serve boundary, so the backtest scores the
    quantity a user would actually receive — never a negative wind speed.
    NaN (no prediction) passes through unchanged.
    """
    match kind:
        case TargetKind.PROBABILITY:
            point = np.clip(point, 0.0, 1.0)
        case _:
            pass
    if variable is None or (variable.minimum is None and variable.maximum is None):
        return point
    lower = -np.inf if variable.minimum is None else variable.minimum
    upper = np.inf if variable.maximum is None else variable.maximum
    return np.clip(point, lower, upper)


def finalize_quantiles(
    quantiles: FloatArray, kind: TargetKind, variable: VariableSpec | None = None
) -> FloatArray:
    """Monotone rearrangement plus the same clamps ``finalize_point`` applies.

    Row-wise ``np.sort`` is the standard fix for quantile crossing: it never
    worsens any proper scoring rule and guarantees emitted quantiles are a
    valid distribution. Every quantile emitter must route through here.
    """
    ordered = np.sort(quantiles, axis=1)
    match kind:
        case TargetKind.PROBABILITY:
            ordered = np.clip(ordered, 0.0, 1.0)
        case _:
            pass
    if variable is None or (variable.minimum is None and variable.maximum is None):
        return ordered
    lower = -np.inf if variable.minimum is None else variable.minimum
    upper = np.inf if variable.maximum is None else variable.maximum
    return np.clip(ordered, lower, upper)


def quantile_blend_result(
    quantiles: FloatArray,
    levels: tuple[float, ...],
    kind: TargetKind,
    variable: VariableSpec | None,
) -> BlendResult:
    """Finalized quantiles served with their own median as the point."""
    finalized = finalize_quantiles(quantiles, kind, variable)
    median = finalized[:, len(levels) // 2]
    return BlendResult(
        point=finalize_point(median, kind, variable),
        quantiles=finalized,
        quantile_levels=levels,
    )


def coefficient_state(
    method_id: str,
    variable: VariableSpec | None,
    fit_status: str,
    parameters: FloatArray | None,
    names: tuple[str, ...],
) -> dict[str, object]:
    """Glass-box ``to_state`` payload shared by coefficient-vector heads."""
    return {
        "schema_version": 1,
        "method_id": method_id,
        "variable": variable.name if variable else None,
        "fit_status": fit_status,
        "fitted": parameters is not None,
        "coefficients": None
        if parameters is None
        else {
            name: float(value) for name, value in zip(names, parameters, strict=True)
        },
    }


def fit_shrunk_buckets(
    product: Product,
    lead_hours: FloatArray,
    fit_one: Callable[[np.ndarray], FloatArray],
) -> "FittedBuckets[FloatArray]":
    """Per-bucket coefficient fits with linear shrinkage toward the global fit."""
    fitter = PerBucketFitter(
        buckets=buckets_for_product(product),
        fit_one=fit_one,
        blend=lambda local, glob, w: w * local + (1.0 - w) * glob,
    )
    return fitter.fit(lead_hours)


@dataclass
class PerBucketFitter[S]:
    """Fit one state object per lead bucket, with a global-fit fallback.

    ``fit_one`` receives the row subset for a bucket and returns the state.
    Without ``blend``, buckets with fewer than ``min_rows`` rows fall back to
    the global state — an all-or-nothing cliff, kept for states whose
    parameters cannot be safely interpolated. Supplying ``blend`` replaces the
    cliff with empirical-Bayes-style shrinkage: every non-empty bucket serves
    ``blend(local, global, w)`` with weight ``w = n / (n + prior_rows)``, so
    the global fit acts as a prior worth ``prior_rows`` rows of evidence. The
    default prior of ``min_rows / 4`` puts ``w = 0.8`` at exactly ``min_rows``
    rows — the local fit dominates right where the old cliff granted it full
    weight — while thinner buckets shade smoothly toward the global fit
    instead of a barely-qualified local fit serving its sampling noise as a
    constant bias.
    """

    buckets: tuple[LeadBucket, ...]
    fit_one: Callable[[np.ndarray], S]
    min_rows: int = 24
    blend: Callable[[S, S, float], S] | None = None
    prior_rows: float | None = None

    def fit(self, lead: FloatArray) -> "FittedBuckets[S]":
        all_rows = np.arange(lead.shape[0])
        global_state = self.fit_one(all_rows)
        states: dict[str, S] = {}
        for bucket in self.buckets:
            rows = all_rows[(lead >= bucket.lo) & (lead < bucket.hi)]
            if (state := self._bucket_state(rows, global_state)) is not None:
                states[bucket.label] = state
        return FittedBuckets(
            buckets=self.buckets, states=states, global_state=global_state
        )

    def _bucket_state(self, rows: np.ndarray, global_state: S) -> S | None:
        """One bucket's state: shrunk when ``blend`` is set, the cliff otherwise."""
        n = int(rows.shape[0])
        if self.blend is None:
            return self.fit_one(rows) if n >= self.min_rows else None
        if n == 0:
            return None
        prior = self.min_rows / 4.0 if self.prior_rows is None else self.prior_rows
        return self.blend(self.fit_one(rows), global_state, n / (n + prior))


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
