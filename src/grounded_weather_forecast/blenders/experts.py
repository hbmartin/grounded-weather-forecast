"""Online expert aggregation: EWA and BOA with sleeping experts and fixed share.

The experts are the grounded sources. Weights update sequentially in
training-row order, separately per lead bucket, because skill orderings differ
by horizon. Two mechanisms make this family the drift specialist of the lineup:

- **Sleeping experts.** A source outside its horizon is simply absent from the
  round: it is neither updated nor penalized, because the reduction assigns a
  sleeping expert the awake mixture's loss, i.e. an update factor of exactly
  one. Ragged provider horizons need no special casing.
- **Fixed share.** After each multiplicative update, a small fraction of the
  awake mass is redistributed uniformly. This floors every expert's weight and
  bounds the weight ratio. Without it the learning rate decays, an expert that
  dominated early accumulates an insurmountable lead, and the aggregator cannot
  follow a regime change — a provider silently swapping its backend model would
  keep its weight forever. With it, the weights track the best *sequence* of
  experts, and recovery takes days rather than never.

``ewa`` uses a known-horizon exponential update. ``boa`` (Bernstein Online
Aggregation) uses a second-order update whose per-expert rate shrinks with
accumulated regret variance, so it adapts faster when losses are volatile.
"""

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, Self

import numpy as np

from grounded_weather_forecast.blenders.grounding import AffineGrounding
from grounded_weather_forecast.blenders.protocol import finalize_point
from grounded_weather_forecast.blenders.registry import register
from grounded_weather_forecast.contracts import (
    BlendResult,
    BoolArray,
    FloatArray,
    ForecastMatrix,
    Product,
    SupervisedSlice,
    TargetKind,
    VariableSpec,
)
from grounded_weather_forecast.leads import bucket_for_product

_MAX_ETA = 0.5
# The drift-vs-average-case knob. The loser's steady-state weight is roughly
# share / (2 * per-round learning rate), so a share this small costs nothing
# against a stationary best expert while still capping the weight ratio, which
# is what lets a recovering source climb back after a regime change.
_SHARE = 0.005
_MIN_AWAKE = 2
_GLOBAL_BUCKET = "__global__"
_STATE_SCHEMA_VERSION = 2


@dataclass
class _BucketState:
    weights: FloatArray
    regret_variance: FloatArray
    horizon: int
    steps: int = 0


@dataclass(frozen=True)
class _BucketProgress:
    """Identity of the exact history prefix already consumed by one bucket."""

    resolution_us: int
    issue_us: int
    lead_hours: float
    rows: int
    prefix_digest: str


def _uniform(n_experts: int) -> _BucketState:
    return _BucketState(
        weights=np.full(n_experts, 1.0 / max(n_experts, 1)),
        regret_variance=np.full(n_experts, 1e-12),
        horizon=1,
    )


def _copy_bucket(state: _BucketState) -> _BucketState:
    """A detached copy. ``_BucketState`` holds arrays that ``_step`` mutates
    in place, so a shallow ``replace`` would alias them back."""
    return _BucketState(
        weights=state.weights.copy(),
        regret_variance=state.regret_variance.copy(),
        horizon=state.horizon,
        steps=state.steps,
    )


def _awake_weights(weights: FloatArray, awake: BoolArray) -> FloatArray | None:
    """Weights renormalized over the awake set; ``None`` if nobody is awake."""
    if not awake.any():
        return None
    subset = weights[awake]
    total = float(subset.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(subset.shape[0], 1.0 / subset.shape[0])
    return subset / total


@dataclass
class OnlineExperts:
    """Sequential expert aggregation over grounded sources."""

    method_id: str
    scheme: Literal["ewa", "boa"]
    share: float = _SHARE
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _grounding: AffineGrounding = field(default_factory=AffineGrounding)
    _grounding_state: dict[str, object] | None = None
    _states: dict[str, _BucketState] = field(default_factory=dict)
    _progress: dict[str, _BucketProgress] = field(default_factory=dict)
    _sources: tuple[str, ...] = ()

    def _update_factor(
        self,
        state: _BucketState,
        normalized: FloatArray,
        mixture_loss: float,
        awake: BoolArray,
    ) -> FloatArray:
        k = int(awake.sum())
        match self.scheme:
            case "ewa":
                eta = min(_MAX_ETA, float(np.sqrt(8.0 * np.log(k) / state.horizon)))
                exponent = -eta * (normalized[awake] - mixture_loss)
            case "boa":
                regret = mixture_loss - normalized[awake]
                state.regret_variance[awake] += regret**2
                eta = np.minimum(
                    _MAX_ETA, np.sqrt(np.log(k) / state.regret_variance[awake])
                )
                exponent = eta * regret - (eta * regret) ** 2
        return np.exp(np.clip(exponent, -50.0, 50.0))

    def _step(self, state: _BucketState, losses: FloatArray, awake: BoolArray) -> None:
        """One sequential round; ``losses`` is finite on the awake set.

        The loss scale, the expert count in the learning rate, and the mixture
        all come from the awake set alone, so a source that is asleep (or never
        present at all) cannot perturb the update.
        """
        if int(awake.sum()) < _MIN_AWAKE:
            return
        weights = _awake_weights(state.weights, awake)
        if weights is None:
            return
        state.steps += 1
        # Per-round range normalization keeps losses in [0, 1], the precondition
        # for both regret bounds, without assuming a global loss scale.
        scale = max(float(losses[awake].max()), 1e-9)
        normalized = losses / scale
        mixture_loss = float((weights * normalized[awake]).sum())

        updated = state.weights[awake] * self._update_factor(
            state, normalized, mixture_loss, awake
        )
        mass = float(updated.sum())
        if not np.isfinite(mass) or mass <= 0.0:
            updated = np.full(updated.shape[0], 1.0 / updated.shape[0])
            mass = 1.0
        # Fixed share, applied inside the awake block so the mass relationship
        # to sleeping experts is left untouched.
        shared = (1.0 - self.share) * updated + self.share * mass / updated.shape[0]
        state.weights[awake] = shared
        total = float(state.weights.sum())
        if total > 0.0:  # uniform rescale; relative weights are unchanged
            state.weights /= total

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._states = {}
        self._progress = {}
        grounding = AffineGrounding().fit(train)
        self._replay(train, grounding)
        return self

    def advance(self, train: SupervisedSlice) -> Self:
        """Continue from per-bucket target-resolution cursors.

        A forecast's loss becomes available at its target time, not its issue
        time. Each lead bucket therefore owns a cursor ordered by target time,
        issue time, and lead. A digest of the processed prefix detects archive
        corrections or retention changes; callers then discard the state and
        replay rather than silently combining incompatible histories.
        """
        if self._sources and self._sources != train.x.sources:
            msg = "expert source order changed; full replay required"
            raise ValueError(msg)
        resolution, _ = self._row_times(train.x)
        if self._progress and (
            not resolution.size
            or max(progress.resolution_us for progress in self._progress.values())
            > int(resolution.max())
        ):
            msg = "expert state extends beyond the causal training history"
            raise ValueError(msg)
        grounding = AffineGrounding().fit(train)
        grounding_state = grounding.to_state()
        # EWA's known-horizon learning rate changes when the queue grows, and
        # every expert scheme's historic losses change when grounding refits.
        # Validate against the persisted prefix on a detached object, then
        # replay the complete queue so one state never mixes incompatible loss
        # definitions or learning-rate horizons.
        requires_replay = bool(self._progress) and (
            self.scheme == "ewa" or self._grounding_state != grounding_state
        )
        if requires_replay:
            probe = deepcopy(self)
            probe._replay(train, grounding)  # noqa: SLF001 - detached self-check
            self._states = {}
            self._progress = {}
        self._replay(train, grounding)
        self._kind = train.variable.kind
        self._variable = train.variable
        return self

    @staticmethod
    def _time_us(x: ForecastMatrix, name: str) -> np.ndarray | None:
        if name not in x.features.columns:
            return None
        values = x.features[name].to_numpy()
        try:
            return values.astype("datetime64[us]").astype(np.int64)
        except (TypeError, ValueError):
            return np.asarray(
                [np.datetime64(value, "us").astype(np.int64) for value in values],
                dtype=np.int64,
            )

    @classmethod
    def _row_times(cls, x: ForecastMatrix) -> tuple[np.ndarray, np.ndarray]:
        issue = cls._time_us(x, "issue_time")
        if issue is None:
            issue = np.arange(x.n_rows, dtype=np.int64)
        resolution = cls._time_us(x, "truth_known_at")
        if resolution is None:
            resolution = cls._time_us(x, "valid_time")
        if resolution is None:
            resolution = cls._time_us(x, "forecast_date")
        if resolution is None:
            resolution = issue + np.rint(x.lead_hours * 3_600_000_000).astype(np.int64)
        return resolution, issue

    @staticmethod
    def _digest_rows(train: SupervisedSlice, rows: np.ndarray) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "sources": train.x.sources,
                    "product": train.x.product.value,
                    "variable": train.variable.name,
                },
                sort_keys=True,
            ).encode()
        )
        values = train.x.values[rows].astype("<f8", copy=True)
        values[np.isnan(values)] = np.nan
        digest.update(values.tobytes())
        digest.update(train.y[rows].astype("<f8", copy=False).tobytes())
        digest.update(train.x.lead_hours[rows].astype("<f8", copy=False).tobytes())
        resolution, issue = OnlineExperts._row_times(train.x)
        digest.update(resolution[rows].astype("<i8", copy=False).tobytes())
        digest.update(issue[rows].astype("<i8", copy=False).tobytes())
        return digest.hexdigest()

    @staticmethod
    def _after_cursor(
        resolution: int,
        issue: int,
        lead: float,
        progress: _BucketProgress,
    ) -> bool:
        return (resolution, issue, lead) > (
            progress.resolution_us,
            progress.issue_us,
            progress.lead_hours,
        )

    def _bucket_rows(
        self,
        train: SupervisedSlice,
        bucket: str,
        buckets: list[str],
        resolution: np.ndarray,
        issue: np.ndarray,
    ) -> np.ndarray:
        rows = np.asarray(
            [row for row, label in enumerate(buckets) if label == bucket],
            dtype=np.int64,
        )
        if not rows.size:
            return rows
        order = np.lexsort((train.x.lead_hours[rows], issue[rows], resolution[rows]))
        return rows[order]

    def _replay(
        self, train: SupervisedSlice, grounding: AffineGrounding | None = None
    ) -> None:
        """Advance every bucket's cursor, swapping state in only on success.

        ``advance`` raises ``ValueError`` as a documented signal, so it must
        not leave a half-advanced object behind: every mutation lands on
        locals until the last bucket has validated. That covers the grounding
        and ``_sources`` too, which used to be assigned before the first
        bucket was even read.
        """
        active = self._grounding if grounding is None else grounding
        corrected = active.transform(train.x)
        n_experts = len(train.x.sources)
        buckets = [
            bucket_for_product(train.x.product, lead) or _GLOBAL_BUCKET
            for lead in train.x.lead_hours
        ]
        horizons = Counter(buckets)
        resolution, issue = self._row_times(train.x)
        states = {label: _copy_bucket(state) for label, state in self._states.items()}
        progress: dict[str, _BucketProgress] = {}
        for bucket in sorted(set(buckets) | set(self._progress)):
            rows = self._bucket_rows(train, bucket, buckets, resolution, issue)
            previous = self._progress.get(bucket)
            if not rows.size:
                # Only a bucket carried over from `_progress` can be empty; one
                # drawn from this matrix always has rows. Report the retention
                # loss for what it is rather than as a changed history.
                msg = (
                    f"processed expert bucket {bucket!r} vanished from the "
                    f"training history after "
                    f"{previous.rows if previous else 0} consumed rows; "
                    "retention or filtering changed"
                )
                raise ValueError(msg)
            if previous is not None:
                prefix = np.asarray(
                    [
                        row
                        for row in rows
                        if not self._after_cursor(
                            int(resolution[row]),
                            int(issue[row]),
                            float(train.x.lead_hours[row]),
                            previous,
                        )
                    ],
                    dtype=np.int64,
                )
                if (
                    prefix.shape[0] != previous.rows
                    or self._digest_rows(train, prefix) != previous.prefix_digest
                ):
                    msg = f"processed expert history changed for bucket {bucket!r}"
                    raise ValueError(msg)
            else:
                prefix = np.empty(0, dtype=np.int64)
            for row in rows[prefix.shape[0] :]:
                if bucket not in states:
                    fresh = _uniform(n_experts)
                    fresh.horizon = horizons[bucket]
                    states[bucket] = fresh
                else:
                    states[bucket].horizon = max(
                        states[bucket].horizon, horizons[bucket]
                    )
                awake = train.x.availability[row] & np.isfinite(corrected[row])
                losses = np.where(awake, (corrected[row] - train.y[row]) ** 2, 0.0)
                self._step(states[bucket], losses, awake)
            last = int(rows[-1])
            progress[bucket] = _BucketProgress(
                resolution_us=int(resolution[last]),
                issue_us=int(issue[last]),
                lead_hours=float(train.x.lead_hours[last]),
                rows=int(rows.shape[0]),
                prefix_digest=self._digest_rows(train, rows),
            )
        self._sources = train.x.sources
        self._states = states
        self._progress = progress
        self._grounding = active
        self._grounding_state = active.to_state()

    def _state_for(self, product: Product, lead: float, n_experts: int) -> _BucketState:
        """Fitted state for a lead, or a uniform default for unseen buckets."""
        state = self._states.get(bucket_for_product(product, lead) or _GLOBAL_BUCKET)
        if state is None or state.weights.shape[0] != n_experts:
            return _uniform(n_experts)
        return state

    def predict(self, x: ForecastMatrix) -> BlendResult:
        corrected = self._grounding.transform(x)
        n_experts = len(x.sources)
        point = np.full(x.n_rows, np.nan)
        for row in range(x.n_rows):
            awake = x.availability[row] & np.isfinite(corrected[row])
            state = self._state_for(x.product, float(x.lead_hours[row]), n_experts)
            weights = _awake_weights(state.weights, awake)
            if weights is None:
                continue
            point[row] = float((weights * corrected[row][awake]).sum())
        return BlendResult(point=finalize_point(point, self._kind, self._variable))

    def bucket_weights(self, bucket_label: str) -> FloatArray | None:
        """Final normalized weights for one bucket (inspection and tests)."""
        state = self._states.get(bucket_label)
        if state is None:
            return None
        return _awake_weights(
            state.weights, np.ones(state.weights.shape[0], dtype=bool)
        )

    def to_state(self) -> dict[str, object]:
        return {
            "schema_version": _STATE_SCHEMA_VERSION,
            "scheme": self.scheme,
            "share": self.share,
            "sources": list(self._sources),
            "buckets": {
                label: {
                    "weights": state.weights.tolist(),
                    "regret_variance": state.regret_variance.tolist(),
                    "horizon": state.horizon,
                    "steps": state.steps,
                }
                for label, state in self._states.items()
            },
            "progress": {
                label: {
                    "resolution_us": progress.resolution_us,
                    "issue_us": progress.issue_us,
                    "lead_hours": progress.lead_hours,
                    "rows": progress.rows,
                    "prefix_digest": progress.prefix_digest,
                }
                for label, progress in self._progress.items()
            },
            "grounding": self._grounding_state or self._grounding.to_state(),
        }

    def observability_state(self) -> dict[str, object]:
        """``to_state`` without replay cursors: the dashboard-facing view."""
        state = self.to_state()
        state.pop("progress", None)
        return state

    @classmethod
    def from_state(cls, state: dict[str, object], method_id: str) -> "OnlineExperts":
        """Rehydrate persisted weights; grounding re-fits on the next advance."""
        if state.get("schema_version") != _STATE_SCHEMA_VERSION:
            msg = "legacy or unknown expert state schema; full replay required"
            raise ValueError(msg)
        experts = cls(method_id=method_id, scheme=_state_scheme(state))
        if experts.scheme != method_id:
            msg = (
                f"expert state scheme {experts.scheme!r} does not match "
                f"requested method {method_id!r}"
            )
            raise ValueError(msg)
        experts.share = _state_share(state)
        experts._sources = _state_sources(state)
        n_experts = len(experts._sources)
        buckets = state.get("buckets")
        if isinstance(buckets, dict):
            experts._states = {
                str(label): _bucket_state(str(label), raw, n_experts)
                for label, raw in buckets.items()
            }
        progress = state.get("progress")
        if not isinstance(progress, dict):
            msg = "expert state has no processed-prefix metadata"
            raise ValueError(msg)
        experts._progress = {
            str(label): _bucket_progress(str(label), raw)
            for label, raw in progress.items()
        }
        if set(experts._states) != set(experts._progress):
            msg = "expert state buckets and progress do not match"
            raise ValueError(msg)
        grounding = state.get("grounding")
        if not isinstance(grounding, dict):
            msg = "expert state has no grounding metadata"
            raise ValueError(msg)
        experts._grounding_state = {str(key): value for key, value in grounding.items()}
        return experts


def _state_scheme(state: Mapping[str, object]) -> Literal["ewa", "boa"]:
    match state.get("scheme"):
        case "ewa":
            return "ewa"
        case "boa":
            return "boa"
        case other:
            msg = f"unknown expert scheme in state: {other!r}"
            raise ValueError(msg)


def _state_sources(state: Mapping[str, object]) -> tuple[str, ...]:
    sources = state.get("sources")
    if not isinstance(sources, list) or not sources:
        msg = "expert state names no sources; full replay required"
        raise ValueError(msg)
    normalized: list[str] = []
    for source in sources:
        if not isinstance(source, str) or not source:
            msg = "expert state carries an invalid source name"
            raise ValueError(msg)
        normalized.append(source)
    if len(set(normalized)) != len(normalized):
        msg = "expert state carries duplicate source names"
        raise ValueError(msg)
    return tuple(normalized)


def _state_share(state: Mapping[str, object]) -> float:
    share = state.get("share")
    if (
        isinstance(share, bool)
        or not isinstance(share, (int, float))
        or not np.isfinite(share)
        or not 0.0 <= float(share) <= 1.0
    ):
        msg = f"invalid expert fixed-share value: {share!r}"
        raise ValueError(msg)
    return float(share)


def _bucket_state(label: str, raw: object, n_experts: int) -> _BucketState:
    """One rehydrated bucket, validated against the state's own source list.

    A weight vector of the wrong length used to survive here and fail much
    later inside ``_step`` as an opaque IndexError.
    """
    if not isinstance(raw, dict):
        msg = f"corrupt expert bucket {label!r}"
        raise ValueError(msg)
    weights = np.asarray(raw.get("weights", []), dtype=np.float64)
    regret_variance = np.asarray(raw.get("regret_variance", []), dtype=np.float64)
    expected = (n_experts,)
    if weights.shape != expected or regret_variance.shape != expected:
        msg = (
            f"expert bucket {label!r} carries weights/regret shapes "
            f"{weights.shape} and {regret_variance.shape}; expected {expected}; "
            "full replay required"
        )
        raise ValueError(msg)
    if (
        not np.isfinite(weights).all()
        or (weights < 0.0).any()
        or not np.isfinite(weights.sum())
        or float(weights.sum()) <= 0.0
    ):
        msg = f"expert bucket {label!r} carries invalid weights"
        raise ValueError(msg)
    if not np.isfinite(regret_variance).all() or (regret_variance < 0.0).any():
        msg = f"expert bucket {label!r} carries invalid regret variance"
        raise ValueError(msg)
    horizon = raw.get("horizon", 1)
    steps = raw.get("steps", 0)
    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, (int, float))
        or not np.isfinite(horizon)
        or int(horizon) != horizon
        or int(horizon) <= 0
    ):
        msg = f"expert bucket {label!r} carries invalid horizon {horizon!r}"
        raise ValueError(msg)
    if (
        isinstance(steps, bool)
        or not isinstance(steps, (int, float))
        or not np.isfinite(steps)
        or int(steps) != steps
        or int(steps) < 0
    ):
        msg = f"expert bucket {label!r} carries invalid steps {steps!r}"
        raise ValueError(msg)
    return _BucketState(
        weights=weights,
        regret_variance=regret_variance,
        horizon=int(horizon),
        steps=int(steps),
    )


def _bucket_progress(label: str, raw: object) -> _BucketProgress:
    if not isinstance(raw, dict):
        msg = f"corrupt expert progress for bucket {label!r}"
        raise ValueError(msg)
    resolution_us = raw.get("resolution_us")
    issue_us = raw.get("issue_us")
    lead_hours = raw.get("lead_hours")
    rows = raw.get("rows")
    prefix_digest = raw.get("prefix_digest")
    numeric = (int, float)
    if (
        isinstance(resolution_us, bool)
        or isinstance(issue_us, bool)
        or isinstance(lead_hours, bool)
        or isinstance(rows, bool)
        or not isinstance(resolution_us, numeric)
        or not isinstance(issue_us, numeric)
        or not isinstance(lead_hours, numeric)
        or not isinstance(rows, numeric)
        or not isinstance(prefix_digest, str)
        or not prefix_digest
        or not all(
            np.isfinite(value) for value in (resolution_us, issue_us, lead_hours, rows)
        )
        or int(rows) != rows
        or int(rows) < 0
    ):
        msg = f"corrupt expert progress for bucket {label!r}"
        raise ValueError(msg)
    return _BucketProgress(
        resolution_us=int(resolution_us),
        issue_us=int(issue_us),
        lead_hours=float(lead_hours),
        rows=int(rows),
        prefix_digest=prefix_digest,
    )


def _ewa() -> OnlineExperts:
    return OnlineExperts(method_id="ewa", scheme="ewa")


def _boa() -> OnlineExperts:
    return OnlineExperts(method_id="boa", scheme="boa")


register("ewa", _ewa)
register("boa", _boa)
