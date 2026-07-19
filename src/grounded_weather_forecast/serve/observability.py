"""Write-only observability snapshots of fitted blender internals.

Serving keeps its rehydration state in ``artifacts/state``; this module
mirrors a compact, read-only view of every fitted blender's internals into
``artifacts/observability`` for the operator dashboard. A bounded trajectory
history is kept only for the online-expert methods, whose weight paths are
the provider-backend-swap detector. Snapshot failures are always swallowed:
serving output is identical whether they land or not.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import polars as pl

from grounded_weather_forecast import __version__
from grounded_weather_forecast.artifacts import ArtifactError, ArtifactStore
from grounded_weather_forecast.config import Config
from grounded_weather_forecast.evaluation import dataset_fingerprint
from grounded_weather_forecast.storage import atomic_write_parquet, locked_path

_LOCK_TIMEOUT_SECONDS = 5.0
# Mirrors predict._STATEFUL_METHODS without importing it (no cycle).
_TRAJECTORY_METHODS = frozenset({"ewa", "boa"})

OBSERVABILITY_HISTORY_SCHEMA = pl.Schema(
    {
        "captured_at": pl.Datetime("us", "UTC"),
        "issue_time": pl.Datetime("us", "UTC"),
        "method_id": pl.String(),
        "product": pl.String(),
        "variable": pl.String(),
        "dataset_fingerprint": pl.String(),
        "state_json": pl.String(),
    }
)


@runtime_checkable
class SupportsObservabilityState(Protocol):
    def observability_state(self) -> dict[str, Any]: ...


@runtime_checkable
class SupportsToState(Protocol):
    def to_state(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ObservabilitySnapshot:
    """The latest persisted internals of one fitted (method, slice)."""

    method_id: str
    product: str
    variable: str
    dataset_fingerprint: str
    created_at: str
    issue_time: str | None
    state: dict[str, Any]


def observability_root(config: Config) -> Path:
    return config.artifacts_dir / "observability"


def observability_state(blender: object) -> dict[str, Any] | None:
    """A blender's compact internals, or None when it exposes none."""
    match blender:
        case SupportsObservabilityState():
            return blender.observability_state()
        case SupportsToState():
            return blender.to_state()
        case _:
            return None


def snapshot_observability(
    blender: object,
    *,
    method_id: str,
    product: str,
    variable: str,
    config: Config,
    issue_time: datetime,
) -> None:
    """Persist a compact snapshot of a fitted blender; never raises."""
    try:
        state = observability_state(blender)
        if state is None:
            return
        fingerprint = dataset_fingerprint(config)
        store = ArtifactStore(observability_root(config))
        store.save(
            fingerprint=fingerprint,
            method_id=method_id,
            product=product,
            variable=variable,
            state=state,
            meta={
                "issue_time": issue_time.isoformat(),
                "kind": "observability",
                "code_version": __version__,
            },
            # The dataset fingerprint changes on every `build-dataset`, so
            # without reclamation each rebuild mints a permanent new tree per
            # (method x product x variable) and this directory grows forever.
            # It runs inside `save`'s lock so it cannot delete a tree another
            # `predict` is mid-way through writing.
            reclaim_unreferenced=True,
        )
        if method_id in _TRAJECTORY_METHODS:
            _append_history(
                config,
                method_id=method_id,
                product=product,
                variable=variable,
                fingerprint=fingerprint,
                issue_time=issue_time,
                state=state,
            )
    except Exception:
        # Deliberately broad: this is write-only telemetry and the docstring
        # promises it never raises. ``observability_state`` runs arbitrary
        # blender code (e.g. LightGBM's own error type, which subclasses
        # Exception directly), and no snapshot failure may ever fail a
        # `predict`. KeyboardInterrupt/SystemExit are BaseException and still
        # propagate.
        return


def _append_history(
    config: Config,
    *,
    method_id: str,
    product: str,
    variable: str,
    fingerprint: str,
    issue_time: datetime,
    state: dict[str, Any],
) -> None:
    path = observability_root(config) / "history.parquet"
    row = pl.DataFrame(
        [
            {
                "captured_at": datetime.now(tz=UTC),
                "issue_time": issue_time,
                "method_id": method_id,
                "product": product,
                "variable": variable,
                "dataset_fingerprint": fingerprint,
                "state_json": json.dumps(state, sort_keys=True),
            }
        ],
        schema=OBSERVABILITY_HISTORY_SCHEMA,
    )
    horizon = datetime.now(tz=UTC) - timedelta(days=config.backtest.rolling_window_days)
    path.parent.mkdir(parents=True, exist_ok=True)
    with locked_path(path, timeout=_LOCK_TIMEOUT_SECONDS):
        combined = pl.concat([_read_history(path), row]) if path.exists() else row
        atomic_write_parquet(combined.filter(pl.col("captured_at") >= horizon), path)


def _read_history(path: Path) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    missing = [
        pl.lit(None, dtype=dtype).alias(column)
        for column, dtype in OBSERVABILITY_HISTORY_SCHEMA.items()
        if column not in frame.columns
    ]
    return (
        frame.with_columns(*missing)
        .select(OBSERVABILITY_HISTORY_SCHEMA.names())
        .cast(OBSERVABILITY_HISTORY_SCHEMA, strict=False)
    )


def load_observability_history(artifacts_dir: Path) -> pl.DataFrame:
    """Trajectory rows, deduped keep-last per (issue, method, slice)."""
    path = artifacts_dir / "observability" / "history.parquet"
    if not path.exists():
        return pl.DataFrame(schema=OBSERVABILITY_HISTORY_SCHEMA)
    return (
        _read_history(path)
        .sort("captured_at")
        .unique(subset=["issue_time", "method_id", "product", "variable"], keep="last")
        .sort("captured_at")
    )


def load_observability_states(artifacts_dir: Path) -> tuple[ObservabilitySnapshot, ...]:
    """Latest snapshot per (product, variable, method); unreadable slots skipped."""
    store = ArtifactStore(artifacts_dir / "observability")
    try:
        latest = store.read_latest()
    except (OSError, json.JSONDecodeError):
        return ()
    snapshots: list[ObservabilitySnapshot] = []
    for _key, pointer in sorted(latest.items()):
        try:
            identity = {
                "fingerprint": str(pointer["fingerprint"]),
                "method_id": str(pointer["method_id"]),
                "product": str(pointer["product"]),
                "variable": str(pointer["variable"]),
            }
            state = store.load_state(**identity)
            manifest = store.load_manifest(**identity)
        except (ArtifactError, KeyError, OSError, TypeError, ValueError):
            continue
        issue = manifest.get("issue_time")
        snapshots.append(
            ObservabilitySnapshot(
                method_id=identity["method_id"],
                product=identity["product"],
                variable=identity["variable"],
                dataset_fingerprint=identity["fingerprint"],
                created_at=str(manifest.get("created_at", "")),
                issue_time=str(issue) if issue is not None else None,
                state=state,
            )
        )
    return tuple(snapshots)
