"""Append-only ledger of every CLI invocation, for the operator dashboard."""

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from filelock import Timeout

from grounded_weather_forecast.config import Config
from grounded_weather_forecast.storage import atomic_write_parquet, locked_path

_LOCK_TIMEOUT_SECONDS = 5.0
_RETENTION_DAYS = 90
_MAX_ROWS = 50_000

RUNS_SCHEMA = pl.Schema(
    {
        "run_id": pl.String(),
        "command": pl.String(),
        "args_json": pl.String(),
        "started_at": pl.Datetime("us", "UTC"),
        "ended_at": pl.Datetime("us", "UTC"),
        "duration_ms": pl.Int64(),
        "exit_code": pl.Int64(),
        "error": pl.String(),
        "dataset_fingerprint": pl.String(),
        "config_fingerprint": pl.String(),
        "code_version": pl.String(),
    }
)


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One CLI invocation: what ran, when, and against which artifacts."""

    run_id: str
    command: str
    args_json: str
    started_at: datetime
    ended_at: datetime
    exit_code: int | None
    error: str | None
    dataset_fingerprint: str
    config_fingerprint: str
    code_version: str


def run_id_for(command: str, started_at: datetime) -> str:
    seed = f"{command}|{started_at.isoformat()}|{os.getpid()}"
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def runs_path(config: Config) -> Path:
    return config.dataset.dir / "runs.parquet"


def prune_runs(frame: pl.DataFrame, *, now: datetime) -> pl.DataFrame:
    """Bound the ledger by age, then by row count as a burst backstop."""
    if frame.is_empty():
        return frame
    horizon = now - timedelta(days=_RETENTION_DAYS)
    return frame.filter(pl.col("started_at") >= horizon).tail(_MAX_ROWS)


def append_run(record: RunRecord, path: Path) -> None:
    """Append one ledger row; telemetry failures never reach the command."""
    try:
        fresh = _to_frame(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        with locked_path(path, timeout=_LOCK_TIMEOUT_SECONDS):
            combined = pl.concat([load_runs(path), fresh]) if path.exists() else fresh
            # The whole file is rewritten under the lock anyway, so pruning is
            # free here — and without it the ledger grows without bound at the
            # documented 10-minute `predict` cadence.
            atomic_write_parquet(prune_runs(combined, now=record.ended_at), path)
    except (OSError, ValueError, Timeout, pl.exceptions.PolarsError):
        return


def load_runs(path: Path) -> pl.DataFrame:
    """Read the ledger; absent files and older schemas load as null-filled."""
    if not path.exists():
        return pl.DataFrame(schema=RUNS_SCHEMA)
    try:
        frame = pl.read_parquet(path)
        missing = [
            pl.lit(None, dtype=dtype).alias(column)
            for column, dtype in RUNS_SCHEMA.items()
            if column not in frame.columns
        ]
        return (
            frame.with_columns(*missing)
            .select(RUNS_SCHEMA.names())
            .cast(RUNS_SCHEMA, strict=False)
        )
    except (OSError, pl.exceptions.PolarsError):
        return pl.DataFrame(schema=RUNS_SCHEMA)


def _to_frame(record: RunRecord) -> pl.DataFrame:
    duration_ms = int((record.ended_at - record.started_at).total_seconds() * 1000)
    row = {
        "run_id": record.run_id,
        "command": record.command,
        "args_json": record.args_json,
        "started_at": record.started_at,
        "ended_at": record.ended_at,
        "duration_ms": duration_ms,
        "exit_code": record.exit_code,
        "error": record.error,
        "dataset_fingerprint": record.dataset_fingerprint,
        "config_fingerprint": record.config_fingerprint,
        "code_version": record.code_version,
    }
    return pl.DataFrame([row], schema=RUNS_SCHEMA)
