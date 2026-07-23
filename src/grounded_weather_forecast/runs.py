"""Rolling ledger of every CLI invocation, for the operator dashboard.

Bounded by age (90 days) and row count (50,000) so the 10-minute `predict`
cadence cannot grow it without limit.
"""

import hashlib
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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


def _as_utc(moment: datetime) -> datetime:
    """A timezone-aware UTC instant.

    A naive ``RunRecord`` timestamp used to lose the whole row: ``_to_frame``
    silently reinterpreted it as UTC, then ``prune_runs`` raised SchemaError
    comparing a tz-aware column against a tz-naive literal, and ``append_run``
    swallowed that — so the ledger was never written and nothing said so.
    """
    return (
        moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)
    )


def prune_runs(frame: pl.DataFrame, *, now: datetime) -> pl.DataFrame:
    """Bound the ledger by age, then by row count as a burst backstop.

    A null ``started_at`` has no age to judge, so it is kept and left to the
    row-count backstop rather than silently dropped — ``load_runs`` promises
    that older schemas load null-filled, and pruning must honour that.
    """
    if frame.is_empty():
        return frame
    horizon = _as_utc(now) - timedelta(days=_RETENTION_DAYS)
    return frame.filter(
        pl.col("started_at").is_null() | (pl.col("started_at") >= horizon)
    ).tail(_MAX_ROWS)


def append_run(record: RunRecord, path: Path, *, now: datetime | None = None) -> None:
    """Append one ledger row; telemetry failures never reach the command."""
    try:
        fresh = _to_frame(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        with locked_path(path, timeout=_LOCK_TIMEOUT_SECONDS):
            combined = pl.concat([load_runs(path), fresh]) if path.exists() else fresh
            # The whole file is rewritten under the lock anyway, so pruning is
            # free here — and without it the ledger grows without bound at the
            # documented 10-minute `predict` cadence. The horizon comes from
            # the wall clock, never from `record`: one row carrying a skewed
            # future timestamp would otherwise set a horizon in the future and
            # delete every genuinely recent row along with it.
            atomic_write_parquet(
                prune_runs(combined, now=now or datetime.now(tz=UTC)),
                path,
            )
    except (OSError, ValueError, Timeout, pl.exceptions.PolarsError):
        return


def read_runs(path: Path) -> pl.DataFrame:
    """Read and normalize the ledger, raising when an existing file is unusable."""
    if not path.exists():
        return pl.DataFrame(schema=RUNS_SCHEMA)
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


def load_runs(path: Path) -> pl.DataFrame:
    """Read the ledger; absent files and older schemas load as null-filled.

    A file that exists but cannot be parsed also reads as empty, which means
    the next ``append_run`` rewrites it with only the fresh row. That is the
    intended trade for telemetry — a corrupt ledger must not fail a command —
    but it does discard whatever the file held.
    """
    try:
        return read_runs(path)
    except (OSError, pl.exceptions.PolarsError):
        return pl.DataFrame(schema=RUNS_SCHEMA)


def _to_frame(record: RunRecord) -> pl.DataFrame:
    started_at = _as_utc(record.started_at)
    ended_at = _as_utc(record.ended_at)
    duration_ms = int((ended_at - started_at).total_seconds() * 1000)
    row = {
        "run_id": record.run_id,
        "command": record.command,
        "args_json": record.args_json,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_ms": duration_ms,
        "exit_code": record.exit_code,
        "error": record.error,
        "dataset_fingerprint": record.dataset_fingerprint,
        "config_fingerprint": record.config_fingerprint,
        "code_version": record.code_version,
    }
    return pl.DataFrame([row], schema=RUNS_SCHEMA)
