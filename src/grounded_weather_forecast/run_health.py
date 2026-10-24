"""Pure eligibility and timestamp rules for completed CLI run evidence."""

from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from typing import Any

import polars as pl


def aware_moment(value: object) -> datetime | None:
    try:
        moment = (
            value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        )
        return moment.astimezone(UTC) if moment.tzinfo is not None else None
    except (ValueError, TypeError, OverflowError):
        return None


def eligible_run(row: Mapping[str, Any], command: str | None = None) -> bool:
    return (command is None or row.get("command") == command) and row.get(
        "record_kind"
    ) in (None, "invocation", "stage")


def completed_at(row: Mapping[str, Any], now: datetime) -> datetime | None:
    """Validate a relevant row; incomplete/future runs are not completed evidence."""
    if row.get("exit_code") is not None and type(row["exit_code"]) is not int:
        raise ValueError("invalid_run_exit_code")
    if row.get("failure_kind") is not None and not isinstance(row["failure_kind"], str):
        raise ValueError("invalid_run_failure_kind")
    if row.get("ended_at") is None:
        return None
    if (ended := aware_moment(row["ended_at"])) is None:
        raise ValueError("invalid_run_ended_at")
    if ended > now:
        return None
    if (started := aware_moment(row.get("started_at"))) is None or started > ended:
        raise ValueError("invalid_run_started_at")
    return ended


def completed_attempts(
    runs: pl.DataFrame, command: str, now: datetime
) -> Iterator[tuple[datetime, Mapping[str, Any]]]:
    required = {"command", "started_at", "ended_at", "exit_code"}
    if not required.issubset(runs.columns):
        return
    for row in runs.filter(pl.col("command") == command).iter_rows(named=True):
        if not eligible_run(row, command):
            continue
        try:
            if (ended := completed_at(row, now)) is not None:
                yield ended, row
        except (ValueError, TypeError, OverflowError):
            continue


def attempt_order(
    item: tuple[datetime, Mapping[str, Any]],
) -> tuple[datetime, datetime]:
    ended, row = item
    return ended, aware_moment(row["started_at"]) or ended


def latest_attempt(
    runs: pl.DataFrame, command: str, now: datetime
) -> Mapping[str, Any] | None:
    result = max(
        completed_attempts(runs, command, now), key=attempt_order, default=None
    )
    return result[1] if result is not None else None
