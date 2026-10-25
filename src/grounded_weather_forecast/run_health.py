"""Pure eligibility, chronology, and uncertainty rules for CLI run evidence."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl

from grounded_weather_forecast.clocks import aware_utc


def aware_moment(value: object) -> datetime | None:
    """Compatibility adapter for the shared, strictly aware clock validator."""
    return aware_utc(value)


def eligible_run(row: Mapping[str, Any], command: str | None = None) -> bool:
    return (command is None or row.get("command") == command) and row.get(
        "record_kind"
    ) in (None, "invocation", "stage")


def completed_at(row: Mapping[str, Any], now: datetime) -> datetime | None:
    """Validate relevant evidence; valid incomplete/future runs are excluded."""
    ended = aware_moment(row.get("ended_at"))
    if ended is not None and ended > now:
        return None
    if row.get("exit_code") is not None and type(row["exit_code"]) is not int:
        raise ValueError("invalid_run_exit_code")
    if row.get("failure_kind") is not None and not isinstance(row["failure_kind"], str):
        raise ValueError("invalid_run_failure_kind")
    if row.get("ended_at") is None:
        return None
    if ended is None:
        raise ValueError("invalid_run_ended_at")
    if (started := aware_moment(row.get("started_at"))) is None or started > ended:
        raise ValueError("invalid_run_started_at")
    return ended


@dataclass(frozen=True, slots=True)
class RunAttempt:
    ended_at: datetime
    row: Mapping[str, Any]
    position: int

    @property
    def order(self) -> tuple[datetime, datetime, int]:
        return (
            self.ended_at,
            aware_moment(self.row["started_at"]) or self.ended_at,
            self.position,
        )


@dataclass(frozen=True, slots=True)
class AttemptAssessment:
    attempts: tuple[RunAttempt, ...] = ()
    invalid_ended: tuple[datetime | None, ...] = ()

    @property
    def latest(self) -> RunAttempt | None:
        return self.attempts[-1] if self.attempts else None

    @property
    def unresolved(self) -> bool:
        latest = self.latest
        return any(
            clock is None or latest is None or clock >= latest.ended_at
            for clock in self.invalid_ended
        )

    @property
    def superseded(self) -> bool:
        return bool(self.invalid_ended) and not self.unresolved


def assess_attempts(
    runs: pl.DataFrame, command: str, now: datetime
) -> AttemptAssessment:
    """Keep damaged relevant rows visible; exact valid ties use ledger order."""
    cutoff = aware_moment(now)
    if cutoff is None:
        raise ValueError("run cutoff must be timezone-aware")
    if runs.is_empty():
        return AttemptAssessment()
    if not {"command", "started_at", "ended_at", "exit_code"}.issubset(runs.columns):
        return AttemptAssessment(invalid_ended=(None,))
    attempts: list[RunAttempt] = []
    invalid: list[datetime | None] = []
    for position, row in enumerate(
        runs.filter(pl.col("command") == command).iter_rows(named=True)
    ):
        if not eligible_run(row, command):
            continue
        try:
            if (ended := completed_at(row, cutoff)) is not None:
                attempts.append(RunAttempt(ended, row, position))
        except (ValueError, TypeError, OverflowError):
            ended = aware_moment(row.get("ended_at"))
            if ended is None or ended <= cutoff:
                invalid.append(ended)
    return AttemptAssessment(
        tuple(sorted(attempts, key=lambda attempt: attempt.order)), tuple(invalid)
    )


def latest_attempt(
    runs: pl.DataFrame, command: str, now: datetime
) -> Mapping[str, Any] | None:
    assessment = assess_attempts(runs, command, now)
    return (
        assessment.latest.row
        if assessment.latest is not None and not assessment.unresolved
        else None
    )
