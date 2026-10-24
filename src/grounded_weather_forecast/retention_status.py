"""Recorded protection checks and pruning attempts, without live archive reads."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import polars as pl

from grounded_weather_forecast.run_health import (
    attempt_order,
    aware_moment,
    completed_attempts,
)
from grounded_weather_forecast.status import ComponentDiagnostic


@dataclass(frozen=True, slots=True)
class RetentionVerification:
    state: Literal["clear", "suspended", "not_recorded"] = "not_recorded"
    recorded_at: datetime | None = None
    details: tuple[str, ...] = ()
    paths: tuple[str | None, ...] = ()

    def diagnostics(self) -> tuple[ComponentDiagnostic, ...]:
        if self.state == "not_recorded":
            return ()
        fallback = (
            "published-score protection verified"
            if self.state == "clear"
            else "published-score deletion suspended: protection could not be verified"
        )
        return tuple(
            ComponentDiagnostic(
                "retention_verification",
                0,
                f"retention_{self.state}",
                detail.removeprefix(f"{self.paths[index]}: ")
                if index < len(self.paths) and self.paths[index]
                else detail,
                self.paths[index] if index < len(self.paths) else None,
                self.recorded_at.isoformat() if self.recorded_at else None,
            )
            for index, detail in enumerate(self.details or (fallback,))
        )

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "verification": [self.state],
                "recorded_at": [
                    self.recorded_at.isoformat() if self.recorded_at else None
                ],
                "detail": ["; ".join(self.details) or None],
                "affected_paths": [
                    "; ".join(path for path in self.paths if path) or None
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class RetentionStatus:
    verification: RetentionVerification = field(default_factory=RetentionVerification)
    attempt_at: datetime | None = None
    attempt_state: Literal["not_recorded", "completed", "failed", "unverified"] = (
        "not_recorded"
    )

    @property
    def severity(self) -> Literal["ok", "amber", "info"]:
        if self.verification.state == "suspended" or self.attempt_state == "failed":
            return "amber"
        return (
            "ok"
            if self.verification.state == "clear" and self.attempt_state == "completed"
            else "info"
        )

    def frame(self) -> pl.DataFrame:
        return self.verification.frame().with_columns(
            pl.lit(self.attempt_at.isoformat() if self.attempt_at else None).alias(
                "latest_attempt_at"
            ),
            pl.lit(self.attempt_state).alias("latest_attempt"),
        )


def _recorded_check(row: dict[str, Any], ended: datetime) -> RetentionVerification:
    try:
        raw = json.loads(row.get("failure_details_json") or "null")
        if not isinstance(raw, list):
            return RetentionVerification()
        items = [
            item
            for item in raw
            if isinstance(item, dict)
            and item.get("stage") == "retention_verification"
            and item.get("exit_code") == 0
        ]
        kinds = {item.get("failure_kind") for item in items}
        if not kinds or not kinds <= {"retention_clear", "retention_suspended"}:
            return RetentionVerification()
        clocks = [
            aware_moment(item["recorded_at"])
            if item.get("recorded_at") is not None
            else ended
            for item in items
        ]
        if (
            any(clock is None or clock > ended for clock in clocks)
            or len(set(clocks)) != 1
        ):
            return RetentionVerification()
        return RetentionVerification(
            "suspended" if "retention_suspended" in kinds else "clear",
            clocks[0],
            tuple(str(item.get("detail") or "")[:400] for item in items),
            tuple(
                item.get("path") if isinstance(item.get("path"), str) else None
                for item in items
            ),
        )
    except (ValueError, TypeError):
        return RetentionVerification()


def retention_status(runs: pl.DataFrame, *, as_of: datetime) -> RetentionStatus:
    attempts = sorted(
        completed_attempts(runs, "prune-scores", as_of), key=attempt_order
    )
    verification = RetentionVerification()
    state: Literal["not_recorded", "completed", "failed", "unverified"] = "not_recorded"
    latest = None
    for ended, row in attempts:
        check = _recorded_check(dict(row), ended)
        if check.state != "not_recorded":
            verification = check
        state = (
            "failed"
            if row.get("error") or row.get("exit_code") != 0
            else "completed"
            if check.state != "not_recorded"
            else "unverified"
        )
        latest = ended
    return RetentionStatus(verification, latest, state)


def latest_retention_verification(
    runs: pl.DataFrame, *, as_of: datetime
) -> RetentionVerification:
    """Compatibility adapter: an unverified attempt cannot erase a recorded check."""
    return retention_status(runs, as_of=as_of).verification
