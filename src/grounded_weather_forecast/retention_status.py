"""Recorded protection checks and pruning attempts, without live archive reads."""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal

import polars as pl

from grounded_weather_forecast.run_health import assess_attempts, aware_moment
from grounded_weather_forecast.status import EXIT_CONTENTION, ComponentDiagnostic
from grounded_weather_forecast.verification_text import normalize_detail, public_detail

VERIFICATION_SAMPLE_LIMIT = 20


@dataclass(frozen=True, slots=True)
class RetentionVerification:
    state: Literal["clear", "suspended", "not_recorded"] = "not_recorded"
    recorded_at: datetime | None = None
    details: tuple[str, ...] = ()
    paths: tuple[str | None, ...] = ()
    check_id: str | None = None
    audit_path: str | None = None
    issue_count: int | None = None
    audit_error: str | None = None
    categories: tuple[str, ...] = ()
    dry_run: bool | None = None

    @property
    def total(self) -> int:
        return self.issue_count if self.issue_count is not None else len(self.details)

    @property
    def omitted(self) -> int:
        return max(0, self.total - min(len(self.details), VERIFICATION_SAMPLE_LIMIT))

    def diagnostics(self) -> tuple[ComponentDiagnostic, ...]:
        if self.state == "not_recorded":
            return ()
        fallback = (
            "published-score protection verified"
            if self.state == "clear"
            else "published-score deletion suspended: protection could not be verified"
        )
        common = ComponentDiagnostic(
            stage="retention_verification",
            exit_code=0,
            failure_kind=f"retention_{self.state}",
            recorded_at=self.recorded_at.isoformat() if self.recorded_at else None,
            check_id=self.check_id,
            dry_run=self.dry_run,
        )
        summary = (
            (
                replace(
                    common,
                    detail=fallback,
                    audit_path=self.audit_path,
                    issue_count=self.total,
                    omitted_count=self.omitted,
                    audit_error=self.audit_error,
                ),
            )
            if self.check_id is not None or self.omitted
            else ()
        )
        samples = self.details[:VERIFICATION_SAMPLE_LIMIT]
        if not summary and not samples:
            samples = (fallback,)
        return summary + tuple(
            replace(
                common,
                detail=normalize_detail(
                    detail, self.paths[index] if index < len(self.paths) else None
                ),
                path=self.paths[index] if index < len(self.paths) else None,
            )
            for index, detail in enumerate(samples)
        )

    def frame(
        self,
        *,
        path_label: Callable[[str], str] | None = None,
        audit_path_label: Callable[[str], str] | None = None,
    ) -> pl.DataFrame:
        label = path_label or str
        details = [
            public_detail(
                detail, self.paths[index] if index < len(self.paths) else None
            )
            for index, detail in enumerate(self.details[:VERIFICATION_SAMPLE_LIMIT])
        ]
        return pl.DataFrame(
            {
                "verification": [self.state],
                "recorded_at": [
                    self.recorded_at.isoformat() if self.recorded_at else None
                ],
                "detail": ["; ".join(details) or None],
                "affected_paths": [
                    "; ".join(
                        label(path)
                        for path in self.paths[:VERIFICATION_SAMPLE_LIMIT]
                        if path
                    )
                    or None
                ],
                "issue_count": [self.total],
                "omitted_count": [self.omitted],
                "audit_path": [
                    (audit_path_label or str)(self.audit_path)
                    if self.audit_path
                    else None
                ],
                "audit_error": [
                    public_detail(self.audit_error) if self.audit_error else None
                ],
                "dry_run": pl.Series([self.dry_run], dtype=pl.Boolean),
            }
        )


type AttemptState = Literal[
    "not_recorded", "completed", "failed", "unverified", "contended", "unknown"
]


@dataclass(frozen=True, slots=True)
class RetentionStatus:
    verification: RetentionVerification = field(default_factory=RetentionVerification)
    attempt_at: datetime | None = None
    attempt_state: AttemptState = "not_recorded"
    outstanding_failure: bool = False
    as_of: datetime | None = None
    attempt_dry_run: bool | None = None

    @property
    def severity(self) -> Literal["ok", "amber", "info"]:
        if (
            self.verification.state == "suspended"
            or self.outstanding_failure
            or self.attempt_state in {"failed", "unknown"}
        ):
            return "amber"
        return (
            "ok"
            if self.verification.state == "clear" and self.attempt_state == "completed"
            else "info"
        )

    @property
    def check_age_minutes(self) -> float | None:
        if self.as_of is None or self.verification.recorded_at is None:
            return None
        return (self.as_of - self.verification.recorded_at).total_seconds() / 60

    def frame(
        self,
        *,
        path_label: Callable[[str], str] | None = None,
        audit_path_label: Callable[[str], str] | None = None,
    ) -> pl.DataFrame:
        return self.verification.frame(
            path_label=path_label, audit_path_label=audit_path_label
        ).with_columns(
            pl.lit(self.attempt_at.isoformat() if self.attempt_at else None).alias(
                "latest_attempt_at"
            ),
            pl.lit(self.attempt_state).alias("latest_attempt"),
            pl.lit(self.check_age_minutes, dtype=pl.Float64).alias("check_age_minutes"),
            pl.lit(self.outstanding_failure).alias("outstanding_failure"),
            pl.lit(self.attempt_dry_run, dtype=pl.Boolean).alias(
                "latest_attempt_dry_run"
            ),
        )


def _argument_mode(row: Mapping[str, Any]) -> bool | None:
    try:
        arguments = json.loads(row.get("args_json") or "null")
    except (ValueError, TypeError):
        return None
    value = arguments.get("dry_run") if isinstance(arguments, dict) else None
    return value if type(value) is bool else None


def _check_mode(items: list[dict[str, Any]], *, fallback: bool | None) -> bool | None:
    values = [item["dry_run"] for item in items if item.get("dry_run") is not None]
    if not values:
        return fallback
    return (
        values[0]
        if all(type(value) is bool and value == values[0] for value in values)
        else None
    )


def _check_batch(
    items: list[dict[str, Any]],
    ended: datetime,
    *,
    mode: bool | None,
) -> RetentionVerification:
    kinds = {item.get("failure_kind") for item in items}
    clocks = [
        aware_moment(item["recorded_at"])
        if item.get("recorded_at") is not None
        else ended
        for item in items
    ]
    if (
        not kinds
        or not kinds <= {"retention_clear", "retention_suspended"}
        or any(clock is None or clock > ended for clock in clocks)
        or len(set(clocks)) != 1
    ):
        return RetentionVerification()
    summary = next(
        (
            item
            for item in items
            if type(item.get("issue_count")) is int and item["issue_count"] >= 0
        ),
        {},
    )
    samples = [item for item in items if "issue_count" not in item][
        :VERIFICATION_SAMPLE_LIMIT
    ]

    def text(key: str) -> str | None:
        value = summary.get(key)
        return value if isinstance(value, str) else None

    return RetentionVerification(
        "suspended" if "retention_suspended" in kinds else "clear",
        clocks[0],
        tuple(str(item.get("detail") or "")[:400] for item in samples),
        tuple(
            item.get("path") if isinstance(item.get("path"), str) else None
            for item in samples
        ),
        items[0].get("check_id"),
        text("audit_path"),
        summary.get("issue_count", 0 if kinds == {"retention_clear"} else len(items)),
        text("audit_error"),
        dry_run=_check_mode(items, fallback=mode),
    )


def _recorded_checks(
    row: Mapping[str, Any], ended: datetime
) -> tuple[RetentionVerification, ...]:
    try:
        raw = json.loads(row.get("failure_details_json") or "null")
        if not isinstance(raw, list):
            return ()
        groups: dict[tuple[str, object], list[dict[str, Any]]] = {}
        for item in raw:
            if (
                not isinstance(item, dict)
                or item.get("stage") != "retention_verification"
                or item.get("exit_code") != 0
            ):
                continue
            identity = (
                ("id", item["check_id"])
                if isinstance(item.get("check_id"), str)
                else ("clock", item.get("recorded_at"))
            )
            groups.setdefault(identity, []).append(item)
        checks = [
            _check_batch(items, ended, mode=_argument_mode(row))
            for items in groups.values()
        ]
        return tuple(
            sorted(
                (check for check in checks if check.recorded_at is not None),
                key=lambda check: check.recorded_at or ended,
            )
        )
    except (ValueError, TypeError):
        return ()


def retention_status(runs: pl.DataFrame, *, as_of: datetime) -> RetentionStatus:
    assessment = assess_attempts(runs, "prune-scores", as_of)
    verification = RetentionVerification()
    state: AttemptState = "not_recorded"
    latest = None
    outstanding = False
    mode = None
    for attempt in assessment.attempts:
        row, latest = attempt.row, attempt.ended_at
        checks = _recorded_checks(row, latest)
        mode = checks[-1].dry_run if checks else _argument_mode(row)
        if checks:
            check = checks[-1]
            if check.recorded_at is not None and (
                verification.recorded_at is None
                or check.recorded_at >= verification.recorded_at
            ):
                verification = check
        if row.get("exit_code") == EXIT_CONTENTION and not row.get("error"):
            state = "contended"
        elif row.get("error") or row.get("exit_code") != 0:
            state, outstanding = "failed", True
        elif checks:
            state = "completed"
            if checks[-1].state == "clear" and mode is False:
                outstanding = False
        else:
            state = "unverified"
    if assessment.unresolved:
        state = "unknown"
        mode = None
        clocks = assessment.invalid_ended
        latest = None if None in clocks else max(clocks) if clocks else latest
    return RetentionStatus(verification, latest, state, outstanding, as_of, mode)


def latest_retention_verification(
    runs: pl.DataFrame, *, as_of: datetime
) -> RetentionVerification:
    """Compatibility adapter: an unverified attempt cannot erase a recorded check."""
    return retention_status(runs, as_of=as_of).verification
