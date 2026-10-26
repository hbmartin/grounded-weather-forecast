"""Recorded protection checks and pruning attempts, without live archive reads."""

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from itertools import islice
from typing import Any, Literal

import polars as pl

from grounded_weather_forecast.run_health import assess_attempts, aware_moment
from grounded_weather_forecast.status import EXIT_CONTENTION, ComponentDiagnostic
from grounded_weather_forecast.verification_text import (
    VerificationIssue,
    category_code,
    public_explanation,
)

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
    reason_codes: tuple[str | None, ...] = ()
    audit_candidate_path: str | None = None
    audit_reason_code: str | None = None
    audit_cause_code: str | None = None
    _details_normalized: bool = field(default=False, repr=False, compare=False)

    @property
    def total(self) -> int:
        return self.issue_count if self.issue_count is not None else len(self.details)

    @property
    def omitted(self) -> int:
        return max(0, self.total - min(len(self.details), VERIFICATION_SAMPLE_LIMIT))

    @property
    def audit_state(self) -> str:
        if self.audit_candidate_path:
            return "unconfirmed"
        if self.audit_error:
            return "unavailable"
        if self.audit_path:
            return "recorded"
        return "not_needed" if self.state == "clear" else "not_recorded"

    def iter_issues(self) -> Iterator[VerificationIssue]:
        """Normalize legacy parallel metadata once; full audits may stream all rows."""
        for index, detail in enumerate(self.details):
            yield VerificationIssue.from_legacy(
                detail,
                path=self.paths[index] if index < len(self.paths) else None,
                category=self.categories[index]
                if index < len(self.categories)
                else None,
                reason_code=self.reason_codes[index]
                if index < len(self.reason_codes)
                else None,
                normalize_prefix=not self._details_normalized,
            )

    def sample_issues(self) -> tuple[VerificationIssue, ...]:
        """One bounded projection shared by all diagnostic and public renderers."""
        return tuple(islice(self.iter_issues(), VERIFICATION_SAMPLE_LIMIT))

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
                    audit_candidate_path=self.audit_candidate_path,
                    audit_reason_code=self.audit_reason_code,
                    audit_cause_code=self.audit_cause_code,
                ),
            )
            if self.check_id is not None or self.omitted
            else ()
        )
        samples = self.sample_issues()
        if not summary and not samples:
            return (replace(common, detail=fallback),)
        return summary + tuple(
            replace(
                common,
                detail=issue.detail,
                path=str(issue.path) if issue.path is not None else None,
                category=issue.category,
                reason_code=issue.reason_code,
            )
            for issue in samples
        )

    def frame(
        self,
        *,
        path_label: Callable[[str], str] | None = None,
        audit_path_label: Callable[[str], str] | None = None,
    ) -> pl.DataFrame:
        label = path_label or str
        samples = self.sample_issues()
        details = dict.fromkeys(
            public_explanation(category=issue.category, reason_code=issue.reason_code)
            for issue in samples
        )
        return pl.DataFrame(
            {
                "verification": [self.state],
                "recorded_at": [
                    self.recorded_at.isoformat() if self.recorded_at else None
                ],
                "detail": ["; ".join(details) or None],
                "affected_paths": [
                    "; ".join(
                        label(str(issue.path))
                        for issue in samples
                        if issue.path is not None
                    )
                    or None
                ],
                "issue_count": [self.total],
                "omitted_count": [self.omitted],
                "audit_state": [self.audit_state],
                "audit_path": [
                    (audit_path_label or str)(self.audit_path)
                    if self.audit_path
                    else None
                ],
                "audit_error": [
                    public_explanation(
                        reason_code=self.audit_reason_code,
                        cause_code=self.audit_cause_code,
                        audit=True,
                    )
                    if self.audit_error
                    else None
                ],
                "audit_candidate_path": [
                    (audit_path_label or str)(self.audit_candidate_path)
                    if self.audit_candidate_path
                    else None
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
        categories=tuple(category_code(item.get("category")) for item in samples),
        reason_codes=tuple(
            item.get("reason_code")
            if isinstance(item.get("reason_code"), str)
            else None
            for item in samples
        ),
        audit_candidate_path=text("audit_candidate_path"),
        audit_reason_code=text("audit_reason_code"),
        audit_cause_code=text("audit_cause_code"),
        # Component detail is already raw evidence, separate from its path/category.
        # Never strip an evidence prefix a second time on journal reads.
        _details_normalized=True,
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
