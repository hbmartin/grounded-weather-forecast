"""Recorded protection checks and pruning attempts, without live archive reads."""

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal

import polars as pl

from grounded_weather_forecast.run_health import assess_attempts, aware_moment
from grounded_weather_forecast.status import (
    EXIT_CONTENTION,
    ComponentDiagnostic,
    DiagnosticIdentity,
    assess_progress_identity,
    effective_progress_id,
)
from grounded_weather_forecast.verification_text import (
    IssueSource,
    VerificationIssue,
    canonical_issues,
    optional_code,
    public_explanation,
)

VERIFICATION_SAMPLE_LIMIT = 20
_VERDICT_DETAILS = {
    "retention_clear": "published-score protection verified",
    "retention_suspended": "published-score deletion suspended: protection could not be verified",
}


def _verification_issues(
    issues: Sequence[IssueSource],
    details: tuple[str, ...] | None,
    paths: tuple[str | None, ...] | None,
    categories: tuple[str, ...] | None,
    reasons: tuple[str | None, ...] | None,
    *,
    normalize_prefix: bool = False,
) -> tuple[VerificationIssue, ...]:
    issues = canonical_issues(issues)
    if all(value is None for value in (details, paths, categories, reasons)):
        return issues
    raw = details if details is not None else tuple(issue.detail for issue in issues)
    sources = (
        paths
        if paths is not None
        else tuple(str(i.path) if i.path is not None else None for i in issues)
    )
    kinds = categories if categories is not None else tuple(i.category for i in issues)
    codes = reasons if reasons is not None else tuple(i.reason_code for i in issues)
    return tuple(
        VerificationIssue.from_legacy(
            detail,
            path=sources[index] if index < len(sources) else None,
            category=kinds[index] if index < len(kinds) else None,
            reason_code=codes[index] if index < len(codes) else None,
            normalize_prefix=normalize_prefix and details is not None,
        )
        for index, detail in enumerate(raw)
    )


@dataclass(frozen=True, slots=True, init=False)
class RetentionVerification:
    state: Literal["clear", "suspended", "not_recorded"] = "not_recorded"
    recorded_at: datetime | None = None
    issues: tuple[VerificationIssue, ...] = ()
    check_id: str | None = None
    audit_path: str | None = None
    issue_count: int | None = None
    audit_error: str | None = None
    dry_run: bool | None = None
    audit_candidate_path: str | None = None
    audit_reason_code: str | None = None
    audit_cause_code: str | None = None
    audit_operation: str | None = None
    progress_id: str | None = None
    issue_count_known: bool | None = None

    def __init__(
        self,
        state: Literal["clear", "suspended", "not_recorded"] = "not_recorded",
        recorded_at: datetime | None = None,
        details: tuple[str, ...] | None = None,
        paths: tuple[str | None, ...] | None = None,
        check_id: str | None = None,
        audit_path: str | None = None,
        issue_count: int | None = None,
        audit_error: str | None = None,
        categories: tuple[str, ...] | None = None,
        dry_run: bool | None = None,  # noqa: FBT001 -- preserve positional construction compatibility
        reason_codes: tuple[str | None, ...] | None = None,
        audit_candidate_path: str | None = None,
        audit_reason_code: str | None = None,
        audit_cause_code: str | None = None,
        *,
        issues: Sequence[IssueSource] = (),
        audit_operation: str | None = None,
        progress_id: str | None = None,
        issue_count_known: bool | None = None,
    ) -> None:
        # Constructor and replacement inputs are literal evidence. Prefix stripping
        # belongs exclusively to the explicit legacy adapter below.
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "recorded_at", recorded_at)
        object.__setattr__(
            self,
            "issues",
            _verification_issues(issues, details, paths, categories, reason_codes),
        )
        object.__setattr__(self, "check_id", check_id)
        object.__setattr__(
            self, "progress_id", effective_progress_id(progress_id, check_id)
        )
        object.__setattr__(self, "audit_path", audit_path)
        object.__setattr__(self, "issue_count", issue_count)
        if issue_count_known is not None and type(issue_count_known) is not bool:
            raise ValueError("issue count knowledge must be boolean or null")
        if issue_count_known is False and (
            state != "suspended" or self.issues or issue_count not in (None, 0)
        ):
            raise ValueError("unknown issue count requires a sample-free suspension")
        object.__setattr__(self, "issue_count_known", issue_count_known)
        object.__setattr__(self, "audit_error", audit_error)
        object.__setattr__(self, "dry_run", dry_run)
        object.__setattr__(self, "audit_candidate_path", audit_candidate_path)
        object.__setattr__(self, "audit_reason_code", optional_code(audit_reason_code))
        object.__setattr__(self, "audit_cause_code", optional_code(audit_cause_code))
        object.__setattr__(self, "audit_operation", optional_code(audit_operation))

    @classmethod
    def from_legacy_details(
        cls,
        state: Literal["clear", "suspended", "not_recorded"] = "not_recorded",
        recorded_at: datetime | None = None,
        details: tuple[str, ...] | None = None,
        paths: tuple[str | None, ...] | None = None,
        check_id: str | None = None,
        audit_path: str | None = None,
        issue_count: int | None = None,
        audit_error: str | None = None,
        categories: tuple[str, ...] | None = None,
        dry_run: bool | None = None,  # noqa: FBT001 -- constructor-compatible arguments
        reason_codes: tuple[str | None, ...] | None = None,
        audit_candidate_path: str | None = None,
        audit_reason_code: str | None = None,
        audit_cause_code: str | None = None,
        *,
        issues: Sequence[IssueSource] = (),
        audit_operation: str | None = None,
        progress_id: str | None = None,
        issue_count_known: bool | None = None,
    ) -> "RetentionVerification":
        """Ingest labelled legacy strings, removing exact recorded prefixes once."""
        normalized = _verification_issues(
            issues,
            details,
            paths,
            categories,
            reason_codes,
            normalize_prefix=True,
        )
        return cls(
            state,
            recorded_at,
            check_id=check_id,
            audit_path=audit_path,
            issue_count=issue_count,
            audit_error=audit_error,
            dry_run=dry_run,
            audit_candidate_path=audit_candidate_path,
            audit_reason_code=audit_reason_code,
            audit_cause_code=audit_cause_code,
            issues=normalized,
            audit_operation=audit_operation,
            progress_id=progress_id,
            issue_count_known=issue_count_known,
        )

    @property
    def details(self) -> tuple[str, ...]:
        """Canonical raw evidence; formatting is an explicit separate projection."""
        return tuple(issue.detail for issue in self.issues)

    @property
    def formatted_details(self) -> tuple[str, ...]:
        return tuple(issue.diagnostic() for issue in self.issues)

    @property
    def paths(self) -> tuple[str | None, ...]:
        return tuple(
            str(issue.path) if issue.path is not None else None for issue in self.issues
        )

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(issue.category for issue in self.issues)

    @property
    def reason_codes(self) -> tuple[str | None, ...]:
        return tuple(issue.reason_code for issue in self.issues)

    @property
    def total(self) -> int:
        return self.issue_count if self.issue_count is not None else len(self.issues)

    @property
    def omitted(self) -> int:
        return max(0, self.total - min(len(self.issues), VERIFICATION_SAMPLE_LIMIT))

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
        """Stream canonical evidence without interpreting its raw prose again."""
        return iter(self.issues)

    def sample_issues(self) -> tuple[VerificationIssue, ...]:
        """One bounded projection shared by all diagnostic and public renderers."""
        return self.issues[:VERIFICATION_SAMPLE_LIMIT]

    def audit_explanation(self) -> str:
        return public_explanation(
            reason_code=self.audit_reason_code,
            cause_code=self.audit_cause_code,
            audit=True,
            audit_state=self.audit_state,
            operation=self.audit_operation,
        )

    def diagnostics(self) -> tuple[ComponentDiagnostic, ...]:
        if self.state == "not_recorded":
            return ()
        fallback = _VERDICT_DETAILS[f"retention_{self.state}"]
        common = ComponentDiagnostic(
            stage="retention_verification",
            exit_code=0,
            failure_kind=f"retention_{self.state}",
            recorded_at=self.recorded_at.isoformat() if self.recorded_at else None,
            check_id=self.check_id,
            progress_id=self.progress_id,
            dry_run=self.dry_run,
            issue_count_known=self.issue_count_known,
        )
        summary = (
            (
                replace(
                    common,
                    detail=fallback,
                    audit_path=self.audit_path,
                    issue_count=self.total,
                    omitted_count=None
                    if self.issue_count_known is False
                    else self.omitted,
                    audit_error=self.audit_error,
                    audit_candidate_path=self.audit_candidate_path,
                    audit_reason_code=self.audit_reason_code,
                    audit_cause_code=self.audit_cause_code,
                    operation=self.audit_operation,
                ),
            )
            if self.progress_id is not None or self.omitted
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
                "issue_count": pl.Series(
                    [None if self.issue_count_known is False else self.total],
                    dtype=pl.Int64,
                ),
                "omitted_count": pl.Series(
                    [None if self.issue_count_known is False else self.omitted],
                    dtype=pl.Int64,
                ),
                "audit_state": [self.audit_state],
                "audit_path": [
                    (audit_path_label or str)(self.audit_path)
                    if self.audit_path
                    else None
                ],
                "audit_error": [self.audit_explanation() if self.audit_error else None],
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
    verification_unknown: bool = False

    @property
    def verification_warning(self) -> str | None:
        if not self.verification_unknown:
            return None
        return (
            "Recorded verification evidence is damaged or inconsistent; the last valid check, if any, is retained. "
            "A clear check recorded strictly after the damaged run ends is required to resolve this uncertainty."
        )

    @property
    def severity(self) -> Literal["ok", "amber", "info"]:
        if (
            self.verification.state == "suspended"
            or self.verification_unknown
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
        frame = self.verification.frame(
            path_label=path_label, audit_path_label=audit_path_label
        )
        if self.verification_warning is not None:
            frame = frame.with_columns(
                pl.concat_str(
                    pl.lit(self.verification_warning),
                    pl.col("detail"),
                    separator="; ",
                    ignore_nulls=True,
                ).alias("detail")
            )
        return frame.with_columns(
            pl.lit(self.attempt_at.isoformat() if self.attempt_at else None).alias(
                "latest_attempt_at"
            ),
            pl.lit(self.attempt_state).alias("latest_attempt"),
            pl.lit(self.check_age_minutes, dtype=pl.Float64).alias("check_age_minutes"),
            pl.lit(self.outstanding_failure).alias("outstanding_failure"),
            pl.lit(self.verification_unknown).alias("verification_unknown"),
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


def _check_identity(items: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    owners = {
        effective_progress_id(item.get("progress_id"), item.get("check_id"))
        for item in items
    }
    check_ids = {item["check_id"] for item in items if item.get("check_id") is not None}
    if len(owners) != 1 or len(check_ids) > 1:
        raise ValueError("conflicting verification identities")
    return owners.pop(), next(iter(check_ids), None)


def _has_issue_metadata(item: Mapping[str, Any]) -> bool:
    # Malformed additive codes still describe a sample; the legacy issue adapter
    # normalizes them safely instead of losing the recorded protection verdict.
    # An explicitly empty category/path on a suspension is recorded sample
    # metadata, normalized to the existing unknown-category/null-path defaults.
    # Verdict-only legacy records have no such sample metadata.
    if item.get("failure_kind") == "retention_suspended" and any(
        item.get(key) == "" for key in ("category", "path")
    ):
        return True
    return any(
        item.get(key) is not None and item[key] != ""
        for key in ("category", "path", "reason_code")
    )


def _check_parts(
    items: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for item in items:
        metadata = _has_issue_metadata(item)
        if "issue_count" in item:
            if (
                type(item["issue_count"]) is not int
                or item["issue_count"] < 0
                or metadata
            ):
                raise ValueError("invalid verification summary")
            summaries.append(item)
        else:
            rows.append(item)
    if len(summaries) > 1:
        raise ValueError("conflicting verification summaries")
    summary = summaries[0] if summaries else {}
    samples = _legacy_samples(rows, summary)
    total = summary.get("issue_count", len(samples))
    if total < len(samples):
        raise ValueError("verification total is smaller than its samples")
    omitted = summary.get("omitted_count")
    if omitted is not None and (
        type(omitted) is not int
        or omitted != max(0, total - min(len(samples), VERIFICATION_SAMPLE_LIMIT))
    ):
        raise ValueError("invalid verification omitted count")
    return summary, samples[:VERIFICATION_SAMPLE_LIMIT], total


def _legacy_samples(
    rows: list[dict[str, Any]], summary: dict[str, Any]
) -> list[dict[str, Any]]:
    """Only exact wire verdict markers may stand in for absent legacy samples."""
    verdicts: list[dict[str, Any]] = []
    actual: list[dict[str, Any]] = []
    for item in rows:
        if not _has_issue_metadata(item) and item.get("detail") == _VERDICT_DETAILS.get(
            item["failure_kind"]
        ):
            verdicts.append(item)
        else:
            actual.append(item)
    if not verdicts:
        return rows
    if not summary or summary.get("issue_count_known") is False:
        return actual
    total = summary["issue_count"]
    omitted = summary.get("omitted_count")
    if omitted is not None and (type(omitted) is not int or not 0 <= omitted <= total):
        raise ValueError("invalid verification omitted count")
    expected = (
        total - omitted
        if omitted is not None
        else min(total, VERIFICATION_SAMPLE_LIMIT)
    )
    needed = expected - len(actual)
    if needed == 0:
        return actual
    if needed == len(verdicts):
        return rows
    raise ValueError("ambiguous legacy verification samples")


def _count_knowledge(items: list[dict[str, Any]], *, has_samples: bool) -> bool | None:
    values = [
        item["issue_count_known"]
        for item in items
        if item.get("issue_count_known") is not None
    ]
    if values:
        if any(type(value) is not bool or value != values[0] for value in values):
            raise ValueError("invalid issue count knowledge")
        return values[0]
    if (
        not has_samples
        and not any("issue_count" in item for item in items)
        and any(item.get("failure_kind") == "retention_suspended" for item in items)
    ):
        return False
    return None


def _check_batch(
    items: list[dict[str, Any]],
    ended: datetime,
    *,
    mode: bool | None,
) -> RetentionVerification:
    progress_id, check_id = _check_identity(items)
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
        raise ValueError("invalid verification verdict or clock")
    summary, samples, total = _check_parts(items)

    def text(key: str) -> str | None:
        value = summary.get(key)
        return value if isinstance(value, str) else None

    return RetentionVerification(
        "suspended" if "retention_suspended" in kinds else "clear",
        clocks[0],
        check_id=check_id,
        audit_path=text("audit_path"),
        issue_count=total,
        audit_error=text("audit_error"),
        dry_run=_check_mode(items, fallback=mode),
        issues=tuple(
            VerificationIssue.from_legacy(
                str(item.get("detail") or "")[:400],
                path=item.get("path"),
                category=item.get("category"),
                reason_code=item.get("reason_code"),
                normalize_prefix=False,
            )
            for item in samples
        ),
        audit_candidate_path=text("audit_candidate_path"),
        audit_reason_code=text("audit_reason_code"),
        audit_cause_code=text("audit_cause_code"),
        audit_operation=text("operation"),
        progress_id=progress_id,
        issue_count_known=_count_knowledge(items, has_samples=bool(samples)),
    )


@dataclass(frozen=True, slots=True)
class InvalidCheckEvidence:
    reason: str
    progress_id: str | None = None
    check_id: str | None = None


@dataclass(frozen=True, slots=True)
class RecordedCheckAssessment:
    checks: tuple[RetentionVerification, ...] = ()
    invalid: tuple[InvalidCheckEvidence, ...] = ()


@dataclass(slots=True)
class _CheckGroups:
    groups: dict[tuple[str, object], list[dict[str, Any]]] = field(default_factory=dict)
    identities: dict[tuple[str, object], DiagnosticIdentity] = field(
        default_factory=dict
    )
    check_owners: dict[str, set[str | None]] = field(default_factory=dict)
    bad_owners: set[str] = field(default_factory=set)
    bad_checks: set[str] = field(default_factory=set)
    invalid: list[InvalidCheckEvidence] = field(default_factory=list)

    def reject(self, item: object, reason: str) -> None:
        identity = assess_progress_identity(
            item.get("progress_id") if isinstance(item, dict) else None,
            item.get("check_id") if isinstance(item, dict) else None,
        )
        self.invalid.append(
            InvalidCheckEvidence(reason, identity.progress_id, identity.check_id)
        )
        if identity.progress_id is not None:
            self.bad_owners.add(identity.progress_id)
        if identity.check_id is not None:
            self.bad_checks.add(identity.check_id)

    def add(self, item: dict[str, Any]) -> None:
        identity = assess_progress_identity(
            item.get("progress_id"), item.get("check_id")
        )
        if not identity.valid or type(item.get("exit_code")) is not int:
            self.reject(item, "invalid_identity_or_exit")
            return
        if identity.check_id is not None:
            self.check_owners.setdefault(identity.check_id, set()).add(identity.owner)
        clock = item.get("recorded_at")
        if clock is not None and not isinstance(clock, str):
            self.reject(item, "invalid_recorded_clock")
            return
        key = (
            ("owner", identity.owner)
            if identity.owner is not None
            else (
                "clock",
                clock,
            )
        )
        self.groups.setdefault(key, []).append(item)
        self.identities[key] = identity

    def assess(self, ended: datetime, *, mode: bool | None) -> RecordedCheckAssessment:
        bad_checks = self.bad_checks | {
            check for check, owners in self.check_owners.items() if len(owners) > 1
        }
        invalid = list(self.invalid)
        checks: list[RetentionVerification] = []
        for key, items in self.groups.items():
            identity = self.identities[key]
            try:
                if identity.owner in self.bad_owners or any(
                    item.get("check_id") in bad_checks for item in items
                ):
                    raise ValueError("conflicting verification identities")
                checks.append(_check_batch(items, ended, mode=mode))
            except (ValueError, TypeError):
                invalid.append(
                    InvalidCheckEvidence(
                        "invalid_check_group", identity.progress_id, identity.check_id
                    )
                )
        return RecordedCheckAssessment(
            tuple(sorted(checks, key=lambda check: check.recorded_at or ended)),
            tuple(invalid),
        )


def _damaged_diagnostic(item: object) -> bool:
    if not isinstance(item, dict):
        return True
    stage = item.get("stage")
    if not isinstance(stage, str) or not stage.strip():
        return True
    metadata_invalid = item.get("metadata_invalid")
    if metadata_invalid is not None and metadata_invalid is not False:
        return True
    if item.get("failure_kind") == "invalid_diagnostic_metadata":
        return True
    return stage != "retention_verification" and item.get("failure_kind") in (
        "retention_clear",
        "retention_suspended",
    )


def _recorded_check_assessment(
    row: Mapping[str, Any], ended: datetime
) -> RecordedCheckAssessment:
    try:
        raw = json.loads(row.get("failure_details_json") or "null")
    except (ValueError, TypeError):
        return RecordedCheckAssessment(
            invalid=(InvalidCheckEvidence("invalid_diagnostic_json"),)
        )
    if raw is None:
        return RecordedCheckAssessment()
    if not isinstance(raw, list):
        return RecordedCheckAssessment(
            invalid=(InvalidCheckEvidence("invalid_diagnostic_container"),)
        )
    groups = _CheckGroups()
    for item in raw:
        if (
            isinstance(item, dict)
            and item.get("stage") == "retention_audit_cleanup"
            and item.get("failure_kind")
            not in ("retention_clear", "retention_suspended")
        ):
            # Cleanup shares lineage with verification, not its validity domain.
            continue
        if _damaged_diagnostic(item):
            groups.reject(item, "invalid_diagnostic_metadata")
        elif item.get("stage") == "retention_verification":
            if type(item.get("exit_code")) is int and item["exit_code"] != 0:
                # Non-completed diagnostics never asserted protection, including
                # legacy contention records; their run outcome is assessed separately.
                continue
            groups.add(item)
    return groups.assess(ended, mode=_argument_mode(row))


def _recorded_checks(
    row: Mapping[str, Any], ended: datetime
) -> tuple[RetentionVerification, ...]:
    """Compatibility projection: preserve every independently valid check."""
    return _recorded_check_assessment(row, ended).checks


def _verification_uncertainty(
    recorded: RecordedCheckAssessment,
    ended: datetime,
    boundary: datetime | None,
) -> datetime | None:
    if recorded.invalid:
        boundary = ended
    if boundary is not None and any(
        check.state == "clear"
        and check.recorded_at is not None
        and check.recorded_at > boundary
        for check in recorded.checks
    ):
        return None
    return boundary


def _execution_recovery(
    row: Mapping[str, Any],
    checks: tuple[RetentionVerification, ...],
    *,
    mode: bool | None,
    outstanding: bool,
) -> tuple[AttemptState, bool]:
    if row.get("exit_code") == EXIT_CONTENTION and not row.get("error"):
        return "contended", outstanding
    if row.get("error") or row.get("exit_code") != 0:
        return "failed", True
    if not checks:
        return "unverified", outstanding
    if checks[-1].state == "clear" and mode is False:
        return "completed", False
    return "completed", outstanding


def retention_status(runs: pl.DataFrame, *, as_of: datetime) -> RetentionStatus:
    assessment = assess_attempts(runs, "prune-scores", as_of)
    verification = RetentionVerification()
    state: AttemptState = "not_recorded"
    latest = None
    outstanding = False
    mode = None
    uncertainty_until: datetime | None = None
    for attempt in assessment.attempts:
        row, latest = attempt.row, attempt.ended_at
        recorded = _recorded_check_assessment(row, latest)
        checks = recorded.checks
        uncertainty_until = _verification_uncertainty(
            recorded, latest, uncertainty_until
        )
        mode = checks[-1].dry_run if checks else _argument_mode(row)
        if checks:
            check = checks[-1]
            if check.recorded_at is not None and (
                verification.recorded_at is None
                or check.recorded_at >= verification.recorded_at
            ):
                verification = check
        state, outstanding = _execution_recovery(
            row, checks, mode=mode, outstanding=outstanding
        )
    if assessment.unresolved:
        state = "unknown"
        mode = None
        clocks = assessment.invalid_ended
        latest = None if None in clocks else max(clocks) if clocks else latest
    return RetentionStatus(
        verification,
        latest,
        state,
        outstanding,
        as_of,
        mode,
        uncertainty_until is not None,
    )


def latest_retention_verification(
    runs: pl.DataFrame, *, as_of: datetime
) -> RetentionVerification:
    """Compatibility adapter: an unverified attempt cannot erase a recorded check."""
    return retention_status(runs, as_of=as_of).verification
