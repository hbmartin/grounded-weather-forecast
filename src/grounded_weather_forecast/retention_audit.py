"""Full pruning verification audits, with bounded journal samples and safe expiry."""

import json
import os
import re
import stat
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum, StrEnum
from pathlib import Path
from typing import Literal
from uuid import uuid4

import polars as pl
from filelock import FileLock, Timeout

from grounded_weather_forecast.artifact_labels import issue_label
from grounded_weather_forecast.clocks import aware_utc
from grounded_weather_forecast.config import Config
from grounded_weather_forecast.retention_status import (
    VERIFICATION_SAMPLE_LIMIT,
    RetentionVerification,
)
from grounded_weather_forecast.runs import normalize_error_detail, read_runs, runs_path
from grounded_weather_forecast.serve.approvals import ApprovalIssue
from grounded_weather_forecast.status import (
    ComponentDiagnostic,
    effective_progress_id,
    record_verification_progress,
)
from grounded_weather_forecast.storage import temporary_sibling
from grounded_weather_forecast.verification_text import (
    IssueSource,
    VerificationIssue,
    canonical_issues,
    reason_for_error,
)

AUDIT_RETENTION_DAYS = 90
_AUDIT_NAME = re.compile(r"[0-9a-f]{32}\.jsonl")
_STAGING_NAME = re.compile(r"\.[0-9a-f]{32}\.jsonl\.[0-9a-f]{16}\.tmp")
_AUDIT_ERRORS = (
    OSError,
    ValueError,
    TypeError,
    OverflowError,
    pl.exceptions.PolarsError,
)


class _MissingContext(Enum):
    PATH = "path not supplied"


def audit_directory(config: Config) -> Path:
    return config.artifacts_dir / "history" / "pruning_verifications"


@contextmanager
def _audit_lock(directory: Path) -> Iterator[None]:
    try:
        lock = FileLock(directory / ".audit.lock", timeout=0)
        lock.acquire()
    except (Timeout,) as error:  # noqa: B013 -- only acquisition establishes contention
        raise _AuditPublicationError(
            "audit_lock_contended", error, "lock_acquire", directory
        ) from error
    except (OSError,) as error:  # noqa: B013
        raise _AuditPublicationError(
            "audit_io_failed", error, "lock_acquire", directory
        ) from error
    primary: BaseException | None = None
    try:
        yield
    except (BaseException,) as error:  # noqa: B013 -- track only this operation's failure
        primary = error
        raise
    finally:
        try:
            lock.release()
        except (OSError,) as error:  # noqa: B013 -- cleanup must preserve cancellation
            if primary is None:
                raise _AuditPublicationError(
                    "audit_cleanup_failed", error, "lock_release", directory
                ) from error
            primary.add_note(f"audit lock release also failed: {type(error).__name__}")


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _AuditPublicationError(OSError):
    def __init__(
        self,
        reason_code: str,
        cause: Exception,
        operation: str | None = None,
        path: Path | None | _MissingContext = _MissingContext.PATH,
    ) -> None:
        self.reason_code = reason_code
        self.operation = operation
        self.path_supplied = path is not _MissingContext.PATH
        self.path = path if isinstance(path, Path) else None
        self.cause = cause
        super().__init__(reason_code, cause)
        self.args = (reason_code, cause, operation, path)

    def __str__(self) -> str:
        cause = self.args[1]
        return f"{type(cause).__name__}: {cause}"


@contextmanager
def _audit_operation(
    operation: str,
    path: Path | None,
) -> Iterator[None]:
    try:
        yield
    except (_AuditPublicationError,) as error:  # noqa: B013 -- preserve known context
        if error.operation is None or not error.path_supplied:
            raise _AuditPublicationError(
                error.reason_code,
                error.cause,
                error.operation or operation,
                error.path if error.path_supplied else path,
            ) from error
        raise
    except (*_AUDIT_ERRORS,) as error:
        source = error.filename if isinstance(error, OSError) else None
        location = Path(source) if isinstance(source, str) and source else path
        raise _AuditPublicationError(
            "audit_io_failed", error, operation, location
        ) from error


def _legacy_issues(
    verification: RetentionVerification,
) -> tuple[VerificationIssue, ...]:
    # The compatibility API may carry more than twenty issues; full audits retain
    # every one, while sample_issues() bounds all journal and public projections.
    return tuple(verification.iter_issues())


def _publish_audit(
    path: Path,
    verification: RetentionVerification,
    *,
    dry_run: bool,
    issues: Sequence[VerificationIssue],
    on_link: Callable[[], None],
) -> None:
    temporary = temporary_sibling(path)
    primary: BaseException | None = None
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "pruning_verification",
                        "check_id": verification.check_id,
                        "recorded_at": verification.recorded_at.isoformat()
                        if verification.recorded_at
                        else None,
                        "dry_run": dry_run,
                    }
                )
                + "\n"
            )
            for issue in issues:
                stream.write(
                    json.dumps(
                        {
                            "kind": "issue",
                            "path": str(issue.path) if issue.path is not None else None,
                            "category": issue.category,
                            "detail": issue.detail,
                            "reason_code": issue.reason_code,
                        }
                    )
                    + "\n"
                )
            stream.write(
                json.dumps({"kind": "complete", "issue_count": len(issues)}) + "\n"
            )
            stream.flush()
            try:
                os.fsync(stream.fileno())
            except (OSError,) as error:  # noqa: B013
                raise _AuditPublicationError(
                    "audit_sync_failed", error, "staging_sync", temporary
                ) from error
        # Exclusive publication: even an ID collision must never rewrite an audit.
        try:
            os.link(temporary, path)
        except (OSError,) as error:  # noqa: B013
            raise _AuditPublicationError(
                "audit_link_failed", error, "link", path
            ) from error
        on_link()
        try:
            _sync_directory(path.parent)
        except (OSError,) as error:  # noqa: B013
            raise _AuditPublicationError(
                "audit_sync_failed", error, "directory_sync", path.parent
            ) from error
    except (BaseException,) as error:  # noqa: B013 -- preserve cancellation through cleanup
        primary = error
        raise
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except (OSError,) as error:  # noqa: B013 -- preserve the original interruption
            if primary is None:
                raise _AuditPublicationError(
                    "audit_cleanup_failed", error, "staging_cleanup", temporary
                ) from error
            primary.add_note(
                f"audit staging cleanup also failed: {type(error).__name__}"
            )


def publish_verification(
    config: Config,
    verification: RetentionVerification,
    *,
    dry_run: bool,
    issues: tuple[ApprovalIssue | VerificationIssue, ...] | None = None,
) -> RetentionVerification:
    problems = (
        tuple(VerificationIssue.from_issue(issue) for issue in issues)
        if issues is not None
        else _legacy_issues(verification)
    )
    identity = uuid4().hex
    identified = replace(
        verification,
        check_id=identity,
        progress_id=identity,
        issue_count=len(problems),
        dry_run=dry_run,
        issues=problems[:VERIFICATION_SAMPLE_LIMIT],
        audit_path=None,
        audit_candidate_path=None,
        audit_reason_code="audit_incomplete" if problems else None,
        audit_cause_code=None,
        audit_operation=None,
        audit_error="audit publication not completed" if problems else None,
    )
    record_verification_progress(identified.diagnostics())
    current = identified

    def linked() -> None:
        nonlocal current
        current = replace(identified, audit_candidate_path=reference)
        record_verification_progress(current.diagnostics())

    if identified.total:
        path = audit_directory(config) / f"{identified.check_id}.jsonl"
        reference = str(path.relative_to(config.artifacts_dir))
        try:
            with _audit_operation("directory_create", path.parent):
                path.parent.mkdir(parents=True, exist_ok=True)
            with _audit_lock(path.parent), _audit_operation("staging_write", path):
                _publish_audit(
                    path,
                    identified,
                    dry_run=dry_run,
                    issues=problems,
                    on_link=linked,
                )
            current = replace(
                current,
                audit_path=reference,
                audit_candidate_path=None,
                audit_error=None,
                audit_reason_code=None,
                audit_cause_code=None,
                audit_operation=None,
            )
        except (OSError, ValueError, TypeError) as problem:
            current = replace(
                current,
                audit_error=f"audit unavailable: {type(problem).__name__}: {problem}"[
                    :400
                ],
                audit_reason_code=_publication_reason(problem),
                audit_cause_code=reason_for_error(problem),
                audit_operation=problem.operation
                if isinstance(problem, _AuditPublicationError)
                else None,
            )
    record_verification_progress(current.diagnostics())
    return current


def _publication_reason(error: Exception) -> str:
    if isinstance(error, _AuditPublicationError):
        return error.reason_code
    return "audit_io_failed"


def _referenced_audits(config: Config) -> set[str]:
    config.dataset.dir.stat()  # A vanished dataset cannot establish absent references.
    ledger = runs_path(config)
    try:
        ledger.stat()
    except (FileNotFoundError,):  # noqa: B013 -- only genuine absence means no references
        try:
            ledger.lstat()
        except (FileNotFoundError,):  # noqa: B013
            return set()
        raise ValueError("run ledger is a dangling reference") from None
    references = set()
    for value in read_runs(ledger)["failure_details_json"].drop_nulls():
        raw = json.loads(value)
        if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
            raise ValueError("run diagnostic references could not be verified")
        for item in raw:
            for key in ("audit_path", "audit_candidate_path"):
                reference = item.get(key)
                if reference is None:
                    continue
                if not isinstance(reference, str):
                    raise ValueError("invalid audit reference")
                expected = Path("history/pruning_verifications") / Path(reference).name
                if Path(reference) != expected or not _AUDIT_NAME.fullmatch(
                    expected.name
                ):
                    raise ValueError("invalid audit reference")
                references.add(expected.name)
            # Older interrupted publishers recorded a check ID but no audit path.
            # A complete candidate has independently validated its matching header.
            identity = item.get("check_id")
            if (
                item.get("stage") == "retention_verification"
                and isinstance(identity, str)
                and _AUDIT_NAME.fullmatch(f"{identity}.jsonl")
            ):
                references.add(f"{identity}.jsonl")
    return references


def _expired_audit(path: Path, now: datetime) -> bool:
    # Validate the whole stream before deleting a recognized completed audit.
    with path.open(encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if (
            not isinstance(header, dict)
            or header.get("schema_version") != 1
            or header.get("kind") != "pruning_verification"
            or header.get("check_id") != path.stem
        ):
            return False
        moment = aware_utc(header.get("recorded_at"))
        if moment is None or moment >= now - timedelta(days=AUDIT_RETENTION_DAYS):
            return False
        count = 0
        complete = False
        for line in stream:
            item = json.loads(line)
            if not isinstance(item, dict) or complete:
                return False
            if item.get("kind") == "issue":
                count += 1
            elif item.get("kind") == "complete" and item.get("issue_count") == count:
                complete = True
            else:
                return False
        return complete


def _signature(path: Path) -> tuple[int, ...] | None:
    info = path.lstat()
    return (
        (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if stat.S_ISREG(info.st_mode)
        else None
    )


type CleanupFailureRole = Literal["abort_cause", "file_local"]


def _cleanup_role(category: str) -> CleanupFailureRole | None:
    match category:
        case "audit_cleanup_aborted":
            return "abort_cause"
        case "audit_cleanup_file_failed":
            return "file_local"
        case _:
            return None


@dataclass(frozen=True, slots=True)
class AuditCleanupFailure:
    operation: str | None
    path: Path | None
    raw_detail: str
    reason_code: str | None = None
    cause_code: str | None = None
    error_type: str | None = None
    failure_role: CleanupFailureRole | None = None

    def __post_init__(self) -> None:
        if self.failure_role not in {None, "abort_cause", "file_local"}:
            raise ValueError("unknown audit cleanup failure role")

    @classmethod
    def capture(
        cls,
        error: Exception,
        *,
        operation: str,
        path: Path | None,
        failure_role: CleanupFailureRole | None = None,
    ) -> "AuditCleanupFailure":
        if isinstance(error, _AuditPublicationError):
            return cls(
                error.operation or operation,
                error.path if error.path_supplied else path,
                str(error.cause),
                error.reason_code,
                reason_for_error(error.cause),
                type(error.cause).__name__,
                failure_role,
            )
        return cls(
            operation,
            path,
            str(error),
            None,
            reason_for_error(error),
            type(error).__name__,
            failure_role,
        )

    def issue(self, *, aborted: bool | None = None) -> VerificationIssue:
        role = self.failure_role
        if aborted is not None:
            supplied: CleanupFailureRole = "abort_cause" if aborted else "file_local"
            if role is not None and role != supplied:
                raise ValueError(
                    "cleanup issue role conflicts with recorded failure role"
                )
            role = supplied
        match role:
            case "abort_cause":
                category = "audit_cleanup_aborted"
            case "file_local":
                category = "audit_cleanup_file_failed"
            case _:
                category = "audit_cleanup_failure"
        return VerificationIssue(
            self.path,
            category,
            normalize_error_detail(self.raw_detail) or "Audit operation failed.",
            self.cause_code
            if self.reason_code in {None, "audit_io_failed"}
            else self.reason_code,
        )


@dataclass(frozen=True, slots=True)
class AuditCleanupEntry:
    """One bounded issue and its optional complete original failure evidence."""

    issue: VerificationIssue
    failure: AuditCleanupFailure | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "issue", VerificationIssue.from_issue(self.issue))
        if self.failure is not None:
            failure = self.failure
            if failure.failure_role is None and (
                role := _cleanup_role(self.issue.category)
            ):
                failure = replace(failure, failure_role=role)
                object.__setattr__(self, "failure", failure)
            if self.issue != failure.issue():
                raise ValueError("cleanup issue does not match its failure evidence")


def _failure_matches(issue: VerificationIssue, failure: AuditCleanupFailure) -> bool:
    role = _cleanup_role(issue.category)
    if failure.failure_role is not None and failure.failure_role != role:
        return False
    expected = (
        failure.issue(aborted=role == "abort_cause")
        if role is not None and failure.failure_role is None
        else failure.issue()
    )
    return issue == expected


def _failure_positions(
    issues: tuple[VerificationIssue, ...],
    failures: tuple[AuditCleanupFailure, ...],
    *,
    reverse: bool,
) -> tuple[int, ...]:
    """Find the earliest/latest order-preserving compatible association."""
    indexes = iter(range(len(issues) - 1, -1, -1) if reverse else range(len(issues)))
    positions = []
    for failure in reversed(failures) if reverse else failures:
        for index in indexes:
            issue = issues[index]
            if _failure_matches(issue, failure):
                positions.append(index)
                break
        else:
            raise ValueError("cleanup failure evidence does not match supplied issues")
    return tuple(reversed(positions)) if reverse else tuple(positions)


def _annotated_entries(
    issues: tuple[VerificationIssue, ...],
    failures: tuple[AuditCleanupFailure, ...],
) -> tuple[AuditCleanupEntry, ...]:
    if len(failures) > len(issues):
        raise ValueError("cleanup failure evidence must match supplied issues")
    first = _failure_positions(issues, failures, reverse=False)
    if first != _failure_positions(issues, failures, reverse=True):
        raise ValueError("ambiguous cleanup failure evidence; use aligned_failures")
    by_index = dict(zip(first, failures, strict=True))
    return tuple(
        AuditCleanupEntry(issue, by_index.get(index))
        for index, issue in enumerate(issues)
    )


def _cleanup_entries(
    entries: Sequence[AuditCleanupEntry],
    issues: Sequence[IssueSource] | None,
    failures: Sequence[AuditCleanupFailure] | None,
    *,
    aligned_failures: Sequence[AuditCleanupFailure | None] | None,
) -> tuple[AuditCleanupEntry, ...]:
    raw = tuple(failures or ())
    if aligned_failures is not None:
        if raw:
            raise ValueError("compact and aligned failure evidence cannot be combined")
        problems = (
            canonical_issues(issues)
            if issues is not None
            else tuple(entry.issue for entry in entries)
        )
        if len(problems) != len(aligned_failures):
            raise ValueError("aligned failure evidence must match the issue count")
        return tuple(
            AuditCleanupEntry(issue, failure)
            for issue, failure in zip(problems, aligned_failures, strict=True)
        )
    if issues is not None:
        return _annotated_entries(canonical_issues(issues), raw)
    if raw:
        # Failure-only compatibility input replaces its source, including on replace().
        return tuple(AuditCleanupEntry(failure.issue(), failure) for failure in raw)
    return tuple(entries)


class CleanupOutcome(StrEnum):
    UNKNOWN = "unknown"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True, init=False)
class AuditCleanupResult:
    entries: tuple[AuditCleanupEntry, ...] = ()
    aborted: bool = False
    omitted_count: int = 0
    recorded_at: datetime | None = None
    dry_run: bool | None = None
    deleted_audit_count: int | None = None
    deleted_staging_count: int | None = None
    eligible_audit_count: int | None = None
    eligible_staging_count: int | None = None
    completed: bool | None = None

    def __init__(
        self,
        issues: Sequence[IssueSource] | None = None,
        aborted: bool = False,  # noqa: FBT001, FBT002 -- positional compatibility
        omitted_count: int = 0,
        recorded_at: datetime | None = None,
        dry_run: bool | None = None,  # noqa: FBT001 -- positional compatibility
        deleted_audit_count: int | None = None,
        deleted_staging_count: int | None = None,
        raw_failures: Sequence[AuditCleanupFailure] | None = None,
        *,
        entries: Sequence[AuditCleanupEntry] = (),
        eligible_audit_count: int | None = None,
        eligible_staging_count: int | None = None,
        completed: bool | None = None,
        aligned_failures: Sequence[AuditCleanupFailure | None] | None = None,
    ) -> None:
        if aborted and completed is True:
            raise ValueError("an aborted cleanup cannot be completed")
        normalized = _cleanup_entries(
            entries,
            issues,
            raw_failures,
            aligned_failures=aligned_failures,
        )
        if not aborted and any(
            entry.issue.category == "audit_cleanup_aborted" for entry in normalized
        ):
            raise ValueError("abort-cause evidence requires aborted=True")
        values = {
            "entries": normalized,
            "aborted": aborted,
            "omitted_count": omitted_count,
            "recorded_at": recorded_at,
            "dry_run": dry_run,
            "deleted_audit_count": deleted_audit_count,
            "deleted_staging_count": deleted_staging_count,
            "eligible_audit_count": eligible_audit_count,
            "eligible_staging_count": eligible_staging_count,
            "completed": completed,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def issues(self) -> tuple[VerificationIssue, ...]:
        return tuple(entry.issue for entry in self.entries)

    @property
    def raw_failures(self) -> tuple[AuditCleanupFailure, ...]:
        return tuple(
            entry.failure for entry in self.entries if entry.failure is not None
        )

    @property
    def aligned_failures(self) -> tuple[AuditCleanupFailure | None, ...]:
        """Lossless positional view, including explicitly absent original evidence."""
        return tuple(entry.failure for entry in self.entries)

    @property
    def outcome(self) -> CleanupOutcome:
        if self.aborted or self.completed is False:
            return CleanupOutcome.INCOMPLETE
        if self.completed is True:
            return CleanupOutcome.COMPLETED
        return CleanupOutcome.UNKNOWN

    def summary(self) -> str:
        match self.outcome:
            case CleanupOutcome.INCOMPLETE:
                state = "did not complete"
            case CleanupOutcome.UNKNOWN:
                state = "final status is unknown"
            case CleanupOutcome.COMPLETED:
                state = "completed with issues" if self.entries else "completed"
        mode = " (dry run)" if self.dry_run is True else ""
        parts = [f"Audit cleanup {state}{mode}."]
        if (
            self.eligible_audit_count is not None
            and self.eligible_staging_count is not None
        ):
            if self.outcome is not CleanupOutcome.COMPLETED:
                prefix = "Verified eligible so far:"
            elif self.dry_run is True:
                prefix = "Would delete"
            else:
                prefix = "Verified eligible:"
            parts.append(
                f"{prefix} {self.eligible_audit_count} audits and {self.eligible_staging_count} staging files."
            )
        if (
            self.deleted_audit_count is not None
            and self.deleted_staging_count is not None
        ):
            prefix = (
                "Recorded deletions:"
                if self.outcome is CleanupOutcome.UNKNOWN
                else "Deleted"
            )
            parts.append(
                f"{prefix} {self.deleted_audit_count} audits and {self.deleted_staging_count} staging files."
            )
        else:
            parts.append("Completed deletion counts were not recorded.")
        return " ".join(parts)

    def public_notes(self, config: Config) -> tuple[str, ...]:
        notes = tuple(issue_label(entry.issue, config) for entry in self.entries)
        if self.omitted_count:
            notes += (f"{self.omitted_count} additional audit cleanup issues omitted.",)
        return notes

    def diagnostics(
        self, *, check_id: str | None = None, progress_id: str | None = None
    ) -> tuple[ComponentDiagnostic, ...]:
        if self == AuditCleanupResult():
            return ()
        common = ComponentDiagnostic(
            stage="retention_audit_cleanup",
            exit_code=0,
            recorded_at=self.recorded_at.isoformat() if self.recorded_at else None,
            check_id=check_id,
            progress_id=effective_progress_id(progress_id, check_id),
            dry_run=self.dry_run,
        )
        match self.outcome:
            case CleanupOutcome.INCOMPLETE:
                kind = "audit_cleanup_aborted"
            case CleanupOutcome.UNKNOWN:
                kind = "audit_cleanup_unverified"
            case CleanupOutcome.COMPLETED:
                kind = (
                    "audit_cleanup_issues" if self.entries else "audit_cleanup_complete"
                )
        summary = replace(
            common,
            failure_kind=kind,
            detail=self.summary(),
            issue_count=len(self.entries) + self.omitted_count,
            omitted_count=self.omitted_count,
            deleted_audit_count=self.deleted_audit_count,
            deleted_staging_count=self.deleted_staging_count,
            eligible_audit_count=self.eligible_audit_count,
            eligible_staging_count=self.eligible_staging_count,
            cleanup_completed=None
            if self.outcome is CleanupOutcome.UNKNOWN
            else self.outcome is CleanupOutcome.COMPLETED,
        )
        samples = tuple(
            replace(
                common,
                detail=entry.issue.detail,
                path=str(entry.issue.path) if entry.issue.path is not None else None,
                category=entry.issue.category,
                reason_code=entry.issue.reason_code,
                operation=entry.failure.operation
                if entry.failure is not None
                else None,
                audit_cause_code=entry.failure.cause_code
                if entry.failure is not None
                else None,
                error_type=entry.failure.error_type
                if entry.failure is not None
                else None,
            )
            for entry in self.entries
        )
        return (summary, *samples)


@dataclass(frozen=True, slots=True)
class AuditCleanupRecord:
    """Bind one immutable source/check pair to the diagnostic batch it produced."""

    result: AuditCleanupResult
    check_id: str | None = None
    progress_id: str | None = None
    diagnostics: tuple[ComponentDiagnostic, ...] = field(init=False)

    def __post_init__(self) -> None:
        identity = effective_progress_id(self.progress_id, self.check_id)
        object.__setattr__(self, "progress_id", identity)
        object.__setattr__(
            self,
            "diagnostics",
            self.result.diagnostics(check_id=self.check_id, progress_id=identity),
        )


@dataclass(slots=True)
class _CleanupNotes:
    samples: list[AuditCleanupFailure] = field(default_factory=list)
    count: int = 0
    abort: AuditCleanupFailure | None = None
    deleted_audit_count: int = 0
    deleted_staging_count: int = 0
    eligible_audit_count: int = 0
    eligible_staging_count: int = 0

    def add(
        self, path: Path, error: Exception, *, operation: str = "candidate_verify"
    ) -> None:
        self.count += 1
        if len(self.samples) < VERIFICATION_SAMPLE_LIMIT:
            self.samples.append(
                AuditCleanupFailure.capture(
                    error, operation=operation, path=path, failure_role="file_local"
                )
            )

    def finish(
        self, recorded_at: datetime | None, *, dry_run: bool
    ) -> AuditCleanupResult:
        prefix = (self.abort,) if self.abort is not None else ()
        budget = VERIFICATION_SAMPLE_LIMIT - len(prefix)
        shown = min(self.count, budget)
        if self.count > shown:
            shown = max(0, budget - 1)
        failures = prefix + tuple(self.samples[:shown])
        return AuditCleanupResult(
            entries=tuple(
                AuditCleanupEntry(
                    failure.issue(),
                    failure,
                )
                for failure in failures
            ),
            aborted=self.abort is not None,
            omitted_count=self.count - shown,
            recorded_at=recorded_at,
            dry_run=dry_run,
            deleted_audit_count=self.deleted_audit_count,
            deleted_staging_count=self.deleted_staging_count,
            eligible_audit_count=self.eligible_audit_count,
            eligible_staging_count=self.eligible_staging_count,
            completed=self.abort is None,
        )


def _cleanup_candidates(
    directory: Path,
    now: datetime,
    notes: "_CleanupNotes",
) -> dict[Path, tuple[int, ...]]:
    candidates = {}
    with _audit_operation("enumerate", directory), os.scandir(directory) as entries:
        paths = sorted(directory / entry.name for entry in entries)
    for path in paths:
        staging = bool(_STAGING_NAME.fullmatch(path.name))
        if not (_AUDIT_NAME.fullmatch(path.name) or staging):
            continue
        try:
            signature = _signature(path)
            if signature is None:
                continue
            expired = (
                signature[3] / 1e9
                < (now - timedelta(days=AUDIT_RETENTION_DAYS)).timestamp()
                if staging
                else _expired_audit(path, now)
            )
            if expired and _signature(path) == signature:
                candidates[path] = signature
        except (OSError, ValueError, TypeError, OverflowError, StopIteration) as error:
            notes.add(path, error)
    return candidates


def cleanup_audits_result(
    config: Config, *, now: datetime, dry_run: bool
) -> AuditCleanupResult:
    """Expire known audit/staging files only after coordinated revalidation."""
    notes = _CleanupNotes()
    cutoff = None
    try:
        with _audit_operation("cutoff_validate", None):
            cutoff = aware_utc(now)
            if cutoff is None:
                raise ValueError("audit cutoff must be timezone-aware")
        directory = audit_directory(config)
        with _audit_operation("directory_stat", directory):
            try:
                directory.stat()
            except (FileNotFoundError,):  # noqa: B013 -- a dangling directory is uncertain
                if not directory.is_symlink():
                    return notes.finish(cutoff, dry_run=dry_run)
                raise ValueError("audit directory is a dangling reference") from None
        with _audit_lock(directory):
            candidates = _cleanup_candidates(directory, cutoff, notes)
            references: set[str] = set()
            if any(_AUDIT_NAME.fullmatch(path.name) for path in candidates):
                with _audit_operation("references", None):
                    references = _referenced_audits(config)
            for path, signature in candidates.items():
                try:
                    if path.name in references:
                        continue
                    with _audit_operation("revalidate", path):
                        unchanged = _signature(path) == signature
                    if unchanged:
                        committed = bool(_AUDIT_NAME.fullmatch(path.name))
                        if committed:
                            notes.eligible_audit_count += 1
                        else:
                            notes.eligible_staging_count += 1
                        if not dry_run:
                            with _audit_operation("delete", path):
                                path.unlink()
                            if committed:
                                notes.deleted_audit_count += 1
                            else:
                                notes.deleted_staging_count += 1
                except (OSError,) as error:  # noqa: B013 -- independent files still proceed
                    notes.add(path, error, operation="delete")
    except (*_AUDIT_ERRORS,) as error:
        notes.abort = AuditCleanupFailure.capture(
            error, operation="cleanup", path=None, failure_role="abort_cause"
        )
    return notes.finish(cutoff, dry_run=dry_run)


def cleanup_audits(config: Config, *, now: datetime, dry_run: bool) -> tuple[str, ...]:
    """Compatibility adapter returning fixed, safe public cleanup notes."""
    return cleanup_audits_result(config, now=now, dry_run=dry_run).public_notes(config)
