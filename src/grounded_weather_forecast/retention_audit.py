"""Full pruning verification audits, with bounded journal samples and safe expiry."""

import json
import os
import re
import stat
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import polars as pl
from filelock import FileLock

from grounded_weather_forecast.clocks import aware_utc
from grounded_weather_forecast.config import Config
from grounded_weather_forecast.retention_status import (
    VERIFICATION_SAMPLE_LIMIT,
    RetentionVerification,
)
from grounded_weather_forecast.runs import read_runs, runs_path
from grounded_weather_forecast.serve.approvals import ApprovalIssue
from grounded_weather_forecast.status import record_verification_progress
from grounded_weather_forecast.storage import temporary_sibling
from grounded_weather_forecast.verification_text import (
    VerificationIssue,
    normalize_detail,
    reason_for_error,
)

AUDIT_RETENTION_DAYS = 90
_AUDIT_NAME = re.compile(r"[0-9a-f]{32}\.jsonl")
_STAGING_NAME = re.compile(r"\.[0-9a-f]{32}\.jsonl\.[0-9a-f]{16}\.tmp")


def audit_directory(config: Config) -> Path:
    return config.artifacts_dir / "history" / "pruning_verifications"


@contextmanager
def _audit_lock(directory: Path) -> Iterator[None]:
    lock = FileLock(directory / ".audit.lock", timeout=0)
    lock.acquire()
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
                raise
            primary.add_note(f"audit lock release also failed: {type(error).__name__}")


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _AuditPublicationError(OSError):
    def __init__(self, reason_code: str, cause: OSError) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code, cause)

    def __str__(self) -> str:
        cause = self.args[1]
        return f"{type(cause).__name__}: {cause}"


def _legacy_issues(
    verification: RetentionVerification,
) -> tuple[VerificationIssue, ...]:
    return tuple(
        VerificationIssue(
            Path(source)
            if index < len(verification.paths) and (source := verification.paths[index])
            else None,
            verification.categories[index]
            if index < len(verification.categories)
            else "verification_invalid",
            normalize_detail(
                detail,
                verification.paths[index] if index < len(verification.paths) else None,
                verification.categories[index]
                if index < len(verification.categories)
                else None,
            ),
            verification.reason_codes[index]
            if index < len(verification.reason_codes)
            else None,
        )
        for index, detail in enumerate(verification.details)
    )


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
            os.fsync(stream.fileno())
        # Exclusive publication: even an ID collision must never rewrite an audit.
        try:
            os.link(temporary, path)
        except (OSError,) as error:  # noqa: B013
            raise _AuditPublicationError(
                reason_for_error(error) or "audit_link_failed", error
            ) from error
        on_link()
        try:
            _sync_directory(path.parent)
        except (OSError,) as error:  # noqa: B013
            raise _AuditPublicationError("audit_sync_failed", error) from error
    except (BaseException,) as error:  # noqa: B013 -- preserve cancellation through cleanup
        primary = error
        raise
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except (OSError,) as error:  # noqa: B013 -- preserve the original interruption
            if primary is None:
                raise _AuditPublicationError("audit_cleanup_failed", error) from error
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
        tuple(
            VerificationIssue(
                issue.path, issue.category, issue.detail, issue.reason_code
            )
            for issue in issues
        )
        if issues is not None
        else _legacy_issues(verification)
    )
    identified = replace(
        verification,
        check_id=uuid4().hex,
        issue_count=len(problems),
        dry_run=dry_run,
        details=tuple(
            issue.diagnostic() for issue in problems[:VERIFICATION_SAMPLE_LIMIT]
        ),
        paths=tuple(
            str(issue.path) if issue.path is not None else None
            for issue in problems[:VERIFICATION_SAMPLE_LIMIT]
        ),
        categories=tuple(
            issue.category for issue in problems[:VERIFICATION_SAMPLE_LIMIT]
        ),
        reason_codes=tuple(
            issue.reason_code for issue in problems[:VERIFICATION_SAMPLE_LIMIT]
        ),
        audit_path=None,
        audit_candidate_path=None,
        audit_reason_code="audit_incomplete" if problems else None,
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
            path.parent.mkdir(parents=True, exist_ok=True)
            with _audit_lock(path.parent):
                _publish_audit(
                    path, identified, dry_run=dry_run, issues=problems, on_link=linked
                )
            current = replace(
                current,
                audit_path=reference,
                audit_candidate_path=None,
                audit_error=None,
                audit_reason_code=None,
            )
        except (OSError, ValueError, TypeError) as problem:
            current = replace(
                current,
                audit_error=f"audit unavailable: {type(problem).__name__}: {problem}"[
                    :400
                ],
                audit_reason_code=(
                    problem.reason_code
                    if isinstance(problem, _AuditPublicationError)
                    else "audit_cleanup_failed"
                    if current.audit_candidate_path
                    else "audit_lock_contended"
                    if isinstance(problem, TimeoutError)
                    else reason_for_error(problem) or "audit_io_failed"
                ),
            )
    record_verification_progress(current.diagnostics())
    return current


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


@dataclass(slots=True)
class _CleanupNotes:
    samples: list[str] = field(default_factory=list)
    count: int = 0
    abort: str | None = None

    def add(self, note: str) -> None:
        self.count += 1
        if len(self.samples) < VERIFICATION_SAMPLE_LIMIT:
            self.samples.append(note)

    def finish(self) -> tuple[str, ...]:
        prefix = [self.abort] if self.abort else []
        budget = VERIFICATION_SAMPLE_LIMIT - len(prefix)
        shown = min(self.count, budget)
        if self.count > shown:
            shown = max(0, budget - 1)
            return tuple(
                prefix
                + self.samples[:shown]
                + [f"{self.count - shown} additional audit cleanup issues omitted."]
            )
        return tuple(prefix + self.samples[:shown])


def _cleanup_candidates(
    directory: Path,
    now: datetime,
    notes: "_CleanupNotes",
) -> dict[Path, tuple[int, ...]]:
    candidates = {}
    with os.scandir(directory) as entries:
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
            notes.add(f"audit cleanup kept {path.name}: {type(error).__name__}"[:400])
    return candidates


def cleanup_audits(config: Config, *, now: datetime, dry_run: bool) -> tuple[str, ...]:
    """Expire known audit/staging files only after coordinated revalidation."""
    notes = _CleanupNotes()
    try:
        cutoff = aware_utc(now)
        if cutoff is None:
            raise ValueError("audit cutoff must be timezone-aware")
        directory = audit_directory(config)
        try:
            directory.stat()
        except (FileNotFoundError,):  # noqa: B013 -- a dangling directory is uncertain
            if not directory.is_symlink():
                return ()
            raise ValueError("audit directory is a dangling reference") from None
        with _audit_lock(directory):
            candidates = _cleanup_candidates(directory, cutoff, notes)
            references = (
                _referenced_audits(config)
                if any(_AUDIT_NAME.fullmatch(path.name) for path in candidates)
                else set()
            )
            for path, signature in candidates.items():
                try:
                    if (
                        path.name not in references
                        and _signature(path) == signature
                        and not dry_run
                    ):
                        path.unlink()
                except (OSError,) as error:  # noqa: B013 -- one damaged file cannot block independent cleanup
                    notes.add(
                        f"audit cleanup kept {path.name}: {type(error).__name__}"[:400]
                    )
    except (
        OSError,
        ValueError,
        TypeError,
        OverflowError,
        pl.exceptions.PolarsError,
    ) as error:
        notes.abort = f"audit cleanup not completed: {type(error).__name__}: {error}"[
            :400
        ]
    return notes.finish()
