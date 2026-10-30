"""Exit-status protocol shared by CLI commands and installed job wrappers."""

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields
from pathlib import Path

EXIT_NO_FOLDS = 3
EXIT_CONTENTION = 75
PROMOTION_ONLY_FAILURE_KINDS = frozenset(
    {
        "promotion_evidence_conflict",
        "promotion_evidence_invalid",
        "promotion_publication_failed",
        "promotion_snapshot_stale",
    }
)
# Zero is a maintenance command with expected pending work; one is legacy data.
PENDING_EXIT_CODES = (0, 1, EXIT_NO_FOLDS)
# Private, invocation-local transport for the standalone maintenance wrapper.
COMMAND_OUTCOME_PATH = "_GWF_COMMAND_OUTCOME_PATH"


@dataclass(frozen=True, slots=True)
class ComponentDiagnostic:
    stage: str
    exit_code: int
    failure_kind: str | None = None
    detail: str | None = None
    path: str | None = None
    recorded_at: str | None = None
    check_id: str | None = None
    audit_path: str | None = None
    issue_count: int | None = None
    omitted_count: int | None = None
    audit_error: str | None = None
    dry_run: bool | None = None
    category: str | None = None
    reason_code: str | None = None
    audit_candidate_path: str | None = None
    audit_reason_code: str | None = None
    audit_cause_code: str | None = None
    operation: str | None = None
    deleted_audit_count: int | None = None
    deleted_staging_count: int | None = None
    eligible_audit_count: int | None = None
    eligible_staging_count: int | None = None
    cleanup_completed: bool | None = None
    error_type: str | None = None
    progress_id: str | None = None


_progress: ContextVar[list[ComponentDiagnostic] | None] = ContextVar(
    "command_progress", default=None
)


@contextmanager
def diagnostic_progress() -> Iterator[list[ComponentDiagnostic]]:
    """Isolate typed partial results to one invocation or embedded stage."""
    progress: list[ComponentDiagnostic] = []
    token = _progress.set(progress)
    try:
        yield progress
    finally:
        _progress.reset(token)


def record_progress(diagnostics: tuple[ComponentDiagnostic, ...]) -> None:
    if (progress := _progress.get()) is not None:
        progress[:] = merge_diagnostics(tuple(progress), diagnostics)


@dataclass(frozen=True, slots=True)
class DiagnosticIdentity:
    """Validated ownership and independent hints for quarantining damaged data."""

    owner: str | None
    valid: bool
    progress_id: str | None
    check_id: str | None


def _identity_hint(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def assess_progress_identity(
    progress_id: object, check_id: object
) -> DiagnosticIdentity:
    """Inspect optional recorded metadata without giving malformed data ownership."""
    progress, check = _identity_hint(progress_id), _identity_hint(check_id)
    valid = (progress_id is None or progress is not None) and (
        check_id is None or check is not None
    )
    return DiagnosticIdentity(
        progress or check if valid else None, valid, progress, check
    )


def effective_progress_id(progress_id: object, check_id: object) -> str | None:
    """Resolve one owner consistently; malformed recorded identities are not legacy."""
    identity = assess_progress_identity(progress_id, check_id)
    if not identity.valid:
        raise ValueError("diagnostic identity must be a nonempty string or null")
    return identity.owner


_RETENTION_STAGES = frozenset({"retention_verification", "retention_audit_cleanup"})
type BatchKey = tuple[str, str]


@dataclass(slots=True)
class _DiagnosticBatch:
    key: BatchKey | None
    identity_valid: bool
    items: list[ComponentDiagnostic]


def _diagnostic_batches(
    incoming: tuple[ComponentDiagnostic, ...],
    identities: dict[int, DiagnosticIdentity],
) -> list[_DiagnosticBatch]:
    batches: list[_DiagnosticBatch] = []
    active: dict[BatchKey, _DiagnosticBatch] = {}
    for item in incoming:
        if id(item) not in identities:
            identities[id(item)] = assess_progress_identity(
                item.progress_id, item.check_id
            )
        identity = identities[id(item)]
        key = (
            (item.stage, identity.owner)
            if item.stage in _RETENTION_STAGES and identity.owner is not None
            else None
        )
        summary = (
            type(item.issue_count) is int
            and item.issue_count >= 0
            and item.category is None
        )
        if key is None:
            batches.append(_DiagnosticBatch(None, identity.valid, [item]))
        else:
            if key not in active or summary:
                active[key] = _DiagnosticBatch(key, identity_valid=True, items=[])
                batches.append(active[key])
            active[key].items.append(item)
    return batches


def record_verification_progress(diagnostics: tuple[ComponentDiagnostic, ...]) -> None:
    """Compatibility entry point sharing atomic replacement with final merging."""
    record_progress(diagnostics)


def merge_diagnostics(
    *batches: tuple[ComponentDiagnostic, ...],
) -> tuple[ComponentDiagnostic, ...]:
    identities: dict[int, DiagnosticIdentity] = {}
    ordered: list[_DiagnosticBatch | None] = []
    positions: dict[BatchKey, int] = {}
    for batch in batches:
        for incoming in _diagnostic_batches(batch, identities):
            if incoming.key is not None:
                if incoming.key in positions:
                    ordered[positions[incoming.key]] = None
                positions[incoming.key] = len(ordered)
            ordered.append(incoming)
    seen: set[ComponentDiagnostic] = set()
    result: list[ComponentDiagnostic] = []
    for group in ordered:
        if group is None:
            continue
        for item in group.items:
            if group.key is not None or not group.identity_valid:
                result.append(item)
            elif item not in seen:
                result.append(item)
                seen.add(item)
    return tuple(result)


def _serialized_identity(value: object) -> object:
    """Bound invalid values, retaining an unmistakably invalid JSON object."""
    if value is None or _identity_hint(value) is not None:
        return value
    from grounded_weather_forecast.runs import normalize_error_detail  # noqa: PLC0415

    try:
        detail = repr(value)
    except (Exception,):  # noqa: B013 -- optional evidence must not erase a run
        detail = "Identity value could not be represented."
    return {
        "invalid_identity_type": type(value).__name__,
        "detail": normalize_error_detail(detail),
    }


def _diagnostic_payload(item: ComponentDiagnostic) -> dict[str, object]:
    from grounded_weather_forecast.runs import normalize_error_detail  # noqa: PLC0415

    identity = assess_progress_identity(item.progress_id, item.check_id)
    payload: dict[str, object] = {
        field.name: getattr(item, field.name)
        for field in fields(item)
        if getattr(item, field.name) is not None
        or field.name in {"stage", "exit_code", "failure_kind", "detail"}
    }
    for key in ("progress_id", "check_id"):
        if key in payload:
            payload[key] = _serialized_identity(payload[key])
    if identity.valid and identity.progress_id == identity.check_id:
        payload.pop("progress_id", None)
    payload["detail"] = normalize_error_detail(item.detail)
    return payload


def failure_details_json(diagnostics: tuple[ComponentDiagnostic, ...]) -> str | None:
    """Bound each component independently so a second error cannot disappear."""
    if not diagnostics:
        return None
    return json.dumps([_diagnostic_payload(item) for item in diagnostics])


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    """Internal command result; the CLI boundary exposes only exit_code."""

    exit_code: int = 0
    failure_kind: str | None = None
    error_detail: str | None = None
    diagnostics: tuple[ComponentDiagnostic, ...] = ()

    @classmethod
    def normalize(cls, value: "CommandOutcome | int") -> "CommandOutcome":
        return value if isinstance(value, CommandOutcome) else cls(value)


def write_outcome_hint(outcome: CommandOutcome) -> None:
    """Best-effort child classification; the process exit remains authoritative."""
    if path := os.environ.get(COMMAND_OUTCOME_PATH):
        try:
            Path(path).write_text(
                json.dumps(
                    {
                        "exit_code": outcome.exit_code,
                        "failure_kind": outcome.failure_kind,
                    }
                ),
                encoding="utf-8",
            )
        except (OSError,):  # noqa: B013 -- hints never change the command result
            return


def read_outcome_hint(path: Path, exit_code: int) -> CommandOutcome:
    """Absent, damaged or mismatched hints cannot grant a continuation exemption."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if (
            isinstance(raw, dict)
            and type(raw.get("exit_code")) is int
            and raw["exit_code"] == exit_code
            and isinstance(kind := raw.get("failure_kind"), str)
        ):
            return CommandOutcome(exit_code, kind)
    except (OSError, ValueError):
        pass
    return CommandOutcome(exit_code)
