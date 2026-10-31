"""Exit-status protocol shared by CLI commands and installed job wrappers."""

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import cast

from grounded_weather_forecast.diagnostic_values import (
    actual_integer,
    actual_string,
    declared_types,
    field_valid,
    primitive,
)

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
VERIFICATION_VERDICT_DETAILS = {
    "retention_clear": "published-score protection verified",
    "retention_suspended": "published-score deletion suspended: protection could not be verified",
}


def is_verification_verdict(value: object) -> bool:
    kind = actual_string(value)
    return kind is not None and kind in VERIFICATION_VERDICT_DETAILS


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
    issue_count_known: bool | None = None


_FIELD_TYPES = declared_types(ComponentDiagnostic)


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
        try:
            progress[:] = merge_diagnostics(tuple(progress), diagnostics)
        except (Exception,):  # noqa: B013 -- optional telemetry cannot stop work
            progress.extend((diagnostic_damage(),))


@dataclass(frozen=True, slots=True)
class DiagnosticIdentity:
    """Validated ownership and independent hints for quarantining damaged data."""

    owner: str | None
    valid: bool
    progress_id: str | None
    check_id: str | None


def _identity_hint(value: object) -> str | None:
    text = actual_string(value)
    return text if text is not None and text.strip() else None


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


def diagnostic_damage() -> ComponentDiagnostic:
    return ComponentDiagnostic(
        "diagnostic_metadata",
        0,
        "invalid_diagnostic_metadata",
        "Optional diagnostic evidence could not be processed.",
    )


def _field_valid(name: str, value: object) -> bool:
    return field_valid(name, value, _FIELD_TYPES)


def _primitive(value: object) -> object:
    """Compatibility projection using the same real-subclass rules as validation."""
    return primitive(value)


def _exact_value(value: object, ancestors: tuple[int, ...] = ()) -> object:
    """Hash only built-ins; opaque/cyclic data deduplicates by object, not preview."""
    kind = type(value)
    if kind in (str, int, bool, bytes, type(None)):
        return (kind.__name__, value)
    if kind is float:
        return ("float", repr(value))
    if id(value) in ancestors or len(ancestors) >= 20:
        return ("object", id(value))
    parents = (*ancestors, id(value))
    if kind in (tuple, list):
        return (
            kind.__name__,
            tuple(
                _exact_value(part, parents)
                for part in cast(list[object] | tuple[object, ...], value)
            ),
        )
    if kind is dict:
        return (
            "dict",
            tuple(
                (_exact_value(key, parents), _exact_value(part, parents))
                for key, part in cast(dict[object, object], value).items()
            ),
        )
    return ("object", id(value))


def _diagnostic_batches(
    incoming: tuple[ComponentDiagnostic, ...],
    identities: dict[int, DiagnosticIdentity],
) -> Iterator[tuple[ComponentDiagnostic, BatchKey | None, bool]]:
    for item in incoming:
        if not issubclass(type(item), ComponentDiagnostic):
            yield diagnostic_damage(), None, False
            continue
        if id(item) not in identities:
            identities[id(item)] = assess_progress_identity(
                item.progress_id, item.check_id
            )
        identity = identities[id(item)]
        stage = actual_string(item.stage)
        key = (
            (stage, identity.owner)
            if stage is not None
            and stage in _RETENTION_STAGES
            and identity.owner is not None
            else None
        )
        count = _primitive(item.issue_count)
        summary = type(count) is int and count >= 0 and item.category is None
        yield item, key, summary


def record_verification_progress(diagnostics: tuple[ComponentDiagnostic, ...]) -> None:
    """Compatibility entry point sharing atomic replacement with final merging."""
    record_progress(diagnostics)


def merge_diagnostics(
    *batches: tuple[ComponentDiagnostic, ...],
) -> tuple[ComponentDiagnostic, ...]:
    identities: dict[int, DiagnosticIdentity] = {}
    ordered: list[ComponentDiagnostic | None] = []
    positions: dict[BatchKey, list[int]] = {}
    seen: set[object] = set()
    for batch in batches:
        active: set[BatchKey] = set()
        for item, key, summary in _diagnostic_batches(batch, identities):
            if key is not None:
                if key not in active or summary:
                    for index in positions.get(key, ()):
                        ordered[index] = None
                    positions[key] = []
                    active.add(key)
                positions[key].append(len(ordered))
            else:
                exact = tuple(
                    _exact_value(getattr(item, field.name)) for field in fields(item)
                )
                if exact in seen:
                    continue
                seen.add(exact)
            ordered.append(item)
    return tuple(item for item in ordered if item is not None)


def _serialized_identity(value: object) -> object:
    """Bound invalid values, retaining an unmistakably invalid JSON object."""
    if value is None:
        return None
    if (text := _identity_hint(value)) is not None:
        return text
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

    assessed: dict[str, tuple[object, bool]] = {}
    schema = declared_types(type(item))
    for field in fields(item):
        value = getattr(item, field.name)
        valid = (
            _field_valid(field.name, value)
            if type(item) is ComponentDiagnostic
            else field_valid(field.name, value, schema)
        )
        assessed[field.name] = (value, valid)
    identity = assess_progress_identity(
        assessed["progress_id"][0], assessed["check_id"][0]
    )
    payload: dict[str, object] = {}
    for name, (value, valid) in assessed.items():
        if value is not None or name in {
            "stage",
            "exit_code",
            "failure_kind",
            "detail",
        }:
            payload[name] = _primitive(value) if valid else _invalid_field(value)
    for key in ("progress_id", "check_id"):
        if key in payload:
            payload[key] = _serialized_identity(assessed[key][0])
    if identity.valid and identity.progress_id == identity.check_id:
        payload.pop("progress_id", None)
    detail, valid_detail = assessed["detail"]
    if valid_detail:
        payload["detail"] = normalize_error_detail(_primitive(detail))
    if not all(valid for _, valid in assessed.values()):
        payload["metadata_invalid"] = True
    return payload


def _invalid_field(value: object) -> dict[str, object]:
    from grounded_weather_forecast.runs import normalize_error_detail  # noqa: PLC0415

    try:
        detail = normalize_error_detail(repr(value))
    except (Exception,):  # noqa: B013 -- hostile optional values stay untrusted
        detail = "Value could not be represented."
    return {"invalid_field_type": type(value).__name__, "detail": detail}


def failure_details_json(diagnostics: tuple[ComponentDiagnostic, ...]) -> str | None:
    """Bound each component independently so a second error cannot disappear."""
    if not diagnostics:
        return None
    payloads: list[dict[str, object]] = []
    for item in diagnostics:
        try:
            payloads.append(_diagnostic_payload(item))
        except (Exception,):  # noqa: B013 -- preserve other components and core row
            payloads.append(_diagnostic_payload(diagnostic_damage()))
    return json.dumps(payloads)


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    """Internal command result; the CLI boundary exposes only exit_code."""

    exit_code: int = 0
    failure_kind: str | None = None
    error_detail: str | None = None
    diagnostics: tuple[ComponentDiagnostic, ...] = ()

    @classmethod
    def normalize(cls, value: "CommandOutcome | int") -> "CommandOutcome":
        outcome = (
            cast(CommandOutcome, value)
            if issubclass(type(value), cls)
            else cls(cast(int, value))
        )
        code = actual_integer(outcome.exit_code)
        damage: tuple[ComponentDiagnostic, ...] = ()
        kind = actual_string(outcome.failure_kind)
        detail = actual_string(outcome.error_detail)
        if outcome.failure_kind is not None and kind is None:
            damage += (invalid_metadata("failure_kind", outcome.failure_kind),)
        if code is None or not -(2**63) <= code < 2**63:
            damage += (invalid_metadata("exit_code", outcome.exit_code),)
            code, kind = 1, "invalid_command_outcome"
        if outcome.error_detail is not None and detail is None:
            damage += (invalid_metadata("error_detail", outcome.error_detail),)
        if (
            type(outcome.exit_code) is int
            and code == outcome.exit_code
            and (outcome.failure_kind is None or type(outcome.failure_kind) is str)
            and (outcome.error_detail is None or type(outcome.error_detail) is str)
            and not damage
        ):
            return outcome
        return cls(code, kind, detail, _with_damage(outcome.diagnostics, damage))


def _with_damage(
    diagnostics: tuple[ComponentDiagnostic, ...],
    damage: tuple[ComponentDiagnostic, ...],
) -> tuple[ComponentDiagnostic, ...]:
    try:
        return (*diagnostics, *damage)
    except (Exception,):  # noqa: B013 -- optional evidence cannot replace an outcome
        return (*damage, diagnostic_damage())


def invalid_metadata(field: str, value: object) -> ComponentDiagnostic:
    """Share the safe damage identity while keeping bounded offending evidence."""
    from grounded_weather_forecast.runs import normalize_error_detail  # noqa: PLC0415

    detail = _invalid_field(value)
    return replace(
        diagnostic_damage(),
        detail=normalize_error_detail(
            f"{field}: {detail['invalid_field_type']}: {detail['detail']}"
        ),
    )


def write_outcome_hint(outcome: CommandOutcome) -> None:
    """Best-effort child classification; the process exit remains authoritative."""
    try:
        if path := os.environ.get(COMMAND_OUTCOME_PATH):
            normalized = CommandOutcome.normalize(outcome)
            if normalized is not outcome:
                record_progress(normalized.diagnostics)
            Path(path).write_text(
                json.dumps(
                    {
                        "exit_code": normalized.exit_code,
                        "failure_kind": normalized.failure_kind,
                    }
                ),
                encoding="utf-8",
            )
    except (Exception,):  # noqa: B013 -- optional hints never change command results
        record_progress((diagnostic_damage(),))
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
