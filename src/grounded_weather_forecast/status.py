"""Exit-status protocol shared by CLI commands and installed job wrappers."""

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
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
        progress.extend(diagnostics)


def record_verification_progress(diagnostics: tuple[ComponentDiagnostic, ...]) -> None:
    """Replace one identified check atomically, preserving other partial results."""
    if (progress := _progress.get()) is None:
        return
    identities = {
        item.check_id
        for item in diagnostics
        if item.stage == "retention_verification" and item.check_id is not None
    }
    progress[:] = [
        item
        for item in progress
        if item.stage != "retention_verification" or item.check_id not in identities
    ] + list(diagnostics)


def merge_diagnostics(
    *batches: tuple[ComponentDiagnostic, ...],
) -> tuple[ComponentDiagnostic, ...]:
    return tuple(dict.fromkeys(item for batch in batches for item in batch))


def failure_details_json(diagnostics: tuple[ComponentDiagnostic, ...]) -> str | None:
    """Bound each component independently so a second error cannot disappear."""
    if not diagnostics:
        return None
    from grounded_weather_forecast.runs import normalize_error_detail  # noqa: PLC0415

    return json.dumps(
        [
            {
                key: value
                for key, value in asdict(item).items()
                if key
                not in {
                    "path",
                    "recorded_at",
                    "check_id",
                    "audit_path",
                    "issue_count",
                    "omitted_count",
                    "audit_error",
                    "dry_run",
                    "category",
                    "reason_code",
                    "audit_candidate_path",
                    "audit_reason_code",
                }
                or value is not None
            }
            | {"detail": normalize_error_detail(item.detail)}
            for item in diagnostics
        ]
    )


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
