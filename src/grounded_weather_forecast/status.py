"""Exit-status protocol shared by CLI commands and installed job wrappers."""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

EXIT_NO_FOLDS = 3
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


def failure_details_json(diagnostics: tuple[ComponentDiagnostic, ...]) -> str | None:
    """Bound each component independently so a second error cannot disappear."""
    if not diagnostics:
        return None
    from grounded_weather_forecast.runs import normalize_error_detail  # noqa: PLC0415

    return json.dumps(
        [
            asdict(item) | {"detail": normalize_error_detail(item.detail)}
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
