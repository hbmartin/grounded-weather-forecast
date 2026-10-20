"""Exit-status protocol shared by CLI commands and installed job wrappers."""

from dataclasses import dataclass

EXIT_NO_FOLDS = 3
# Zero is a maintenance command with expected pending work; one is legacy data.
PENDING_EXIT_CODES = (0, 1, EXIT_NO_FOLDS)


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    """Internal command result; the CLI boundary exposes only exit_code."""

    exit_code: int = 0
    failure_kind: str | None = None
    error_detail: str | None = None

    @classmethod
    def normalize(cls, value: "CommandOutcome | int") -> "CommandOutcome":
        return value if isinstance(value, CommandOutcome) else cls(value)
