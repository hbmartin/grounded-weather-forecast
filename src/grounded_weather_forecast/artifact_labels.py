"""Public artifact labels without rewriting prose or resolving filesystem links."""

import os
from dataclasses import dataclass
from pathlib import Path

from grounded_weather_forecast.config import Config
from grounded_weather_forecast.verification_text import (
    VerificationIssue,
    public_explanation,
)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))  # noqa: PTH100 -- lexical normalization must not follow symlinks


def artifact_label(path: Path, config: Config) -> str:
    target = _absolute(path)
    roots = [
        (config.artifacts_dir, "artifacts"),
        (config.dataset.dir, "dataset"),
        (config.predict.history_path.parent, "predict"),
    ]
    ordered = sorted(
        ((_absolute(root), label) for root, label in roots),
        key=lambda item: len(item[0].parts),
        reverse=True,
    )
    for root, label in ordered:
        if target.is_relative_to(root):
            return (Path(label) / target.relative_to(root)).as_posix()
    return path.name


def diagnostic_label(
    detail: str,
    path: str | None,
    config: Config,
    *,
    category: str | None = None,
    reason_code: str | None = None,
) -> str:
    return issue_label(
        VerificationIssue(
            Path(path) if path else None,
            category or "verification_invalid",
            detail,
            reason_code,
        ),
        config,
    )


def issue_label(issue: VerificationIssue, config: Config) -> str:
    explanation = public_explanation(
        category=issue.category, reason_code=issue.reason_code
    )
    return _label_explanation(issue.path, explanation, config)


def _label_explanation(path: Path | None, explanation: str, config: Config) -> str:
    return (
        f"{artifact_label(path, config)}: {explanation}"
        if path is not None
        else explanation
    )


def audit_label(reference: str, config: Config) -> str:
    return artifact_label(config.artifacts_dir / reference, config)


@dataclass(frozen=True, slots=True)
class CompatibilityNotice:
    path: Path | None = None
    category: str = "legacy_identity_missing"

    def label(self, config: Config) -> str:
        if self.category != "legacy_identity_missing":
            return "Legacy compatibility information is available; see diagnostic evidence."
        explanation = public_explanation(category=self.category)
        return _label_explanation(self.path, explanation, config)


class CompatibilityNote(str):
    """A fixed explanation with a formatted artifact label; legacy str API retained."""

    __slots__ = ()

    def __new__(cls, path: Path, config: Config) -> "CompatibilityNote":
        return super().__new__(
            cls,
            CompatibilityNotice(path).label(config),
        )

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (_restore_compatibility_note, (str(self),))


def _restore_compatibility_note(value: str) -> CompatibilityNote:
    # Object reconstruction preserves the old string API. Renderers use structured
    # notices or fixed informational text rather than trusting this subtype.
    return str.__new__(CompatibilityNote, value)
