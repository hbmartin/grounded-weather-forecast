"""Public artifact labels without rewriting prose or resolving filesystem links."""

import os
from pathlib import Path

from grounded_weather_forecast.config import Config
from grounded_weather_forecast.verification_text import public_detail


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
    explanation = public_detail(
        detail, path, category=category, reason_code=reason_code
    )
    if path:
        return f"{artifact_label(Path(path), config)}: {explanation}"
    return explanation


def audit_label(reference: str, config: Config) -> str:
    return artifact_label(config.artifacts_dir / reference, config)


class CompatibilityNote(str):
    """A fixed explanation with a formatted artifact label; legacy str API retained."""

    __slots__ = ()

    def __new__(cls, path: Path, config: Config) -> "CompatibilityNote":
        return super().__new__(
            cls,
            f"{artifact_label(path, config)}: "
            + public_detail("", category="legacy_identity_missing"),
        )

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (_restore_compatibility_note, (str(self),))


def _restore_compatibility_note(value: str) -> CompatibilityNote:
    # Python object reconstruction preserves the already formatted trusted value;
    # JSON/legacy strings still take the neutral public rendering path.
    return str.__new__(CompatibilityNote, value)
