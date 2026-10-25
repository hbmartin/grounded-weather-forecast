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


def diagnostic_label(detail: str, path: str | None, config: Config) -> str:
    explanation = public_detail(detail, path)
    if path:
        return f"{artifact_label(Path(path), config)}: {explanation}"
    return explanation


def audit_label(reference: str, config: Config) -> str:
    return artifact_label(config.artifacts_dir / reference, config)
