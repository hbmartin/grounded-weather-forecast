"""Read the immutable build clock only when it describes the evaluated inputs."""

import json
from datetime import UTC, datetime
from pathlib import Path

from grounded_weather_forecast.dataset.matrix import _file_digest, matrix_path


def evaluation_snapshot(
    directory: Path, product: str, source: str, *, truth_minute: bool = False
) -> datetime | None:
    """Legacy matrices without metadata retain the historical outcome rules.

    Callers hold the pipeline lock while loading inputs and this metadata.
    Invalid input integrity is an error. Missing or invalid maturity-only clocks
    cannot grant awaiting_truth, but never prevent ordinary scoring.
    """
    path = directory / (
        "manifest.json" if source == "live" else "manifest_synthetic.json"
    )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError,):  # noqa: B013 -- project convention
        try:
            path.lstat()
        except (FileNotFoundError,):  # noqa: B013 -- project convention
            return None
        raise ValueError(f"snapshot manifest is a dangling link: {path}") from None
    try:
        inputs = {f"{product}_matrix": matrix_path(directory, product, source)}
        if truth_minute:
            inputs["truth_minute"] = directory / "truth_minute.parquet"
        for name, matrix in inputs.items():
            if _file_digest(matrix) != raw["files"][name]["sha256_16"]:
                raise ValueError(f"{name} differs from snapshot manifest")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid evaluation snapshot {path}: {exc}") from exc
    return _truth_cutoff(raw)


def _truth_cutoff(raw: dict[str, object]) -> datetime | None:
    """Publication time and virtual issue clocks cannot establish truth maturity."""
    try:
        moment = datetime.fromisoformat(str(raw["truth_snapshot_at"]))
        published = datetime.fromisoformat(str(raw["created_at"]))
        if moment.tzinfo is None or published.tzinfo is None or moment > published:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return moment.astimezone(UTC)
