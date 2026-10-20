"""Read the immutable build clock only when it describes the evaluated inputs."""

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from grounded_weather_forecast.dataset.matrix import _file_digest, matrix_path


def evaluation_snapshot(
    directory: Path, product: str, source: str, *, truth_minute: bool = False
) -> datetime | None:
    """Legacy matrices without metadata retain the historical outcome rules.

    Callers hold the pipeline lock while loading inputs and this metadata.
    Present but invalid metadata is an input error, never a pending exemption.
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
        moment = datetime.fromisoformat(raw["created_at"])
        if moment.tzinfo is None:
            raise ValueError("snapshot timestamp must include a timezone")
        inputs = {f"{product}_matrix": matrix_path(directory, product, source)}
        if truth_minute:
            inputs["truth_minute"] = directory / "truth_minute.parquet"
        for name, matrix in inputs.items():
            if _file_digest(matrix) != raw["files"][name]["sha256_16"]:
                raise ValueError(f"{name} differs from snapshot manifest")
        latest = (
            pl.scan_parquet(inputs[f"{product}_matrix"])
            .select(pl.col("issue_time").max())
            .collect()
            .item()
        )
        if not isinstance(latest, datetime) or latest.tzinfo is None or latest > moment:
            raise ValueError("snapshot timestamp precedes the input issue clock")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid evaluation snapshot {path}: {exc}") from exc
    return moment.astimezone(UTC)
