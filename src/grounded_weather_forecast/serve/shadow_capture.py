"""Opt-in export of the exact inputs used by an issued forecast.

No fitting, collection, publication, or enrollment happens here. The exporter
is disabled unless prediction receives an explicit capture directory.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from grounded_weather_forecast.config import Config, config_fingerprint
from grounded_weather_forecast.evaluation import code_identity
from grounded_weather_forecast.serve.schema import Forecast
from grounded_weather_forecast.shadow_artifacts import (
    ARTIFACT_VERSION,
    existing_directory,
    issuance_name,
    private_directory,
    sync_directory,
    sync_files,
)


def temperature_frame(frame: pl.DataFrame, *, training: bool) -> pl.DataFrame:
    columns = [
        "issue_time",
        "valid_time",
        "lead_hours",
        "lead_bucket",
        "source_kind",
        "valid_hour_local",
        "obs__temp_c",
        *(c for c in frame.columns if c.startswith("age__")),
        *(c for c in frame.columns if c.startswith("fx__") and c.endswith("__temp_c")),
    ]
    if training:
        columns.append("t__temp_c__mean")
    return frame.select([c for c in columns if c in frame.columns])


def capture_issue(
    root: Path,
    config: Config,
    document: Forecast,
    *,
    issue: pl.DataFrame,
    training: pl.DataFrame,
    truth: pl.DataFrame,
    completions: pl.DataFrame,
    historical: bool,
) -> Path:
    """Publish an exclusive local capsule, retaining only temperature inputs."""
    if issue.is_empty() or not document.hourly:
        raise ValueError("temperature shadow requires hourly inputs")
    if any(point.truth_semantics.get("temp_c") != "mean" for point in document.hourly):
        raise ValueError(
            "temperature shadow requires production interval-mean semantics"
        )
    private_directory(root)
    if not historical:
        existing_directory(root, private=True)
    issued_at = datetime.fromisoformat(document.issued_at).astimezone(UTC)
    target = root / issuance_name(issued_at)
    if target.is_symlink() or target.is_junction():
        existing_directory(target, private=not historical)
    if target.exists():
        if not historical:
            existing_directory(target, private=True)
        raise FileExistsError(f"shadow inputs already captured: {target}")
    inputs = temperature_frame(issue, training=False)
    if any(c.startswith("t__") for c in inputs.columns):
        raise ValueError("shadow issue inputs must not contain truth labels")
    with TemporaryDirectory(prefix="shadow-input-", dir=root) as temporary:
        folder = Path(temporary)
        inputs.write_parquet(folder / "issue.parquet")
        temperature_frame(training, training=True).write_parquet(
            folder / "training.parquet"
        )
        truth.select("valid_hour", "t__temp_c__mean").write_parquet(
            folder / "truth_hourly.parquet"
        )
        completions.write_parquet(folder / "provider_completions.parquet")
        (folder / "production.json").write_text(document.to_json(), encoding="utf-8")
        provenance = {
            "production_code_identity": code_identity(),
            "config_fingerprint": config_fingerprint(config),
            "dataset_fingerprint": document.dataset_fingerprint,
            "availability_assumptions": "hourly truth known at valid time +2h; station collection ts is an availability proxy; no independent receipt/commit ledger is assumed",
            "provider_completion_clock": "provider_completions.parquet retains actual archive run-completion clocks visible in the serving read, including clocks after rounded issuance; age__ columns retain selected fetch ages; native initialization and per-source commit receipts are not inferred",
            "station_receipt_clock": "not separately recorded; observation_at is the collected observation clock, not a certified receipt/commit timestamp",
        }
        manifest = {
            "artifact_version": ARTIFACT_VERSION,
            "input_as_of": document.issued_at,
            "capture_kind": "historical_reconstruction"
            if historical
            else "serving_snapshot",
            "capsule_exported_at": datetime.now(UTC).isoformat(),
            "provenance": provenance,
            "input_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(folder.iterdir())
            },
        }
        (folder / "capsule.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        sync_files(folder, folder.iterdir())
        # The forecast pipeline serializes capture; reject existing issues and
        # never replace a non-empty capsule directory.
        folder.rename(target)
        sync_directory(root)
    return target
