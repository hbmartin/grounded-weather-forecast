"""A consistent, in-memory view of the small live serving dataset."""

import hashlib
import io
import json
from dataclasses import dataclass, fields

import polars as pl

from grounded_weather_forecast.config import Config
from grounded_weather_forecast.dataset.matrix import DatasetPaths
from grounded_weather_forecast.storage import dataset_lock


class DatasetIntegrityError(ValueError):
    """The live files do not match their published manifest."""


@dataclass(frozen=True, slots=True)
class LiveDatasetSnapshot:
    fingerprint: str
    hourly: pl.DataFrame
    daily: pl.DataFrame
    truth_minute: pl.DataFrame


def load_live_snapshot(
    config: Config, *, lock_timeout: float | None = None
) -> LiveDatasetSnapshot:
    """Copy and validate under the publication lock, then parse outside it."""
    paths = DatasetPaths.in_dir(config.dataset.dir)
    all_files = {
        field.name: getattr(paths, field.name)
        for field in fields(DatasetPaths)
        if field.name != "manifest"
    }
    with dataset_lock(config.dataset.dir, timeout=lock_timeout):
        if not paths.manifest.exists():
            if any(path.exists() for path in all_files.values()):
                raise DatasetIntegrityError(
                    "live dataset files exist without a manifest"
                )
            return LiveDatasetSnapshot(
                "unknown", pl.DataFrame(), pl.DataFrame(), pl.DataFrame()
            )
        manifest_bytes = paths.manifest.read_bytes()
        try:
            file_bytes = {name: path.read_bytes() for name, path in all_files.items()}
        except OSError as exc:
            raise DatasetIntegrityError(f"live dataset file missing: {exc}") from exc
        try:
            manifest = json.loads(manifest_bytes)
            fingerprint = str(manifest["fingerprint"])
            if set(manifest["files"]) != set(all_files):
                raise DatasetIntegrityError(
                    "live dataset manifest has incomplete file list"
                )
            for name, blob in file_bytes.items():
                expected = str(manifest["files"][name]["sha256_16"])
                actual = hashlib.sha256(blob).hexdigest()[:16]
                if actual != expected:
                    raise DatasetIntegrityError(
                        f"{name} differs from the published manifest"
                    )
            identity = {
                name: [
                    manifest["files"][name]["rows"],
                    manifest["files"][name]["sha256_16"],
                ]
                for name in all_files
            }
            computed = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest()[:16]
            if fingerprint != computed:
                raise DatasetIntegrityError(
                    "live dataset fingerprint differs from manifest files"
                )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, DatasetIntegrityError):
                raise
            raise DatasetIntegrityError(
                f"invalid live dataset manifest: {exc}"
            ) from exc
    try:
        return LiveDatasetSnapshot(
            fingerprint=fingerprint,
            hourly=pl.read_parquet(io.BytesIO(file_bytes["hourly_matrix"])),
            daily=pl.read_parquet(io.BytesIO(file_bytes["daily_matrix"])),
            truth_minute=pl.read_parquet(io.BytesIO(file_bytes["truth_minute"])),
        )
    except pl.exceptions.PolarsError as exc:
        raise DatasetIntegrityError(f"corrupt live dataset Parquet: {exc}") from exc
