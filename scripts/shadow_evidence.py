"""Local evidence checks; hashes detect inconsistency, not same-user forgery."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import polars as pl

from grounded_weather_forecast.shadow_artifacts import (
    ARTIFACT_VERSION,
    BINDING,
    TARGET,
    EvidenceError,
    current_artifact,
    digest,
    forecast_rows,
    hex_identity,
    issuance_name,
    json_object,
    object_value,
    required_string,
    sha256,
    sync_directory,
    timestamp,
)


def regular(path: Path) -> None:
    if path.is_symlink() or path.is_junction():
        raise EvidenceError(f"{path}: evidence must be a regular local file")
    if not stat.S_ISREG(path.stat().st_mode):
        raise EvidenceError(f"{path}: evidence must be a regular local file")


def capsule_for(root: Path, origin: datetime) -> Path:
    path = root / issuance_name(origin)
    if (
        path.is_symlink()
        or path.is_junction()
        or not path.is_dir()
        or path.resolve().parent != root.resolve()
    ):
        raise EvidenceError("capsule must remain within the selected capsule root")
    return path


def publication(value: object) -> tuple[bytes, dict[str, Any]]:
    receipt = object_value(value, "publisher receipt")
    if (
        type(receipt.get("schema_version")) is not int
        or receipt.get("schema_version") != 1
        or receipt.get("commit_binding") != BINDING
        or type(receipt.get("public_schema_version")) is not int
        or receipt.get("public_schema_version") != 2
    ):
        raise EvidenceError(
            "unsupported publisher receipt; an exact-push receipt is required"
        )
    hex_identity(receipt.get("commit"), "publisher receipt.commit", 40)
    fields = (
        "remote",
        "branch",
        "publisher_version",
        "source_sha256",
        "forecast_sha256",
        "issued_at",
        "published_at",
        "forecast_base64",
    )
    for key in fields:
        required_string(receipt.get(key), f"publisher receipt.{key}")
    for key in ("source_sha256", "forecast_sha256"):
        hex_identity(receipt[key], f"publisher receipt.{key}", 64)
    try:
        payload = base64.b64decode(receipt["forecast_base64"], validate=True)
    except ValueError as error:
        raise EvidenceError("invalid publisher receipt payload") from error
    document = json_object(payload, "publisher receipt payload")
    if (
        hashlib.sha256(payload).hexdigest() != receipt.get("forecast_sha256")
        or document.get("issued_at") != receipt.get("issued_at")
        or document.get("published_at") != receipt.get("published_at")
        or document.get("publisher_version") != receipt.get("publisher_version")
        or type(document.get("schema_version")) is not int
        or document.get("schema_version") != 2
    ):
        raise EvidenceError("publisher receipt payload mismatch")
    timestamp(receipt["issued_at"])
    timestamp(receipt["published_at"])
    forecast_rows(document, public=True)
    return payload, document


def assert_equivalent(actual: Any, expected: Any) -> None:
    """Reproduction tolerance is numerical, never a substitute for byte digests."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise EvidenceError("record reproduction differs")
        for key, value in expected.items():
            assert_equivalent(actual[key], value)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise EvidenceError("record reproduction differs")
        for left, right in zip(actual, expected, strict=True):
            assert_equivalent(left, right)
    elif isinstance(expected, float):
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (float, int))
            or not math.isfinite(actual)
            or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)
        ):
            raise EvidenceError("record reproduction differs")
    elif type(actual) is not type(expected) or actual != expected:
        raise EvidenceError("record reproduction differs")


def mature(protocol: dict[str, Any], now: datetime) -> None:
    if now < timestamp(protocol["window_end"]) + timedelta(hours=98):
        raise EvidenceError(
            "prospective truth and evaluation require window_end +98 hours"
        )


def truth_frame(path: Path | bytes) -> pl.DataFrame:
    if isinstance(path, Path):
        regular(path)
        payload = path.read_bytes()
    else:
        payload = path
    try:
        frame = pl.read_parquet(BytesIO(payload))
    except pl.exceptions.PolarsError as error:
        raise EvidenceError("truth: invalid Parquet input") from error
    if not {"valid_hour", TARGET} <= set(frame.columns):
        raise EvidenceError("truth requires QC interval-mean temperature labels")
    dtype = frame.schema["valid_hour"]
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None:
        raise EvidenceError("truth hours must include timezone")
    # Check nanosecond residues before casting so sub-hour evidence cannot round
    # into a valid hour during normalization.
    if frame.filter(pl.col("valid_hour").dt.epoch("ns") % (3600 * 10**9) != 0).height:
        raise EvidenceError("truth must contain unique nonempty exact hours")
    frame = frame.with_columns(
        pl.col("valid_hour").dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC"))
    )
    times = frame["valid_hour"]
    if (
        not len(times)
        or times.null_count()
        or times.n_unique() != len(times)
        or any(t.minute or t.second or t.microsecond for t in times)
    ):
        raise EvidenceError("truth must contain unique nonempty exact hours")
    if (
        not frame.schema[TARGET].is_numeric()
        or frame.filter(pl.col(TARGET).is_infinite() | pl.col(TARGET).is_nan()).height
    ):
        raise EvidenceError("truth must contain numeric finite-or-null QC labels")
    return frame


@contextmanager
def staged_directory(output: Path) -> Iterator[Path]:
    """Reserve a sibling lease; publish a complete directory only after success."""
    output.parent.mkdir(parents=True, exist_ok=True)
    lease = output.with_name(f".{output.name}.publishing")
    fd = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temporary = None
    try:
        os.close(fd)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        temporary = Path(tempfile.mkdtemp(prefix="shadow-stage-", dir=output.parent))
        folder = temporary / "complete"
        folder.mkdir(mode=0o700)
        yield folder
        for path in folder.iterdir():
            regular(path)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        sync_directory(folder)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        folder.rename(output)
        sync_directory(output.parent)
    finally:
        # Cleanup cannot turn a published complete result into a reported failure.
        with suppress(OSError):
            lease.unlink(missing_ok=True)
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def freeze_truth(
    truth: Path,
    protocol_path: Path,
    output: Path,
    now: datetime,
    *,
    protocol_bytes: bytes | None = None,
    enrollment_bytes: bytes | None = None,
) -> Path:
    protocol_bytes = (
        protocol_path.read_bytes() if protocol_bytes is None else protocol_bytes
    )
    enrollment_bytes = (
        protocol_path.with_suffix(".lock.json").read_bytes()
        if enrollment_bytes is None
        else enrollment_bytes
    )
    protocol = json_object(protocol_bytes, "protocol")
    mature(protocol, now)
    regular(truth)
    payload = truth.read_bytes()
    frame = truth_frame(payload)
    # Validate the copied bytes, so the lock describes exactly the data scored.
    with staged_directory(output) as folder:
        target = folder / "truth.parquet"
        target.write_bytes(payload)
        lock = {
            "artifact_version": ARTIFACT_VERSION,
            "protocol_sha256": sha256(protocol_bytes),
            "enrollment_sha256": sha256(enrollment_bytes),
            "truth_sha256": sha256(payload),
            "sealed_at": now.isoformat(),
            "target": TARGET,
            "rows": frame.height,
            "first_hour": cast(datetime, frame["valid_hour"].min()).isoformat(),
            "last_hour": cast(datetime, frame["valid_hour"].max()).isoformat(),
            "qc": "operator-reviewed QC input; sealing does not independently certify QC",
        }
        (folder / "truth.lock.json").write_text(json.dumps(lock, indent=2))
    return output


def locked_truth(
    root: Path,
    protocol: Path,
    replacement: Path | None,
    now: datetime,
    *,
    protocol_bytes: bytes | None = None,
    enrollment_bytes: bytes | None = None,
) -> tuple[bytes, bytes]:
    if root.is_symlink() or root.is_junction() or not root.is_dir():
        raise EvidenceError("truth lock must be a local directory")
    lock_path, truth = root / "truth.lock.json", root / "truth.parquet"
    regular(lock_path)
    regular(truth)
    truth_bytes, lock_bytes = truth.read_bytes(), lock_path.read_bytes()
    frame = truth_frame(truth_bytes)
    lock = json_object(lock_bytes, "truth lock")
    protocol_bytes = protocol.read_bytes() if protocol_bytes is None else protocol_bytes
    enrollment_bytes = (
        protocol.with_suffix(".lock.json").read_bytes()
        if enrollment_bytes is None
        else enrollment_bytes
    )
    mature(
        json_object(protocol_bytes, "protocol"),
        timestamp(required_string(lock.get("sealed_at"), "truth lock.sealed_at")),
    )
    if (
        not current_artifact(lock.get("artifact_version"))
        or lock.get("protocol_sha256") != sha256(protocol_bytes)
        or lock.get("enrollment_sha256") != sha256(enrollment_bytes)
        or lock.get("truth_sha256") != sha256(truth_bytes)
        or lock.get("target") != TARGET
        or lock.get("rows") != frame.height
        or lock.get("first_hour")
        != cast(datetime, frame["valid_hour"].min()).isoformat()
        or lock.get("last_hour")
        != cast(datetime, frame["valid_hour"].max()).isoformat()
        or timestamp(required_string(lock.get("sealed_at"), "truth lock.sealed_at"))
        > now
    ):
        raise EvidenceError("frozen truth identity differs")
    if replacement is not None:
        regular(replacement)
        if digest(replacement) != lock["truth_sha256"]:
            raise EvidenceError("supplied truth differs from the frozen truth")
    return truth_bytes, lock_bytes
