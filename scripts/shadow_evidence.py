"""Local evidence checks; hashes detect inconsistency, not same-user forgery."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import polars as pl

ARTIFACT_VERSION = 2
TARGET = "t__temp_c__mean"
BINDING = "publisher successful explicit-commit push"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp(text: str) -> datetime:
    value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        raise ValueError("timestamps must include timezone")
    return value.astimezone(UTC)


def regular(path: Path) -> None:
    if path.is_symlink() or path.is_junction() or not path.is_file():
        raise ValueError("evidence must be a regular local file")


def capsule_for(root: Path, origin: datetime) -> Path:
    path = root / origin.strftime("%Y%m%dT%H%M%SZ")
    if (
        path.is_symlink()
        or path.is_junction()
        or not path.is_dir()
        or path.resolve().parent != root.resolve()
    ):
        raise ValueError("capsule must remain within the selected capsule root")
    return path


def publication(receipt: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    if not isinstance(receipt, dict):
        raise ValueError("unsupported publisher receipt; expected an object")
    if (
        type(receipt.get("schema_version")) is not int
        or receipt.get("schema_version") != 1
        or receipt.get("commit_binding") != BINDING
        or type(receipt.get("public_schema_version")) is not int
        or receipt.get("public_schema_version") != 2
    ):
        raise ValueError(
            "unsupported publisher receipt; an exact-push receipt is required"
        )
    commit = receipt.get("commit", "")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(c not in "0123456789abcdef" for c in commit)
    ):
        raise ValueError("publisher receipt requires a full commit identity")
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
    if any(not isinstance(receipt.get(key), str) or not receipt[key] for key in fields):
        raise ValueError("publisher receipt requires nonempty string fields")
    try:
        payload = base64.b64decode(receipt["forecast_base64"], validate=True)
        document = json.loads(payload)
    except (KeyError, TypeError, ValueError, binascii.Error, UnicodeError) as error:
        raise ValueError("invalid publisher receipt payload") from error
    if not isinstance(document, dict):
        raise ValueError("invalid publisher receipt payload; expected an object")
    if (
        hashlib.sha256(payload).hexdigest() != receipt.get("forecast_sha256")
        or document.get("issued_at") != receipt.get("issued_at")
        or document.get("published_at") != receipt.get("published_at")
        or document.get("publisher_version") != receipt.get("publisher_version")
        or document.get("schema_version") != 2
    ):
        raise ValueError("publisher receipt payload mismatch")
    timestamp(receipt["issued_at"])
    timestamp(receipt["published_at"])
    return payload, document


def assert_equivalent(actual: Any, expected: Any) -> None:
    """Reproduction tolerance is numerical, never a substitute for byte digests."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError("record reproduction differs")
        for key, value in expected.items():
            assert_equivalent(actual[key], value)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError("record reproduction differs")
        for left, right in zip(actual, expected, strict=True):
            assert_equivalent(left, right)
    elif isinstance(expected, float):
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (float, int))
            or not math.isfinite(actual)
            or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)
        ):
            raise ValueError("record reproduction differs")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError("record reproduction differs")


def mature(protocol: dict[str, Any], now: datetime) -> None:
    if now < timestamp(protocol["window_end"]) + timedelta(hours=98):
        raise ValueError(
            "prospective truth and evaluation require window_end +98 hours"
        )


def truth_frame(path: Path) -> pl.DataFrame:
    regular(path)
    frame = pl.read_parquet(path)
    if not {"valid_hour", TARGET} <= set(frame.columns):
        raise ValueError("truth requires QC interval-mean temperature labels")
    dtype = frame.schema["valid_hour"]
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None:
        raise ValueError("truth hours must include timezone")
    frame = frame.with_columns(pl.col("valid_hour").dt.convert_time_zone("UTC"))
    times = frame["valid_hour"]
    if (
        not len(times)
        or times.null_count()
        or times.n_unique() != len(times)
        or any(t.minute or t.second or t.microsecond for t in times)
    ):
        raise ValueError("truth must contain unique nonempty exact hours")
    if (
        not frame.schema[TARGET].is_numeric()
        or frame.filter(pl.col(TARGET).is_infinite() | pl.col(TARGET).is_nan()).height
    ):
        raise ValueError("truth must contain numeric finite-or-null QC labels")
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
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        folder.rename(output)
    finally:
        # Cleanup cannot turn a published complete result into a reported failure.
        with suppress(OSError):
            lease.unlink(missing_ok=True)
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def freeze_truth(truth: Path, protocol_path: Path, output: Path, now: datetime) -> Path:
    protocol = json.loads(protocol_path.read_text())
    mature(protocol, now)
    regular(truth)
    # Validate the copied bytes, so the lock describes exactly the data scored.
    with staged_directory(output) as folder:
        target = folder / "truth.parquet"
        target.write_bytes(truth.read_bytes())
        frame = truth_frame(target)
        lock = {
            "artifact_version": ARTIFACT_VERSION,
            "protocol_sha256": digest(protocol_path),
            "enrollment_sha256": digest(protocol_path.with_suffix(".lock.json")),
            "truth_sha256": digest(target),
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
    root: Path, protocol: Path, replacement: Path | None, now: datetime
) -> Path:
    if root.is_symlink() or root.is_junction() or not root.is_dir():
        raise ValueError("truth lock must be a local directory")
    lock_path, truth = root / "truth.lock.json", root / "truth.parquet"
    regular(lock_path)
    frame = truth_frame(truth)
    lock = json.loads(lock_path.read_text())
    mature(json.loads(protocol.read_text()), timestamp(lock["sealed_at"]))
    if (
        lock.get("artifact_version") != ARTIFACT_VERSION
        or lock.get("protocol_sha256") != digest(protocol)
        or lock.get("enrollment_sha256") != digest(protocol.with_suffix(".lock.json"))
        or lock.get("truth_sha256") != digest(truth)
        or lock.get("target") != TARGET
        or lock.get("rows") != frame.height
        or lock.get("first_hour")
        != cast(datetime, frame["valid_hour"].min()).isoformat()
        or lock.get("last_hour")
        != cast(datetime, frame["valid_hour"].max()).isoformat()
        or timestamp(lock["sealed_at"]) > now
    ):
        raise ValueError("frozen truth identity differs")
    if replacement is not None:
        regular(replacement)
        if digest(replacement) != lock["truth_sha256"]:
            raise ValueError("supplied truth differs from the frozen truth")
    return truth
