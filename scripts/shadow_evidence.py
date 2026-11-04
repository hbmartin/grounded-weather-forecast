"""Local evidence checks; hashes detect inconsistency, not same-user forgery."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import polars as pl
from filelock import BaseFileLock, FileLock, Timeout

from grounded_weather_forecast.shadow_artifacts import (
    ARTIFACT_VERSION,
    BINDING,
    TARGET,
    EvidenceError,
    capsule_directory,
    current_artifact,
    digest,
    finite_number,
    forecast_rows,
    hex_identity,
    issuance_name,
    json_object,
    object_value,
    private_directory,
    regular,
    required_string,
    sha256,
    sync_directory,
    sync_files,
    timestamp,
)

INPUT_FILES = frozenset(
    {
        "issue.parquet",
        "training.parquet",
        "truth_hourly.parquet",
        "production.json",
        "provider_completions.parquet",
    }
)


def manifest_hashes(
    meta: dict[str, Any], children: set[str], *, prospective: bool
) -> dict[str, Any]:
    if prospective and not current_artifact(meta.get("artifact_version")):
        raise EvidenceError("prospective capsules require a fresh v3 enrollment")
    hashes = object_value(meta.get("input_sha256", {}), "capsule.input_sha256")
    if prospective and not hashes.keys() >= INPUT_FILES:
        raise EvidenceError("prospective requires the complete captured input manifest")
    for name, expected in hashes.items():
        if not name or name in {".", ".."} or any(c in name for c in "/\\:"):
            raise EvidenceError(f"invalid capsule filename: {name!r}")
        if name not in children:
            raise EvidenceError(
                f"invalid capsule filename: {name!r} is not a regular file"
            )
        hex_identity(expected, f"capsule.input_sha256.{name}", 64)
    return hashes


@dataclass(eq=False)
class CapsuleSnapshot:
    """One invocation's bytes; parsed projections never mutate these inputs."""

    capsule: Path
    files: dict[str, bytes]
    hashes: dict[str, str]
    prospective: bool
    objects: dict[str, dict[str, Any]] = dataclass_field(default_factory=dict)
    frames: dict[str, pl.DataFrame] = dataclass_field(default_factory=dict)

    @classmethod
    def load(cls, capsule: Path, *, prospective: bool) -> CapsuleSnapshot:
        capsule_directory(capsule, prospective=prospective)
        children = tuple(capsule.iterdir())
        if any(path.is_symlink() or path.is_junction() for path in children):
            raise EvidenceError("capsule symlink is not a self-contained input")
        if any(not path.is_file() for path in children):
            raise EvidenceError("capsule child must be a regular file")
        child_names = {path.name for path in children}
        raw = (capsule / "capsule.json").read_bytes()
        meta = json_object(raw, "capsule manifest")
        timestamp(required_string(meta.get("input_as_of"), "capsule.input_as_of"))
        hashes = manifest_hashes(meta, child_names, prospective=prospective)
        optional = INPUT_FILES | {
            "publication.json",
            "published.json",
            "evidence.lock.json",
        }
        names = set(hashes) | {name for name in optional if (capsule / name).is_file()}
        files = {"capsule.json": raw}
        for name in sorted(names - files.keys()):
            path = capsule / name
            if not path.is_file():
                raise EvidenceError(
                    f"invalid capsule filename: {name!r} is not a regular file"
                )
            regular(path)
            files[name] = path.read_bytes()
        snapshot = cls(
            capsule,
            files,
            {name: sha256(data) for name, data in files.items()},
            prospective,
        )
        snapshot.objects["capsule.json"] = meta
        for name, expected in hashes.items():
            if snapshot.hashes[name] != expected:
                raise EvidenceError(f"captured input changed: {name}")
        return snapshot

    def read(self, name: str) -> bytes:
        if name not in self.files:
            raise EvidenceError(f"{self.capsule / name}: missing captured evidence")
        return self.files[name]

    def document(self, name: str, label: str) -> dict[str, Any]:
        if name not in self.objects:
            self.objects[name] = json_object(self.read(name), label)
        return deepcopy(self.objects[name])

    def parquet(self, name: str) -> pl.DataFrame:
        if name not in self.frames:
            try:
                self.frames[name] = pl.read_parquet(BytesIO(self.read(name)))
            except (pl.exceptions.PolarsError,) as error:  # noqa: B013 -- input boundary
                raise EvidenceError(f"{name}: invalid Parquet input") from error
        return self.frames[name].clone()

    def metadata(self) -> dict[str, Any]:
        meta = self.document("capsule.json", "capsule manifest")
        if "publication.json" in self.files:
            meta["publication"] = self.document(
                "publication.json", "sealed publication receipt"
            )
        return meta

    def digests(self, *, sealed: bool = True) -> dict[str, str]:
        return {
            name: self.hashes[name]
            for name in sorted(self.hashes)
            if sealed or name != "evidence.lock.json"
        }

    def with_files(
        self, files: dict[str, bytes], *, known_hashes: dict[str, str] | None = None
    ) -> CapsuleSnapshot:
        """Extend a view with this owner's bytes and already validated digests."""
        known_hashes = known_hashes or {}
        return CapsuleSnapshot(
            self.capsule,
            self.files | files,
            self.hashes
            | {
                name: known_hashes[name] if name in known_hashes else sha256(raw)
                for name, raw in files.items()
            },
            self.prospective,
            {name: value for name, value in self.objects.items() if name not in files},
            {name: value for name, value in self.frames.items() if name not in files},
        )


def protocol_rules(protocol: dict[str, Any], *, complete: bool = True) -> None:
    """Validate the numeric rules before scoring, without changing their values."""
    counts = {
        "quality": (
            "minimum_matched_utc_dates_per_horizon",
            "minimum_matched_valid_hours_per_horizon",
            "minimum_weeks_with_data",
            "minimum_training_unique_valid_hours",
        ),
        "uncertainty": (
            "bootstrap_samples",
            "primary_block_days",
            "sensitivity_block_days",
            "family_comparisons",
        ),
    }
    fractions = {
        "quality": "minimum_common_coverage",
        "uncertainty": "two_sided_family_alpha",
    }
    for domain, fields in counts.items():
        if not complete and domain not in protocol:
            continue
        rule = object_value(protocol.get(domain), f"protocol.{domain}")
        for field in fields:
            value = rule.get(field)
            if type(value) is not int or value < 1:
                raise EvidenceError(
                    f"protocol.{domain}.{field}: expected positive integer"
                )
        field = fractions[domain]
        value = rule.get(field)
        if not finite_number(value) or not 0 < value <= 1:
            raise EvidenceError(
                f"protocol.{domain}.{field}: expected fraction in (0,1]"
            )
        if domain == "uncertainty":
            value = rule.get("seed")
            if type(value) is not int or value < 0:
                raise EvidenceError(
                    "protocol.uncertainty.seed: expected nonnegative integer"
                )


def protocol_document(payload: bytes) -> dict[str, Any]:
    protocol = json_object(payload, "protocol")
    start, end = (
        timestamp(required_string(protocol.get(key), f"protocol.{key}"))
        for key in ("window_start", "window_end")
    )
    if start >= end:
        raise EvidenceError("protocol: window_start must precede window_end")
    protocol_rules(protocol, complete=False)
    return protocol


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


@dataclass(frozen=True, eq=False)
class ValidatedPublication:
    payload: bytes
    payload_sha256: str
    document: dict[str, Any]
    receipt: dict[str, Any]


def publication(
    value: object, *, payload_snapshot: tuple[bytes, str] | None = None
) -> ValidatedPublication:
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
    except (ValueError,) as error:  # noqa: B013 -- project exception convention
        raise EvidenceError("invalid publisher receipt payload") from error
    document = json_object(payload, "publisher receipt payload")
    if payload_snapshot is not None and payload != payload_snapshot[0]:
        raise EvidenceError("publisher receipt payload mismatch")
    payload_digest = (
        payload_snapshot[1]
        if payload_snapshot is not None
        else hashlib.sha256(payload).hexdigest()
    )
    if (
        payload_digest != receipt.get("forecast_sha256")
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
    return ValidatedPublication(payload, payload_digest, document, receipt)


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


def truth_frame(path: Path | bytes, *, prospective: bool = True) -> pl.DataFrame:
    if isinstance(path, Path):
        regular(path)
        payload = path.read_bytes()
    else:
        payload = path
    try:
        frame = pl.read_parquet(BytesIO(payload))
    except (pl.exceptions.PolarsError,) as error:  # noqa: B013 -- input boundary
        raise EvidenceError("truth: invalid Parquet input") from error
    if not {"valid_hour", TARGET} <= set(frame.columns):
        raise EvidenceError("truth requires QC interval-mean temperature labels")
    dtype = frame.schema["valid_hour"]
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None:
        raise EvidenceError("truth hours must include timezone")
    # Check nanosecond residues before casting so sub-hour evidence cannot round
    # into a valid hour during normalization.
    if (
        prospective
        and frame.filter(
            pl.col("valid_hour").dt.epoch("ns") % (3600 * 10**9) != 0
        ).height
    ):
        raise EvidenceError("truth must contain unique nonempty exact hours")
    if (
        not prospective
        and frame.filter(pl.col("valid_hour").dt.epoch("ns") % 1000 != 0).height
    ):
        raise EvidenceError("replay truth: expected microsecond-precision timestamps")
    frame = frame.with_columns(
        pl.col("valid_hour").dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC"))
    )
    times = frame["valid_hour"]
    if (
        not len(times)
        or times.null_count()
        or times.n_unique() != len(times)
        or (prospective and any(t.minute or t.second or t.microsecond for t in times))
    ):
        raise EvidenceError("truth must contain unique nonempty exact hours")
    if not frame.schema[TARGET].is_numeric() or (
        prospective
        and frame.filter(pl.col(TARGET).is_infinite() | pl.col(TARGET).is_nan()).height
    ):
        raise EvidenceError("truth must contain numeric finite-or-null QC labels")
    return frame


@contextmanager
def native_guard(
    lease: Path, label: str, *, output: Path | None = None
) -> Iterator[None]:
    """Never substitute marker existence for native ownership or mask a primary error."""
    if lease.exists() or lease.is_symlink():
        regular(lease)
    lock = FileLock(lease, timeout=0, mode=0o600, fallback_to_soft=False)
    try:
        lock.acquire()
    except (Timeout,) as error:  # noqa: B013 -- coordination is operational
        raise OSError(f"{label} is already running: {lease}") from error
    except (OSError,) as error:  # noqa: B013 -- native ownership is mandatory
        message = f"{label}: cannot acquire native guard {lease}: {error}"
        if error.errno is None:
            raise OSError(message) from error
        raise OSError(error.errno, message) from error
    try:
        yield
    except (BaseException,) as error:  # noqa: B013 -- retain cancellation and primary failure
        release_guard(lock, lease, label, primary=error)
        raise
    else:
        release_guard(lock, lease, label, output=output)


def release_guard(
    lock: BaseFileLock,
    lease: Path,
    label: str,
    *,
    primary: BaseException | None = None,
    output: Path | None = None,
) -> None:
    try:
        lock.release()
    except (Exception,) as error:  # noqa: B013 -- coordination boundary
        if primary is not None:
            primary.add_note(f"{label} guard release failed at {lease}: {error}")
            return
        preserved = (
            f"; complete output preserved at {output}"
            if output is not None and output.exists()
            else ""
        )
        raise OSError(
            f"{label} guard release failed at {lease}{preserved}: {error}"
        ) from error


@contextmanager
def staged_directory(output: Path) -> Iterator[Path]:
    """Hold an OS-managed guard; publish a complete directory after success."""
    private_directory(output.parent)
    temporary = None
    try:
        with native_guard(
            output.with_name(f".{output.name}.publishing"),
            "output publication",
            output=output,
        ):
            if output.exists() or output.is_symlink():
                raise FileExistsError(output)
            temporary = Path(
                tempfile.mkdtemp(prefix="shadow-stage-", dir=output.parent)
            )
            folder = temporary / "complete"
            folder.mkdir(mode=0o700)
            yield folder
            sync_files(folder, folder.iterdir())
            if output.exists() or output.is_symlink():
                raise FileExistsError(output)
            folder.rename(output)
            try:
                sync_directory(output.parent)
            except (OSError,) as error:  # noqa: B013 -- preserve published evidence
                raise OSError(
                    f"complete output preserved at {output}; directory synchronization failed: {error}"
                ) from error
    finally:
        if temporary is not None:
            with suppress(OSError):
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
    protocol = protocol_document(protocol_bytes)
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
        protocol_document(protocol_bytes),
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
