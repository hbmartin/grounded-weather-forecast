"""Pure shadow contracts and durable local evidence primitives."""

import hashlib
import json
import math
import os
import platform
import re
import stat
import sys
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

ARTIFACT_VERSION = 3
ISSUE_FILE = "issue.parquet"
TRAINING_FILE = "training.parquet"
TARGET = "t__temp_c__mean"
BINDING = "publisher successful explicit-commit push"
THREAD_CONTROLS = (
    "POLARS_MAX_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class EvidenceError(ValueError):
    """A named, expected refusal of supplied experiment evidence."""


def current_artifact(value: object) -> bool:
    return type(value) is int and value == ARTIFACT_VERSION


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest(path: Path) -> str:
    return sha256(path.read_bytes())


def timestamp(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value)
        if result.utcoffset() is None:
            raise ValueError("timezone missing")
        return result.astimezone(UTC)
    except (TypeError, ValueError, OverflowError) as error:
        raise EvidenceError(f"timestamp must include timezone: {value!r}") from error


def issuance_name(origin: datetime) -> str:
    return origin.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def object_value(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise EvidenceError(f"{label}: expected an object")
    return cast("dict[str, Any]", value)


def json_value(payload: bytes | str, label: str) -> Any:
    try:
        value = json.loads(payload)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise EvidenceError(f"{label}: invalid JSON") from error
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > 64:
            raise EvidenceError(f"{label}: excessive JSON nesting")
        if isinstance(item, dict):
            pending.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, list):
            pending.extend((v, depth + 1) for v in item)
    return value


def json_object(payload: bytes | str, label: str) -> dict[str, Any]:
    return object_value(json_value(payload, label), label)


def required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{label}: requires nonempty string fields")
    return value


def hex_identity(value: object, label: str, length: int) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(f"[0-9a-f]{{{length}}}", value) is None
    ):
        raise EvidenceError(
            f"{label}: requires full {'commit identity' if length == 40 else 'SHA-256 identity'}"
        )
    return value


def _string_map(value: object, label: str) -> dict[str, Any]:
    result = object_value(value, label)
    if any(v is not None and not isinstance(v, str) for v in result.values()):
        raise EvidenceError(f"{label}: expected string or null values")
    return result


def finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError,):  # noqa: B013 -- project exception convention
        return False


def forecast_rows(
    document: dict[str, Any], *, public: bool
) -> dict[str, dict[str, Any]]:
    """Validate every nested field consumed by production projection, before writes."""
    label = "published" if public else "source"
    timestamp(required_string(document.get("issued_at"), f"{label}.issued_at"))
    required_string(document.get("dataset_fingerprint"), f"{label}.dataset_fingerprint")
    releases = document.get("release_ids")
    if not isinstance(releases, list) or any(not isinstance(v, str) for v in releases):
        raise EvidenceError(f"{label}.release_ids: expected string array")
    hourly = document.get("hourly")
    if not isinstance(hourly, list) or not hourly:
        raise EvidenceError(f"{label}.hourly: expected nonempty object array")
    variable = "temp_f" if public else "temp_c"
    rows = {}
    for index, value in enumerate(hourly):
        row = object_value(value, f"{label}.hourly[{index}]")
        valid = timestamp(
            required_string(row.get("valid_time"), "hourly.valid_time")
        ).isoformat()
        if valid in rows:
            raise EvidenceError(f"duplicated {label} forecast grid")
        for field in ("methods", "release_ids", "selection_reasons", "truth_semantics"):
            row[field] = _string_map(
                row.get(field, {}), f"{label}.hourly[{index}].{field}"
            )
        values = object_value(row.get("values"), f"{label}.hourly[{index}].values")
        number = values.get(variable)
        if variable not in values or (
            number is not None and (not finite_number(number))
        ):
            raise EvidenceError(
                f"{label}.hourly[{index}].values.{variable}: expected finite number or null"
            )
        row["valid_time"] = valid
        rows[valid] = row
    if public:
        if (
            type(document.get("schema_version")) is not int
            or document["schema_version"] != 2
        ):
            raise EvidenceError("published.schema_version: expected integer 2")
        if _string_map(document.get("units"), "published.units").get(variable) != "F":
            raise EvidenceError("published.units.temp_f: expected F")
        if (
            _string_map(
                document.get("truth_semantics"), "published.truth_semantics"
            ).get(variable)
            != "mean"
        ):
            raise EvidenceError("published temperature must target interval mean truth")
    elif any(
        row.get("truth_semantics", {}).get(variable) != "mean" for row in rows.values()
    ):
        raise EvidenceError("production temperature must target interval mean truth")
    return rows


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def regular(path: Path) -> None:
    if path.is_symlink() or path.is_junction():
        raise EvidenceError(f"{path}: evidence must be a regular local file")
    if not stat.S_ISREG(path.stat().st_mode):
        raise EvidenceError(f"{path}: evidence must be a regular local file")


def system_path(path: Path) -> Path:
    """Canonicalize macOS system prefixes without resolving selected evidence aliases."""
    if sys.platform == "darwin":
        absolute = path.absolute()
        for name in ("tmp", "var"):
            prefix = Path("/") / name
            if absolute.is_relative_to(prefix) and prefix.is_symlink():
                return prefix.resolve() / absolute.relative_to(prefix)
    return path


def required_artifact(path: Path, *, snapshot: bytes | None = None) -> bytes:
    path = system_path(path)
    try:
        regular(path)
        return path.read_bytes() if snapshot is None else snapshot
    except (FileNotFoundError,) as error:  # noqa: B013 -- required supplied evidence
        raise EvidenceError(f"{path}: missing required enrollment artifact") from error


def existing_directory(path: Path, *, private: bool = False) -> None:
    """Inspect supplied directories without creating or repairing their paths."""
    path = system_path(path)
    try:
        metadata = path.lstat()
    except (FileNotFoundError,) as error:  # noqa: B013 -- missing supplied evidence
        raise EvidenceError(f"{path}: missing capsule directory") from error
    if stat.S_ISLNK(metadata.st_mode) or path.is_junction():
        raise EvidenceError(f"{path}: expected a nonaliased local directory")
    if not stat.S_ISDIR(metadata.st_mode):
        raise EvidenceError(f"{path}: expected a local directory")
    if private and os.name == "posix" and metadata.st_mode & 0o077:
        raise EvidenceError(
            f"{path}: private capsule directory permits group/world access"
        )


def capsule_directory(path: Path, *, prospective: bool) -> None:
    path = system_path(path)
    existing_directory(path.parent, private=prospective)
    existing_directory(path, private=prospective)


def directory_child(root: Path, name: str, *, required: bool, label: str) -> Path:
    """Check selected roots and issuance leaves, allowing trusted ancestor prefixes."""
    root = system_path(root)
    if root.exists() or root.is_symlink() or root.is_junction():
        existing_directory(root)
    path = root / name
    if (
        path.is_symlink()
        or path.is_junction()
        or (path.exists() and not path.is_dir())
        or (required and not path.is_dir())
        or path.resolve().parent != root.resolve()
    ):
        raise EvidenceError(f"{path}: {label}")
    return path


def sync_files(directory: Path, paths: Iterable[Path]) -> None:
    """Synchronize explicitly selected regular files before their directory."""
    for path in paths:
        regular(path)
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    sync_directory(directory)


def private_directory(path: Path) -> None:
    path = system_path(path)
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for folder in reversed(missing):
        folder.mkdir(mode=0o700, exist_ok=True)
        sync_directory(folder.parent)
    if path.is_symlink() or path.is_junction() or not path.is_dir():
        raise EvidenceError(f"{path}: expected a private local directory")


def write_bytes_new(path: Path, payload: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        sync_directory(path.parent)
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()


def runtime_identity(lockfile: Path) -> dict[str, Any]:
    import numpy as np  # noqa: PLC0415 -- runtime is sampled after thread setup
    import polars as pl  # noqa: PLC0415
    import scipy  # noqa: PLC0415

    if pl.thread_pool_size() != 2 or any(
        os.environ.get(name) != "2" for name in THREAD_CONTROLS
    ):
        raise EvidenceError(
            "shadow numerical runtime requires two-thread controls before numerical imports"
        )

    def backend(config: Mapping[str, Any]) -> dict[str, Any]:
        dependencies = config.get("Build Dependencies", {})
        return {
            key: {
                name: value
                for name, value in dependencies.get(key, {}).items()
                if name in {"name", "version", "openblas configuration"}
            }
            for key in ("blas", "lapack")
        }

    return {
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "platform": platform.system(),
        "architecture": platform.machine(),
        "libraries": {
            name: version(name) for name in ("numpy", "scipy", "polars", "pyarrow")
        },
        "backend": {
            "numpy": backend(np.__config__.CONFIG),
            "scipy": backend(scipy.__config__.CONFIG),
        },
        "polars_threads": pl.thread_pool_size(),
        "thread_controls": {name: os.environ.get(name) for name in THREAD_CONTROLS},
        "dependency_lock_sha256": digest(lockfile),
    }
