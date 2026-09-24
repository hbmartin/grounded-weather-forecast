"""Locked, atomic filesystem writes shared by persistent artifact stores."""

import os
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import polars as pl
from filelock import FileLock

# Not a config key on purpose: config_fingerprint hashes repr(config), so an
# operational knob there would invalidate promoted evidence every time the
# operator tuned it.
PIPELINE_LOCK_TIMEOUT_S = 60.0
DATASET_LOCK_TIMEOUT_S = 60.0


@contextmanager
def pipeline_lock(dataset_dir: Path, timeout: float | None = None) -> Iterator[None]:
    """Serialize whole-pipeline mutators on one coarse lock.

    build-dataset, backtest, report, and prune-scores all read or rewrite the
    scores directory; running two of them at once (a manual cycle against the
    scheduled maintain chain) lets prune delete files a concurrent report has
    already globbed. ``predict`` and ``publish`` deliberately stay outside and
    use the short dataset publication lock instead, so serving never waits
    behind an hour-long report. Raises ``filelock.Timeout`` on contention.
    """
    lock_path = dataset_dir / "pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = PIPELINE_LOCK_TIMEOUT_S if timeout is None else timeout
    with FileLock(lock_path, timeout=resolved):
        yield


@contextmanager
def dataset_lock(dataset_dir: Path, timeout: float | None = None) -> Iterator[None]:
    """Serialize the short live-dataset publication/read window."""
    lock_path = dataset_dir / "dataset.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = DATASET_LOCK_TIMEOUT_S if timeout is None else timeout
    with FileLock(lock_path, timeout=resolved):
        yield


@contextmanager
def locked_path(path: Path, timeout: float = -1) -> Iterator[None]:
    """Hold the sidecar lock for ``path`` across a read-modify-write cycle.

    A negative ``timeout`` blocks forever; telemetry writers pass a finite
    timeout so they can drop a row instead of stalling the command.
    """
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_path, timeout=timeout):
        yield


def _temporary_sibling(path: Path) -> Path:
    """Create an exclusive sibling using normal umask-derived permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        candidate = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
        try:
            descriptor = os.open(
                candidate,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o666,
            )
        except FileExistsError:  # pragma: no cover - cryptographic-name collision
            continue
        os.close(descriptor)
        return candidate
    msg = f"could not allocate a temporary sibling for {path}"
    raise FileExistsError(msg)


def _preserve_mode(temporary: Path, destination: Path) -> None:
    """Copy the destination mode only after the staged contents are written."""
    if destination.exists():
        temporary.chmod(stat.S_IMODE(destination.stat().st_mode))


def output_target(path: Path) -> Path:
    """Keep a caller's output symlink while replacing its target atomically."""
    return path.resolve(strict=False) if path.is_symlink() else path


def stage_parquet(frame: pl.DataFrame, path: Path) -> Path:
    """Write parquet to a uniquely named sibling without replacing ``path``."""
    temporary = _temporary_sibling(path)
    try:
        frame.write_parquet(temporary)
        _preserve_mode(temporary, path)
    except (Exception, KeyboardInterrupt):
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def stage_text(text: str, path: Path) -> Path:
    """Write text to a uniquely named sibling without replacing ``path``."""
    temporary = _temporary_sibling(path)
    try:
        temporary.write_text(text, encoding="utf-8")
        _preserve_mode(temporary, path)
    except (Exception, KeyboardInterrupt):
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    """Write parquet beside its destination, then replace it atomically."""
    temporary = stage_parquet(frame, path)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(text: str, path: Path) -> None:
    """Write text beside its destination, then replace it atomically."""
    temporary = stage_text(text, path)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
