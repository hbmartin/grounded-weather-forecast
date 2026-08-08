"""Locked, atomic filesystem writes shared by persistent artifact stores."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import polars as pl
from filelock import FileLock

# Not a config key on purpose: config_fingerprint hashes repr(config), so an
# operational knob there would invalidate promoted evidence every time the
# operator tuned it.
PIPELINE_LOCK_TIMEOUT_S = 60.0


@contextmanager
def pipeline_lock(dataset_dir: Path, timeout: float | None = None) -> Iterator[None]:
    """Serialize whole-pipeline mutators on one coarse lock.

    build-dataset, backtest, report, and prune-scores all read or rewrite the
    scores directory; running two of them at once (a manual cycle against the
    scheduled maintain chain) lets prune delete files a concurrent report has
    already globbed. ``predict`` deliberately stays outside — serving must
    never wait behind an hour-long report; its glob races are handled by
    retrying the scan instead. Raises ``filelock.Timeout`` on contention.
    """
    lock_path = dataset_dir / "pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = PIPELINE_LOCK_TIMEOUT_S if timeout is None else timeout
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


def atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    """Write parquet beside its destination, then replace it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        frame.write_parquet(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(text: str, path: Path) -> None:
    """Write text beside its destination, then replace it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent,
        suffix=path.suffix,
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as tmp:
        tmp.write(text)
        temporary = Path(tmp.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
