"""Locked, atomic filesystem writes shared by persistent artifact stores."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import polars as pl
from filelock import FileLock


@contextmanager
def locked_path(path: Path) -> Iterator[None]:
    """Hold the sidecar lock for ``path`` across a read-modify-write cycle."""
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_path):
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
