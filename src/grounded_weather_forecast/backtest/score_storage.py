"""Score publication ownership and recovery of unpublished temporary files."""

import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock, Timeout

SCORE_WRITE_LOCK = ".scores-write.lock"
_SCORE_TEMPORARY = re.compile(r"\.scores_.*\.parquet\.[0-9a-f]{16}\.tmp")


@contextmanager
def score_write_lock(directory: Path, *, timeout: float = -1) -> Iterator[None]:
    """Serialize score publishers and reclamation; readers never take this lock.

    Keep the small lock file after release: unlinking a live lock's inode would
    let another process lock a different inode under the same name.
    """
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(directory / SCORE_WRITE_LOCK, timeout=timeout):
        yield


@dataclass(frozen=True, slots=True)
class ScoreTempCleanup:
    """Removed (or dry-run eligible) files, independent of published evidence."""

    deleted: tuple[Path, ...] = ()
    freed_mb: float = 0.0
    skipped: tuple[str, ...] = ()


def _reclaim_temporaries(directory: Path, *, dry_run: bool) -> ScoreTempCleanup:
    deleted: list[Path] = []
    skipped: list[str] = []
    freed = 0
    for path in sorted(directory.glob(".scores_*.tmp")):
        if _SCORE_TEMPORARY.fullmatch(path.name) is None:
            continue
        try:
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                continue
            if not dry_run:
                path.unlink()
        except (FileNotFoundError,):  # noqa: B013 -- harmless external removal
            continue
        except (OSError,) as error:  # noqa: B013 -- project tuple convention
            skipped.append(f"{path.name}: temporary cleanup failed: {error}")
            continue
        deleted.append(path)
        freed += metadata.st_size
    return ScoreTempCleanup(tuple(deleted), freed / 1e6, tuple(skipped))


def cleanup_score_temporaries(
    directory: Path, *, dry_run: bool = False
) -> ScoreTempCleanup:
    """Reclaim abandoned scores without waiting behind an active library writer.

    CLI callers also hold the pipeline lock, coordinating with older CLI
    publishers. All current score publishers participate in the directory lock.
    No age test is needed while exclusive ownership proves no publisher is live.
    """
    if not directory.exists():
        return ScoreTempCleanup()
    try:
        with score_write_lock(directory, timeout=0):
            return _reclaim_temporaries(directory, dry_run=dry_run)
    except (Timeout,):  # noqa: B013 -- project tuple convention
        return ScoreTempCleanup(skipped=("temporary cleanup: score writer is active",))
    except (OSError,) as error:  # noqa: B013
        return ScoreTempCleanup(skipped=(f"temporary cleanup failed: {error}",))
