"""Independent score publication leases and recovery of unpublished files."""

import logging
import os
import stat
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock, Timeout

from grounded_weather_forecast.storage import (
    locked_path,
    publish_parquet,
    temporary_destination,
    temporary_sibling,
)

SCORE_STAGE_NAMESPACE = "score-stage-v1"
SCORE_WRITE_LOCK = f".{SCORE_STAGE_NAMESPACE}.lock"
SCORE_COORDINATION_TIMEOUT_S = 60.0
_logger = logging.getLogger(__name__)

type _DirectoryIdentity = tuple[int, int, int]


class _LockState(threading.local):
    """Initialize explicitly typed bookkeeping independently for each thread."""

    def __init__(self) -> None:
        self.directories: frozenset[_DirectoryIdentity] = frozenset()


_held = _LockState()


class ScoreCoordinationTimeout(Timeout):
    """A brief score-directory bookkeeping lock could not be acquired in time."""


class ScoreCoordinationReleaseError(RuntimeError):
    """A guard release failed; the lock's ownership may be uncertain."""

    def __init__(self, lock_path: Path, cause: Exception) -> None:
        self.lock_path = lock_path
        self.cause = cause
        super().__init__(
            f"directory guard release failed for {lock_path}; "
            f"ownership uncertain: {cause}"
        )


def _release_guard(guard: AbstractContextManager[None], lock_path: Path) -> None:
    try:
        guard.__exit__(None, None, None)
    except (Exception,) as error:  # noqa: B013 -- resource release only
        raise ScoreCoordinationReleaseError(lock_path, error) from error


@contextmanager
def score_write_lock(
    directory: Path, *, timeout: float | None = None
) -> Iterator[None]:
    """Brief directory coordination, reentrant only on this process and thread.

    Filesystem identity makes directory aliases equivalent. Payload leases use
    fresh lock objects instead, including during recovery in the writer thread.
    """
    directory.mkdir(parents=True, exist_ok=True)
    metadata = directory.stat()
    identity = (os.getpid(), metadata.st_dev, metadata.st_ino)
    held = _held.directories
    if identity in held:
        yield
        return
    guard = locked_path(
        directory / f".{SCORE_STAGE_NAMESPACE}",
        SCORE_COORDINATION_TIMEOUT_S if timeout is None else timeout,
    )
    try:
        guard.__enter__()
    except (Timeout,) as error:  # noqa: B013 -- translate acquisition only
        raise ScoreCoordinationTimeout(error.lock_file) from error
    _held.directories = held | {identity}
    try:
        yield
    except (BaseException,) as error:  # noqa: B013
        try:
            _release_guard(guard, directory / SCORE_WRITE_LOCK)
        except (Exception,) as secondary:  # noqa: B013
            error.add_note(f"score coordination release failed: {secondary}")
        raise
    else:
        _release_guard(guard, directory / SCORE_WRITE_LOCK)
    finally:
        _held.directories = held


def _lease_path(temporary: Path) -> Path:
    return temporary.with_name(f"{temporary.name}.lease")


def _cleanup_action(action: Callable[[], object], phase: str, path: Path) -> None:
    try:
        action()
    except (BaseException,) as error:  # noqa: B013 -- annotate, never suppress
        error.add_note(f"score {phase} failed for {path}")
        raise


def _cleanup_diagnostic(error: Exception) -> str:
    return "; ".join((str(error), *getattr(error, "__notes__", ())))


def _warn_committed_cleanup(path: Path, error: Exception) -> None:
    if isinstance(error, ScoreCoordinationTimeout):
        _logger.warning(
            "published %s; cleanup deferred: directory guard busy (%s); "
            "a later backtest or prune-scores can recover leftovers",
            path,
            error.lock_file,
        )
    else:
        _logger.warning("published %s; %s", path, _cleanup_diagnostic(error))


def _dispose_stage(temporary: Path, lease: FileLock) -> None:
    """Release ownership even when bookkeeping fails; leftovers are recoverable."""
    try:
        with score_write_lock(temporary.parent, timeout=0):
            _cleanup_action(
                lambda: temporary.unlink(missing_ok=True), "payload removal", temporary
            )
            _cleanup_action(lease.release, "lease release", _lease_path(temporary))
            _cleanup_action(
                lambda: _lease_path(temporary).unlink(missing_ok=True),
                "lease companion removal",
                _lease_path(temporary),
            )
    except (BaseException,) as error:  # noqa: B013
        if lease.is_locked:
            try:
                lease.release()
            except (Exception,) as secondary:  # noqa: B013
                error.add_note(
                    f"score lease release failed for {_lease_path(temporary)}: {secondary}"
                )
        raise


@contextmanager
def score_parquet_destination(path: Path) -> Iterator[Path]:
    """Own one payload through encoding, producer closure and atomic publication.

    Other payloads can be written in parallel, even to the same destination.
    Publication keeps atomic last-replacement-wins semantics. Only allocation,
    replacement and removal use the short directory guard.
    """
    temporary: Path | None = None
    lease: FileLock | None = None
    committed = False
    try:
        with score_write_lock(path.parent):
            temporary = temporary_sibling(path, namespace=SCORE_STAGE_NAMESPACE)
            lease = FileLock(_lease_path(temporary))
            lease.acquire(timeout=0)
        yield temporary
        with score_write_lock(path.parent):
            publish_parquet(temporary, path)
            committed = True
    except (BaseException,) as error:  # noqa: B013
        if temporary is not None and lease is not None:
            try:
                _dispose_stage(temporary, lease)
            except (Exception,) as secondary:  # noqa: B013
                error.add_note(
                    f"score stage cleanup failed: {_cleanup_diagnostic(secondary)}"
                )
        if committed and isinstance(error, Exception):
            _warn_committed_cleanup(path, error)
        else:
            raise
    else:
        try:
            _dispose_stage(temporary, lease)
        except (Exception,) as error:  # noqa: B013
            # The destination is already committed. Do not report it as failed.
            _warn_committed_cleanup(path, error)


@dataclass(frozen=True, slots=True)
class ScoreTempCleanup:
    """Removed (or dry-run eligible) payloads, independent of published evidence."""

    deleted: tuple[Path, ...] = ()
    freed_mb: float = 0.0
    skipped: tuple[str, ...] = ()


def _regular(path: Path) -> bool:
    return stat.S_ISREG(path.lstat().st_mode)


def _stage_payloads(directory: Path) -> list[Path]:
    payloads: set[Path] = set()
    for entry in directory.glob(f".{SCORE_STAGE_NAMESPACE}.*"):
        path = entry.with_name(entry.name.removesuffix(".lease"))
        destination = temporary_destination(path, namespace=SCORE_STAGE_NAMESPACE)
        if (
            destination
            and destination.name.startswith("scores_")
            and destination.suffix == ".parquet"
        ):
            payloads.add(path)
    return sorted(payloads)


def _reclaim_stage(
    path: Path, *, dry_run: bool, deleted: list[Path], skipped: list[str]
) -> int:
    """Account for payload deletion before releasing a potentially faulty lease."""
    lease_path = _lease_path(path)
    if (
        path.is_symlink()
        or lease_path.is_symlink()
        or (path.exists() and not _regular(path))
        or (lease_path.exists() and not _regular(lease_path))
    ):
        return 0
    lease = FileLock(lease_path)
    freed = 0
    primary: BaseException | None = None
    try:
        lease.acquire(timeout=0)
    except (Timeout,):  # noqa: B013
        skipped.append(f"{path.name}: temporary cleanup: score writer is active")
        return 0
    try:
        if path.exists():
            size = path.stat().st_size
            if not dry_run:
                _cleanup_action(path.unlink, "payload removal", path)
            deleted.append(path)
            freed = size
    except (BaseException,) as error:  # noqa: B013 -- preserve during release
        primary = error
        raise
    finally:
        try:
            lease.release()
        except (Exception,) as error:  # noqa: B013
            note = f"{lease_path}: temporary lease release failed: {error}"
            if primary is None:
                skipped.append(note)
            else:
                primary.add_note(note)
        else:
            if not dry_run:
                try:
                    lease_path.unlink(missing_ok=True)
                except (OSError,) as error:  # noqa: B013
                    note = f"{lease_path}: temporary lease companion removal failed: {error}"
                    if primary is None:
                        skipped.append(note)
                    else:
                        primary.add_note(note)
    return freed


def cleanup_score_temporaries(
    directory: Path, *, dry_run: bool = False
) -> ScoreTempCleanup:
    """Reclaim only inactive v1 stages; leave legacy/uncooperative writers alone.

    Directory and individual lease acquisition are nonblocking. A dry run may
    create lock metadata, but never explicitly unlinks payloads or companions.
    Deletion accounting survives ordinary errors while releasing either lock.
    """
    deleted: list[Path] = []
    skipped: list[str] = []
    freed = 0
    try:
        if not directory.exists():
            return ScoreTempCleanup()
        if not directory.is_dir():
            return ScoreTempCleanup(
                skipped=(f"temporary cleanup failed: not a directory: {directory}",)
            )
        candidates = _stage_payloads(directory)
    except (OSError,) as error:  # noqa: B013
        return ScoreTempCleanup(skipped=(f"temporary discovery failed: {error}",))
    for path in candidates:
        try:
            with score_write_lock(directory, timeout=0):
                try:
                    freed += _reclaim_stage(
                        path, dry_run=dry_run, deleted=deleted, skipped=skipped
                    )
                except (FileNotFoundError,):  # noqa: B013
                    pass
                except (OSError,) as error:  # noqa: B013
                    skipped.append(
                        f"{path.name}: temporary cleanup failed: {_cleanup_diagnostic(error)}"
                    )
        except (ScoreCoordinationTimeout,):  # noqa: B013
            skipped.append(
                "temporary cleanup deferred: score directory coordination is busy"
            )
            break
        except (ScoreCoordinationReleaseError,) as error:  # noqa: B013
            skipped.append(f"temporary cleanup stopped: {error}")
            break
        except (OSError,) as error:  # noqa: B013
            skipped.append(
                f"{path.name}: temporary cleanup failed: {_cleanup_diagnostic(error)}"
            )
    return ScoreTempCleanup(tuple(deleted), freed / 1e6, tuple(skipped))
