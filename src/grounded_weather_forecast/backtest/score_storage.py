"""Independent score publication leases and recovery of unpublished files."""

import logging
import os
import stat
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

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


@dataclass(frozen=True, slots=True)
class ScoreTempCleanup:
    """Removed (or dry-run eligible) payloads and nonfatal diagnostics."""

    deleted: tuple[Path, ...] = ()
    freed_mb: float = 0.0
    skipped: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ScoreCoordinationFailure:
    """Serializable diagnostics without exception frames or lock references."""

    lock_path: Path
    cause_type: str
    cause_message: str
    notes: tuple[str, ...] = ()

    @classmethod
    def from_cause(cls, lock_path: Path, cause: BaseException) -> Self:
        return cls(
            lock_path,
            f"{type(cause).__module__}.{type(cause).__qualname__}",
            str(cause),
            tuple(getattr(cause, "__notes__", ())),
        )

    def exception(self) -> "ScoreCoordinationReleaseError":
        error = ScoreCoordinationReleaseError(
            self.lock_path, RuntimeError(f"{self.cause_type}: {self.cause_message}")
        )
        for note in self.notes:
            error.add_note(note)
        return error

    def diagnostic(self) -> str:
        return _cleanup_diagnostic(self.exception())


_failed_directories: dict[_DirectoryIdentity, ScoreCoordinationFailure] = {}
_failed_directories_lock = threading.Lock()


def _reset_failed_directories() -> None:
    """A fork child must not inherit a failure marker or a locked mutex."""
    global _failed_directories, _failed_directories_lock  # noqa: PLW0603
    _failed_directories = {}
    _failed_directories_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_failed_directories)


def _check_guard_failure(identity: _DirectoryIdentity) -> None:
    with _failed_directories_lock:
        failure = _failed_directories.get(identity)
    if failure is not None:
        raise failure.exception()


def check_score_coordination(directory: Path) -> None:
    """Raise a saved guard failure without acquiring or creating any resource.

    Product exception handlers use this to detect secondary release failures.
    Successful committed writes deliberately do not call this probe.
    """
    try:
        metadata = directory.stat()
    except (FileNotFoundError,):  # noqa: B013 -- no directory can have a saved owner
        return
    _check_guard_failure((os.getpid(), metadata.st_dev, metadata.st_ino))


class ScoreCoordinationTimeout(Timeout):
    """A brief score-directory bookkeeping lock could not be acquired in time."""


class ScoreCoordinationProbeError(RuntimeError):
    """A secondary diagnostic cannot establish whether shared ownership is safe."""

    def __init__(self, directory: Path, primary: Exception, probe: OSError) -> None:
        self.directory = directory
        self.primary_type = type(primary).__name__
        self.primary_message = str(primary)
        self.probe_type = type(probe).__name__
        self.probe_message = str(probe)
        super().__init__(
            f"score coordination check failed for {directory}: {self.probe_type}: "
            f"{self.probe_message}; primary failure: {self.primary_type}: {self.primary_message}"
        )
        for note in (
            *getattr(primary, "__notes__", ()),
            *getattr(probe, "__notes__", ()),
        ):
            self.add_note(note)


class ScoreCoordinationReleaseError(RuntimeError):
    """A guard release failed; the lock's ownership may be uncertain."""

    def __init__(
        self,
        lock_path: Path,
        cause: Exception,
        *,
        partial_result: ScoreTempCleanup | None = None,
    ) -> None:
        self.lock_path = lock_path
        self.cause = cause
        self.__cause__ = cause
        self.partial_result = partial_result
        super().__init__(
            f"directory guard release failed for {lock_path}; "
            f"ownership uncertain: {cause}"
        )

    def __reduce__(
        self,
    ) -> tuple[type[Self], tuple[Path, Exception], dict[str, object]]:
        return type(self), (self.lock_path, self.cause), self.__dict__

    def snapshot(self) -> ScoreCoordinationFailure:
        failure = ScoreCoordinationFailure.from_cause(self.lock_path, self.cause)
        return ScoreCoordinationFailure(
            failure.lock_path,
            failure.cause_type,
            failure.cause_message,
            (*failure.notes, *getattr(self, "__notes__", ())),
        )


def _release_guard(
    guard: AbstractContextManager[None],
    lock_path: Path,
    identity: _DirectoryIdentity,
) -> None:
    try:
        guard.__exit__(None, None, None)
    except (BaseException,) as error:  # noqa: B013 -- record uncertain ownership
        failure = ScoreCoordinationFailure.from_cause(lock_path, error)
        with _failed_directories_lock:
            _failed_directories.setdefault(identity, failure)
        if isinstance(error, Exception):
            raise ScoreCoordinationReleaseError(lock_path, error) from error
        raise


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
    _check_guard_failure(identity)
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
        _check_guard_failure(identity)
        raise ScoreCoordinationTimeout(error.lock_file) from error
    _held.directories = held | {identity}
    try:
        yield
    except (BaseException,) as error:  # noqa: B013
        try:
            _release_guard(guard, directory / SCORE_WRITE_LOCK, identity)
        except (Exception,) as secondary:  # noqa: B013
            error.add_note(f"score coordination release failed: {secondary}")
        raise
    else:
        _release_guard(guard, directory / SCORE_WRITE_LOCK, identity)
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
            _cleanup_diagnostic(error),
        )
    else:
        _logger.warning("published %s; %s", path, _cleanup_diagnostic(error))


def _dispose_stage(temporary: Path, lease: FileLock) -> None:
    """Release ownership even when bookkeeping fails; leftovers are recoverable."""
    lease_path = _lease_path(temporary)
    held_on_entry = lease.is_locked
    lease_attempted = False
    companion_removed = False
    try:
        with score_write_lock(temporary.parent, timeout=0):
            _cleanup_action(
                lambda: temporary.unlink(missing_ok=True), "payload removal", temporary
            )
            lease_attempted = True
            _cleanup_action(lease.release, "lease release (first attempt)", lease_path)
            _cleanup_action(
                lambda: lease_path.unlink(missing_ok=True),
                "lease companion removal",
                lease_path,
            )
            companion_removed = True
    except (BaseException,) as error:  # noqa: B013
        if lease.is_locked:
            attempt = "retry" if lease_attempted else "fallback attempt"
            try:
                lease.release()
            except (Exception,) as secondary:  # noqa: B013
                error.add_note(
                    f"score lease release {attempt} failed for {lease_path}: "
                    f"{_cleanup_diagnostic(secondary)}"
                )
            else:
                error.add_note(
                    f"score lease release {attempt} succeeded for {lease_path}"
                )
        ownership = "not held on entry"
        if lease.is_locked:
            ownership = "still held"
        elif held_on_entry:
            ownership = "reported released"
        error.add_note(
            f"score lease ownership for {lease_path}: "
            f"{ownership}; companion removal "
            f"{'completed' if companion_removed else 'not completed'}"
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


@dataclass
class _RecoveryProgress:
    deleted: list[Path] = field(default_factory=list)
    freed_bytes: int = 0
    skipped: list[str] = field(default_factory=list)

    def diagnostic(self, note: str, primary: BaseException | None = None) -> None:
        if primary is None:
            self.skipped.append(note)
        else:
            primary.add_note(note)

    def annotate(self, error: BaseException, *, dry_run: bool) -> None:
        error.add_note(
            f"score temporary recovery partial progress: {len(self.deleted)} "
            f"payloads {'eligible' if dry_run else 'deleted'}, "
            f"{self.freed_bytes} bytes"
        )

    def result(self) -> ScoreTempCleanup:
        return ScoreTempCleanup(
            tuple(self.deleted),
            self.freed_bytes / 1e6,
            tuple(self.skipped),
        )


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


def _reclaim_stage(path: Path, *, dry_run: bool, progress: _RecoveryProgress) -> None:
    """Account for payload deletion before releasing a potentially faulty lease."""
    lease_path = _lease_path(path)
    if not path.exists() and not lease_path.exists():
        return
    if (
        path.is_symlink()
        or lease_path.is_symlink()
        or (path.exists() and not _regular(path))
        or (lease_path.exists() and not _regular(lease_path))
    ):
        return
    lease = FileLock(lease_path)
    primary: BaseException | None = None
    try:
        lease.acquire(timeout=0)
    except (Timeout,):  # noqa: B013
        progress.diagnostic(f"{path.name}: temporary cleanup: score writer is active")
        return
    try:
        if path.exists():
            size = path.stat().st_size
            if not dry_run:
                _cleanup_action(path.unlink, "payload removal", path)
            progress.deleted.append(path)
            progress.freed_bytes += size
    except (FileNotFoundError,):  # noqa: B013 -- a vanished payload is benign
        pass
    except (BaseException,) as error:  # noqa: B013 -- preserve during release
        primary = error
        raise
    finally:
        try:
            lease.release()
        except (Exception,) as error:  # noqa: B013
            if primary is None and not isinstance(error, OSError):
                raise
            note = f"{lease_path}: temporary lease release failed: {error}"
            progress.diagnostic(note, primary)
        else:
            if not dry_run:
                try:
                    lease_path.unlink(missing_ok=True)
                except (Exception,) as error:  # noqa: B013 -- secondary cleanup
                    if primary is None and not isinstance(error, OSError):
                        raise
                    note = f"{lease_path}: temporary lease companion removal failed: {error}"
                    progress.diagnostic(note, primary)


def _recover_candidates(
    directory: Path,
    candidates: list[Path],
    *,
    dry_run: bool,
    progress: _RecoveryProgress,
) -> None:
    for path in candidates:
        try:
            if not directory.exists():
                progress.diagnostic(
                    f"temporary cleanup stopped: score directory disappeared: {directory}"
                )
                break
            with score_write_lock(directory, timeout=0):
                try:
                    _reclaim_stage(path, dry_run=dry_run, progress=progress)
                except (FileNotFoundError,):  # noqa: B013
                    pass
                except (OSError,) as error:  # noqa: B013
                    progress.diagnostic(
                        f"{path.name}: temporary cleanup failed: {_cleanup_diagnostic(error)}"
                    )
        except (ScoreCoordinationTimeout,):  # noqa: B013
            progress.diagnostic(
                "temporary cleanup deferred: score directory coordination is busy"
            )
            break
        except (OSError,) as error:  # noqa: B013
            progress.diagnostic(
                f"temporary cleanup stopped: score directory coordination failed for "
                f"{directory}: {_cleanup_diagnostic(error)}"
            )
            break


def cleanup_score_temporaries(
    directory: Path, *, dry_run: bool = False
) -> ScoreTempCleanup:
    """Reclaim only inactive v1 stages; leave legacy/uncooperative writers alone.

    Directory and individual lease acquisition are nonblocking. A dry run may
    create lock metadata, but never explicitly unlinks payloads or companions.
    Deletion accounting survives ordinary errors while releasing either lock.
    A guard-release failure raises with a structured partial result so a caller
    must explicitly handle uncertain ownership. Unexpected errors carry the same
    partial-progress note. Observed directory disappearance stops recovery;
    removal after the existence check can race with parent-directory creation
    by score_write_lock, locked_path or FileLock.
    """
    progress = _RecoveryProgress()
    try:
        try:
            if not directory.exists():
                return progress.result()
            if not directory.is_dir():
                progress.diagnostic(
                    f"temporary cleanup failed: not a directory: {directory}"
                )
                return progress.result()
            candidates = _stage_payloads(directory)
        except (OSError,) as error:  # noqa: B013
            progress.diagnostic(f"temporary discovery failed: {error}")
            return progress.result()
        _recover_candidates(directory, candidates, dry_run=dry_run, progress=progress)
    except (BaseException,) as error:  # noqa: B013 -- annotate, never suppress
        progress.annotate(error, dry_run=dry_run)
        if isinstance(error, ScoreCoordinationReleaseError):
            error.partial_result = progress.result()
        raise
    return progress.result()
