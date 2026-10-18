"""Recovery fairness, resource diagnostics and retained best-effort disposal."""

import json
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier, Event, Thread

import polars as pl
import pytest
from conftest import write_config

from grounded_weather_forecast import cli
from grounded_weather_forecast.backtest import score_storage as storage
from grounded_weather_forecast.backtest.scores import empty_scores, write_scores
from grounded_weather_forecast.reports.operations import prune_scores_files
from grounded_weather_forecast.storage import temporary_sibling


def orphan(directory: Path, name: str = "orphan", contents: bytes = b"orphan") -> Path:
    path = temporary_sibling(
        directory / f"scores_{name}.parquet", namespace=storage.SCORE_STAGE_NAMESPACE
    )
    path.write_bytes(contents)
    return path


def test_enumeration_does_not_hold_directory_guard(tmp_path, monkeypatch):
    staged = orphan(tmp_path)
    discover = storage._stage_payloads

    def scan(directory):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                write_scores, empty_scores(), directory / "scores_writer.parquet"
            )
            future.result(timeout=5)
        assert staged.exists()
        return discover(directory)

    monkeypatch.setattr(storage, "_stage_payloads", scan)
    result = storage.cleanup_score_temporaries(tmp_path)
    assert result.deleted == (staged,)
    assert (tmp_path / "scores_writer.parquet").exists()


def test_writer_publishes_between_recovery_entries(tmp_path, monkeypatch):
    stages = [orphan(tmp_path, name) for name in ("first", "second")]
    monkeypatch.setattr(storage, "_stage_payloads", lambda _directory: stages)
    guard = storage.score_write_lock
    released, published = Event(), Event()
    failures = []

    def write():
        try:
            assert released.wait(timeout=5)
            # The second recovery entry has not been processed yet.
            assert stages[1].exists()
            write_scores(empty_scores(), tmp_path / "scores_writer.parquet")
        except (BaseException,) as error:  # noqa: B013 -- project convention
            failures.append(error)
        finally:
            published.set()

    writer = Thread(target=write)
    writer.start()
    recovery_entries = 0

    @contextmanager
    def coordinated(directory, **kwargs):
        nonlocal recovery_entries
        with guard(directory, **kwargs):
            yield
        if kwargs.get("timeout") == 0 and not released.is_set():
            recovery_entries += 1
            released.set()
            assert published.wait(timeout=5)

    monkeypatch.setattr(storage, "score_write_lock", coordinated)
    try:
        result = storage.cleanup_score_temporaries(tmp_path)
    finally:
        released.set()
        writer.join(timeout=5)
    assert not writer.is_alive()
    assert not failures
    assert recovery_entries == 1
    assert result.deleted == tuple(stages)
    assert pl.read_parquet(tmp_path / "scores_writer.parquet").is_empty()


def test_recovery_stops_and_keeps_accounting_when_next_entry_is_busy(
    tmp_path, monkeypatch
):
    first, second = orphan(tmp_path, "first"), orphan(tmp_path, "second")
    monkeypatch.setattr(storage, "_stage_payloads", lambda _directory: [first, second])
    guard = storage.score_write_lock
    calls = 0

    @contextmanager
    def busy(directory, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise storage.ScoreCoordinationTimeout(
                str(directory / storage.SCORE_WRITE_LOCK)
            )
        with guard(directory, **kwargs):
            yield

    monkeypatch.setattr(storage, "score_write_lock", busy)
    result = storage.cleanup_score_temporaries(tmp_path)
    assert result.deleted == (first,)
    assert result.freed_mb == 6 / 1e6
    assert "deferred" in result.skipped[0]
    assert second.exists()


@pytest.mark.parametrize("fault", ["enumeration", "validation", "payload", "companion"])
@pytest.mark.parametrize("entrypoint", ["cleanup", "prune", "cli"])
def test_programming_errors_escape_recovery(tmp_path, monkeypatch, fault, entrypoint):
    config = write_config(tmp_path)
    directory = config.dataset.dir / "scores"
    staged = orphan(directory)

    def bug(*_args, **_kwargs):
        raise TypeError("programming fixture")

    if fault == "enumeration":
        monkeypatch.setattr(storage, "_stage_payloads", bug)
    elif fault == "validation":
        monkeypatch.setattr(storage, "_regular", bug)
    else:
        unlink = Path.unlink
        target = (
            staged if fault == "payload" else staged.with_name(staged.name + ".lease")
        )

        def fail_target(path, *args, **kwargs):
            if path == target:
                bug()
            return unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_target)

    def invoke():
        if entrypoint == "cleanup":
            storage.cleanup_score_temporaries(directory)
        elif entrypoint == "prune":
            prune_scores_files(config, dry_run=False)
        else:
            cli.main(["--config", str(tmp_path / "config.toml"), "prune-scores"])

    with pytest.raises(TypeError, match="programming fixture"):
        invoke()


def test_discovery_os_error_is_diagnostic(tmp_path, monkeypatch):
    def unreadable(_directory):
        raise PermissionError("discovery fixture")

    monkeypatch.setattr(storage, "_stage_payloads", unreadable)
    result = storage.cleanup_score_temporaries(tmp_path)
    assert not result.deleted
    assert "temporary discovery failed" in result.skipped[0]
    assert "discovery fixture" in result.skipped[0]


def test_guard_release_failure_has_resource_cause_and_stops_recovery(
    tmp_path, monkeypatch
):
    first, second = orphan(tmp_path, "first"), orphan(tmp_path, "second")
    monkeypatch.setattr(storage, "_stage_payloads", lambda _directory: [first, second])
    guard = storage.locked_path
    cause = RuntimeError("unlock fixture")

    @contextmanager
    def faulty_guard(*args, **kwargs):
        with guard(*args, **kwargs):
            yield
        raise cause

    monkeypatch.setattr(storage, "locked_path", faulty_guard)
    with (
        pytest.raises(storage.ScoreCoordinationReleaseError) as caught,
        storage.score_write_lock(tmp_path),
    ):
        pass
    assert caught.value.lock_path == tmp_path / storage.SCORE_WRITE_LOCK
    assert caught.value.cause is cause
    assert caught.value.__cause__ is cause
    result = storage.cleanup_score_temporaries(tmp_path)
    assert result.deleted == (first,)
    assert result.freed_mb == 6 / 1e6
    assert second.exists()
    assert "directory guard release failed" in result.skipped[0]
    assert str(tmp_path / storage.SCORE_WRITE_LOCK) in result.skipped[0]
    assert "ownership uncertain" in result.skipped[0]


@pytest.mark.parametrize("phase", ["payload", "lease", "companion"])
def test_committed_warning_identifies_cleanup_phase(
    tmp_path, monkeypatch, caplog, phase
):
    destination = tmp_path / "scores_output.parquet"
    if phase == "lease":
        release = storage.FileLock.release

        def fail_release(self, *args, **kwargs):
            held = self.is_locked
            release(self, *args, **kwargs)
            if held and str(self.lock_file).endswith(".lease"):
                raise OSError("cleanup fixture")

        monkeypatch.setattr(storage.FileLock, "release", fail_release)
    else:
        unlink = Path.unlink

        def fail_unlink(path, *args, **kwargs):
            if path.name.endswith(".tmp" if phase == "payload" else ".lease"):
                raise OSError("cleanup fixture")
            return unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_unlink)
    write_scores(empty_scores(), destination)
    assert pl.read_parquet(destination).is_empty()
    expected = {
        "payload": "payload removal",
        "lease": "lease release",
        "companion": "lease companion removal",
    }[phase]
    assert expected in caplog.text
    assert str(tmp_path) in caplog.text
    assert "cleanup fixture" in caplog.text


@contextmanager
def guard_holder(directory: Path) -> Iterator[tuple[Thread, Event]]:
    """Always release the holder and join it only if the test started it."""
    ready, release = Event(), Event()
    guard = storage.score_write_lock
    failures: list[BaseException] = []

    def hold() -> None:
        try:
            with guard(directory):
                ready.set()
                assert release.wait(timeout=5)
        except (BaseException,) as error:  # noqa: B013 -- project convention
            failures.append(error)

    holder = Thread(target=hold)
    try:
        yield holder, ready
    finally:
        release.set()
        if holder.ident is not None:
            holder.join(timeout=5)
        assert not holder.is_alive()
        assert not failures


def test_thread_cleanup_preserves_failure_before_start(tmp_path: Path) -> None:
    with (
        pytest.raises(ValueError, match="before start fixture"),
        guard_holder(tmp_path) as (holder, _ready),
    ):
        raise ValueError("before start fixture")
    assert holder.ident is None
    assert not holder.is_alive()


@pytest.mark.parametrize(
    "primary", [None, ValueError, storage.ScoreCoordinationTimeout]
)
def test_zero_wait_disposal_and_later_recovery_are_retained(
    tmp_path, primary, monkeypatch, caplog
):
    dispose = storage._dispose_stage
    staged_path = []

    def contend(staged, lease):
        staged_path.append(staged)
        holder.start()
        assert ready.wait(timeout=5)
        return dispose(staged, lease)

    monkeypatch.setattr(storage, "_dispose_stage", contend)
    monkeypatch.setattr(storage, "SCORE_COORDINATION_TIMEOUT_S", 0.05)
    destination = tmp_path / "scores_output.parquet"
    destination.write_bytes(b"original")

    def fail_producer():
        with storage.score_parquet_destination(destination) as staged:
            staged.write_bytes(b"x" * 5000)
            raise ValueError("producer fixture")

    def timeout_publication():
        with storage.score_parquet_destination(destination) as staged:
            staged_path.append(staged)
            staged.write_bytes(b"x" * 5000)
            holder.start()
            assert ready.wait(timeout=5)
            monkeypatch.setattr(storage, "_dispose_stage", dispose)

    with guard_holder(tmp_path) as (holder, ready):
        if primary is None:
            write_scores(empty_scores(), destination)
            assert "cleanup deferred: directory guard busy" in caplog.text
            assert pl.read_parquet(destination).is_empty()
        elif primary is ValueError:
            with pytest.raises(ValueError, match="producer fixture") as caught:
                fail_producer()
            assert "could not be acquired" in caught.value.__notes__[0]
            assert destination.read_bytes() == b"original"
        else:
            with pytest.raises(storage.ScoreCoordinationTimeout) as caught:
                timeout_publication()
            assert "could not be acquired" in caught.value.__notes__[0]
            assert destination.read_bytes() == b"original"
        assert list(tmp_path.glob("*.lease"))
        if primary is not None:
            assert staged_path[0].stat().st_size == 5000
    result = storage.cleanup_score_temporaries(tmp_path)
    assert not result.skipped
    assert len(result.deleted) == (0 if primary is None else 1)
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.lease"))


def test_thread_local_bookkeeping_is_independent_and_typed(tmp_path):
    main_state = storage._held.directories
    with storage.score_write_lock(tmp_path):
        owned = storage._held.directories
        assert len(owned) == len(main_state) + 1
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert (
                pool.submit(lambda: storage._held.directories).result(timeout=5)
                == frozenset()
            )
        with storage.score_write_lock(tmp_path):
            assert storage._held.directories == owned
    assert storage._held.directories == main_state


@pytest.mark.parametrize("failing", [False, True])
@pytest.mark.parametrize("trial", range(3))
def test_repeated_writes_with_concurrent_recovery(
    tmp_path, failing, trial, record_property
):
    directory = tmp_path / "scores"
    directory.mkdir()
    destinations = [
        directory / f"scores_worker_{worker}.parquet" for worker in range(4)
    ]
    expected = [
        {
            f"worker-{worker}-iteration-{iteration}".encode().ljust(5000, b"!")
            for iteration in range(200)
        }
        for worker in range(4)
    ]
    for destination in destinations:
        destination.write_bytes(b"original")
        if os.name != "nt":
            destination.chmod(0o640)
    start, stop = Barrier(4), Event()

    def write(worker):
        start.wait(timeout=5)
        for iteration in range(200):
            contents = f"worker-{worker}-iteration-{iteration}".encode().ljust(
                5000, b"!"
            )
            try:
                with storage.score_parquet_destination(destinations[worker]) as staged:
                    assert staged.exists()
                    staged.write_bytes(contents)
                    assert staged.exists()
                    if failing:
                        raise ValueError("expected producer failure")
            except (ValueError,):  # noqa: B013 -- project convention
                if not failing:
                    raise

    def recover():
        while not stop.is_set():
            storage.cleanup_score_temporaries(directory)
            stop.wait(0.002)

    def read():
        while not stop.is_set():
            for worker, destination in enumerate(destinations):
                assert destination.read_bytes() in expected[worker] | {b"original"}
            stop.wait(0.002)

    with ThreadPoolExecutor(max_workers=6) as pool:
        recovery, reader = pool.submit(recover), pool.submit(read)
        writers = [pool.submit(write, worker) for worker in range(4)]
        try:
            for future in writers:
                future.result(timeout=30)
        finally:
            stop.set()
        recovery.result(timeout=5)
        reader.result(timeout=5)
    initial = {
        "trial": trial,
        "failing": failing,
        "payloads": len(list(directory.glob("*.tmp"))),
        "leases": len(list(directory.glob("*.lease"))),
    }
    (tmp_path / "stress-result.json").write_text(json.dumps(initial) + "\n")
    record_property("initial_leftovers", json.dumps(initial))
    result = storage.cleanup_score_temporaries(directory)
    assert not result.skipped
    assert not list(directory.glob("*.tmp"))
    assert not list(directory.glob("*.lease"))
    for worker, destination in enumerate(destinations):
        assert destination.read_bytes() == (
            b"original"
            if failing
            else f"worker-{worker}-iteration-199".encode().ljust(5000, b"!")
        )
        if os.name != "nt":
            assert destination.stat().st_mode & 0o777 == 0o640


def test_guard_release_failure_stops_after_a_handled_io_error(tmp_path, monkeypatch):
    first, second = orphan(tmp_path, "first"), orphan(tmp_path, "second")
    monkeypatch.setattr(storage, "_stage_payloads", lambda _directory: [first, second])
    unlink, guard = Path.unlink, storage.locked_path

    def fail_first(path, *args, **kwargs):
        if path == first:
            raise PermissionError("payload fixture")
        return unlink(path, *args, **kwargs)

    @contextmanager
    def fail_release(*args, **kwargs):
        with guard(*args, **kwargs):
            yield
        raise OSError("unlock fixture")

    monkeypatch.setattr(Path, "unlink", fail_first)
    monkeypatch.setattr(storage, "locked_path", fail_release)
    result = storage.cleanup_score_temporaries(tmp_path)
    assert not result.deleted
    assert first.exists()
    assert second.exists()
    assert "payload fixture" in result.skipped[0]
    assert "directory guard release failed" in result.skipped[1]
    assert "unlock fixture" in result.skipped[1]


@pytest.mark.parametrize(
    "primary", [TypeError, KeyboardInterrupt, SystemExit, GeneratorExit]
)
def test_secondary_guard_release_failure_preserves_primary(
    tmp_path, monkeypatch, primary
):
    guard = storage.locked_path

    @contextmanager
    def fail_release(*args, **kwargs):
        with guard(*args, **kwargs):
            yield
        raise RuntimeError("unlock fixture")

    monkeypatch.setattr(storage, "locked_path", fail_release)
    with pytest.raises(primary) as caught, storage.score_write_lock(tmp_path):
        raise primary(23)
    assert caught.value.args == (23,)
    assert "directory guard release failed" in caught.value.__notes__[0]
    assert "unlock fixture" in caught.value.__notes__[0]
    assert str(tmp_path / storage.SCORE_WRITE_LOCK) in caught.value.__notes__[0]


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_new_guard_release_interrupt_propagates_with_original_context(
    tmp_path, monkeypatch, interrupt
):
    guard = storage.locked_path

    @contextmanager
    def interrupt_release(*args, **kwargs):
        with guard(*args, **kwargs):
            yield
        raise interrupt(23)

    monkeypatch.setattr(storage, "locked_path", interrupt_release)
    with pytest.raises(interrupt) as caught, storage.score_write_lock(tmp_path):
        raise ValueError("primary fixture")
    assert caught.value.args == (23,)
    assert isinstance(caught.value.__context__, ValueError)


@pytest.mark.parametrize(
    "primary", [TypeError, KeyboardInterrupt, SystemExit, GeneratorExit]
)
@pytest.mark.parametrize("secondary_phase", ["lease", "companion"])
@pytest.mark.parametrize("secondary_error", [OSError, TypeError, RuntimeError])
def test_recovery_secondary_failure_annotates_pending_primary(
    tmp_path, monkeypatch, primary, secondary_phase, secondary_error
):
    staged = orphan(tmp_path)
    unlink, release = Path.unlink, storage.FileLock.release

    def fail_unlink(path, *args, **kwargs):
        if path == staged:
            raise primary(23)
        if secondary_phase == "companion" and path.name.endswith(".lease"):
            raise secondary_error("secondary fixture")
        return unlink(path, *args, **kwargs)

    def fail_release(self, *args, **kwargs):
        held = self.is_locked
        release(self, *args, **kwargs)
        if (
            secondary_phase == "lease"
            and held
            and str(self.lock_file).endswith(".lease")
        ):
            raise secondary_error("secondary fixture")

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    monkeypatch.setattr(storage.FileLock, "release", fail_release)
    with pytest.raises(primary) as caught:
        storage.cleanup_score_temporaries(tmp_path)
    assert caught.value.args == (23,)
    assert any("secondary fixture" in note for note in caught.value.__notes__)
    assert any(str(staged) in note for note in caught.value.__notes__)
