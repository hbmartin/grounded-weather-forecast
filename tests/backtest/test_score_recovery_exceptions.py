"""Recovery progress and failure diagnostics across library and CLI boundaries."""

import multiprocessing
import os
import pickle
import signal
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Never

import polars as pl
import pytest
from conftest import synthetic_hourly_matrix, write_config
from score_recovery_helpers import GuardFailureFactory, OrphanFactory

from grounded_weather_forecast import cli, runs
from grounded_weather_forecast.backtest import engine, scores
from grounded_weather_forecast.backtest import score_storage as storage
from grounded_weather_forecast.contracts import (
    HOURLY_VARIABLES,
    TruthSemantics,
    truth_col,
)
from grounded_weather_forecast.reports import operations
from grounded_weather_forecast.serve import selection
from grounded_weather_forecast.storage import pipeline_lock


@pytest.mark.parametrize("phase", ["lease", "companion"])
@pytest.mark.parametrize("secondary_error", [OSError, TypeError])
def test_disappeared_payload_preserves_secondary_cleanup_failure(
    orphan: OrphanFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    secondary_error: type[Exception],
) -> None:
    first, second = orphan(tmp_path, "first"), orphan(tmp_path, "second")
    unlink, release = Path.unlink, storage.FileLock.release

    def fail_unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path == first:
            unlink(path)
            raise FileNotFoundError("payload disappeared")
        if phase == "companion" and path == storage._lease_path(first):
            raise secondary_error("secondary fixture")
        unlink(path, missing_ok=missing_ok)

    def fail_release(self: storage.FileLock, *, force: bool = False) -> None:
        held = self.is_locked
        release(self, force=force)
        if (
            phase == "lease"
            and held
            and Path(self.lock_file) == storage._lease_path(first)
        ):
            raise secondary_error("secondary fixture")

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    monkeypatch.setattr(storage.FileLock, "release", fail_release)
    if secondary_error is TypeError:
        with pytest.raises(TypeError, match="secondary fixture") as caught:
            storage.cleanup_score_temporaries(tmp_path)
        assert second.exists()
        assert "0 payloads deleted, 0 bytes" in caught.value.__notes__[-1]
    else:
        result = storage.cleanup_score_temporaries(tmp_path)
        assert result.deleted == (second,)
        assert result.freed_mb == 6 / 1e6
        assert len(result.skipped) == 1
        assert "secondary fixture" in result.skipped[0]
        assert str(storage._lease_path(first)) in result.skipped[0]
    assert not first.exists()


@pytest.mark.parametrize("dry_run", [False, True])
def test_recovery_raises_typed_release_failure_and_partial_progress(
    orphan: OrphanFactory,
    tmp_path: Path,
    failed_score_guard_release: list[storage.FileLock],
    *,
    dry_run: bool,
) -> None:
    first, second = orphan(tmp_path, "first"), orphan(tmp_path, "second")
    with pytest.raises(storage.ScoreCoordinationReleaseError) as caught:
        storage.cleanup_score_temporaries(tmp_path, dry_run=dry_run)
    error = caught.value
    result = error.partial_result
    assert result is not None
    assert error.lock_path == tmp_path / storage.SCORE_WRITE_LOCK
    assert isinstance(error.cause, OSError)
    assert error.__cause__ is error.cause
    assert (
        f"1 payloads {'eligible' if dry_run else 'deleted'}, 6 bytes"
        in error.__notes__[-1]
    )
    assert result.deleted == (first,)
    assert result.freed_mb == 6 / 1e6
    assert first.exists() == dry_run
    assert second.exists()
    assert not result.skipped
    assert len(failed_score_guard_release) == 1
    assert failed_score_guard_release[0].is_locked
    with (
        pytest.raises(storage.ScoreCoordinationReleaseError, match="unlock fixture"),
        storage.score_write_lock(tmp_path, timeout=0),
    ):
        pass


def test_pruning_continues_after_recovery_guard_release_failure(
    orphan: OrphanFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_score_guard_release: list[storage.FileLock],
) -> None:
    config = write_config(tmp_path)
    directory = config.dataset.dir / "scores"
    first, second = orphan(directory, "first"), orphan(directory, "second")
    published = [
        directory / f"scores_hourly_live_expanding_e{i}.parquet" for i in range(4)
    ]
    for i, path in enumerate(published):
        path.write_bytes(b"published")
        os.utime(path, (i + 1, i + 1))
    monkeypatch.setattr(
        operations, "_protected_evaluations", lambda *_args, **_kwargs: set()
    )
    monkeypatch.setattr(
        operations.evidence,
        "load_ledger",
        lambda *_args: pl.DataFrame({"evaluation_id": ["e0"]}),
    )
    result = operations.prune_scores_files(config, dry_run=False)
    assert result.temporary_deleted == (first,)
    assert result.temporary_freed_mb == 6 / 1e6
    assert second.exists()
    assert result.deleted == (published[0],)
    assert not published[0].exists()
    assert all(path.exists() for path in published[1:])
    assert result.coordination_release_failure is not None
    assert "unlock fixture" in result.coordination_release_failure.diagnostic()
    assert failed_score_guard_release[0].is_locked


@pytest.mark.parametrize("completed", [0, 1])
def test_directory_acquisition_error_stops_once_with_partial_accounting(
    orphan: OrphanFactory, tmp_path, monkeypatch, completed
):
    stages = [orphan(tmp_path, str(i)) for i in range(3)]
    monkeypatch.setattr(storage, "_stage_payloads", lambda _directory: stages)
    guard = storage.locked_path
    attempts = []

    @contextmanager
    def denied(*args, **kwargs):
        attempts.append(True)
        if len(attempts) > completed:
            raise PermissionError("directory fixture")
        with guard(*args, **kwargs):
            yield

    monkeypatch.setattr(storage, "locked_path", denied)
    result = storage.cleanup_score_temporaries(tmp_path)
    assert len(attempts) == completed + 1
    assert result.deleted == tuple(stages[:completed])
    assert result.freed_mb == completed * 6 / 1e6
    assert len(result.skipped) == 1
    assert "score directory coordination failed" in result.skipped[0]
    assert str(tmp_path) in result.skipped[0]
    assert all(stage.name not in result.skipped[0] for stage in stages)
    assert all(stage.exists() for stage in stages[completed:])


@pytest.mark.parametrize("dry_run", [False, True])
def test_vanished_candidate_does_not_create_companion(
    orphan: OrphanFactory, tmp_path, monkeypatch, dry_run
):
    staged = orphan(tmp_path)
    discover = storage._stage_payloads

    def vanished(directory):
        candidates = discover(directory)
        staged.unlink()
        return candidates

    monkeypatch.setattr(storage, "_stage_payloads", vanished)
    assert (
        storage.cleanup_score_temporaries(tmp_path, dry_run=dry_run)
        == storage.ScoreTempCleanup()
    )
    assert not storage._lease_path(staged).exists()


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("completed", [0, 1])
def test_observed_directory_disappearance_stops_without_recreation(
    orphan: OrphanFactory, tmp_path, monkeypatch, dry_run, completed
):
    directory = tmp_path / "scores"
    stages = [orphan(directory, str(i)) for i in range(2)]
    monkeypatch.setattr(storage, "_stage_payloads", lambda _directory: stages)
    guard = storage.score_write_lock
    exists = Path.exists
    probes = 0

    def disappear(path):
        nonlocal probes
        if path == directory:
            probes += 1
            # Initial discovery checks existence once, then each entry checks it.
            if probes == completed + 2:
                for child in directory.iterdir():
                    child.unlink()
                directory.rmdir()
        return exists(path)

    monkeypatch.setattr(Path, "exists", disappear)
    result = storage.cleanup_score_temporaries(directory, dry_run=dry_run)
    assert not exists(directory)
    assert result.deleted == tuple(stages[:completed])
    assert result.freed_mb == completed * 6 / 1e6
    assert len(result.skipped) == 1
    assert "directory disappeared" in result.skipped[0]
    # A normal writer continues to create its output directory.
    with guard(directory):
        assert exists(directory)


@pytest.mark.parametrize("dry_run", [False, True])
def test_unexpected_error_carries_completed_progress(
    orphan: OrphanFactory, tmp_path, monkeypatch, dry_run
):
    first, second = orphan(tmp_path, "first"), orphan(tmp_path, "second")
    monkeypatch.setattr(storage, "_stage_payloads", lambda _directory: [first, second])
    regular = storage._regular

    def faulty_validation(path):
        if path == second:
            raise TypeError("validation fixture")
        return regular(path)

    monkeypatch.setattr(storage, "_regular", faulty_validation)
    with pytest.raises(TypeError, match="validation fixture") as caught:
        storage.cleanup_score_temporaries(tmp_path, dry_run=dry_run)
    assert (
        f"1 payloads {'eligible' if dry_run else 'deleted'}, 6 bytes"
        in caught.value.__notes__[-1]
    )
    assert first.exists() == dry_run
    assert second.exists()


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_new_companion_interrupt_retains_original_context(
    orphan: OrphanFactory, tmp_path, monkeypatch, interrupt
):
    staged = orphan(tmp_path)
    unlink = Path.unlink

    def fail(path, *args, **kwargs):
        if path == staged:
            raise ValueError("primary fixture")
        if path == storage._lease_path(staged):
            raise interrupt(23)
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail)
    with pytest.raises(interrupt) as caught:
        storage.cleanup_score_temporaries(tmp_path)
    assert caught.value.args == (23,)
    assert isinstance(caught.value.__context__, ValueError)
    assert "0 payloads deleted, 0 bytes" in caught.value.__notes__[-1]


@pytest.mark.parametrize("interrupt", ["KeyboardInterrupt", "SystemExit"])
def test_companion_secondary_failure_preserves_process_status(tmp_path, interrupt):
    worker = """
import builtins,sys
from pathlib import Path
from grounded_weather_forecast.backtest import score_storage as storage
from grounded_weather_forecast.storage import temporary_sibling
directory = Path(sys.argv[1])
staged = temporary_sibling(directory / 'scores_test.parquet', namespace=storage.SCORE_STAGE_NAMESPACE)
staged.write_bytes(b'fixture')
unlink = Path.unlink
def faulty(path, *args, **kwargs):
    if path == staged:
        raise getattr(builtins, sys.argv[2])(23)
    if path == storage._lease_path(staged):
        raise RuntimeError('secondary fixture')
    return unlink(path, *args, **kwargs)
Path.unlink = faulty
storage.cleanup_score_temporaries(directory)
"""
    result = subprocess.run(  # noqa: S603 -- fixed fixture script, no external input
        [sys.executable, "-c", worker, str(tmp_path), interrupt],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    expected = (
        (130 if os.name == "nt" else -signal.SIGINT)
        if interrupt == "KeyboardInterrupt"
        else 23
    )
    assert result.returncode == expected
    assert "secondary fixture" in result.stderr or interrupt == "SystemExit"


def test_timeout_warning_includes_secondary_notes(tmp_path, caplog):
    error = storage.ScoreCoordinationTimeout("directory-lock-fixture")
    error.add_note("score lease release retry failed: lease-fixture")
    storage._warn_committed_cleanup(tmp_path / "scores_output.parquet", error)
    assert "directory guard busy" in caplog.text
    assert "directory-lock-fixture" in caplog.text
    assert "lease-fixture" in caplog.text


def test_busy_guard_reports_failed_first_fallback_release(
    orphan: OrphanFactory, tmp_path, monkeypatch, caplog
):
    staged = orphan(tmp_path)
    lease = storage.FileLock(storage._lease_path(staged))
    lease.acquire(timeout=0)
    release = storage.FileLock.release

    @contextmanager
    def busy(*_args, **_kwargs):
        raise storage.ScoreCoordinationTimeout("directory fixture")
        yield  # pragma: no cover -- context manager never enters

    def fail_release(self, *args, **kwargs):
        if self is lease:
            raise OSError("fallback fixture")
        return release(self, *args, **kwargs)

    monkeypatch.setattr(storage, "score_write_lock", busy)
    monkeypatch.setattr(storage.FileLock, "release", fail_release)
    try:
        with pytest.raises(storage.ScoreCoordinationTimeout) as caught:
            storage._dispose_stage(staged, lease)
        storage._warn_committed_cleanup(
            tmp_path / "scores_output.parquet", caught.value
        )
        assert "fallback attempt failed" in caplog.text
        assert "fallback fixture" in caplog.text
        assert "still held" in caplog.text
        assert "companion removal not completed" in caplog.text
        assert "retry failed" not in caplog.text
    finally:
        release(lease)


@pytest.mark.parametrize("failures", [1, 2])
def test_release_retry_reports_attempts_and_final_state(
    orphan: OrphanFactory, tmp_path, monkeypatch, failures
):
    staged = orphan(tmp_path)
    lease = storage.FileLock(storage._lease_path(staged))
    lease.acquire(timeout=0)
    release = storage.FileLock.release
    attempts = 0

    def flaky(self, *args, **kwargs):
        nonlocal attempts
        if self is lease:
            attempts += 1
            if attempts <= failures:
                raise OSError("lease fixture")
        return release(self, *args, **kwargs)

    monkeypatch.setattr(storage.FileLock, "release", flaky)
    try:
        with pytest.raises(OSError, match="lease fixture") as caught:
            storage._dispose_stage(staged, lease)
        assert attempts == 2
        assert lease.is_locked == (failures == 2)
        diagnostic = storage._cleanup_diagnostic(caught.value)
        assert "first attempt" in diagnostic
        assert f"retry {'succeeded' if failures == 1 else 'failed'}" in diagnostic
        assert ("reported released" if failures == 1 else "still held") in diagnostic
        assert "companion removal not completed" in diagnostic
    finally:
        release(lease)
    assert storage._lease_path(staged).exists()
    assert not storage.cleanup_score_temporaries(tmp_path).skipped
    assert not storage._lease_path(staged).exists()


def test_release_error_pickle_retains_metadata_and_notes(tmp_path):
    error = storage.ScoreCoordinationReleaseError(
        tmp_path / "guard.lock",
        OSError("cause fixture"),
        partial_result=storage.ScoreTempCleanup(
            (tmp_path / "payload",), 0.5, ("prior",)
        ),
    )
    error.add_note("note fixture")
    restored = pickle.loads(pickle.dumps(error))  # noqa: S301 -- local fixture only
    assert type(restored) is type(error)
    assert str(restored) == str(error)
    assert restored.lock_path == error.lock_path
    assert type(restored.cause) is OSError
    assert str(restored.cause) == "cause fixture"
    assert restored.__cause__ is restored.cause
    assert restored.__notes__ == ["note fixture"]
    assert restored.partial_result == error.partial_result


def raise_release_error(path: Path) -> Never:
    error = storage.ScoreCoordinationReleaseError(path, OSError("worker fixture"))
    error.partial_result = storage.ScoreTempCleanup((path.parent / "payload",), 0.5)
    error.add_note("worker note")
    raise error


def test_release_error_crosses_process_pool(tmp_path):
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        future = pool.submit(raise_release_error, tmp_path / "guard.lock")
        with pytest.raises(storage.ScoreCoordinationReleaseError) as caught:
            future.result(timeout=20)
    assert caught.value.lock_path == tmp_path / "guard.lock"
    assert isinstance(caught.value.cause, OSError)
    assert str(caught.value.cause) == "worker fixture"
    assert caught.value.__notes__ == ["worker note"]
    assert caught.value.partial_result == storage.ScoreTempCleanup(
        (tmp_path / "payload",), 0.5
    )


@pytest.mark.parametrize("initially_held", [False, True])
def test_disposal_guard_error_reports_completed_companion_removal(
    orphan: OrphanFactory,
    tmp_path: Path,
    fail_guard_release: GuardFailureFactory,
    *,
    initially_held: bool,
) -> None:
    staged = orphan(tmp_path)
    lease = storage.FileLock(storage._lease_path(staged))
    if initially_held:
        lease.acquire(timeout=0)
    fail_guard_release(OSError("guard release fixture"))
    try:
        with pytest.raises(storage.ScoreCoordinationReleaseError) as caught:
            storage._dispose_stage(staged, lease)
        detail = storage._cleanup_diagnostic(caught.value)
        state = "reported released" if initially_held else "not held on entry"
        assert f": {state}; companion removal completed" in detail
        assert "deferred" not in detail
        assert not staged.exists()
        assert not storage._lease_path(staged).exists()
        assert not lease.is_locked
    finally:
        lease.release(force=True)


@pytest.mark.parametrize("command", ["backtest", "maintain", "recover"])
def test_startup_release_failure_stops_writers_and_is_journaled(
    orphan: OrphanFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failed_score_guard_release: list[storage.FileLock],
    command: str,
) -> None:
    config = write_config(tmp_path)
    directory = config.dataset.dir / "scores"
    first, second = orphan(directory, "first"), orphan(directory, "second")
    records: list[runs.RunRecord] = []
    writers: list[bool] = []
    promoted: list[bool] = []

    def write(*_args: object) -> int:
        writers.append(True)
        with storage.score_write_lock(directory):
            return 1

    monkeypatch.setattr(storage, "SCORE_COORDINATION_TIMEOUT_S", 0.01)
    monkeypatch.setattr(cli, "_run_matrix_backtest", write)
    monkeypatch.setattr(cli, "_run_minutely_backtest", write)
    monkeypatch.setattr(cli, "_cmd_build_dataset", lambda *_args: 0)
    monkeypatch.setattr(cli, "_recovery_needed", lambda *_args: True)
    monkeypatch.setattr(
        selection, "select_methods", lambda *_a, **_k: promoted.append(True)
    )
    monkeypatch.setattr(
        runs, "append_run", lambda record, _path: records.append(record)
    )
    arguments = [
        "--config",
        str(tmp_path / "config.toml"),
        command,
        "--methods",
        "equal_weight",
    ]
    if command != "recover":
        arguments.extend(["--products", "hourly"])
    assert cli.main(arguments) == 1
    assert not writers
    assert not promoted
    assert not first.exists()
    assert second.exists()
    assert len(failed_score_guard_release) == 1
    assert failed_score_guard_release[0].is_locked
    with pipeline_lock(config.dataset.dir, timeout=0):
        pass
    output = capsys.readouterr()
    assert "deleted 1 abandoned score temporary" in output.out
    assert "1 payloads deleted, 6 bytes" in output.err
    assert "timed out" not in output.err
    assert "unlock fixture" not in output.out
    assert output.err.count("directory guard release failed") == 1
    failures = [record for record in records if record.exit_code == 1]
    assert len(failures) == (1 if command == "backtest" else 2)
    for record in failures:
        assert record.failure_kind == "score_coordination_release"
        assert record.error_detail is not None
        assert "unlock fixture" in record.error_detail
        assert ".score-stage-v1.lock" in record.error_detail
        assert "1 payloads deleted, 6 bytes" in record.error_detail


@pytest.mark.parametrize("command", ["backtest", "maintain", "recover"])
def test_cli_release_failure_is_handled_and_journaled(
    tmp_path, monkeypatch, capsys, command, fail_guard_release: GuardFailureFactory
):
    config = write_config(tmp_path)
    directory = config.dataset.dir / "scores"
    directory.mkdir(parents=True)
    synthetic_hourly_matrix(days=1, max_lead=1).with_columns(
        *(
            pl.lit(1.0).alias(
                truth_col(
                    variable.name,
                    TruthSemantics.INSTANTANEOUS
                    if variable.has_dual_semantics
                    else None,
                )
            )
            for variable in HOURLY_VARIABLES
        )
    ).write_parquet(config.dataset.dir / "hourly_matrix_live.parquet")
    destination = directory / "scores_hourly_live_expanding_fixed.parquet"
    destination.write_bytes(b"original")
    if os.name != "nt":
        destination.chmod(0o640)
    closed, promoted, records = [], [], []

    def produce(*_args, **_kwargs):
        try:
            yield (
                scores.empty_scores()
                .clear(1)
                .with_columns(
                    pl.lit("fixed").alias("evaluation_id"),
                    pl.lit("hourly").alias("product"),
                    pl.lit("live").alias("source_kind"),
                    pl.lit("expanding").alias("window"),
                )
                .cast(scores.COMPACT_SCORES_SCHEMA)
            )
        finally:
            closed.append(True)

    fail_guard_release(OSError("guard release fixture"))
    monkeypatch.setattr(engine, "iter_backtest_scores", produce)
    monkeypatch.setattr(
        selection, "select_methods", lambda *_a, **_k: promoted.append(True)
    )
    monkeypatch.setattr(
        runs, "append_run", lambda record, _path: records.append(record)
    )
    monkeypatch.setattr(cli, "_cmd_build_dataset", lambda *_args: 0)
    monkeypatch.setattr(cli, "_recovery_needed", lambda *_args: True)
    arguments = [
        "--config",
        str(tmp_path / "config.toml"),
        command,
        "--methods",
        "equal_weight",
    ]
    if command != "recover":
        arguments.extend(["--products", "hourly"])
    assert cli.main(arguments) == 1
    assert closed == [True]
    assert not promoted
    assert destination.read_bytes() == b"original"
    if os.name != "nt":
        assert destination.stat().st_mode & 0o777 == 0o640
    with pipeline_lock(config.dataset.dir, timeout=0):
        pass
    assert "directory guard release failed" in capsys.readouterr().err
    failures = [record for record in records if record.exit_code == 1]
    assert len(failures) == (1 if command == "backtest" else 2)
    for record in failures:
        assert record.failure_kind == "score_coordination_release"
        assert "guard release fixture" in record.error_detail
        assert ".score-stage-v1.lock" in record.error_detail
