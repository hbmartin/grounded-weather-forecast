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

import polars as pl
import pytest
from conftest import write_config

from grounded_weather_forecast import cli, runs
from grounded_weather_forecast.backtest import engine, scores
from grounded_weather_forecast.backtest import score_storage as storage
from grounded_weather_forecast.serve import selection
from grounded_weather_forecast.storage import pipeline_lock, temporary_sibling


def orphan(directory, name="orphan", contents=b"orphan"):
    staged = temporary_sibling(
        directory / f"scores_{name}.parquet", namespace=storage.SCORE_STAGE_NAMESPACE
    )
    staged.write_bytes(contents)
    return staged


@pytest.mark.parametrize("completed", [0, 1])
def test_directory_acquisition_error_stops_once_with_partial_accounting(
    tmp_path, monkeypatch, completed
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
def test_vanished_candidate_does_not_create_companion(tmp_path, monkeypatch, dry_run):
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
    tmp_path, monkeypatch, dry_run, completed
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
def test_unexpected_error_carries_completed_progress(tmp_path, monkeypatch, dry_run):
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
    tmp_path, monkeypatch, interrupt
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
    tmp_path, monkeypatch, caplog
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
        assert "retry failed" not in caplog.text
    finally:
        release(lease)


@pytest.mark.parametrize("failures", [1, 2])
def test_release_retry_reports_attempts_and_final_state(
    tmp_path, monkeypatch, failures
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
    finally:
        release(lease)
    assert storage._lease_path(staged).exists()
    assert not storage.cleanup_score_temporaries(tmp_path).skipped
    assert not storage._lease_path(staged).exists()


def test_release_error_pickle_retains_metadata_and_notes(tmp_path):
    error = storage.ScoreCoordinationReleaseError(
        tmp_path / "guard.lock", OSError("cause fixture")
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


def raise_release_error(path):
    error = storage.ScoreCoordinationReleaseError(path, OSError("worker fixture"))
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
    assert str(caught.value.cause) == "worker fixture"
    assert caught.value.__notes__ == ["worker note"]


@pytest.mark.parametrize("command", ["backtest", "maintain", "recover"])
def test_cli_release_failure_is_handled_and_journaled(
    tmp_path, monkeypatch, capsys, command
):
    config = write_config(tmp_path)
    directory = config.dataset.dir / "scores"
    directory.mkdir(parents=True)
    pl.DataFrame({"source_kind": ["live"]}).write_parquet(
        config.dataset.dir / "hourly_matrix_live.parquet"
    )
    destination = directory / "scores_hourly_live_expanding_fixed.parquet"
    destination.write_bytes(b"original")
    if os.name != "nt":
        destination.chmod(0o640)
    closed, promoted, records = [], [], []
    guard = storage.locked_path

    @contextmanager
    def faulty_guard(*args, **kwargs):
        with guard(*args, **kwargs):
            yield
        raise OSError("guard release fixture")

    def produce(*_args):
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

    monkeypatch.setattr(storage, "locked_path", faulty_guard)
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


def test_thread_cleanup_preserves_failure_before_start(tmp_path, monkeypatch):
    from test_score_recovery_diagnostics import (  # noqa: PLC0415
        test_zero_wait_disposal_and_later_recovery_are_retained as exercise,
    )

    def fail(*_args, **_kwargs):
        raise ValueError("before start fixture")

    monkeypatch.setitem(exercise.__globals__, "write_scores", fail)
    with pytest.raises(ValueError, match="before start fixture"):
        exercise(tmp_path, None, monkeypatch, None)
