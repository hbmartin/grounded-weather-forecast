"""Sticky coordination failures across products, threads and process boundaries."""

import argparse
import gc
import multiprocessing
import os
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Never

import polars as pl
import pytest
from conftest import write_config
from score_recovery_helpers import (
    GuardFailureFactory,
    OrphanFactory,
    release_guards,
)

from grounded_weather_forecast import cli, runs
from grounded_weather_forecast.backtest import score_storage as storage
from grounded_weather_forecast.backtest.outcomes import ProductBacktestOutcome
from grounded_weather_forecast.config import Config
from grounded_weather_forecast.reports import operations
from grounded_weather_forecast.serve import selection
from grounded_weather_forecast.storage import pipeline_lock


@pytest.mark.parametrize("products", ["hourly", "hourly,daily"])
@pytest.mark.parametrize("phase", ["publication", "disposal"])
def test_committed_guard_failure_warns_then_rejects_next_product(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    products: str,
    phase: str,
) -> None:
    config = write_config(tmp_path)
    release = storage.FileLock.release
    held: list[storage.FileLock] = []
    records: list[runs.RunRecord] = []
    written: list[str] = []
    promoted: list[bool] = []
    calls = 0

    def fail_release(self: storage.FileLock, *, force: bool = False) -> None:
        nonlocal calls
        if Path(self.lock_file).name == storage.SCORE_WRITE_LOCK and self.is_locked:
            calls += 1
            if calls == (2 if phase == "publication" else 3):
                held.append(self)
                raise OSError("committed unlock fixture")
        release(self, force=force)

    def write(
        config: Config, _args: argparse.Namespace, product: str, _methods: list[str]
    ) -> ProductBacktestOutcome:
        with storage.score_parquet_destination(
            config.dataset.dir / "scores" / f"scores_{product}.parquet"
        ) as staged:
            written.append(product)
            staged.write_bytes(b"committed fixture")
        return ProductBacktestOutcome(product, "evaluated", 1)

    def promote(*_args: object, **_kwargs: object) -> dict[str, object]:
        promoted.append(True)
        return selection.PromotionResult({})

    monkeypatch.setattr(storage.FileLock, "release", fail_release)
    monkeypatch.setattr(cli, "_run_matrix_backtest", write)
    monkeypatch.setattr(selection, "promote_methods", promote)
    monkeypatch.setattr(
        runs, "append_run", lambda record, _path: records.append(record)
    )
    # Fail if a later acquisition reaches FileLock instead of the saved failure.
    monkeypatch.setattr(storage, "SCORE_COORDINATION_TIMEOUT_S", 0)
    try:
        code = cli.main(
            [
                "--config",
                str(tmp_path / "config.toml"),
                "backtest",
                "--methods",
                "equal_weight",
                "--products",
                products,
            ]
        )
        assert written == ["hourly"]
        assert (
            config.dataset.dir / "scores/scores_hourly.parquet"
        ).read_bytes() == b"committed fixture"
        assert "committed unlock fixture" in caplog.text
        if products == "hourly":
            assert code == 0
            assert promoted == [True]
            assert records[-1].failure_kind is None
        else:
            assert code == 1
            assert not promoted
            assert records[-1].failure_kind == "score_coordination_release"
            assert "committed unlock fixture" in (records[-1].error_detail or "")
            assert not (config.dataset.dir / "scores/scores_daily.parquet").exists()
            assert "timed out" not in capsys.readouterr().err
        assert held[0].is_locked
        with pipeline_lock(config.dataset.dir, timeout=0):
            pass
    finally:
        release_guards(held, release)


def test_failed_directory_is_shared_across_threads_and_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_guard_release: GuardFailureFactory,
) -> None:
    directory = tmp_path / "scores"
    original = storage.locked_path
    fail_guard_release(OSError("sticky fixture"))
    with (
        pytest.raises(storage.ScoreCoordinationReleaseError),
        storage.score_write_lock(directory),
    ):
        pass
    monkeypatch.setattr(storage, "locked_path", original)
    aliases = [directory]
    if os.name != "nt":
        alias = tmp_path / "alias"
        alias.symlink_to(directory, target_is_directory=True)
        aliases.append(alias)

    def denied(path: Path) -> storage.ScoreCoordinationReleaseError:
        with (
            pytest.raises(storage.ScoreCoordinationReleaseError) as caught,
            storage.score_write_lock(path, timeout=0),
        ):
            pytest.fail("failed directory entered")
        return caught.value

    def unexpected(*_args: object, **_kwargs: object) -> Never:
        pytest.fail("failed directory attempted OS acquisition")

    monkeypatch.setattr(storage, "locked_path", unexpected)
    with ThreadPoolExecutor(max_workers=1) as pool:
        errors = [pool.submit(denied, path).result(timeout=5) for path in aliases]
    assert len({id(error) for error in errors}) == len(errors)
    assert all("sticky fixture" in str(error) for error in errors)
    monkeypatch.setattr(storage, "locked_path", original)
    with storage.score_write_lock(tmp_path / "independent", timeout=0):
        pass


def test_primary_failure_is_preserved_and_directory_remains_failed(
    tmp_path: Path, fail_guard_release: GuardFailureFactory
) -> None:
    fail_guard_release(OSError("secondary unlock fixture"))
    with (
        pytest.raises(ValueError, match="primary fixture") as caught,
        storage.score_write_lock(tmp_path),
    ):
        raise ValueError("primary fixture")
    assert "secondary unlock fixture" in caught.value.__notes__[0]
    with (
        pytest.raises(
            storage.ScoreCoordinationReleaseError, match="secondary unlock fixture"
        ),
        storage.score_write_lock(tmp_path),
    ):
        pass


def test_failure_is_checked_before_reentrant_bypass(tmp_path: Path) -> None:
    with storage.score_write_lock(tmp_path):
        identity = next(iter(storage._held.directories))
        storage._failed_directories[identity] = (
            storage.ScoreCoordinationFailure.from_cause(
                tmp_path / storage.SCORE_WRITE_LOCK, OSError("nested fixture")
            )
        )
        with (
            pytest.raises(
                storage.ScoreCoordinationReleaseError, match="nested fixture"
            ),
            storage.score_write_lock(tmp_path),
        ):
            pass


def test_snapshots_and_partial_results_do_not_retain_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, orphan: OrphanFactory
) -> None:
    staged = orphan(tmp_path)
    release = storage.FileLock._release
    references: list[weakref.ReferenceType[storage.FileLock]] = []

    def fail_once(self: storage.FileLock) -> None:
        if Path(self.lock_file).name == storage.SCORE_WRITE_LOCK and not references:
            references.append(weakref.ref(self))
            raise OSError("lifetime fixture")
        release(self)

    def recover() -> tuple[storage.ScoreTempCleanup, storage.ScoreCoordinationFailure]:
        try:
            storage.cleanup_score_temporaries(tmp_path)
        except (storage.ScoreCoordinationReleaseError,) as error:  # noqa: B013
            if error.partial_result is None:
                pytest.fail("partial recovery progress missing")
            return error.partial_result, error.snapshot()
        pytest.fail("release failure was suppressed")

    enabled = gc.isenabled()
    gc.disable()
    try:
        monkeypatch.setattr(storage.FileLock, "_release", fail_once)
        partial, failure = recover()
        assert references[0]() is None
        assert partial.deleted == (staged,)
        assert "lifetime fixture" in failure.diagnostic()
        assert "1 payloads deleted, 6 bytes" in failure.notes[-1]
        assert storage._failed_directories
    finally:
        if enabled:
            gc.enable()


def probe_directory(directory: Path) -> None:
    """Run in a fresh interpreter, without the parent's failure markers."""
    with storage.score_write_lock(directory, timeout=0):
        pass


def test_fresh_process_can_use_failed_directory(
    tmp_path: Path, fail_guard_release: GuardFailureFactory
) -> None:
    fail_guard_release(OSError("parent fixture"))
    with (
        pytest.raises(storage.ScoreCoordinationReleaseError),
        storage.score_write_lock(tmp_path),
    ):
        pass
    process = multiprocessing.get_context("spawn").Process(
        target=probe_directory, args=(tmp_path,)
    )
    process.start()
    try:
        process.join(timeout=10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork reset")
def test_fork_resets_failed_registry_and_inherited_mutex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_guard_release: GuardFailureFactory,
) -> None:
    original = storage.locked_path
    fail_guard_release(OSError("fork fixture"))
    with (
        pytest.raises(storage.ScoreCoordinationReleaseError),
        storage.score_write_lock(tmp_path),
    ):
        pass
    monkeypatch.setattr(storage, "locked_path", original)
    storage._failed_directories_lock.acquire()
    try:
        process = multiprocessing.get_context("fork").Process(
            target=probe_directory, args=(tmp_path,)
        )
        process.start()
        process.join(timeout=5)
        assert process.exitcode == 0
    finally:
        storage._failed_directories_lock.release()
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
    assert storage._failed_directories


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("protected", [False, True])
def test_prune_failure_preserves_retention_counts_and_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    orphan: OrphanFactory,
    failed_score_guard_release: list[storage.FileLock],
    *,
    dry_run: bool,
    protected: bool,
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
    records: list[runs.RunRecord] = []
    monkeypatch.setattr(
        operations,
        "_protected_evaluations",
        lambda *_a, **_k: None if protected else set(),
    )
    monkeypatch.setattr(
        operations.evidence,
        "load_ledger",
        lambda *_args: pl.DataFrame({"evaluation_id": ["e0"]}),
    )
    monkeypatch.setattr(
        runs, "append_run", lambda record, _path: records.append(record)
    )
    arguments = ["--config", str(tmp_path / "config.toml"), "prune-scores"]
    if dry_run:
        arguments.append("--dry-run")
    assert cli.main(arguments) == 1
    assert first.exists() == dry_run
    assert second.exists()
    assert published[0].exists() == (dry_run or protected)
    assert all(path.exists() for path in published[1:])
    output = capsys.readouterr()
    assert "1 abandoned score temporary files" in output.out
    assert "unlock fixture" not in output.out
    assert output.err.count("directory guard release failed") == 1
    assert f"1 payloads {'eligible' if dry_run else 'deleted'}, 6 bytes" in output.err
    assert records[-1].failure_kind == "score_coordination_release"
    assert records[-1].exit_code == 1
    assert failed_score_guard_release[0].is_locked
    with pipeline_lock(config.dataset.dir, timeout=0):
        pass


def test_teardown_attempts_all_guards_after_failure(tmp_path: Path) -> None:
    guards = [storage.FileLock(tmp_path / f"guard-{i}.lock") for i in range(2)]
    for guard in guards:
        guard.acquire(timeout=0)
    attempted: list[storage.FileLock] = []

    def release(guard: storage.FileLock, *, force: bool = False) -> None:
        attempted.append(guard)
        guard.release(force=force)
        if guard is guards[0]:
            raise OSError("teardown fixture")

    try:
        with pytest.raises(
            ExceptionGroup, match="score guard teardown failures"
        ) as caught:
            release_guards(guards, release)
        assert attempted == guards
        assert all(not guard.is_locked for guard in guards)
        assert str(caught.value.exceptions[0]) == "teardown fixture"
    finally:
        release_guards(guards, storage.FileLock.release)


def test_secondary_unlock_failure_on_last_product_suppresses_promotion(
    tmp_path, monkeypatch
):
    config = write_config(tmp_path)
    original_publish = storage.publish_parquet
    original_release = storage.FileLock.release
    held = []
    failing = False
    promoted = []
    records = []

    def publish(staged, path):
        nonlocal failing
        if "daily" in path.name:
            failing = True
            raise OSError("primary publication failure")
        original_publish(staged, path)

    def release(self, *, force=False):
        if (
            failing
            and Path(self.lock_file).name == storage.SCORE_WRITE_LOCK
            and self.is_locked
        ):
            held.append(self)
            raise OSError("secondary unlock failure")
        original_release(self, force=force)

    def product(config, args, name, methods):
        with storage.score_parquet_destination(
            config.dataset.dir / "scores" / f"scores_{name}.parquet"
        ) as staged:
            staged.write_bytes(b"fixture")
        return ProductBacktestOutcome(name, "evaluated", 1)

    monkeypatch.setattr(storage, "publish_parquet", publish)
    monkeypatch.setattr(storage.FileLock, "release", release)
    monkeypatch.setattr(cli, "_run_matrix_backtest", product)
    monkeypatch.setattr(
        selection, "promote_methods", lambda *_a, **_k: promoted.append(True)
    )
    monkeypatch.setattr(
        runs, "append_run", lambda record, _path: records.append(record)
    )
    try:
        assert (
            cli.main(
                [
                    "--config",
                    str(tmp_path / "config.toml"),
                    "backtest",
                    "--products",
                    "hourly,daily",
                    "--methods",
                    "equal_weight",
                ]
            )
            == 1
        )
        assert not promoted
        assert records[-1].failure_kind == "score_coordination_release"
        assert "primary publication failure" in records[-1].error_detail
        assert "secondary unlock failure" in records[-1].error_detail
        with pipeline_lock(config.dataset.dir, timeout=0):
            pass
    finally:
        release_guards(held, original_release)
