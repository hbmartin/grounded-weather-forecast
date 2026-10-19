"""Process-level ownership, hard-stop recovery, and lazy writer dependencies."""

import os
import queue
import signal
import stat
import subprocess
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import Thread

import polars as pl
import pytest
from conftest import write_config
from filelock import Timeout

from grounded_weather_forecast import cli
from grounded_weather_forecast.backtest import engine, score_storage, scores
from grounded_weather_forecast.backtest.score_storage import (
    SCORE_STAGE_NAMESPACE,
    SCORE_WRITE_LOCK,
    ScoreCoordinationReleaseError,
    ScoreCoordinationTimeout,
    cleanup_score_temporaries,
    score_write_lock,
)
from grounded_weather_forecast.backtest.scores import (
    empty_scores,
    load_scores,
    write_scores,
)
from grounded_weather_forecast.reports.operations import prune_scores_files
from grounded_weather_forecast.serve import selection
from grounded_weather_forecast.storage import pipeline_lock, temporary_sibling

ROOT = Path(__file__).resolve().parents[2]
TEMP_NAME = (
    ".score-stage-v1.scores_hourly_live_expanding_fixed.parquet.0123456789abcdef.tmp"
)
DESTINATION = "scores_hourly_live_expanding_fixed.parquet"
WORKER = """
import sys
from contextlib import contextmanager
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from benchmark_backtest_memory import fixture_batches
from grounded_weather_forecast.backtest import scores, score_storage
import polars as pl
directory = Path(sys.argv[1])
frame = next(fixture_batches(scores, int(sys.argv[5]), compact=True))
frame = frame.with_columns(pl.lit(sys.argv[6]).alias("evaluation_id")).cast(scores.COMPACT_SCORES_SCHEMA)
phase = sys.argv[4]
def pause():
    if phase == 'noisy':
        sys.stderr.write('x' * (1024 * 1024 + 1))
        sys.stderr.flush()
    print('READY', flush=True)
    sys.stdin.readline()
allocate = score_storage.temporary_sibling
def observed_allocate(*args, **kwargs):
    path = allocate(*args, **kwargs)
    if phase == 'allocated':
        pause()
    return path
score_storage.temporary_sibling = observed_allocate
publish = score_storage.publish_parquet
def observed_publish(*args, **kwargs):
    publish(*args, **kwargs)
    if phase == 'published':
        pause()
score_storage.publish_parquet = observed_publish
original_stage = scores.score_parquet_destination
@contextmanager
def observed_stage(path):
    print('WAITING', flush=True)
    with original_stage(path) as temporary:
        yield temporary
        if phase in ('writing', 'noisy'):
            pause()
scores.score_parquet_destination = observed_stage
if phase == 'legacy':
    from grounded_weather_forecast.storage import stage_parquet, publish_parquet
    print('WAITING', flush=True)
    path = directory / 'scores_legacy.parquet'
    temporary = stage_parquet(frame, path)
    try:
        pause()
        publish_parquet(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
elif sys.argv[2] == 'stream':
    scores.SCORE_BATCH_ROWS = 1
    scores.write_score_batches(iter([frame]), directory, 'hourly', 'live', 'expanding')
else:
    scores.write_scores(frame.cast(scores.SCORES_SCHEMA), directory / f'scores_hourly_live_expanding_{sys.argv[6]}.parquet')
"""


def stderr_tail(process):
    stream = process.score_stderr
    stream.flush()
    stream.seek(max(0, stream.seek(0, 2) - 8192))
    return stream.read().decode(errors="replace")


@pytest.fixture
def writer_processes():
    processes = []
    stack = ExitStack()

    def launch(
        directory,
        mode="stream",
        *,
        ready=True,
        phase="writing",
        rows=2,
        evaluation_id="fixed",
    ):
        errors = stack.enter_context(tempfile.TemporaryFile(mode="w+b"))  # noqa: SIM115 -- fixture owns streams until process teardown
        process = subprocess.Popen(  # noqa: S603 -- local fixture, constructed argv
            [
                sys.executable,
                "-c",
                WORKER,
                str(directory),
                mode,
                str(ROOT / "scripts"),
                phase,
                str(rows),
                evaluation_id,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
            env=os.environ
            | {
                "POLARS_MAX_THREADS": "2",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
        )
        process.score_stderr = errors
        processes.append(process)
        lines = queue.Queue()

        def collect():
            for line in process.stdout:
                lines.put(line.strip())
            lines.put(None)

        Thread(target=collect, daemon=True).start()
        assert lines.get(timeout=20) == "WAITING", stderr_tail(process)
        if ready:
            assert lines.get(timeout=20) == "READY", stderr_tail(process)
        return process, lines

    yield launch
    for process in processes:
        if process.poll() is None:
            process.kill()
    for process in processes:
        process.wait(timeout=20)
        process.stdin.close()
        process.stdout.close()
    stack.close()


def release(process):
    process.stdin.write("release\n")
    process.stdin.flush()
    assert process.wait(timeout=20) == 0, stderr_tail(process)


def payloads(directory):
    return sorted(directory.glob(f".{SCORE_STAGE_NAMESPACE}.*.tmp"))


@pytest.mark.parametrize("mode", ["stream", "materialized"])
def test_active_library_writers_are_independent(tmp_path, writer_processes, mode):
    process, _ = writer_processes(tmp_path, mode)
    (temporary,) = payloads(tmp_path)
    before = temporary.read_bytes()
    cleanup = cleanup_score_temporaries(tmp_path)
    assert not cleanup.deleted
    assert "score writer is active" in cleanup.skipped[0]
    assert temporary.read_bytes() == before
    # Brief directory guard is free throughout encoding/producer closure.
    with score_write_lock(tmp_path, timeout=0):
        pass
    other, _ = writer_processes(tmp_path, rows=3)
    assert len(payloads(tmp_path)) == 2
    independent, _ = writer_processes(tmp_path / "independent")
    release(independent)
    release(process)
    release(other)
    assert load_scores(tmp_path / DESTINATION).height == 3
    assert not payloads(tmp_path)
    assert (tmp_path / SCORE_WRITE_LOCK).is_file()


@pytest.mark.skipif(os.name == "nt", reason="POSIX interrupt semantics")
def test_waiting_writer_can_be_interrupted(tmp_path, writer_processes):
    with score_write_lock(tmp_path):
        waiting, lines = writer_processes(tmp_path, ready=False)
        with pytest.raises(queue.Empty):
            lines.get(timeout=0.1)
        waiting.send_signal(signal.SIGINT)
        assert waiting.wait(timeout=20) in (-signal.SIGINT, 130), stderr_tail(waiting)
    assert not payloads(tmp_path)


def test_noisy_worker_stderr_cannot_block_handshake(tmp_path, writer_processes):
    process, _ = writer_processes(tmp_path, phase="noisy")
    assert process.score_stderr.seek(0, 2) > 1024 * 1024
    assert len(stderr_tail(process)) == 8192
    release(process)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal and mode semantics")
@pytest.mark.parametrize("stop_name", ["SIGTERM", "SIGKILL"])
@pytest.mark.parametrize("command", ["backtest", "prune-scores"])
def test_next_cli_run_reclaims_hard_stopped_scores(
    tmp_path, writer_processes, stop_name, command, monkeypatch, capsys
):
    stop = getattr(signal, stop_name)
    config = write_config(tmp_path)
    directory = config.dataset.dir / "scores"
    directory.mkdir(parents=True)
    destination = directory / DESTINATION
    destination.write_bytes(b"existing destination")
    destination.chmod(0o640)
    process, _ = writer_processes(directory)
    process.send_signal(stop)
    assert process.wait(timeout=20) == -stop
    temporary = payloads(directory)
    assert len(temporary) == 1
    assert temporary[0].stat().st_size > 0
    assert destination.read_bytes() == b"existing destination"
    promoted = []
    monkeypatch.setattr(
        selection, "select_methods", lambda *_a, **_k: promoted.append(True)
    )
    args = ["--config", str(tmp_path / "config.toml"), command]
    if command == "backtest":
        args.extend(["--products", "hourly", "--methods", "equal_weight"])
    code = cli.main(args)
    assert code == (1 if command == "backtest" else 0)
    assert "deleted 1 abandoned score temporary files" in capsys.readouterr().out
    assert not temporary[0].exists()
    assert destination.read_bytes() == b"existing destination"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640
    assert not promoted


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink fixture")
def test_reclamation_is_exact_and_dry_run_keeps_files(tmp_path):
    stale = temporary_sibling(tmp_path / DESTINATION, namespace=SCORE_STAGE_NAMESPACE)
    stale.write_bytes(b"abandoned")
    untouched = [
        tmp_path / DESTINATION,
        tmp_path / ".unrelated.parquet.0123456789abcdef.tmp",
        tmp_path / ".scores_hourly_live_expanding_fixed.parquet.0123456789abcdef.tmp",
        tmp_path / TEMP_NAME.replace("0123456789abcdef", "FEDCBA9876543210"),
        tmp_path / TEMP_NAME.replace("0123456789abcdef", "1234"),
    ]
    for path in untouched:
        path.write_bytes(b"preserve")
    subdirectory = tmp_path / TEMP_NAME.replace("fixed", "directory")
    subdirectory.mkdir()
    link = tmp_path / TEMP_NAME.replace("fixed", "link")
    link.symlink_to(untouched[0])
    dry = cleanup_score_temporaries(tmp_path, dry_run=True)
    assert dry.deleted == (stale,)
    assert dry.freed_mb == len(b"abandoned") / 1e6
    assert stale.exists()
    assert cleanup_score_temporaries(tmp_path) == dry
    assert not stale.exists()
    assert all(path.read_bytes() == b"preserve" for path in untouched)
    assert subdirectory.is_dir()
    assert link.is_symlink()


def test_missing_directory_is_not_created(tmp_path):
    missing = tmp_path / "missing"
    assert not cleanup_score_temporaries(missing).deleted
    assert not missing.exists()


def test_cleanup_lock_failure_is_diagnostic(tmp_path):
    invalid = tmp_path / "file"
    invalid.touch()
    result = cleanup_score_temporaries(invalid)
    assert not result.deleted
    assert "temporary cleanup failed" in result.skipped[0]


def test_disappeared_temporary_is_harmless(tmp_path, monkeypatch):
    stale = temporary_sibling(tmp_path / DESTINATION, namespace=SCORE_STAGE_NAMESPACE)
    stale.touch()
    unlink = Path.unlink

    def disappeared(path, *args, **kwargs):
        unlink(path, *args, **kwargs)
        if not kwargs.get("missing_ok", False):
            raise FileNotFoundError(path)

    monkeypatch.setattr(Path, "unlink", disappeared)
    result = cleanup_score_temporaries(tmp_path)
    assert not result.deleted
    assert not result.skipped


def test_cleanup_error_is_reported_and_does_not_block_other_files(
    tmp_path, monkeypatch
):
    blocked = tmp_path / TEMP_NAME
    good = tmp_path / TEMP_NAME.replace("fixed", "other")
    blocked.write_bytes(b"blocked")
    good.write_bytes(b"good")
    unlink = Path.unlink

    def fail_one(path, *args, **kwargs):
        if path == blocked:
            raise PermissionError("fixture permission failure")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_one)
    result = cleanup_score_temporaries(tmp_path)
    assert result.deleted == (good,)
    assert result.freed_mb == 4 / 1e6
    assert "fixture permission failure" in result.skipped[0]
    assert blocked.exists()


def test_prune_recovery_does_not_require_catalog_or_active_release(tmp_path):
    config = write_config(tmp_path)
    directory = config.dataset.dir / "scores"
    directory.mkdir(parents=True)
    stale = temporary_sibling(directory / DESTINATION, namespace=SCORE_STAGE_NAMESPACE)
    stale.write_bytes(b"orphan")
    dry = prune_scores_files(config, dry_run=True)
    assert dry.temporary_deleted == (stale,)
    assert dry.temporary_freed_mb == 6 / 1e6
    assert dry.deleted == ()
    assert dry.freed_mb == 0
    assert stale.exists()
    result = prune_scores_files(config, dry_run=False)
    assert result.temporary_deleted == (stale,)
    assert not stale.exists()


def test_cli_prune_lists_temporary_dry_run_separately(tmp_path, capsys):
    config = write_config(tmp_path)
    directory = config.dataset.dir / "scores"
    directory.mkdir(parents=True)
    stale = temporary_sibling(directory / DESTINATION, namespace=SCORE_STAGE_NAMESPACE)
    stale.touch()
    assert (
        cli.main(
            ["--config", str(tmp_path / "config.toml"), "prune-scores", "--dry-run"]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "would delete 0 scores files" in output
    assert "would delete 1 abandoned score temporary files" in output
    assert stale.exists()


def test_reader_imports_do_not_load_pyarrow_but_writer_does(tmp_path):
    script = """
import importlib.abc
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
import polars
class BlockArrow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pyarrow' or fullname.startswith('pyarrow.'):
            raise AssertionError('reader imported ' + fullname)
block = BlockArrow()
sys.meta_path.insert(0, block)
from grounded_weather_forecast.backtest import scores
from grounded_weather_forecast.serve import score_scan, selection
assert not any(name == 'pyarrow' or name.startswith('pyarrow.') for name in sys.modules)
sys.meta_path.remove(block)
result = scores.write_score_batches(iter([]), Path(sys.argv[1]), 'hourly', 'live', 'expanding')
assert 'pyarrow.parquet' in sys.modules
assert scores.load_scores(result.path).schema == scores.SCORES_SCHEMA
"""
    completed = subprocess.run(  # noqa: S603 -- local import probe
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("kind", "expected"), [("KeyboardInterrupt", 130), ("SystemExit", 23)]
)
def test_cleanup_interrupt_process_status(kind, expected):
    script = f"""
from grounded_weather_forecast.backtest.scores import _close_callback
def close():
    raise {kind}(23)
with _close_callback(close):
    raise ValueError('original error')
"""
    completed = subprocess.run(  # noqa: S603 -- local interruption probe
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode in (
        {-signal.SIGINT, expected} if kind == "KeyboardInterrupt" else {expected}
    )
    if kind == "KeyboardInterrupt":
        assert "original error" in completed.stderr


def test_nested_writes_and_same_thread_recovery_protect_active_payload(tmp_path):
    destination = tmp_path / DESTINATION
    with (
        score_write_lock(tmp_path),
        score_storage.score_parquet_destination(destination) as staged,
    ):
        staged.write_bytes(b"outer")
        # Reentrant directory guard, independent non-reentrant payload leases.
        result = cleanup_score_temporaries(tmp_path)
        assert not result.deleted
        assert "score writer is active" in result.skipped[0]
        write_scores(empty_scores(), tmp_path / "scores_daily_live.parquet")
        assert staged.read_bytes() == b"outer"
    assert destination.read_bytes() == b"outer"
    assert not payloads(tmp_path)


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory aliases")
def test_directory_alias_is_reentrant(tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with score_write_lock(tmp_path), score_write_lock(alias, timeout=0):
        write_scores(empty_scores(), alias / DESTINATION)


@pytest.mark.parametrize("phase", ["allocated", "published"])
@pytest.mark.parametrize("stop_name", ["SIGTERM", "SIGKILL"])
@pytest.mark.skipif(os.name == "nt", reason="POSIX hard-stop semantics")
def test_recovery_handles_allocation_and_published_lease_remnants(
    tmp_path, writer_processes, phase, stop_name
):
    process, _ = writer_processes(tmp_path, phase=phase)
    assert cleanup_score_temporaries(tmp_path).skipped
    process.send_signal(getattr(signal, stop_name))
    process.wait(timeout=20)
    if phase == "allocated":
        assert len(payloads(tmp_path)) == 1
        assert not list(tmp_path.glob("*.lease"))
    else:
        assert not payloads(tmp_path)
        assert list(tmp_path.glob("*.lease"))
    result = cleanup_score_temporaries(tmp_path)
    assert len(result.deleted) == (1 if phase == "allocated" else 0)
    assert result.freed_mb == 0
    assert not result.skipped
    assert not payloads(tmp_path)
    assert not list(tmp_path.glob("*.lease"))
    if phase == "published":
        assert load_scores(tmp_path / DESTINATION).height == 2


def test_old_reaper_cannot_match_new_namespace_and_new_reaper_ignores_legacy(tmp_path):
    legacy = tmp_path / ".scores_hourly_live.parquet.0123456789abcdef.tmp"
    legacy.write_bytes(b"live legacy writer")
    with score_storage.score_parquet_destination(tmp_path / DESTINATION) as staged:
        staged.write_bytes(b"new")
        assert list(tmp_path.glob(".scores_*.parquet.*.tmp")) == [legacy]
        for path in tmp_path.glob(".scores_*.parquet.*.tmp"):
            if path != legacy:
                path.unlink()
        assert staged.exists()
        assert not cleanup_score_temporaries(tmp_path).deleted
    assert legacy.read_bytes() == b"live legacy writer"
    assert not cleanup_score_temporaries(tmp_path).deleted


@pytest.mark.parametrize("command", ["backtest", "maintain", "recover"])
def test_cli_coordination_timeout_closes_producer_and_releases_pipeline(
    tmp_path, writer_processes, monkeypatch, capsys, command
):
    config = write_config(tmp_path)
    directory = config.dataset.dir / "scores"
    holder, _ = writer_processes(directory, phase="allocated")
    pl.DataFrame({"source_kind": ["live"]}).write_parquet(
        config.dataset.dir / "hourly_matrix_live.parquet"
    )
    closed = []
    promoted = []

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

    monkeypatch.setattr(engine, "iter_backtest_scores", produce)
    monkeypatch.setattr(score_storage, "SCORE_COORDINATION_TIMEOUT_S", 0.05)
    monkeypatch.setattr(
        selection, "select_methods", lambda *_a, **_k: promoted.append(True)
    )
    # Keep the real maintain dispatcher and journal, bypass preceding work only.
    for name in ("_cmd_build_dataset", "_cmd_truth_qc"):
        monkeypatch.setattr(cli, name, lambda *_a, **_k: 0)
    monkeypatch.setattr(cli, "_recovery_needed", lambda *_a, **_k: True)
    arguments = [
        "--config",
        str(tmp_path / "config.toml"),
        command,
        "--methods",
        "equal_weight",
    ]
    if command != "recover":
        arguments.extend(["--products", "hourly"])
    code = cli.main(arguments)
    assert code == 75
    assert closed == [True]
    assert not promoted
    assert "score directory coordination timed out" in capsys.readouterr().err
    with pipeline_lock(config.dataset.dir, timeout=0):
        pass
    holder.terminate()
    holder.wait(timeout=20)


def test_only_acquisition_timeout_is_translated(tmp_path):
    with pytest.raises(Timeout) as caught, score_write_lock(tmp_path):
        raise Timeout("body")
    assert not isinstance(caught.value, ScoreCoordinationTimeout)


@pytest.mark.parametrize("phase", ["guard", "lease", "companion"])
@pytest.mark.parametrize("error_type", [OSError, RuntimeError])
def test_cleanup_preserves_deleted_accounting_on_release_failure(
    tmp_path, monkeypatch, phase, error_type
):
    staged = temporary_sibling(tmp_path / DESTINATION, namespace=SCORE_STAGE_NAMESPACE)
    staged.write_bytes(b"accounted")
    if phase == "guard":
        original = score_storage.locked_path

        @contextmanager
        def guard(*args, **kwargs):
            with original(*args, **kwargs):
                yield
            raise error_type("release fixture")

        monkeypatch.setattr(score_storage, "locked_path", guard)
    elif phase == "lease":
        original = score_storage.FileLock.release

        def release_lease(self, *args, **kwargs):
            was_locked = self.is_locked
            original(self, *args, **kwargs)
            if was_locked and str(self.lock_file).endswith(".lease"):
                raise error_type("release fixture")

        monkeypatch.setattr(score_storage.FileLock, "release", release_lease)
    else:
        original = Path.unlink

        def unlink(path, *args, **kwargs):
            if path.name.endswith(".lease"):
                raise error_type("release fixture")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", unlink)
    if phase == "guard":
        with pytest.raises(ScoreCoordinationReleaseError) as caught:
            cleanup_score_temporaries(tmp_path)
        assert caught.value.partial_result is not None
        assert caught.value.partial_result.deleted == (staged,)
        assert caught.value.partial_result.freed_mb == 9 / 1e6
        assert "1 payloads deleted, 9 bytes" in caught.value.__notes__[-1]
        return
    if error_type is RuntimeError:
        with pytest.raises(RuntimeError, match="release fixture") as caught:
            cleanup_score_temporaries(tmp_path)
        assert not staged.exists()
        assert "1 payloads deleted, 9 bytes" in caught.value.__notes__[-1]
        return
    result = cleanup_score_temporaries(tmp_path)
    assert result.deleted == (staged,)
    assert result.freed_mb == 9 / 1e6
    assert "release fixture" in result.skipped[0]


def test_dry_run_does_not_explicitly_unlink_payload_or_lease(tmp_path, monkeypatch):
    staged = temporary_sibling(tmp_path / DESTINATION, namespace=SCORE_STAGE_NAMESPACE)
    staged.write_bytes(b"preview")

    def fail_unlink(*_args, **_kwargs):
        pytest.fail("dry run attempted unlink")

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    result = cleanup_score_temporaries(tmp_path, dry_run=True)
    assert result.deleted == (staged,)
    assert result.freed_mb == 7 / 1e6
    assert not result.skipped


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_stage_cleanup_interrupt_precedence(tmp_path, monkeypatch, interrupt):
    original = score_storage._dispose_stage

    def dispose(*args):
        original(*args)
        raise interrupt(23)

    monkeypatch.setattr(score_storage, "_dispose_stage", dispose)
    with (
        pytest.raises(interrupt) as caught,
        score_storage.score_parquet_destination(tmp_path / DESTINATION),
    ):
        raise ValueError("original")
    assert isinstance(caught.value.__context__, ValueError)
    assert caught.value.args == (23,)


def test_committed_housekeeping_failure_is_warning_not_write_failure(
    tmp_path, monkeypatch, caplog
):
    original = score_storage._dispose_stage

    def dispose(*args):
        original(*args)
        raise OSError("housekeeping fixture")

    monkeypatch.setattr(score_storage, "_dispose_stage", dispose)
    write_scores(empty_scores(), tmp_path / DESTINATION)
    assert load_scores(tmp_path / DESTINATION).is_empty()
    assert "housekeeping fixture" in caplog.text


def test_primary_failure_survives_ordinary_stage_cleanup_error(tmp_path, monkeypatch):
    original = score_storage._dispose_stage

    def dispose(*args):
        original(*args)
        raise OSError("cleanup fixture")

    monkeypatch.setattr(score_storage, "_dispose_stage", dispose)
    with (
        pytest.raises(ValueError, match="original") as caught,
        score_storage.score_parquet_destination(tmp_path / DESTINATION),
    ):
        raise ValueError("original")
    assert "cleanup fixture" in caught.value.__notes__[0]


def test_legacy_active_writer_survives_recovery(tmp_path, writer_processes):
    process, _ = writer_processes(tmp_path, phase="legacy")
    (legacy,) = tmp_path.glob(".scores_legacy.parquet.*.tmp")
    before = legacy.read_bytes()
    assert not cleanup_score_temporaries(tmp_path).deleted
    assert legacy.read_bytes() == before
    release(process)
    assert (tmp_path / "scores_legacy.parquet").exists()


def test_directory_coordination_does_not_block_other_directories(
    tmp_path, writer_processes
):
    holder, _ = writer_processes(tmp_path, phase="allocated")
    independent, _ = writer_processes(tmp_path / "independent")
    release(independent)
    release(holder)


@pytest.mark.parametrize("phase", ["allocation", "publication", "cleanup"])
def test_directory_release_error_preserves_failure_or_commit(
    tmp_path, monkeypatch, caplog, phase
):
    original = score_storage.locked_path
    calls = 0

    @contextmanager
    def guard(*args, **kwargs):
        nonlocal calls
        calls += 1
        current = calls
        with original(*args, **kwargs):
            yield
        if current == {"allocation": 1, "publication": 2, "cleanup": 3}[phase]:
            raise OSError("guard release fixture")

    monkeypatch.setattr(score_storage, "locked_path", guard)
    if phase == "allocation":
        with pytest.raises(
            ScoreCoordinationReleaseError, match="guard release fixture"
        ):
            write_scores(empty_scores(), tmp_path / DESTINATION)
        assert not (tmp_path / DESTINATION).exists()
    else:
        write_scores(empty_scores(), tmp_path / DESTINATION)
        assert load_scores(tmp_path / DESTINATION).is_empty()
        assert "guard release fixture" in caplog.text
    if phase == "allocation":
        # Failed coordination prevents another cleanup acquisition in this process.
        assert len(payloads(tmp_path)) == 1
    else:
        assert not payloads(tmp_path)


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_primary_interrupt_survives_failed_lease_release(
    tmp_path, monkeypatch, interrupt
):
    original_unlink = Path.unlink
    original_release = score_storage.FileLock.release

    def unlink(path, *args, **kwargs):
        if path.name.endswith(".tmp"):
            raise interrupt(23)
        return original_unlink(path, *args, **kwargs)

    def release_lease(self, *args, **kwargs):
        held = self.is_locked
        original_release(self, *args, **kwargs)
        if held and str(self.lock_file).endswith(".lease"):
            raise OSError("lease release fixture")

    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(score_storage.FileLock, "release", release_lease)
    with pytest.raises(interrupt) as caught:
        write_scores(empty_scores(), tmp_path / DESTINATION)
    assert caught.value.args == (23,)
    assert any("lease release fixture" in note for note in caught.value.__notes__)


def test_competing_directory_guard_waits_then_proceeds(tmp_path, writer_processes):
    with score_write_lock(tmp_path):
        waiting, lines = writer_processes(tmp_path, ready=False)
        with pytest.raises(queue.Empty):
            lines.get(timeout=0.1)
    assert lines.get(timeout=20) == "READY", stderr_tail(waiting)
    release(waiting)


def test_dry_run_lists_inactive_stage_while_writer_is_active(
    tmp_path, writer_processes
):
    active, _ = writer_processes(tmp_path)
    stale = temporary_sibling(tmp_path / DESTINATION, namespace=SCORE_STAGE_NAMESPACE)
    stale.write_bytes(b"stale")
    result = cleanup_score_temporaries(tmp_path, dry_run=True)
    assert result.deleted == (stale,)
    assert result.freed_mb == 5 / 1e6
    assert "score writer is active" in result.skipped[0]
    assert stale.exists()
    release(active)
    assert cleanup_score_temporaries(tmp_path).deleted == (stale,)


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink fixture")
def test_recovery_does_not_follow_lease_symlink(tmp_path):
    staged = temporary_sibling(tmp_path / DESTINATION, namespace=SCORE_STAGE_NAMESPACE)
    staged.write_bytes(b"preserve")
    target = tmp_path / "unrelated"
    target.write_bytes(b"target")
    staged.with_name(staged.name + ".lease").symlink_to(target)
    result = cleanup_score_temporaries(tmp_path)
    assert not result.deleted
    assert staged.read_bytes() == b"preserve"
    assert target.read_bytes() == b"target"


def test_directory_release_does_not_mask_body_failure(tmp_path, monkeypatch):
    original = score_storage.locked_path

    @contextmanager
    def guard(*args, **kwargs):
        with original(*args, **kwargs):
            yield
        raise OSError("release fixture")

    monkeypatch.setattr(score_storage, "locked_path", guard)
    with (
        pytest.raises(ValueError, match="primary") as caught,
        score_write_lock(tmp_path),
    ):
        raise ValueError("primary")
    assert "release fixture" in caught.value.__notes__[0]


def test_writers_to_different_files_run_in_parallel(tmp_path, writer_processes):
    first, _ = writer_processes(tmp_path)
    second, _ = writer_processes(tmp_path, evaluation_id="other", rows=3)
    assert len(payloads(tmp_path)) == 2
    release(second)
    release(first)
    assert load_scores(tmp_path / DESTINATION).height == 2
    assert load_scores(tmp_path / DESTINATION.replace("fixed", "other")).height == 3
