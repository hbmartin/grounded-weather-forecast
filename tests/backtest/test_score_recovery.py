"""Process-level ownership, hard-stop recovery, and lazy writer dependencies."""

import os
import queue
import signal
import stat
import subprocess
import sys
from pathlib import Path
from threading import Thread

import pytest
from conftest import write_config
from filelock import Timeout

from grounded_weather_forecast import cli
from grounded_weather_forecast.backtest.score_storage import (
    SCORE_WRITE_LOCK,
    cleanup_score_temporaries,
    score_write_lock,
)
from grounded_weather_forecast.backtest.scores import load_scores
from grounded_weather_forecast.reports.operations import prune_scores_files
from grounded_weather_forecast.serve import selection

ROOT = Path(__file__).resolve().parents[2]
TEMP_NAME = ".scores_hourly_live_expanding_fixed.parquet.0123456789abcdef.tmp"
DESTINATION = "scores_hourly_live_expanding_fixed.parquet"
WORKER = """
import sys
from contextlib import contextmanager
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from benchmark_backtest_memory import fixture_batches
from grounded_weather_forecast.backtest import scores
from grounded_weather_forecast.storage import atomic_parquet_destination
directory = Path(sys.argv[1])
frame = next(fixture_batches(scores, 2, compact=True))
original_lock = scores.score_write_lock
@contextmanager
def observed_lock(directory):
    print('WAITING', flush=True)
    with original_lock(directory):
        print('LOCKED', flush=True)
        yield
scores.score_write_lock = observed_lock
def pause():
    print('READY', flush=True)
    sys.stdin.readline()
if sys.argv[2] == 'stream':
    scores.SCORE_BATCH_ROWS = 1
    def batches():
        yield frame
        pause()
    scores.write_score_batches(batches(), directory, 'hourly', 'live', 'expanding')
else:
    def write(frame, path):
        with atomic_parquet_destination(path) as temporary:
            frame.write_parquet(temporary)
            pause()
    scores.atomic_write_parquet = write
    scores.write_scores(frame.cast(scores.SCORES_SCHEMA), directory / 'scores_hourly_live_expanding_fixed.parquet')
"""


@pytest.fixture
def writer_processes():
    processes = []

    def launch(directory, mode="stream", *, ready=True):
        process = subprocess.Popen(  # noqa: S603 -- local fixture, constructed argv
            [sys.executable, "-c", WORKER, str(directory), mode, str(ROOT / "scripts")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ
            | {
                "POLARS_MAX_THREADS": "2",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
        )
        processes.append(process)
        lines = queue.Queue()

        def collect():
            for line in process.stdout:
                lines.put(line.strip())
            lines.put(None)

        Thread(target=collect, daemon=True).start()
        assert lines.get(timeout=20) == "WAITING"
        if ready:
            assert lines.get(timeout=20) == "LOCKED"
            assert lines.get(timeout=20) == "READY"
        return process, lines

    yield launch
    for process in processes:
        if process.poll() is None:
            process.kill()
    for process in processes:
        process.wait(timeout=20)
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()


def release(process):
    process.stdin.write("release\n")
    process.stdin.flush()
    assert process.wait(timeout=20) == 0, process.stderr.read()


@pytest.mark.parametrize("mode", ["stream", "materialized"])
def test_active_library_writer_blocks_cleanup_and_other_writes(
    tmp_path, writer_processes, mode
):
    process, _ = writer_processes(tmp_path, mode)
    temporaries = list(tmp_path.glob(".scores_*.tmp"))
    assert len(temporaries) == 1
    before = temporaries[0].read_bytes()
    cleanup = cleanup_score_temporaries(tmp_path)
    assert not cleanup.deleted
    assert cleanup.skipped == ("temporary cleanup: score writer is active",)
    assert temporaries[0].read_bytes() == before
    with pytest.raises(Timeout), score_write_lock(tmp_path, timeout=0):
        pytest.fail("active writer did not hold its directory lock")
    other, lines = writer_processes(tmp_path, ready=False)
    with pytest.raises(queue.Empty):
        lines.get(timeout=0.2)
    independent, _ = writer_processes(tmp_path / "independent")
    release(independent)
    release(process)
    assert lines.get(timeout=20) == "LOCKED"
    assert lines.get(timeout=20) == "READY"
    release(other)
    assert load_scores(tmp_path / DESTINATION).height == 2
    assert not list(tmp_path.glob(".scores_*.tmp"))
    assert (tmp_path / SCORE_WRITE_LOCK).is_file()


@pytest.mark.skipif(os.name == "nt", reason="POSIX interrupt semantics")
def test_waiting_writer_can_be_interrupted(tmp_path, writer_processes):
    holder, _ = writer_processes(tmp_path)
    waiting, _ = writer_processes(tmp_path, ready=False)
    waiting.send_signal(signal.SIGINT)
    assert waiting.wait(timeout=20) in (-signal.SIGINT, 130)
    assert len(list(tmp_path.glob(".scores_*.tmp"))) == 1
    assert cleanup_score_temporaries(tmp_path).skipped
    release(holder)
    assert load_scores(tmp_path / DESTINATION).height == 2


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
    temporary = list(directory.glob(".scores_*.tmp"))
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
    stale = tmp_path / TEMP_NAME
    stale.write_bytes(b"abandoned")
    untouched = [
        tmp_path / DESTINATION,
        tmp_path / ".unrelated.parquet.0123456789abcdef.tmp",
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
    stale = tmp_path / TEMP_NAME
    stale.touch()
    unlink = Path.unlink

    def disappeared(path, *args, **kwargs):
        unlink(path, *args, **kwargs)
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
    stale = directory / TEMP_NAME
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
    stale = directory / TEMP_NAME
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
