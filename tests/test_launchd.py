import fcntl
import importlib.util
import json
import os
import subprocess
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).parents[1] / "docs/launchd/daily-maintenance.py"


def run_maintenance(
    monkeypatch, tmp_path, *, failure=None, failure_code=-9, restore=False
):
    directory = tmp_path / "Library/Application Support/grounded-weather-forecast"
    directory.mkdir(parents=True, exist_ok=True)
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        step = command[command.index("--config") + 2]
        return subprocess.CompletedProcess(
            command, failure_code if step == failure else 0
        )

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(os, "chdir", lambda _: None)
    monkeypatch.setattr(subprocess, "run", run)
    with patch.dict(os.environ):
        module = load_script(SCRIPT)
        code = module.main(["--restore"] if restore else [])
    return code, commands, directory


def load_script(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_maintenance_uses_fresh_measured_children_and_stamps_success(
    monkeypatch, tmp_path
):
    code, commands, directory = run_maintenance(monkeypatch, tmp_path)
    assert code == 0
    assert commands
    assert all(command[:2] == ["/usr/bin/time", "-l"] for command in commands)
    assert all("grounded-weather-forecast" in command for command in commands)
    assert (directory / "last-maintenance").exists()


def test_killed_report_is_failure_without_stamp_or_pruning(
    monkeypatch, tmp_path, capsys
):
    code, commands, directory = run_maintenance(monkeypatch, tmp_path, failure="report")
    assert code == 1
    assert "report exited 137" in capsys.readouterr().out
    assert not (directory / "last-maintenance").exists()
    assert not any("prune-scores" in command for command in commands)


def test_restore_refreshes_evidence_without_marking_dataset_rebuilt(
    monkeypatch, tmp_path
):
    directory = tmp_path / "Library/Application Support/grounded-weather-forecast"
    directory.mkdir(parents=True)
    stamp = directory / "last-maintenance"
    stamp.touch()
    before = stamp.stat().st_mtime_ns
    code, commands, _ = run_maintenance(monkeypatch, tmp_path, restore=True)
    assert code == 0
    assert any("backtest" in command for command in commands)
    assert not any("build-dataset" in command for command in commands)
    assert stamp.stat().st_mtime_ns == before


@pytest.mark.parametrize("restore", [False, True])
def test_duplicate_job_skips_every_stage(monkeypatch, tmp_path, restore):
    directory = tmp_path / "Library/Application Support/grounded-weather-forecast"
    directory.mkdir(parents=True)
    with (directory / "maintenance-job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code, commands, _ = run_maintenance(monkeypatch, tmp_path, restore=restore)
    assert code == 0
    assert commands == []
    assert not (directory / "last-maintenance").exists()


@pytest.mark.parametrize("name", ["daily-maintenance.py", "hourly-predict-publish.py"])
def test_import_has_no_execution_side_effects(monkeypatch, name):
    before = dict(os.environ)

    def unexpected(*_args, **_kwargs):
        pytest.fail("import executed a scheduled job")

    monkeypatch.setattr(os, "chdir", unexpected)
    monkeypatch.setattr(subprocess, "run", unexpected)
    monkeypatch.setattr(subprocess, "Popen", unexpected)
    load_script(SCRIPT.with_name(name))
    assert dict(os.environ) == before


@pytest.mark.parametrize("failure", [None, "report", "build-dataset"])
def test_maintenance_releases_lock_and_restores_test_environment(
    monkeypatch, tmp_path, failure
):
    before = dict(os.environ)
    _, _, directory = run_maintenance(monkeypatch, tmp_path, failure=failure)
    assert dict(os.environ) == before
    with (directory / "maintenance-job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_healthy_hourly_check_cannot_block_daily_job(monkeypatch, tmp_path):
    hourly = load_script(SCRIPT.with_name("hourly-predict-publish.py"))
    forecast = tmp_path / "forecast.json"
    forecast.write_text(json.dumps({"status": "ready"}))
    monkeypatch.setattr(hourly, "OUT", str(forecast))
    directory = tmp_path / "Library/Application Support/grounded-weather-forecast"
    directory.mkdir(parents=True)
    monkeypatch.setattr(hourly, "HERE", directory)
    entered = threading.Event()
    release = threading.Event()
    real_read = Path.read_text
    failures = []

    def paused_read(path, *args, **kwargs):
        data = real_read(path, *args, **kwargs)
        if path == forecast:
            entered.set()
            if not release.wait(10):
                raise TimeoutError("daily job did not complete")
        return data

    def check():
        try:
            hourly.maybe_spawn_restore()
        except Exception as error:
            failures.append(error)

    monkeypatch.setattr(Path, "read_text", paused_read)
    for _ in range(20):
        entered.clear()
        release.clear()
        worker = threading.Thread(target=check)
        worker.start()
        try:
            assert entered.wait(10)
            code, commands, _ = run_maintenance(monkeypatch, tmp_path)
            assert code == 0
            assert commands
            (directory / "last-maintenance").unlink()
            release.set()
            worker.join(10)
            assert not worker.is_alive()
            assert failures == []
        finally:
            release.set()
            worker.join(10)


def test_actual_restore_spawns_detached_child_without_lock_probe(monkeypatch, tmp_path):
    hourly = load_script(SCRIPT.with_name("hourly-predict-publish.py"))
    forecast = tmp_path / "forecast.json"
    forecast.write_text(
        json.dumps({"status": "degraded", "status_reason": "implementation changed"})
    )
    monkeypatch.setattr(hourly, "OUT", str(forecast))
    monkeypatch.setattr(hourly, "HERE", tmp_path)
    monkeypatch.setattr(hourly, "RESTORE_LOG", tmp_path / "restore.log")
    monkeypatch.setenv("UV_TEST_PARENT", "parent-only")
    monkeypatch.setenv("VIRTUAL_ENV", "parent-venv")
    calls = []

    class Child:
        pid = 123

    def spawn(command, **kwargs):
        calls.append((command, kwargs))
        return Child()

    monkeypatch.setattr(subprocess, "Popen", spawn)
    with (tmp_path / "maintenance-job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hourly.maybe_spawn_restore()
    assert len(calls) == 1
    assert calls[0][0][-1] == "--restore"
    assert calls[0][1]["start_new_session"] is True
    assert "VIRTUAL_ENV" not in calls[0][1]["env"]
    assert not any(key.startswith("UV_") for key in calls[0][1]["env"])


def test_fresh_stamp_skips_stages_and_releases_lock(monkeypatch, tmp_path):
    directory = tmp_path / "Library/Application Support/grounded-weather-forecast"
    directory.mkdir(parents=True)
    (directory / "last-maintenance").touch()
    code, commands, _ = run_maintenance(monkeypatch, tmp_path)
    assert code == 0
    assert commands == []
    with (directory / "maintenance-job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_exception_releases_job_lock_and_logs_failed_exit(
    monkeypatch, tmp_path, capsys
):
    directory = tmp_path / "Library/Application Support/grounded-weather-forecast"
    directory.mkdir(parents=True)
    module = load_script(SCRIPT)

    def crash(_step):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(os, "chdir", lambda _: None)
    monkeypatch.setattr(module, "run_stage", crash)
    before = dict(os.environ)
    with (
        patch.dict(os.environ),
        pytest.raises(RuntimeError, match="unexpected failure"),
    ):
        module.main([])
    assert dict(os.environ) == before
    assert "exit=1" in capsys.readouterr().out
    assert not (directory / "last-maintenance").exists()
    with (directory / "maintenance-job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_hourly_failed_predict_stops_restore_and_publish(monkeypatch, capsys):
    module = load_script(SCRIPT.with_name("hourly-predict-publish.py"))
    before = dict(os.environ)
    monkeypatch.setattr(os, "chdir", lambda _: None)
    monkeypatch.setattr(module, "run_predict", lambda: 1)

    def unexpected():
        pytest.fail("failed prediction continued to restore or publish")

    monkeypatch.setattr(module, "maybe_spawn_restore", unexpected)
    monkeypatch.setattr(module, "publish", unexpected)
    with (
        patch.dict(os.environ),
        patch.object(module.sys, "path", module.sys.path.copy()),
    ):
        assert module.main() == 1
    assert dict(os.environ) == before
    assert "exit=1" in capsys.readouterr().out


def test_backtest_without_folds_remains_successful_and_prunes(monkeypatch, tmp_path):
    code, commands, directory = run_maintenance(
        monkeypatch, tmp_path, failure="backtest", failure_code=1
    )
    assert code == 0
    assert (directory / "last-maintenance").exists()
    assert any("prune-scores" in command for command in commands)


@pytest.mark.parametrize("sunday", [False, True])
def test_maintenance_stage_order_and_restore_scope(monkeypatch, sunday):
    module = load_script(SCRIPT)

    class Clock:
        @staticmethod
        def now(_timezone):
            return Clock()

        def weekday(self):
            return 6 if sunday else 0

    monkeypatch.setattr(module, "datetime", Clock)
    expected = [
        ["ingest-ensembles"],
        ["build-dataset"],
        ["alignment"],
        ["backtest", "--source", "live"],
        *([["backtest", "--source", "synthetic"]] if sunday else []),
        ["truth-qc"],
        ["report"],
        ["prune-scores"],
    ]
    assert module.maintenance_steps(restore=False) == expected
    assert module.maintenance_steps(restore=True) == [
        ["backtest", "--source", "live"],
        ["truth-qc"],
        ["report"],
        ["prune-scores"],
    ]
