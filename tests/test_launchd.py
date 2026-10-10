import atexit
import fcntl
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "docs/launchd/daily-maintenance.py"


def run_maintenance(monkeypatch, tmp_path, *, failure=None, restore=False):
    directory = tmp_path / "Library/Application Support/grounded-weather-forecast"
    directory.mkdir(parents=True, exist_ok=True)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        step = command[command.index("--config") + 2]
        return subprocess.CompletedProcess(command, -9 if step == failure else 0)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(os, "chdir", lambda _: None)
    monkeypatch.setattr(atexit, "register", lambda _: None)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *(["--restore"] if restore else [])])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(SCRIPT))
    return result.value.code, commands, directory


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


def test_duplicate_job_skips_every_stage(monkeypatch, tmp_path):
    directory = tmp_path / "Library/Application Support/grounded-weather-forecast"
    directory.mkdir(parents=True)
    with (directory / "maintenance-job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code, commands, _ = run_maintenance(monkeypatch, tmp_path)
    assert code == 0
    assert commands == []
    assert not (directory / "last-maintenance").exists()
