import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import types
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from grounded_weather_forecast.status import EXIT_NO_FOLDS

SCRIPT = Path(__file__).parents[1] / "docs/launchd/daily-maintenance.py"


@pytest.fixture(autouse=True)
def project_layout(monkeypatch, tmp_path):
    monkeypatch.delenv("UV", raising=False)
    project = tmp_path / "weather projects" / "forecast checkout"
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        '[project]\nname = "grounded-weather-forecast"\n'
    )
    (project / "config.toml").write_text(
        '[station]\ndb_path = "station data/observations.db"\n'
    )
    (project / "data").mkdir()
    for product in ("hourly", "daily"):
        (project / "data" / f"{product}_matrix_synthetic.parquet").touch()
    database = project / "station data/observations.db"
    database.parent.mkdir()
    database.touch()
    station = project.parent / "ambientweather2sqlite"
    station.mkdir()
    (station / "pyproject.toml").write_text(
        '[project]\nname = "ambientweather2sqlite"\n'
    )
    binary = tmp_path / "custom bin" / "uv"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\nexit 99\n")
    binary.chmod(0o755)
    monkeypatch.setenv(
        "PATH", f"{binary.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    )
    monkeypatch.chdir(project)
    return project


def run_maintenance(
    monkeypatch,
    tmp_path,
    *,
    failure=None,
    failure_code=-9,
    restore=False,
    steps=None,
    old_stamp=False,
    failures=None,
    sunday=False,
):
    directory = tmp_path / "Library/Application Support/grounded-weather-forecast"
    directory.mkdir(parents=True, exist_ok=True)
    commands = []
    if old_stamp:
        stamp = directory / "last-maintenance"
        stamp.touch()
        os.utime(stamp, (1, 1))

    def run(command, **_kwargs):
        commands.append(command)
        argv = tuple(command[command.index("--config") + 2 :])
        step = argv[0]
        code = (failures or {}).get(argv, failure_code if step == failure else 0)
        return subprocess.CompletedProcess(command, code)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(os, "chdir", lambda _: None)
    monkeypatch.setattr(subprocess, "run", run)
    with patch.dict(os.environ):
        module = load_script(SCRIPT)

        class Clock:
            @staticmethod
            def now(_timezone):
                return datetime(2026, 10, 18 if sunday else 19, tzinfo=UTC)

        monkeypatch.setattr(module, "datetime", Clock)
        if steps is not None:
            monkeypatch.setattr(module, "maintenance_steps", lambda **_: steps)
        code = module.main(["--restore"] if restore else [])
    return code, commands, directory


def load_script(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(path.parent), *sys.path]):
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
    paths = hourly.RuntimePaths(tmp_path, "uv")
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
            hourly.maybe_spawn_restore(paths)
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
    paths = hourly.RuntimePaths(tmp_path, "uv")
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
        hourly.maybe_spawn_restore(paths)
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

    def crash(_step, _paths):
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
    monkeypatch.setattr(module, "run_predict", lambda _paths: 1)

    def unexpected(_paths):
        pytest.fail("failed prediction continued to restore or publish")

    monkeypatch.setattr(module, "maybe_spawn_restore", unexpected)
    monkeypatch.setattr(module, "publish", unexpected)
    with (
        patch.dict(os.environ),
        patch.object(module.sys, "path", module.sys.path.copy()),
    ):
        assert module.main([]) == 1
    assert dict(os.environ) == before
    assert "exit=1" in capsys.readouterr().out


def test_backtest_exit_one_fails_without_success_stamp(monkeypatch, tmp_path, capsys):
    code, commands, directory = run_maintenance(
        monkeypatch, tmp_path, failure="backtest", failure_code=1
    )
    assert code == 1
    assert not (directory / "last-maintenance").exists()
    assert any("prune-scores" in command for command in commands)
    output = capsys.readouterr().out
    assert "maintenance failed stages: backtest --source live" in output
    assert "exit=1" in output


@pytest.mark.parametrize("source", ["live", "synthetic"])
@pytest.mark.parametrize("failure_code", [1, 2, 75, -9])
def test_backtest_failures_keep_previous_success_stamp(
    monkeypatch, tmp_path, capsys, source, failure_code
):
    code, _, directory = run_maintenance(
        monkeypatch,
        tmp_path,
        old_stamp=True,
        failures={("backtest", "--source", source): failure_code},
        steps=[
            ["backtest", "--source", "live"],
            ["backtest", "--source", "synthetic"],
            ["report"],
        ],
    )
    assert code == 1
    assert (directory / "last-maintenance").stat().st_mtime_ns == 1_000_000_000
    output = capsys.readouterr().out
    assert f"maintenance failed stages: backtest --source {source}" in output
    assert "END mode=maintain" in output
    assert "exit=1" in output


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


@pytest.mark.parametrize("source", ["live", "synthetic"])
@pytest.mark.parametrize("later_failure", [None, "report", "truth-qc"])
def test_pending_evaluation_preserves_stamp_and_continues(
    monkeypatch, tmp_path, source, later_failure, capsys
):
    failures = {("backtest", "--source", source): EXIT_NO_FOLDS}
    if later_failure:
        failures[(later_failure,)] = 1
    code, commands, directory = run_maintenance(
        monkeypatch,
        tmp_path,
        old_stamp=True,
        failures=failures,
        steps=[
            ["backtest", "--source", "live"],
            ["backtest", "--source", "synthetic"],
            ["truth-qc"],
            ["report"],
            ["prune-scores"],
        ],
    )
    assert code == int(later_failure is not None)
    assert (directory / "last-maintenance").stat().st_mtime_ns == 1_000_000_000
    assert any("report" in command for command in commands)
    assert any("prune-scores" in command for command in commands) == (
        later_failure != "report"
    )
    assert f"evaluation pending: backtest --source {source}" in capsys.readouterr().out


def test_failed_report_with_arguments_blocks_pruning(monkeypatch, tmp_path):
    code, commands, _ = run_maintenance(
        monkeypatch,
        tmp_path,
        failures={("report", "--example-option"): 1},
        steps=[["report", "--example-option"], ["prune-scores"]],
    )
    assert code == 1
    assert not any("prune-scores" in command for command in commands)


def test_pending_automatic_restore_stays_incomplete(monkeypatch, tmp_path):
    code, commands, directory = run_maintenance(
        monkeypatch,
        tmp_path,
        restore=True,
        old_stamp=True,
        failures={("backtest", "--source", "live"): EXIT_NO_FOLDS},
    )
    assert code == EXIT_NO_FOLDS
    assert (directory / "last-maintenance").stat().st_mtime_ns == 1_000_000_000
    assert any("report" in command for command in commands)
    assert not any("synthetic" in command for command in commands)


@pytest.mark.parametrize("mode", ["auto", "required", "off"])
@pytest.mark.parametrize("present", [(), ("hourly",), ("hourly", "daily")])
@pytest.mark.parametrize("sunday", [False, True])
def test_synthetic_scheduling_modes(
    monkeypatch, tmp_path, project_layout, mode, present, sunday, capsys
):
    dataset = project_layout / "custom dataset"
    dataset.mkdir()
    for product in present:
        (dataset / f"{product}_matrix_synthetic.parquet").touch()
    with (project_layout / "config.toml").open("a") as file:
        file.write(
            f'\n[dataset]\ndir = "custom dataset"\n[maintenance]\nsynthetic_backtest = "{mode}"\n'
        )
    code, commands, directory = run_maintenance(monkeypatch, tmp_path, sunday=sunday)
    expected_failure = (
        sunday
        and mode != "off"
        and len(present) != 2
        and (mode == "required" or len(present) == 1)
    )
    assert code == int(expected_failure)
    assert (directory / "last-maintenance").exists() == (not expected_failure)
    synthetic = [command for command in commands if "synthetic" in command]
    assert bool(synthetic) == (sunday and mode != "off" and len(present) == 2)
    if sunday and mode == "auto" and not present:
        assert "both matrices absent" in capsys.readouterr().out


def test_synthetic_permission_fault_is_not_an_optional_skip(monkeypatch, tmp_path):
    original = Path.stat

    def denied(path, *args, **kwargs):
        if path.name == "hourly_matrix_synthetic.parquet":
            raise PermissionError("fixture denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied)
    code, commands, directory = run_maintenance(monkeypatch, tmp_path, sunday=True)
    assert code == 1
    assert not (directory / "last-maintenance").exists()
    assert not any("synthetic" in command for command in commands)
    assert any("report" in command for command in commands)


@pytest.mark.parametrize("fresh_stamp", [False, True])
def test_invalid_synthetic_setting_fails_configuration(
    monkeypatch, tmp_path, project_layout, fresh_stamp
):
    with (project_layout / "config.toml").open("a") as file:
        file.write('\n[maintenance]\nsynthetic_backtest = "sometimes"\n')
    stamp = (
        tmp_path
        / "Library/Application Support/grounded-weather-forecast/last-maintenance"
    )
    if fresh_stamp:
        stamp.parent.mkdir(parents=True)
        stamp.touch()
        before = stamp.stat().st_mtime_ns
    code, commands, directory = run_maintenance(monkeypatch, tmp_path)
    assert code == 2
    assert commands == []
    assert (directory / "last-maintenance").exists() == fresh_stamp
    if fresh_stamp:
        assert stamp.stat().st_mtime_ns == before


@pytest.mark.parametrize("source", ["live", "synthetic"])
def test_exit_and_crash_lines_include_source(monkeypatch, source, capsys):
    module = load_script(SCRIPT)
    paths = module.resolve_runtime()
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: subprocess.CompletedProcess([], 1)
    )
    assert module.run_stage(["backtest", "--source", source], paths) == 1
    assert f"backtest --source {source} exited 1" in capsys.readouterr().out

    def fail(*_a, **_k):
        raise OSError("fixture crash")

    monkeypatch.setattr(subprocess, "run", fail)
    assert module.run_stage(["backtest", "--source", source], paths) == 70
    assert f"backtest --source {source} crashed" in capsys.readouterr().out


def test_publishing_uses_active_project_independently_of_forecast_database(
    project_layout,
):
    hourly = load_script(SCRIPT.with_name("hourly-predict-publish.py"))
    paths = hourly.resolve_publishing()
    assert paths.project == project_layout
    assert paths.station_project == project_layout.parent / "ambientweather2sqlite"
    assert not hasattr(paths, "station_db")
    assert Path(paths.uv).parent.name == "custom bin"


def test_station_project_override_ignores_forecast_database(tmp_path, project_layout):
    hourly = load_script(SCRIPT.with_name("hourly-predict-publish.py"))
    collector = tmp_path / "separate collector"
    collector.mkdir()
    (collector / "pyproject.toml").write_text(
        '[project]\nname = "ambientweather2sqlite"\n'
    )
    database = tmp_path / "separate observations.db"
    database.touch()
    (project_layout / "config.toml").write_text(
        f"[station]\ndb_path = {json.dumps(str(database))}\n"
    )
    paths = hourly.resolve_publishing(collector)
    assert paths.station_project == collector
    assert not hasattr(paths, "station_db")


def test_station_child_uses_collector_configuration(monkeypatch):
    hourly = load_script(SCRIPT.with_name("hourly-predict-publish.py"))
    paths = hourly.resolve_publishing()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("VIRTUAL_ENV", "forecast-environment")
    monkeypatch.setenv("UV_PROJECT", "forecast-project")
    assert hourly.render_station(paths) == 0
    command, kwargs = calls[0]
    assert command[command.index("--project") + 1] == str(paths.station_project)
    assert "--db" not in command
    assert command[command.index("--directory") + 1] == str(paths.station_project)
    assert "VIRTUAL_ENV" not in kwargs["env"]
    assert "UV_PROJECT" not in kwargs["env"]


@pytest.mark.parametrize("name", ["daily-maintenance.py", "hourly-predict-publish.py"])
@pytest.mark.parametrize(
    "fault", ["project", "config-missing", "config-malformed", "uv"]
)
def test_path_failures_run_no_children_and_preserve_stamp(
    monkeypatch, tmp_path, project_layout, capsys, name, fault
):
    module = load_script(SCRIPT.with_name(name))
    directory = tmp_path / "Library/Application Support/grounded-weather-forecast"
    directory.mkdir(parents=True)
    stamp = directory / "last-maintenance"
    stamp.touch()
    os.utime(stamp, (1, 1))
    before = stamp.stat().st_mtime_ns
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    match fault:
        case "project":
            (project_layout / "pyproject.toml").write_text(
                '[project]\nname = "other"\n'
            )
        case "config-missing":
            (project_layout / "config.toml").unlink()
        case "config-malformed":
            (project_layout / "config.toml").write_text("[broken")
        case "uv":
            monkeypatch.setenv("PATH", "")

    def unexpected(*_args, **_kwargs):
        pytest.fail("invalid paths started a child")

    monkeypatch.setattr(subprocess, "run", unexpected)
    monkeypatch.setattr(subprocess, "Popen", unexpected)
    if name.startswith("hourly"):
        monkeypatch.setattr(module, "publish", unexpected)
    with patch.dict(os.environ):
        assert module.main([]) == 2
    assert stamp.stat().st_mtime_ns == before
    output = capsys.readouterr().out
    assert "configuration error:" in output
    assert "exit=2" in output


@pytest.mark.parametrize("optimized", [False, True])
def test_installed_copy_uses_launchd_working_directory(
    tmp_path, project_layout, optimized
):
    installed = tmp_path / "installed scripts"
    installed.mkdir()
    for name in ("launchd_paths.py", "hourly-predict-publish.py"):
        shutil.copyfile(SCRIPT.with_name(name), installed / name)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(  # noqa: S603 -- fixture script and interpreter
        [
            sys.executable,
            *(["-OO"] if optimized else []),
            str(installed / "hourly-predict-publish.py"),
        ],
        cwd=project_layout,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    # The fixture uv deliberately fails: configuration resolves without ever
    # running real weather commands, touching publication state, or using Git.
    assert result.returncode == 1
    assert f"PROJECT_CWD cwd={project_layout}" in result.stdout
    assert "configuration error:" not in result.stdout
    assert "exit=1" in result.stdout


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize(
    "fault",
    [
        "collector",
        "unknown-user",
        "symlink-loop",
        "missing-db",
        "permission",
        "timeout",
    ],
)
def test_station_failure_follows_prediction_and_preserves_fallback(
    monkeypatch, tmp_path, project_layout, capsys, staged, fault
):
    hourly = load_script(SCRIPT.with_name("hourly-predict-publish.py"))
    stage = tmp_path / "staged"
    stage.mkdir()
    if staged:
        (stage / "latest.json").write_text('{"previous":true}')
    monkeypatch.setattr(hourly, "STAGED", stage)
    events = []
    monkeypatch.setattr(
        hourly, "run_predict", lambda paths: events.append("predict") or 0
    )
    monkeypatch.setattr(
        hourly, "maybe_spawn_restore", lambda paths: events.append("restore")
    )
    args = []
    if fault == "collector":
        (project_layout.parent / "ambientweather2sqlite/pyproject.toml").unlink()
    elif fault == "unknown-user":
        args = ["--station-project", "~nosuchweatheruser/x"]
    elif fault == "symlink-loop":
        link = tmp_path / "loop"
        link.symlink_to(link)
        args = ["--station-project", str(link)]

    def child(command, **kwargs):
        events.append("station")
        assert "--db" not in command
        if fault == "permission":
            raise PermissionError("database access denied")
        if fault == "timeout":
            raise subprocess.TimeoutExpired(command, 180)
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr(subprocess, "run", child)
    fake = types.SimpleNamespace(
        choose_source=lambda path: path, run=lambda **kw: events.append("publish") or 0
    )
    monkeypatch.setitem(sys.modules, "publish_forecast", fake)
    with patch.object(sys, "path", sys.path.copy()):
        assert hourly.main(args) == (0 if staged else 1)
    assert events[:2] == ["predict", "restore"]
    assert ("publish" in events) == staged
    assert ("station" in events) == (fault in {"missing-db", "permission", "timeout"})
    if staged:
        assert (stage / "latest.json").read_text() == '{"previous":true}'
    output = capsys.readouterr().out
    assert "station exited" in output
    assert f"exit={0 if staged else 1}" in output


@pytest.mark.parametrize("name", ["daily-maintenance.py", "hourly-predict-publish.py"])
def test_bad_arguments_and_help_have_lifecycle_logs(name, capsys):
    module = load_script(SCRIPT.with_name(name))
    for arguments, expected in [(["--typo"], 2), (["--help"], 0)]:
        assert module.main(arguments) == expected
        output = capsys.readouterr().out
        assert "START mode=" in output
        assert f"exit={expected}" in output


@pytest.mark.parametrize("name", ["daily-maintenance.py", "hourly-predict-publish.py"])
def test_startup_survives_stripped_docstrings_and_missing_cwd(
    monkeypatch, name, capsys
):
    module = load_script(SCRIPT.with_name(name))
    monkeypatch.setattr(module, "__doc__", None)

    def missing():
        raise FileNotFoundError("deleted working directory")

    monkeypatch.setattr(Path, "cwd", missing)
    assert module.main([]) == 2
    output = capsys.readouterr().out
    assert "START mode=" in output and "deleted working directory" in output
    assert "exit=2" in output


def test_uv_precedence_and_launcher_symlink(monkeypatch, project_layout, tmp_path):
    module = load_script(SCRIPT)
    launcher = tmp_path / "approved uv"
    launcher.symlink_to(Path(module.resolve_runtime().uv))
    monkeypatch.setenv("UV", str(launcher))
    monkeypatch.setenv("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    assert module.resolve_runtime().uv == str(launcher)


@pytest.mark.parametrize("value", ["", "/no/such/uv"])
def test_invalid_explicit_uv_does_not_fall_back(monkeypatch, value):
    module = load_script(SCRIPT)
    monkeypatch.setenv("UV", value)
    with pytest.raises(ValueError, match="uv|UV"):
        module.resolve_runtime()


def test_uv_permission_error_keeps_original_detail(monkeypatch):
    module = load_script(SCRIPT)
    executable = Path(module.resolve_runtime().uv)
    monkeypatch.setenv("UV", str(executable))
    original = Path.stat

    def denied(path, *args, **kwargs):
        if path == executable:
            raise PermissionError("fixture permission revoked")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied)
    with pytest.raises(ValueError, match="fixture permission revoked"):
        module.resolve_runtime()


def test_wrapper_reads_configuration_once_and_restore_ignores_synthetic(
    monkeypatch, project_layout, tmp_path
):
    with (project_layout / "config.toml").open("a") as stream:
        stream.write('\n[maintenance]\nsynthetic_backtest="invalid"\n')
    code, commands, _ = run_maintenance(monkeypatch, tmp_path, restore=True)
    assert code == 0 and commands
    module = load_script(SCRIPT)
    original = Path.read_text
    reads = []

    def read(path, **kwargs):
        if path.name == "config.toml":
            reads.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    paths = module.resolve_runtime()
    with pytest.raises(ValueError, match="synthetic_backtest"):
        module.synthetic_mode(paths)
    assert reads == [project_layout / "config.toml"]


@pytest.mark.parametrize(
    "fault", ["missing-directory", "dangling", "directory", "permission"]
)
def test_optional_synthetic_inputs_require_available_storage(
    monkeypatch, tmp_path, fault
):
    module = load_script(SCRIPT)
    directory = tmp_path / "backfill"
    if fault != "missing-directory":
        directory.mkdir()
    if fault == "dangling":
        for product in ("hourly", "daily"):
            (directory / f"{product}_matrix_synthetic.parquet").symlink_to(
                tmp_path / "absent"
            )
    elif fault == "directory":
        (directory / "hourly_matrix_synthetic.parquet").mkdir()
    elif fault == "permission":

        def denied(*_args):
            raise PermissionError("cannot list dataset")

        monkeypatch.setattr(os, "scandir", denied)
    with pytest.raises(OSError):
        module.synthetic_enabled("auto", directory)


def test_daily_package_mismatch_logs_clean_configuration_failure(
    monkeypatch, tmp_path, capsys
):
    load_script(SCRIPT)
    monkeypatch.setitem(sys.modules, "grounded_weather_forecast.status", None)
    code, _, _ = run_maintenance(monkeypatch, tmp_path)
    assert code == 2
    output = capsys.readouterr().out
    assert "START mode=" in output and "exit=2" in output
    assert "grounded_weather_forecast.status" in output


@pytest.mark.parametrize("fault", ["log", "spawn", "value", "subprocess"])
def test_restore_launch_failure_still_publishes_and_returns_failure(
    monkeypatch, tmp_path, capsys, fault
):
    module = load_script(SCRIPT.with_name("hourly-predict-publish.py"))
    events = []
    monkeypatch.setattr(module, "run_predict", lambda _: events.append("predict") or 0)
    module.resolve_runtime().forecast.write_text(
        '{"status":"degraded","status_reason":"implementation changed"}'
    )
    if fault == "log":
        monkeypatch.setattr(module, "RESTORE_LOG", tmp_path / "missing" / "restore.log")
    else:
        monkeypatch.setattr(module, "RESTORE_LOG", tmp_path / "restore.log")

        def fail_spawn(*_a, **_k):
            error = {
                "spawn": OSError,
                "value": ValueError,
                "subprocess": subprocess.SubprocessError,
            }[fault]
            raise error("spawn unavailable")

        monkeypatch.setattr(subprocess, "Popen", fail_spawn)
    monkeypatch.setattr(module, "publish", lambda *_a: events.append("publish") or 0)
    assert module.main([]) == 1
    assert events == ["predict", "publish"]
    output = capsys.readouterr().out
    assert "auto-restore launch failed" in output
    assert "exit=1" in output
    assert "configuration error" not in output


def test_fresh_support_directory_is_initialized(monkeypatch, tmp_path):
    module = load_script(SCRIPT)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(module, "run_stage", lambda *_a: 0)
    monkeypatch.setattr(module, "maintenance_steps", lambda **_k: [["report"]])
    directory = tmp_path / "Library/Application Support/grounded-weather-forecast"
    assert not directory.exists()
    assert module.main([]) == 0
    assert (directory / "last-maintenance").exists()


@pytest.mark.parametrize("fault", ["mkdir", "open", "flock"])
def test_support_lock_setup_faults_are_configuration_errors(
    monkeypatch, tmp_path, capsys, fault
):
    module = load_script(SCRIPT)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    directory = tmp_path / "Library/Application Support/grounded-weather-forecast"
    directory.mkdir(parents=True)

    def denied(*_a, **_k):
        raise PermissionError("support lock fixture")

    if fault == "mkdir":
        monkeypatch.setattr(Path, "mkdir", denied)
    elif fault == "open":
        opened = Path.open

        def deny_lock(path, *args, **kwargs):
            if path.name == "maintenance-job.lock":
                denied()
            return opened(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", deny_lock)
    else:
        monkeypatch.setattr(module.fcntl, "flock", denied)
    assert module.main([]) == 2
    output = capsys.readouterr().out
    assert "START mode=maintain" in output and "exit=2" in output
    assert "support lock fixture" in output


def test_stamp_runtime_failure_keeps_previous_stamp(monkeypatch, tmp_path, capsys):
    directory = tmp_path / "Library/Application Support/grounded-weather-forecast"
    directory.mkdir(parents=True)
    stamp = directory / "last-maintenance"
    stamp.touch()
    os.utime(stamp, (1, 1))
    before = stamp.stat().st_mtime_ns

    def fail_touch(*_a, **_k):
        raise PermissionError("stamp permission denied")

    monkeypatch.setattr(Path, "touch", fail_touch)
    code, commands, _ = run_maintenance(monkeypatch, tmp_path)
    assert code == 1
    assert stamp.stat().st_mtime_ns == before
    assert any("prune-scores" in command for command in commands)
    output = capsys.readouterr().out
    assert "maintenance failed: stamp permission denied" in output
    assert "configuration error" not in output
    assert "exit=1" in output


def test_late_synthetic_import_failure_continues_diagnostics(
    monkeypatch, tmp_path, capsys
):
    module = load_script(SCRIPT)
    directory = tmp_path / "Library/Application Support/grounded-weather-forecast"
    directory.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    steps = [
        ["backtest", "--source", "synthetic"],
        ["truth-qc"],
        ["report"],
        ["prune-scores"],
    ]
    monkeypatch.setattr(module, "maintenance_steps", lambda **_: steps)

    def fail_import(*_a):
        raise ImportError("synthetic runtime unavailable")

    monkeypatch.setattr(module, "synthetic_enabled", fail_import)
    seen = []
    monkeypatch.setattr(
        module, "run_stage", lambda step, _paths: seen.append(step) or 0
    )
    assert module.main([]) == 1
    assert seen == steps[1:]
    assert not (directory / "last-maintenance").exists()
    output = capsys.readouterr().out
    assert "backtest --source synthetic failed" in output
    assert "exit=1" in output


def test_publication_oserror_is_runtime_failure(monkeypatch, capsys):
    module = load_script(SCRIPT.with_name("hourly-predict-publish.py"))
    monkeypatch.setattr(module, "run_predict", lambda _: 0)
    monkeypatch.setattr(module, "maybe_spawn_restore", lambda _: None)

    def fail_publish(*_a):
        raise OSError("publisher write denied")

    monkeypatch.setattr(module, "publish", fail_publish)
    assert module.main([]) == 1
    assert "runtime failure: publisher write denied" in capsys.readouterr().out
