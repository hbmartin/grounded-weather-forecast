"""The pipeline mutex: mutators serialize, serving never waits."""

from conftest import write_config
from filelock import FileLock, Timeout

import grounded_weather_forecast.storage as storage_module
from grounded_weather_forecast.cli import _LOCKED_COMMANDS, EXIT_CONTENTION, main
from grounded_weather_forecast.storage import pipeline_lock


def test_contended_command_exits_tempfail(tmp_path, monkeypatch, capsys):
    config = write_config(tmp_path)
    monkeypatch.setattr(storage_module, "PIPELINE_LOCK_TIMEOUT_S", 0.05)
    with FileLock(config.dataset.dir / "pipeline.lock"):
        code = main(
            ["--config", str(tmp_path / "config.toml"), "prune-scores", "--dry-run"]
        )
    assert code == EXIT_CONTENTION
    assert "another pipeline command" in capsys.readouterr().err


def test_uncontended_command_runs_and_releases(tmp_path):
    config = write_config(tmp_path)
    code = main(
        ["--config", str(tmp_path / "config.toml"), "prune-scores", "--dry-run"]
    )
    assert code == 0
    # The lock must be free again afterwards.
    with pipeline_lock(config.dataset.dir, timeout=0.05):
        pass


def test_serving_commands_stay_outside_the_lock():
    assert "predict" not in _LOCKED_COMMANDS
    assert "ingest-ensembles" not in _LOCKED_COMMANDS
    assert "qc" not in _LOCKED_COMMANDS


def test_pipeline_lock_times_out_against_a_holder(tmp_path):
    (tmp_path / "data").mkdir()
    with pipeline_lock(tmp_path / "data", timeout=5):
        try:
            with pipeline_lock(tmp_path / "data", timeout=0.05):
                raise AssertionError("second holder must not acquire")
        except Timeout:
            pass
