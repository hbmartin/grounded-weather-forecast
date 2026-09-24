"""The pipeline mutex and short dataset snapshot lock."""

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from threading import Event

import pytest
import polars as pl
from conftest import write_config
from filelock import FileLock, Timeout

import grounded_weather_forecast.cli as cli_module
import grounded_weather_forecast.storage as storage_module
import grounded_weather_forecast.serve.predict as predict_module
import grounded_weather_forecast.serve.selection as selection_module
from grounded_weather_forecast.cli import _LOCKED_COMMANDS, EXIT_CONTENTION, main
from grounded_weather_forecast.serve.dataset_snapshot import LiveDatasetSnapshot
from grounded_weather_forecast.runs import load_runs, runs_path
from grounded_weather_forecast.storage import dataset_lock, pipeline_lock


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


def test_predict_exits_tempfail_during_dataset_publication(tmp_path, monkeypatch):
    config = write_config(tmp_path)
    monkeypatch.setattr(storage_module, "DATASET_LOCK_TIMEOUT_S", 0.05)
    with FileLock(config.dataset.dir / "dataset.lock"):
        code = main(["--config", str(tmp_path / "config.toml"), "predict"])
    assert code == EXIT_CONTENTION


def test_publish_exits_tempfail_during_dataset_publication(tmp_path, monkeypatch):
    config = write_config(tmp_path)
    monkeypatch.setattr(storage_module, "DATASET_LOCK_TIMEOUT_S", 0.05)
    with FileLock(config.dataset.dir / "dataset.lock"):
        code = main(
            [
                "--config",
                str(tmp_path / "config.toml"),
                "publish",
                "--out",
                str(tmp_path / "forecast.json"),
            ]
        )
    assert code == EXIT_CONTENTION


def test_selection_and_fitting_run_after_dataset_lock_is_released(
    tmp_path, monkeypatch
):
    config = write_config(tmp_path)
    seen = []

    def assert_unlocked(name):
        with dataset_lock(config.dataset.dir, timeout=0.05):
            seen.append(name)

    monkeypatch.setattr(
        "grounded_weather_forecast.serve.dataset_snapshot.load_live_snapshot",
        lambda *_args, **_kwargs: LiveDatasetSnapshot(
            "pinned", pl.DataFrame(), pl.DataFrame(), pl.DataFrame()
        ),
    )

    def select(*_args, **kwargs):
        assert_unlocked("selection")
        assert kwargs["dataset_id"] == "pinned"
        return {}

    def predict(*_args, **kwargs):
        assert_unlocked("fitting")
        assert kwargs["dataset_snapshot"].fingerprint == "pinned"
        return "document"

    monkeypatch.setattr(selection_module, "select_methods", select)
    monkeypatch.setattr(predict_module, "predict", predict)
    assert (
        cli_module._forecast_document(
            config, method="auto", semantics_flag="auto", now=None
        )
        == "document"
    )
    assert seen == ["selection", "fitting"]


def test_maintain_runs_every_step_in_order(tmp_path, monkeypatch):
    write_config(tmp_path)
    calls = []
    held = False

    @contextmanager
    def tracked_pipeline_lock(dataset_dir, timeout=None):
        nonlocal held
        assert timeout == -1
        held = True
        try:
            yield
        finally:
            held = False

    def record(name):
        assert held
        calls.append(name)
        return 0

    monkeypatch.setattr(storage_module, "pipeline_lock", tracked_pipeline_lock)
    monkeypatch.setattr(
        cli_module, "_cmd_build_dataset", lambda config: record("build")
    )
    monkeypatch.setattr(
        cli_module, "_cmd_backtest", lambda config, args: record("backtest")
    )
    monkeypatch.setattr(cli_module, "_cmd_report", lambda config: record("report"))
    monkeypatch.setattr(
        cli_module, "_cmd_truth_qc", lambda config, args: record("truth-qc")
    )

    code = main(["--config", str(tmp_path / "config.toml"), "maintain"])

    assert code == 0
    assert calls == ["build", "backtest", "report", "truth-qc"]


def test_maintain_stops_after_failed_step(tmp_path, monkeypatch):
    write_config(tmp_path)
    calls = []
    monkeypatch.setattr(
        cli_module, "_cmd_build_dataset", lambda config: calls.append("build") or 0
    )
    monkeypatch.setattr(
        cli_module, "_cmd_backtest", lambda config, args: calls.append("backtest") or 1
    )
    monkeypatch.setattr(
        cli_module, "_cmd_report", lambda config: calls.append("report") or 0
    )

    code = main(["--config", str(tmp_path / "config.toml"), "maintain"])

    assert code == 1
    assert calls == ["build", "backtest"]


def test_maintain_records_child_stages_and_marks_only_no_folds(tmp_path, monkeypatch):
    config = write_config(tmp_path)
    monkeypatch.setattr(cli_module, "_cmd_build_dataset", lambda _config: 0)

    def no_folds(_config, args):
        vars(args)["_failure_kind"] = "no_folds"
        return 1

    monkeypatch.setattr(cli_module, "_cmd_backtest", no_folds)
    assert main(["--config", str(tmp_path / "config.toml"), "maintain"]) == 1
    rows = load_runs(runs_path(config))
    parent = rows.filter(pl.col("record_kind") == "invocation").row(0, named=True)
    children = rows.filter(pl.col("record_kind") == "stage")
    assert children["command"].to_list() == ["build-dataset", "backtest"]
    assert children["parent_run_id"].to_list() == [parent["run_id"]] * 2
    assert parent["failure_kind"] == "no_folds"
    assert children["failure_kind"].to_list() == [None, "no_folds"]


def test_invalid_truth_qc_days_rejected_before_build(tmp_path, monkeypatch):
    config = write_config(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "_cmd_build_dataset",
        lambda _config: (_ for _ in ()).throw(AssertionError("build started")),
    )
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "--config",
                str(tmp_path / "config.toml"),
                "maintain",
                "--truth-qc-days",
                "0",
            ]
        )
    assert raised.value.code == 2
    assert not runs_path(config).exists()


def test_recover_deduplicates_against_an_existing_recovery(tmp_path):
    config = write_config(tmp_path)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(config.artifacts_dir / "auto-restore.lock"):
        code = main(["--config", str(tmp_path / "config.toml"), "recover"])
    assert code == EXIT_CONTENTION


def test_recover_waits_for_pipeline_lock_then_rechecks(tmp_path, monkeypatch):
    config = write_config(tmp_path)
    checked = Event()
    monkeypatch.setattr(
        cli_module,
        "_recovery_needed",
        lambda config, semantics: checked.set() or False,
    )
    started = Event()

    def run_recovery():
        started.set()
        return main(["--config", str(tmp_path / "config.toml"), "recover"])

    with ThreadPoolExecutor(max_workers=1) as pool:
        with pipeline_lock(config.dataset.dir):
            future = pool.submit(run_recovery)
            assert started.wait(timeout=1)
            with pytest.raises(FutureTimeoutError):
                future.result(timeout=0.05)
            assert not checked.is_set()
        assert future.result(timeout=2) == 0
    assert checked.is_set()


def test_recover_runs_chain_only_while_still_degraded(tmp_path, monkeypatch):
    write_config(tmp_path)
    calls = []
    monkeypatch.setattr(cli_module, "_recovery_needed", lambda config, semantics: True)
    monkeypatch.setattr(
        cli_module, "_cmd_build_dataset", lambda config: calls.append("build") or 0
    )
    monkeypatch.setattr(
        cli_module, "_cmd_backtest", lambda config, args: calls.append("backtest") or 0
    )
    monkeypatch.setattr(
        cli_module, "_cmd_report", lambda config: calls.append("report") or 0
    )

    assert main(["--config", str(tmp_path / "config.toml"), "recover"]) == 0
    assert calls == ["build", "backtest", "report"]


def test_recover_flags_reach_recheck_and_backtest(tmp_path, monkeypatch):
    config = write_config(tmp_path)
    captured = {}
    monkeypatch.setattr(
        cli_module,
        "_recovery_needed",
        lambda _config, semantics: captured.setdefault("recheck", semantics) or True,
    )
    monkeypatch.setattr(cli_module, "_cmd_build_dataset", lambda _config: 0)

    def backtest(_config, args):
        captured["backtest"] = (args.semantics, args.methods, args.window)
        return 0

    monkeypatch.setattr(cli_module, "_cmd_backtest", backtest)
    monkeypatch.setattr(cli_module, "_cmd_report", lambda _config: 0)
    assert (
        main(
            [
                "--config",
                str(tmp_path / "config.toml"),
                "recover",
                "--semantics",
                "mean",
                "--methods",
                "equal_weight",
                "--window",
                "rolling",
            ]
        )
        == 0
    )
    assert captured == {
        "recheck": "mean",
        "backtest": ("mean", "equal_weight", "rolling"),
    }
    rows = load_runs(runs_path(config))
    assert rows.filter(pl.col("record_kind") == "stage")["command"].to_list() == [
        "build-dataset",
        "backtest",
        "report",
    ]
