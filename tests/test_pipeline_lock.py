"""The pipeline mutex and short dataset snapshot lock."""

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
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
    config = write_config(tmp_path)
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
    assert calls == ["build", "backtest", "truth-qc", "report"]
    frame = load_runs(runs_path(config))
    stages = frame.filter(pl.col("record_kind") == "stage")
    assert stages["command"].to_list() == [
        "build-dataset",
        "backtest",
        "truth-qc",
        "report",
    ]
    parent = frame.filter(pl.col("record_kind") == "invocation").row(0, named=True)
    assert stages["parent_run_id"].to_list() == [parent["run_id"]] * 4


@pytest.mark.parametrize("report_exit", [0, 3])
def test_maintain_reports_after_truth_qc_failure_and_returns_first_failure(
    tmp_path, monkeypatch, report_exit
):
    config = write_config(tmp_path)
    config.artifacts_dir.mkdir(parents=True)
    artifact = config.artifacts_dir / "truth_qc.json"
    artifact.write_text('{"schema_version": 3, "drift_reason": "previous verdict"}')
    original = artifact.read_bytes()
    calls = []
    monkeypatch.setattr(
        cli_module, "_cmd_build_dataset", lambda _config: calls.append("build") or 0
    )
    monkeypatch.setattr(
        cli_module,
        "_cmd_backtest",
        lambda _config, _args: calls.append("backtest") or 0,
    )
    monkeypatch.setattr(
        cli_module,
        "_cmd_truth_qc",
        lambda _config, _args: calls.append("truth-qc") or 2,
    )
    monkeypatch.setattr(
        cli_module, "_cmd_report", lambda _config: calls.append("report") or report_exit
    )
    assert main(["--config", str(tmp_path / "config.toml"), "maintain"]) == 2
    assert calls == ["build", "backtest", "truth-qc", "report"]
    stages = load_runs(runs_path(config)).filter(pl.col("record_kind") == "stage")
    assert stages["command"].to_list() == [
        "build-dataset",
        "backtest",
        "truth-qc",
        "report",
    ]
    assert stages["exit_code"].to_list() == [0, 0, 2, report_exit]
    assert artifact.read_bytes() == original


@pytest.mark.parametrize("stage", ["truth-qc", "report"])
def test_maintain_unexpected_exception_still_propagates(tmp_path, monkeypatch, stage):
    config = write_config(tmp_path)
    calls = []
    monkeypatch.setattr(cli_module, "_cmd_build_dataset", lambda _config: 0)
    monkeypatch.setattr(cli_module, "_cmd_backtest", lambda *_args: 0)

    def fail(name):
        calls.append(name)
        if name == stage:
            raise RuntimeError("unexpected stage failure")
        return 1

    monkeypatch.setattr(cli_module, "_cmd_truth_qc", lambda *_args: fail("truth-qc"))
    monkeypatch.setattr(cli_module, "_cmd_report", lambda *_args: fail("report"))
    with pytest.raises(RuntimeError, match="unexpected stage failure"):
        main(["--config", str(tmp_path / "config.toml"), "maintain"])
    assert calls == (["truth-qc"] if stage == "truth-qc" else ["truth-qc", "report"])
    stages = load_runs(runs_path(config)).filter(pl.col("record_kind") == "stage")
    assert stages["command"][-1] == stage
    assert stages["error"][-1] == "RuntimeError"
    assert stages["error_detail"][-1] == "unexpected stage failure"
    assert stages["exit_code"][-1] is None


def test_qc_stage_records_child_arguments_and_parent_keeps_invocation(
    tmp_path, monkeypatch
):
    import json

    config = write_config(tmp_path)
    monkeypatch.setattr(cli_module, "_cmd_build_dataset", lambda *_args: 0)
    monkeypatch.setattr(cli_module, "_cmd_backtest", lambda *_args: 0)
    monkeypatch.setattr(cli_module, "_cmd_truth_qc", lambda *_args: 0)
    monkeypatch.setattr(cli_module, "_cmd_report", lambda *_args: 0)
    assert (
        main(
            [
                "--config",
                str(tmp_path / "config.toml"),
                "maintain",
                "--truth-qc-days",
                "17",
            ]
        )
        == 0
    )
    rows = load_runs(runs_path(config))
    parent = rows.filter(pl.col("record_kind") == "invocation").row(0, named=True)
    qc = rows.filter(pl.col("command") == "truth-qc").row(0, named=True)
    assert json.loads(qc["args_json"]) == {"days": 17}
    assert json.loads(parent["args_json"])["truth_qc_days"] == 17
    assert qc["parent_run_id"] == parent["run_id"]


def test_pipeline_child_records_do_not_inherit_parent_failure_kind(tmp_path):
    import argparse

    config = write_config(tmp_path)
    args = argparse.Namespace(command="maintain", _failure_kind="parent-only")
    assert (
        cli_module._run_pipeline_steps(
            config, args, (("truth-qc", lambda: 0), ("report", lambda: 0))
        )
        == 0
    )
    rows = load_runs(runs_path(config))
    assert rows["failure_kind"].to_list() == [None, None]
    assert args._failure_kind == "parent-only"


def test_stage_namespaces_are_public_and_metadata_is_explicit(tmp_path, monkeypatch):
    import argparse
    from datetime import UTC, datetime

    config = write_config(tmp_path)
    args = argparse.Namespace(
        command="maintain",
        truth_qc_days=17,
        _failure_kind="parent-only",
        _run_started_at=datetime.now(UTC),
        _unknown_private="never copy",
        _error_detail="old parent detail",
    )
    truth_args = argparse.Namespace(
        days=17, _error_detail="old QC detail", _other="private"
    )
    recorded = []
    real_record = cli_module._record_run

    def record(config, stage, **metadata):
        assert not any(key.startswith("_") for key in vars(stage))
        recorded.append((vars(stage).copy(), metadata))
        real_record(config, stage, **metadata)

    def fail_qc():
        assert not hasattr(truth_args, "_error_detail")
        truth_args._error_detail = "  handled\nneighbor failure  "
        return 1

    def report():
        assert not hasattr(args, "_error_detail")
        return 0

    monkeypatch.setattr(cli_module, "_record_run", record)
    assert (
        cli_module._run_pipeline_steps(
            config,
            args,
            (("truth-qc", fail_qc), ("report", report)),
            continue_on_failure=frozenset({"truth-qc"}),
            stage_args={"truth-qc": truth_args},
        )
        == 1
    )
    assert recorded[0][0] == {"days": 17, "command": "truth-qc"}
    assert recorded[0][1]["error_detail"] == "  handled\nneighbor failure  "
    assert recorded[1][1]["error_detail"] is None
    assert all(metadata["failure_kind"] is None for _, metadata in recorded)
    assert all(
        metadata["parent_run_id"] == cli_module._parent_run_id(args)
        for _, metadata in recorded
    )
    rows = load_runs(runs_path(config))
    assert rows["error_detail"].to_list() == ["handled neighbor failure", None]
    assert args._failure_kind == "parent-only"


@pytest.mark.parametrize("rule", ["mcs", "legacy", "seq_mcs"])
@pytest.mark.parametrize("previous_artifact", [False, True])
def test_neighbor_fetch_failure_still_refreshes_real_report(
    tmp_path, monkeypatch, capsys, rule, previous_artifact
):
    from grounded_weather_forecast.backtest.scores import write_scores
    from grounded_weather_forecast.dashboard.context import collect_context
    from grounded_weather_forecast.dataset import neighbors
    from grounded_weather_forecast.reports import evidence
    from score_fixtures import scores

    config = write_config(tmp_path, extra_toml=f'[promotion]\nrule = "{rule}"')
    config.artifacts_dir.mkdir(parents=True)
    artifact = config.artifacts_dir / "truth_qc.json"
    if previous_artifact:
        import json

        artifact.write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "generated_at": (
                        datetime.now(tz=UTC) - timedelta(hours=1)
                    ).isoformat(),
                    "drift_reason": "previous verdict",
                    "drift_verdict": {
                        "exceeded": False,
                        "latched": False,
                        "reason": "previous quiet verdict",
                        "series": [
                            {"date": "2026-10-10", "difference": 0.1, "cusum": 0.2}
                        ],
                    },
                }
            )
        )
    original = artifact.read_bytes() if previous_artifact else None
    directory = config.dataset.dir / "scores"
    directory.mkdir(parents=True)
    good = directory / "scores_hourly_live_expanding_readable.parquet"
    write_scores(scores(), good)
    monkeypatch.setattr(cli_module, "_cmd_build_dataset", lambda _config: 0)
    monkeypatch.setattr(cli_module, "_cmd_backtest", lambda *_args: 0)
    # Exercise the real truth-QC handler without fetching a live provider.
    monkeypatch.setattr(
        "grounded_weather_forecast.dataset.matrix.build_truth",
        lambda _config: (pl.DataFrame(), pl.DataFrame(), pl.DataFrame()),
    )

    def unavailable(*_args, **_kwargs):
        raise OSError("neighbor HTTP unavailable")

    monkeypatch.setattr(neighbors, "fetch_neighbor_checks", unavailable)
    report = cli_module._cmd_report

    def locked_report(config):
        with pytest.raises(Timeout):
            with pipeline_lock(config.dataset.dir, timeout=0.01):
                pass
        stages = load_runs(runs_path(config)).filter(pl.col("record_kind") == "stage")
        assert stages["command"][-1] == "truth-qc"
        assert stages["exit_code"][-1] == 1
        return report(config)

    monkeypatch.setattr(cli_module, "_cmd_report", locked_report)
    assert main(["--config", str(tmp_path / "config.toml"), "maintain"]) == 1
    assert (config.reports_dir / "dashboard.html").exists()
    assert (
        config.reports_dir / "leaderboard_hourly_live_expanding_readable.md"
    ).exists()
    assert (config.reports_dir / "pipeline_health.md").exists()
    assert not evidence.read_ledger(
        config.artifacts_dir
        / "history"
        / f"{evidence.EVALUATIONS_LEDGER.name}.parquet",
        evidence.EVALUATIONS_LEDGER.schema,
    ).is_empty()
    ctx = collect_context(config)
    from grounded_weather_forecast.dashboard.zones.data_trust import _truth_cross_check

    panel = _truth_cross_check(ctx)
    assert panel.status == "amber"
    assert "latest truth-QC refresh failed" in (panel.intro or panel.empty_reason)
    if previous_artifact:
        assert ctx.truth_qc["drift_reason"] == "previous verdict"
        assert artifact.read_bytes() == original
        assert panel.chart is not None
        assert "previous quiet verdict" in panel.intro
    else:
        assert ctx.truth_qc is None
        assert not artifact.exists()
    stages = ctx.runs.filter(pl.col("record_kind") == "stage")
    assert stages["exit_code"].to_list() == [0, 0, 1, 0]
    assert stages["error_detail"].to_list() == [
        None,
        None,
        "neighbor HTTP unavailable",
        None,
    ]
    assert "neighbor HTTP unavailable" in (panel.intro or panel.empty_reason)
    from grounded_weather_forecast.dashboard.derive import derive
    from grounded_weather_forecast.dashboard.writer import _alert_inputs
    from grounded_weather_forecast.reports.alerts import evaluate_alerts

    alerts = [
        a
        for a in evaluate_alerts(_alert_inputs(ctx, derive(ctx)))
        if a.panel_id == "truth-qc-freshness"
    ]
    assert len(alerts) == 1
    assert "neighbor HTTP unavailable" in alerts[0].message
    assert "neighbor HTTP unavailable" in capsys.readouterr().out
    assert main(["--config", str(tmp_path / "config.toml"), "truth-qc"]) == 1
    invocation = load_runs(runs_path(config)).row(-1, named=True)
    assert invocation["record_kind"] == "invocation"
    assert invocation["error"] is None
    assert invocation["error_detail"] == "neighbor HTTP unavailable"
    if previous_artifact:
        assert artifact.read_bytes() == original
    # A real subsequent standalone refresh supersedes the failed stage.
    checks = neighbors.NeighborChecks(
        daily_drift=pl.DataFrame(),
        rolling_correlation=pl.DataFrame(),
        comparison=pl.DataFrame(),
        drift_alert=None,
        correlation_alert=None,
        n_neighbors=0,
        overlap_hours=0,
        drift_reason="no overlap",
        correlation_reason="no overlap",
    )
    monkeypatch.setattr(
        neighbors, "fetch_neighbor_checks", lambda *_args, **_kwargs: checks
    )
    assert main(["--config", str(tmp_path / "config.toml"), "truth-qc"]) == 0
    refreshed = _truth_cross_check(collect_context(config))
    assert refreshed.status == "info"
    assert load_runs(runs_path(config))["error_detail"][-1] is None
    assert "failed" not in refreshed.intro
    assert any(
        stat.label == "refreshed" and stat.value != "unknown"
        for stat in refreshed.stats
    )


def test_maintain_runs_truth_qc_before_partial_report_failure(
    tmp_path, monkeypatch, capsys
):
    from grounded_weather_forecast.backtest.scores import write_scores
    from grounded_weather_forecast.dashboard.context import collect_context
    from score_fixtures import scores

    config = write_config(tmp_path)
    directory = config.dataset.dir / "scores"
    directory.mkdir(parents=True)
    good = directory / "scores_hourly_live_expanding_readable.parquet"
    write_scores(scores(), good)
    damaged = directory / "scores_hourly_live_expanding_truncated.parquet"
    damaged.write_bytes(b"PAR1\x00\x00")
    calls = []
    monkeypatch.setattr(
        cli_module, "_cmd_build_dataset", lambda _config: calls.append("build") or 0
    )
    monkeypatch.setattr(
        cli_module,
        "_cmd_backtest",
        lambda _config, _args: calls.append("backtest") or 0,
    )
    monkeypatch.setattr(
        cli_module,
        "_cmd_truth_qc",
        lambda _config, _args: calls.append("truth-qc") or 0,
    )
    report = cli_module._cmd_report

    def record_report(config):
        calls.append("report")
        return report(config)

    monkeypatch.setattr(cli_module, "_cmd_report", record_report)
    assert main(["--config", str(tmp_path / "config.toml"), "maintain"]) == 1
    assert calls == ["build", "backtest", "truth-qc", "report"]
    dashboard = (config.reports_dir / "dashboard.html").read_text()
    assert damaged.name in dashboard
    assert (
        config.reports_dir / "leaderboard_hourly_live_expanding_readable.md"
    ).exists()
    assert collect_context(config).unreadable_scores == (damaged.stem,)
    stages = load_runs(runs_path(config)).filter(pl.col("record_kind") == "stage")
    assert stages["command"].to_list() == [
        "build-dataset",
        "backtest",
        "truth-qc",
        "report",
    ]
    assert stages["exit_code"].to_list() == [0, 0, 0, 1]
    assert damaged.name in capsys.readouterr().err
    assert damaged.exists()


@pytest.mark.parametrize("failed_stage", ["build", "backtest"])
def test_maintain_stops_after_failed_step(tmp_path, monkeypatch, failed_stage):
    write_config(tmp_path)
    calls = []
    monkeypatch.setattr(
        cli_module,
        "_cmd_build_dataset",
        lambda config: calls.append("build") or int(failed_stage == "build"),
    )
    monkeypatch.setattr(
        cli_module, "_cmd_backtest", lambda config, args: calls.append("backtest") or 1
    )
    monkeypatch.setattr(
        cli_module, "_cmd_report", lambda config: calls.append("report") or 0
    )
    monkeypatch.setattr(
        cli_module, "_cmd_truth_qc", lambda *_args: calls.append("truth-qc") or 0
    )

    code = main(["--config", str(tmp_path / "config.toml"), "maintain"])

    assert code == 1
    assert calls == (["build"] if failed_stage == "build" else ["build", "backtest"])


@pytest.mark.parametrize("failure", [None, "truth-qc", "report"])
def test_maintain_records_pending_and_does_not_hide_later_failure(
    tmp_path, monkeypatch, failure
):
    config = write_config(tmp_path)
    monkeypatch.setattr(cli_module, "_cmd_build_dataset", lambda _config: 0)

    def no_folds(_config, args):
        vars(args)["_failure_kind"] = "no_folds"
        return cli_module.EXIT_NO_FOLDS

    monkeypatch.setattr(cli_module, "_cmd_backtest", no_folds)
    monkeypatch.setattr(
        cli_module, "_cmd_truth_qc", lambda *_a: int(failure == "truth-qc")
    )
    monkeypatch.setattr(cli_module, "_cmd_report", lambda *_a: int(failure == "report"))
    assert main(["--config", str(tmp_path / "config.toml"), "maintain"]) == int(
        failure is not None
    )
    rows = load_runs(runs_path(config))
    parent = rows.filter(pl.col("record_kind") == "invocation").row(0, named=True)
    children = rows.filter(pl.col("record_kind") == "stage")
    assert children["command"].to_list() == [
        "build-dataset",
        "backtest",
        "truth-qc",
        "report",
    ]
    assert children["parent_run_id"].to_list() == [parent["run_id"]] * 4
    assert parent["failure_kind"] == ("no_folds" if failure is None else None)
    assert children["failure_kind"].to_list() == [None, "no_folds", None, None]


def test_recovery_remains_incomplete_with_pending_backtest(tmp_path, monkeypatch):
    write_config(tmp_path)
    monkeypatch.setattr(cli_module, "_recovery_needed", lambda *_a: True)
    monkeypatch.setattr(cli_module, "_cmd_build_dataset", lambda *_a: 0)

    def pending(_config, args):
        args._failure_kind = "no_folds"
        return cli_module.EXIT_NO_FOLDS

    monkeypatch.setattr(cli_module, "_cmd_backtest", pending)

    def unexpected(*_a):
        pytest.fail("incomplete recovery was reported as finished")

    monkeypatch.setattr(cli_module, "_cmd_report", unexpected)
    assert (
        main(["--config", str(tmp_path / "config.toml"), "recover"])
        == cli_module.EXIT_NO_FOLDS
    )


def test_pending_maintenance_exception_is_not_a_pending_parent(tmp_path, monkeypatch):
    config = write_config(tmp_path)
    monkeypatch.setattr(cli_module, "_cmd_build_dataset", lambda *_a: 0)

    def pending(_config, args):
        args._failure_kind = "no_folds"
        return cli_module.EXIT_NO_FOLDS

    monkeypatch.setattr(cli_module, "_cmd_backtest", pending)
    monkeypatch.setattr(cli_module, "_cmd_truth_qc", lambda *_a: 0)

    def crash(*_a):
        raise OSError("report fixture failed")

    monkeypatch.setattr(cli_module, "_cmd_report", crash)
    with pytest.raises(OSError, match="report fixture failed"):
        main(["--config", str(tmp_path / "config.toml"), "maintain"])
    parent = (
        load_runs(runs_path(config))
        .filter(pl.col("record_kind") == "invocation")
        .row(0, named=True)
    )
    assert parent["error"] == "OSError"
    assert parent["failure_kind"] is None


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
