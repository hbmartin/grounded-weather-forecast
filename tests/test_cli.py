import json
import sqlite3
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from conftest import make_station_db, write_config

from grounded_weather_forecast import __version__
from grounded_weather_forecast import cli as cli_module
from grounded_weather_forecast.backtest.engine import BacktestRequest
from grounded_weather_forecast.cli import build_parser, main
from grounded_weather_forecast.contracts import HOURLY_VARIABLES, TruthSemantics
from grounded_weather_forecast.dataset.neighbors import NeighborChecks
from grounded_weather_forecast.runs import load_runs
from grounded_weather_forecast.serve.predict import UnsupportedMethodError


def test_backfill_start_and_truth_qc_days_are_parsed():
    parser = build_parser()
    backfill = parser.parse_args(
        ["backfill", "--provider", "open_meteo", "--start", "2026-02-01"]
    )
    truth_qc = parser.parse_args(["truth-qc", "--days", "45"])

    assert backfill.start == date(2026, 2, 1)
    assert truth_qc.days == 45


def test_mean_semantics_binds_only_dual_semantics_variables(tmp_path: Path) -> None:
    config = write_config(tmp_path)

    semantics = cli_module._semantics_by_variable(config, "mean", HOURLY_VARIABLES)

    assert semantics["temp_c"] is TruthSemantics.INTERVAL_MEAN
    assert semantics["wind_gust_ms"] is TruthSemantics.INSTANTANEOUS
    assert semantics["precip_mm"] is TruthSemantics.INSTANTANEOUS
    assert semantics["pop"] is TruthSemantics.INSTANTANEOUS
    # Selection filters evidence on this map while the engine records
    # semantics_for; any disagreement strands that variable's evidence on the
    # no-evidence fallback whenever --semantics mean is requested.
    request = BacktestRequest(
        variables=HOURLY_VARIABLES,
        methods=("equal_weight",),
        semantics=semantics,
    )
    recorded = {
        variable.name: request.semantics_for(variable) for variable in HOURLY_VARIABLES
    }
    assert recorded == semantics


class TestQcCommand:
    def test_qc_happy_path(self, tmp_path, capsys):
        write_config(tmp_path)
        make_station_db(
            tmp_path / "station.db",
            [
                ("2026-07-13 19:21:03", {"outTemp": 70.0, "outHumi": 50.0}),
                ("2026-07-13 19:22:03", {"outTemp": 71.0, "outHumi": 51.0}),
            ],
        )
        code = main(["--config", str(tmp_path / "config.toml"), "qc"])
        out = capsys.readouterr().out
        assert code == 0
        assert "observations: 2 samples" in out
        assert "hourly truth" in out

    def test_qc_empty_db(self, tmp_path, capsys):
        write_config(tmp_path)
        sqlite3.connect(tmp_path / "station.db").close()
        code = main(["--config", str(tmp_path / "config.toml"), "qc"])
        assert code == 1
        assert "no observations" in capsys.readouterr().out

    def test_bad_config(self, tmp_path, capsys):
        (tmp_path / "config.toml").write_text("[station]\n", encoding="utf-8")
        code = main(["--config", str(tmp_path / "config.toml"), "qc"])
        assert code == 2
        assert "config error" in capsys.readouterr().out

    def test_truth_qc_reports_unknown_verdicts_without_failing(
        self, tmp_path, capsys, monkeypatch
    ):
        config = write_config(
            tmp_path,
            extra_toml='[truth_qc]\nsynoptic_token = "test-token"\n',
        )
        make_station_db(
            tmp_path / "station.db",
            [("2026-07-13 19:21:03", {"outTemp": 70.0, "avgwind": 1.0})],
        )
        empty_daily = pl.DataFrame(
            schema={"date": pl.Date(), "station_minus_consensus_c": pl.Float64()}
        )
        empty_correlation = pl.DataFrame(
            schema={
                "valid_hour": pl.Datetime("us", "UTC"),
                "correlation": pl.Float64(),
            }
        )
        empty_comparison = pl.DataFrame(
            schema={
                "valid_hour": pl.Datetime("us", "UTC"),
                "t__temp_c__inst": pl.Float64(),
                "consensus_c": pl.Float64(),
                "difference": pl.Float64(),
            }
        )
        checks = NeighborChecks(
            daily_drift=empty_daily,
            rolling_correlation=empty_correlation,
            comparison=empty_comparison,
            drift_alert=None,
            correlation_alert=None,
            n_neighbors=0,
            overlap_hours=0,
            drift_reason="no overlap",
            correlation_reason="no overlap",
        )
        monkeypatch.setattr(
            "grounded_weather_forecast.dataset.neighbors.fetch_neighbor_checks",
            lambda *_args, **_kwargs: checks,
        )

        code = main(
            ["--config", str(tmp_path / "config.toml"), "truth-qc", "--days", "30"]
        )

        artifact = json.loads(
            (config.artifacts_dir / "truth_qc.json").read_text(encoding="utf-8")
        )
        # A cold start has no neighbour overlap yet. That is a normal state,
        # not a failure: the shipped `maintain` cron chains truth-qc, so a
        # non-zero exit would report the whole nightly job as broken for the
        # first week, and the run ledger would record it as a real fault.
        assert code == 0
        assert artifact["schema_version"] == 2
        assert artifact["drift_alert"] is None
        assert artifact["correlation_alert"] is None
        assert artifact["shield_alert"] is None
        output = capsys.readouterr().out
        assert "overlap: 0 hours" in output
        assert "no check is evaluable yet" in output


def test_predict_reports_unsupported_method_without_traceback(
    tmp_path, capsys, monkeypatch
):
    write_config(tmp_path)

    def unsupported(*_args, **_kwargs):
        raise UnsupportedMethodError(
            "method 'persistence' does not support daily.temp_max_c"
        )

    monkeypatch.setattr(
        "grounded_weather_forecast.serve.predict.predict",
        unsupported,
    )

    code = main(
        [
            "--config",
            str(tmp_path / "config.toml"),
            "predict",
            "--method",
            "persistence",
            "--no-history",
        ]
    )

    assert code == 1
    assert "cannot predict: method 'persistence'" in capsys.readouterr().out


def test_ensemble_store_failure_is_an_actionable_cli_error(
    tmp_path, capsys, monkeypatch
):
    write_config(tmp_path)
    monkeypatch.setattr(
        "grounded_weather_forecast.dataset.ensembles.ingest_ensembles",
        lambda _config: pl.DataFrame({"model": ["gefs"]}),
    )

    def fail_append(_path, _fresh):
        raise OSError("disk full")

    monkeypatch.setattr(
        "grounded_weather_forecast.dataset.ensembles.append_ensembles",
        fail_append,
    )

    code = main(["--config", str(tmp_path / "config.toml"), "ingest-ensembles"])

    assert code == 1
    assert "ensemble ingest failed: disk full" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("failing_stage", "exception"),
    [
        ("build_truth", OSError("truth unavailable")),
        ("verify_history", ValueError("history invalid")),
        ("compare_to_backtest", ValueError("comparison invalid")),
    ],
)
def test_report_skips_any_self_verification_failure(
    tmp_path, capsys, monkeypatch, failing_stage, exception
):
    config = write_config(tmp_path)
    scores_dir = config.dataset.dir / "scores"
    scores_dir.mkdir(parents=True)
    (scores_dir / "scores_hourly_live.parquet").touch()
    config.predict.history_path.parent.mkdir(parents=True, exist_ok=True)
    config.predict.history_path.touch()
    scores = pl.DataFrame({"source_kind": ["live"]})
    empty = pl.DataFrame()
    written_sections = []

    monkeypatch.setattr(
        "grounded_weather_forecast.backtest.scores.load_scores",
        lambda _path: scores,
    )
    # dashboard.context binds load_scores at import; patching through the
    # string target imports it NOW so the mock cannot become its permanent
    # module-level binding when this test is the first to touch it.
    monkeypatch.setattr(
        "grounded_weather_forecast.dashboard.context.load_scores",
        lambda _path: scores,
    )
    monkeypatch.setattr(
        "grounded_weather_forecast.reports.leaderboard.leaderboard",
        lambda _scores, **_kwargs: empty,
    )
    monkeypatch.setattr(
        "grounded_weather_forecast.reports.leaderboard.aggregate_leaderboard",
        lambda _board: empty,
    )
    monkeypatch.setattr(
        "grounded_weather_forecast.reports.leaderboard.slice_winners",
        lambda *_args, **_kwargs: empty,
    )
    monkeypatch.setattr(
        "grounded_weather_forecast.reports.eprocess.promotion_comparison",
        lambda *_args, **_kwargs: empty,
    )
    monkeypatch.setattr(
        "grounded_weather_forecast.reports.render.print_summary",
        lambda *_args, **_kwargs: None,
    )

    def write_report(_directory, name, _title, sections):
        written_sections.append(sections)
        return tmp_path / f"{name}.md"

    monkeypatch.setattr(
        "grounded_weather_forecast.reports.render.write_markdown_report",
        write_report,
    )
    monkeypatch.setattr(
        "grounded_weather_forecast.dashboard.write_dashboard",
        lambda _config: tmp_path / "dashboard.html",
    )

    def maybe_fail(stage, result):
        if failing_stage == stage:
            raise exception
        return result

    monkeypatch.setattr(
        "grounded_weather_forecast.dataset.matrix.build_truth",
        lambda _config: maybe_fail("build_truth", (empty, empty, empty)),
    )
    monkeypatch.setattr(
        "grounded_weather_forecast.reports.verification.verify_history",
        lambda *_args, **_kwargs: maybe_fail("verify_history", empty),
    )
    monkeypatch.setattr(
        "grounded_weather_forecast.reports.verification.compare_to_backtest",
        lambda *_args, **_kwargs: maybe_fail("compare_to_backtest", empty),
    )

    assert cli_module._cmd_report(config) == 0
    assert "self-verification skipped" in capsys.readouterr().out
    assert all(
        title != "Self-verification (served vs realized)"
        for sections in written_sections
        for title, _frame in sections
    )


class TestRunLedger:
    def test_successful_run_is_recorded(self, tmp_path):
        config = write_config(tmp_path)
        make_station_db(
            tmp_path / "station.db",
            [
                ("2026-07-13 19:21:03", {"outTemp": 70.0, "outHumi": 50.0}),
                ("2026-07-13 19:22:03", {"outTemp": 71.0, "outHumi": 51.0}),
            ],
        )
        code = main(["--config", str(tmp_path / "config.toml"), "qc"])
        assert code == 0
        frame = load_runs(config.dataset.dir / "runs.parquet")
        assert frame.height == 1
        row = frame.row(0, named=True)
        assert row["command"] == "qc"
        assert row["exit_code"] == 0
        assert row["error"] is None
        assert row["dataset_fingerprint"] == "unknown"
        assert row["code_version"] == __version__
        assert row["duration_ms"] >= 0

    def test_config_error_has_no_configured_ledger_destination(self, tmp_path):
        (tmp_path / "config.toml").write_text("[station]\n", encoding="utf-8")
        code = main(["--config", str(tmp_path / "config.toml"), "qc"])
        assert code == 2
        assert not (tmp_path / "data" / "runs.parquet").exists()

    def test_raised_command_is_recorded_and_propagates(self, tmp_path, monkeypatch):
        config = write_config(tmp_path)

        def interrupt(_config):
            raise KeyboardInterrupt

        monkeypatch.setattr("grounded_weather_forecast.cli._cmd_qc", interrupt)
        with pytest.raises(KeyboardInterrupt):
            main(["--config", str(tmp_path / "config.toml"), "qc"])
        row = load_runs(config.dataset.dir / "runs.parquet").row(0, named=True)
        assert row["error"] == "KeyboardInterrupt"
        assert row["exit_code"] is None

    def test_fingerprint_captured_from_manifest(self, tmp_path):
        config = write_config(tmp_path)
        make_station_db(
            tmp_path / "station.db",
            [("2026-07-13 19:21:03", {"outTemp": 70.0, "outHumi": 50.0})],
        )
        config.dataset.dir.mkdir(parents=True, exist_ok=True)
        (config.dataset.dir / "manifest.json").write_text(
            json.dumps({"fingerprint": "feedface00000000"}), encoding="utf-8"
        )
        main(["--config", str(tmp_path / "config.toml"), "qc"])
        row = load_runs(config.dataset.dir / "runs.parquet").row(0, named=True)
        assert row["dataset_fingerprint"] == "feedface00000000"
        assert row["config_fingerprint"] not in ("", "unknown")

    def test_unexpected_telemetry_error_does_not_break_command(
        self, tmp_path, monkeypatch
    ):
        write_config(tmp_path)
        make_station_db(
            tmp_path / "station.db",
            [
                ("2026-07-13 19:21:03", {"outTemp": 70.0, "outHumi": 50.0}),
                ("2026-07-13 19:22:03", {"outTemp": 71.0, "outHumi": 51.0}),
            ],
        )

        def fail_run_id(_command, _started_at):
            raise RuntimeError("unexpected telemetry bug")

        monkeypatch.setattr(
            "grounded_weather_forecast.runs.run_id_for",
            fail_run_id,
        )

        assert main(["--config", str(tmp_path / "config.toml"), "qc"]) == 0


def test_report_writes_evidence_ledgers_and_churn(tmp_path, capsys):
    from datetime import timedelta

    from conftest import synthetic_hourly_matrix, utc

    from grounded_weather_forecast.backtest.engine import run_backtest
    from grounded_weather_forecast.backtest.scores import scores_path, write_scores
    from grounded_weather_forecast.contracts import hourly_variable

    config = write_config(
        tmp_path,
        extra_toml="\n[backtest]\ninitial_train_days = 10\nstep_days = 5\n",
    )
    matrix = synthetic_hourly_matrix(days=25, biases={"alpha": 3.0})
    scores = run_backtest(
        matrix,
        BacktestRequest(
            variables=(hourly_variable("temp_c"),),
            methods=(
                "equal_weight",
                "grounded_equal_weight",
                "best_provider",
                "damped_grounded_equal_weight",
            ),
        ),
        config,
    )
    write_scores(scores, scores_path(config.dataset.dir / "scores", "hourly", "live"))
    releases = config.artifacts_dir / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    for name, when, method in (
        ("relA", utc(2026, 8, 1), "equal_weight"),
        ("relB", utc(2026, 8, 2), "grounded_equal_weight"),
    ):
        (releases / f"{name}.json").write_text(
            json.dumps(
                {
                    "release_id": name,
                    "promoted_at": when.isoformat(),
                    "dataset_fingerprint": "ds1",
                    "config_fingerprint": "cfg1",
                    "selections": {
                        "hourly.temp_c.0-1h": {
                            "method_id": method,
                            "reason": "test",
                            "evaluation_id": "eval1",
                            "code_version": "code1",
                            "n": 100,
                            "mae": 1.0,
                            "truth_semantics": "mean",
                        }
                    },
                }
            )
        )

    assert cli_module._cmd_report(config) == 0
    quality_path = config.artifacts_dir / "history" / "quality.parquet"
    assert quality_path.exists()
    first_height = pl.read_parquet(quality_path).height
    assert first_height > 0
    assert (config.reports_dir / "selection_churn.md").exists()
    churn_path = config.artifacts_dir / "history" / "churn.parquet"
    churn_height = pl.read_parquet(churn_path).height
    capsys.readouterr()

    # a second run must be a complete no-op on every ledger
    assert cli_module._cmd_report(config) == 0
    assert pl.read_parquet(quality_path).height == first_height
    assert pl.read_parquet(churn_path).height == churn_height


def test_report_served_ledger_records_only_the_products_own_rows(tmp_path, monkeypatch):
    from conftest import synthetic_hourly_matrix

    from grounded_weather_forecast.backtest.engine import run_backtest
    from grounded_weather_forecast.backtest.scores import scores_path, write_scores
    from grounded_weather_forecast.contracts import hourly_variable

    config = write_config(
        tmp_path,
        extra_toml="\n[backtest]\ninitial_train_days = 10\nstep_days = 5\n",
    )
    matrix = synthetic_hourly_matrix(days=25, biases={"alpha": 3.0})
    scores = run_backtest(
        matrix,
        BacktestRequest(
            variables=(hourly_variable("temp_c"),),
            methods=("equal_weight", "best_provider"),
        ),
        config,
    )
    write_scores(scores, scores_path(config.dataset.dir / "scores", "hourly", "live"))
    config.predict.history_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"placeholder": [1]}).write_parquet(config.predict.history_path)
    # Whole-history verify frame spanning two products: the report loop only
    # has an HOURLY board here, so a daily row reaching the ledger would be
    # the cross-product poisoning regression (first-write-wins dedupe then
    # blocks the correct rows forever).
    live = pl.DataFrame(
        {
            "product": ["hourly", "daily"],
            "variable": ["temp_c", "temp_max_c"],
            "truth_semantics": ["mean", "inst"],
            "lead_bucket": ["0-1h", "D1"],
            "method_id": ["equal_weight", "equal_weight"],
            "dataset_fingerprint": ["ds1", "ds1"],
            "release_id": ["relA", "relA"],
            "n": [10, 10],
            "live_mae": [1.0, 1.0],
            "live_rmse": [1.0, 1.0],
            "live_bias": [0.0, 0.0],
        }
    )
    monkeypatch.setattr(
        "grounded_weather_forecast.dataset.matrix.build_truth",
        lambda _config: (pl.DataFrame(), pl.DataFrame(), pl.DataFrame()),
    )
    monkeypatch.setattr(
        "grounded_weather_forecast.reports.verification.verify_history",
        lambda *_args, **_kwargs: live,
    )

    assert cli_module._cmd_report(config) == 0
    served_path = config.artifacts_dir / "history" / "served_quality.parquet"
    assert served_path.exists()
    served = pl.read_parquet(served_path)
    assert served.height > 0
    assert set(served["product"].to_list()) == {"hourly"}
