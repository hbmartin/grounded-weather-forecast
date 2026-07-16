from datetime import UTC, datetime, timedelta

import polars as pl
from conftest import synthetic_hourly_matrix, utc, write_config

from grounded_weather_forecast.backtest.engine import BacktestRequest, run_backtest
from grounded_weather_forecast.backtest.scores import scores_path, write_scores
from grounded_weather_forecast.contracts import hourly_variable
from grounded_weather_forecast.serve.selection import (
    FALLBACK_METHOD,
    Selection,
    method_for,
    select_methods,
    selection_report,
)


def scored_config(tmp_path, extra=""):
    config = write_config(
        tmp_path,
        extra_toml="\n[backtest]\ninitial_train_days = 10\nstep_days = 5\n" + extra,
    )
    matrix = synthetic_hourly_matrix(days=25, biases={"alpha": 3.0})
    scores = run_backtest(
        matrix,
        BacktestRequest(
            variables=(hourly_variable("temp_c"),),
            methods=("equal_weight", "grounded_equal_weight", "best_provider"),
        ),
        config,
    )
    write_scores(scores, scores_path(config.dataset.dir / "scores", "hourly", "live"))
    return config


class TestSelectMethods:
    def test_winner_per_slice(self, tmp_path):
        config = scored_config(tmp_path)
        selections = select_methods(config, config.dataset.dir / "scores")
        assert selections
        for (product, variable, _bucket), chosen in selections.items():
            assert product == "hourly"
            assert variable == "temp_c"
            assert chosen.n > 0
            assert chosen.mae is not None
            assert "lowest backtest MAE" in chosen.reason
        # grounding removes alpha's +3C bias, so it must win somewhere
        assert any(c.method_id == "grounded_equal_weight" for c in selections.values())
        assert all(c.evaluation_id for c in selections.values())
        assert all(c.release_id for c in selections.values())
        assert list((config.artifacts_dir / "releases").glob("*.json"))

    def test_config_pin_overrides(self, tmp_path):
        config = scored_config(
            tmp_path,
            extra='\n[predict.methods]\n"hourly.temp_c" = "best_provider"\n',
        )
        selections = select_methods(config, config.dataset.dir / "scores")
        assert {c.method_id for c in selections.values()} == {"best_provider"}
        assert all(c.reason == "pinned in config" for c in selections.values())

    def test_report_frame(self, tmp_path):
        config = scored_config(tmp_path)
        report = selection_report(select_methods(config, config.dataset.dir / "scores"))
        assert set(report.columns) >= {
            "product",
            "variable",
            "lead_bucket",
            "method_id",
            "reason",
        }

    def test_no_scores_is_empty(self, tmp_path):
        config = write_config(tmp_path)
        assert select_methods(config, tmp_path / "nothing") == {}

    def test_historical_issue_rejects_future_evaluation(self, tmp_path):
        config = scored_config(tmp_path)
        as_of = utc(2026, 1, 1) - timedelta(days=1)
        assert select_methods(config, config.dataset.dir / "scores", as_of=as_of) == {}

    def test_historical_issue_loads_release_that_already_existed(self, tmp_path):
        config = scored_config(tmp_path)
        promoted = select_methods(config, config.dataset.dir / "scores")
        restored = select_methods(
            config,
            config.dataset.dir / "scores",
            as_of=datetime.now(tz=UTC) + timedelta(minutes=1),
        )
        assert restored
        assert {choice.release_id for choice in restored.values()} == {
            choice.release_id for choice in promoted.values()
        }


class TestMethodFor:
    def test_falls_back_explicitly(self, tmp_path):
        config = write_config(tmp_path)
        chosen = method_for({}, "hourly", "temp_c", "0-1h", config)
        assert chosen.method_id == FALLBACK_METHOD
        assert "no backtest evidence" in chosen.reason

    def test_pin_wins_over_scores(self, tmp_path):
        config = write_config(
            tmp_path, extra_toml='\n[predict.methods]\n"hourly.temp_c" = "gbm"\n'
        )
        selections = {("hourly", "temp_c", "0-1h"): Selection("equal_weight", "x")}
        assert (
            method_for(selections, "hourly", "temp_c", "0-1h", config).method_id
            == "gbm"
        )

    def test_unknown_bucket_falls_back(self, tmp_path):
        config = write_config(tmp_path)
        assert (
            method_for({}, "hourly", "temp_c", None, config).method_id
            == FALLBACK_METHOD
        )


class TestScoresProvenance:
    def test_reads_both_kinds_separately(self, tmp_path):
        config = scored_config(tmp_path)
        synthetic = synthetic_hourly_matrix(days=25, source_kind="synthetic")
        scores = run_backtest(
            synthetic,
            BacktestRequest(
                variables=(hourly_variable("temp_c"),), methods=("equal_weight",)
            ),
            config,
        )
        write_scores(
            scores, scores_path(config.dataset.dir / "scores", "hourly", "synthetic")
        )
        selections = select_methods(config, config.dataset.dir / "scores")
        # both files load without a MixedProvenanceError
        assert selections
        live_evaluation = pl.read_parquet(
            scores_path(config.dataset.dir / "scores", "hourly", "live")
        )["evaluation_id"][0]
        assert {choice.evaluation_id for choice in selections.values()} == {
            live_evaluation
        }
        assert isinstance(
            pl.read_parquet(
                scores_path(config.dataset.dir / "scores", "hourly", "synthetic")
            ),
            pl.DataFrame,
        )
        assert utc(2026, 1, 1)  # sanity: fixture epoch unchanged
