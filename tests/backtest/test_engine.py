import polars as pl
import pytest
from conftest import synthetic_hourly_matrix, write_config

from omni_forecast.backtest.engine import (
    BacktestRequest,
    run_backtest,
    variables_from_names,
)
from omni_forecast.backtest.scores import (
    SCORES_SCHEMA,
    empty_scores,
    load_scores,
    scores_path,
    write_scores,
)
from omni_forecast.contracts import (
    HOURLY_VARIABLES,
    MixedProvenanceError,
    hourly_variable,
)


@pytest.fixture
def config(tmp_path):
    return write_config(
        tmp_path,
        extra_toml="\n[backtest]\ninitial_train_days = 10\nstep_days = 5\n",
    )


@pytest.fixture
def matrix():
    return synthetic_hourly_matrix(days=25, biases={"alpha": 2.0})


class TestRunBacktest:
    def test_scores_frame(self, config, matrix):
        request = BacktestRequest(
            variables=(hourly_variable("temp_c"),),
            methods=("equal_weight", "climatology"),
        )
        scores = run_backtest(matrix, request, config)
        assert scores.schema == SCORES_SCHEMA
        assert set(scores["method_id"].unique()) == {"equal_weight", "climatology"}
        assert scores["y_pred"].null_count() == 0
        assert scores["source_kind"].unique().to_list() == ["live"]
        assert scores["evaluation_id"].null_count() == 0
        assert scores["dataset_fingerprint"].unique().to_list() == ["unknown"]
        assert scores["source_set_json"].unique().to_list() == ['["alpha", "beta"]']
        assert scores["semantics"].unique().to_list() == ["inst"]
        assert (scores["lead_hours"] > 0).all()
        # test rows strictly after each fold origin
        assert (scores["issue_time"] > scores["fold_origin"]).all()

    def test_empty_matrix(self, config):
        request = BacktestRequest(
            variables=(hourly_variable("temp_c"),), methods=("equal_weight",)
        )
        empty = pl.DataFrame(
            schema={
                "issue_time": pl.Datetime("us", "UTC"),
                "source_kind": pl.String(),
            }
        )
        scores = run_backtest(empty, request, config)
        assert scores.is_empty()
        assert scores.schema == empty_scores().schema

    def test_missing_variable_skipped(self, config, matrix):
        request = BacktestRequest(
            variables=(hourly_variable("wind_speed_ms"),),
            methods=("equal_weight",),
        )
        scores = run_backtest(matrix, request, config)
        assert scores.is_empty()


class TestScoresIO:
    def test_round_trip_and_provenance(self, tmp_path, config, matrix):
        request = BacktestRequest(
            variables=(hourly_variable("temp_c"),), methods=("equal_weight",)
        )
        scores = run_backtest(matrix, request, config)
        path = scores_path(tmp_path, "hourly", "live")
        write_scores(scores, path)
        loaded = load_scores(path)
        assert loaded.height == scores.height

        mixed = pl.concat(
            [scores, scores.with_columns(pl.lit("synthetic").alias("source_kind"))]
        )
        mixed_path = scores_path(tmp_path, "hourly", "mixed")
        write_scores(mixed, mixed_path)
        with pytest.raises(MixedProvenanceError):
            load_scores(mixed_path)
        assert load_scores(mixed_path, allow_mixed=True).height == mixed.height

    def test_score_paths_preserve_window_and_evaluation(self, tmp_path):
        first = scores_path(tmp_path, "hourly", "live", "expanding", "run-a")
        second = scores_path(tmp_path, "hourly", "live", "rolling", "run-b")
        assert first != second
        assert "expanding_run-a" in first.stem


class TestVariableLookup:
    def test_lookup(self):
        specs = variables_from_names(["temp_c", "pop"], HOURLY_VARIABLES)
        assert [s.name for s in specs] == ["temp_c", "pop"]

    def test_unknown(self):
        with pytest.raises(ValueError, match="unknown variables"):
            variables_from_names(["nope"], HOURLY_VARIABLES)
