import polars as pl
import pytest
from conftest import synthetic_hourly_matrix, write_config

from grounded_weather_forecast.backtest.engine import BacktestRequest, run_backtest
from grounded_weather_forecast.contracts import hourly_variable
from grounded_weather_forecast.reports.leaderboard import (
    aggregate_leaderboard,
    leaderboard,
    slice_winners,
)


@pytest.fixture(scope="module")
def scores(tmp_path_factory):
    config = write_config(
        tmp_path_factory.mktemp("cfg"),
        extra_toml="\n[backtest]\ninitial_train_days = 10\nstep_days = 5\n",
    )
    # alpha carries a large bias, so equal_weight should beat best-of-one=alpha
    matrix = synthetic_hourly_matrix(
        days=25, biases={"alpha": 4.0}, noise_sd=0.3, seed=7
    )
    request = BacktestRequest(
        variables=(hourly_variable("temp_c"),),
        methods=("equal_weight", "best_provider", "climatology", "persistence"),
    )
    return run_backtest(matrix, request, config)


class TestLeaderboard:
    def test_columns_and_views(self, scores):
        board = leaderboard(scores)
        expected = {
            "product",
            "variable",
            "lead_bucket",
            "method_id",
            "n",
            "n_total",
            "coverage",
            "mae",
            "rmse",
            "bias",
            "pct_within",
            "brier",
            "skill_vs_best_provider",
            "dm_p_vs_best_provider",
            "skill_vs_equal_weight",
            "dm_p_vs_equal_weight",
        }
        assert expected <= set(board.columns)
        assert board["n"].min() > 0
        # reference row leaves its own skill columns null
        best_rows = board.filter(pl.col("method_id") == "best_provider")
        assert best_rows["skill_vs_best_provider"].null_count() == best_rows.height

    def test_best_provider_beats_biased_source_and_ew_close(self, scores):
        board = leaderboard(scores)
        agg = aggregate_leaderboard(board)
        mae_by_method = {row["method_id"]: row["mae"] for row in agg.to_dicts()}
        # best_provider learns to pick beta (unbiased); alpha bias would be 4.0
        assert mae_by_method["best_provider"] < 1.0
        # persistence degrades with lead; should be worst overall
        assert mae_by_method["persistence"] > mae_by_method["equal_weight"]

    def test_slice_winners_unique_per_slice(self, scores):
        winners = slice_winners(leaderboard(scores))
        keys = winners.select("product", "variable", "lead_bucket")
        assert keys.unique().height == winners.height

    def test_empty_scores(self):
        from grounded_weather_forecast.backtest.scores import empty_scores

        assert leaderboard(empty_scores()).is_empty()

    def test_all_methods_use_one_common_case_mask(self, scores):
        method = scores["method_id"][0]
        issue = scores.filter(pl.col("method_id") == method)["issue_time"][0]
        sparse = scores.with_columns(
            pl.when((pl.col("method_id") == method) & (pl.col("issue_time") == issue))
            .then(None)
            .otherwise(pl.col("y_pred"))
            .alias("y_pred")
        )
        board = leaderboard(sparse)
        for group in board.partition_by("lead_bucket"):
            assert group["n"].n_unique() == 1
            assert group["coverage"].n_unique() == 1

    def test_aggregate_rmse_combines_squared_error(self):
        board = pl.DataFrame(
            {
                "product": ["hourly", "hourly"],
                "variable": ["temp_c", "temp_c"],
                "method_id": ["equal_weight", "equal_weight"],
                "n": [10, 10],
                "mae": [1.0, 1.0],
                "rmse": [1.0, 3.0],
            }
        )
        result = aggregate_leaderboard(board).row(0, named=True)
        assert result["rmse"] == pytest.approx(5.0**0.5)
