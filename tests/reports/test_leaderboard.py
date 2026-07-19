import json
from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from conftest import synthetic_hourly_matrix, write_config

from grounded_weather_forecast.backtest.engine import BacktestRequest, run_backtest
from grounded_weather_forecast.backtest.scores import empty_scores
from grounded_weather_forecast.contracts import hourly_variable
from grounded_weather_forecast.metrics.probabilistic import crps_from_quantiles
from grounded_weather_forecast.reports.leaderboard import (
    aggregate_leaderboard,
    leaderboard,
    slice_winners,
)
from grounded_weather_forecast.timeutil import utc


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
        assert leaderboard(empty_scores()).is_empty()

    def test_methods_are_scored_on_their_own_cases(self, scores):
        """A sparse method loses its own missing cases; the others keep theirs.

        The old behavior shrank every method to the all-methods intersection,
        which punished complete methods for one sparse method's holes.
        """
        method = scores["method_id"][0]
        issue = scores.filter(pl.col("method_id") == method)["issue_time"][0]
        sparse = scores.with_columns(
            pl.when((pl.col("method_id") == method) & (pl.col("issue_time") == issue))
            .then(None)
            .otherwise(pl.col("y_pred"))
            .alias("y_pred")
        )
        board = leaderboard(sparse)
        full_board = leaderboard(scores)
        sparse_rows = board.filter(pl.col("method_id") == method)
        other_rows = board.filter(pl.col("method_id") != method)
        full_other = full_board.filter(pl.col("method_id") != method)
        # the sparse method's own n shrank somewhere...
        assert (
            sparse_rows["n"].sum()
            < full_board.filter(pl.col("method_id") == method)["n"].sum()
        )
        # ...while every other method kept its full case set
        assert other_rows["n"].sum() == full_other["n"].sum()
        assert "n_valid_times" in board.columns

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

    def test_ineligible_slice_has_no_winner(self):
        board = pl.DataFrame(
            {
                "product": ["hourly"],
                "variable": ["temp_c"],
                "lead_bucket": ["1-3h"],
                "method_id": ["challenger"],
                "n": [1],
                "n_valid_times": [1],
                "coverage": [0.1],
                "mae": [0.1],
            }
        )
        assert slice_winners(board).is_empty()

    def test_quantile_crps_uses_probability_levels_and_pit_needs_50_rows(self):
        start = utc(2026, 3, 1)
        levels = (0.1, 0.5, 0.9)

        def probabilistic_scores(n):
            return pl.DataFrame(
                {
                    "product": ["hourly"] * n,
                    "variable": ["temp_c"] * n,
                    "lead_bucket": ["1-3h"] * n,
                    "method_id": ["distribution"] * n,
                    "issue_time": [start + timedelta(hours=i) for i in range(n)],
                    "valid_time": [start + timedelta(hours=i + 1) for i in range(n)],
                    "lead_hours": [1.0] * n,
                    "y_pred": [0.0] * n,
                    "y_true": [0.4] * n,
                    "quantile_levels_json": [json.dumps(levels)] * n,
                    "quantiles_json": [json.dumps([-1.0, 0.0, 2.0])] * n,
                }
            )

        thin = leaderboard(probabilistic_scores(8)).row(0, named=True)
        grids = np.tile(np.asarray([-1.0, 0.0, 2.0]), (8, 1))
        expected = crps_from_quantiles(np.full(8, 0.4), grids, levels)
        assert thin["crps"] == pytest.approx(expected)
        assert thin["pit_chi2_p"] is None
        mature = leaderboard(probabilistic_scores(50)).row(0, named=True)
        assert mature["pit_chi2_p"] is not None

    def test_nominal_coverage_requires_exact_probability_levels(self):
        start = utc(2026, 3, 1)
        n = 8
        scores = pl.DataFrame(
            {
                "product": ["hourly"] * n,
                "variable": ["temp_c"] * n,
                "lead_bucket": ["1-3h"] * n,
                "method_id": ["distribution"] * n,
                "issue_time": [start + timedelta(hours=i) for i in range(n)],
                "valid_time": [start + timedelta(hours=i + 1) for i in range(n)],
                "lead_hours": [1.0] * n,
                "y_pred": [0.0] * n,
                "y_true": [0.0] * n,
                "quantile_levels_json": [json.dumps((0.2, 0.5, 0.8))] * n,
                "quantiles_json": [json.dumps((-1.0, 0.0, 1.0))] * n,
            }
        )

        row = leaderboard(scores).row(0, named=True)

        assert row["crps"] is not None
        assert row["coverage80"] is None
        assert row["coverage90"] is None
        assert row["sharpness"] is None


class TestDmCollapsesPseudoReplication:
    """Dozens of snapshots forecasting the same valid hour must not
    manufacture DM significance: losses collapse to one per valid_time."""

    def make_scores(self, replicates):
        rng = np.random.default_rng(0)
        start = utc(2026, 3, 1)
        rows = []
        for i in range(12):
            valid = start + timedelta(hours=i)
            loss_reference = 1.0 + abs(float(rng.normal(0.0, 0.3)))
            loss_challenger = loss_reference + float(rng.normal(0.0, 0.5))
            for r in range(replicates):
                lead = 24.0 + 0.25 * r
                issue = valid - timedelta(hours=lead)
                for method, loss in (
                    ("challenger", abs(loss_challenger)),
                    ("equal_weight", loss_reference),
                ):
                    rows.append(
                        {
                            "product": "hourly",
                            "variable": "temp_c",
                            "lead_bucket": "24-48h",
                            "method_id": method,
                            "issue_time": issue,
                            "valid_time": valid,
                            "lead_hours": lead,
                            "y_pred": loss,
                            "y_true": 0.0,
                        }
                    )
        return pl.DataFrame(rows)

    def test_replication_leaves_p_value_unchanged(self):
        single = leaderboard(self.make_scores(1))
        replicated = leaderboard(self.make_scores(30))
        column = "dm_p_vs_equal_weight"
        p_single = single.filter(pl.col("method_id") == "challenger")[column][0]
        p_replicated = replicated.filter(pl.col("method_id") == "challenger")[column][0]
        assert p_single is not None
        assert p_replicated == pytest.approx(p_single)
        # n still reports every scored row; only the DM test collapses
        n_replicated = replicated.filter(pl.col("method_id") == "challenger")["n"][0]
        assert n_replicated == 360
