import json
from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from conftest import synthetic_hourly_matrix, write_config

from grounded_weather_forecast.backtest.engine import BacktestRequest, run_backtest
from grounded_weather_forecast.backtest.scores import empty_scores
from grounded_weather_forecast.contracts import hourly_variable
from grounded_weather_forecast.metrics.dm import diebold_mariano
from grounded_weather_forecast.reports.leaderboard import (
    _dm_columns,
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

    def test_concatenated_truth_targets_stay_separate_slices(
        self, scores: pl.DataFrame
    ) -> None:
        single = leaderboard(scores)
        mixed = leaderboard(
            pl.concat([scores, scores.with_columns(pl.lit("mean").alias("semantics"))])
        )

        assert set(mixed["truth_semantics"].to_list()) == {"inst", "mean"}
        instantaneous = mixed.filter(pl.col("truth_semantics") == "inst")
        assert instantaneous.height == single.height
        winners = slice_winners(mixed)
        winner_columns = ["product", "variable", "lead_bucket", "method_id"]
        expected = (
            slice_winners(single).select(winner_columns).sort(winner_columns).to_dicts()
        )
        assert expected
        for semantics in ("inst", "mean"):
            actual = (
                winners.filter(pl.col("truth_semantics") == semantics)
                .select(winner_columns)
                .sort(winner_columns)
                .to_dicts()
            )
            assert len(actual) == len(expected)
            assert actual == expected

    def test_null_semantics_merge_into_instantaneous_slice(
        self, scores: pl.DataFrame
    ) -> None:
        nullable = scores.with_columns(
            pl.Series(
                "semantics",
                [None if index % 2 else "inst" for index in range(scores.height)],
                dtype=pl.String,
            )
        )

        assert leaderboard(nullable).equals(leaderboard(scores))

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

    def test_non_finite_truth_rows_are_not_score_cases(self):
        """Historical malformed frames cannot poison metrics or coverage."""
        start = utc(2026, 3, 1)
        malformed = pl.DataFrame(
            {
                "product": ["hourly"] * 5,
                "variable": ["temp_c"] * 5,
                "lead_bucket": ["1-3h"] * 5,
                "method_id": ["candidate"] * 5,
                "issue_time": [start + timedelta(hours=i) for i in range(5)],
                "valid_time": [start + timedelta(hours=i + 1) for i in range(5)],
                "lead_hours": [1.0] * 5,
                "y_pred": [1.0, 100.0, 100.0, 100.0, 100.0],
                "y_true": [0.0, None, float("nan"), float("inf"), float("-inf")],
            }
        )

        row = leaderboard(malformed).row(0, named=True)

        assert row["n"] == 1
        assert row["n_total"] == 1
        assert row["n_valid_times"] == 1
        assert row["coverage"] == 1.0
        assert row["mae"] == 1.0
        assert leaderboard(malformed.slice(1)).is_empty()

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

    def test_quantile_crps_is_ensemble_crps_and_pit_needs_50_rows(self):
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
        # Energy-form ensemble CRPS with members (-1, 0, 2) against y = 0.4:
        #   mean|X - y| = (1.4 + 0.4 + 1.6) / 3 = 3.4 / 3
        #   mean|X - X'| over all 9 pairs = 2 * (1 + 3 + 2) / 9 = 4 / 3
        #   CRPS = 3.4 / 3 - (4 / 3) / 2 = 1.4 / 3
        assert thin["crps"] == pytest.approx(1.4 / 3)
        assert thin["pit_chi2_p"] is None
        mature = leaderboard(probabilistic_scores(50)).row(0, named=True)
        assert mature["pit_chi2_p"] is not None

    def test_rows_without_usable_quantiles_yield_null_crps(self):
        """Null grids and null grid members drop out; too few survivors
        leave the CRPS column null instead of crashing the estimator."""
        start = utc(2026, 3, 1)
        n = 10
        grids = [None, None, json.dumps([None, 0.0, 2.0])] + [
            json.dumps([-1.0, 0.0, 2.0])
        ] * (n - 3)
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
                "y_true": [0.4] * n,
                "quantile_levels_json": [json.dumps((0.1, 0.5, 0.9))] * n,
                "quantiles_json": grids,
            }
        )

        row = leaderboard(scores).row(0, named=True)

        assert row["crps"] is None
        assert row["pinball"] is None

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


class TestDmComparisonProjection:
    @pytest.mark.parametrize(
        ("product", "lead_lo", "horizon"),
        [("hourly", 2.0, 3), ("daily", 48.0, 3), ("minutely", 0.05, 4)],
    )
    def test_wide_partial_cases_preserve_collapsed_losses(
        self, product, lead_lo, horizon
    ):
        start = utc(2026, 3, 1)
        rows = []
        method_losses, reference_losses = [], []
        for index in range(12):
            paired_method, paired_reference = [], []
            for replicate in range(2):
                valid = start + timedelta(hours=index)
                issue = valid - timedelta(hours=lead_lo + replicate / 2)
                truth = 20.0 + index / 5
                method_loss = 0.25 + index / 10 + replicate / 5
                reference_loss = 1.0 + index * 0.03 + replicate / 10
                method_null = (index, replicate) == (3, 0)
                reference_null = (index, replicate) == (5, 1)
                for method, available, prediction, method_truth in (
                    (
                        "candidate",
                        index > 0,
                        None if method_null else truth + method_loss,
                        truth,
                    ),
                    (
                        "reference",
                        index < 11,
                        None if reference_null else truth + reference_loss,
                        999.0,  # Pairwise losses must use the candidate's truth.
                    ),
                ):
                    if available:
                        rows.append(
                            {
                                "method_id": method,
                                "issue_time": issue,
                                "valid_time": valid,
                                "y_pred": prediction,
                                "y_true": method_truth,
                                "feature_set_json": json.dumps(["x" * 8192]),
                                "quantiles_json": json.dumps([0.0] * 128),
                            }
                        )
                if 0 < index < 11 and not method_null and not reference_null:
                    paired_method.append(method_loss)
                    paired_reference.append(reference_loss)
            if paired_method:
                method_losses.append(np.mean(paired_method))
                reference_losses.append(np.mean(paired_reference))
        scores = pl.DataFrame(rows)
        method = scores.filter(pl.col("method_id") == "candidate")
        skill, p_value = _dm_columns(scores, method, "reference", lead_lo, product)
        method_loss = np.asarray(method_losses)
        reference_loss = np.asarray(reference_losses)

        assert skill == pytest.approx(
            1.0 - method_loss.mean() / reference_loss.mean(), rel=1e-12, abs=1e-12
        )
        assert p_value == pytest.approx(
            diebold_mariano(method_loss, reference_loss, horizon).p_value,
            rel=1e-12,
            abs=1e-12,
        )

    @pytest.mark.parametrize("case", ["missing", "no_overlap", "nulls", "zero_loss"])
    def test_reference_edge_cases(self, case):
        start = utc(2026, 3, 1)
        candidate = pl.DataFrame(
            {
                "method_id": ["candidate"] * 12,
                "issue_time": [start + timedelta(hours=i) for i in range(12)],
                "valid_time": [start + timedelta(hours=i + 1) for i in range(12)],
                "y_pred": [0.5 + i / 10 for i in range(12)],
                "y_true": [0.0] * 12,
                "feature_set_json": ["x" * 8192] * 12,
            }
        )
        reference = candidate.with_columns(
            pl.lit("other" if case == "missing" else "reference").alias("method_id"),
            pl.lit(None if case == "nulls" else 0.0, dtype=pl.Float64).alias("y_pred"),
        )
        if case == "no_overlap":
            reference = reference.with_columns(
                pl.col("issue_time") + pl.duration(days=1)
            )
        result = _dm_columns(
            pl.concat([candidate, reference]), candidate, "reference", 1.0, "hourly"
        )

        if case == "zero_loss":
            assert result[0] is None
            assert result[1] == pytest.approx(
                diebold_mariano(
                    candidate["y_pred"].to_numpy(), np.zeros(12), 2
                ).p_value,
                rel=1e-12,
                abs=1e-12,
            )
        else:
            assert result == (None, None)
