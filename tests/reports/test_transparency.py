"""Promotion transparency: every winner row explains its gap and gate."""

from datetime import timedelta

import numpy as np
import polars as pl

from grounded_weather_forecast.reports.leaderboard import (
    _mcs_gate,
    blocked_promotions,
    slice_winners,
)
from grounded_weather_forecast.timeutil import utc


def board_row(
    method_id,
    mae,
    n=100,
    coverage=1.0,
    skill=None,
    p=None,
):
    return {
        "product": "hourly",
        "variable": "humidity_pct",
        "lead_bucket": "168-240h",
        "method_id": method_id,
        "n": n,
        "n_total": n,
        "n_valid_times": n,
        "coverage": coverage,
        "mae": mae,
        "rmse": mae,
        "bias": 0.0,
        "skill_vs_best_provider": skill,
        "dm_p_vs_best_provider": p,
        "skill_vs_equal_weight": skill,
        "dm_p_vs_equal_weight": p,
    }


def make_board(rows):
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("skill_vs_best_provider").cast(pl.Float64),
        pl.col("dm_p_vs_best_provider").cast(pl.Float64),
        pl.col("skill_vs_equal_weight").cast(pl.Float64),
        pl.col("dm_p_vs_equal_weight").cast(pl.Float64),
    )


class TestWinnerAnnotations:
    def test_promoted_winner_has_zero_gap_and_no_gate(self):
        board = make_board(
            [
                board_row("persistence", 12.35, skill=0.3, p=0.01),
                board_row("equal_weight", 18.09),
                board_row("best_provider", 19.0),
            ]
        )
        row = slice_winners(board, rule="legacy").row(0, named=True)
        assert row["method_id"] == "persistence"
        assert row["gate"] is None
        assert row["mae_gap"] == 0.0
        assert row["best_method_id"] == "persistence"

    def test_dm_gate_blocks_and_reports_the_gap(self):
        board = make_board(
            [
                board_row("persistence", 12.35, skill=0.3, p=0.4),
                board_row("equal_weight", 18.09),
                board_row("best_provider", 19.0),
            ]
        )
        row = slice_winners(board, rule="legacy").row(0, named=True)
        assert row["method_id"] == "equal_weight"
        assert row["gate"] == "dm_not_significant"
        assert row["best_method_id"] == "persistence"
        assert row["gap_ratio"] == (18.09 / 12.35) - 1.0

    def test_missing_reference_gate(self):
        board = make_board(
            [
                board_row("persistence", 12.35, skill=0.3, p=0.01),
                board_row("equal_weight", 18.09),
            ]
        )
        row = slice_winners(board, rule="legacy").row(0, named=True)
        assert row["method_id"] == "equal_weight"
        assert row["gate"] == "missing_reference"

    def test_eligibility_gate_names_the_unseen_minimum(self):
        board = make_board(
            [
                board_row("persistence", 12.35, coverage=0.5),
                board_row("equal_weight", 18.09),
                board_row("best_provider", 19.0),
            ]
        )
        row = slice_winners(board, rule="legacy").row(0, named=True)
        assert row["method_id"] == "equal_weight"
        assert row["gate"] == "eligibility"
        assert row["best_method_id"] == "persistence"
        assert row["best_mae"] == 12.35

    def test_all_ineligible_yields_typed_empty_frame(self):
        board = make_board([board_row("equal_weight", 18.09, coverage=0.1)])
        winners = slice_winners(board, rule="legacy")
        assert winners.is_empty()
        assert "gate" in winners.columns
        assert "gap_ratio" in winners.columns


class TestBlockedPromotions:
    def blocked_fixture(self):
        board = make_board(
            [
                board_row("persistence", 12.35, skill=0.3, p=0.4),
                board_row("equal_weight", 18.09),
                board_row("best_provider", 19.0),
            ]
        )
        return slice_winners(board, rule="legacy")

    def test_gap_above_threshold_is_flagged(self):
        blocked = blocked_promotions(self.blocked_fixture(), threshold=0.15)
        assert blocked.height == 1
        assert blocked.row(0, named=True)["gate"] == "dm_not_significant"

    def test_gap_below_threshold_is_silent(self):
        blocked = blocked_promotions(self.blocked_fixture(), threshold=2.0)
        assert blocked.is_empty()


def mcs_scores(n_times, candidate_offset=0.0, reference_offset=0.0, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    issue = utc(2026, 7, 31)
    for index in range(n_times):
        valid = utc(2026, 8, 1) + timedelta(hours=index)
        truth = 20.0 + rng.normal(0.0, 0.1)
        noise = rng.normal(0.0, 0.05, 2)
        rows.append(
            ("candidate", issue, valid, truth + candidate_offset + noise[0], truth)
        )
        rows.append(
            ("equal_weight", issue, valid, truth + reference_offset + noise[1], truth)
        )
    return pl.DataFrame(
        rows,
        schema={
            "method_id": pl.String,
            "issue_time": pl.Datetime("us", "UTC"),
            "valid_time": pl.Datetime("us", "UTC"),
            "y_pred": pl.Float64,
            "y_true": pl.Float64,
        },
        orient="row",
    )


class TestMcsGateReasons:
    CANDIDATE = {"method_id": "candidate", "mae": 1.0}
    REFERENCES = ({"method_id": "equal_weight", "mae": 2.0},)
    METHODS = ("candidate", "equal_weight")

    def test_no_matrix(self):
        empty = mcs_scores(0)
        result, gate = _mcs_gate(
            self.CANDIDATE, self.REFERENCES, empty, self.METHODS, alpha=0.1
        )
        assert result["method_id"] == "equal_weight"
        assert gate == "mcs_no_matrix"

    def test_thin_matrix(self):
        result, gate = _mcs_gate(
            self.CANDIDATE, self.REFERENCES, mcs_scores(3), self.METHODS, alpha=0.1
        )
        assert result["method_id"] == "equal_weight"
        assert gate == "mcs_thin_matrix"

    def test_not_separated(self):
        scores = mcs_scores(20, candidate_offset=0.5, reference_offset=0.5)
        result, gate = _mcs_gate(
            self.CANDIDATE, self.REFERENCES, scores, self.METHODS, alpha=0.1
        )
        assert result["method_id"] == "equal_weight"
        assert gate == "mcs_not_separated"

    def test_promoted(self):
        scores = mcs_scores(40, candidate_offset=0.05, reference_offset=5.0)
        result, gate = _mcs_gate(
            self.CANDIDATE, self.REFERENCES, scores, self.METHODS, alpha=0.1
        )
        assert result["method_id"] == "candidate"
        assert gate is None
