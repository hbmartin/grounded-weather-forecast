"""Winner's-curse corrections: bias sign, hybrid guards, gate safety."""

from datetime import timedelta

import numpy as np
import polars as pl

from grounded_weather_forecast.config import PromotionConfig
from grounded_weather_forecast.reports.leaderboard import (
    blocked_promotions,
    leaderboard,
    slice_winners,
)
from grounded_weather_forecast.reports.winner_curse import (
    _slice_correction,
    winner_curse_adjusted,
    winner_curse_verdicts,
)
from grounded_weather_forecast.timeutil import utc


def tied_losses(n=200, k=10, seed=42):
    rng = np.random.default_rng(seed)
    losses = np.abs(rng.normal(0.0, 1.25, (n, k)))
    ids = tuple(f"m{i}" for i in range(k))
    return losses, ids


class TestSliceCorrection:
    def test_near_ties_produce_negative_bias_and_flag(self):
        losses, ids = tied_losses()
        column_means = losses.mean(axis=0)
        served = ids[int(np.argmin(column_means))]
        result = _slice_correction(
            losses, ids, served, n_bootstrap=500, block_length=None, alpha=0.1
        )
        # ten methods with identical true MAE: the reported min is biased low
        assert result["winner_bias"] < -0.01
        # the conditional estimate pulls the winner back up past its raw mean
        assert result["mae_hybrid_shift"] > 0.0
        assert result["mae_hybrid_upper_shift"] > result["mae_hybrid_shift"]
        assert result["near_tie_flag"] is True

    def test_separated_winner_needs_no_correction(self):
        losses, ids = tied_losses()
        rng = np.random.default_rng(7)
        losses = losses.copy()
        losses[:, 3] = np.abs(rng.normal(0.0, 0.5, losses.shape[0]))
        result = _slice_correction(
            losses, ids, "m3", n_bootstrap=500, block_length=None, alpha=0.1
        )
        assert abs(result["winner_bias"]) < 5e-3
        assert abs(result["mae_hybrid_shift"]) < 5e-3
        assert result["near_tie_flag"] is False

    def test_served_method_disagreeing_with_matrix_argmin_skips_hybrid(self):
        losses, ids = tied_losses()
        column_means = losses.mean(axis=0)
        not_winner = ids[int(np.argmax(column_means))]
        result = _slice_correction(
            losses, ids, not_winner, n_bootstrap=200, block_length=None, alpha=0.1
        )
        # the unconditional bias still applies; the conditional math would
        # condition on the wrong identity event
        assert result["winner_bias"] is not None
        assert result["mae_hybrid_shift"] is None
        assert result["near_tie_flag"] is True

    def test_thin_matrix_yields_nulls(self):
        losses, ids = tied_losses(n=4)
        result = _slice_correction(
            losses, ids, "m0", n_bootstrap=100, block_length=None, alpha=0.1
        )
        assert result["winner_bias"] is None

    def test_deterministic_across_calls(self):
        losses, ids = tied_losses()
        served = ids[int(np.argmin(losses.mean(axis=0)))]
        first = _slice_correction(
            losses, ids, served, n_bootstrap=300, block_length=None, alpha=0.1
        )
        second = _slice_correction(
            losses, ids, served, n_bootstrap=300, block_length=None, alpha=0.1
        )
        assert first == second


def curse_scores(n_times=120, k=6, spread=0.0, seed=3):
    """k methods over shared valid times; spread=0 means a pure near-tie."""
    rng = np.random.default_rng(seed)
    rows = []
    issue = utc(2026, 7, 1)
    for index in range(n_times):
        valid = utc(2026, 7, 2) + timedelta(hours=index)
        truth = 20.0 + rng.normal(0.0, 0.1)
        rows.extend(
            {
                "method_id": f"m{m}",
                "product": "hourly",
                "variable": "temp_c",
                "semantics": "inst",
                "lead_bucket": "24-48h",
                "lead_hours": 30.0,
                "issue_time": issue,
                "valid_time": valid,
                "y_pred": truth + m * spread + rng.normal(0.0, 1.0),
                "y_true": truth,
                "source_kind": "live",
                "evaluation_id": "eval1",
                "n": 1,
            }
            for m in range(k)
        )
    return pl.DataFrame(rows)


def winners_row(gate=None, mae=0.8, method_id="m0"):
    return {
        "product": "hourly",
        "variable": "temp_c",
        "truth_semantics": "inst",
        "lead_bucket": "24-48h",
        "method_id": method_id,
        "n": 100,
        "mae": mae,
        "best_method_id": method_id,
        "best_mae": mae,
        "best_n": 100,
        "mae_gap": 0.0,
        "gap_ratio": 0.0,
        "gate": gate,
    }


class TestWinnerCurseAdjusted:
    def adjusted(self, gate=None, scores=None, method_id=None):
        scores = curse_scores() if scores is None else scores
        if method_id is None:
            # the common-case argmin, so the conditional arm engages
            collapsed = (
                scores.group_by("method_id")
                .agg((pl.col("y_pred") - pl.col("y_true")).abs().mean().alias("mae"))
                .sort("mae")
            )
            method_id = str(collapsed["method_id"][0])
        winners = pl.DataFrame([winners_row(gate=gate, method_id=method_id)])
        return winner_curse_adjusted(
            winners, scores, promotion=PromotionConfig(mcs_bootstrap=300)
        )

    def test_argmin_winner_gains_all_columns(self):
        adjusted = self.adjusted()
        row = adjusted.row(0, named=True)
        assert row["winner_bias"] < 0.0
        assert row["mae_debiased"] > row["mae"]
        assert row["mae_hybrid"] is not None
        assert row["mae_hybrid_upper"] >= row["mae_hybrid"]

    def test_gated_reference_rows_stay_null(self):
        adjusted = self.adjusted(gate="mcs_not_separated")
        row = adjusted.row(0, named=True)
        assert row["winner_bias"] is None
        assert row["mae_debiased"] is None

    def test_eligibility_gate_is_still_an_argmin_selection(self):
        adjusted = self.adjusted(gate="eligibility")
        assert adjusted.row(0, named=True)["winner_bias"] is not None

    def test_missing_scores_adds_typed_null_columns(self):
        winners = pl.DataFrame([winners_row()])
        adjusted = winner_curse_adjusted(winners, None, promotion=None)
        assert adjusted["winner_bias"].dtype == pl.Float64
        assert adjusted["winner_bias"][0] is None

    def test_empty_winners_keep_typed_columns(self):
        empty = pl.DataFrame([winners_row()]).head(0)
        adjusted = winner_curse_adjusted(empty, curse_scores(), promotion=None)
        assert adjusted.is_empty()
        assert "mae_hybrid" in adjusted.columns

    def test_decision_surfaces_are_byte_identical(self):
        scores = curse_scores(spread=0.4)
        board = leaderboard(scores, references=())
        winners = slice_winners(board)
        adjusted = winner_curse_adjusted(winners, scores, promotion=None)
        # every pre-existing column unchanged; blocked_promotions agrees
        for column in winners.columns:
            assert adjusted[column].to_list() == winners[column].to_list()
        assert blocked_promotions(adjusted, 0.15).height == (
            blocked_promotions(winners, 0.15).height
        )


class TestWinnerCurseVerdicts:
    def test_scalars_summarize_the_columns(self):
        winners = pl.DataFrame(
            {
                "winner_bias": [-0.05, -0.01, None],
                "near_tie_flag": [True, False, None],
            }
        )
        verdicts = winner_curse_verdicts(winners)
        assert abs(verdicts["winner_bias_mean"] - (-0.03)) < 1e-12
        assert verdicts["winner_bias_slices"] == 2.0
        assert verdicts["near_tie_slices"] == 1.0

    def test_empty_or_uncorrected_yields_nothing(self):
        assert winner_curse_verdicts(pl.DataFrame()) == {}
        assert winner_curse_verdicts(pl.DataFrame({"mae": [1.0]})) == {}
