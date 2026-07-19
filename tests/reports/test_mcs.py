from datetime import timedelta

import numpy as np
import polars as pl

from grounded_weather_forecast.reports.leaderboard import leaderboard, slice_winners
from grounded_weather_forecast.reports.mcs import (
    collapsed_loss_matrix,
    model_confidence_set,
)
from grounded_weather_forecast.serve.selection import Selection, apply_live_gate
from grounded_weather_forecast.timeutil import utc


def losses_matrix(n=120, n_methods=10, best=None, seed=5):
    rng = np.random.default_rng(seed)
    losses = 1.0 + rng.normal(0.0, 0.3, size=(n, n_methods))
    if best is not None:
        losses[:, best] -= 0.5  # genuinely dominant
    return np.abs(losses)


class TestModelConfidenceSet:
    def test_equal_methods_mostly_survive(self):
        """Winner's-curse guard: near-equal methods stay indistinguishable."""
        ids = tuple(f"m{i}" for i in range(10))
        result = model_confidence_set(losses_matrix(), ids, alpha=0.1)
        assert len(result.survivors) >= 8

    def test_dominant_method_prunes_the_field(self):
        ids = tuple(f"m{i}" for i in range(10))
        result = model_confidence_set(losses_matrix(best=3), ids, alpha=0.1)
        assert "m3" in result.survivors
        assert len(result.survivors) < 5

    def test_thin_data_eliminates_nothing(self):
        ids = ("a", "b", "c")
        result = model_confidence_set(losses_matrix(n=5)[:, :3], ids)
        assert result.survivors == ids

    def test_deterministic_loss_difference_eliminates_worse_method(self):
        losses = np.column_stack((np.zeros(40), np.full(40, 2.0)))
        result = model_confidence_set(losses, ("challenger", "equal_weight"))
        assert result.survivors == ("challenger",)


def scores_frame(gap: float, n_times=80, seed=9):
    """Two methods with independent errors; ``gap`` shrinks the challenger's.

    Independent noise streams matter: a deterministic per-case edge would be
    genuinely significant at any size, which is not a "near tie".
    """
    start = utc(2026, 3, 1)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_times):
        valid = start + timedelta(hours=i)
        issue = valid - timedelta(hours=30)
        truth = 10.0
        challenger_noise = rng.normal(0.0, 1.0 - gap)
        reference_noise = rng.normal(0.0, 1.0)
        rows.append(("challenger", issue, valid, truth + challenger_noise, truth))
        rows.append(("equal_weight", issue, valid, truth + reference_noise, truth))
        rows.append(("best_provider", issue, valid, truth + reference_noise, truth))
    return pl.DataFrame(
        {
            "method_id": [r[0] for r in rows],
            "issue_time": [r[1] for r in rows],
            "valid_time": [r[2] for r in rows],
            "y_pred": [r[3] for r in rows],
            "y_true": [r[4] for r in rows],
            "product": ["hourly"] * len(rows),
            "variable": ["temp_c"] * len(rows),
            "lead_bucket": ["24-48h"] * len(rows),
            "lead_hours": [30.0] * len(rows),
        }
    )


class TestMcsPromotionGate:
    def test_clear_winner_promotes(self):
        scores = scores_frame(gap=0.5)
        board = leaderboard(scores)
        winners = slice_winners(board, scores=scores, rule="mcs", alpha=0.1)
        assert winners["method_id"][0] == "challenger"

    def test_near_tie_keeps_the_reference(self):
        # seed 10: the challenger leads on sample MAE (the winner's-curse
        # configuration) yet the MCS keeps both, so the reference serves
        scores = scores_frame(gap=0.0, seed=10)
        board = leaderboard(scores)
        ranked = board.sort("mae")
        assert ranked["method_id"][0] == "challenger"  # lucky on the sample
        winners = slice_winners(board, scores=scores, rule="mcs", alpha=0.1)
        assert winners["method_id"][0] == "equal_weight"

    def test_collapsed_matrix_shape(self):
        scores = scores_frame(gap=0.1, n_times=20)
        built = collapsed_loss_matrix(scores)
        assert built is not None
        matrix, methods = built
        assert matrix.shape == (20, 3)
        assert methods == ("best_provider", "challenger", "equal_weight")

    def test_ineligible_sparse_method_does_not_shrink_common_cases(self):
        scores = scores_frame(gap=0.1, n_times=20)
        sparse = (
            scores.filter(pl.col("method_id") == "challenger")
            .head(1)
            .with_columns(pl.lit("sparse").alias("method_id"))
        )
        built = collapsed_loss_matrix(
            pl.concat([scores, sparse]),
            method_ids=("best_provider", "challenger", "equal_weight"),
        )
        assert built is not None
        matrix, methods = built
        assert matrix.shape == (20, 3)
        assert "sparse" not in methods


class TestLiveGate:
    def make_selection(self, mae=1.0):
        return {
            ("hourly", "temp_c", "24-48h"): Selection(
                method_id="gbm",
                reason="won",
                n=100,
                mae=mae,
                evaluation_id="eval-gbm",
                dataset_fingerprint="dataset-a",
                release_id="release-a",
            )
        }

    def fallback(self):
        return {
            ("hourly", "temp_c", "24-48h"): Selection(
                method_id="equal_weight",
                reason="reference",
                n=88,
                mae=1.2,
                evaluation_id="eval-reference",
                dataset_fingerprint="dataset-a",
                release_id="release-a",
            )
        }

    def live_frame(
        self,
        live_mae,
        n=50,
        *,
        bucket="24-48h",
        dataset="dataset-a",
        release="release-a",
    ):
        return pl.DataFrame(
            {
                "product": ["hourly"],
                "variable": ["temp_c"],
                "lead_bucket": [bucket],
                "method_id": ["gbm"],
                "dataset_fingerprint": [dataset],
                "release_id": [release],
                "n": [n],
                "live_mae": [live_mae],
                "live_rmse": [live_mae],
                "live_bias": [0.0],
            }
        )

    def test_materially_worse_live_mae_demotes(self):
        gated = apply_live_gate(
            self.make_selection(mae=1.0),
            self.live_frame(live_mae=2.0),
            factor=1.5,
            min_n=24,
            fallbacks=self.fallback(),
        )
        selected = gated[("hourly", "temp_c", "24-48h")]
        assert selected.method_id == "equal_weight"
        assert "demoted gbm" in selected.reason
        assert selected.n == 88
        assert selected.mae == 1.2
        assert selected.evaluation_id == "eval-reference"

    def test_delivering_method_survives(self):
        gated = apply_live_gate(
            self.make_selection(mae=1.0),
            self.live_frame(live_mae=1.1),
            factor=1.5,
            min_n=24,
            fallbacks=self.fallback(),
        )
        assert gated[("hourly", "temp_c", "24-48h")].method_id == "gbm"

    def test_thin_live_evidence_is_ignored(self):
        gated = apply_live_gate(
            self.make_selection(mae=1.0),
            self.live_frame(live_mae=5.0, n=5),
            factor=1.5,
            min_n=24,
            fallbacks=self.fallback(),
        )
        assert gated[("hourly", "temp_c", "24-48h")].method_id == "gbm"

    def test_empty_live_frame_is_noop(self):
        selections = self.make_selection()
        assert (
            apply_live_gate(selections, pl.DataFrame(), factor=1.5, min_n=24)
            == selections
        )

    def test_another_slice_cannot_demote(self):
        """Evidence about a different slice says nothing about this one."""
        gated = apply_live_gate(
            self.make_selection(),
            self.live_frame(5.0, bucket="1-3h"),
            factor=1.5,
            min_n=24,
            fallbacks=self.fallback(),
        )
        assert gated[("hourly", "temp_c", "24-48h")].method_id == "gbm"

    def test_an_ineligible_release_cannot_demote(self):
        """A release written under an incompatible config is not evidence.

        This is the safety the old dataset/release equality check was reaching
        for, expressed where it belongs: eligibility is decided from the
        ledger, not from an identity that rotates with data volume.
        """
        gated = apply_live_gate(
            self.make_selection(),
            self.live_frame(5.0, release="release-b"),
            factor=1.5,
            min_n=24,
            fallbacks=self.fallback(),
            eligible_releases=frozenset({"release-a"}),
        )
        assert gated[("hourly", "temp_c", "24-48h")].method_id == "gbm"

    def test_evidence_survives_a_nightly_dataset_rotation(self):
        """Regression: leads >= 24h could never accumulate enough evidence.

        `maintain` rebuilds the dataset nightly, rotating both the dataset
        fingerprint and the release id, so a 24-48h forecast's truth always
        landed after its own cohort had been retired. Neither cohort here
        reaches min_n alone; only pooling crosses the threshold.
        """
        live = pl.concat(
            [
                self.live_frame(
                    2.0, n=12, dataset="dataset-yesterday", release="release-yesterday"
                ),
                self.live_frame(2.0, n=12, dataset="dataset-a", release="release-a"),
            ]
        )
        gated = apply_live_gate(
            self.make_selection(mae=1.0),
            live,
            factor=1.5,
            min_n=24,
            fallbacks=self.fallback(),
            eligible_releases=frozenset({"release-yesterday", "release-a"}),
        )
        selected = gated[("hourly", "temp_c", "24-48h")]
        assert selected.method_id == "equal_weight"
        assert "demoted gbm" in selected.reason

    def test_pooled_live_mae_is_n_weighted(self):
        """Pooling must weight cohorts by their row counts, not average them."""
        live = pl.concat(
            [
                self.live_frame(1.1, n=100, release="release-a"),
                self.live_frame(5.0, n=100, release="release-b"),
            ]
        )
        gated = apply_live_gate(
            self.make_selection(mae=1.0),
            live,
            factor=1.5,
            min_n=24,
            fallbacks=self.fallback(),
            eligible_releases=frozenset({"release-a", "release-b"}),
        )
        assert "live MAE 3.050" in gated[("hourly", "temp_c", "24-48h")].reason

    def test_legacy_live_rows_cannot_drive_automatic_demotion(self):
        legacy = self.live_frame(5.0).drop("dataset_fingerprint", "release_id")
        assert (
            apply_live_gate(self.make_selection(), legacy, factor=1.5, min_n=24)
            == self.make_selection()
        )

    def test_a_rotated_dataset_fingerprint_no_longer_blocks_the_gate(self):
        """The dataset fingerprint hashes row counts, so it is not identity."""
        gated = apply_live_gate(
            self.make_selection(mae=1.0),
            self.live_frame(5.0, dataset="dataset-b"),
            factor=1.5,
            min_n=24,
            fallbacks=self.fallback(),
        )
        assert gated[("hourly", "temp_c", "24-48h")].method_id == "equal_weight"


class TestPValuesMonotone:
    def test_elimination_p_values_recorded(self):
        ids = tuple(f"m{i}" for i in range(6))
        result = model_confidence_set(losses_matrix(n_methods=6, best=0), ids)
        eliminated = set(ids) - set(result.survivors)
        assert set(result.p_values) == eliminated
        assert all(0.0 <= p < 0.1 for p in result.p_values.values())
