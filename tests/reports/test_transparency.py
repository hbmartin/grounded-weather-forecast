"""Promotion transparency: every winner row explains its gap and gate."""

from datetime import timedelta

import numpy as np
import polars as pl

from grounded_weather_forecast.config import PromotionConfig
from grounded_weather_forecast.reports.eprocess import (
    EProcessEntry,
    EProcessStore,
    pair_key,
)
from grounded_weather_forecast.reports.leaderboard import (
    _mcs_gate,
    bh_adjusted,
    blocked_promotions,
    ebh_adjusted,
    gate_references,
    slice_winners,
    union_references,
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
        "skill_vs_damped_grounded_equal_weight": skill,
        "dm_p_vs_damped_grounded_equal_weight": p,
    }


_SKILL_COLUMNS = (
    "skill_vs_best_provider",
    "dm_p_vs_best_provider",
    "skill_vs_equal_weight",
    "dm_p_vs_equal_weight",
    "skill_vs_damped_grounded_equal_weight",
    "dm_p_vs_damped_grounded_equal_weight",
)


def make_board(rows):
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        *(pl.col(column).cast(pl.Float64) for column in _SKILL_COLUMNS)
    )


class TestWinnerAnnotations:
    def test_promoted_winner_has_zero_gap_and_no_gate(self):
        board = make_board(
            [
                board_row("persistence", 12.35, skill=0.3, p=0.01),
                board_row("equal_weight", 18.09),
                board_row("best_provider", 19.0),
                board_row("damped_grounded_equal_weight", 17.0),
            ]
        )
        row = slice_winners(board, rule="legacy").row(0, named=True)
        assert row["method_id"] == "persistence"
        assert row["gate"] is None
        assert row["mae_gap"] == 0.0
        assert row["best_method_id"] == "persistence"

    def test_dm_gate_blocks_and_falls_back_to_the_best_reference(self):
        board = make_board(
            [
                board_row("persistence", 12.35, skill=0.3, p=0.4),
                board_row("equal_weight", 18.09),
                board_row("best_provider", 19.0),
                board_row("damped_grounded_equal_weight", 17.0),
            ]
        )
        row = slice_winners(board, rule="legacy").row(0, named=True)
        # The fallback is the best reference by MAE, not equal_weight by name.
        assert row["method_id"] == "damped_grounded_equal_weight"
        assert row["gate"] == "dm_not_significant"
        assert row["best_method_id"] == "persistence"
        assert row["gap_ratio"] == (17.0 / 12.35) - 1.0

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
                board_row("damped_grounded_equal_weight", 17.0),
            ]
        )
        row = slice_winners(board, rule="legacy").row(0, named=True)
        # damped is the eligible minimum AND a reference: it serves outright.
        assert row["method_id"] == "damped_grounded_equal_weight"
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
                board_row("damped_grounded_equal_weight", 17.0),
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
    references = (
        "equal_weight",
        "best_provider",
        "damped_grounded_equal_weight",
    )
    rows = []
    issue = utc(2026, 7, 31)
    for index in range(n_times):
        valid = utc(2026, 8, 1) + timedelta(hours=index)
        truth = 20.0 + rng.normal(0.0, 0.1)
        rows.append(
            (
                "candidate",
                issue,
                valid,
                truth + candidate_offset + rng.normal(0.0, 0.05),
                truth,
            )
        )
        rows.extend(
            (
                reference,
                issue,
                valid,
                truth + reference_offset + rng.normal(0.0, 0.05),
                truth,
            )
            for reference in references
        )
    frame = pl.DataFrame(
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
    return frame.with_columns(
        pl.lit("hourly").alias("product"),
        pl.lit("humidity_pct").alias("variable"),
        pl.lit("168-240h").alias("lead_bucket"),
        pl.lit(200.0).alias("lead_hours"),
    )


class TestDecisionScopedMatrix:
    def test_sparse_bystander_cannot_thin_the_gate(self):
        scores = mcs_scores(40, candidate_offset=0.05, reference_offset=5.0)
        sparse = (
            scores.filter(pl.col("method_id") == "candidate")
            .head(3)
            .with_columns(pl.lit("sparse").alias("method_id"))
        )
        board = make_board(
            [
                board_row("candidate", 1.0),
                board_row("sparse", 1.5),
                board_row("equal_weight", 5.0),
                board_row("best_provider", 5.1),
                board_row("damped_grounded_equal_weight", 5.2),
            ]
        )
        winners = slice_winners(
            board, scores=pl.concat([scores, sparse]), rule="mcs", alpha=0.1
        )
        row = winners.row(0, named=True)
        # Before the decision-scoped matrix, the sparse bystander's 3 common
        # cases collapsed the all-method intersection below the 8-row floor
        # and the gate returned "mcs_thin_matrix" instead of promoting.
        assert row["method_id"] == "candidate"
        assert row["gate"] is None


class TestPerVariableReferences:
    OVERRIDES = PromotionConfig(
        references={
            "humidity_pct": ("grounded_equal_weight", "damped_grounded_equal_weight")
        }
    )

    def overridden_board(self):
        # The raw references sit an offset off truth (the pressure pattern);
        # the configured grounded references are competitive.
        return make_board(
            [
                board_row("persistence", 12.35, skill=0.3, p=0.01),
                board_row("equal_weight", 38.0),
                board_row("best_provider", 39.0),
                board_row("grounded_equal_weight", 13.0),
                board_row("damped_grounded_equal_weight", 13.5),
            ]
        )

    def test_gate_references_resolves_override(self):
        assert gate_references("humidity_pct", self.OVERRIDES) == (
            "grounded_equal_weight",
            "damped_grounded_equal_weight",
        )
        assert gate_references("temp_c", self.OVERRIDES)[0] == "best_provider"
        assert gate_references("humidity_pct", None)[0] == "best_provider"

    def test_union_keeps_default_columns_first(self):
        union = union_references(self.OVERRIDES)
        assert union[:3] == (
            "best_provider",
            "equal_weight",
            "damped_grounded_equal_weight",
        )
        assert "grounded_equal_weight" in union

    def test_gate_failure_falls_back_to_a_grounded_reference(self):
        board = self.overridden_board().with_columns(
            pl.lit(0.4).alias("dm_p_vs_grounded_equal_weight").cast(pl.Float64),
            pl.lit(0.4).alias("dm_p_vs_damped_grounded_equal_weight").cast(pl.Float64),
            pl.lit(0.1).alias("skill_vs_grounded_equal_weight").cast(pl.Float64),
        )
        row = slice_winners(board, rule="legacy", promotion=self.OVERRIDES).row(
            0, named=True
        )
        # The 38-off raw mean is no longer a permissible fallback.
        assert row["method_id"] == "grounded_equal_weight"
        assert row["gate"] == "dm_not_significant"

    def test_override_absent_reference_hits_missing_reference(self):
        board = make_board(
            [
                board_row("persistence", 12.35, skill=0.3, p=0.01),
                board_row("equal_weight", 38.0),
                board_row("best_provider", 39.0),
                board_row("damped_grounded_equal_weight", 13.5),
            ]
        )
        row = slice_winners(board, rule="legacy", promotion=self.OVERRIDES).row(
            0, named=True
        )
        assert row["method_id"] == "damped_grounded_equal_weight"
        assert row["gate"] == "missing_reference"


class TestBhAdjusted:
    def test_q_values_are_monotone_and_bounded(self):
        board = make_board(
            [
                board_row("a", 1.0, skill=0.3, p=0.01),
                board_row("b", 2.0, skill=0.2, p=0.04),
                board_row("c", 3.0, skill=0.1, p=0.5),
                board_row("equal_weight", 4.0),
            ]
        )
        adjusted = bh_adjusted(board)
        q = (
            adjusted.drop_nulls("dm_q_vs_equal_weight")
            .sort("dm_p_vs_equal_weight")["dm_q_vs_equal_weight"]
            .to_list()
        )
        p = (
            adjusted.drop_nulls("dm_q_vs_equal_weight")
            .sort("dm_p_vs_equal_weight")["dm_p_vs_equal_weight"]
            .to_list()
        )
        assert q == sorted(q)
        assert all(qv >= pv for qv, pv in zip(q, p, strict=True))
        assert all(0.0 <= qv <= 1.0 for qv in q)
        # BH arithmetic on (0.01, 0.04, 0.5) with m=3: q = (0.03, 0.06, 0.5)
        assert q[0] == 0.03


def store_with(log_e_by_method, reference="equal_weight", tmp_path=None):
    import math
    from pathlib import Path

    store = EProcessStore(
        path=Path(tmp_path or "/dev/null"),
        config_fingerprint="cfg1",
        code_version="code1",
    )
    for method_id, e_value in log_e_by_method.items():
        key = pair_key("hourly", "humidity_pct", "", "168-240h", method_id, reference)
        store.entries[key] = EProcessEntry(log_e=math.log(e_value), t=10)
    return store


class TestEbhAdjusted:
    def test_flags_top_wealth_pairs_and_reports_threshold(self):
        board = make_board(
            [
                board_row("a", 1.0, skill=0.3, p=0.01),
                board_row("b", 2.0, skill=0.2, p=0.04),
                board_row("c", 3.0, skill=0.1, p=0.5),
                board_row("untracked", 3.5, skill=0.0, p=0.9),
                board_row("equal_weight", 4.0),
            ]
        )
        store = store_with({"a": 400.0, "b": 30.0, "c": 1.5})
        adjusted = ebh_adjusted(board, store, alpha=0.05)
        by_method = {row["method_id"]: row for row in adjusted.iter_rows(named=True)}
        # K = 3 tracked pairs; sorted e = (400, 30, 1.5); k * e_[k] / K vs
        # 1/alpha = 20: k=1 -> 133, k=2 -> 20, k=3 -> 1.5, so k* = 2 and the
        # discovery bar is K / (alpha * k*) = 30.
        assert by_method["a"]["ebh_sig_vs_equal_weight"]
        assert by_method["b"]["ebh_sig_vs_equal_weight"]
        assert not by_method["c"]["ebh_sig_vs_equal_weight"]
        assert not by_method["untracked"]["ebh_sig_vs_equal_weight"]
        assert by_method["untracked"]["e_vs_equal_weight"] is None
        assert by_method["equal_weight"]["e_vs_equal_weight"] is None
        assert abs(by_method["a"]["ebh_threshold_vs_equal_weight"] - 30.0) < 1e-9

    def test_no_rejections_reports_the_standing_bar(self):
        board = make_board(
            [
                board_row("a", 1.0, skill=0.3, p=0.01),
                board_row("b", 2.0, skill=0.2, p=0.04),
                board_row("equal_weight", 4.0),
            ]
        )
        store = store_with({"a": 2.0, "b": 1.2})
        adjusted = ebh_adjusted(board, store, alpha=0.05)
        assert not adjusted["ebh_sig_vs_equal_weight"].any()
        # Nothing rejects, so the bar is K / alpha = 40 — the wealth a pair
        # must reach for the first FDR discovery.
        bars = adjusted["ebh_threshold_vs_equal_weight"].drop_nulls().unique()
        assert bars.to_list() == [40.0]

    def test_empty_board_passes_through(self):
        empty = pl.DataFrame()
        assert ebh_adjusted(empty, store_with({}), alpha=0.05).is_empty()


def daily_scores(
    buckets=("D3-4", "D5-7", "D8-10"),
    times_per_bucket=5,
    candidate_offset=0.05,
    reference_offset=5.0,
    seed=21,
):
    rng = np.random.default_rng(seed)
    references = ("equal_weight", "best_provider", "damped_grounded_equal_weight")
    rows = []
    issue = utc(2026, 7, 1)
    for bucket_index, bucket in enumerate(buckets):
        for index in range(times_per_bucket):
            valid = utc(2026, 7, 3) + timedelta(
                days=bucket_index * times_per_bucket + index
            )
            truth = 30.0 + rng.normal(0.0, 0.1)
            rows.append(
                (
                    "candidate",
                    bucket,
                    issue,
                    valid,
                    truth + candidate_offset + rng.normal(0.0, 0.05),
                    truth,
                )
            )
            rows.extend(
                (
                    reference,
                    bucket,
                    issue,
                    valid,
                    truth + reference_offset + rng.normal(0.0, 0.05),
                    truth,
                )
                for reference in references
            )
    frame = pl.DataFrame(
        rows,
        schema={
            "method_id": pl.String,
            "lead_bucket": pl.String,
            "issue_time": pl.Datetime("us", "UTC"),
            "valid_time": pl.Datetime("us", "UTC"),
            "y_pred": pl.Float64,
            "y_true": pl.Float64,
        },
        orient="row",
    )
    return frame.with_columns(
        pl.lit("daily").alias("product"),
        pl.lit("temp_max_c").alias("variable"),
        pl.lit(120.0).alias("lead_hours"),
    )


def daily_board_row(method_id, mae, n_valid_times=4):
    return {
        "product": "daily",
        "variable": "temp_max_c",
        "lead_bucket": "D5-7",
        "method_id": method_id,
        "n": 20,
        "n_total": 20,
        "n_valid_times": n_valid_times,
        "coverage": 1.0,
        "mae": mae,
        "rmse": mae,
        "bias": 0.0,
        **{column: None for column in _SKILL_COLUMNS},
    }


class TestPooledDailyGate:
    def far_board(self):
        return make_board(
            [
                daily_board_row("candidate", 1.0),
                daily_board_row("equal_weight", 5.0),
                daily_board_row("best_provider", 5.1),
                daily_board_row("damped_grounded_equal_weight", 5.2),
            ]
        )

    def test_thin_far_bucket_promotes_through_the_pool(self):
        # Each fine bucket has 5 valid times (below the 8 floor); the pool
        # reaches 15 and the candidate separates cleanly.
        winners = slice_winners(
            self.far_board(), scores=daily_scores(), rule="mcs", alpha=0.1
        )
        row = winners.row(0, named=True)
        assert row["lead_bucket"] == "D5-7"  # selection stays per fine bucket
        assert row["method_id"] == "candidate"
        assert row["gate"] == "pooled_D3-10"

    def test_pool_below_the_floor_serves_nothing(self):
        # 2 valid times per fine bucket pools to 6 — the pooled board is
        # still ineligible, so the slice stays unserved (cold-start later)
        # rather than promoting on six dates of evidence.
        winners = slice_winners(
            self.far_board(),
            scores=daily_scores(times_per_bucket=2),
            rule="mcs",
            alpha=0.1,
        )
        assert winners.is_empty()

    def test_hourly_slices_never_pool(self):
        board = make_board(
            [
                board_row("persistence", 12.35, n=4),
                board_row("equal_weight", 18.09, n=4),
            ]
        ).with_columns(pl.lit(4).alias("n_valid_times"))
        winners = slice_winners(board, rule="mcs", scores=mcs_scores(4))
        assert winners.is_empty()


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
