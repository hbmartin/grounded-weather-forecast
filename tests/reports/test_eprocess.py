"""The anytime-valid sequential promotion gate."""

import json
import math
from datetime import timedelta

import numpy as np
import polars as pl

from grounded_weather_forecast.config import PromotionConfig
from grounded_weather_forecast.reports.eprocess import (
    EProcessStore,
    pair_key,
    promotion_comparison,
)
from grounded_weather_forecast.reports.leaderboard import slice_winners
from grounded_weather_forecast.timeutil import utc

_REFERENCES = ("best_provider", "equal_weight", "damped_grounded_equal_weight")
_SKILL_COLUMNS = tuple(
    column
    for reference in _REFERENCES
    for column in (f"skill_vs_{reference}", f"dm_p_vs_{reference}")
)


def make_store(tmp_path, config_fingerprint="cfg1", code_version="code1"):
    return EProcessStore.load(
        tmp_path / "eprocess" / "hourly_live.json", config_fingerprint, code_version
    )


def times(n, start=None):
    start = start if start is not None else utc(2026, 8, 1)
    return [start + timedelta(hours=index) for index in range(n)]


class TestBettingMartingale:
    def test_null_crossing_frequency_is_controlled(self, tmp_path):
        alpha = 0.1
        crossings = 0
        replications = 200
        for seed in range(replications):
            rng = np.random.default_rng(seed)
            store = make_store(tmp_path / str(seed))
            candidate = rng.normal(1.0, 0.3, 300)
            reference = rng.normal(1.0, 0.3, 300)  # equal skill: H0
            sup_log_e = -math.inf
            for index in range(300):
                entry = store.update_pair(
                    "key",
                    candidate[index : index + 1],
                    reference[index : index + 1],
                    times(1, utc(2026, 8, 1) + timedelta(hours=index)),
                )
                sup_log_e = max(sup_log_e, entry.log_e)
            if sup_log_e >= math.log(1.0 / alpha):
                crossings += 1
        assert crossings / replications <= 0.15

    def test_genuine_skill_crosses_and_stays(self, tmp_path):
        rng = np.random.default_rng(7)
        store = make_store(tmp_path)
        candidate = rng.normal(1.0, 0.1, 60)
        reference = rng.normal(6.0, 0.1, 60)
        entry = store.update_pair("key", candidate, reference, times(60))
        assert entry.log_e >= math.log(10.0)
        assert entry.t == 60

    def test_update_is_idempotent_under_the_cursor(self, tmp_path):
        rng = np.random.default_rng(8)
        store = make_store(tmp_path)
        candidate = rng.normal(1.0, 0.1, 30)
        reference = rng.normal(2.0, 0.1, 30)
        first = store.update_pair("key", candidate, reference, times(30))
        state = (first.log_e, first.t, first.lam, first.ons_a, first.scale)
        second = store.update_pair("key", candidate, reference, times(30))
        assert (
            second.log_e,
            second.t,
            second.lam,
            second.ons_a,
            second.scale,
        ) == state

    def test_lambda_stays_in_the_supermartingale_range(self, tmp_path):
        store = make_store(tmp_path)
        rng = np.random.default_rng(9)
        entry = store.update_pair(
            "key", rng.normal(0.0, 1.0, 200), rng.normal(5.0, 1.0, 200), times(200)
        )
        assert 0.0 <= entry.lam <= 0.5


class TestStorePersistence:
    def test_round_trip(self, tmp_path):
        store = make_store(tmp_path)
        rng = np.random.default_rng(10)
        store.update_pair(
            "hourly|temp_c|mean|0-1h|cand|ref",
            rng.normal(1.0, 0.1, 20),
            rng.normal(2.0, 0.1, 20),
            times(20),
        )
        store.save()
        reloaded = make_store(tmp_path)
        assert reloaded.entries.keys() == store.entries.keys()
        key = "hourly|temp_c|mean|0-1h|cand|ref"
        assert reloaded.entries[key].log_e == store.entries[key].log_e
        assert (
            reloaded.entries[key].cursor_valid_time
            == store.entries[key].cursor_valid_time
        )

    def test_fingerprint_change_resets_wealth(self, tmp_path):
        store = make_store(tmp_path)
        rng = np.random.default_rng(11)
        store.update_pair(
            "key", rng.normal(1.0, 0.1, 20), rng.normal(2.0, 0.1, 20), times(20)
        )
        store.save()
        reset = make_store(tmp_path, config_fingerprint="cfg2")
        assert reset.entries == {}
        assert reset.resets == 1

    def test_corrupt_file_loads_empty(self, tmp_path):
        path = tmp_path / "eprocess" / "hourly_live.json"
        path.parent.mkdir(parents=True)
        path.write_text("not json")
        store = make_store(tmp_path)
        assert store.entries == {}

    def test_saved_payload_is_readable_json(self, tmp_path):
        store = make_store(tmp_path)
        store.update_pair("key", np.array([1.0]), np.array([2.0]), times(1))
        store.save()
        payload = json.loads(store.path.read_text())
        assert payload["config_fingerprint"] == "cfg1"
        assert "key" in payload["entries"]

    def test_pair_key_shape(self):
        key = pair_key("hourly", "temp_c", "mean", "0-1h", "cand", "ref")
        assert key == "hourly|temp_c|mean|0-1h|cand|ref"


def board_row(method_id, mae, skill=None, p=None):
    row = {
        "product": "hourly",
        "variable": "temp_c",
        "truth_semantics": "mean",
        "lead_bucket": "12-24h",
        "method_id": method_id,
        "n": 100,
        "n_total": 100,
        "n_valid_times": 100,
        "coverage": 1.0,
        "mae": mae,
        "rmse": mae,
        "bias": 0.0,
    }
    for reference in _REFERENCES:
        row[f"skill_vs_{reference}"] = skill
        row[f"dm_p_vs_{reference}"] = p
    return row


def make_board(rows):
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        *(pl.col(column).cast(pl.Float64) for column in _SKILL_COLUMNS)
    )


def seq_scores(n_times, candidate_offset=0.0, reference_offset=5.0, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    issue = utc(2026, 7, 31)
    for index in range(n_times):
        valid = utc(2026, 8, 1) + timedelta(hours=index)
        truth = 20.0 + rng.normal(0.0, 0.1)
        rows.append(("candidate", issue, valid, truth + candidate_offset + 0.01, truth))
        rows.extend(
            (reference, issue, valid, truth + reference_offset + 0.01, truth)
            for reference in _REFERENCES
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
        pl.lit("temp_c").alias("variable"),
        pl.lit("12-24h").alias("lead_bucket"),
        pl.lit(18.0).alias("lead_hours"),
        pl.lit("mean").alias("semantics"),
    )


def seq_board():
    return make_board(
        [
            board_row("candidate", 1.0),
            board_row("best_provider", 6.0),
            board_row("equal_weight", 6.1),
            board_row("damped_grounded_equal_weight", 6.2),
        ]
    )


class TestSeqGate:
    def test_accumulated_evidence_promotes(self, tmp_path):
        store = make_store(tmp_path)
        winners = slice_winners(
            seq_board(),
            scores=seq_scores(60),
            rule="seq_mcs",
            alpha=0.1,
            eprocess_store=store,
        )
        row = winners.row(0, named=True)
        assert row["method_id"] == "candidate"
        assert row["gate"] is None

    def test_thin_evidence_serves_the_best_reference(self, tmp_path):
        store = make_store(tmp_path)
        winners = slice_winners(
            seq_board(),
            scores=seq_scores(3),
            rule="seq_mcs",
            alpha=0.1,
            eprocess_store=store,
        )
        row = winners.row(0, named=True)
        assert row["method_id"] == "best_provider"
        assert row["gate"] == "seq_e_below_threshold"

    def test_rerunning_the_gate_does_not_move_the_decision(self, tmp_path):
        store = make_store(tmp_path)
        scores = seq_scores(60)
        first = slice_winners(
            seq_board(), scores=scores, rule="seq_mcs", eprocess_store=store
        )
        wealth = {key: entry.log_e for key, entry in store.entries.items()}
        second = slice_winners(
            seq_board(), scores=scores, rule="seq_mcs", eprocess_store=store
        )
        assert first.equals(second)
        assert {key: entry.log_e for key, entry in store.entries.items()} == wealth


class TestPromotionComparison:
    def test_both_rules_reported_side_by_side(self, tmp_path):
        store = make_store(tmp_path)
        comparison = promotion_comparison(
            seq_board(),
            seq_scores(60),
            alpha=0.1,
            promotion=PromotionConfig(),
            store=store,
        )
        row = comparison.row(0, named=True)
        assert {"mcs_choice", "mcs_gate", "seq_choice", "seq_gate", "agree"} <= set(
            comparison.columns
        )
        assert row["seq_choice"] == "candidate"
        assert isinstance(row["agree"], bool)
