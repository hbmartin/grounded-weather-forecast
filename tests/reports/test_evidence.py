"""The append-only quality-evidence ledgers."""

from datetime import timedelta

import numpy as np
import polars as pl
from conftest import write_config
from filelock import FileLock

from grounded_weather_forecast.reports import evidence
from grounded_weather_forecast.reports.eprocess import EProcessEntry, EProcessStore
from grounded_weather_forecast.reports.evidence import (
    CHURN_LEDGER,
    CHURN_SCHEMA,
    EPROCESS_WEALTH_LEDGER,
    LEDGERS,
    QUALITY_LEDGER,
    QUALITY_SCHEMA,
    SERVED_QUALITY_LEDGER,
    SERVED_QUALITY_SCHEMA,
    VERDICTS_LEDGER,
    append_ledger,
    gate_verdicts,
    ledger_path,
    prune_ledger,
    quality_delta_line,
    quality_rows,
    read_ledger,
    recalibration_verdicts,
    record_selection_churn,
    selection_churn,
    served_quality_rows,
    wealth_rows,
)
from grounded_weather_forecast.timeutil import utc

NOW = utc(2026, 8, 6)


def quality_row(evaluation_id="eval1", method_id="equal_weight", recorded_at=NOW):
    return {
        "recorded_at": recorded_at,
        "evaluation_id": evaluation_id,
        "evaluation_created_at": NOW,
        "product": "hourly",
        "source_kind": "live",
        "variable": "temp_c",
        "truth_semantics": "mean",
        "lead_bucket": "0-1h",
        "method_id": method_id,
        "n": 100,
        "n_valid_times": 90,
        "coverage": 1.0,
        "mae": 1.5,
        "rmse": 2.0,
        "bias": 0.1,
        "coverage80": None,
        "coverage90": None,
        "crps": None,
        "pinball": None,
        "recent_mae": 1.4,
        "recent_n": 30,
        "code_version": "code1",
        "config_fingerprint": "cfg1",
        "dataset_fingerprint": "ds1",
    }


def quality_frame(rows):
    # schema= (not schema_overrides=) so row dicts predating newer ledger
    # columns null-fill them, mirroring the engine's write-side tolerance.
    return pl.DataFrame(rows, schema=dict(QUALITY_SCHEMA)).select(
        QUALITY_SCHEMA.names()
    )


class TestLedgerEngine:
    def test_append_and_read_round_trip(self, tmp_path):
        path = tmp_path / "quality.parquet"
        append_ledger(quality_frame([quality_row()]), path, QUALITY_LEDGER, now=NOW)
        loaded = read_ledger(path, QUALITY_SCHEMA)
        assert loaded.height == 1
        assert loaded.schema == QUALITY_SCHEMA

    def test_duplicate_append_is_noop_and_keeps_first_recorded_at(self, tmp_path):
        path = tmp_path / "quality.parquet"
        first = quality_frame([quality_row(recorded_at=NOW)])
        later = quality_frame([quality_row(recorded_at=NOW + timedelta(days=1))])
        append_ledger(first, path, QUALITY_LEDGER, now=NOW)
        append_ledger(later, path, QUALITY_LEDGER, now=NOW + timedelta(days=1))
        loaded = read_ledger(path, QUALITY_SCHEMA)
        assert loaded.height == 1
        assert loaded["recorded_at"][0] == NOW

    def test_rerendered_old_evaluation_is_noop(self, tmp_path):
        path = tmp_path / "quality.parquet"
        append_ledger(
            quality_frame([quality_row("evalA")]), path, QUALITY_LEDGER, now=NOW
        )
        both = quality_frame([quality_row("evalA"), quality_row("evalB")])
        append_ledger(both, path, QUALITY_LEDGER, now=NOW)
        loaded = read_ledger(path, QUALITY_SCHEMA)
        assert loaded.height == 2
        assert set(loaded["evaluation_id"]) == {"evalA", "evalB"}

    def test_null_release_id_dedupes(self, tmp_path):
        path = tmp_path / "served_quality.parquet"
        row = {
            "recorded_at": NOW,
            "as_of_date": NOW.date(),
            "product": "hourly",
            "variable": "temp_c",
            "truth_semantics": "mean",
            "lead_bucket": "0-1h",
            "method_id": "equal_weight",
            "release_id": None,
            "dataset_fingerprint": "ds1",
            "n": 24,
            "live_mae": 1.0,
            "live_rmse": 1.5,
            "live_bias": 0.0,
            "backtest_mae": 0.9,
            "mae_gap": 0.1,
            "code_version": "code1",
            "config_fingerprint": "cfg1",
        }
        frame = pl.DataFrame(
            [row], schema_overrides=dict(SERVED_QUALITY_SCHEMA)
        ).select(SERVED_QUALITY_SCHEMA.names())
        append_ledger(frame, path, SERVED_QUALITY_LEDGER, now=NOW)
        append_ledger(frame, path, SERVED_QUALITY_LEDGER, now=NOW)
        assert read_ledger(path, SERVED_QUALITY_SCHEMA).height == 1

    def test_retention_prunes_by_age_and_keeps_null_timestamps(self):
        frame = quality_frame(
            [
                quality_row("old", recorded_at=NOW - timedelta(days=1000)),
                quality_row("new", recorded_at=NOW),
            ]
        ).with_columns(
            pl.when(pl.col("evaluation_id") == "old")
            .then(pl.lit(NOW - timedelta(days=1000)))
            .otherwise(pl.col("recorded_at"))
            .alias("recorded_at")
        )
        with_null = pl.concat(
            [
                frame,
                quality_frame([quality_row("nulltime")]).with_columns(
                    pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("recorded_at")
                ),
            ]
        )
        pruned = prune_ledger(with_null, QUALITY_LEDGER, now=NOW)
        assert set(pruned["evaluation_id"]) == {"new", "nulltime"}

    def test_row_cap_tails(self):
        spec = evidence.LedgerSpec(
            name="tiny",
            schema=QUALITY_SCHEMA,
            dedupe_keys=QUALITY_LEDGER.dedupe_keys,
            max_rows=2,
        )
        frame = quality_frame([quality_row(f"eval{i}") for i in range(5)])
        assert prune_ledger(frame, spec, now=NOW).height == 2

    def test_corrupt_ledger_rewritten_with_fresh_only(self, tmp_path):
        path = tmp_path / "quality.parquet"
        path.write_text("not parquet")
        append_ledger(quality_frame([quality_row()]), path, QUALITY_LEDGER, now=NOW)
        assert read_ledger(path, QUALITY_SCHEMA).height == 1

    def test_read_ledger_null_fills_missing_columns(self, tmp_path):
        path = tmp_path / "quality.parquet"
        pl.DataFrame({"evaluation_id": ["evalA"]}).write_parquet(path)
        loaded = read_ledger(path, QUALITY_SCHEMA)
        assert loaded.height == 1
        assert loaded["mae"][0] is None

    def test_append_never_raises_under_held_lock(self, tmp_path):
        path = tmp_path / "quality.parquet"
        lock = FileLock(path.with_suffix(".parquet.lock"))
        with lock:
            append_ledger(quality_frame([quality_row()]), path, QUALITY_LEDGER, now=NOW)
        assert not path.exists()

    def test_every_spec_registered(self):
        assert set(LEDGERS) == {
            "quality",
            "churn",
            "verdicts",
            "eprocess_wealth",
            "served_quality",
        }


def scores_frame(evaluation_id="eval1", days=30, created=NOW):
    rng = np.random.default_rng(5)
    rows = []
    for day in range(days):
        valid = utc(2026, 7, 1) + timedelta(days=day)
        rows.append(
            {
                "method_id": "equal_weight",
                "variable": "temp_c",
                "product": "hourly",
                "source_kind": "live",
                "evaluation_id": evaluation_id,
                "evaluation_created_at": created,
                "dataset_fingerprint": "ds1",
                "source_set_json": "[]",
                "feature_set_json": "[]",
                "semantics": "mean",
                "code_version": "code1",
                "config_fingerprint": "cfg1",
                "window": "expanding",
                "fold_origin": created,
                "issue_time": valid - timedelta(hours=1),
                "valid_time": valid,
                "lead_hours": 1.0,
                "lead_bucket": "0-1h",
                # errors: 1.0 in the old era, 3.0 in the recent 14 days
                "y_pred": 20.0
                + (3.0 if day >= days - 14 else 1.0)
                + rng.normal(0.0, 0.001),
                "y_true": 20.0,
                "quantile_levels_json": "[]",
                "quantiles_json": None,
            }
        )
    return pl.DataFrame(rows)


class TestQualityRows:
    def board_for(self, scores):
        from grounded_weather_forecast.reports.leaderboard import leaderboard

        return leaderboard(scores, references=())

    def test_recent_window_uses_only_the_last_14_days(self):
        scores = scores_frame()
        rows = quality_rows(scores, self.board_for(scores), now=NOW)
        row = rows.row(0, named=True)
        assert row["recent_n"] == 14
        assert abs(row["recent_mae"] - 3.0) < 0.01
        assert abs(row["mae"] - (14 * 3.0 + 16 * 1.0) / 30) < 0.01

    def test_multi_evaluation_frame_uses_newest_only(self):
        old = scores_frame("evalOld", created=NOW - timedelta(days=1))
        new = scores_frame("evalNew", created=NOW)
        rows = quality_rows(
            pl.concat([old, new]), self.board_for(pl.concat([old, new])), now=NOW
        )
        assert set(rows["evaluation_id"]) == {"evalNew"}

    def test_empty_inputs_yield_empty_frame(self):
        scores = scores_frame()
        assert quality_rows(pl.DataFrame(), pl.DataFrame(), now=NOW).is_empty()
        assert quality_rows(scores, pl.DataFrame(), now=NOW).is_empty()


def quantile_scores_frame(periods=120, step_hours=6, miss_every=4):
    """Every miss_every-th recent row falls outside the interval; the old
    era misses entirely, so pooled and recent coverage must disagree."""
    base = scores_frame(days=1).row(0, named=True)
    start = utc(2026, 7, 1)
    anchor = start + timedelta(hours=step_hours * (periods - 1))
    rows = []
    for index in range(periods):
        valid = start + timedelta(hours=step_hours * index)
        recent = valid > anchor - timedelta(days=14)
        miss = (index % miss_every == 0) if recent else True
        rows.append(
            {
                **base,
                "valid_time": valid,
                "issue_time": valid - timedelta(hours=1),
                "y_pred": 20.0,
                "y_true": 30.0 if miss else 20.0,
                "quantile_levels_json": "[0.05, 0.1, 0.9, 0.95]",
                "quantiles_json": "[17.0, 18.0, 22.0, 23.0]",
            }
        )
    return pl.DataFrame(rows)


class TestRecentQuantileMetrics:
    def board_for(self, scores):
        from grounded_weather_forecast.reports.leaderboard import leaderboard

        return leaderboard(scores, references=())

    def test_recent_window_coverage_diverges_from_pooled(self):
        scores = quantile_scores_frame()
        rows = quality_rows(scores, self.board_for(scores), now=NOW)
        row = rows.row(0, named=True)
        assert abs(row["recent_coverage80"] - 0.75) < 0.05
        assert row["recent_coverage90"] == row["recent_coverage80"]
        assert row["recent_crps"] is not None and row["recent_crps"] > 0.0
        # the pooled board metric drowns the recent repair in the old era
        assert row["coverage80"] < row["recent_coverage80"]

    def test_point_only_method_stays_null(self):
        scores = scores_frame()
        rows = quality_rows(scores, self.board_for(scores), now=NOW)
        assert rows["recent_coverage80"][0] is None
        assert rows["recent_crps"][0] is None

    def test_thin_recent_window_stays_null(self):
        scores = quantile_scores_frame(periods=30, step_hours=24)
        rows = quality_rows(scores, self.board_for(scores), now=NOW)
        assert rows["recent_coverage80"][0] is None


class TestDiscoveryVerdicts:
    def test_counts_both_corrections(self):
        board = pl.DataFrame(
            {
                "method_id": ["a", "b", "c"],
                "dm_q_vs_ref": [0.01, 0.2, None],
                "e_vs_ref": [50.0, 1.0, None],
                "ebh_sig_vs_ref": [True, False, False],
            }
        )
        verdicts = evidence.discovery_verdicts(board, alpha=0.05)
        assert verdicts["pbh_discoveries"] == 1.0
        assert verdicts["pbh_pairs"] == 2.0
        assert verdicts["ebh_discoveries"] == 1.0
        assert verdicts["ebh_pairs"] == 2.0

    def test_empty_board_yields_nothing(self):
        assert evidence.discovery_verdicts(pl.DataFrame(), alpha=0.05) == {}


def release_payload(release_id, promoted_at, selections):
    return {
        "release_id": release_id,
        "promoted_at": promoted_at.isoformat(),
        "dataset_fingerprint": "ds1",
        "config_fingerprint": "cfg1",
        "selections": selections,
    }


def selection_entry(method_id, mae=1.0, n=100):
    return {
        "method_id": method_id,
        "reason": "test",
        "evaluation_id": "eval1",
        "code_version": "code1",
        "n": n,
        "mae": mae,
        "truth_semantics": "mean",
    }


class TestSelectionChurn:
    def test_diff_marks_changed_added_removed(self):
        previous = release_payload(
            "relA",
            NOW - timedelta(days=1),
            {
                "hourly.temp_c.0-1h": selection_entry("equal_weight", 1.8),
                "hourly.temp_c.1-3h": selection_entry("harmonic", 1.3),
                "hourly.pop.0-1h": selection_entry("damped", 0.0),
            },
        )
        current = release_payload(
            "relB",
            NOW,
            {
                "hourly.temp_c.0-1h": selection_entry("anchored_trend", 1.2),
                "hourly.temp_c.1-3h": selection_entry("harmonic", 1.25),
                "daily.temp_max_c.D1": selection_entry("inverse_mse", 1.1),
            },
        )
        churn = selection_churn(previous, current, now=NOW, code_version="code1")
        by_key = {row["slice_key"]: row for row in churn.iter_rows(named=True)}
        assert by_key["hourly.temp_c.0-1h"]["changed"] is True
        assert by_key["hourly.temp_c.1-3h"]["changed"] is False
        assert by_key["hourly.pop.0-1h"]["changed"] is True  # removed
        assert by_key["hourly.pop.0-1h"]["to_method"] is None
        assert by_key["daily.temp_max_c.D1"]["changed"] is True  # added
        assert by_key["daily.temp_max_c.D1"]["from_method"] is None
        row = by_key["hourly.temp_c.0-1h"]
        assert row["product"] == "hourly"
        assert row["variable"] == "temp_c"
        assert row["lead_bucket"] == "0-1h"
        assert row["from_mae"] == 1.8
        assert row["to_mae"] == 1.2

    def test_zero_or_one_release_yields_empty(self, tmp_path):
        config = write_config(tmp_path)
        assert record_selection_churn(config, now=NOW).is_empty()
        releases = config.artifacts_dir / "releases"
        releases.mkdir(parents=True)
        (releases / "a.json").write_text(
            pl.DataFrame().write_json()
        )  # corrupt-ish: not a release
        assert record_selection_churn(config, now=NOW).is_empty()
        assert not ledger_path(config, CHURN_LEDGER).exists()

    def test_record_skips_corrupt_and_is_idempotent(self, tmp_path):
        import json

        config = write_config(tmp_path)
        releases = config.artifacts_dir / "releases"
        releases.mkdir(parents=True)
        (releases / "bad.json").write_text("{ not json")
        for name, when, method in (
            ("relA", NOW - timedelta(days=1), "equal_weight"),
            ("relB", NOW, "harmonic"),
        ):
            (releases / f"{name}.json").write_text(
                json.dumps(
                    release_payload(
                        name, when, {"hourly.temp_c.0-1h": selection_entry(method)}
                    )
                )
            )
        first = record_selection_churn(config, now=NOW)
        assert first.height == 1
        record_selection_churn(config, now=NOW)
        assert read_ledger(ledger_path(config, CHURN_LEDGER), CHURN_SCHEMA).height == 1


class TestVerdicts:
    def recalib_frame(self, rows):
        return pl.DataFrame(
            rows,
            schema={
                "product": pl.String,
                "variable": pl.String,
                "semantics": pl.String,
                "lead_bucket": pl.String,
                "method_id": pl.String,
                "fit_scope": pl.String,
                "n_fit": pl.Int64,
                "n_eval": pl.Int64,
                "transform": pl.String,
                "coverage80": pl.Float64,
                "coverage90": pl.Float64,
                "pinball": pl.Float64,
            },
        )

    def cell(self, method, transform, coverage80, pinball=1.0):
        return (
            "hourly",
            "temp_c",
            "mean",
            "0-1h",
            method,
            "bucket",
            100,
            50,
            transform,
            coverage80,
            None,
            pinball,
        )

    def test_winner_by_coverage_then_pinball(self):
        frame = self.recalib_frame(
            [
                self.cell("idr", "raw", 0.5),
                self.cell("idr", "pit", 0.7),
                self.cell("idr", "cqr", 0.82),
                self.cell("emos", "raw", 0.79, pinball=2.0),
                self.cell("emos", "pit", 0.81, pinball=1.0),  # tie distance, wins
                self.cell("emos", "cqr", 0.81, pinball=3.0),
            ]
        )
        verdicts = recalibration_verdicts(frame)
        assert verdicts["recalib_cells"] == 2.0
        assert verdicts["recalib_win_share_cqr"] == 0.5
        assert verdicts["recalib_win_share_pit"] == 0.5
        assert verdicts["recalib_win_share_raw"] == 0.0
        assert (
            verdicts["recalib_win_share_raw"]
            + verdicts["recalib_win_share_pit"]
            + verdicts["recalib_win_share_cqr"]
        ) == 1.0

    def test_all_null_cells_excluded_and_empty_frame(self):
        assert recalibration_verdicts(pl.DataFrame()) == {}
        frame = self.recalib_frame(
            [self.cell("idr", transform, None) for transform in ("raw", "pit", "cqr")]
        )
        assert recalibration_verdicts(frame) == {}

    def test_gate_verdicts(self):
        comparison = pl.DataFrame(
            {"agree": [True, True, False, None]},
            schema={"agree": pl.Boolean},
        )
        verdicts = gate_verdicts(comparison)
        assert abs(verdicts["gate_agree_rate"] - 2 / 3) < 1e-9
        assert verdicts["gate_disagreements"] == 1.0
        assert verdicts["gate_slices"] == 3.0
        assert gate_verdicts(pl.DataFrame()) == {}


class TestWealthRows:
    def make_store(self, tmp_path, resets=0):
        store = EProcessStore(
            path=tmp_path / "hourly_live.json",
            config_fingerprint="cfg1",
            code_version="code1",
        )
        store.resets = resets
        store.entries["k1"] = EProcessEntry(log_e=1.5, t=10, lam=0.3, scale=2.0)
        return store

    def test_snapshot_rows_and_reset_era(self, tmp_path):
        config = write_config(tmp_path)
        path = ledger_path(config, EPROCESS_WEALTH_LEDGER)
        first = wealth_rows("hourly", self.make_store(tmp_path), now=NOW)
        append_ledger(first, path, EPROCESS_WEALTH_LEDGER, now=NOW)
        # unchanged wealth: no new rows
        append_ledger(first, path, EPROCESS_WEALTH_LEDGER, now=NOW)
        assert read_ledger(path, EPROCESS_WEALTH_LEDGER.schema).height == 1
        # after a reset, the same t must append as a new era
        reset_rows = wealth_rows("hourly", self.make_store(tmp_path, resets=1), now=NOW)
        append_ledger(reset_rows, path, EPROCESS_WEALTH_LEDGER, now=NOW)
        assert read_ledger(path, EPROCESS_WEALTH_LEDGER.schema).height == 2

    def test_empty_store_yields_empty(self, tmp_path):
        store = EProcessStore(
            path=tmp_path / "x.json", config_fingerprint="cfg1", code_version="code1"
        )
        assert wealth_rows("hourly", store, now=NOW).is_empty()


class TestServedQualityRows:
    def test_missing_backtest_columns_are_null_filled(self):
        live = pl.DataFrame(
            {
                "product": ["hourly"],
                "variable": ["temp_c"],
                "truth_semantics": ["mean"],
                "lead_bucket": ["0-1h"],
                "method_id": ["equal_weight"],
                "dataset_fingerprint": ["ds1"],
                "release_id": [None],
                "n": [24],
                "live_mae": [1.0],
                "live_rmse": [1.4],
                "live_bias": [0.1],
            },
            schema_overrides={"release_id": pl.String},
        )
        rows = served_quality_rows(
            live, config_fingerprint_value="cfg1", code_version="code1", now=NOW
        )
        row = rows.row(0, named=True)
        assert row["backtest_mae"] is None
        assert row["as_of_date"] == NOW.date()


class TestQualityDeltaLine:
    def test_two_evaluations_produce_a_line(self, tmp_path):
        config = write_config(tmp_path)
        path = ledger_path(config, QUALITY_LEDGER)
        older = quality_frame([quality_row("evalA")]).with_columns(
            pl.lit(NOW - timedelta(days=1)).alias("evaluation_created_at"),
            pl.lit(2.0).alias("mae"),
        )
        newer = quality_frame([quality_row("evalB")]).with_columns(
            pl.lit(NOW).alias("evaluation_created_at"), pl.lit(1.0).alias("mae")
        )
        append_ledger(older, path, QUALITY_LEDGER, now=NOW)
        append_ledger(newer, path, QUALITY_LEDGER, now=NOW)
        line = quality_delta_line(config)
        assert line is not None
        assert "hourly live" in line
        assert "2.000 -> 1.000" in line
        assert "-50.0%" in line

    def test_single_evaluation_is_none(self, tmp_path):
        config = write_config(tmp_path)
        append_ledger(
            quality_frame([quality_row()]),
            ledger_path(config, QUALITY_LEDGER),
            QUALITY_LEDGER,
            now=NOW,
        )
        assert quality_delta_line(config) is None

    def test_no_ledger_is_none(self, tmp_path):
        assert quality_delta_line(write_config(tmp_path)) is None


class TestVerdictsLedgerSpec:
    def test_dedupe_keys(self):
        assert VERDICTS_LEDGER.dedupe_keys == ("evaluation_id", "product", "name")


class TestWinnerCurseRecorder:
    def winners(self):
        return pl.DataFrame(
            {"winner_bias": [-0.04, -0.02], "near_tie_flag": [True, False]}
        )

    def test_records_scalars_and_annotates_the_delta_line(self, tmp_path):
        from conftest import write_config

        config = write_config(tmp_path)
        scores = scores_frame()
        evidence.record_winner_curse(config, "hourly", scores, self.winners(), now=NOW)
        ledger = pl.read_parquet(evidence.ledger_path(config, evidence.VERDICTS_LEDGER))
        names = set(ledger["name"].to_list())
        assert {"winner_bias_mean", "winner_bias_slices", "near_tie_slices"} <= names
        assert "argmin winner bias" in evidence._winner_bias_note(config, "hourly")

    def test_uncorrected_winners_record_nothing(self, tmp_path):
        from conftest import write_config

        config = write_config(tmp_path)
        evidence.record_winner_curse(
            config, "hourly", scores_frame(), pl.DataFrame({"mae": [1.0]}), now=NOW
        )
        assert not evidence.ledger_path(config, evidence.VERDICTS_LEDGER).exists()
