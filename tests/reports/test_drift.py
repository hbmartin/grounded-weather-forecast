import json
from datetime import timedelta

import numpy as np
import polars as pl
from conftest import synthetic_hourly_matrix, utc

from grounded_weather_forecast.contracts import hourly_variable
from grounded_weather_forecast.reports.drift import (
    CONSENSUS_SKIPPED_TIER,
    RESIDUAL_SKIPPED_TIER,
    consensus_alarms,
    drift_report,
    page_hinkley,
    residual_alarms,
    write_drift_artifact,
)

TEMP = hourly_variable("temp_c")
PRECIP = hourly_variable("precip_mm")


def swap_matrix(offset=5.0, days=40, sources=("a", "b", "c", "d", "e")):
    """Source ``a`` swaps its backend three days before the end."""
    matrix = synthetic_hourly_matrix(days=days, sources=sources, noise_sd=0.4, seed=51)
    cutover = matrix["issue_time"].max() - pl.duration(days=3)
    return matrix.with_columns(
        pl.when(pl.col("issue_time") > cutover)
        .then(pl.col("fx__a__temp_c") + offset)
        .otherwise(pl.col("fx__a__temp_c"))
        .alias("fx__a__temp_c")
    )


def precip_matrix(forecasts, truths):
    """A single-source, single-lead precip matrix from parallel value lists."""
    issues = [utc(2026, 6, 1) + timedelta(hours=12 * i) for i in range(len(forecasts))]
    return pl.DataFrame(
        {
            "issue_time": issues,
            "valid_time": [issue + timedelta(hours=1) for issue in issues],
            "lead_hours": [1.0] * len(forecasts),
            "t__precip_mm": truths,
            "fx__a__precip_mm": forecasts,
        },
        schema_overrides={"t__precip_mm": pl.Float64, "fx__a__precip_mm": pl.Float64},
    )


def dry_july_matrix(snapshots=120, burst_mm=8.6):
    """A dry month of perfect zero forecasts with one missed monsoon burst."""
    truths = [0.0] * snapshots
    truths[snapshots // 2] = burst_mm
    return precip_matrix(forecasts=[0.0] * snapshots, truths=truths)


def drizzle_bias_matrix(snapshots=120, bias_mm=0.4, seed=7):
    """Dry truth throughout; the source starts inventing drizzle mid-window."""
    rng = np.random.default_rng(seed)
    onset = snapshots // 3
    forecasts = [
        0.0 if i < onset else bias_mm + float(rng.normal(0.0, 0.05))
        for i in range(snapshots)
    ]
    return precip_matrix(forecasts=forecasts, truths=[0.0] * snapshots)


class TestPageHinkley:
    def test_detects_a_step(self):
        rng = np.random.default_rng(3)
        series = np.concatenate([rng.normal(0.0, 1.0, 200), rng.normal(3.0, 1.0, 100)])
        alarmed, excursion = page_hinkley(series)
        assert alarmed
        assert excursion > 12.0

    def test_quiet_on_stationary(self):
        rng = np.random.default_rng(4)
        alarmed, _ = page_hinkley(rng.normal(0.0, 1.0, 300))
        assert not alarmed

    def test_detects_a_downward_step(self):
        rng = np.random.default_rng(3)
        series = np.concatenate([rng.normal(0.0, 1.0, 200), rng.normal(-3.0, 1.0, 100)])
        alarmed, excursion = page_hinkley(series)
        assert alarmed
        assert excursion > 12.0


class TestConsensusTier:
    def test_swapped_source_alarms_fast(self):
        alarms = consensus_alarms(swap_matrix(), TEMP)
        assert any(a.source == "a" for a in alarms)
        assert all(a.tier == "consensus" for a in alarms)

    def test_stationary_sources_stay_quiet(self):
        matrix = synthetic_hourly_matrix(
            days=40, sources=("a", "b", "c", "d", "e"), noise_sd=0.4, seed=52
        )
        assert consensus_alarms(matrix, TEMP) == []

    def test_needs_a_crowd(self):
        matrix = synthetic_hourly_matrix(days=40, noise_sd=0.4, seed=53)  # 2 sources
        assert consensus_alarms(matrix, TEMP) == []

    def test_zero_variance_baseline_notes_instead_of_alarming(self):
        """A +0.01 mm drift over an all-zero baseline used to produce z-scores
        in the tens of thousands (future-work #22a); it must skip, not page."""
        issues = [utc(2026, 6, 1) + timedelta(hours=i) for i in range(40 * 24)]
        cutover = issues[-1] - timedelta(days=2)
        columns: dict[str, object] = {
            "issue_time": issues,
            "valid_time": [issue + timedelta(hours=1) for issue in issues],
            "lead_hours": [1.0] * len(issues),
            "t__precip_mm": [0.0] * len(issues),
            "fx__a__precip_mm": [0.01 if issue > cutover else 0.0 for issue in issues],
        }
        for source in ("b", "c", "d", "e"):
            columns[f"fx__{source}__precip_mm"] = [0.0] * len(issues)
        matrix = pl.DataFrame(
            columns,
            schema_overrides={
                name: pl.Float64
                for name in columns
                if name.endswith("precip_mm") or name == "lead_hours"
            },
        )
        alarms = consensus_alarms(matrix, PRECIP)
        assert alarms
        assert all(a.tier == CONSENSUS_SKIPPED_TIER for a in alarms)
        assert all(abs(a.statistic) <= 1.0 for a in alarms)


class TestResidualTier:
    def test_persistent_shift_alarms(self):
        alarms = residual_alarms(swap_matrix(offset=6.0), TEMP)
        assert any(a.source == "a" and a.tier == "residual" for a in alarms)

    def test_report_combines_tiers(self):
        report = drift_report(swap_matrix(offset=6.0), (TEMP,))
        assert not report.is_empty()
        assert set(report["tier"].unique().to_list()) <= {"consensus", "residual"}
        assert report["lead_bucket"].null_count() == 0

    def test_horizon_row_duplication_does_not_change_residual_alarms(self):
        matrix = swap_matrix(offset=6.0)
        original = residual_alarms(matrix, TEMP)
        duplicated = residual_alarms(pl.concat([matrix] * 16), TEMP)
        assert duplicated == original

    def test_healthy_variable_is_never_skipped(self):
        rows = residual_alarms(swap_matrix(offset=6.0), TEMP)
        assert rows
        assert not any(r.tier == RESIDUAL_SKIPPED_TIER for r in rows)

    def test_stationary_temp_stays_quiet_without_notes(self):
        matrix = synthetic_hourly_matrix(
            days=40, sources=("a", "b", "c", "d", "e"), noise_sd=0.4, seed=52
        )
        assert residual_alarms(matrix, TEMP) == []

    def test_artifact_is_schema_version_two(self, tmp_path):
        path = tmp_path / "drift.json"
        write_drift_artifact(drift_report(swap_matrix(), (TEMP,)), path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 2


class TestResidualTierZeroInflated:
    def test_dry_month_single_burst_skips_with_a_note(self):
        rows = residual_alarms(dry_july_matrix(), PRECIP)
        assert [r.tier for r in rows] == [RESIDUAL_SKIPPED_TIER]
        assert "skipped_degenerate" in rows[0].detail
        assert rows[0].source == "a"

    def test_jittery_dry_month_burst_neither_alarms_nor_notes(self):
        """Tiny forecast jitter defeats the degenerate guard; the clipped
        standardization must still keep a lone burst below lambda."""
        rng = np.random.default_rng(11)
        forecasts = np.abs(rng.normal(0.0, 0.05, 120)).tolist()
        truths = [0.0] * 120
        truths[60] = 8.6
        assert residual_alarms(precip_matrix(forecasts, truths), PRECIP) == []

    def test_sustained_drizzle_bias_still_alarms(self):
        rows = residual_alarms(drizzle_bias_matrix(), PRECIP)
        assert any(r.source == "a" and r.tier == "residual" for r in rows)
        assert not any(r.tier == RESIDUAL_SKIPPED_TIER for r in rows)

    def test_artifact_splits_notes_from_alarms(self, tmp_path):
        path = tmp_path / "drift.json"
        write_drift_artifact(drift_report(dry_july_matrix(), (PRECIP,)), path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 2
        assert payload["alarms"] == []
        notes = payload["notes"]
        assert notes
        assert all(note["tier"] == RESIDUAL_SKIPPED_TIER for note in notes)
        assert "skipped_degenerate" in notes[0]["detail"]

    def test_artifact_keeps_real_alarms_out_of_notes(self, tmp_path):
        path = tmp_path / "drift.json"
        write_drift_artifact(drift_report(swap_matrix(offset=6.0), (TEMP,)), path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert any(alarm["tier"] == "residual" for alarm in payload["alarms"])
        assert payload["notes"] == []
