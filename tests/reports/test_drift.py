import json

import numpy as np
import polars as pl
from conftest import synthetic_hourly_matrix

from grounded_weather_forecast.contracts import hourly_variable
from grounded_weather_forecast.reports.drift import (
    consensus_alarms,
    drift_report,
    page_hinkley,
    residual_alarms,
    write_drift_artifact,
)

TEMP = hourly_variable("temp_c")


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

    def test_artifact_is_schema_version_two(self, tmp_path):
        path = tmp_path / "drift.json"
        write_drift_artifact(drift_report(swap_matrix(), (TEMP,)), path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 2
