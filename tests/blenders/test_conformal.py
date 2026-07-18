from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from conftest import synthetic_hourly_matrix

from grounded_weather_forecast.blenders import get_factory
from grounded_weather_forecast.contracts import hourly_variable
from grounded_weather_forecast.dataset.matrix import to_supervised_slice
from grounded_weather_forecast.metrics.probabilistic import empirical_coverage

TEMP = hourly_variable("temp_c")


def coverage80(result, y, rows):
    lower = result.quantiles[rows, 1]  # level 0.1
    upper = result.quantiles[rows, 4]  # level 0.9
    return empirical_coverage(y[rows], lower, upper)


class TestConformal:
    def test_stationary_coverage_near_nominal(self):
        matrix = synthetic_hourly_matrix(days=60, noise_sd=1.0, seed=41)
        train = to_supervised_slice(matrix, TEMP)
        conformal = get_factory("conformal_gew")().fit(train)
        result = conformal.predict(train.x)
        assert result.quantiles is not None
        late = np.arange(train.x.n_rows) > train.x.n_rows // 2
        assert coverage80(result, train.y, late) == pytest.approx(0.8, abs=0.06)

    def test_variance_regime_shift_recovers_coverage(self):
        """Noise triples mid-archive; the tracker re-covers within the tail."""
        matrix = synthetic_hourly_matrix(days=80, noise_sd=1.0, seed=42)
        midpoint = (
            matrix["issue_time"].min()
            + (matrix["issue_time"].max() - matrix["issue_time"].min()) / 2
        )
        rng = np.random.default_rng(7)
        extra = rng.normal(0.0, 3.0, matrix.height)
        matrix = matrix.with_columns(
            pl.when(pl.col("issue_time") > midpoint)
            .then(pl.col("fx__alpha__temp_c") + pl.Series(extra))
            .otherwise(pl.col("fx__alpha__temp_c"))
            .alias("fx__alpha__temp_c")
        )
        train = to_supervised_slice(matrix, TEMP)
        conformal = get_factory("conformal_gew")().fit(train)
        result = conformal.predict(train.x)
        issue = train.x.features["issue_time"]
        tail = (issue > matrix["issue_time"].max() - timedelta(days=10)).to_numpy()
        head = (issue < midpoint).to_numpy()
        assert coverage80(result, train.y, tail) == pytest.approx(0.8, abs=0.08)
        # the widened tail intervals must be wider than the calm-era scores
        # would demand: sharpness grew with the regime
        tail_width = float(
            np.mean(result.quantiles[tail, 4] - result.quantiles[tail, 1])
        )
        head_width_needed = float(
            np.quantile(np.abs(train.y[head] - result.point[head]), 0.8) * 2
        )
        assert tail_width > head_width_needed

    def test_state_serializes(self):
        matrix = synthetic_hourly_matrix(days=20, noise_sd=1.0, seed=43)
        train = to_supervised_slice(matrix, TEMP)
        conformal = get_factory("conformal_gew")().fit(train)
        state = conformal.to_state()
        assert state["coverages"] == [0.5, 0.8, 0.9]
        assert len(state["cells"]) > 0
        assert state["schema_version"] == 2
        assert state["calibration"]["strategy"] == "chronological_70_30"
        assert state["calibration"]["proper_rows"] >= 60
        assert state["calibration"]["calibration_rows"] >= 20

    def test_only_later_out_of_sample_rows_update_cells(self):
        matrix = synthetic_hourly_matrix(days=20, noise_sd=1.0, seed=45)
        train = to_supervised_slice(matrix, TEMP)
        conformal = get_factory("conformal_gew")().fit(train)
        state = conformal.to_state()
        updates = sum(cell["updates"] for cell in state["cells"].values())
        assert updates == state["calibration"]["calibration_rows"]
        assert updates < train.x.n_rows

    def test_proper_training_excludes_truth_unresolved_at_cutoff(self):
        matrix = synthetic_hourly_matrix(days=20, noise_sd=1.0, seed=46)
        train = to_supervised_slice(matrix, TEMP)
        conformal = get_factory("conformal_gew")().fit(train)
        state = conformal.to_state()
        issue = train.x.features["issue_time"].cast(pl.Int64).to_numpy()
        cutoff = state["calibration"]["cutoff_issue_us"]
        naively_resolved = int(
            np.sum(
                (issue < cutoff)
                & (train.x.features["valid_time"].cast(pl.Int64).to_numpy() <= cutoff)
            )
        )
        assert state["calibration"]["proper_rows"] < naively_resolved

    def test_thin_cells_emit_no_quantiles(self):
        matrix = synthetic_hourly_matrix(days=1, max_lead=6, seed=44)
        train = to_supervised_slice(matrix, TEMP)
        conformal = get_factory("conformal_gew")().fit(train)
        result = conformal.predict(train.x)
        # 12 rows total: no cell reaches _MIN_UPDATES, base passes through
        assert result.quantiles is None
