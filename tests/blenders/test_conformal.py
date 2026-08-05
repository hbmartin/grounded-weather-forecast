from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from conftest import synthetic_hourly_matrix

from grounded_weather_forecast.blenders import conformal as conformal_module
from grounded_weather_forecast.blenders import get_factory
from grounded_weather_forecast.contracts import ForecastMatrix, hourly_variable
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
        assert state["schema_version"] == 3
        assert state["calibration"]["strategy"] == "chronological_70_30"
        assert state["calibration"]["proper_rows"] >= 60
        assert state["calibration"]["calibration_rows"] >= 20

    def test_only_later_out_of_sample_rows_update_cells(self):
        matrix = synthetic_hourly_matrix(days=20, noise_sd=1.0, seed=45)
        train = to_supervised_slice(matrix, TEMP)
        conformal = get_factory("conformal_gew")().fit(train)
        state = conformal.to_state()
        bucket_updates = sum(
            cell["updates"]
            for key, cell in state["cells"].items()
            if not key.startswith("__global__")
        )
        global_updates = sum(
            cell["updates"]
            for key, cell in state["cells"].items()
            if key.startswith("__global__")
        )
        assert bucket_updates == state["calibration"]["calibration_rows"]
        assert global_updates == state["calibration"]["calibration_rows"]
        assert bucket_updates < train.x.n_rows

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

    def test_mean_shift_regime_recovers_coverage(self):
        """The base goes ~26 units off mid-archive (the station-pressure
        pattern); warm-started, scale-aware radii must cover the tail."""
        matrix = synthetic_hourly_matrix(days=80, noise_sd=4.0, seed=48)
        midpoint = (
            matrix["issue_time"].min()
            + (matrix["issue_time"].max() - matrix["issue_time"].min()) / 2
        )
        matrix = matrix.with_columns(
            pl.when(pl.col("issue_time") > midpoint)
            .then(pl.col(column) + 26.0)
            .otherwise(pl.col(column))
            .alias(column)
            for column in matrix.columns
            if column.startswith("fx__")
        )
        train = to_supervised_slice(matrix, TEMP)
        conformal = get_factory("conformal_gew")().fit(train)
        result = conformal.predict(train.x)
        assert result.quantiles is not None
        issue = train.x.features["issue_time"]
        tail = (issue > matrix["issue_time"].max() - timedelta(days=10)).to_numpy()
        assert coverage80(result, train.y, tail) >= 0.7
        # the radii must have reached the offset scale, not ramped from zero
        halfwidth = float(
            np.mean(result.quantiles[tail, 4] - result.quantiles[tail, 1]) / 2
        )
        assert halfwidth >= 10.0

    def test_far_lead_rows_served_by_global_rung(self):
        matrix = synthetic_hourly_matrix(days=60, noise_sd=1.0, seed=49)
        train = to_supervised_slice(matrix, TEMP)
        conformal = get_factory("conformal_gew")().fit(train)
        far = ForecastMatrix.build(
            sources=train.x.sources,
            values=train.x.values[:8],
            lead_hours=np.full(8, 200.0),
            features=train.x.features[:8],
            product=train.x.product,
        )
        result = conformal.predict(far)
        # no 168-240h cell was ever trained; the pooled cell serves radii
        assert result.quantiles is not None
        assert np.isfinite(result.quantiles).all()


class TestCellState:
    def make_cell(self):
        return conformal_module._fresh_cell()

    def test_warm_start_equals_buffered_conservative_quantile(self):
        cell = self.make_cell()
        scores = [float(value) for value in range(1, 21)]  # 1..20
        for score in scores:
            cell.update(score)
        # rank rule min(n, ceil((n+1)*c)) on n=20: c=0.5 -> 11th, 0.8 -> 17th,
        # 0.9 -> 19th order statistic
        assert cell.radii.tolist() == [11.0, 17.0, 19.0]
        assert cell.error_sums.tolist() == [0.0, 0.0, 0.0]

    def test_large_offset_scores_covered_at_readiness(self):
        cell = self.make_cell()
        rng = np.random.default_rng(3)
        scores = rng.normal(26.0, 2.0, 300)
        covered = []
        for score in scores:
            if cell.ready():
                covered.append(float(score) <= cell.effective_radius(1))
            cell.update(float(score))
        assert np.mean(covered) >= 0.7

    def test_persistent_misses_cannot_wind_up_past_the_scale_bound(self):
        cell = self.make_cell()
        for _ in range(20):
            cell.update(1.0)
        for _ in range(500):
            radius_before = float(cell.radii[1])
            bound = radius_before + 2.6 * max(cell.recent)
            assert cell.effective_radius(1) <= bound
            cell.update(10.0 + cell.effective_radius(1))  # always a miss
        # the tangent integrator stays bounded by ~2.6x the trailing max score
        assert cell.effective_radius(1) <= float(cell.radii[1]) + 2.6 * max(cell.recent)
