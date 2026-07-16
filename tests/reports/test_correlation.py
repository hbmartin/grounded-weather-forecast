import pytest
from conftest import synthetic_hourly_matrix

from grounded_weather_forecast.contracts import hourly_variable
from grounded_weather_forecast.reports.correlation import error_correlation

TEMP = hourly_variable("temp_c")


class TestErrorCorrelation:
    def test_shared_bias_correlates(self):
        # identical noise seed component -> highly correlated errors when the
        # noise dominates and both sources share the same truth signal
        matrix = synthetic_hourly_matrix(days=20, noise_sd=0.01, biases={})
        table = error_correlation(matrix, TEMP)
        assert table.columns == ["source", "alpha", "beta"]
        assert table.height == 2
        diagonal = table.filter(table["source"] == "alpha")["alpha"][0]
        assert diagonal == pytest.approx(1.0)

    def test_independent_noise_weakly_correlated(self):
        matrix = synthetic_hourly_matrix(days=20, noise_sd=1.0)
        table = error_correlation(matrix, TEMP)
        off_diagonal = table.filter(table["source"] == "alpha")["beta"][0]
        assert abs(off_diagonal) < 0.3

    def test_missing_truth_column(self):
        matrix = synthetic_hourly_matrix(days=3).drop(
            "t__temp_c__inst", "t__temp_c__mean"
        )
        assert error_correlation(matrix, TEMP).is_empty()
