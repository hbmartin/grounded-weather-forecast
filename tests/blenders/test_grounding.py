import numpy as np
import pytest
from conftest import synthetic_hourly_matrix

from grounded_weather_forecast.blenders import get_factory
from grounded_weather_forecast.blenders.grounding import (
    BIAS_ONLY,
    FREE_SLOPE,
    AffineGrounding,
    fit_affine,
)
from grounded_weather_forecast.contracts import hourly_variable
from grounded_weather_forecast.dataset.matrix import to_supervised_slice
from grounded_weather_forecast.metrics.deterministic import mae

TEMP = hourly_variable("temp_c")


class TestFitAffine:
    def test_free_slope_recovers_bias_and_scale(self):
        rng = np.random.default_rng(0)
        x = rng.normal(10, 5, 500)
        y = 2.0 + 0.9 * x + rng.normal(0, 0.01, 500)
        intercept, slope = fit_affine(x, y, FREE_SLOPE)
        assert intercept == pytest.approx(2.0, abs=0.05)
        assert slope == pytest.approx(0.9, abs=0.01)

    def test_bias_only_is_the_default(self):
        rng = np.random.default_rng(0)
        x = rng.normal(10, 5, 500)
        y = x - 3.0 + rng.normal(0, 0.01, 500)
        intercept, slope = fit_affine(x, y)
        assert slope == 1.0
        assert intercept == pytest.approx(-3.0, abs=0.05)

    def test_bias_only_ignores_slope_and_removes_mean_error(self):
        rng = np.random.default_rng(1)
        x = rng.normal(10, 5, 500)
        y = 2.0 + 0.5 * x  # a real slope the bias-only fit must not chase
        intercept, slope = fit_affine(x, y, BIAS_ONLY)
        assert slope == 1.0
        # the correction still removes the training bias exactly
        assert float(np.mean(intercept + slope * x - y)) == pytest.approx(0.0, abs=1e-9)

    def test_shrinkage_interpolates(self):
        rng = np.random.default_rng(2)
        x = rng.normal(10, 5, 500)
        y = 2.0 + 0.6 * x + rng.normal(0, 0.01, 500)
        _, half = fit_affine(x, y, 0.5)
        assert half == pytest.approx(0.8, abs=0.01)  # halfway between 1.0 and 0.6

    def test_thin_data_is_identity(self):
        assert fit_affine(np.ones(5), np.ones(5), FREE_SLOPE) == (0.0, 1.0)

    def test_degenerate_variance_is_identity(self):
        x = np.full(100, 7.0)
        y = np.arange(100.0)
        assert fit_affine(x, y, FREE_SLOPE) == (0.0, 1.0)


class TestSeasonalRegimeShift:
    """The reason bias-only is the default: a free slope shrinks toward the
    training mean, so a seasonally unrepresentative training window makes the
    correction re-introduce the very bias it exists to remove."""

    def make(self, level, n=800, seed=3):
        rng = np.random.default_rng(seed)
        truth = rng.normal(level, 4.0, n)
        forecast = truth + rng.normal(2.0, 3.0, n)  # +2 warm bias, noisy
        return forecast, truth

    def test_free_slope_injects_bias_under_regime_shift(self):
        warm_x, warm_y = self.make(level=21.0)  # summer training window
        cold_x, cold_y = self.make(level=13.0, seed=4)  # cooler test window

        affine = fit_affine(warm_x, warm_y, FREE_SLOPE)
        bias_only = fit_affine(warm_x, warm_y, BIAS_ONLY)
        assert affine[1] < 0.95  # regression dilution: slope lands below 1

        affine_bias = float(np.mean(affine[0] + affine[1] * cold_x - cold_y))
        bias_only_bias = float(np.mean(bias_only[0] + bias_only[1] * cold_x - cold_y))
        raw_bias = float(np.mean(cold_x - cold_y))
        assert abs(bias_only_bias) < 0.3  # level-shift equivariant: still clean
        assert affine_bias > 1.0  # tilted toward the warm training mean
        assert abs(bias_only_bias) < abs(raw_bias)  # and it beat no correction


class TestAffineGrounding:
    def test_removes_known_bias(self):
        matrix = synthetic_hourly_matrix(
            days=30, biases={"alpha": 4.0, "beta": -2.0}, noise_sd=0.2
        )
        train = to_supervised_slice(matrix, TEMP)
        grounding = AffineGrounding().fit(train)
        corrected = grounding.transform(train.x)
        raw_bias_alpha = float(np.nanmean(train.x.values[:, 0] - train.y))
        corrected_bias_alpha = float(np.nanmean(corrected[:, 0] - train.y))
        assert abs(raw_bias_alpha) > 3.5
        assert abs(corrected_bias_alpha) < 0.2
        corrected_bias_beta = float(np.nanmean(corrected[:, 1] - train.y))
        assert abs(corrected_bias_beta) < 0.2

    def test_unknown_source_passes_through(self):
        matrix = synthetic_hourly_matrix(days=10)
        train = to_supervised_slice(matrix, TEMP)
        grounding = AffineGrounding().fit(train)
        renamed = to_supervised_slice(matrix, TEMP)
        object.__setattr__(renamed.x, "sources", ("ghost_a", "ghost_b"))
        untouched = grounding.transform(renamed.x)
        np.testing.assert_array_equal(untouched, renamed.x.values)

    def test_state_is_json_shaped(self):
        matrix = synthetic_hourly_matrix(days=15)
        train = to_supervised_slice(matrix, TEMP)
        state = AffineGrounding().fit(train).to_state()
        assert set(state) == {"alpha", "beta"}
        assert len(state["alpha"]["global"]) == 2


class TestGroundedBlenders:
    def test_grounding_beats_raw_equal_weight_on_biased_sources(self):
        matrix = synthetic_hourly_matrix(
            days=40, biases={"alpha": 4.0, "beta": 3.0}, noise_sd=0.3, seed=5
        )
        train = to_supervised_slice(matrix, TEMP)
        raw = get_factory("equal_weight")().fit(train).predict(train.x)
        grounded = get_factory("grounded_equal_weight")().fit(train).predict(train.x)
        assert mae(grounded.point, train.y) < 0.5 * mae(raw.point, train.y)

    def test_inverse_mse_downweights_noisy_source(self):
        matrix = synthetic_hourly_matrix(
            days=40,
            noise_sd={"alpha": 4.0, "beta": 0.3},
            seed=6,
        )
        train = to_supervised_slice(matrix, TEMP)
        equal = get_factory("grounded_equal_weight")().fit(train).predict(train.x)
        weighted = get_factory("inverse_mse")().fit(train).predict(train.x)
        assert mae(weighted.point, train.y) < mae(equal.point, train.y)
