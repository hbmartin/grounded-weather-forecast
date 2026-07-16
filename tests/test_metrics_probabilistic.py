import numpy as np
import pytest

from grounded_weather_forecast.metrics.probabilistic import (
    brier,
    crps_ensemble,
    crps_from_quantiles,
    empirical_coverage,
    pinball_loss,
    pit_from_quantiles,
    reliability_bins,
)


class TestPinball:
    def test_hand_computed(self):
        # y=10, q=8, level 0.9: 0.9 * 2 = 1.8 ; y=10, q=12, level 0.9: 0.1 * 2 = 0.2
        y = np.array([10.0, 10.0])
        q = np.array([8.0, 12.0])
        assert pinball_loss(y, q, 0.9) == pytest.approx(1.0)

    def test_median_pinball_is_half_mae(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=100)
        q = rng.normal(size=100)
        assert pinball_loss(y, q, 0.5) == pytest.approx(np.mean(np.abs(y - q)) / 2)

    def test_bad_level(self):
        with pytest.raises(ValueError, match="quantile level"):
            pinball_loss(np.zeros(1), np.zeros(1), 1.0)


class TestCrps:
    def test_from_quantiles_shape_check(self):
        with pytest.raises(ValueError, match="quantiles shape"):
            crps_from_quantiles(np.zeros(3), np.zeros((3, 2)), (0.5,))

    def test_sharper_is_better(self):
        y = np.zeros(200)
        levels = (0.1, 0.25, 0.5, 0.75, 0.9)
        z = np.array([-1.2816, -0.6745, 0.0, 0.6745, 1.2816])
        sharp = np.tile(z * 0.5, (200, 1))
        wide = np.tile(z * 2.0, (200, 1))
        assert crps_from_quantiles(y, sharp, levels) < crps_from_quantiles(
            y, wide, levels
        )

    def test_ensemble_perfect_forecast(self):
        y = np.array([1.0, 2.0])
        ensemble = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        assert crps_ensemble(y, ensemble) == pytest.approx(0.0)

    def test_ensemble_degenerate_is_mae(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=50)
        point = rng.normal(size=50)
        ensemble = np.repeat(point[:, None], 5, axis=1)
        assert crps_ensemble(y, ensemble) == pytest.approx(
            float(np.mean(np.abs(y - point)))
        )


class TestBrier:
    def test_golden(self):
        pop = np.array([1.0, 0.0, 0.5])
        occurred = np.array([1.0, 0.0, 0.0])
        assert brier(pop, occurred) == pytest.approx(0.25 / 3)


class TestReliability:
    def test_bins(self):
        pop = np.array([0.05, 0.05, 0.95, 0.95])
        occurred = np.array([0.0, 1.0, 1.0, 1.0])
        table = reliability_bins(pop, occurred, n_bins=10)
        assert table.height == 10
        first = table.row(0, named=True)
        last = table.row(9, named=True)
        assert first["count"] == 2
        assert first["observed_freq"] == pytest.approx(0.5)
        assert last["count"] == 2
        assert last["observed_freq"] == pytest.approx(1.0)
        assert table["count"].sum() == 4


class TestCoverageAndPit:
    def test_coverage(self):
        y = np.array([0.0, 5.0, 10.0])
        lower = np.array([-1.0, 6.0, 9.0])
        upper = np.array([1.0, 7.0, 11.0])
        assert empirical_coverage(y, lower, upper) == pytest.approx(2.0 / 3.0)

    def test_pit_properties(self):
        levels = (0.1, 0.5, 0.9)
        quantiles = np.array([[-1.0, 0.0, 1.0]])
        assert pit_from_quantiles(np.array([0.0]), quantiles, levels)[0] == 0.5
        assert pit_from_quantiles(np.array([-5.0]), quantiles, levels)[0] == 0.0
        assert pit_from_quantiles(np.array([5.0]), quantiles, levels)[0] == 1.0
        mid = pit_from_quantiles(np.array([0.5]), quantiles, levels)[0]
        assert 0.5 < mid < 0.9
