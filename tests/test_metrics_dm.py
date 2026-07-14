import numpy as np
import pytest

from omni_forecast.metrics.dm import diebold_mariano


class TestDieboldMariano:
    def test_identical_losses(self):
        losses = np.abs(np.random.default_rng(0).normal(size=50))
        result = diebold_mariano(losses, losses)
        assert result.statistic == 0.0
        assert result.p_value == 1.0
        assert result.n == 50
        assert not result.significant

    def test_clearly_better_method_is_significant(self):
        rng = np.random.default_rng(1)
        loss_b = np.abs(rng.normal(size=200)) + 1.0
        loss_a = loss_b - 1.0 + rng.normal(scale=0.05, size=200)
        result = diebold_mariano(loss_a, loss_b)
        assert result.statistic < 0  # negative favors A
        assert result.p_value < 1e-6
        assert result.significant
        assert result.mean_loss_diff == pytest.approx(-1.0, abs=0.05)

    def test_sign_symmetry(self):
        rng = np.random.default_rng(2)
        loss_a = np.abs(rng.normal(size=100))
        loss_b = np.abs(rng.normal(size=100))
        forward = diebold_mariano(loss_a, loss_b)
        backward = diebold_mariano(loss_b, loss_a)
        assert forward.statistic == pytest.approx(-backward.statistic)
        assert forward.p_value == pytest.approx(backward.p_value)

    def test_multi_step_horizon_widens_variance(self):
        rng = np.random.default_rng(3)
        base = rng.normal(size=300)
        # serially correlated loss diff via moving average
        diff = np.convolve(base, np.ones(6) / 6, mode="same")
        loss_b = np.abs(rng.normal(size=300)) + 1.0
        loss_a = loss_b + diff + 0.05
        one_step = diebold_mariano(loss_a, loss_b, horizon_steps=1)
        six_step = diebold_mariano(loss_a, loss_b, horizon_steps=6)
        assert abs(six_step.statistic) < abs(one_step.statistic)

    def test_noise_is_not_significant(self):
        rng = np.random.default_rng(4)
        loss_a = np.abs(rng.normal(size=100))
        loss_b = np.abs(rng.normal(size=100))
        result = diebold_mariano(loss_a, loss_b)
        assert result.p_value > 0.01

    def test_validation(self):
        with pytest.raises(ValueError, match="at least"):
            diebold_mariano(np.zeros(3), np.zeros(3))
        with pytest.raises(ValueError, match="shape mismatch"):
            diebold_mariano(np.zeros(10), np.zeros(11))
        with pytest.raises(ValueError, match="horizon_steps"):
            diebold_mariano(np.zeros(10), np.zeros(10), horizon_steps=0)
        with pytest.raises(ValueError, match="less than sample count"):
            diebold_mariano(np.zeros(10), np.zeros(10), horizon_steps=10)
