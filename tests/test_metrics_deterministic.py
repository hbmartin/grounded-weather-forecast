import numpy as np
import pytest

from grounded_weather_forecast.metrics.deterministic import (
    EmptyScoreError,
    bias,
    mae,
    mae_skill,
    pct_within,
    pop_hit_rate,
    rmse,
)

PRED = np.array([1.0, 2.0, 3.0, 4.0])
Y = np.array([1.0, 1.0, 5.0, 4.0])  # errors: 0, +1, -2, 0


class TestGoldens:
    def test_mae(self):
        assert mae(PRED, Y) == pytest.approx(0.75)

    def test_rmse(self):
        assert rmse(PRED, Y) == pytest.approx(np.sqrt(5.0 / 4.0))

    def test_bias(self):
        assert bias(PRED, Y) == pytest.approx(-0.25)

    def test_pct_within(self):
        assert pct_within(PRED, Y, 0.0) == pytest.approx(0.5)
        assert pct_within(PRED, Y, 1.0) == pytest.approx(0.75)
        assert pct_within(PRED, Y, 2.0) == pytest.approx(1.0)


class TestSkill:
    def test_positive_when_better(self):
        reference = np.array([0.0, 0.0, 0.0, 0.0])
        assert mae_skill(PRED, Y, reference) > 0.0

    def test_zero_against_self(self):
        assert mae_skill(PRED, Y, PRED) == pytest.approx(0.0)

    def test_perfect_reference(self):
        assert mae_skill(PRED, Y, Y) == -np.inf
        assert mae_skill(Y, Y, Y) == 0.0


class TestPopHitRate:
    def test_hit_rate(self):
        pop = np.array([0.9, 0.2, 0.7, 0.1])
        occurred = np.array([1.0, 0.0, 0.0, 1.0])
        assert pop_hit_rate(pop, occurred) == pytest.approx(0.5)


class TestValidation:
    def test_empty_raises(self):
        with pytest.raises(EmptyScoreError):
            mae(np.array([]), np.array([]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            rmse(np.zeros(3), np.zeros(4))
