"""Diebold-Mariano equal-predictive-accuracy test with the HLN correction.

Compares two aligned loss series (e.g. absolute errors of two methods on the
same cases). Uses a Bartlett-kernel HAC variance so multi-step forecast
horizons with serially correlated loss differentials are handled, and the
Harvey-Leybourne-Newbold small-sample correction with a Student-t reference.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from grounded_weather_forecast.contracts import FloatArray

MIN_SAMPLES = 8


@dataclass(frozen=True, slots=True)
class DMResult:
    statistic: float
    p_value: float
    n: int
    mean_loss_diff: float

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def _bartlett_hac_variance(diff: FloatArray, horizon_steps: int) -> float:
    n = diff.shape[0]
    centered = diff - diff.mean()
    gamma0 = float(centered @ centered) / n
    variance = gamma0
    max_lag = min(horizon_steps - 1, n - 1)
    for lag in range(1, max_lag + 1):
        gamma = float(centered[lag:] @ centered[:-lag]) / n
        weight = 1.0 - lag / horizon_steps
        variance += 2.0 * weight * gamma
    return max(variance, 0.0) / n


def _hln_factor(n: int, horizon_steps: int) -> float:
    h = horizon_steps
    adjusted = (n + 1 - 2 * h + h * (h - 1) / n) / n
    return float(np.sqrt(max(adjusted, 0.0)))


def diebold_mariano(
    loss_a: FloatArray, loss_b: FloatArray, horizon_steps: int = 1
) -> DMResult:
    """Test H0: equal expected loss. Negative statistic favors method A."""
    if loss_a.shape != loss_b.shape:
        msg = f"shape mismatch: {loss_a.shape} vs {loss_b.shape}"
        raise ValueError(msg)
    if horizon_steps < 1:
        msg = f"horizon_steps must be >= 1: {horizon_steps}"
        raise ValueError(msg)
    n = loss_a.shape[0]
    if n < MIN_SAMPLES:
        msg = f"need at least {MIN_SAMPLES} paired losses, got {n}"
        raise ValueError(msg)
    if horizon_steps >= n:
        msg = f"horizon_steps must be less than sample count {n}: {horizon_steps}"
        raise ValueError(msg)
    diff = loss_a - loss_b
    mean_diff = float(diff.mean())
    variance = _bartlett_hac_variance(diff, horizon_steps)
    if variance == 0.0:
        statistic = 0.0 if mean_diff == 0.0 else float(np.sign(mean_diff) * np.inf)
        p_value = 1.0 if mean_diff == 0.0 else 0.0
        return DMResult(statistic, p_value, n, mean_diff)
    statistic = mean_diff / float(np.sqrt(variance)) * _hln_factor(n, horizon_steps)
    p_value = 2.0 * float(stats.t.sf(abs(statistic), df=n - 1))
    return DMResult(statistic, p_value, n, mean_diff)
