"""Point-forecast metrics. All functions take equal-length float64 arrays."""

import numpy as np

from grounded_weather_forecast.contracts import FloatArray


class EmptyScoreError(ValueError):
    """A metric was requested over zero samples."""


def _check(pred: FloatArray, y: FloatArray) -> None:
    if pred.shape != y.shape:
        msg = f"shape mismatch: pred {pred.shape} vs y {y.shape}"
        raise ValueError(msg)
    if pred.size == 0:
        msg = "cannot score zero samples"
        raise EmptyScoreError(msg)


def mae(pred: FloatArray, y: FloatArray) -> float:
    _check(pred, y)
    return float(np.mean(np.abs(pred - y)))


def rmse(pred: FloatArray, y: FloatArray) -> float:
    _check(pred, y)
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def bias(pred: FloatArray, y: FloatArray) -> float:
    """Mean error; positive means the forecast runs high."""
    _check(pred, y)
    return float(np.mean(pred - y))


def mae_skill(pred: FloatArray, y: FloatArray, reference: FloatArray) -> float:
    """1 - MAE/MAE_ref: positive beats the reference, 0 ties, negative loses."""
    reference_mae = mae(reference, y)
    if reference_mae == 0.0:
        return 0.0 if mae(pred, y) == 0.0 else -np.inf
    return 1.0 - mae(pred, y) / reference_mae


def pct_within(pred: FloatArray, y: FloatArray, tolerance: float) -> float:
    """Fraction of forecasts within ``tolerance`` of truth (consumer view)."""
    _check(pred, y)
    return float(np.mean(np.abs(pred - y) <= tolerance))


def pop_hit_rate(
    pop: FloatArray, occurred: FloatArray, threshold: float = 0.5
) -> float:
    """Fraction of correct rain/no-rain calls at a probability threshold."""
    _check(pop, occurred)
    return float(np.mean((pop >= threshold) == (occurred > 0.5)))
