"""Probabilistic scores: CRPS, pinball, Brier, reliability, coverage, PIT.

scoringrules is wrapped entirely inside this module so it can be swapped out
in one place if its API moves.
"""

import numpy as np
import polars as pl
import scoringrules as sr

from grounded_weather_forecast.contracts import FloatArray


def pinball_loss(y: FloatArray, quantile_pred: FloatArray, level: float) -> float:
    """Mean pinball (quantile) loss at one level in (0, 1)."""
    if not 0.0 < level < 1.0:
        msg = f"quantile level must be in (0, 1): {level}"
        raise ValueError(msg)
    error = y - quantile_pred
    return float(np.mean(np.maximum(level * error, (level - 1.0) * error)))


def crps_from_quantiles(
    y: FloatArray, quantiles: FloatArray, levels: tuple[float, ...]
) -> float:
    """CRPS approximated from a quantile grid: 2x mean pinball across levels."""
    if quantiles.shape != (y.shape[0], len(levels)):
        msg = f"quantiles shape {quantiles.shape} != ({y.shape[0]}, {len(levels)})"
        raise ValueError(msg)
    losses = [pinball_loss(y, quantiles[:, i], level) for i, level in enumerate(levels)]
    return 2.0 * float(np.mean(losses))


def crps_ensemble(y: FloatArray, ensemble: FloatArray) -> float:
    """Mean CRPS of an ensemble forecast (rows: cases, columns: members)."""
    return float(np.mean(sr.crps_ensemble(y, ensemble)))


def brier(pop: FloatArray, occurred: FloatArray) -> float:
    """Brier score for probability-of-precipitation against binary outcomes."""
    return float(np.mean((pop - occurred) ** 2))


def reliability_bins(
    pop: FloatArray, occurred: FloatArray, n_bins: int = 10
) -> pl.DataFrame:
    """Reliability table: forecast probability vs observed frequency per bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(pop, edges[1:-1]), 0, n_bins - 1)
    frame = pl.DataFrame(
        {"bin": index, "pop": pop, "occurred": occurred.astype(np.float64)}
    )
    stats = (
        frame.group_by("bin")
        .agg(
            pl.col("pop").mean().alias("forecast_mean"),
            pl.col("occurred").mean().alias("observed_freq"),
            pl.len().alias("count"),
        )
        .sort("bin")
    )
    mids = pl.Series("bin_mid", (edges[:-1] + edges[1:]) / 2.0)
    return (
        pl.DataFrame({"bin": np.arange(n_bins), "bin_mid": mids})
        .join(stats, on="bin", how="left")
        .with_columns(pl.col("count").fill_null(0))
    )


def empirical_coverage(y: FloatArray, lower: FloatArray, upper: FloatArray) -> float:
    """Fraction of truths inside [lower, upper]."""
    return float(np.mean((y >= lower) & (y <= upper)))


def pit_from_quantiles(
    y: FloatArray, quantiles: FloatArray, levels: tuple[float, ...]
) -> FloatArray:
    """Approximate PIT values by interpolating truth into the quantile grid."""
    if quantiles.shape != (y.shape[0], len(levels)):
        msg = f"quantiles shape {quantiles.shape} != ({y.shape[0]}, {len(levels)})"
        raise ValueError(msg)
    levels_arr = np.asarray(levels, dtype=np.float64)
    pit = np.empty(y.shape[0], dtype=np.float64)
    for i in range(y.shape[0]):
        pit[i] = np.interp(
            y[i],
            quantiles[i, :],
            levels_arr,
            left=0.0,
            right=1.0,
        )
    return pit
