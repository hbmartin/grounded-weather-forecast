"""Frozen October 26 candidate functions, copied verbatim; no registry changes.

Original experiment source-file SHA256 (before extraction into this module):
962d0a9c21d3e0e1d1babaf562635fd2e64a559ab27e6909fe0aba1caa327c28
Only temperature simplex_decay21 / decay21 are enrolled by the shadow runner.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
from scipy.optimize import minimize

from grounded_weather_forecast.blenders.protocol import masked_average
from grounded_weather_forecast.contracts import (
    FloatArray,
    TruthSemantics,
    hourly_variable,
)
from grounded_weather_forecast.dataset.matrix import to_supervised_slice

LEAD_EDGES = (0.0, 6.0, 24.0, 72.0, 168.0, 360.0)
SHRINK_DAYS = 7.0


def groups(leads: FloatArray) -> np.ndarray:
    return np.clip(np.digitize(leads, LEAD_EDGES[1:-1]), 0, 4)


def safe_blend(values: FloatArray, weights: FloatArray) -> FloatArray:
    return masked_average(values, np.isfinite(values), weights)


def weighted_bias(values: FloatArray, y: FloatArray, weights: FloatArray) -> FloatArray:
    errors = values - y[:, None]
    finite = np.isfinite(errors)
    denom = (finite * weights[:, None]).sum(axis=0)
    return np.divide(
        np.where(finite, errors, 0.0).T @ weights,
        denom,
        out=np.zeros(values.shape[1]),
        where=denom > 0,
    )


def bias_corrections(
    train: pl.DataFrame, variable: str, origin: datetime, method: str
) -> tuple[FloatArray, FloatArray]:
    """Calendar-time decay; one vote per unique valid hour in each lead group."""
    spec = hourly_variable(variable)
    s = to_supervised_slice(train, spec, semantics=semantics(variable))
    age = (origin.timestamp() - train["valid_time"].dt.epoch("s").to_numpy()) / 86400.0
    half_life = 42.0 if method == "decay42" else 21.0
    weights = np.exp2(-age / half_life)
    if method == "rolling21":
        weights = (age <= 21).astype(float)
    # Each hour's repeated lead-group appearances get equal total global weight.
    multiplicity = train.select(pl.len().over("valid_time").alias("n"))["n"].to_numpy()
    global_weights = weights / multiplicity
    global_bias = weighted_bias(s.x.values, s.y, global_weights)
    correction = np.zeros((5, len(s.x.sources)))
    for group in range(5):
        mask = groups(s.x.lead_hours) == group
        local = weighted_bias(s.x.values[mask], s.y[mask], weights[mask])
        days = np.array(
            [
                train.filter(
                    pl.Series(mask & (weights > 0) & np.isfinite(s.x.values[:, j]))
                )["valid_time"]
                .dt.date()
                .n_unique()
                for j in range(len(s.x.sources))
            ],
            dtype=float,
        )
        strength = days / (days + SHRINK_DAYS)
        correction[group] = strength * local + (1 - strength) * global_bias
    return correction, weights


def fit_simplex(
    values: FloatArray, y: FloatArray, row_weights: FloatArray
) -> FloatArray:
    """Constrained loss uses the same availability renormalization as inference."""
    n_sources = values.shape[1]
    equal = np.full(n_sources, 1 / n_sources)
    usable = np.isfinite(values).any(axis=1) & (row_weights > 0)
    if usable.sum() < 20:
        return equal
    scale = max(float(np.nanstd(y[usable])), 0.1)

    def objective(w: FloatArray) -> float:
        pred = safe_blend(values[usable], w)
        loss = np.average(
            ((pred - y[usable]) / scale) ** 2, weights=row_weights[usable]
        )
        return float(loss + 2.0 * np.sum((w - equal) ** 2))

    fit = minimize(
        objective,
        equal,
        method="SLSQP",
        bounds=[(1e-4, 1.0)] * n_sources,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"maxiter": 100, "ftol": 1e-8},
    )
    # Additional fixed 50% shrinkage toward equal even after penalized fitting.
    return 0.5 * fit.x + 0.5 * equal if fit.success else equal


def semantics(variable: str) -> TruthSemantics:
    return (
        TruthSemantics.INTERVAL_MEAN
        if variable in {"temp_c", "humidity_pct", "wind_speed_ms"}
        else TruthSemantics.INSTANTANEOUS
    )


def truth_column(variable: str) -> str:
    suffix = "__mean" if variable in {"temp_c", "humidity_pct", "wind_speed_ms"} else ""
    return "t__" + variable + suffix


def independent_training(frame: pl.DataFrame) -> pl.DataFrame:
    """Suppress repeated same-valid-time rows within each broad lead group."""
    return (
        frame.with_columns(
            pl.Series("broad_group", groups(frame["lead_hours"].to_numpy()))
        )
        .sort("issue_time")
        .unique(subset=["valid_time", "broad_group"], keep="last")
        .sort("issue_time", "valid_time")
        .drop("broad_group")
    )


def rolling_climate(
    train: pl.DataFrame, test: pl.DataFrame, variable: str, origin: datetime
) -> FloatArray:
    known = train.unique("valid_time").filter(
        pl.col("valid_time") >= origin - timedelta(days=30)
    )
    if known.is_empty():
        known = train.unique("valid_time")
    reduce = np.mean if variable == "pop" else np.median
    target = truth_column(variable)
    global_value = float(reduce(known[target].to_numpy()))
    point = np.full(test.height, global_value)
    for hour in range(8):
        pool = known.filter(pl.col("valid_hour_local") // 3 == hour)
        if pool["valid_time"].dt.date().n_unique() >= 3:
            point[test["valid_hour_local"].to_numpy() // 3 == hour] = reduce(
                pool[target].to_numpy()
            )
    return point


def observed_climate(
    truth: pl.DataFrame, test: pl.DataFrame, variable: str, origin: datetime
) -> FloatArray:
    """Stronger baseline: all fully known observations, including nonforecast days."""
    known = (
        truth.rename({"valid_hour": "valid_time"})
        .filter(
            (pl.col("valid_time") + timedelta(hours=2) <= origin)
            & pl.col(truth_column(variable)).is_finite()
        )
        .with_columns(
            pl.col("valid_time")
            .dt.convert_time_zone("America/Los_Angeles")
            .dt.hour()
            .alias("valid_hour_local")
        )
    )
    future = test.with_columns(
        pl.col("valid_time")
        .dt.convert_time_zone("America/Los_Angeles")
        .dt.hour()
        .alias("valid_hour_local")
    )
    return rolling_climate(known, future, variable, origin)
