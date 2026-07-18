"""Model Confidence Set: the honest replacement for argmin-MAE promotion.

Hansen, Lunde & Nason (2011): instead of crowning whichever method's sample
MAE is lowest — a winner's-curse machine on a 20+-method x 10-bucket grid —
compute the *set* of methods statistically indistinguishable from the best.
Thin data yields a large set, which is the correct answer: keep the
incumbent. Losses arrive already collapsed to one value per valid_time (the
same pseudo-replication fix the DM test uses), and the bootstrap resamples
moving blocks of that time series so serial dependence survives resampling.

This is the t-max variant with a moving-block bootstrap — the compact core
of the published procedure, not a wrapper around a heavyweight dependency.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from grounded_weather_forecast.contracts import FloatArray

_DEFAULT_BOOTSTRAP = 500
_MIN_TIMES = 8
_SEED = 20260718


@dataclass(frozen=True, slots=True)
class McsResult:
    """Surviving method ids and the elimination p-value trail."""

    survivors: tuple[str, ...]
    p_values: dict[str, float]

    def contains(self, method_id: str) -> bool:
        return method_id in self.survivors


def _block_indices(
    rng: np.random.Generator, n_times: int, block_length: int, n_bootstrap: int
) -> np.ndarray:
    n_blocks = int(np.ceil(n_times / block_length))
    starts = rng.integers(0, n_times, size=(n_bootstrap, n_blocks))
    offsets = np.arange(block_length)
    indices = (starts[:, :, np.newaxis] + offsets) % n_times
    return indices.reshape(n_bootstrap, -1)[:, :n_times]


def model_confidence_set(
    losses: FloatArray,
    method_ids: tuple[str, ...],
    alpha: float = 0.1,
    n_bootstrap: int = _DEFAULT_BOOTSTRAP,
) -> McsResult:
    """MCS over a (n_times, n_methods) loss matrix (one row per valid time).

    Returns every method the data cannot distinguish from the best at level
    ``alpha``. With fewer than ``_MIN_TIMES`` rows nothing can be eliminated
    and all methods survive — thinness is never evidence.
    """
    n_times, n_methods = losses.shape
    if n_methods != len(method_ids):
        msg = f"{n_methods} loss columns for {len(method_ids)} method ids"
        raise ValueError(msg)
    if n_times < _MIN_TIMES or n_methods < 2:
        return McsResult(survivors=tuple(method_ids), p_values={})
    rng = np.random.default_rng(_SEED)
    block_length = max(1, round(n_times ** (1.0 / 3.0)))
    indices = _block_indices(rng, n_times, block_length, n_bootstrap)
    active = list(range(n_methods))
    p_values: dict[str, float] = {}
    mcs_p = 0.0
    while len(active) > 1:
        subset = losses[:, active]
        relative = subset - subset.mean(axis=1, keepdims=True)
        means = relative.mean(axis=0)
        boot_means = relative[indices].mean(axis=1)  # (B, k)
        deviations = boot_means - means
        std = np.sqrt(np.mean(deviations**2, axis=0))
        zero_variance = std <= 0.0
        deterministic = np.where(
            means > 0.0, np.inf, np.where(means < 0.0, -np.inf, 0.0)
        )
        t_stats = np.divide(
            means,
            std,
            out=deterministic,
            where=~zero_variance,
        )
        t_max = float(t_stats.max())
        boot_t = np.divide(
            deviations,
            std,
            out=np.zeros_like(deviations),
            where=~zero_variance,
        )
        boot_t_max = boot_t.max(axis=1)
        p_raw = float(np.mean(boot_t_max >= t_max))
        # elimination p-values are non-decreasing along the sequence
        mcs_p = max(mcs_p, p_raw)
        if mcs_p >= alpha:
            break
        worst = int(np.argmax(t_stats))
        p_values[method_ids[active[worst]]] = mcs_p
        active.pop(worst)
    survivors = tuple(method_ids[index] for index in active)
    return McsResult(survivors=survivors, p_values=p_values)


def collapsed_loss_matrix(
    slice_scores: pl.DataFrame,
    *,
    method_ids: tuple[str, ...] | None = None,
) -> tuple[FloatArray, tuple[str, ...]] | None:
    """Common-case losses collapsed per valid_time, methods as columns.

    Only cases every method scored enter (MCS compares the set, so its inputs
    must be common) — the leaderboard's own-case scoring is unaffected.
    """
    frame = slice_scores.drop_nulls("y_pred")
    if method_ids is not None:
        frame = frame.filter(pl.col("method_id").is_in(method_ids))
    methods = tuple(sorted(frame["method_id"].unique().to_list()))
    if len(methods) < 2:
        return None
    cases = (
        frame.group_by("issue_time", "valid_time")
        .agg(pl.col("method_id").n_unique().alias("k"))
        .filter(pl.col("k") == len(methods))
        .select("issue_time", "valid_time")
    )
    common = frame.join(cases, on=["issue_time", "valid_time"], how="inner")
    if common.is_empty():
        return None
    collapsed = (
        common.with_columns((pl.col("y_pred") - pl.col("y_true")).abs().alias("loss"))
        .group_by("valid_time", "method_id")
        .agg(pl.col("loss").mean())
        .pivot(on="method_id", index="valid_time", values="loss")
        .sort("valid_time")
        .drop_nulls()
    )
    if collapsed.is_empty():
        return None
    matrix = collapsed.select(list(methods)).to_numpy().astype(np.float64)
    return matrix, methods
