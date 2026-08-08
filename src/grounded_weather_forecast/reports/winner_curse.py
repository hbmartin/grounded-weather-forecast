"""Winner's-curse-corrected reporting for promoted slice winners.

Every slice promotes the argmin-MAE method among the eligible candidates and
reports that winner's MAE — selected on the same data that estimates it, so
the number is biased low, worst where many methods are near-tied. The BH/e-BH
columns control the DISCOVERY error rate; this module corrects the reported
MAGNITUDE, two mutually exclusive estimands side by side:

- **Bootstrap bias** (Efron-Tibshirani plug-in on the min functional): the
  moving-block bootstrap the MCS gate already uses resamples valid times
  jointly across candidates; ``winner_bias = E*[min_k mean*] - min_k mean``
  (<= 0) and ``mae_debiased = mae - winner_bias``. Unconditional: "how much
  does taking a minimum flatter the report here". First-order only at exact
  ties (Fang-Santos), but it grows correctly with the number of near-ties.
- **AKM hybrid** (Andrews-Kitagawa-McCloskey, QJE 2024): condition on WHICH
  method won. With X = negated mean losses ~ N(mu, Sigma), the winner event
  is polyhedral, so the winner's coordinate is truncated normal on a
  closed-form interval; ``mae_hybrid`` is the median-unbiased estimate and
  ``mae_hybrid_upper`` the (1 - alpha) upper bound, both under the hybrid
  scheme (beta = alpha/10 projection guard) that stays finite at near-ties.

Agreement between the two validates the report; divergence beyond the
winner's bootstrap SE is flagged as ``near_tie_flag``. Report-layer only —
the gate, the selection payload, and every raw mae/gap column are unchanged
(the ``bh_adjusted`` convention). Corrections apply only where the served
row was chosen by argmin (``gate`` is null or ``"eligibility"``); gated rows
serve a reference that was not selected on the minimum and carry nulls.

The bootstrap matrix is common-case and valid_time-collapsed while the
reported mae is own-case — the bias is therefore applied as a labeled DELTA
to the reported number, not recomputed from a different sample.
"""

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
from scipy.stats import norm

from grounded_weather_forecast.reports.mcs import (
    _block_indices,
    collapsed_loss_matrix,
)

if TYPE_CHECKING:
    from grounded_weather_forecast.config import PromotionConfig

_SEED = 20260718  # the mcs.py determinism convention
_MIN_TIMES = 8
_DEFAULT_BOOTSTRAP = 500
_DEFAULT_ALPHA = 0.1
_PROJECTION_DRAWS = 2000
_BISECT_STEPS = 80
_DENOMINATOR_FLOOR = 1e-12
_CORRECTED_GATES = (None, "eligibility")

CURSE_COLUMNS: tuple[tuple[str, pl.DataType], ...] = (
    ("winner_bias", pl.Float64()),
    ("mae_debiased", pl.Float64()),
    ("mae_hybrid", pl.Float64()),
    ("mae_hybrid_upper", pl.Float64()),
    ("near_tie_flag", pl.Boolean()),
)


def _truncnorm_cdf(x: float, mu: float, sd: float, lo: float, hi: float) -> float:
    """CDF of N(mu, sd) truncated to [lo, hi]; NaN when degenerate."""
    a = norm.cdf((lo - mu) / sd) if math.isfinite(lo) else 0.0
    b = norm.cdf((hi - mu) / sd) if math.isfinite(hi) else 1.0
    denominator = b - a
    if denominator < _DENOMINATOR_FLOOR:
        return float("nan")
    return float((norm.cdf((x - mu) / sd) - a) / denominator)


def _solve_mu(
    x: float,
    sd: float,
    lo: float,
    hi: float,
    quantile: float,
    *,
    cap: float | None = None,
) -> float | None:
    """mu with F_TN(x; mu, [lo', hi']) = quantile; F is decreasing in mu.

    ``cap`` is the hybrid's projection half-width: the truncation interval is
    additionally intersected with [mu - cap, mu + cap], which keeps the
    estimate within cap of x however tight the winner event's truncation is.
    """
    span = (cap if cap is not None else 10.0 * sd) + 10.0 * sd
    low, high = x - span, x + span
    for _ in range(_BISECT_STEPS):
        mu = (low + high) / 2.0
        lo_mu = max(lo, mu - cap) if cap is not None else lo
        hi_mu = min(hi, mu + cap) if cap is not None else hi
        if lo_mu >= hi_mu:
            value = float("nan")
        else:
            value = _truncnorm_cdf(x, mu, sd, lo_mu, hi_mu)
        if math.isnan(value):
            # Degenerate interval: x outside [lo_mu, hi_mu] means mu is too
            # far on one side; steer by comparison with x.
            if mu < x:
                low = mu
            else:
                high = mu
            continue
        if value > quantile:
            # F too large -> x sits high in the truncated distribution ->
            # mu is too small.
            low = mu
        else:
            high = mu
    mu = (low + high) / 2.0
    return mu if math.isfinite(mu) else None


def _truncation_bounds(
    x: np.ndarray, sigma_row: np.ndarray, winner: int
) -> tuple[float, float]:
    """Closed-form winner-event truncation of the winner's coordinate.

    {X_w >= X_j for all j}: substituting X_j = Z_j + (Sigma_jw/Sigma_ww) X_w
    turns each competitor into a one-sided bound on X_w depending on the
    sign of (Sigma_ww - Sigma_jw).
    """
    sigma_ww = sigma_row[winner]
    lo, hi = -math.inf, math.inf
    for j in range(x.shape[0]):
        if j == winner:
            continue
        z_j = x[j] - (sigma_row[j] / sigma_ww) * x[winner]
        gap = sigma_ww - sigma_row[j]
        if abs(gap) < _DENOMINATOR_FLOOR:
            continue
        bound = sigma_ww * z_j / gap
        if gap > 0.0:
            lo = max(lo, bound)
        else:
            hi = min(hi, bound)
    return lo, hi


def _projection_halfwidth(
    sigma: np.ndarray, winner_sd: float, beta: float, rng: np.random.Generator
) -> float:
    """c_beta * sd_w: the hybrid guard from the projection CI."""
    scales = np.sqrt(np.clip(np.diag(sigma), _DENOMINATOR_FLOOR, None))
    try:
        draws = rng.multivariate_normal(
            np.zeros(sigma.shape[0]), sigma, size=_PROJECTION_DRAWS, method="svd"
        )
    except (np.linalg.LinAlgError, ValueError):
        return float("nan")
    statistic = np.max(np.abs(draws) / scales, axis=1)
    return float(np.quantile(statistic, 1.0 - beta)) * winner_sd


def _slice_correction(
    losses: np.ndarray,
    method_ids: Sequence[str],
    served_method: str,
    *,
    n_bootstrap: int,
    block_length: int | None,
    alpha: float,
) -> dict[str, object]:
    """Both estimators for one slice's collapsed common-case loss matrix."""
    n_times, n_methods = losses.shape
    result: dict[str, object] = dict.fromkeys(
        ("winner_bias", "mae_hybrid_shift", "mae_hybrid_upper_shift"), None
    )
    result["near_tie_flag"] = None
    if n_times < _MIN_TIMES or n_methods < 2:
        return result
    rng = np.random.default_rng(_SEED)
    length = block_length or max(1, round(n_times ** (1.0 / 3.0)))
    indices = _block_indices(rng, n_times, length, n_bootstrap)
    replicate_means = losses[indices].mean(axis=1)  # (B, k)
    column_means = losses.mean(axis=0)
    bias = float(np.mean(replicate_means.min(axis=1)) - column_means.min())
    result["winner_bias"] = bias
    matrix_winner = int(np.argmin(column_means))
    if method_ids[matrix_winner] != served_method:
        # The common-case argmin disagrees with the served (own-case) winner:
        # the conditional-on-identity math would condition on the wrong
        # event. The unconditional bootstrap bias above still applies.
        result["near_tie_flag"] = True
        return result
    sigma = np.cov(replicate_means, rowvar=False)
    sigma = np.atleast_2d(sigma)
    winner_sd = float(math.sqrt(max(sigma[matrix_winner, matrix_winner], 0.0)))
    if winner_sd < _DENOMINATOR_FLOOR:
        return result
    x = -column_means  # maximize: winner = argmax
    lo, hi = _truncation_bounds(x, sigma[matrix_winner], matrix_winner)
    beta = alpha / 10.0
    cap = _projection_halfwidth(sigma, winner_sd, beta, rng)
    if not math.isfinite(cap):
        return result
    adjusted = (alpha - beta) / (1.0 - beta)
    x_w = float(x[matrix_winner])
    median = _solve_mu(x_w, winner_sd, lo, hi, 0.5, cap=cap)
    # An upper bound on the winner's true MAE is a LOWER bound on mu in the
    # negated (maximize) space: the one-sided level uses the hybrid-adjusted
    # size so the guard's beta spend is accounted for.
    lower = _solve_mu(x_w, winner_sd, lo, hi, 1.0 - adjusted, cap=cap)
    if median is not None:
        result["mae_hybrid_shift"] = -median - column_means[matrix_winner]
    if lower is not None:
        result["mae_hybrid_upper_shift"] = -lower - column_means[matrix_winner]
    se = float(np.std(replicate_means[:, matrix_winner]))
    if median is not None:
        debiased_common = column_means[matrix_winner] - bias
        result["near_tie_flag"] = bool(abs(-median - debiased_common) > se)
    return result


def winner_curse_adjusted(
    winners: pl.DataFrame,
    scores: pl.DataFrame | None,
    *,
    promotion: "PromotionConfig | None" = None,
) -> pl.DataFrame:
    """Winners plus the two curse-corrected estimates; gate unchanged.

    ``mae_debiased``/``mae_hybrid``/``mae_hybrid_upper`` are the reported
    own-case mae shifted by deltas estimated on the common-case collapsed
    matrix — a different (smaller) sample, which the delta framing makes
    explicit rather than hiding.
    """
    required = {"product", "variable", "lead_bucket", "method_id", "mae", "gate"}
    if winners.is_empty() or scores is None or not required <= set(winners.columns):
        empty = [
            pl.lit(None, dtype=dtype).alias(name)
            for name, dtype in CURSE_COLUMNS
            if name not in winners.columns
        ]
        return winners.with_columns(empty) if empty else winners
    from grounded_weather_forecast.reports.leaderboard import (  # noqa: PLC0415
        _with_default_semantics,
    )

    normalized = _with_default_semantics(scores)
    alpha = promotion.alpha if promotion is not None else _DEFAULT_ALPHA
    n_bootstrap = (
        promotion.mcs_bootstrap if promotion is not None else _DEFAULT_BOOTSTRAP
    )
    block_length = (
        promotion.mcs_block_length if promotion is not None else None
    ) or None
    rows: list[dict[str, object]] = []
    for row in winners.iter_rows(named=True):
        record: dict[str, object] = dict.fromkeys(
            (name for name, _ in CURSE_COLUMNS), None
        )
        if row.get("gate") in _CORRECTED_GATES and row.get("mae") is not None:
            slice_scores = normalized.filter(
                (pl.col("product") == row["product"])
                & (pl.col("variable") == row["variable"])
                & (pl.col("lead_bucket") == row["lead_bucket"])
            )
            if "truth_semantics" in row and "semantics" in normalized.columns:
                slice_scores = slice_scores.filter(
                    pl.col("semantics") == row["truth_semantics"]
                )
            collapsed = collapsed_loss_matrix(slice_scores)
            if collapsed is not None:
                losses, method_ids = collapsed
                correction = _slice_correction(
                    losses,
                    method_ids,
                    str(row["method_id"]),
                    n_bootstrap=n_bootstrap,
                    block_length=block_length,
                    alpha=alpha,
                )
                mae = float(row["mae"])
                bias = correction["winner_bias"]
                record["winner_bias"] = bias
                if bias is not None:
                    record["mae_debiased"] = mae - float(cast("float", bias))
                if correction["mae_hybrid_shift"] is not None:
                    record["mae_hybrid"] = mae + float(
                        cast("float", correction["mae_hybrid_shift"])
                    )
                if correction["mae_hybrid_upper_shift"] is not None:
                    record["mae_hybrid_upper"] = mae + float(
                        cast("float", correction["mae_hybrid_upper_shift"])
                    )
                record["near_tie_flag"] = correction["near_tie_flag"]
        rows.append(record)
    columns = [
        pl.Series(name, [record[name] for record in rows], dtype=dtype)
        for name, dtype in CURSE_COLUMNS
    ]
    return winners.with_columns(columns)


# One winner-SE of slack before a new argmin evicts a serving incumbent. A
# module constant, not a config key: a new config field would rotate
# config_fingerprint and invalidate every score and release once per tweak.
_RETENTION_SE_MULT = 1.0


def should_retain_incumbent(
    scores: pl.DataFrame | None,
    winner_row: Mapping[str, object],
    incumbent_mae: float,
    *,
    promotion: "PromotionConfig | None" = None,
) -> bool:
    """One-SE incumbent retention: the argmin's near-tie guard.

    Retain when the incumbent's current-board MAE sits within
    ``_RETENTION_SE_MULT`` bootstrap SEs of the winner's — the same
    moving-block bootstrap the bias estimate uses, so "statistically
    indistinguishable" means one thing everywhere. Conservative on any
    missing ingredient: no scores, a thin collapsed matrix, or the winner
    absent from the common-case matrix all mean no retention.
    """
    raw_mae = winner_row.get("mae")
    if scores is None or raw_mae is None:
        return False
    from grounded_weather_forecast.reports.leaderboard import (  # noqa: PLC0415
        _with_default_semantics,
    )

    slice_scores = _with_default_semantics(scores).filter(
        (pl.col("product") == winner_row["product"])
        & (pl.col("variable") == winner_row["variable"])
        & (pl.col("lead_bucket") == winner_row["lead_bucket"])
    )
    if "truth_semantics" in winner_row and "semantics" in slice_scores.columns:
        slice_scores = slice_scores.filter(
            pl.col("semantics") == winner_row["truth_semantics"]
        )
    collapsed = collapsed_loss_matrix(slice_scores)
    if collapsed is None:
        return False
    losses, method_ids = collapsed
    n_times, n_methods = losses.shape
    served = str(winner_row["method_id"])
    if n_times < _MIN_TIMES or n_methods < 2 or served not in method_ids:
        return False
    rng = np.random.default_rng(_SEED)
    n_bootstrap = (
        promotion.mcs_bootstrap if promotion is not None else _DEFAULT_BOOTSTRAP
    )
    block_length = (
        promotion.mcs_block_length if promotion is not None else None
    ) or max(1, round(n_times ** (1.0 / 3.0)))
    indices = _block_indices(rng, n_times, block_length, n_bootstrap)
    replicate_means = losses[indices].mean(axis=1)
    se = float(np.std(replicate_means[:, method_ids.index(served)]))
    if not math.isfinite(se) or se <= 0.0:
        return False
    return float(incumbent_mae) - float(cast("float", raw_mae)) <= (
        _RETENTION_SE_MULT * se
    )


def winner_curse_verdicts(winners: pl.DataFrame) -> dict[str, float]:
    """Scalar diagnostics for the verdicts ledger."""
    if winners.is_empty() or "winner_bias" not in winners.columns:
        return {}
    biases = winners["winner_bias"].drop_nulls()
    if biases.is_empty():
        return {}
    flags = winners["near_tie_flag"].drop_nulls()
    return {
        "winner_bias_mean": float(cast("float", biases.mean())),
        "winner_bias_slices": float(biases.len()),
        "near_tie_slices": float(flags.sum()) if not flags.is_empty() else 0.0,
    }
