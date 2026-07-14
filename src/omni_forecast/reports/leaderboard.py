"""Leaderboards over the scores frame.

Reports ALL the views the shipping criterion might use: absolute errors,
relative skill per variable x lead bucket against reference methods (with
Diebold-Mariano significance and visible n), aggregate-vs-reference, and
consumer-style percent-within-tolerance / PoP hit rate. The leaderboard never
pools live and synthetic scores.
"""

from collections.abc import Mapping

import numpy as np
import polars as pl

from omni_forecast.metrics.deterministic import bias, mae, pct_within, rmse
from omni_forecast.metrics.dm import diebold_mariano
from omni_forecast.metrics.probabilistic import brier

DEFAULT_REFERENCES: tuple[str, ...] = ("best_provider", "equal_weight")

# Consumer-legible "close enough" tolerances, in each variable's metric unit.
CONSUMER_TOLERANCES: Mapping[str, float] = {
    "temp_c": 5.0 / 3.0,  # 3 degF
    "dew_point_c": 5.0 / 3.0,
    "temp_max_c": 5.0 / 3.0,
    "temp_min_c": 5.0 / 3.0,
    "humidity_pct": 10.0,
    "wind_speed_ms": 2.0,
    "wind_gust_ms": 3.0,
    "pressure_sea_hpa": 2.0,
    "precip_mm": 1.0,
    "precip_sum_mm": 2.5,
}

_MIN_DM_SAMPLES = 8


def _dm_columns(
    slice_scores: pl.DataFrame,
    method_scores: pl.DataFrame,
    reference: str,
    lead_lo: float,
    product: str,
) -> tuple[float | None, float | None]:
    """Skill and DM p-value of a method against one reference method."""
    reference_scores = slice_scores.filter(pl.col("method_id") == reference)
    if reference_scores.is_empty():
        return None, None
    paired = method_scores.join(
        reference_scores.select("issue_time", "valid_time", "y_pred", "y_true"),
        on=("issue_time", "valid_time"),
        how="inner",
        suffix="_ref",
    ).drop_nulls(["y_pred", "y_pred_ref"])
    if paired.height == 0:
        return None, None
    y = paired["y_true"].to_numpy()
    loss_method = np.abs(paired["y_pred"].to_numpy() - y)
    loss_reference = np.abs(paired["y_pred_ref"].to_numpy() - y)
    reference_mae = float(loss_reference.mean())
    skill = 1.0 - float(loss_method.mean()) / reference_mae if reference_mae else None
    if paired.height < _MIN_DM_SAMPLES:
        return skill, None
    lead_steps = lead_lo / 24.0 if product == "daily" else lead_lo
    horizon_steps = max(1, min(int(lead_steps) + 1, 48, paired.height - 1))
    result = diebold_mariano(loss_method, loss_reference, horizon_steps)
    return skill, result.p_value


def leaderboard(
    scores: pl.DataFrame,
    references: tuple[str, ...] = DEFAULT_REFERENCES,
) -> pl.DataFrame:
    """Per (product, variable, lead bucket, method): every reported view."""
    rows: list[dict[str, object]] = []
    slice_keys = ("product", "variable", "lead_bucket")
    for slice_key, raw_slice_scores in scores.partition_by(
        list(slice_keys), as_dict=True
    ).items():
        product, variable, lead_bucket = (str(part) for part in slice_key)
        methods = raw_slice_scores["method_id"].unique().sort().to_list()
        n_methods = len(methods)
        cases = ["issue_time", "valid_time"]
        common_cases = (
            raw_slice_scores.filter(pl.col("y_pred").is_not_null())
            .group_by(cases)
            .agg(pl.col("method_id").n_unique().alias("available_methods"))
            .filter(pl.col("available_methods") == n_methods)
            .select(cases)
        )
        n_total = raw_slice_scores.select(cases).unique().height
        slice_scores = raw_slice_scores.join(common_cases, on=cases, how="inner")
        if slice_scores.is_empty():
            continue
        lead_lo = float(np.min(slice_scores["lead_hours"].to_numpy()))
        tolerance = CONSUMER_TOLERANCES.get(variable)
        for method_id in methods:
            method_scores = slice_scores.filter(
                pl.col("method_id") == method_id
            ).drop_nulls("y_pred")
            if method_scores.is_empty():
                continue
            pred = method_scores["y_pred"].to_numpy()
            y = method_scores["y_true"].to_numpy()
            row: dict[str, object] = {
                "product": product,
                "variable": variable,
                "lead_bucket": lead_bucket,
                "method_id": method_id,
                "n": method_scores.height,
                "n_total": n_total,
                "coverage": method_scores.height / n_total if n_total else 0.0,
                "mae": mae(pred, y),
                "rmse": rmse(pred, y),
                "bias": bias(pred, y),
                "pct_within": pct_within(pred, y, tolerance)
                if tolerance is not None
                else None,
                "brier": brier(pred, y) if variable == "pop" else None,
            }
            for reference in references:
                skill, p_value = (
                    (None, None)
                    if method_id == reference
                    else _dm_columns(
                        slice_scores,
                        method_scores,
                        reference,
                        lead_lo,
                        product,
                    )
                )
                row[f"skill_vs_{reference}"] = skill
                row[f"dm_p_vs_{reference}"] = p_value
            rows.append(row)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort("product", "variable", "lead_bucket", "mae")


def aggregate_leaderboard(board: pl.DataFrame) -> pl.DataFrame:
    """Headline view: per (product, variable, method), n-weighted overall MAE."""
    if board.is_empty():
        return board
    return (
        board.group_by("product", "variable", "method_id")
        .agg(
            pl.col("n").sum().alias("n"),
            ((pl.col("mae") * pl.col("n")).sum() / pl.col("n").sum()).alias("mae"),
            (
                ((pl.col("rmse") ** 2 * pl.col("n")).sum() / pl.col("n").sum()).sqrt()
            ).alias("rmse"),
        )
        .sort("product", "variable", "mae")
    )


def slice_winners(board: pl.DataFrame) -> pl.DataFrame:
    """Promote a challenger only with coverage and significant reference skill."""
    if board.is_empty():
        return board
    winners: list[dict[str, object]] = []
    keys = ["product", "variable", "lead_bucket"]
    for group in board.partition_by(keys, as_dict=True).values():
        eligible = group.filter((pl.col("coverage") >= 0.8) & (pl.col("n") >= 8))
        ranked = (eligible if not eligible.is_empty() else group).sort("mae")
        candidate = ranked.row(0, named=True)
        references = ranked.filter(pl.col("method_id").is_in(DEFAULT_REFERENCES))
        if (
            candidate["method_id"] not in DEFAULT_REFERENCES
            and not references.is_empty()
        ):
            reference = references.sort("mae").row(0, named=True)
            reference_id = str(reference["method_id"])
            skill = candidate.get(f"skill_vs_{reference_id}")
            p_value = candidate.get(f"dm_p_vs_{reference_id}")
            if skill is None or skill <= 0.0 or p_value is None or p_value >= 0.05:
                candidate = reference
        winners.append(candidate)
    return (
        pl.DataFrame(winners)
        .select("product", "variable", "lead_bucket", "method_id", "n", "mae")
        .sort(*keys)
    )
