"""Leaderboards over the scores frame.

Reports ALL the views the shipping criterion might use: absolute errors,
relative skill per variable x lead bucket against reference methods (with
Diebold-Mariano significance and visible n), aggregate-vs-reference, and
consumer-style percent-within-tolerance / PoP hit rate. The leaderboard never
pools live and synthetic scores.
"""

import json
from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl
from scipy import stats

from grounded_weather_forecast.metrics.deterministic import bias, mae, pct_within, rmse
from grounded_weather_forecast.metrics.dm import diebold_mariano
from grounded_weather_forecast.metrics.probabilistic import (
    brier,
    crps_from_quantiles,
    empirical_coverage,
    pinball_loss,
    pit_from_quantiles,
)

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
_MIN_PIT_SAMPLES = 50
_PIT_BINS = 10
_PROBABILISTIC_EMPTY: Mapping[str, float | None] = {
    "crps": None,
    "pinball": None,
    "coverage80": None,
    "coverage90": None,
    "pit_chi2_p": None,
    "sharpness": None,
}


def _level_index(levels: Sequence[float], target: float) -> int | None:
    matches = np.flatnonzero(
        np.isclose(np.asarray(levels), target, rtol=0.0, atol=1e-9)
    )
    return int(matches[0]) if matches.size else None


def _interval_metrics(
    y: np.ndarray,
    grids: np.ndarray,
    levels: Sequence[float],
    lower: float,
    upper: float,
) -> tuple[float | None, float | None]:
    lower_index = _level_index(levels, lower)
    upper_index = _level_index(levels, upper)
    if lower_index is None or upper_index is None:
        return None, None
    coverage = empirical_coverage(
        y,
        grids[:, lower_index],
        grids[:, upper_index],
    )
    sharpness = float(np.mean(grids[:, upper_index] - grids[:, lower_index]))
    return coverage, sharpness


def _probabilistic_columns(method_scores: pl.DataFrame) -> dict[str, float | None]:
    """CRPS/pinball/coverage/PIT/sharpness from stored quantile grids.

    Null for point-only methods. Wired per the improvement program: the
    metrics were implemented and tested long before anything emitted a
    distribution; this is where they finally reach the leaderboard.
    """
    empty = dict(_PROBABILISTIC_EMPTY)
    if "quantiles_json" not in method_scores.columns:
        return empty
    with_quantiles = method_scores.drop_nulls("quantiles_json")
    if with_quantiles.is_empty():
        return empty
    levels = tuple(json.loads(with_quantiles["quantile_levels_json"][0]))
    if not levels:
        return empty
    grids = np.asarray(
        [
            [np.nan if value is None else float(value) for value in json.loads(row)]
            for row in with_quantiles["quantiles_json"].to_list()
        ]
    )
    y = with_quantiles["y_true"].to_numpy().astype(np.float64)
    usable = np.isfinite(grids).all(axis=1) & np.isfinite(y)
    if int(usable.sum()) < _MIN_DM_SAMPLES:
        return empty
    grids, y = grids[usable], y[usable]
    pinball = float(
        np.mean([pinball_loss(y, grids[:, i], level) for i, level in enumerate(levels)])
    )
    pit = pit_from_quantiles(y, grids, tuple(levels))
    counts, _ = np.histogram(pit, bins=_PIT_BINS, range=(0.0, 1.0))
    coverage80, sharpness = _interval_metrics(y, grids, levels, 0.1, 0.9)
    coverage90, _ = _interval_metrics(y, grids, levels, 0.05, 0.95)
    return {
        "crps": crps_from_quantiles(y, grids, levels),
        "pinball": pinball,
        "coverage80": coverage80,
        "coverage90": coverage90,
        "pit_chi2_p": (
            float(stats.chisquare(counts).pvalue)
            if y.shape[0] >= _MIN_PIT_SAMPLES
            else None
        ),
        "sharpness": sharpness,
    }


def _collapsed_losses(paired: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """One mean absolute loss per valid_time, in temporal order.

    Dozens of consecutive snapshots forecast the same valid hour, so raw
    per-row losses are massively pseudo-replicated; collapsing makes the DM
    variance see the real effective sample and its true autocorrelation axis.
    """
    collapsed = (
        paired.group_by("valid_time")
        .agg(
            pl.col("loss_method").mean(),
            pl.col("loss_reference").mean(),
        )
        .sort("valid_time")
    )
    return (
        collapsed["loss_method"].to_numpy(),
        collapsed["loss_reference"].to_numpy(),
    )


def _dm_columns(
    slice_scores: pl.DataFrame,
    method_scores: pl.DataFrame,
    reference: str,
    lead_lo: float,
    product: str,
) -> tuple[float | None, float | None]:
    """Skill and DM p-value of a method against one reference method.

    Compared on pairwise-common cases only, then collapsed per valid_time.
    """
    reference_scores = slice_scores.filter(pl.col("method_id") == reference)
    if reference_scores.is_empty():
        return None, None
    paired = (
        method_scores.join(
            reference_scores.select("issue_time", "valid_time", "y_pred", "y_true"),
            on=("issue_time", "valid_time"),
            how="inner",
            suffix="_ref",
        )
        .drop_nulls(["y_pred", "y_pred_ref"])
        .with_columns(
            (pl.col("y_pred") - pl.col("y_true")).abs().alias("loss_method"),
            (pl.col("y_pred_ref") - pl.col("y_true")).abs().alias("loss_reference"),
        )
    )
    if paired.height == 0:
        return None, None
    loss_method, loss_reference = _collapsed_losses(paired)
    reference_mae = float(loss_reference.mean())
    skill = 1.0 - float(loss_method.mean()) / reference_mae if reference_mae else None
    if loss_method.shape[0] < _MIN_DM_SAMPLES:
        return skill, None
    lead_steps = lead_lo / 24.0 if product == "daily" else lead_lo
    horizon_steps = max(1, min(int(lead_steps) + 1, 48, loss_method.shape[0] - 1))
    result = diebold_mariano(loss_method, loss_reference, horizon_steps)
    return skill, result.p_value


def leaderboard(
    scores: pl.DataFrame,
    references: tuple[str, ...] = DEFAULT_REFERENCES,
) -> pl.DataFrame:
    """Per (product, variable, lead bucket, method): every reported view."""
    rows: list[dict[str, object]] = []
    slice_keys = ("product", "variable", "lead_bucket")
    cases = ["issue_time", "valid_time"]
    for slice_key, slice_scores in scores.partition_by(
        list(slice_keys), as_dict=True
    ).items():
        product, variable, lead_bucket = (str(part) for part in slice_key)
        methods = slice_scores["method_id"].unique().sort().to_list()
        n_total = slice_scores.select(cases).unique().height
        lead_lo = float(np.min(slice_scores["lead_hours"].to_numpy()))
        tolerance = CONSUMER_TOLERANCES.get(variable)
        for method_id in methods:
            # Each method is scored on its own non-null cases; only the DM
            # comparison inside _dm_columns restricts to pairwise-common ones.
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
                "n_valid_times": method_scores["valid_time"].n_unique(),
                "coverage": method_scores.height / n_total if n_total else 0.0,
                "mae": mae(pred, y),
                "rmse": rmse(pred, y),
                "bias": bias(pred, y),
                "pct_within": pct_within(pred, y, tolerance)
                if tolerance is not None
                else None,
                "brier": brier(pred, y) if variable == "pop" else None,
                **_probabilistic_columns(method_scores),
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
    # Nullable metric columns (pct_within, brier, skill/DM p-values) may hold
    # None for every early row; scan all rows so a late float still unifies.
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        "product", "variable", "lead_bucket", "mae"
    )


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


def _legacy_gate(
    candidate: dict[str, object], reference: dict[str, object]
) -> dict[str, object]:
    reference_id = str(reference["method_id"])
    skill = candidate.get(f"skill_vs_{reference_id}")
    p_value = candidate.get(f"dm_p_vs_{reference_id}")
    strong = (
        isinstance(skill, (int, float))
        and skill > 0.0
        and isinstance(p_value, (int, float))
        and p_value < 0.05
    )
    return candidate if strong else reference


def _reference_fallback(
    references: tuple[dict[str, object], ...],
) -> dict[str, object]:
    """The named serving incumbent, or the best remaining reference."""
    for reference in references:
        if reference["method_id"] == "equal_weight":
            return reference

    def row_mae(row: dict[str, object]) -> float:
        value = row["mae"]
        return float(value) if isinstance(value, (int, float)) else float("inf")

    return min(references, key=row_mae)


def _mcs_gate(
    candidate: dict[str, object],
    references: tuple[dict[str, object], ...],
    slice_scores: pl.DataFrame,
    eligible_methods: tuple[str, ...],
    alpha: float,
) -> dict[str, object]:
    """Promote only when every reference is excluded from the MCS.

    Sparse, ineligible methods are deliberately excluded before constructing
    the common-case matrix. Thin data never falls back to a more permissive
    test: the best eligible reference continues serving.
    """
    from grounded_weather_forecast.reports.mcs import (  # noqa: PLC0415
        collapsed_loss_matrix,
        model_confidence_set,
    )

    fallback = _reference_fallback(references)
    built = collapsed_loss_matrix(slice_scores, method_ids=eligible_methods)
    if built is None:
        return fallback
    matrix, methods = built
    if matrix.shape[0] < _MIN_DM_SAMPLES:
        return fallback
    result = model_confidence_set(matrix, methods, alpha=alpha)
    candidate_id = str(candidate["method_id"])
    reference_ids = tuple(str(reference["method_id"]) for reference in references)
    if result.contains(candidate_id) and all(
        not result.contains(reference_id) for reference_id in reference_ids
    ):
        return candidate
    return fallback


def slice_winners(
    board: pl.DataFrame,
    scores: pl.DataFrame | None = None,
    rule: str = "legacy",
    alpha: float = 0.1,
) -> pl.DataFrame:
    """Promote a challenger only past the configured statistical gate.

    ``rule="mcs"`` (with the raw ``scores``) uses the Model Confidence Set;
    ``"legacy"`` keeps the single-DM gate. Coverage and effective-n gates
    apply either way.
    """
    if board.is_empty():
        return board
    winners: list[dict[str, object]] = []
    keys = ["product", "variable", "lead_bucket"]
    for slice_key, group in board.partition_by(keys, as_dict=True).items():
        eligible = group.filter(
            (pl.col("coverage") >= 0.8)
            & (pl.col("n") >= 8)
            & (pl.col("n_valid_times") >= 8)
        )
        if eligible.is_empty():
            continue
        ranked = eligible.sort("mae")
        candidate = ranked.row(0, named=True)
        reference_rows = tuple(
            ranked.filter(pl.col("method_id").is_in(DEFAULT_REFERENCES))
            .sort("mae")
            .iter_rows(named=True)
        )
        if candidate["method_id"] not in DEFAULT_REFERENCES:
            present_references = {
                str(reference["method_id"]) for reference in reference_rows
            }
            if not set(DEFAULT_REFERENCES) <= present_references:
                if reference_rows:
                    candidate = _reference_fallback(reference_rows)
                else:
                    continue
            elif rule == "mcs" and scores is not None:
                product, variable, lead_bucket = slice_key
                slice_scores = scores.filter(
                    (pl.col("product") == product)
                    & (pl.col("variable") == variable)
                    & (pl.col("lead_bucket") == lead_bucket)
                )
                candidate = _mcs_gate(
                    candidate,
                    reference_rows,
                    slice_scores,
                    tuple(str(method) for method in ranked["method_id"].to_list()),
                    alpha,
                )
            elif any(
                _legacy_gate(candidate, reference) is not candidate
                for reference in reference_rows
            ):
                candidate = _reference_fallback(reference_rows)
        winners.append(candidate)
    if not winners:
        return board.select(
            "product", "variable", "lead_bucket", "method_id", "n", "mae"
        ).head(0)
    return (
        pl.DataFrame(winners)
        .select("product", "variable", "lead_bucket", "method_id", "n", "mae")
        .sort(*keys)
    )
