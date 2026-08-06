"""Leaderboards over the scores frame.

Reports ALL the views the shipping criterion might use: absolute errors,
relative skill per variable x lead bucket against reference methods (with
Diebold-Mariano significance and visible n), aggregate-vs-reference, and
consumer-style percent-within-tolerance / PoP hit rate. The leaderboard never
pools live and synthetic scores.
"""

import json
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy import stats

from grounded_weather_forecast.contracts import TruthSemantics

if TYPE_CHECKING:
    from grounded_weather_forecast.config import PromotionConfig
    from grounded_weather_forecast.reports.eprocess import EProcessStore
from grounded_weather_forecast.metrics.deterministic import bias, mae, pct_within, rmse
from grounded_weather_forecast.metrics.dm import diebold_mariano
from grounded_weather_forecast.metrics.probabilistic import (
    brier,
    crps_ensemble,
    empirical_coverage,
    pinball_loss,
    pit_from_quantiles,
)

# The safety class: a candidate serves only by excluding every reference
# from the confidence set, and gate failures fall back to the best reference.
# damped_grounded_equal_weight joined 2026-08-04 after one clean promoted
# cycle — it contains equal_weight as its alpha -> 1 boundary and stays safe
# at long leads where the raw mean provably is not (the week-2 bias episode,
# research/week2-provider-diagnostic-2026-08-04.md).
DEFAULT_REFERENCES: tuple[str, ...] = (
    "best_provider",
    "equal_weight",
    "damped_grounded_equal_weight",
)


def gate_references(
    variable: str, promotion: "PromotionConfig | None"
) -> tuple[str, ...]:
    """The reference class the gate holds this variable's candidates against.

    ``[promotion.references]`` overrides per variable: where the raw
    provider-space references are bias-dominated against station truth
    (pressure's ~26 hPa console offset; the sheltered anemometer), grounded
    variants make the gate and its fallback meaningful again.
    """
    if promotion is not None:
        override = promotion.references.get(variable)
        if override:
            return tuple(override)
    return DEFAULT_REFERENCES


def union_references(promotion: "PromotionConfig | None") -> tuple[str, ...]:
    """Defaults plus every configured override, defaults-first and deduped.

    ``leaderboard()`` computes skill/DM columns for this union so the pinned
    ``skill_vs_equal_weight``-style columns never change meaning while
    overridden variables still get columns for their own references.
    """
    ordered = list(DEFAULT_REFERENCES)
    if promotion is not None:
        for override in promotion.references.values():
            ordered.extend(method for method in override if method not in ordered)
    return tuple(ordered)


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


def _with_default_semantics(scores: pl.DataFrame) -> pl.DataFrame:
    """Treat missing semantic values as the legacy instantaneous target."""
    if "semantics" not in scores.columns:
        return scores
    return scores.with_columns(
        pl.col("semantics")
        .fill_null(TruthSemantics.INSTANTANEOUS.value)
        .alias("semantics")
    )


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
        # Energy-form ensemble CRPS with the sorted quantile grid as the
        # members: mean|X - y| - mean|X - X'| / 2. Unlike the weighted-pinball
        # rectangle rule it needs no level spacing assumptions; rows already
        # dropped above (no quantiles, non-finite grid) stay null.
        "crps": crps_ensemble(y, np.sort(grids, axis=1)),
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
    scores = _with_default_semantics(scores)
    if "lead_bucket" in scores.columns:
        # Historical score files may carry rows past the last bucket edge
        # (14-16-day provider dailies before the matrix-level filter landed);
        # a null-bucket slice can never serve and only pollutes the board.
        scores = scores.filter(pl.col("lead_bucket").is_not_null())
    rows: list[dict[str, object]] = []
    # Semantics joins the slice identity so a concatenated frame cannot pool
    # two truth targets into one row; per-evaluation frames are unaffected.
    slice_keys = (
        ("product", "variable", "semantics", "lead_bucket")
        if "semantics" in scores.columns
        else ("product", "variable", "lead_bucket")
    )
    cases = ["issue_time", "valid_time"]
    for slice_key, slice_scores in scores.partition_by(
        list(slice_keys), as_dict=True
    ).items():
        parts = dict(zip(slice_keys, (str(part) for part in slice_key), strict=True))
        product = parts["product"]
        variable = parts["variable"]
        lead_bucket = parts["lead_bucket"]
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
                "truth_semantics": parts.get(
                    "semantics", TruthSemantics.INSTANTANEOUS.value
                ),
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
    group_columns = ["product", "variable"]
    if "truth_semantics" in board.columns:
        group_columns.append("truth_semantics")
    group_columns.append("method_id")
    return (
        board.group_by(*group_columns)
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
    """The best reference by MAE; equal_weight only breaks exact ties.

    Preferring the incumbent by name defeated the point of a multi-member
    safety class: with damped_grounded_equal_weight as a reference, a gate
    failure at long leads should serve the damped blend, not the raw mean
    whose shared provider bias motivated it.
    """

    def rank(row: dict[str, object]) -> tuple[float, int]:
        value = row["mae"]
        mae = float(value) if isinstance(value, (int, float)) else float("inf")
        return (mae, 0 if row["method_id"] == "equal_weight" else 1)

    return min(references, key=rank)


def _mcs_gate(
    candidate: dict[str, object],
    references: tuple[dict[str, object], ...],
    slice_scores: pl.DataFrame,
    eligible_methods: tuple[str, ...],
    alpha: float,
    n_bootstrap: int = 500,
    block_length: int | None = None,
) -> tuple[dict[str, object], str | None]:
    """Promote only when every reference is excluded from the MCS.

    Sparse, ineligible methods are deliberately excluded before constructing
    the common-case matrix. Thin data never falls back to a more permissive
    test: the best eligible reference continues serving. The second element
    names which outcome blocked the candidate (None when it promoted), so
    the report can say why a slice serves above its board minimum.
    """
    from grounded_weather_forecast.reports.mcs import (  # noqa: PLC0415
        collapsed_loss_matrix,
        model_confidence_set,
    )

    fallback = _reference_fallback(references)
    built = collapsed_loss_matrix(slice_scores, method_ids=eligible_methods)
    if built is None:
        return fallback, "mcs_no_matrix"
    matrix, methods = built
    if matrix.shape[0] < _MIN_DM_SAMPLES:
        return fallback, "mcs_thin_matrix"
    result = model_confidence_set(
        matrix,
        methods,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        block_length=block_length,
    )
    candidate_id = str(candidate["method_id"])
    reference_ids = tuple(str(reference["method_id"]) for reference in references)
    if result.contains(candidate_id) and all(
        not result.contains(reference_id) for reference_id in reference_ids
    ):
        return candidate, None
    return fallback, "mcs_not_separated"


# Far daily buckets sit below the 8-valid-times eligibility floor for weeks
# (n_valid_times 7/4/4 on 2026-08-05) and the MCS has its own >= 8 collapsed
# times requirement, so lowering the board floor could not promote anything.
# Pooling D3-10 for the GATE ONLY (scoring and selection stay per fine
# bucket) reaches 15 unique dates today.
_DAILY_GATE_POOL: Mapping[str, tuple[str, ...]] = {
    "D3-4": ("D3-4", "D5-7", "D8-10"),
    "D5-7": ("D3-4", "D5-7", "D8-10"),
    "D8-10": ("D3-4", "D5-7", "D8-10"),
}
_POOLED_GATE = "pooled_D3-10"
_POOLABLE_REASONS = frozenset({"mcs_thin_matrix", "mcs_no_matrix", "seq_no_matrix"})


def _pooled_daily_gate(
    parts: Mapping[str, str],
    normalized_scores: pl.DataFrame | None,
    references: tuple[str, ...],
    rule: str,
    alpha: float,
    n_bootstrap: int,
    block_length: int | None,
    eprocess_store: "EProcessStore | None",
) -> tuple[dict[str, object], str] | None:
    """Gate a far daily bucket on pooled D3-10 evidence.

    Returns the pooled decision (winner row from the pooled board, gate
    label) or None when pooling is inapplicable or still yields nothing
    eligible. A method promoted here carries ``gate="pooled_D3-10"`` so the
    report shows the evidence scope; a method good at D3-4 but bad at D8-10
    stays visible on the fine-bucket board rows.
    """
    pool = _DAILY_GATE_POOL.get(parts.get("lead_bucket", ""))
    if (
        pool is None
        or parts.get("product") != "daily"
        or normalized_scores is None
        or rule == "legacy"
    ):
        return None
    pooled_scores = normalized_scores.filter(
        (pl.col("product") == "daily")
        & (pl.col("variable") == parts["variable"])
        & pl.col("lead_bucket").is_in(pool)
    )
    if "truth_semantics" in parts and "semantics" in normalized_scores.columns:
        pooled_scores = pooled_scores.filter(
            pl.col("semantics") == parts["truth_semantics"]
        )
    if pooled_scores.is_empty():
        return None
    pooled_scores = pooled_scores.with_columns(
        pl.lit(_POOLED_GATE.removeprefix("pooled_")).alias("lead_bucket")
    )
    pooled_board = leaderboard(pooled_scores, references=references)
    if pooled_board.is_empty():
        return None
    eligible = pooled_board.filter(
        (pl.col("coverage") >= 0.8)
        & (pl.col("n") >= 8)
        & (pl.col("n_valid_times") >= 8)
    )
    if eligible.is_empty():
        return None
    ranked = eligible.sort("mae")
    candidate = ranked.row(0, named=True)
    reference_rows = tuple(
        ranked.filter(pl.col("method_id").is_in(references))
        .sort("mae")
        .iter_rows(named=True)
    )
    if candidate["method_id"] in references:
        return candidate, _POOLED_GATE
    present = {str(row["method_id"]) for row in reference_rows}
    if not set(references) <= present:
        if reference_rows:
            return _reference_fallback(reference_rows), "missing_reference"
        return None
    if rule == "seq_mcs" and eprocess_store is not None:
        candidate, gate = _seq_gate(
            candidate,
            reference_rows,
            pooled_scores,
            {**parts, "lead_bucket": _POOLED_GATE.removeprefix("pooled_")},
            alpha,
            eprocess_store,
        )
    else:
        candidate, gate = _mcs_gate(
            candidate,
            reference_rows,
            pooled_scores,
            (
                str(candidate["method_id"]),
                *(str(row["method_id"]) for row in reference_rows),
            ),
            alpha,
            n_bootstrap=n_bootstrap,
            block_length=block_length,
        )
    return candidate, (_POOLED_GATE if gate is None else gate)


def _seq_gate(
    candidate: dict[str, object],
    references: tuple[dict[str, object], ...],
    slice_scores: pl.DataFrame,
    key_parts: Mapping[str, str],
    alpha: float,
    store: "EProcessStore",
) -> tuple[dict[str, object], str | None]:
    """Promote when the candidate's e-process beats 1/alpha vs EVERY reference.

    Anytime-valid across nightly re-runs (Ville's inequality on the betting
    supermartingale): consulting the gate after every backtest refresh needs
    no alpha bookkeeping, and a promotion, once earned, does not flap on one
    night's noise. Updates are cursor-deduped inside the store, so calling
    this twice on the same scores is a no-op.
    """
    from grounded_weather_forecast.reports.eprocess import (  # noqa: PLC0415
        pair_key,
    )
    from grounded_weather_forecast.reports.mcs import (  # noqa: PLC0415
        collapsed_loss_frame,
    )

    fallback = _reference_fallback(references)
    candidate_id = str(candidate["method_id"])
    threshold = math.log(1.0 / alpha)
    worst_log_e = math.inf
    for reference in references:
        reference_id = str(reference["method_id"])
        built = collapsed_loss_frame(
            slice_scores, method_ids=(candidate_id, reference_id)
        )
        if built is None:
            return fallback, "seq_no_matrix"
        collapsed, _methods = built
        key = pair_key(
            key_parts.get("product", ""),
            key_parts.get("variable", ""),
            key_parts.get("truth_semantics", ""),
            key_parts.get("lead_bucket", ""),
            candidate_id,
            reference_id,
        )
        entry = store.update_pair(
            key,
            collapsed[candidate_id].to_numpy().astype(np.float64),
            collapsed[reference_id].to_numpy().astype(np.float64),
            collapsed["valid_time"].to_list(),
        )
        worst_log_e = min(worst_log_e, entry.log_e)
    if worst_log_e >= threshold:
        return candidate, None
    return fallback, "seq_e_below_threshold"


def slice_winners(
    board: pl.DataFrame,
    scores: pl.DataFrame | None = None,
    rule: str = "legacy",
    alpha: float = 0.1,
    *,
    promotion: "PromotionConfig | None" = None,
    eprocess_store: "EProcessStore | None" = None,
) -> pl.DataFrame:
    """Promote a challenger only past the configured statistical gate.

    ``rule="mcs"`` (with the raw ``scores``) uses the Model Confidence Set;
    ``rule="seq_mcs"`` (with ``scores`` and an ``eprocess_store``) the
    anytime-valid betting e-process; ``"legacy"`` keeps the single-DM gate.
    Coverage and effective-n gates apply either way. ``promotion`` supplies
    per-variable reference overrides and MCS bootstrap parameters.
    """
    if board.is_empty():
        # A young archive's board is a bare frame with no columns at all;
        # returning it verbatim broke every consumer that filters winner
        # columns (the dashboard's reference-fallback stat, most visibly).
        empty_keys = ["product", "variable", "lead_bucket"]
        if "truth_semantics" in board.columns:
            empty_keys.insert(2, "truth_semantics")
        return _empty_winners(empty_keys)
    normalized_scores = _with_default_semantics(scores) if scores is not None else None
    n_bootstrap = promotion.mcs_bootstrap if promotion is not None else 500
    block_length = promotion.mcs_block_length if promotion is not None else None
    winners: list[dict[str, object]] = []
    keys = ["product", "variable", "lead_bucket"]
    if "truth_semantics" in board.columns:
        keys.insert(2, "truth_semantics")
    output_columns = (
        *keys,
        "method_id",
        "n",
        "mae",
        "best_method_id",
        "best_mae",
        "best_n",
        "mae_gap",
        "gap_ratio",
        "gate",
    )
    for slice_key, group in board.partition_by(keys, as_dict=True).items():
        parts = dict(zip(keys, (str(part) for part in slice_key), strict=True))
        references = gate_references(parts["variable"], promotion)
        best = group.sort("mae").row(0, named=True)
        eligible = group.filter(
            (pl.col("coverage") >= 0.8)
            & (pl.col("n") >= 8)
            & (pl.col("n_valid_times") >= 8)
        )
        if eligible.is_empty():
            pooled = _pooled_daily_gate(
                parts,
                normalized_scores,
                references,
                rule,
                alpha,
                n_bootstrap,
                block_length,
                eprocess_store,
            )
            if pooled is not None:
                pooled_candidate, pooled_gate = pooled
                winners.append(
                    _annotate_winner(
                        {**pooled_candidate, **parts},
                        best,
                        eligible,
                        pooled_gate,
                        infer_gate=False,
                    )
                )
            continue
        ranked = eligible.sort("mae")
        candidate = ranked.row(0, named=True)
        gate: str | None = None
        reference_rows = tuple(
            ranked.filter(pl.col("method_id").is_in(references))
            .sort("mae")
            .iter_rows(named=True)
        )
        slice_scores: pl.DataFrame | None = None
        if normalized_scores is not None:
            slice_scores = normalized_scores.filter(
                (pl.col("product") == parts["product"])
                & (pl.col("variable") == parts["variable"])
                & (pl.col("lead_bucket") == parts["lead_bucket"])
            )
            if "truth_semantics" in parts and "semantics" in normalized_scores.columns:
                slice_scores = slice_scores.filter(
                    pl.col("semantics") == parts["truth_semantics"]
                )
        if candidate["method_id"] not in references:
            present_references = {
                str(reference["method_id"]) for reference in reference_rows
            }
            if not set(references) <= present_references:
                if reference_rows:
                    candidate = _reference_fallback(reference_rows)
                    gate = "missing_reference"
                else:
                    continue
            elif rule == "mcs" and slice_scores is not None:
                # Decision-scoped matrix: the gate decides "candidate vs the
                # references", so only those methods enter the common-case
                # loss matrix. Intersecting cases over the full pool let every
                # newly registered (and sometimes abstaining) method shrink
                # the sample and mechanically widen the max-statistic null —
                # the 2026-08-04 cycle demoted four standing winners that way.
                candidate, gate = _mcs_gate(
                    candidate,
                    reference_rows,
                    slice_scores,
                    (
                        str(candidate["method_id"]),
                        *(str(row["method_id"]) for row in reference_rows),
                    ),
                    alpha,
                    n_bootstrap=n_bootstrap,
                    block_length=block_length,
                )
            elif (
                rule == "seq_mcs"
                and slice_scores is not None
                and eprocess_store is not None
            ):
                candidate, gate = _seq_gate(
                    candidate,
                    reference_rows,
                    slice_scores,
                    parts,
                    alpha,
                    eprocess_store,
                )
            elif any(
                _legacy_gate(candidate, reference) is not candidate
                for reference in reference_rows
            ):
                candidate = _reference_fallback(reference_rows)
                gate = "dm_not_significant"
        if gate in _POOLABLE_REASONS:
            pooled = _pooled_daily_gate(
                parts,
                normalized_scores,
                references,
                rule,
                alpha,
                n_bootstrap,
                block_length,
                eprocess_store,
            )
            if pooled is not None:
                pooled_candidate, pooled_gate = pooled
                winners.append(
                    _annotate_winner(
                        {**pooled_candidate, **parts},
                        best,
                        eligible,
                        pooled_gate,
                        infer_gate=False,
                    )
                )
                continue
        winners.append(_annotate_winner(candidate, best, eligible, gate))
    if not winners:
        return _empty_winners(keys)
    return (
        pl.DataFrame(winners, infer_schema_length=None)
        .with_columns(
            pl.col("gate").cast(pl.String),
            pl.col("best_mae").cast(pl.Float64),
            pl.col("mae_gap").cast(pl.Float64),
            pl.col("gap_ratio").cast(pl.Float64),
        )
        .select(*output_columns)
        .sort(*keys)
    )


def _annotate_winner(
    candidate: dict[str, object],
    best: dict[str, object],
    eligible: pl.DataFrame,
    gate: str | None,
    *,
    infer_gate: bool = True,
) -> dict[str, object]:
    """Attach the served-vs-board-minimum gap and the gate that caused it.

    ``infer_gate=False`` preserves a caller-supplied label (the pooled daily
    gate), which the best-vs-eligible inference would otherwise overwrite.
    """
    best_id = str(best["method_id"])
    if not infer_gate:
        pass
    elif str(candidate["method_id"]) == best_id:
        gate = None
    elif not (eligible["method_id"] == best_id).any():
        # The board minimum never even reached the statistical gates.
        gate = "eligibility"
    best_mae = best["mae"] if isinstance(best["mae"], (int, float)) else None
    served_mae = (
        candidate["mae"] if isinstance(candidate["mae"], (int, float)) else None
    )
    mae_gap = (
        float(served_mae) - float(best_mae)
        if served_mae is not None and best_mae is not None
        else None
    )
    gap_ratio = (
        float(served_mae) / float(best_mae) - 1.0
        if served_mae is not None and best_mae is not None and best_mae > 0
        else None
    )
    return {
        **candidate,
        "best_method_id": best_id,
        "best_mae": best_mae,
        "best_n": best["n"],
        "mae_gap": mae_gap,
        "gap_ratio": gap_ratio,
        "gate": gate,
    }


def _empty_winners(keys: list[str]) -> pl.DataFrame:
    """Typed empty frame: the winner columns no longer all exist on the board."""
    return pl.DataFrame(
        schema={
            **dict.fromkeys(keys, pl.String),
            "method_id": pl.String,
            "n": pl.Int64,
            "mae": pl.Float64,
            "best_method_id": pl.String,
            "best_mae": pl.Float64,
            "best_n": pl.Int64,
            "mae_gap": pl.Float64,
            "gap_ratio": pl.Float64,
            "gate": pl.String,
        }
    )


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """BH step-up adjusted q-values; NaN passes through untouched."""
    q_values = np.full(p_values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p_values))
    m = finite.size
    if m == 0:
        return q_values
    order = np.argsort(p_values[finite])
    ranked = p_values[finite][order]
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q_values[finite[order]] = np.clip(adjusted, 0.0, 1.0)
    return q_values


def bh_adjusted(board: pl.DataFrame) -> pl.DataFrame:
    """Board plus ``dm_q_vs_*`` columns: BH-FDR across the slice family.

    Each reference's DM p-values form one family over the whole board (the
    winner's-curse surface future-work #21 names); the q-value is the
    smallest FDR at which that row would survive. Report-layer only — the
    gate is unchanged.
    """
    if board.is_empty():
        return board
    result = board
    for column in board.columns:
        if not column.startswith("dm_p_vs_"):
            continue
        p_values = board[column].cast(pl.Float64).fill_null(float("nan")).to_numpy()
        q_values = _benjamini_hochberg(np.asarray(p_values, dtype=np.float64))
        q_column = column.replace("dm_p_vs_", "dm_q_vs_")
        result = result.with_columns(
            pl.Series(q_column, q_values).replace(float("nan"), None)
        )
    return result


def blocked_promotions(winners: pl.DataFrame, threshold: float = 0.15) -> pl.DataFrame:
    """Slices serving measurably above their board minimum, worst first.

    The gate may be right or wrong per slice — thin data is a legitimate
    reason to hold a reference — but a 46 % gap should never be silent.
    """
    if winners.is_empty() or "gate" not in winners.columns:
        return winners
    return winners.filter(
        pl.col("gate").is_not_null()
        & (
            (pl.col("gap_ratio") > threshold)
            | (pl.col("gap_ratio").is_null() & (pl.col("mae_gap") > 0.0))
        )
    ).sort("gap_ratio", descending=True, nulls_last=True)
