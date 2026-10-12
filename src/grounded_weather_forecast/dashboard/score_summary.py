"""Small dashboard score summaries; full frames live only during collection."""

import json
from dataclasses import dataclass

import numpy as np
import polars as pl

from grounded_weather_forecast.config import Config
from grounded_weather_forecast.evaluation import code_identity, config_fingerprint
from grounded_weather_forecast.metrics.probabilistic import (
    pit_from_quantiles,
    reliability_bins,
)
from grounded_weather_forecast.reports.eprocess import EProcessStore
from grounded_weather_forecast.reports.leaderboard import (
    leaderboard,
    slice_winners,
    union_references,
)
from grounded_weather_forecast.reports.winner_curse import winner_curse_adjusted


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    source_kinds: frozenset[str]
    fold_origins: frozenset[object]
    board: pl.DataFrame | None = None
    winners: pl.DataFrame | None = None
    pit_counts: tuple[int, ...] = ()
    pit_rows: int = 0
    reliability: tuple[tuple[float, float, int], ...] = ()
    unusable: bool = False
    pit_unusable: bool = False
    reliability_unusable: bool = False


def _eprocess_store(
    config: Config, stem: str, cache: dict[str, EProcessStore]
) -> EProcessStore | None:
    """Read-only wealth for the seq_mcs rule; ``report`` is the only writer."""
    if config.promotion.rule != "seq_mcs" or "_live" not in stem:
        return None
    product = stem.split("_")[1] if "_" in stem else stem
    if product not in cache:
        cache[product] = EProcessStore.load(
            config.artifacts_dir / "eprocess" / f"{product}_live.json",
            config_fingerprint(config),
            code_identity(),
        )
    return cache[product]


def pit_values(scores: pl.DataFrame) -> np.ndarray:
    needed = {"quantiles_json", "quantile_levels_json", "y_true"}
    if not needed <= set(scores.columns):
        return np.empty(0)
    usable = scores.filter(
        pl.col("quantiles_json").is_not_null() & pl.col("y_true").is_not_null()
    )
    if usable.is_empty():
        return np.empty(0)
    groups: dict[str, list[tuple[list[float], float]]] = {}
    for row in usable.iter_rows(named=True):
        try:
            quantiles = json.loads(row["quantiles_json"])
        except (TypeError, ValueError):
            continue
        if not isinstance(quantiles, list) or any(q is None for q in quantiles):
            continue
        key = str(row["quantile_levels_json"])
        groups.setdefault(key, []).append((quantiles, float(row["y_true"])))
    if not groups:
        return np.empty(0)
    key, entries = max(groups.items(), key=lambda item: len(item[1]))
    try:
        levels = tuple(float(level) for level in json.loads(key))
    except (TypeError, ValueError):
        return np.empty(0)
    quantile_matrix = np.asarray([entry[0] for entry in entries], dtype=np.float64)
    truths = np.asarray([entry[1] for entry in entries], dtype=np.float64)
    if quantile_matrix.shape[1] != len(levels):
        return np.empty(0)
    return pit_from_quantiles(truths, quantile_matrix, levels)


def reliability_summary(scores: pl.DataFrame) -> tuple[tuple[float, float, int], ...]:
    if not {"variable", "y_pred", "y_true"} <= set(scores.columns):
        return ()
    pop = scores.filter(
        (pl.col("variable") == "pop")
        & pl.col("y_pred").is_not_null()
        & pl.col("y_true").is_not_null()
    )
    if pop.height < 20:
        return ()
    bins = reliability_bins(
        pop["y_pred"].to_numpy().astype(np.float64),
        pop["y_true"].to_numpy().astype(np.float64),
    ).filter(pl.col("count") > 0)
    return tuple(
        (float(row["forecast_mean"]), float(row["observed_freq"]), int(row["count"]))
        for row in bins.iter_rows(named=True)
        if row["forecast_mean"] is not None and row["observed_freq"] is not None
    )


def summarize_scores(
    config: Config, stem: str, scores: pl.DataFrame, stores: dict[str, EProcessStore]
) -> ScoreSummary:
    kinds = (
        frozenset(str(value) for value in scores["source_kind"].unique())
        if "source_kind" in scores.columns and not scores.is_empty()
        else frozenset()
    )
    folds = (
        frozenset(scores["fold_origin"].unique())
        if "fold_origin" in scores.columns and not scores.is_empty()
        else frozenset()
    )
    if scores.is_empty():
        return ScoreSummary(source_kinds=kinds, fold_origins=folds)
    try:
        board = leaderboard(scores, references=union_references(config.promotion))
        winners = winner_curse_adjusted(
            slice_winners(
                board,
                scores=scores,
                rule=config.promotion.rule,
                alpha=config.promotion.alpha,
                promotion=config.promotion,
                eprocess_store=_eprocess_store(config, stem, stores),
            ),
            scores,
            promotion=config.promotion,
        )
    except (OSError, ValueError, pl.exceptions.PolarsError):
        # Readable metadata remains useful even when statistical scoring fails.
        return ScoreSummary(source_kinds=kinds, fold_origins=folds, unusable=True)
    pit_unusable = False
    counts: tuple[int, ...] = ()
    pit_rows = 0
    try:
        pit = pit_values(scores)
        counts = (
            tuple(
                int(value) for value in np.histogram(pit, bins=10, range=(0.0, 1.0))[0]
            )
            if pit.size
            else ()
        )
        pit_rows = int(pit.size)
    except (OSError, ValueError, pl.exceptions.PolarsError):
        pit_unusable = True
    reliability_unusable = False
    reliability: tuple[tuple[float, float, int], ...] = ()
    try:
        reliability = reliability_summary(scores)
    except (OSError, ValueError, pl.exceptions.PolarsError):
        reliability_unusable = True
    return ScoreSummary(
        source_kinds=kinds,
        fold_origins=folds,
        board=board,
        winners=winners,
        pit_counts=counts,
        pit_rows=pit_rows,
        reliability=reliability,
        pit_unusable=pit_unusable,
        reliability_unusable=reliability_unusable,
    )
