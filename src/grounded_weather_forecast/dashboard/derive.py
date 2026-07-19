"""Shared derived frames computed once per dashboard build.

Everything here reuses the existing reports functions on the frames the
context collector loaded; zone builders and the alert evaluator both consume
the results so nothing is computed twice.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

import polars as pl

from grounded_weather_forecast.contracts import TruthSemantics, hourly_variable
from grounded_weather_forecast.dashboard.context import DashboardContext
from grounded_weather_forecast.reports.correlation import error_correlation
from grounded_weather_forecast.reports.leaderboard import leaderboard, slice_winners
from grounded_weather_forecast.reports.verification import (
    compare_to_backtest,
    verify_history,
)


@dataclass(frozen=True, slots=True)
class Derived:
    boards: Mapping[str, pl.DataFrame] = field(default_factory=dict)
    winners: Mapping[str, pl.DataFrame] = field(default_factory=dict)
    live_stem: str | None = None
    verification: pl.DataFrame = field(default_factory=pl.DataFrame)
    correlation: pl.DataFrame | None = None
    k_eff: float | None = None


def _k_eff(correlation: pl.DataFrame) -> float | None:
    sources = [column for column in correlation.columns if column != "source"]
    if len(sources) < 2:
        return None
    values = []
    for row_index, row in enumerate(correlation.iter_rows(named=True)):
        for column_index, source in enumerate(sources):
            if column_index <= row_index:
                continue
            value = row[source]
            if isinstance(value, (int, float)):
                values.append(float(value))
    if not values:
        return None
    mean_r = max(0.0, min(1.0, sum(values) / len(values)))
    n = len(sources)
    return n / (1.0 + (n - 1) * mean_r)


def derive(ctx: DashboardContext) -> Derived:
    boards: dict[str, pl.DataFrame] = {}
    winners: dict[str, pl.DataFrame] = {}
    live_stem: str | None = None
    for stem, scores in ctx.score_frames.items():
        if scores.is_empty():
            continue
        try:
            board = leaderboard(scores)
            boards[stem] = board
            winners[stem] = slice_winners(
                board,
                scores=scores,
                rule=ctx.config.promotion.rule,
                alpha=ctx.config.promotion.alpha,
            )
        except (ValueError, pl.exceptions.PolarsError):
            continue
        if "_live" in stem and live_stem is None:
            live_stem = stem

    verification = pl.DataFrame()
    if (
        live_stem is not None
        and ctx.history is not None
        and ctx.truth_hourly is not None
    ):
        try:
            live = verify_history(
                ctx.config.predict.history_path,
                ctx.truth_hourly,
                truth_minute=ctx.truth_minute,
                truth_daily=ctx.truth_daily,
            )
            if not live.is_empty():
                verification = compare_to_backtest(live, boards[live_stem])
        except (OSError, ValueError, pl.exceptions.PolarsError):
            verification = pl.DataFrame()

    correlation: pl.DataFrame | None = None
    k_eff: float | None = None
    if ctx.hourly_matrix is not None and not ctx.hourly_matrix.is_empty():
        try:
            correlation = error_correlation(
                ctx.hourly_matrix,
                hourly_variable("temp_c"),
                TruthSemantics.INSTANTANEOUS,
            )
            if not correlation.is_empty():
                k_eff = _k_eff(correlation)
        except (ValueError, pl.exceptions.PolarsError):
            correlation = None

    return Derived(
        boards=boards,
        winners=winners,
        live_stem=live_stem,
        verification=verification,
        correlation=correlation,
        k_eff=k_eff,
    )
