"""Shared derived frames computed once per dashboard build.

Score statistics come from the collector's compact summaries. Verification
and correlation are derived here once for zone builders and alert evaluation.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

import polars as pl

from grounded_weather_forecast.contracts import (
    TruthSemantics,
    finite_number,
    hourly_variable,
)
from grounded_weather_forecast.dashboard.context import DashboardContext
from grounded_weather_forecast.reports.correlation import error_correlation
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
    live_scores_unusable: bool = False
    """A live score file exists but could not be read or scored."""


def _k_eff(correlation: pl.DataFrame) -> float | None:
    """Effective independent-source count, or ``None`` when not evaluable.

    Non-finite entries are dropped rather than clamped: ``min(1.0, nan)``
    returns ``1.0``, which would report "no independence" — the most alarming
    possible answer — from an absence of evidence. Dropping them at the source
    is what makes the mean safe, so the clamp below only ever bounds a real
    correlation into range.
    """
    sources = [column for column in correlation.columns if column != "source"]
    if len(sources) < 2:
        return None
    values = []
    for row_index, row in enumerate(correlation.iter_rows(named=True)):
        for column_index, source in enumerate(sources):
            if column_index <= row_index:
                continue
            if (value := finite_number(row[source])) is not None:
                values.append(value)
    if not values:
        return None
    mean_r = max(0.0, min(1.0, sum(values) / len(values)))
    n = len(sources)
    return n / (1.0 + (n - 1) * mean_r)


def derive(ctx: DashboardContext) -> Derived:
    boards = {
        stem: summary.board
        for stem, summary in ctx.score_summaries.items()
        if summary.board is not None
    }
    winners = {
        stem: summary.winners
        for stem, summary in ctx.score_summaries.items()
        if summary.winners is not None
    }
    live_stem = next((stem for stem in boards if "_live" in stem), None)
    live_unusable = any("_live" in stem for stem in ctx.unreadable_scores) or any(
        "_live" in stem and summary.unusable
        for stem, summary in ctx.score_summaries.items()
    )

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
            live_unusable = True

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
        live_scores_unusable=live_unusable,
    )
