"""Compatibility adapter for direct dashboard freshness callers."""

from grounded_weather_forecast.dashboard.context import DashboardContext
from grounded_weather_forecast.reports.truth_qc import (
    TruthQCFreshness,
)
from grounded_weather_forecast.reports.truth_qc import (
    truth_qc_freshness as assess_freshness,
)


def truth_qc_freshness(ctx: DashboardContext) -> TruthQCFreshness:
    return assess_freshness(ctx.truth_qc, ctx.runs, ctx.now)
