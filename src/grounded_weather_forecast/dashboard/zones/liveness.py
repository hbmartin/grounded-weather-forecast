"""Zone A: liveness and freshness."""

from datetime import datetime, timedelta

import polars as pl

from grounded_weather_forecast.contracts import age_col
from grounded_weather_forecast.dashboard.charts import bar_chart
from grounded_weather_forecast.dashboard.context import DashboardContext
from grounded_weather_forecast.dashboard.copy import PANEL_COPY, ZONE_INTROS
from grounded_weather_forecast.dashboard.derive import Derived
from grounded_weather_forecast.dashboard.model import Panel, Stat, Zone
from grounded_weather_forecast.dashboard.zones.common import (
    empty_panel,
    fmt,
)
from grounded_weather_forecast.serve.predict import OBS_STALENESS


def _serving_status(ctx: DashboardContext) -> Panel:
    forecast = ctx.latest_forecast
    if forecast is None:
        return empty_panel(
            "a1",
            "a1",
            "Serving status",
            "red",
            "nothing has ever been served: no archived forecast document "
            "exists; run `predict` once the dataset is built",
        )
    degraded = forecast.status == "degraded"
    stats = (
        Stat("status", forecast.status, "amber" if degraded else "ok"),
        Stat("issued", forecast.issued_at),
        Stat(
            "releases",
            ", ".join(forecast.release_ids) if forecast.release_ids else "none",
            "amber" if not forecast.release_ids else "ok",
        ),
        Stat("dataset", forecast.dataset_fingerprint),
    )
    intro = (
        f"degraded: {forecast.status_reason}"
        if degraded and forecast.status_reason
        else None
    )
    return Panel(
        panel_id="a1",
        title="Serving status",
        status="amber" if degraded else "ok",
        copy=PANEL_COPY["a1"],
        stats=stats,
        intro=intro,
        raw_html="",
    )


def _observation_lag(ctx: DashboardContext) -> Panel:
    frame = ctx.truth_minute
    if frame is None or frame.is_empty() or "ts" not in frame.columns:
        return empty_panel(
            "a2",
            "a2",
            "Station observation lag",
            "red",
            "no station observations on disk — the truth pipeline has never "
            "run, or the collector is down",
        )
    newest = frame["ts"].max()
    if not isinstance(newest, datetime):
        return empty_panel(
            "a2",
            "a2",
            "Station observation lag",
            "red",
            "observation timestamps are unreadable",
        )
    lag = ctx.now - newest
    cap = timedelta(hours=ctx.config.forecasts.max_forecast_age_hours)
    status = "red" if lag > cap else "amber" if lag > OBS_STALENESS else "ok"
    return Panel(
        panel_id="a2",
        title="Station observation lag",
        status=status,
        copy=PANEL_COPY["a2"],
        stats=(
            Stat("last observation", newest.isoformat()),
            Stat("lag", f"{lag.total_seconds() / 60:.0f} min", status),
            Stat("anchor", "lost" if lag > OBS_STALENESS else "live", status),
        ),
        raw_html="",
    )


def _provider_ages(ctx: DashboardContext, sources: tuple[str, ...]) -> Panel:
    matrix = ctx.hourly_matrix
    if matrix is None or matrix.is_empty() or "issue_time" not in matrix.columns:
        return empty_panel(
            "a3",
            "a3",
            "Provider vintage ages",
            "amber",
            "no live matrix snapshot yet — run `build-dataset` after the "
            "first provider fetches land",
        )
    newest = matrix.filter(pl.col("issue_time") == matrix["issue_time"].max()).row(
        0, named=True
    )
    cap = ctx.config.forecasts.max_forecast_age_hours
    labels: list[str] = []
    ages: list[float | None] = []
    colors: list[str] = []
    for index, source in enumerate(sources or _matrix_sources(matrix)):
        age = newest.get(age_col(source))
        labels.append(source)
        ages.append(float(age) if isinstance(age, (int, float)) else None)
        aged_out = not isinstance(age, (int, float)) or age > cap
        colors.append("muted" if aged_out else f"series-{(index % 8) + 1}")
    stale = sum(1 for age in ages if age is None or age > cap)
    status = "red" if stale == len(ages) else "amber" if stale else "ok"
    return Panel(
        panel_id="a3",
        title="Provider vintage ages",
        status=status,
        copy=PANEL_COPY["a3"],
        stats=(
            Stat("providers fresh", f"{len(ages) - stale}/{len(ages)}", status),
            Stat("freshness cap", f"{fmt(cap)} h"),
            Stat("snapshot", str(newest.get("issue_time", "—"))),
        ),
        intro=(
            "Grey bars are providers missing from (or aged out of) the newest snapshot."
        ),
        chart=bar_chart(
            labels,
            [("hours since fetch", ages)],
            y_label="hours",
            horizontal=True,
            colors=colors,
        ),
    )


def _matrix_sources(matrix: pl.DataFrame) -> tuple[str, ...]:
    return tuple(
        sorted(
            column.removeprefix("age__")
            for column in matrix.columns
            if column.startswith("age__")
        )
    )


def build(ctx: DashboardContext, derived: Derived) -> Zone:  # noqa: ARG001
    sources = ()
    if ctx.manifest is not None:
        raw = ctx.manifest.get("sources")
        if isinstance(raw, list):
            sources = tuple(str(source) for source in raw)
    lag_panel = _observation_lag(ctx)
    return Zone(
        zone_id="A",
        title="Liveness & freshness",
        intro=ZONE_INTROS["A"],
        panels=(_serving_status(ctx), lag_panel, _provider_ages(ctx, sources)),
    )
