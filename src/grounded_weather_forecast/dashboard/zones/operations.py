"""Zone I: pipeline operations, read from the edge ledgers and the run log.

Every panel is a trend view over evidence the nightly report already
banked (``reports/operations.py``) or the CLI run ledger; an empty ledger
renders a "young history" placeholder, never an alarm. The one exception
is i1: a freshness row whose alarm string is non-empty is a live fault by
definition — that is the panel the two historical week-long silent
failures would have turned red on day one.
"""

from typing import cast

import polars as pl

from grounded_weather_forecast.dashboard.charts import line_chart
from grounded_weather_forecast.dashboard.context import DashboardContext
from grounded_weather_forecast.dashboard.copy import PANEL_COPY, ZONE_INTROS
from grounded_weather_forecast.dashboard.derive import Derived
from grounded_weather_forecast.dashboard.model import (
    Panel,
    PanelStatus,
    Stat,
    TableSpec,
    Zone,
)
from grounded_weather_forecast.dashboard.zones.common import empty_panel, fmt
from grounded_weather_forecast.reports.operations import FRESHNESS_THRESHOLDS

_MAX_POINTS = 60
_MAX_SERIES = 8
_CHANGES_ROWS = 10
_RUNTIME_COMMANDS = ("build-dataset", "backtest", "report", "predict")
_RUNTIME_AMBER = 1.5
_SUCCESS_AMBER = 0.8

_AGE_LABELS = {
    "truth_age_minutes": "truth",
    "collector_age_minutes": "collector",
    "served_history_age_minutes": "served history",
    "forecast_document_age_minutes": "forecast doc",
}


def _ledger(ctx: DashboardContext, name: str) -> pl.DataFrame:
    return ctx.evidence_history.get(name, pl.DataFrame())


def _labels(frame: pl.DataFrame, column: str = "as_of_date") -> list[str]:
    ordered = frame.sort(column)[column].cast(pl.String).to_list()
    return list(dict.fromkeys(ordered))[-_MAX_POINTS:]


def _freshness_panel(ctx: DashboardContext) -> Panel:
    pipeline = _ledger(ctx, "pipeline")
    if pipeline.is_empty():
        return empty_panel(
            "i1",
            "i1",
            "End-to-end freshness",
            "info",
            "young history — no pipeline freshness rows yet",
        )
    newest = pipeline.sort("as_of_date").tail(1).row(0, named=True)
    alarms = str(newest["alarms"] or "")
    stats = []
    for column, label in _AGE_LABELS.items():
        age = newest[column]
        limit = FRESHNESS_THRESHOLDS[column]
        status: PanelStatus = "red" if age is None or age > limit else "ok"
        stats.append(
            Stat(
                label=f"{label} age",
                value="—" if age is None else f"{age:.0f}m",
                status=status,
            )
        )
    labels = _labels(pipeline)
    by_date = {
        str(row["as_of_date"]): row
        for row in pipeline.sort("as_of_date").iter_rows(named=True)
    }
    series = [
        (
            label,
            [
                (
                    None if by_date.get(day) is None else by_date[day][column]  # type: ignore[index]
                )
                for day in labels
            ],
        )
        for column, label in _AGE_LABELS.items()
    ]
    return Panel(
        panel_id="i1",
        title="End-to-end freshness",
        status="red" if alarms else "ok",
        copy=PANEL_COPY["i1"],
        stats=tuple(stats),
        chart=line_chart(labels, series, y_label="age (minutes)"),
        intro=f"alarms: {alarms}" if alarms else None,
    )


def _provider_panel(ctx: DashboardContext) -> Panel:
    health = _ledger(ctx, "provider_health")
    if health.is_empty():
        return empty_panel(
            "i2",
            "i2",
            "Provider collector health",
            "info",
            "young history — no provider health rows yet",
        )
    newest_date = health["as_of_date"].max()
    today = health.filter(pl.col("as_of_date") == newest_date).sort("provider")
    worst = cast(
        "float | None",
        today.filter(pl.col("success_rate").is_not_null())["success_rate"].min(),
    )
    status: PanelStatus = (
        "amber" if worst is not None and worst < _SUCCESS_AMBER else "ok"
    )
    labels = _labels(health)
    top = (
        health.group_by("provider")
        .agg(pl.col("daily_rows_24h").sum().alias("volume"))
        .sort(["volume", "provider"], descending=[True, False])
        .head(_MAX_SERIES)["provider"]
        .to_list()
    )
    series = []
    for provider in top:
        points = dict(
            health.filter(pl.col("provider") == provider)
            .with_columns(pl.col("as_of_date").cast(pl.String))
            .select("as_of_date", "max_daily_lead_days")
            .iter_rows()
        )
        series.append((str(provider), [points.get(day) for day in labels]))
    table = TableSpec(
        columns=("provider", "ok/runs", "latency ms", "hourly lead", "daily lead"),
        rows=tuple(
            (
                str(row["provider"]),
                f"{fmt(row['ok_24h'])}/{fmt(row['runs_24h'])}",
                fmt(row["median_latency_ms"], 0),
                fmt(row["max_hourly_lead_hours"], 0),
                fmt(row["max_daily_lead_days"], 0),
            )
            for row in today.iter_rows(named=True)
        ),
    )
    return Panel(
        panel_id="i2",
        title="Provider collector health",
        status=status,
        copy=PANEL_COPY["i2"],
        stats=(
            Stat(
                label="worst success rate",
                value="—" if worst is None else f"{worst:.0%}",
                status=status,
            ),
        ),
        chart=line_chart(labels, series, y_label="max daily lead (days)"),
        table=table,
    )


def _runtime_panel(ctx: DashboardContext) -> Panel:
    runs = ctx.runs
    if runs.is_empty():
        return empty_panel(
            "i3",
            "i3",
            "Stage runtimes",
            "info",
            "young history — the CLI run ledger is empty",
        )
    usable = runs.filter(
        pl.col("command").is_in(_RUNTIME_COMMANDS)
        & pl.col("started_at").is_not_null()
        & pl.col("duration_ms").is_not_null()
    )
    if usable.is_empty():
        return empty_panel(
            "i3", "i3", "Stage runtimes", "info", "no timed pipeline commands yet"
        )
    daily = (
        usable.with_columns(
            pl.col("started_at").dt.date().cast(pl.String).alias("day"),
            (pl.col("duration_ms") / 60_000.0).alias("minutes"),
        )
        .group_by("day", "command")
        .agg(pl.col("minutes").median().alias("minutes"))
        .sort("day")
    )
    labels = _labels(daily, "day")
    series = []
    for command in _RUNTIME_COMMANDS:
        points = dict(
            daily.filter(pl.col("command") == command)
            .select("day", "minutes")
            .iter_rows()
        )
        series.append((command, [points.get(day) for day in labels]))
    reports = daily.filter(pl.col("command") == "report")["minutes"].to_list()
    status: PanelStatus = "ok"
    latest = reports[-1] if reports else None
    if latest is not None and len(reports) >= 3:
        prior = sorted(reports[:-1])
        median = prior[len(prior) // 2]
        if median > 0 and latest > _RUNTIME_AMBER * median:
            status = "amber"
    return Panel(
        panel_id="i3",
        title="Stage runtimes",
        status=status,
        copy=PANEL_COPY["i3"],
        stats=(
            Stat(
                label="latest report",
                value="—" if latest is None else f"{latest:.1f}m",
                status=status,
            ),
        ),
        chart=line_chart(labels, series, y_label="median minutes/day"),
    )


def _funnel_panel(ctx: DashboardContext) -> Panel:
    funnel = _ledger(ctx, "build_funnel")
    if funnel.is_empty():
        return empty_panel(
            "i4",
            "i4",
            "Build funnel",
            "info",
            "young history — no build-funnel rows yet",
        )
    newest_date = funnel["as_of_date"].max()
    today = funnel.filter(pl.col("as_of_date") == newest_date).sort(
        "granularity", "source"
    )
    table = TableSpec(
        columns=(
            "granularity",
            "source",
            "collector lead",
            "long lead",
            "matrix lead",
            "native",
            "path",
            "matrix rows",
        ),
        rows=tuple(
            (
                str(row["granularity"]),
                str(row["source"]),
                fmt(row["collector_max_lead"], 1),
                fmt(row["long_max_lead"], 1),
                fmt(row["matrix_max_lead"], 1),
                fmt(row["matrix_native_max_lead"], 1),
                fmt(row["matrix_path_max_lead"], 1),
                fmt(row["matrix_rows"]),
            )
            for row in today.iter_rows(named=True)
        ),
    )
    return Panel(
        panel_id="i4",
        title="Build funnel",
        status="ok",
        copy=PANEL_COPY["i4"],
        stats=(Stat(label="as of", value=str(newest_date)),),
        table=table,
    )


def _footprint_panel(ctx: DashboardContext) -> Panel:
    catalog = _ledger(ctx, "evaluations")
    if catalog.is_empty():
        return empty_panel(
            "i5",
            "i5",
            "Evidence footprint",
            "info",
            "young history — no evaluations cataloged yet",
        )
    total_mb = catalog["file_size_mb"].sum()
    per_eval = (
        catalog.with_columns(
            pl.col("recorded_at").dt.date().cast(pl.String).alias("day")
        )
        .group_by("day", "product")
        .agg(pl.col("rows").max().alias("rows"))
        .sort("day")
    )
    labels = _labels(per_eval, "day")
    series = []
    for product in sorted(catalog["product"].drop_nulls().unique().to_list()):
        points = dict(
            per_eval.filter(pl.col("product") == product)
            .select("day", "rows")
            .iter_rows()
        )
        series.append((str(product), [points.get(day) for day in labels]))
    return Panel(
        panel_id="i5",
        title="Evidence footprint",
        status="ok",
        copy=PANEL_COPY["i5"],
        stats=(
            Stat(label="evaluations cataloged", value=str(catalog.height)),
            Stat(label="cataloged volume", value=f"{float(total_mb or 0.0):.0f} MB"),
        ),
        chart=line_chart(labels, series, y_label="rows per evaluation"),
    )


def _changes_panel(ctx: DashboardContext) -> Panel:
    changes = _ledger(ctx, "changes")
    if changes.is_empty():
        return empty_panel(
            "i6",
            "i6",
            "Identity changes",
            "info",
            "no config or code identity changes recorded yet",
        )
    recent = changes.sort("recorded_at", descending=True).head(_CHANGES_ROWS)
    table = TableSpec(
        columns=("date", "kind", "from", "to", "changed keys"),
        rows=tuple(
            (
                str(row["as_of_date"]),
                str(row["kind"]),
                str(row["from_value"]),
                str(row["to_value"]),
                fmt(row["detail"]),
            )
            for row in recent.iter_rows(named=True)
        ),
    )
    return Panel(
        panel_id="i6",
        title="Identity changes",
        status="info",
        copy=PANEL_COPY["i6"],
        stats=(Stat(label="recorded transitions", value=str(changes.height)),),
        table=table,
    )


def build(ctx: DashboardContext, derived: Derived) -> Zone:  # noqa: ARG001
    return Zone(
        zone_id="I",
        title="Operations",
        intro=ZONE_INTROS["I"],
        panels=(
            _freshness_panel(ctx),
            _provider_panel(ctx),
            _runtime_panel(ctx),
            _funnel_panel(ctx),
            _footprint_panel(ctx),
            _changes_panel(ctx),
        ),
    )
