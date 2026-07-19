"""Zone C: learning readiness — the binding constraint on a young deployment."""

from collections.abc import Mapping
from datetime import datetime

import polars as pl

from grounded_weather_forecast.dashboard.charts import line_chart
from grounded_weather_forecast.dashboard.context import DashboardContext
from grounded_weather_forecast.dashboard.copy import PANEL_COPY, ZONE_INTROS
from grounded_weather_forecast.dashboard.derive import Derived
from grounded_weather_forecast.dashboard.html import esc
from grounded_weather_forecast.dashboard.model import Panel, Stat, TableSpec, Zone
from grounded_weather_forecast.dashboard.zones.common import empty_panel, fmt


def _issue_span_days(matrix: pl.DataFrame) -> float:
    first = matrix["issue_time"].min()
    last = matrix["issue_time"].max()
    if not isinstance(first, datetime) or not isinstance(last, datetime):
        return 0.0
    return (last - first).total_seconds() / 86400.0


def _fold_count(ctx: DashboardContext) -> int:
    origins: set[object] = set()
    for stem, frame in ctx.score_frames.items():
        if "_live" in stem and "fold_origin" in frame.columns:
            origins.update(frame["fold_origin"].unique().to_list())
    return len(origins)


def _fold_readiness(ctx: DashboardContext) -> Panel:
    matrix = ctx.hourly_matrix
    needed = float(
        ctx.config.backtest.initial_train_days + ctx.config.backtest.step_days
    )
    if matrix is None or matrix.is_empty() or "issue_time" not in matrix.columns:
        return empty_panel(
            "c1",
            "c1",
            "Archive growth & fold readiness",
            "amber",
            "no live archive yet — a fold needs "
            f"{needed:.0f} days of issue-time span "
            "(initial_train_days + step_days); keep the cron polling",
        )
    span = _issue_span_days(matrix)
    fraction = min(1.0, span / needed) if needed else 0.0
    folds = _fold_count(ctx)
    per_day = (
        matrix.select(pl.col("issue_time").dt.date().alias("date"))
        .group_by("date")
        .len()
        .sort("date")
    )
    labels = [str(value) for value in per_day["date"].to_list()]
    counts = [float(value) for value in per_day["len"].to_list()]
    progress_html = (
        '<div class="progress" role="progressbar" '
        f'aria-valuenow="{esc(f"{span:.1f}")}" aria-valuemax="{esc(needed)}">'
        f'<div class="progress-fill" style="width:{fraction * 100:.1f}%"></div>'
        "</div>"
    )
    status = "ok" if folds else "info"
    return Panel(
        panel_id="c1",
        title="Archive growth & fold readiness",
        status=status,
        copy=PANEL_COPY["c1"],
        stats=(
            Stat("live span", f"{span:.1f} d"),
            Stat("needed for first fold", f"{needed:.0f} d"),
            Stat("live folds", str(folds), "info" if not folds else "ok"),
            Stat("snapshots", f"{matrix['issue_time'].n_unique()}"),
        ),
        intro=(
            None
            if folds
            else (
                f"The archive spans {span:.1f} days but a fold needs "
                f"{needed:.0f} — zero live folds is correct behaviour, "
                "not a fault. Keep polling."
            )
        ),
        raw_html=progress_html,
        chart=line_chart(labels, [("snapshots/day", counts)], y_label="snapshots"),
    )


def _synthetic_coverage(ctx: DashboardContext) -> Panel:
    matrix = ctx.synthetic_hourly
    if matrix is None or matrix.is_empty():
        return empty_panel(
            "c2",
            "c2",
            "Synthetic backfill",
            "info",
            "no synthetic matrix — run `backfill` to compare methods before "
            "the live archive matures",
        )
    span = _issue_span_days(matrix)
    min_lead = matrix["lead_hours"].min() if "lead_hours" in matrix.columns else None
    sources = sorted({c.split("__")[1] for c in matrix.columns if c.startswith("fx__")})
    return Panel(
        panel_id="c2",
        title="Synthetic backfill",
        status="ok",
        copy=PANEL_COPY["c2"],
        stats=(
            Stat("span", f"{span:.0f} d"),
            Stat("rows", f"{matrix.height:,}"),
            Stat("sources", ", ".join(sources) if sources else "—"),
            Stat("shortest lead", f"{fmt(min_lead)} h"),
        ),
        intro=(
            "Leads under 24 h are structurally absent from the Open-Meteo "
            "backfill; that gap is drawn empty on purpose."
        ),
        raw_html="",
    )


def _alignment(ctx: DashboardContext) -> Panel:
    alignment = ctx.alignment
    if alignment is None:
        return empty_panel(
            "c3",
            "c3",
            "Truth-semantics alignment",
            "info",
            "no alignment artifact — run `alignment` to study whether "
            "providers publish instantaneous or hour-mean values",
        )
    recommended = alignment.get("recommended")
    data_backed = alignment.get("data_backed")
    if not isinstance(recommended, Mapping):
        recommended = {}
    if not isinstance(data_backed, Mapping):
        data_backed = {}
    rows = []
    classes = []
    defaulted = 0
    for variable in sorted(recommended):
        backed = bool(data_backed.get(variable))
        defaulted += 0 if backed else 1
        rows.append(
            (
                str(variable),
                str(recommended[variable]),
                "data-backed" if backed else "defaulted (n < 72)",
            )
        )
        classes.append(("", "", "" if backed else "cell-bad"))
    min_rows = alignment.get("min_rows", 72)
    return Panel(
        panel_id="c3",
        title="Truth-semantics alignment",
        status="amber" if defaulted else "ok",
        copy=PANEL_COPY["c3"],
        stats=(
            Stat("defaulted variables", str(defaulted), "amber" if defaulted else "ok"),
            Stat("rows needed", str(min_rows)),
        ),
        table=TableSpec(
            columns=("variable", "semantics", "basis"),
            rows=tuple(rows),
            cell_classes=tuple(classes),
        ),
    )


def build(ctx: DashboardContext, derived: Derived) -> Zone:  # noqa: ARG001
    return Zone(
        zone_id="C",
        title="Learning readiness",
        intro=ZONE_INTROS["C"],
        panels=(_fold_readiness(ctx), _synthetic_coverage(ctx), _alignment(ctx)),
    )
