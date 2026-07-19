"""Zone B: ingestion and data trust."""

from collections import defaultdict
from datetime import date

import polars as pl

from grounded_weather_forecast.dashboard.charts import bar_chart, sequential_class
from grounded_weather_forecast.dashboard.context import DashboardContext
from grounded_weather_forecast.dashboard.copy import PANEL_COPY, ZONE_INTROS
from grounded_weather_forecast.dashboard.derive import Derived
from grounded_weather_forecast.dashboard.model import Panel, Stat, TableSpec, Zone
from grounded_weather_forecast.dashboard.zones.common import empty_panel

_CALENDAR_DAYS = 45


def _station_qc(ctx: DashboardContext) -> Panel:
    qc = ctx.qc
    if qc is None or qc.is_empty():
        return empty_panel(
            "b1",
            "b1",
            "Station QC flags",
            "amber",
            "no QC summary — the station database is unreadable or empty",
        )
    channels = qc["channel"].to_list()
    series = [
        (flag, [float(value) for value in qc[flag].to_list()])
        for flag in ("out_of_bounds", "spike", "flatline")
    ]
    samples = int(qc["samples"].sum())
    clean = int(qc["clean"].sum())
    flat_channels = (
        qc.filter(pl.col("active_flatline"))["channel"].to_list()
        if "active_flatline" in qc.columns
        else []
    )
    status = "amber" if flat_channels else "ok"
    return Panel(
        panel_id="b1",
        title="Station QC flags",
        status=status,
        copy=PANEL_COPY["b1"],
        stats=(
            Stat("samples", f"{samples:,}"),
            Stat("clean", f"{clean / samples:.1%}" if samples else "—"),
            Stat(
                "flatline channels",
                ", ".join(flat_channels) if flat_channels else "none",
                status,
            ),
        ),
        chart=bar_chart(channels, series, y_label="flagged samples"),
    )


def _coverage_calendar(ctx: DashboardContext) -> Panel:
    frame = ctx.truth_hourly
    if frame is None or frame.is_empty():
        return empty_panel(
            "b2", "b2", "Truth coverage calendar", "amber", "no hourly truth yet"
        )
    coverage_columns = sorted(c for c in frame.columns if c.endswith("_cov"))
    time_column = next(
        (c for c in ("valid_hour", "valid_time", "ts") if c in frame.columns), None
    )
    if not coverage_columns or time_column is None:
        return empty_panel(
            "b2",
            "b2",
            "Truth coverage calendar",
            "amber",
            "hourly truth carries no coverage columns",
        )
    daily = (
        frame.with_columns(pl.col(time_column).dt.date().alias("date"))
        .group_by("date")
        .agg(*(pl.col(c).mean() for c in coverage_columns))
        .sort("date", descending=True)
        .head(_CALENDAR_DAYS)
    )
    floor = ctx.config.dataset.min_hour_coverage
    columns = ("date", *(c.removesuffix("_cov") for c in coverage_columns))
    rows: list[tuple[str, ...]] = []
    classes: list[tuple[str, ...]] = []
    titles: list[tuple[str, ...]] = []
    for row in daily.iter_rows(named=True):
        day = row["date"]
        cells = [day.isoformat() if isinstance(day, date) else str(day)]
        row_classes = [""]
        row_titles = [""]
        for column in coverage_columns:
            value = row[column]
            cells.append(f"{value:.2f}" if value is not None else "")
            row_classes.append(sequential_class(value, 0.0, 1.0))
            row_titles.append(
                f"below the {floor} floor: nulled from truth"
                if value is not None and value < floor
                else ""
            )
        rows.append(tuple(cells))
        classes.append(tuple(row_classes))
        titles.append(tuple(row_titles))
    overall = [
        float(value)
        for column in coverage_columns
        if isinstance(value := daily[column].mean(), (int, float))
    ]
    thin = bool(overall) and min(overall) < floor
    return Panel(
        panel_id="b2",
        title="Truth coverage calendar",
        status="amber" if thin else "ok",
        copy=PANEL_COPY["b2"],
        intro=f"Daily mean coverage, newest first (last {_CALENDAR_DAYS} days).",
        table=TableSpec(
            columns=columns,
            rows=tuple(rows),
            cell_classes=tuple(classes),
            cell_titles=tuple(titles),
        ),
    )


def _provider_nulls(ctx: DashboardContext) -> Panel:
    matrix = ctx.hourly_matrix
    if matrix is None or matrix.is_empty():
        return empty_panel(
            "b3", "b3", "Provider QC nulling", "amber", "no live matrix yet"
        )
    by_source: dict[str, list[float]] = defaultdict(list)
    for column in matrix.columns:
        if not column.startswith("fx__"):
            continue
        source = column.split("__")[1]
        share = matrix[column].null_count() / max(matrix.height, 1)
        by_source[source].append(share)
    if not by_source:
        return empty_panel(
            "b3",
            "b3",
            "Provider QC nulling",
            "amber",
            "the live matrix carries no forecast columns",
        )
    sources = sorted(by_source)
    shares = [sum(by_source[s]) / len(by_source[s]) for s in sources]
    worst = max(shares)
    return Panel(
        panel_id="b3",
        title="Provider QC nulling",
        status="amber" if worst > 0.5 else "ok",
        copy=PANEL_COPY["b3"],
        intro=(
            "Null share includes leads a provider never publishes; compare "
            "providers against each other, not against zero."
        ),
        chart=bar_chart(
            sources, [("null share of forecast cells", shares)], y_label="share"
        ),
    )


def _provenance(ctx: DashboardContext) -> Panel:
    kinds: dict[str, set[str]] = {}
    frames = {
        "hourly matrix (live)": ctx.hourly_matrix,
        "hourly matrix (synthetic)": ctx.synthetic_hourly,
        **{f"scores {stem}": frame for stem, frame in ctx.score_frames.items()},
    }
    for name, frame in frames.items():
        if frame is None or frame.is_empty() or "source_kind" not in frame.columns:
            continue
        kinds[name] = {str(value) for value in frame["source_kind"].unique()}
    if not kinds:
        return empty_panel(
            "b4",
            "b4",
            "Provenance wall",
            "info",
            "no provenance-carrying frames on disk yet",
        )
    mixed = {name for name, values in kinds.items() if len(values) > 1}
    rows = tuple(
        (name, ", ".join(sorted(values))) for name, values in sorted(kinds.items())
    )
    return Panel(
        panel_id="b4",
        title="Provenance wall",
        status="red" if mixed else "ok",
        copy=PANEL_COPY["b4"],
        stats=(
            Stat(
                "live/synthetic pooling",
                "VIOLATED" if mixed else "never",
                "red" if mixed else "ok",
            ),
        ),
        table=TableSpec(columns=("frame", "source_kind"), rows=rows),
    )


def build(ctx: DashboardContext, derived: Derived) -> Zone:  # noqa: ARG001
    return Zone(
        zone_id="B",
        title="Ingestion & data trust",
        intro=ZONE_INTROS["B"],
        panels=(
            _station_qc(ctx),
            _coverage_calendar(ctx),
            _provider_nulls(ctx),
            _provenance(ctx),
        ),
    )
