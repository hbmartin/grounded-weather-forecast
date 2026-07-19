"""Zone B: ingestion and data trust."""

from collections import defaultdict
from datetime import date

import polars as pl

from grounded_weather_forecast.contracts import finite_number
from grounded_weather_forecast.dashboard.charts import bar_chart, sequential_class
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
from grounded_weather_forecast.dashboard.zones.common import empty_panel

_CALENDAR_DAYS = 45
_FLAGGED_AMBER = 0.05
_FLAGGED_RED = 0.25


def _flagged_status(share: float) -> PanelStatus:
    if share >= _FLAGGED_RED:
        return "red"
    return "amber" if share >= _FLAGGED_AMBER else "ok"


def _channel_flagged_shares(qc: pl.DataFrame) -> list[float | None]:
    """Per-channel flagged share, ``None`` where the channel reported nothing."""
    if "missing" not in qc.columns:
        return []
    shares: list[float | None] = []
    for row in qc.iter_rows(named=True):
        reported = int(row["samples"]) - int(row["missing"])
        shares.append(None if reported <= 0 else 1.0 - int(row["clean"]) / reported)
    return shares


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
    # `clean` counts samples that are both QC_OK and non-null, so the flagged
    # share must be measured against what the station actually reported.
    # Dividing by `samples` would count an uninstalled sensor's nulls as
    # flagged and alarm on a station that raised no QC flag at all; absence is
    # already reported separately by the `missing` column.
    reported = (
        samples - int(qc["missing"].sum()) if "missing" in qc.columns else samples
    )
    flat_channels = (
        qc.filter(pl.col("active_flatline"))["channel"].to_list()
        if "active_flatline" in qc.columns
        else []
    )
    # A stuck sensor is current state; a high flagged share is a calibration
    # problem. Both belong in the verdict — counting only the first renders a
    # channel that is 100% out-of-bounds as green.
    flagged_share = 1.0 - (clean / reported) if reported else 0.0
    # Judge on the WORST channel, not the pooled average. One wholly bad
    # channel among two dozen clean ones is a dead sensor, but it moves the
    # aggregate by only a few percent and rendered the panel green — the exact
    # failure this share was added to catch.
    worst_share = max(
        (share for share in (_channel_flagged_shares(qc)) if share is not None),
        default=flagged_share,
    )
    flagged_status = _flagged_status(max(flagged_share, worst_share))
    status = "amber" if flat_channels and flagged_status == "ok" else flagged_status
    return Panel(
        panel_id="b1",
        title="Station QC flags",
        status=status,
        copy=PANEL_COPY["b1"],
        stats=(
            Stat("samples", f"{samples:,}"),
            Stat(
                "clean",
                f"{clean / reported:.1%}" if reported else "—",
                flagged_status,
            ),
            Stat(
                "flatline channels",
                ", ".join(flat_channels) if flat_channels else "none",
                "amber" if flat_channels else "ok",
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
    # `isinstance` admits NaN and `min([nan]) < floor` is False, so an
    # all-NaN coverage column rendered "ok" while its cells read "nan".
    # Unusable coverage is not healthy coverage.
    means = [finite_number(daily[column].mean()) for column in coverage_columns]
    overall = [value for value in means if value is not None]
    unusable = len(overall) < len(means)
    thin = bool(overall) and min(overall) < floor
    return Panel(
        panel_id="b2",
        title="Truth coverage calendar",
        status="amber" if thin or unusable else "ok",
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
