"""Zone H: quality over time, read from the artifacts/history ledgers.

Every panel is a trend view over an append-only ledger the nightly report
maintains (``reports/evidence.py``); an empty ledger renders a "young
history" placeholder, never an alarm — absence of history is a young
deployment, not a fault (corrupt ledger files surface separately through
``unreadable_artifacts``).
"""

import math
from collections.abc import Mapping, Sequence

import polars as pl

from grounded_weather_forecast.dashboard.charts import bar_chart, line_chart
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

_MAX_SERIES = 8
_MAX_POINTS = 60
_TOP_PAIRS = 6
_DIFF_ROWS = 20
_CHURN_AMBER = 0.30
_CHURN_RED = 0.60
_TREND_AMBER = 1.15
_GAP_AMBER_DAYS = 4
_AGREE_AMBER = 0.80
_MIN_TREND_N = 8
_COVERAGE_TARGET = 0.80
_COVERAGE_AMBER = 0.10


def _densify(labels: Sequence[str], values: Mapping[str, float]) -> list[float | None]:
    return [values.get(label) for label in labels]


def _series_is_worsening(points: Sequence[float | None]) -> bool:
    finite = [point for point in points if point is not None]
    if len(finite) < 2:
        return False
    prior = finite[-8:-1]
    if not prior:
        return False
    median = sorted(prior)[len(prior) // 2]
    return median > 0.0 and finite[-1] > _TREND_AMBER * median


def _backtest_trend_panel(ctx: DashboardContext) -> Panel:
    quality = ctx.evidence_history.get("quality", pl.DataFrame())
    usable = (
        quality.filter(
            (pl.col("source_kind") == "live")
            & (pl.col("n") >= _MIN_TREND_N)
            & pl.col("recent_mae").is_not_null()
        )
        if not quality.is_empty()
        else quality
    )
    if usable.is_empty():
        return empty_panel(
            "h1",
            "h1",
            "Recent-window backtest MAE",
            "info",
            "young history — the quality ledger has no live rows yet",
        )
    per_slice = usable.group_by(
        ["evaluation_id", "evaluation_created_at", "product", "variable", "lead_bucket"]
    ).agg(
        pl.col("recent_mae").min().alias("bucket_best"),
        pl.col("recent_n").max().alias("bucket_n"),
    )
    per_variable = (
        per_slice.drop_nulls(["bucket_best", "bucket_n"])
        .group_by(["evaluation_id", "evaluation_created_at", "product", "variable"])
        .agg(
            (
                (pl.col("bucket_best") * pl.col("bucket_n")).sum()
                / pl.col("bucket_n").sum()
            ).alias("recent_mae"),
            pl.col("bucket_n").sum().alias("weight"),
        )
        .with_columns(
            (pl.col("product") + "." + pl.col("variable")).alias("series"),
            pl.col("evaluation_created_at").dt.date().cast(pl.String).alias("label"),
        )
        .sort("evaluation_created_at")
    )
    labels = list(dict.fromkeys(per_variable["label"].to_list()))[-_MAX_POINTS:]
    top = (
        per_variable.group_by("series")
        .agg(pl.col("weight").sum().alias("total"))
        .sort(["total", "series"], descending=[True, False])
        .head(_MAX_SERIES)["series"]
        .to_list()
    )
    series = []
    worsening = 0
    for name in top:
        points = dict(
            per_variable.filter(pl.col("series") == name)
            .select("label", "recent_mae")
            .iter_rows()
        )
        values = _densify(labels, points)
        worsening += int(_series_is_worsening(values))
        series.append((name, values))
    status: PanelStatus = "info" if len(labels) < 2 else "amber" if worsening else "ok"
    return Panel(
        panel_id="h1",
        title="Recent-window backtest MAE",
        status=status,
        copy=PANEL_COPY["h1"],
        stats=(
            Stat("series", str(len(series))),
            Stat("evaluations", str(len(labels))),
            Stat("worsening", str(worsening), "amber" if worsening else "ok"),
        ),
        chart=line_chart(labels, series, y_label="recent MAE (14d window)"),
    )


def _churn_panel(ctx: DashboardContext) -> Panel:
    churn = ctx.evidence_history.get("churn", pl.DataFrame())
    if churn.is_empty():
        return empty_panel(
            "h2",
            "h2",
            "Selection churn per release",
            "info",
            "young history — fewer than two promoted releases recorded",
        )
    transitions = (
        churn.group_by(["from_release_id", "to_release_id", "to_promoted_at"])
        .agg(
            pl.col("changed").mean().alias("rate"),
            pl.len().alias("slices"),
        )
        .sort("to_promoted_at")
        .tail(30)
    )
    labels = [str(value)[:16] for value in transitions["to_promoted_at"].to_list()]
    rates = [float(value) for value in transitions["rate"].to_list()]
    latest = rates[-1]
    status: PanelStatus = (
        "red" if latest >= _CHURN_RED else "amber" if latest >= _CHURN_AMBER else "ok"
    )
    newest_to = transitions["to_release_id"][-1]
    diff = (
        churn.filter((pl.col("to_release_id") == newest_to) & pl.col("changed"))
        .sort("slice_key")
        .head(_DIFF_ROWS)
    )
    table = TableSpec(
        columns=("slice", "from", "to", "from mae", "to mae"),
        rows=tuple(
            (
                row["slice_key"],
                fmt(row["from_method"]),
                fmt(row["to_method"]),
                fmt(row["from_mae"], 3),
                fmt(row["to_mae"], 3),
            )
            for row in diff.iter_rows(named=True)
        ),
    )
    return Panel(
        panel_id="h2",
        title="Selection churn per release",
        status=status,
        copy=PANEL_COPY["h2"],
        stats=(
            Stat("latest churn", f"{latest:.0%}", status),
            Stat("transitions", str(transitions.height)),
        ),
        chart=bar_chart(labels, [("churn rate", rates)], y_label="changed share"),
        table=table,
    )


def _served_promise_panel(ctx: DashboardContext) -> Panel:
    served = ctx.evidence_history.get("served_quality", pl.DataFrame())
    usable = (
        served.drop_nulls(["backtest_mae", "live_mae", "n"])
        if not served.is_empty()
        else served
    )
    if usable.is_empty():
        return empty_panel(
            "h3",
            "h3",
            "Served MAE vs backtest promise",
            "info",
            "young history — no served-vs-promise rows recorded yet",
        )
    pooled = (
        usable.group_by(["as_of_date", "product"])
        .agg(
            ((pl.col("live_mae") * pl.col("n")).sum() / pl.col("n").sum()).alias(
                "live"
            ),
            ((pl.col("backtest_mae") * pl.col("n")).sum() / pl.col("n").sum()).alias(
                "backtest"
            ),
        )
        .sort("as_of_date")
        .with_columns(pl.col("as_of_date").cast(pl.String).alias("label"))
    )
    labels = list(dict.fromkeys(pooled["label"].to_list()))[-_MAX_POINTS:]
    products = sorted(set(pooled["product"].to_list()))[:4]
    series = []
    status: PanelStatus = "ok" if len(labels) >= 2 else "info"
    stats: list[Stat] = []
    for product in products:
        rows = pooled.filter(pl.col("product") == product)
        live = dict(rows.select("label", "live").iter_rows())
        backtest = dict(rows.select("label", "backtest").iter_rows())
        series.append((f"{product} live", _densify(labels, live)))
        series.append((f"{product} backtest", _densify(labels, backtest)))
        recent = rows.tail(7)
        gaps = [
            float(row["live"]) - float(row["backtest"])
            for row in recent.iter_rows(named=True)
        ]
        latest = rows.row(rows.height - 1, named=True)
        gap = float(latest["live"]) - float(latest["backtest"])
        stats.append(Stat(f"{product} gap", fmt(gap, 3)))
        factor = ctx.config.promotion.live_gap_factor
        if latest["backtest"] and float(latest["live"]) > factor * float(
            latest["backtest"]
        ):
            status = "red"
        elif (
            status != "red"
            and sum(1 for value in gaps if value > 0.0) >= _GAP_AMBER_DAYS
        ):
            status = "amber"
    return Panel(
        panel_id="h3",
        title="Served MAE vs backtest promise",
        status=status,
        copy=PANEL_COPY["h3"],
        stats=tuple(stats),
        chart=line_chart(labels, series, y_label="MAE"),
    )


def _verdicts_panel(ctx: DashboardContext) -> Panel:
    verdicts = ctx.evidence_history.get("verdicts", pl.DataFrame())
    if verdicts.is_empty():
        return empty_panel(
            "h4",
            "h4",
            "A/B verdicts over time",
            "info",
            "young history — no recalibration or gate verdicts recorded yet",
        )
    daily = (
        verdicts.with_columns(
            pl.col("recorded_at").dt.date().cast(pl.String).alias("label")
        )
        .group_by(["label", "product", "name"])
        .agg(pl.col("value").mean().alias("value"))
        .sort("label")
    )
    labels = list(dict.fromkeys(daily["label"].to_list()))[-_MAX_POINTS:]
    products = sorted(set(daily["product"].to_list()))[:4]
    names = (
        (
            "recalib_win_share_raw",
            "recalib_win_share_pit",
            "recalib_win_share_cqr",
            "gate_agree_rate",
        )
        if len(products) == 1
        else ("recalib_win_share_cqr", "gate_agree_rate")
    )
    series = []
    latest_agree: float | None = None
    latest_cqr: float | None = None
    for product in products:
        for name in names:
            points = dict(
                daily.filter((pl.col("product") == product) & (pl.col("name") == name))
                .select("label", "value")
                .iter_rows()
            )
            if not points:
                continue
            short = name.replace("recalib_win_share_", "").replace(
                "gate_agree_rate", "gate agree"
            )
            series.append((f"{product} {short}", _densify(labels, points)))
            newest = points.get(labels[-1]) if labels else None
            if name == "gate_agree_rate" and newest is not None:
                latest_agree = newest
            if name == "recalib_win_share_cqr" and newest is not None:
                latest_cqr = newest
    status: PanelStatus = (
        "amber"
        if latest_agree is not None and latest_agree < _AGREE_AMBER
        else "info"
        if len(labels) < 2
        else "ok"
    )
    return Panel(
        panel_id="h4",
        title="A/B verdicts over time",
        status=status,
        copy=PANEL_COPY["h4"],
        stats=(
            Stat("cqr win share", fmt(latest_cqr)),
            Stat(
                "gate agree",
                fmt(latest_agree),
                "amber" if status == "amber" else "ok",
            ),
        ),
        chart=line_chart(labels, series, y_label="share / rate"),
    )


def _wealth_panel(ctx: DashboardContext) -> Panel:
    wealth = ctx.evidence_history.get("eprocess_wealth", pl.DataFrame())
    if wealth.is_empty():
        return empty_panel(
            "h5",
            "h5",
            "E-process wealth trajectories",
            "info",
            "young history — no e-process snapshots recorded yet",
        )
    current_era = wealth.join(
        wealth.group_by("pair_key").agg(pl.col("resets").max()),
        on=["pair_key", "resets"],
        how="inner",
    ).sort("recorded_at")
    latest = (
        current_era.group_by("pair_key")
        .agg(pl.col("log_e").last().alias("latest"), pl.col("resets").max())
        .sort(["latest", "pair_key"], descending=[True, False])
        .head(_TOP_PAIRS)
    )
    threshold = math.log(1.0 / ctx.config.promotion.alpha)
    labels = list(
        dict.fromkeys(
            current_era.with_columns(
                pl.col("recorded_at").dt.date().cast(pl.String).alias("label")
            )["label"].to_list()
        )
    )[-_MAX_POINTS:]
    series = []
    for pair_key in latest["pair_key"].to_list():
        rows = current_era.filter(pl.col("pair_key") == pair_key).with_columns(
            pl.col("recorded_at").dt.date().cast(pl.String).alias("label")
        )
        points: dict[str, float] = {}
        for row in rows.iter_rows(named=True):
            points[row["label"]] = float(row["log_e"])
        series.append((_short_pair(pair_key), _densify(labels, points)))
    above = int((latest["latest"] > threshold).sum())
    return Panel(
        panel_id="h5",
        title="E-process wealth trajectories",
        status="ok",
        copy=PANEL_COPY["h5"],
        stats=(
            Stat("pairs tracked", str(int(wealth["pair_key"].n_unique()))),
            Stat("above threshold", str(above)),
        ),
        chart=line_chart(
            labels,
            series,
            y_label="log wealth",
            reference=("log(1/alpha)", threshold),
        ),
    )


def _short_pair(pair_key: str) -> str:
    parts = pair_key.split("|")
    if len(parts) != 6:
        return pair_key[:40]
    _product, variable, _semantics, bucket, candidate, reference = parts
    return f"{variable}.{bucket} {candidate}~{reference}"


def _recent_coverage_panel(ctx: DashboardContext) -> Panel:
    quality = ctx.evidence_history.get("quality", pl.DataFrame())
    usable = (
        quality.filter(
            (pl.col("source_kind") == "live")
            & pl.col("recent_coverage80").is_not_null()
        )
        if "recent_coverage80" in quality.columns
        else pl.DataFrame()
    )
    if usable.is_empty():
        return empty_panel(
            "h6",
            "h6",
            "Recent-window interval coverage",
            "info",
            "young history — no recent-window coverage recorded yet",
        )
    per_method = (
        usable.with_columns(pl.col("recent_n").fill_null(pl.col("n")).alias("weight"))
        .group_by(["evaluation_id", "evaluation_created_at", "product", "method_id"])
        .agg(
            (
                (pl.col("recent_coverage80") * pl.col("weight")).sum()
                / pl.col("weight").sum()
            ).alias("coverage80"),
            pl.col("weight").sum().alias("weight"),
        )
        .with_columns(
            (pl.col("product") + "." + pl.col("method_id")).alias("series"),
            pl.col("evaluation_created_at").dt.date().cast(pl.String).alias("label"),
        )
        .sort("evaluation_created_at")
    )
    labels = list(dict.fromkeys(per_method["label"].to_list()))[-_MAX_POINTS:]
    top = (
        per_method.group_by("series")
        .agg(pl.col("weight").sum().alias("total"))
        .sort(["total", "series"], descending=[True, False])
        .head(_MAX_SERIES)["series"]
        .to_list()
    )
    series = []
    off_target = 0
    for name in top:
        points = dict(
            per_method.filter(pl.col("series") == name)
            .select("label", "coverage80")
            .iter_rows()
        )
        values = _densify(labels, points)
        newest = next((value for value in reversed(values) if value is not None), None)
        if newest is not None and abs(newest - _COVERAGE_TARGET) > _COVERAGE_AMBER:
            off_target += 1
        series.append((name, values))
    status: PanelStatus = "info" if len(labels) < 2 else "amber" if off_target else "ok"
    return Panel(
        panel_id="h6",
        title="Recent-window interval coverage",
        status=status,
        copy=PANEL_COPY["h6"],
        stats=(
            Stat("interval methods", str(len(series))),
            Stat(
                "off target",
                str(off_target),
                "amber" if off_target else "ok",
            ),
        ),
        chart=line_chart(
            labels,
            series,
            y_label="coverage80 (14d window)",
            reference=("target 0.80", _COVERAGE_TARGET),
        ),
    )


def build(ctx: DashboardContext, derived: Derived) -> Zone:  # noqa: ARG001
    return Zone(
        zone_id="H",
        title="Quality over time",
        intro=ZONE_INTROS["H"],
        panels=(
            _backtest_trend_panel(ctx),
            _churn_panel(ctx),
            _served_promise_panel(ctx),
            _verdicts_panel(ctx),
            _wealth_panel(ctx),
            _recent_coverage_panel(ctx),
        ),
    )
