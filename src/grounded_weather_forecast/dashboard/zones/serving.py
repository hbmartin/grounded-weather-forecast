"""Zone F: serving and self-verification."""

from datetime import timedelta

import polars as pl

from grounded_weather_forecast.contracts import finite_number
from grounded_weather_forecast.dashboard.charts import bar_chart, stacked_area
from grounded_weather_forecast.dashboard.context import DashboardContext
from grounded_weather_forecast.dashboard.copy import PANEL_COPY, ZONE_INTROS
from grounded_weather_forecast.dashboard.derive import Derived
from grounded_weather_forecast.dashboard.model import Panel, Stat, TableSpec, Zone
from grounded_weather_forecast.dashboard.zones.common import empty_panel, fmt

_DEGRADED_WINDOW_DAYS = 1.0
_DEGRADED_AMBER = 0.10
_DEGRADED_RED = 0.50


def _degraded_share(history: pl.DataFrame) -> float | None:
    """Degraded fraction of the rows that carry a selection at all.

    Minutely rows are written with ``selection_reason=None`` -- they are
    interpolated from the hourly path, not selected -- and they outnumber the
    selected rows several to one. Counting them in the denominator turned a
    wholly degraded forecast into a single-digit percentage. ``None`` means
    nothing here was selectable, which is not the same as nothing degraded.
    """
    selected = history.filter(pl.col("selection_reason").is_not_null())
    if selected.is_empty():
        return None
    degraded = selected.filter(
        pl.col("selection_reason").str.starts_with("degraded")
        | pl.col("selection_reason").str.starts_with("no backtest evidence")
    ).height
    return degraded / selected.height


def _verification_panel(ctx: DashboardContext, derived: Derived) -> Panel:
    live = derived.verification
    if live.is_empty():
        if derived.live_scores_unusable:
            return empty_panel(
                "f1",
                "f1",
                "Served vs realized (mae_gap)",
                "red",
                "live scores exist but could not be read or scored — this is a "
                "damaged artifact, not a young archive; check scores_*_live.parquet",
            )
        return empty_panel(
            "f1",
            "f1",
            "Served vs realized (mae_gap)",
            "info",
            "not enough realized served forecasts to verify (each slice "
            "needs >= 5 rows whose truth has since arrived)",
        )
    factor = ctx.config.promotion.live_gap_factor
    labels: list[str] = []
    live_maes: list[float | None] = []
    backtest_maes: list[float | None] = []
    rows: list[tuple[str, ...]] = []
    classes: list[tuple[str, ...]] = []
    worst = "ok"
    for row in live.iter_rows(named=True):
        label = (
            f"{row['product']}.{row['variable']}."
            f"{row['lead_bucket']}.{row['method_id']}"
        )
        labels.append(label)
        # `is not None` admits NaN and every NaN comparison is False, so a
        # broken row rendered green while its cell literally read "nan".
        # A genuinely absent backtest_mae is different: `compare_to_backtest`
        # left-joins, so a slice with no board row is simply not comparable
        # and must not raise the verdict.
        raw = (row["live_mae"], row.get("backtest_mae"), row.get("mae_gap"))
        live_mae, backtest_mae, gap = (finite_number(value) for value in raw)
        live_maes.append(live_mae)
        backtest_maes.append(backtest_mae)
        broken = any(
            value is not None and finite_number(value) is None for value in raw
        )
        gap_class = ""
        if broken:
            gap_class = "cell-unknown"
            worst = "amber" if worst == "ok" else worst
        elif live_mae is not None and backtest_mae is not None and gap is not None:
            if live_mae > factor * backtest_mae:
                gap_class = "cell-bad"
                worst = "red"
            elif gap > 0:
                gap_class = "div-p1"
                worst = "amber" if worst == "ok" else worst
        rows.append(
            (
                label,
                fmt(row.get("n")),
                fmt(row.get("live_mae"), 3),
                fmt(row.get("backtest_mae"), 3),
                fmt(gap, 3),
                fmt(row.get("live_bias"), 3),
            )
        )
        classes.append(("", "", "", "", gap_class, ""))
    return Panel(
        panel_id="f1",
        title="Served vs realized (mae_gap)",
        status=worst,
        copy=PANEL_COPY["f1"],
        chart=bar_chart(
            labels,
            [("live MAE", live_maes), ("backtest MAE", backtest_maes)],
            y_label="MAE",
        ),
        table=TableSpec(
            columns=("slice", "n", "live_mae", "backtest_mae", "mae_gap", "live_bias"),
            rows=tuple(rows),
            cell_classes=tuple(classes),
        ),
    )


def _reasons_panel(ctx: DashboardContext) -> Panel:
    history = ctx.history
    if history is None or history.is_empty():
        return empty_panel(
            "f2",
            "f2",
            "Selection reasons over time",
            "info",
            "no served history yet — every `predict` run appends its "
            "selections to predict_history.parquet",
        )
    daily = (
        history.with_columns(pl.col("issued_at").dt.date().alias("date"))
        .group_by("date", "selection_reason")
        .len()
        .sort("date")
    )
    dates = sorted({str(value) for value in daily["date"].to_list()})
    reason_totals = (
        daily.filter(pl.col("selection_reason").is_not_null())
        .group_by("selection_reason")
        .agg(pl.col("len").sum().alias("total"))
        .sort(["total", "selection_reason"], descending=[True, False])
        .head(8)
    )
    reasons = [str(value) for value in reason_totals["selection_reason"].to_list()]
    series = []
    for reason in reasons:
        counts_by_date = {
            str(row["date"]): int(row["len"])
            for row in daily.filter(pl.col("selection_reason") == reason).iter_rows(
                named=True
            )
        }
        series.append((reason, [float(counts_by_date.get(date, 0)) for date in dates]))
    lifetime_share = _degraded_share(history)
    # Anchor the window to wall-clock, not to the newest served row: anchoring
    # to the history would report "the last day the system happened to serve"
    # under a label promising the last day. A stale archive has no recent
    # rows to judge, and saying so beats reporting a confident 0%; zone A owns
    # the "nothing served recently" verdict.
    recent = history.filter(
        pl.col("issued_at").is_between(
            ctx.now - timedelta(days=_DEGRADED_WINDOW_DAYS),
            ctx.now,
            closed="both",
        )
    )
    recent_share = _degraded_share(recent)
    # Judge on the trailing window: a lifetime share is diluted by every
    # healthy row ever served, so a currently-100%-degraded system can read
    # green forever once enough history has accumulated.
    status = (
        "ok"
        if recent_share is None
        else "red"
        if recent_share >= _DEGRADED_RED
        else "amber"
        if recent_share >= _DEGRADED_AMBER
        else "ok"
    )
    return Panel(
        panel_id="f2",
        title="Selection reasons over time",
        status=status,
        copy=PANEL_COPY["f2"],
        stats=(
            Stat("served rows", f"{history.height:,}"),
            Stat(
                f"degraded share (last {_DEGRADED_WINDOW_DAYS:.0f}d)",
                "—" if recent_share is None else f"{recent_share:.0%}",
                status,
            ),
            Stat(
                "degraded share (lifetime)",
                "—" if lifetime_share is None else f"{lifetime_share:.0%}",
            ),
        ),
        chart=stacked_area(dates, series, y_label="served slices"),
    )


def _releases_panel(ctx: DashboardContext) -> Panel:
    if not ctx.releases:
        return empty_panel(
            "f3",
            "f3",
            "Release lineage",
            "amber",
            "no promotion has ever occurred — either the archive is too "
            "young for live folds, or no slice has met the promotion gates",
        )
    manifest_print = (
        str(ctx.manifest.get("fingerprint", "unknown")) if ctx.manifest else "unknown"
    )
    rows: list[tuple[str, ...]] = []
    classes: list[tuple[str, ...]] = []
    stale = 0
    for release in sorted(
        ctx.releases, key=lambda r: str(r.get("promoted_at", "")), reverse=True
    ):
        dataset_print = str(release.get("dataset_fingerprint", ""))
        selections = release.get("selections")
        evaluation_ids = release.get("evaluation_ids")
        mismatched = dataset_print != manifest_print
        stale += int(mismatched)
        rows.append(
            (
                str(release.get("release_id", "")),
                str(release.get("promoted_at", "")),
                dataset_print,
                str(len(selections)) if isinstance(selections, dict) else "0",
                str(release.get("training_cutoff", "")),
                ", ".join(str(value) for value in evaluation_ids)
                if isinstance(evaluation_ids, list)
                else "",
            )
        )
        classes.append(("", "", "cell-bad" if mismatched else "", "", "", ""))
    newest_stale = bool(classes and classes[0][2])
    return Panel(
        panel_id="f3",
        title="Release lineage",
        status="amber" if newest_stale else "ok",
        copy=PANEL_COPY["f3"],
        stats=(
            Stat("releases", str(len(rows))),
            Stat("current manifest", manifest_print),
        ),
        table=TableSpec(
            columns=(
                "release_id",
                "promoted_at",
                "dataset_fingerprint",
                "slices",
                "training_cutoff",
                "evaluation_ids",
            ),
            rows=tuple(rows),
            cell_classes=tuple(classes),
        ),
    )


def build(ctx: DashboardContext, derived: Derived) -> Zone:
    return Zone(
        zone_id="F",
        title="Serving & self-verification",
        intro=ZONE_INTROS["F"],
        panels=(
            _verification_panel(ctx, derived),
            _reasons_panel(ctx),
            _releases_panel(ctx),
        ),
    )
