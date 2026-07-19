"""Zone D: evaluation — the leaderboard made operational."""

import json

import numpy as np
import polars as pl

from grounded_weather_forecast.dashboard.charts import (
    bar_chart,
    diverging_class,
    histogram,
    reliability_scatter,
    sequential_class,
)
from grounded_weather_forecast.dashboard.context import DashboardContext
from grounded_weather_forecast.dashboard.copy import PANEL_COPY, ZONE_INTROS
from grounded_weather_forecast.dashboard.derive import Derived
from grounded_weather_forecast.dashboard.model import Panel, Stat, TableSpec, Zone
from grounded_weather_forecast.dashboard.zones.common import empty_panel, fmt
from grounded_weather_forecast.leads import (
    DAILY_BUCKET_LABELS,
    HOURLY_BUCKET_LABELS,
)
from grounded_weather_forecast.metrics.probabilistic import (
    pit_from_quantiles,
    reliability_bins,
)
from grounded_weather_forecast.reports.leaderboard import CONSUMER_TOLERANCES

_MAX_BOARD_ROWS = 250
_BOARD_COLUMNS = (
    "product",
    "variable",
    "lead_bucket",
    "method_id",
    "n",
    "coverage",
    "mae",
    "rmse",
    "bias",
    "pct_within",
    "skill_vs_best_provider",
    "dm_p_vs_best_provider",
    "skill_vs_equal_weight",
    "dm_p_vs_equal_weight",
)
_BASELINE_METHODS = ("climatology", "persistence", "best_provider", "equal_weight")


def _board_panel(stem: str, board: pl.DataFrame) -> Panel:
    columns = [c for c in _BOARD_COLUMNS if c in board.columns]
    trimmed = board.select(columns).head(_MAX_BOARD_ROWS)
    bias_index = columns.index("bias") if "bias" in columns else None
    rows: list[tuple[str, ...]] = []
    classes: list[tuple[str, ...]] = []
    for row in trimmed.iter_rows(named=True):
        cells = tuple(fmt(row[c], 3) for c in columns)
        row_classes = [""] * len(columns)
        if bias_index is not None:
            tolerance = CONSUMER_TOLERANCES.get(str(row["variable"]), 1.0)
            row_classes[bias_index] = diverging_class(row["bias"], tolerance)
        rows.append(cells)
        classes.append(tuple(row_classes))
    dropped = board.height - trimmed.height
    return Panel(
        panel_id=f"d1-{stem}",
        title=f"Leaderboard — {stem.removeprefix('scores_')}",
        status="ok",
        copy=PANEL_COPY["d1"],
        intro=(
            f"showing the first {_MAX_BOARD_ROWS} of {board.height} rows; the "
            f"full table is in reports/leaderboard_{stem.removeprefix('scores_')}.md"
            if dropped > 0
            else None
        ),
        stats=(Stat("rows", str(board.height)),),
        table=TableSpec(
            columns=tuple(columns), rows=tuple(rows), cell_classes=tuple(classes)
        ),
    )


def _winners_panel(stem: str, winners: pl.DataFrame, rule: str, alpha: float) -> Panel:
    rows = tuple(
        tuple(fmt(row[c], 3) for c in winners.columns)
        for row in winners.iter_rows(named=True)
    )
    reference_wins = winners.filter(
        pl.col("method_id").is_in(["best_provider", "equal_weight"])
    ).height
    return Panel(
        panel_id=f"d2-{stem}",
        title=f"Slice winners — {stem.removeprefix('scores_')}",
        status="ok",
        copy=PANEL_COPY["d2"],
        stats=(
            Stat("gate", f"{rule} @ alpha={alpha}"),
            Stat(
                "reference fallbacks",
                f"{reference_wins}/{winners.height}" if winners.height else "0/0",
            ),
        ),
        table=TableSpec(columns=tuple(winners.columns), rows=rows),
    )


def _baseline_panel(stem: str, board: pl.DataFrame) -> Panel | None:
    subset = board.filter(
        (pl.col("variable") == "temp_c")
        & pl.col("method_id").is_in(list(_BASELINE_METHODS))
    ).sort("lead_bucket")
    if subset.is_empty():
        return None
    product = subset["product"][0]
    canonical = DAILY_BUCKET_LABELS if product == "daily" else HOURLY_BUCKET_LABELS
    available = set(subset["lead_bucket"].drop_nulls().to_list())
    buckets = [label for label in canonical if label in available]
    series = []
    for method in _BASELINE_METHODS:
        method_rows = {
            row["lead_bucket"]: row["mae"]
            for row in subset.filter(pl.col("method_id") == method).iter_rows(
                named=True
            )
        }
        if method_rows:
            series.append((method, [method_rows.get(bucket) for bucket in buckets]))
    shortest = buckets[0] if buckets else None
    shortest_rows = (
        subset.filter(pl.col("lead_bucket") == shortest)
        if shortest is not None
        else pl.DataFrame()
    )
    shortest_mae = {
        row["method_id"]: row["mae"]
        for row in shortest_rows.iter_rows(named=True)
        if row["mae"] is not None
    }
    climatology = shortest_mae.get("climatology")
    reference = shortest_mae.get("best_provider")
    suspicious = (
        climatology is not None
        and reference is not None
        and float(climatology) < float(reference)
    )
    return Panel(
        panel_id=f"d3-{stem}",
        title=f"Baseline floor (temp_c) — {stem.removeprefix('scores_')}",
        status="amber" if suspicious else "ok",
        copy=PANEL_COPY["d3"],
        chart=bar_chart([str(b) for b in buckets], series, y_label="MAE"),
    )


def _correlation_panel(derived: Derived) -> Panel:
    correlation = derived.correlation
    if correlation is None or correlation.is_empty():
        return empty_panel(
            "d4",
            "d4",
            "Provider error correlation",
            "info",
            "not enough overlapping scored hours (cells need >= 24) to "
            "correlate provider errors yet",
        )
    sources = [c for c in correlation.columns if c != "source"]
    rows = []
    classes = []
    for row in correlation.iter_rows(named=True):
        cells = [str(row["source"])]
        row_classes = [""]
        for source in sources:
            value = row[source]
            cells.append(fmt(value))
            row_classes.append(
                sequential_class(value, 0.0, 1.0) if value is not None else "heat-none"
            )
        rows.append(tuple(cells))
        classes.append(tuple(row_classes))
    stats = [Stat("providers", str(len(sources)))]
    if derived.k_eff is not None:
        stats.append(Stat("k_eff", f"{derived.k_eff:.1f} of {len(sources)}"))
    return Panel(
        panel_id="d4",
        title="Provider error correlation (temp_c)",
        status="ok",
        copy=PANEL_COPY["d4"],
        stats=tuple(stats),
        intro=(
            "k_eff ~= n / (1 + (n-1)*mean_r): the number of effectively "
            "independent opinions in the ensemble."
        ),
        table=TableSpec(
            columns=("source", *sources),
            rows=tuple(rows),
            cell_classes=tuple(classes),
        ),
    )


def _pit_values(scores: pl.DataFrame) -> np.ndarray:
    needed = {"quantiles_json", "quantile_levels_json", "y_true"}
    if not needed <= set(scores.columns):
        return np.empty(0)
    usable = scores.filter(
        pl.col("quantiles_json").is_not_null() & pl.col("y_true").is_not_null()
    )
    if usable.is_empty():
        return np.empty(0)
    groups: dict[str, list[tuple[list[float], float]]] = {}
    for row in usable.iter_rows(named=True):
        try:
            quantiles = json.loads(row["quantiles_json"])
        except (TypeError, ValueError):
            continue
        if not isinstance(quantiles, list) or any(q is None for q in quantiles):
            continue
        key = str(row["quantile_levels_json"])
        groups.setdefault(key, []).append((quantiles, float(row["y_true"])))
    if not groups:
        return np.empty(0)
    key, entries = max(groups.items(), key=lambda item: len(item[1]))
    try:
        levels = tuple(float(level) for level in json.loads(key))
    except (TypeError, ValueError):
        return np.empty(0)
    quantile_matrix = np.asarray([entry[0] for entry in entries], dtype=np.float64)
    truths = np.asarray([entry[1] for entry in entries], dtype=np.float64)
    if quantile_matrix.shape[1] != len(levels):
        return np.empty(0)
    return pit_from_quantiles(truths, quantile_matrix, levels)


def _calibration_panel(stem: str, scores: pl.DataFrame, board: pl.DataFrame) -> Panel:
    probabilistic = (
        board.filter(pl.col("crps").is_not_null())
        if ("crps" in board.columns)
        else pl.DataFrame()
    )
    pit = _pit_values(scores)
    if probabilistic.is_empty() and pit.size == 0:
        return empty_panel(
            "d5",
            "d5",
            f"Calibration — {stem.removeprefix('scores_')}",
            "info",
            "no distributional method has enough scored rows yet (emos, idr "
            "and conformal_* emit quantiles; point methods cannot calibrate)",
        )
    stats: list[Stat] = []
    for column, label in (
        ("crps", "best CRPS"),
        ("coverage80", "coverage@80"),
        ("coverage90", "coverage@90"),
        ("sharpness", "sharpness"),
    ):
        if column in probabilistic.columns and not probabilistic.is_empty():
            value = (
                probabilistic[column].min()
                if column == "crps"
                else (probabilistic[column].mean())
            )
            if value is not None:
                stats.append(Stat(label, fmt(float(str(value)), 3)))
    chart = None
    if pit.size:
        counts, _edges = np.histogram(pit, bins=10, range=(0.0, 1.0))
        labels = [f"{i / 10:.1f}–{(i + 1) / 10:.1f}" for i in range(10)]
        chart = histogram(
            labels,
            [int(count) for count in counts],
            y_label="rows",
            label="PIT",
        )
        stats.append(Stat("PIT rows", str(pit.size)))
    return Panel(
        panel_id=f"d5-{stem}",
        title=f"Calibration — {stem.removeprefix('scores_')}",
        status="ok",
        copy=PANEL_COPY["d5"],
        intro=(
            "A calibrated forecast has a flat PIT histogram; a U shape means "
            "the bands are too narrow, a hump too wide."
        ),
        stats=tuple(stats),
        chart=chart,
        raw_html="" if chart is None else None,
    )


def _reliability_panel(stem: str, scores: pl.DataFrame) -> Panel | None:
    pop = scores.filter(
        (pl.col("variable") == "pop")
        & pl.col("y_pred").is_not_null()
        & pl.col("y_true").is_not_null()
    )
    if pop.height < 20:
        return None
    forecast = pop["y_pred"].to_numpy().astype(np.float64)
    occurred = pop["y_true"].to_numpy().astype(np.float64)
    bins = reliability_bins(forecast, occurred).filter(pl.col("count") > 0)
    points: list[tuple[float, float]] = []
    counts: list[int] = []
    for row in bins.iter_rows(named=True):
        if row["forecast_mean"] is None or row["observed_freq"] is None:
            continue
        points.append((float(row["forecast_mean"]), float(row["observed_freq"])))
        counts.append(int(row["count"]))
    if not points:
        return None
    return Panel(
        panel_id=f"d5r-{stem}",
        title=f"PoP reliability — {stem.removeprefix('scores_')}",
        status="ok",
        copy=PANEL_COPY["d5"],
        chart=reliability_scatter(points, counts=counts),
    )


def build(ctx: DashboardContext, derived: Derived) -> Zone:
    panels: list[Panel] = []
    if not derived.boards:
        panels.append(
            empty_panel(
                "d1",
                "d1",
                "Leaderboard",
                "info",
                "no backtest scores yet — run `backtest --source synthetic` "
                "against a backfill now, and `backtest --source live` once "
                "the archive has folds",
            )
        )
    for stem, board in derived.boards.items():
        scores = ctx.score_frames[stem]
        panels.append(_board_panel(stem, board))
        winners = derived.winners.get(stem)
        if winners is not None:
            panels.append(
                _winners_panel(
                    stem,
                    winners,
                    ctx.config.promotion.rule,
                    ctx.config.promotion.alpha,
                )
            )
        if (baseline := _baseline_panel(stem, board)) is not None:
            panels.append(baseline)
        panels.append(_calibration_panel(stem, scores, board))
        if (reliability := _reliability_panel(stem, scores)) is not None:
            panels.append(reliability)
    panels.append(_correlation_panel(derived))
    return Zone(
        zone_id="D",
        title="Evaluation",
        intro=ZONE_INTROS["D"],
        panels=tuple(panels),
    )
