"""Assemble and write the self-contained dashboard page."""

import json
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path

import polars as pl

from grounded_weather_forecast import __version__
from grounded_weather_forecast.config import Config
from grounded_weather_forecast.contracts import age_col, parse_fx_col
from grounded_weather_forecast.dashboard.context import (
    DashboardContext,
    collect_context,
)
from grounded_weather_forecast.dashboard.derive import Derived, derive
from grounded_weather_forecast.dashboard.html import render_page
from grounded_weather_forecast.dashboard.zones import ALL_ZONES
from grounded_weather_forecast.reports.alerts import AlertInputs, evaluate_alerts

_MINUTELY_CAP = 120
type ProviderInputs = dict[str, dict[str, float | None]]
type VariableInputs = dict[str, ProviderInputs]
type PointInputs = dict[str, VariableInputs]


def _empty() -> pl.DataFrame:
    return pl.DataFrame()


def _as_of_snapshot(
    matrix: pl.DataFrame | None, issue_time: datetime
) -> pl.DataFrame | None:
    if matrix is None or matrix.is_empty() or "issue_time" not in matrix.columns:
        return None
    eligible = matrix.filter(pl.col("issue_time") <= issue_time)
    if eligible.is_empty():
        return None
    return eligible.filter(pl.col("issue_time") == eligible["issue_time"].max())


def _alert_inputs(ctx: DashboardContext, derived: Derived) -> AlertInputs:
    forecast = ctx.latest_forecast
    board = pl.DataFrame()
    if derived.live_stem is not None:
        board = derived.boards[derived.live_stem]
    elif derived.boards:
        board = next(iter(derived.boards.values()))
    return AlertInputs(
        config=ctx.config,
        now=ctx.now,
        manifest=ctx.manifest,
        runs=ctx.runs,
        minute_truth=ctx.truth_minute if ctx.truth_minute is not None else _empty(),
        hourly_truth=ctx.truth_hourly if ctx.truth_hourly is not None else _empty(),
        daily_truth=ctx.truth_daily if ctx.truth_daily is not None else _empty(),
        qc=ctx.qc if ctx.qc is not None else _empty(),
        hourly_matrix=(
            ctx.hourly_matrix if ctx.hourly_matrix is not None else _empty()
        ),
        board=board,
        live_vs_backtest=derived.verification,
        drift=ctx.drift,
        latest_status=(
            (forecast.status, forecast.status_reason) if forecast is not None else None
        ),
        releases=ctx.releases,
        observability_history=ctx.observability_history,
        archive_location=ctx.archive_location,
    )


def _source_ages(
    matrix: pl.DataFrame | None, issue_time: datetime
) -> dict[str, float | None]:
    matching = _as_of_snapshot(matrix, issue_time)
    if matching is None:
        return {}
    row = matching.row(0, named=True)
    return {
        column.removeprefix("age__"): (
            float(value) if isinstance(value := row.get(column), (int, float)) else None
        )
        for column in matching.columns
        if column.startswith("age__")
    }


def _point_key(value: object) -> str:
    match value:
        case datetime() | date():
            return value.isoformat()
        case _:
            return str(value)


def _matrix_inputs(
    matrix: pl.DataFrame | None,
    *,
    issue_time: datetime,
    point_column: str,
    forecast_prefix: str,
    fallback_ages: Mapping[str, float | None],
) -> PointInputs:
    if matrix is None or not {point_column} <= set(matrix.columns):
        return {}
    matching = _as_of_snapshot(matrix, issue_time)
    if matching is None:
        return {}
    forecast_columns = [
        column
        for column in matching.columns
        if column.startswith(f"{forecast_prefix}__")
    ]
    inputs: PointInputs = {}
    for row in matching.iter_rows(named=True):
        point_inputs = inputs.setdefault(_point_key(row[point_column]), {})
        for column in forecast_columns:
            value = row.get(column)
            if not isinstance(value, (int, float)):
                continue
            source, variable = parse_fx_col(column)
            raw_age = row.get(age_col(source), fallback_ages.get(source))
            point_inputs.setdefault(variable, {})[source] = {
                "value": round(float(value), 3),
                "age_hours": (
                    round(float(raw_age), 2)
                    if isinstance(raw_age, (int, float))
                    else None
                ),
            }
    return inputs


def _served_inputs(ctx: DashboardContext) -> dict[str, PointInputs]:
    forecast = ctx.latest_forecast
    if forecast is None:
        return {}
    try:
        issue_time = datetime.fromisoformat(forecast.issued_at)
    except (ValueError,):  # noqa: B013 - project style requires tuple clauses
        return {}
    ages = _source_ages(ctx.hourly_matrix, issue_time)
    return {
        "hourly": _matrix_inputs(
            ctx.hourly_matrix,
            issue_time=issue_time,
            point_column="valid_time",
            forecast_prefix="fx",
            fallback_ages=ages,
        ),
        "daily": _matrix_inputs(
            ctx.daily_matrix,
            issue_time=issue_time,
            point_column="forecast_date",
            forecast_prefix="fxd",
            fallback_ages=ages,
        ),
    }


def _forecast_payload(ctx: DashboardContext) -> dict[str, object] | None:
    if ctx.latest_forecast is None:
        return None
    try:
        document = json.loads(ctx.latest_forecast.to_json())
    except (TypeError, ValueError):
        return None
    minutely = document.get("minutely")
    if isinstance(minutely, list) and len(minutely) > _MINUTELY_CAP:
        document["minutely"] = minutely[:_MINUTELY_CAP]
    return document


def write_dashboard(config: Config, *, now: datetime | None = None) -> Path:
    """Collect artifacts, evaluate alerts, render, and write dashboard.html."""
    ctx = collect_context(config, now=now)
    derived = derive(ctx)
    zones = tuple(build(ctx, derived) for build in ALL_ZONES)
    alerts = evaluate_alerts(_alert_inputs(ctx, derived))
    charts = {
        panel.panel_id: panel.chart.config
        for zone in zones
        for panel in zone.panels
        if panel.chart is not None
    }
    payload: dict[str, object] = {
        "charts": charts,
        "forecast": _forecast_payload(ctx),
        "inputs": _served_inputs(ctx),
    }
    manifest_print = (
        str(ctx.manifest.get("fingerprint", "unknown"))
        if ctx.manifest is not None
        else "unknown"
    )
    page = render_page(
        title="grounded-weather-forecast · operator console",
        generated_at=ctx.now.replace(microsecond=0).isoformat(),
        fingerprints={"dataset": manifest_print},
        version=__version__,
        alerts=alerts,
        zones=zones,
        payload=payload,
    )
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    destination = config.reports_dir / "dashboard.html"
    destination.write_text(page, encoding="utf-8")
    return destination


__all__ = ["write_dashboard"]
