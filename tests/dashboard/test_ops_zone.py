"""Zone I: operations panels reading the edge ledgers and the run log."""

from datetime import UTC, datetime, timedelta

import polars as pl
from conftest import write_config

from grounded_weather_forecast.dashboard.context import collect_context
from grounded_weather_forecast.dashboard.derive import derive
from grounded_weather_forecast.dashboard.zones import operations as ops_zone
from grounded_weather_forecast.reports.evidence import (
    BUILD_FUNNEL_LEDGER,
    CHANGES_LEDGER,
    EVALUATIONS_LEDGER,
    PIPELINE_LEDGER,
    PROVIDER_HEALTH_LEDGER,
    append_ledger,
    ledger_path,
)
from grounded_weather_forecast.runs import RUNS_SCHEMA

NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)


def build_zone(ctx):
    return ops_zone.build(ctx, derive(ctx))


def context_with(tmp_path, entries=(), runs_rows=()):
    config = write_config(tmp_path)
    for spec, rows in entries:
        frame = pl.DataFrame(
            [{c: row.get(c) for c in spec.schema.names()} for row in rows],
            schema=dict(spec.schema),
        )
        append_ledger(frame, ledger_path(config, spec), spec, now=NOW)
    if runs_rows:
        filled = [
            {column: row.get(column) for column in RUNS_SCHEMA.names()}
            for row in runs_rows
        ]
        config.dataset.dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(filled, schema=dict(RUNS_SCHEMA)).write_parquet(
            config.dataset.dir / "runs.parquet"
        )
    return collect_context(config, now=NOW)


def pipeline_row(day, alarms=""):
    return {
        "recorded_at": NOW - timedelta(days=day),
        "as_of_date": (NOW - timedelta(days=day)).date(),
        "truth_age_minutes": 1.0,
        "collector_age_minutes": 20.0,
        "served_history_age_minutes": 40.0,
        "forecast_document_age_minutes": 40.0,
        "alarms": alarms,
    }


def test_young_history_renders_placeholders(tmp_path):
    zone = build_zone(context_with(tmp_path))
    assert zone.zone_id == "I"
    assert [panel.panel_id for panel in zone.panels] == [
        "i1",
        "i2",
        "i3",
        "i4",
        "i5",
        "i6",
    ]
    assert all(panel.empty_reason is not None for panel in zone.panels)


def test_freshness_panel_is_red_with_active_alarms(tmp_path):
    ctx = context_with(
        tmp_path,
        entries=[
            (
                PIPELINE_LEDGER,
                [pipeline_row(1), pipeline_row(0, alarms="truth stale (300m > 120m)")],
            )
        ],
    )
    panel = build_zone(ctx).panels[0]
    assert panel.status == "red"
    assert "truth stale" in (panel.intro or "")
    assert panel.chart is not None


def test_freshness_panel_is_ok_when_quiet(tmp_path):
    ctx = context_with(
        tmp_path, entries=[(PIPELINE_LEDGER, [pipeline_row(1), pipeline_row(0)])]
    )
    panel = build_zone(ctx).panels[0]
    assert panel.status == "ok"
    assert all(stat.status == "ok" for stat in panel.stats)


def test_provider_panel_ambers_on_poor_success_rate(tmp_path):
    rows = [
        {
            "recorded_at": NOW,
            "as_of_date": NOW.date(),
            "provider": provider,
            "runs_24h": 4,
            "ok_24h": ok,
            "success_rate": ok / 4,
            "median_latency_ms": 100.0,
            "daily_rows_24h": 10,
            "max_daily_lead_days": 6.0,
            "max_hourly_lead_hours": 48.0,
        }
        for provider, ok in (("alpha", 4), ("beta", 2))
    ]
    ctx = context_with(tmp_path, entries=[(PROVIDER_HEALTH_LEDGER, rows)])
    panel = build_zone(ctx).panels[1]
    assert panel.status == "amber"
    assert panel.table is not None
    assert len(panel.table.rows) == 2


def run_row(day, command, minutes, exit_code=0):
    started = NOW - timedelta(days=day)
    return {
        "run_id": f"{command}{day}",
        "command": command,
        "started_at": started,
        "duration_ms": int(minutes * 60_000),
        "exit_code": exit_code,
    }


def test_runtime_panel_ambers_on_a_slow_report(tmp_path):
    rows = [run_row(day, "report", 10.0) for day in (3, 2, 1)]
    rows.append(run_row(0, "report", 30.0))
    panel = build_zone(context_with(tmp_path, runs_rows=rows)).panels[2]
    # 30 > 1.5 x median(10, 10, 10)
    assert panel.status == "amber"
    assert panel.stats[0].value == "30.0m"


def test_runtime_panel_stays_ok_on_flat_history(tmp_path):
    rows = [run_row(day, "report", 10.0) for day in (2, 1, 0)]
    panel = build_zone(context_with(tmp_path, runs_rows=rows)).panels[2]
    assert panel.status == "ok"


def test_funnel_footprint_and_changes_render_tables(tmp_path):
    funnel_row = {
        "recorded_at": NOW,
        "as_of_date": NOW.date(),
        "granularity": "daily",
        "source": "alpha",
        "collector_rows": 100,
        "collector_max_lead": 7.0,
        "long_rows": 90,
        "long_max_lead": 7.0,
        "matrix_rows": 80,
        "matrix_max_lead": 6.0,
        "matrix_native_max_lead": 6.0,
        "matrix_path_max_lead": 2.0,
    }
    catalog_row = {
        "recorded_at": NOW,
        "evaluation_id": "e1",
        "file_name": "scores_hourly_live_expanding_e1.parquet",
        "product": "hourly",
        "rows": 1000,
        "file_size_mb": 12.5,
    }
    change_row = {
        "recorded_at": NOW,
        "as_of_date": NOW.date(),
        "kind": "config",
        "from_value": "aaa",
        "to_value": "bbb",
        "detail": "promotion.alpha",
    }
    ctx = context_with(
        tmp_path,
        entries=[
            (BUILD_FUNNEL_LEDGER, [funnel_row]),
            (EVALUATIONS_LEDGER, [catalog_row]),
            (CHANGES_LEDGER, [change_row]),
        ],
    )
    zone = build_zone(ctx)
    funnel_panel, footprint_panel, changes_panel = zone.panels[3:]
    assert funnel_panel.table is not None
    assert funnel_panel.table.rows[0][0] == "daily"
    assert footprint_panel.stats[0].value == "1"
    assert "12 MB" in footprint_panel.stats[1].value
    assert changes_panel.table is not None
    assert changes_panel.table.rows[0][1] == "config"
    assert "promotion.alpha" in changes_panel.table.rows[0][4]
