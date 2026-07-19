from datetime import UTC, datetime, timedelta

import polars as pl
from conftest import synthetic_hourly_matrix, write_config

from grounded_weather_forecast.dashboard.context import (
    DashboardContext,
    collect_context,
)
from grounded_weather_forecast.dashboard.derive import Derived, derive
from grounded_weather_forecast.dashboard.zones import ALL_ZONES
from grounded_weather_forecast.dashboard.zones import (
    evaluation,
    liveness,
    readiness,
    serving,
)
from grounded_weather_forecast.dataset.matrix import matrix_path
from grounded_weather_forecast.serve.history import HISTORY_SCHEMA

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def cold_context(tmp_path):
    return collect_context(write_config(tmp_path), now=NOW)


def test_all_zones_render_not_yet_states_on_cold_context(tmp_path):
    ctx = cold_context(tmp_path)
    derived = derive(ctx)
    zones = [build(ctx, derived) for build in ALL_ZONES]
    assert [zone.zone_id for zone in zones] == list("ABCDEFG")
    for zone in zones:
        assert zone.panels
        for panel in zone.panels:
            assert panel.copy.what
            if panel.chart is None and panel.table is None:
                assert panel.raw_html is not None or panel.empty_reason


def test_zone_a_marks_aged_out_providers_grey(tmp_path):
    config = write_config(tmp_path)
    config.dataset.dir.mkdir(parents=True, exist_ok=True)
    matrix = synthetic_hourly_matrix(days=2)
    age_columns = [c for c in matrix.columns if c.startswith("age__")]
    assert age_columns
    matrix = matrix.with_columns(pl.lit(20.0).alias(age_columns[0]))
    matrix.write_parquet(matrix_path(config.dataset.dir, "hourly", "live"))
    ctx = collect_context(config, now=NOW)
    zone = liveness.build(ctx, Derived())
    ages_panel = next(panel for panel in zone.panels if panel.panel_id == "a3")
    assert ages_panel.chart is not None
    colors = ages_panel.chart.config["data"]["datasets"][0]["backgroundColor"]
    assert "muted" in colors
    assert ages_panel.status == "amber"
    stats = {stat.label: stat.value for stat in ages_panel.stats}
    assert stats["providers fresh"] == "1/2"


def test_baseline_panel_only_flags_climatology_at_shortest_lead():
    board = pl.DataFrame(
        {
            "product": ["hourly"] * 4,
            "variable": ["temp_c"] * 4,
            "method_id": [
                "climatology",
                "best_provider",
                "climatology",
                "best_provider",
            ],
            "lead_bucket": ["0-1h", "0-1h", "240h+", "240h+"],
            "mae": [2.0, 1.0, 1.0, 3.0],
        }
    )

    panel = evaluation._baseline_panel("scores_hourly", board)

    assert panel is not None
    assert panel.status == "ok"


def test_zone_c_states_fold_arithmetic(tmp_path):
    config = write_config(tmp_path)
    config.dataset.dir.mkdir(parents=True, exist_ok=True)
    synthetic_hourly_matrix(days=5).write_parquet(
        matrix_path(config.dataset.dir, "hourly", "live")
    )
    ctx = collect_context(config, now=NOW)
    zone = readiness.build(ctx, Derived())
    fold_panel = next(panel for panel in zone.panels if panel.panel_id == "c1")
    needed = {stat.label: stat.value for stat in fold_panel.stats}
    assert needed["needed for first fold"] == "97 d"
    assert fold_panel.intro is not None
    assert "zero live folds is correct behaviour" in fold_panel.intro


def test_zone_f_selection_reason_shares(tmp_path):
    config = write_config(tmp_path)
    history = pl.DataFrame(
        {
            "issued_at": [NOW - timedelta(days=1), NOW],
            "product": ["hourly", "hourly"],
            "variable": ["temp_c", "temp_c"],
            "valid_time": [NOW, NOW + timedelta(hours=1)],
            "valid_date": [None, None],
            "lead_hours": [1.0, 1.0],
            "method_id": ["equal_weight", "equal_weight"],
            "y_pred": [10.0, 11.0],
            "dataset_fingerprint": ["f", "f"],
            "release_id": [None, None],
            "selection_reason": [
                "no backtest evidence for this slice",
                "no backtest evidence for this slice",
            ],
            "quantiles_json": [None, None],
        },
        schema=HISTORY_SCHEMA,
    )
    ctx = DashboardContext(config=config, now=NOW, history=history)
    zone = serving.build(ctx, Derived())
    reasons_panel = next(panel for panel in zone.panels if panel.panel_id == "f2")
    shares = {stat.label: stat.value for stat in reasons_panel.stats}
    assert shares["degraded share"] == "100%"
    assert reasons_panel.status == "amber"
    assert reasons_panel.chart is not None
