from datetime import UTC, datetime, timedelta

import polars as pl
from conftest import synthetic_hourly_matrix, write_config

from grounded_weather_forecast.dashboard.context import (
    DashboardContext,
    collect_context,
)
from grounded_weather_forecast.dashboard.derive import Derived, derive
from grounded_weather_forecast.dashboard.zones import (
    ALL_ZONES,
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


def test_zone_a_freshness_uses_finite_strict_boundary(tmp_path):
    config = write_config(tmp_path)
    cap = config.forecasts.max_forecast_age_hours
    sources = ("fresh", "at_cap", "nan", "infinite", "boolean", "missing")
    matrix = pl.DataFrame(
        {
            "issue_time": [NOW],
            "age__fresh": [cap - 0.001],
            "age__at_cap": [cap],
            "age__nan": [float("nan")],
            "age__infinite": [float("inf")],
            "age__boolean": [True],
            "age__missing": [None],
        },
        schema_overrides={"issue_time": pl.Datetime("us", "UTC")},
    )
    ctx = DashboardContext(
        config=config,
        now=NOW,
        manifest={"sources": list(sources)},
        hourly_matrix=matrix,
    )

    panel = liveness._provider_ages(ctx, sources)

    assert panel.chart is not None
    stats = {stat.label: stat.value for stat in panel.stats}
    assert stats["providers fresh"] == "1/6"
    colors = panel.chart.config["data"]["datasets"][0]["backgroundColor"]
    assert colors == ["series-1", "muted", "muted", "muted", "muted", "muted"]


def _baseline_board(
    shortest_mae: tuple[float, float], longest_mae: tuple[float, float]
):
    """A two-bucket board: (climatology, best_provider) MAE at each lead."""
    return pl.DataFrame(
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
            "mae": [*shortest_mae, *longest_mae],
        }
    )


def test_baseline_panel_only_flags_climatology_at_shortest_lead():
    """Climatology winning at 240h+ is legitimate; at 0-1h it is leakage.

    The MAE values matter: with climatology at 1.0 in the long bucket the
    old cross-bucket `min()` comparison also returned "ok", so this test
    passed against the very bug it is named for. 0.5 discriminates.
    """
    board = _baseline_board(shortest_mae=(2.0, 1.0), longest_mae=(0.5, 3.0))

    panel = evaluation._baseline_panel("scores_hourly", board)

    assert panel is not None
    assert panel.status == "ok", (
        "climatology beating best_provider at a long lead is not leakage"
    )


def test_baseline_panel_flags_climatology_winning_at_the_shortest_lead():
    """The condition the panel exists to catch must actually fire."""
    board = _baseline_board(shortest_mae=(1.0, 2.0), longest_mae=(3.0, 1.0))

    panel = evaluation._baseline_panel("scores_hourly", board)

    assert panel is not None
    assert panel.status == "amber"


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
    assert shares["degraded share (last 1d)"] == "100%"
    assert shares["degraded share (lifetime)"] == "100%"
    assert reasons_panel.status == "red"
    assert reasons_panel.chart is not None


def _degraded_history(healthy: int, degraded: int) -> pl.DataFrame:
    """Served history with `degraded` recent degraded rows after `healthy` old ones."""
    rows = healthy + degraded
    return pl.DataFrame(
        {
            "issued_at": [NOW - timedelta(days=30)] * healthy + [NOW] * degraded,
            "product": ["hourly"] * rows,
            "variable": ["temp_c"] * rows,
            "valid_time": [NOW + timedelta(hours=1)] * rows,
            "valid_date": [None] * rows,
            "lead_hours": [1.0] * rows,
            "method_id": ["equal_weight"] * rows,
            "y_pred": [10.0] * rows,
            "dataset_fingerprint": ["f"] * rows,
            "release_id": [None] * rows,
            "selection_reason": ["winner: emos"] * healthy
            + ["degraded: no evidence"] * degraded,
            "quantiles_json": [None] * rows,
        },
        schema=HISTORY_SCHEMA,
    )


def test_zone_f_degraded_share_is_judged_on_the_trailing_window(tmp_path):
    """A lifetime share is diluted by history; today's degradation must show."""
    config = write_config(tmp_path)
    # 100% degraded right now, but only 7% of all rows ever served.
    ctx = DashboardContext(
        config=config, now=NOW, history=_degraded_history(healthy=5000, degraded=400)
    )
    panel = next(p for p in serving.build(ctx, Derived()).panels if p.panel_id == "f2")
    shares = {stat.label: stat.value for stat in panel.stats}
    assert shares["degraded share (last 1d)"] == "100%"
    assert shares["degraded share (lifetime)"] == "7%"
    assert panel.status == "red"


def test_zone_f_near_total_degradation_is_not_green(tmp_path):
    """999/1000 degraded must not read 'ok' just because it is under 100%."""
    config = write_config(tmp_path)
    ctx = DashboardContext(
        config=config, now=NOW, history=_degraded_history(healthy=0, degraded=999)
    )
    panel = next(p for p in serving.build(ctx, Derived()).panels if p.panel_id == "f2")
    assert panel.status == "red"


def test_zone_f_unusable_live_scores_are_not_reported_as_a_young_archive(tmp_path):
    """A damaged artifact and an empty one must not render the same."""
    config = write_config(tmp_path)
    ctx = DashboardContext(config=config, now=NOW)
    young = next(p for p in serving.build(ctx, Derived()).panels if p.panel_id == "f1")
    damaged = next(
        p
        for p in serving.build(ctx, Derived(live_scores_unusable=True)).panels
        if p.panel_id == "f1"
    )
    assert young.status == "info"
    assert damaged.status == "red"
    assert "could not be read" in (damaged.empty_reason or "")


def test_zone_f_verification_labels_include_lead_bucket(tmp_path):
    config = write_config(tmp_path)
    live = pl.DataFrame(
        {
            "product": ["hourly", "hourly"],
            "variable": ["temp_c", "temp_c"],
            "lead_bucket": ["0-1h", "1-6h"],
            "method_id": ["boa", "boa"],
            "n": [10, 10],
            "live_mae": [1.0, 1.1],
            "backtest_mae": [0.9, 1.0],
            "mae_gap": [0.1, 0.1],
            "live_bias": [0.0, 0.1],
        }
    )
    panel = serving._verification_panel(
        DashboardContext(config=config, now=NOW),
        Derived(verification=live),
    )

    assert panel.chart is not None
    assert panel.table is not None
    labels = panel.chart.config["data"]["labels"]
    assert labels == [
        "hourly.temp_c.0-1h.boa",
        "hourly.temp_c.1-6h.boa",
    ]
    assert [row[0] for row in panel.table.rows] == labels


def test_zone_f_reasons_use_total_frequency_then_name(tmp_path):
    config = write_config(tmp_path)
    reasons = ["z-dominant"] * 5 + list("abcdefghi")
    count = len(reasons)
    history = pl.DataFrame(
        {
            "issued_at": [NOW] * count,
            "product": ["hourly"] * count,
            "variable": ["temp_c"] * count,
            "valid_time": [NOW + timedelta(hours=index) for index in range(count)],
            "valid_date": [None] * count,
            "lead_hours": [1.0] * count,
            "method_id": ["equal_weight"] * count,
            "y_pred": [10.0] * count,
            "dataset_fingerprint": ["f"] * count,
            "release_id": [None] * count,
            "selection_reason": reasons,
            "quantiles_json": [None] * count,
        },
        schema=HISTORY_SCHEMA,
    )
    panel = serving._reasons_panel(
        DashboardContext(config=config, now=NOW, history=history)
    )

    assert panel.chart is not None
    labels = [dataset["label"] for dataset in panel.chart.config["data"]["datasets"]]
    assert labels == ["z-dominant", "a", "b", "c", "d", "e", "f", "g"]


def _qc_frame(rows):
    """A qc_summary-shaped frame: one row per channel."""
    return pl.DataFrame(
        rows,
        schema={
            "channel": pl.String,
            "samples": pl.Int64,
            "missing": pl.Int64,
            "out_of_bounds": pl.Int64,
            "spike": pl.Int64,
            "flatline": pl.Int64,
            "clean": pl.Int64,
        },
        orient="row",
    )


def _station_qc_panel(tmp_path, qc):
    from dataclasses import replace

    from grounded_weather_forecast.dashboard.zones import data_trust

    ctx = replace(cold_context(tmp_path), qc=qc)
    zone = data_trust.build(ctx, derive(ctx))
    return next(panel for panel in zone.panels if panel.panel_id == "b1")


def test_zone_b_ignores_uninstalled_sensors_in_the_flagged_share(tmp_path):
    """An absent sensor is missing data, not a QC flag.

    `clean` counts QC_OK *and* non-null, so measuring the flagged share
    against total samples alarmed on a station that raised no flag at all.
    """
    clean_channel = ("temp_c", 360, 0, 0, 0, 0, 360)
    absent = ("pressure_station", 360, 360, 0, 0, 0, 0)
    panel = _station_qc_panel(tmp_path, _qc_frame([clean_channel, absent]))
    assert panel.status == "ok"
    assert dict((stat.label, stat.value) for stat in panel.stats)["clean"] == "100.0%"


def test_zone_b_still_flags_a_channel_that_is_wholly_out_of_bounds(tmp_path):
    """The behaviour the flagged share exists to catch must survive."""
    clean_channel = ("temp_c", 360, 0, 0, 0, 0, 360)
    bad = ("humidity_pct", 360, 0, 360, 0, 0, 0)
    panel = _station_qc_panel(tmp_path, _qc_frame([clean_channel, bad]))
    assert panel.status == "red"


def _history_with_minutely(hourly_degraded: int, minutely: int) -> pl.DataFrame:
    """Served history where minutely rows carry no selection_reason.

    `forecast_to_rows` writes `selection_reason=None` for minutely points --
    they are interpolated from the hourly path, not selected.
    """
    rows = hourly_degraded + minutely
    return pl.DataFrame(
        {
            "issued_at": [NOW] * rows,
            "product": ["hourly"] * hourly_degraded + ["minutely"] * minutely,
            "variable": ["temp_c"] * rows,
            "valid_time": [NOW + timedelta(hours=1)] * rows,
            "valid_date": [None] * rows,
            "lead_hours": [1.0] * rows,
            "method_id": ["equal_weight"] * rows,
            "y_pred": [10.0] * rows,
            "dataset_fingerprint": ["f"] * rows,
            "release_id": [None] * rows,
            "selection_reason": ["degraded: no evidence"] * hourly_degraded
            + [None] * minutely,
            "quantiles_json": [None] * rows,
        },
        schema=HISTORY_SCHEMA,
    )


def test_zone_f_degraded_share_ignores_rows_that_carry_no_selection(tmp_path):
    """Regression: minutely rows diluted a total serving outage to single digits.

    A realistic document is ~72% minutely rows. Counting them made a wholly
    degraded forecast read 27.6% -- amber instead of red -- and flipped an
    11% outage all the way to green.
    """
    config = write_config(tmp_path)
    ctx = DashboardContext(
        config=config,
        now=NOW,
        history=_history_with_minutely(hourly_degraded=8, minutely=140),
    )

    panel = next(p for p in serving.build(ctx, Derived()).panels if p.panel_id == "f2")

    shares = {stat.label: stat.value for stat in panel.stats}
    assert shares["degraded share (last 1d)"] == "100%"
    assert panel.status == "red"


def test_zone_f_trailing_window_excludes_future_rows(tmp_path):
    config = write_config(tmp_path)
    history = _degraded_history(healthy=1, degraded=0).with_columns(
        pl.lit(NOW + timedelta(days=2)).alias("issued_at"),
        pl.lit("degraded: clock skew").alias("selection_reason"),
    )
    ctx = DashboardContext(config=config, now=NOW, history=history)

    panel = next(p for p in serving.build(ctx, Derived()).panels if p.panel_id == "f2")

    shares = {stat.label: stat.value for stat in panel.stats}
    assert shares["degraded share (last 1d)"] == "—"
    assert panel.status == "ok"


def test_zone_b_worst_channel_drives_the_qc_verdict(tmp_path):
    """One dead sensor among many clean ones must not average away.

    24 clean channels plus one wholly out-of-bounds channel moves the pooled
    share to 3.8% -- under the 5% floor -- and rendered the panel green.
    """
    clean = [(f"c{index}", 360, 0, 0, 0, 0, 360) for index in range(24)]
    dead = ("humidity_pct", 360, 0, 360, 0, 0, 0)
    panel = _station_qc_panel(tmp_path, _qc_frame([*clean, dead]))
    assert panel.status == "red"


def test_zone_b_non_finite_coverage_is_not_healthy(tmp_path):
    """`isinstance` admits NaN and `min([nan]) < floor` is False."""
    from dataclasses import replace

    from grounded_weather_forecast.dashboard.zones import data_trust

    truth = pl.DataFrame(
        {
            "valid_hour": [NOW - timedelta(hours=index) for index in range(4)],
            "temp_c_cov": [float("nan")] * 4,
        },
        schema_overrides={"valid_hour": pl.Datetime("us", "UTC")},
    )
    ctx = replace(cold_context(tmp_path), truth_hourly=truth)

    zone = data_trust.build(ctx, derive(ctx))
    panel = next(p for p in zone.panels if p.panel_id == "b2")

    assert panel.status == "amber", "all-NaN coverage must not render ok"
