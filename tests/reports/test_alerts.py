import json
from datetime import UTC, datetime, timedelta

import polars as pl
from conftest import write_config

from grounded_weather_forecast.reports.alerts import AlertInputs, evaluate_alerts

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
MANIFEST = {
    "fingerprint": "feedface00000000",
    "sources": ["alpha", "beta"],
    "snapshots": 10,
    "files": {"truth_minute.parquet": {"rows": 100}},
}


def make_inputs(tmp_path, **overrides):
    config = overrides.pop("config", write_config(tmp_path))
    return AlertInputs(config=config, now=NOW, **overrides)


def by_panel(alerts, panel_id):
    return [alert for alert in alerts if alert.panel_id == panel_id]


def test_empty_inputs_degrade_to_not_evaluable(tmp_path):
    alerts = evaluate_alerts(make_inputs(tmp_path))
    reds = [alert for alert in alerts if alert.severity == "red"]
    assert [alert.panel_id for alert in reds] == ["silent-empty"]
    assert "build-dataset" in reds[0].message
    for alert in alerts:
        if alert.panel_id != "silent-empty":
            assert alert.severity == "info"
            assert alert.evaluable is False
            assert alert.message.startswith("not evaluable yet:")


def test_silent_empty_manifest_states_fire_red(tmp_path):
    manifest = {
        "fingerprint": "abc",
        "sources": [],
        "snapshots": 0,
        "files": {"truth_minute.parquet": {"rows": 0}},
    }
    alerts = by_panel(
        evaluate_alerts(make_inputs(tmp_path, manifest=manifest)), "silent-empty"
    )
    messages = " | ".join(alert.message for alert in alerts)
    assert len(alerts) == 3
    assert "zero sources" in messages
    assert "zero snapshots" in messages
    assert "truth_minute.parquet" in messages


def test_location_mismatch_fires_red(tmp_path):
    alerts = by_panel(
        evaluate_alerts(
            make_inputs(tmp_path, manifest=MANIFEST, archive_location=(34.0, -117.0))
        ),
        "silent-empty",
    )
    assert len(alerts) == 1
    assert alerts[0].severity == "red"
    assert "does not match" in alerts[0].message
    assert "LOCATION_TOLERANCE" in alerts[0].threshold


def test_anchor_lost_and_ingestion_stalled(tmp_path):
    stale = pl.DataFrame({"ts": [NOW - timedelta(minutes=45)]})
    alerts = evaluate_alerts(make_inputs(tmp_path, minute_truth=stale))
    (anchor,) = by_panel(alerts, "anchor-lost")
    assert anchor.severity == "amber"
    assert "OBS_STALENESS" in anchor.threshold

    dead = pl.DataFrame({"ts": [NOW - timedelta(hours=13)]})
    alerts = evaluate_alerts(make_inputs(tmp_path, minute_truth=dead))
    (stalled,) = by_panel(alerts, "ingestion-stalled")
    assert stalled.severity == "red"

    fresh = pl.DataFrame({"ts": [NOW - timedelta(minutes=5)]})
    alerts = evaluate_alerts(make_inputs(tmp_path, minute_truth=fresh))
    assert not by_panel(alerts, "anchor-lost")
    assert not by_panel(alerts, "ingestion-stalled")


def test_provider_dropped_and_aged_out(tmp_path):
    matrix = pl.DataFrame(
        {"issue_time": [NOW], "age__alpha": [1.0], "age__beta": [None]},
        schema={
            "issue_time": pl.Datetime("us", "UTC"),
            "age__alpha": pl.Float64,
            "age__beta": pl.Float64,
        },
    )
    alerts = evaluate_alerts(
        make_inputs(tmp_path, manifest=MANIFEST, hourly_matrix=matrix)
    )
    (dropped,) = by_panel(alerts, "provider-dropped")
    assert dropped.severity == "amber"
    assert "beta" in dropped.message

    aged_matrix = matrix.with_columns(pl.lit(20.0).alias("age__beta"))
    alerts = evaluate_alerts(
        make_inputs(tmp_path, manifest=MANIFEST, hourly_matrix=aged_matrix)
    )
    (aged,) = by_panel(alerts, "provider-aged-out")
    assert "beta" in aged.message


def test_truth_thinning(tmp_path):
    thin = pl.DataFrame({"temp_c_cov": [0.5] * 48, "pressure_sea_hpa_cov": [0.9] * 48})
    alerts = evaluate_alerts(make_inputs(tmp_path, hourly_truth=thin))
    (alert,) = by_panel(alerts, "truth-thinning")
    assert alert.severity == "amber"
    assert "temp_c=0.50" in alert.message
    assert "pressure" not in alert.message
    assert "min_hour_coverage" in alert.threshold

    healthy = thin.with_columns(pl.lit(0.95).alias("temp_c_cov"))
    alerts = evaluate_alerts(make_inputs(tmp_path, hourly_truth=healthy))
    assert not by_panel(alerts, "truth-thinning")


def test_stuck_sensor(tmp_path):
    qc = pl.DataFrame(
        {
            "channel": ["temp", "humidity"],
            "flatline": [5, 0],
            "active_flatline": [True, False],
        }
    )
    alerts = evaluate_alerts(make_inputs(tmp_path, qc=qc))
    (alert,) = by_panel(alerts, "stuck-sensor")
    assert "temp (5 flagged samples in history)" in alert.message
    assert "humidity" not in alert.message
    assert "flatline_minutes" in alert.threshold

    recovered = qc.with_columns(pl.lit(False).alias("active_flatline"))
    alerts = evaluate_alerts(make_inputs(tmp_path, qc=recovered))
    assert not by_panel(alerts, "stuck-sensor")


def test_drift_tiers_map_to_severity(tmp_path):
    drift = {
        "alarms": [
            {
                "source": "alpha",
                "variable": "temp_c",
                "lead_bucket": "24-48h",
                "tier": "residual",
                "detail": "excursion 30",
            },
            {
                "source": "beta",
                "variable": "temp_c",
                "lead_bucket": "24-48h",
                "tier": "consensus",
                "detail": "z 7",
            },
        ]
    }
    alerts = by_panel(
        evaluate_alerts(make_inputs(tmp_path, drift=drift)), "provider-drifting"
    )
    severities = {alert.message.split(" ")[0]: alert.severity for alert in alerts}
    assert severities == {"alpha": "red", "beta": "amber"}


def test_grounding_bias_uses_consumer_tolerance(tmp_path):
    board = pl.DataFrame(
        {
            "product": ["hourly", "hourly"],
            "variable": ["temp_c", "temp_c"],
            "lead_bucket": ["24-48h", "48-96h"],
            "method_id": ["grounded_equal_weight", "grounded_equal_weight"],
            "n": [50, 50],
            "mae": [1.2, 1.3],
            "bias": [2.5, 0.3],
        }
    )
    alerts = by_panel(
        evaluate_alerts(make_inputs(tmp_path, board=board)), "grounding-bias"
    )
    assert len(alerts) == 1
    assert "24-48h" in alerts[0].message
    assert "CONSUMER_TOLERANCES" in alerts[0].threshold


def test_baseline_implausible_when_climatology_beats_best_provider(tmp_path):
    board = pl.DataFrame(
        {
            "product": ["hourly", "hourly"],
            "variable": ["temp_c", "temp_c"],
            "lead_bucket": ["0-1h", "0-1h"],
            "method_id": ["climatology", "best_provider"],
            "n": [50, 50],
            "mae": [0.5, 1.0],
            "bias": [0.0, 0.0],
        }
    )
    alerts = by_panel(
        evaluate_alerts(make_inputs(tmp_path, board=board)), "baseline-implausible"
    )
    assert len(alerts) == 1
    assert "structural heuristic" in alerts[0].threshold


def test_baseline_implausible_ignores_long_horizon_climatology_win(tmp_path):
    board = pl.DataFrame(
        {
            "product": ["hourly"] * 4,
            "variable": ["temp_c"] * 4,
            "lead_bucket": ["0-1h", "0-1h", "48-96h", "48-96h"],
            "method_id": ["climatology", "best_provider"] * 2,
            "mae": [2.0, 1.0, 0.5, 1.0],
        }
    )

    alerts = evaluate_alerts(make_inputs(tmp_path, board=board))

    assert not by_panel(alerts, "baseline-implausible")


def _weight_state(leader_first):
    weights = [0.8, 0.2] if leader_first else [0.2, 0.8]
    return json.dumps(
        {
            "sources": ["alpha", "beta"],
            "buckets": {"24-48h": {"weights": weights}},
        }
    )


def test_backend_swap_detects_leader_flip(tmp_path):
    history = pl.DataFrame(
        {
            "captured_at": [NOW - timedelta(days=5), NOW],
            "issue_time": [NOW - timedelta(days=5), NOW],
            "method_id": ["boa", "boa"],
            "product": ["hourly", "hourly"],
            "variable": ["temp_c", "temp_c"],
            "dataset_fingerprint": ["f", "f"],
            "state_json": [_weight_state(True), _weight_state(False)],
        }
    )
    alerts = by_panel(
        evaluate_alerts(make_inputs(tmp_path, observability_history=history)),
        "backend-swap",
    )
    assert len(alerts) == 1
    assert "alpha -> beta" in alerts[0].message

    stable = history.with_columns(pl.lit(_weight_state(True)).alias("state_json"))
    alerts = by_panel(
        evaluate_alerts(make_inputs(tmp_path, observability_history=stable)),
        "backend-swap",
    )
    assert alerts == []


def test_serving_diverged_uses_promotion_knobs(tmp_path):
    live = pl.DataFrame(
        {
            "product": ["hourly", "hourly"],
            "variable": ["temp_c", "temp_c"],
            "method_id": ["boa", "gbm"],
            "n": [30, 5],
            "live_mae": [2.0, 9.0],
            "backtest_mae": [1.0, 1.0],
        }
    )
    alerts = by_panel(
        evaluate_alerts(make_inputs(tmp_path, live_vs_backtest=live)),
        "serving-diverged",
    )
    assert len(alerts) == 1  # the n=5 row is below min_live_n
    assert alerts[0].severity == "red"
    assert "live_gap_factor" in alerts[0].threshold


def test_serving_status_alerts(tmp_path):
    alerts = evaluate_alerts(
        make_inputs(tmp_path, latest_status=("degraded", "cold start: no scores"))
    )
    (degraded,) = by_panel(alerts, "serving-degraded")
    assert "cold start: no scores" in degraded.message

    alerts = evaluate_alerts(make_inputs(tmp_path, latest_status=("ready", None)))
    assert not by_panel(alerts, "serving-degraded")

    runs = pl.DataFrame(
        {
            "command": ["predict"],
            "started_at": [NOW],
            "exit_code": [None],
            "error": ["NoForecastDataError"],
        },
        schema={
            "command": pl.String,
            "started_at": pl.Datetime("us", "UTC"),
            "exit_code": pl.Int64,
            "error": pl.String,
        },
    )
    alerts = evaluate_alerts(make_inputs(tmp_path, runs=runs))
    (refused,) = by_panel(alerts, "serving-refused")
    assert refused.severity == "red"
    assert "NoForecastDataError" in refused.message


def test_artifacts_stale_on_fingerprint_drift(tmp_path):
    releases = (
        {
            "promoted_at": "2026-07-01T00:00:00+00:00",
            "dataset_fingerprint": "old_print",
            "config_fingerprint": "old_config",
        },
    )
    alerts = by_panel(
        evaluate_alerts(make_inputs(tmp_path, manifest=MANIFEST, releases=releases)),
        "artifacts-stale",
    )
    assert len(alerts) == 2
    assert all(alert.severity == "amber" for alert in alerts)


def test_archive_stalled(tmp_path):
    matrix = pl.DataFrame(
        {"issue_time": [NOW - timedelta(hours=13)]},
        schema={"issue_time": pl.Datetime("us", "UTC")},
    )
    alerts = by_panel(
        evaluate_alerts(make_inputs(tmp_path, hourly_matrix=matrix)),
        "archive-stalled",
    )
    assert [alert.severity for alert in alerts] == ["amber"]

    dead = pl.DataFrame(
        {"issue_time": [NOW - timedelta(hours=40)]},
        schema={"issue_time": pl.Datetime("us", "UTC")},
    )
    alerts = by_panel(
        evaluate_alerts(make_inputs(tmp_path, hourly_matrix=dead)),
        "archive-stalled",
    )
    assert [alert.severity for alert in alerts] == ["red"]


def test_alerts_sorted_most_severe_first(tmp_path):
    dead = pl.DataFrame({"ts": [NOW - timedelta(hours=13)]})
    alerts = evaluate_alerts(
        make_inputs(
            tmp_path,
            minute_truth=dead,
            latest_status=("degraded", "no promoted release"),
        )
    )
    order = {"red": 0, "amber": 1, "info": 2}
    ranks = [order[alert.severity] for alert in alerts]
    assert ranks == sorted(ranks)
