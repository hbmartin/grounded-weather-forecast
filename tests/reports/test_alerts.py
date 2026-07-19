import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
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
    now = overrides.pop("now", NOW)
    return AlertInputs(config=config, now=now, **overrides)


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


def test_provider_freshness_classifies_invalid_and_boundary_ages(tmp_path):
    sources = ["fresh", "at_cap", "old", "nan", "infinite", "boolean", "missing"]
    manifest = {**MANIFEST, "sources": sources}
    matrix = pl.DataFrame(
        {
            "issue_time": [NOW],
            "age__fresh": [11.999],
            "age__at_cap": [12.0],
            "age__old": [13.0],
            "age__nan": [float("nan")],
            "age__infinite": [float("inf")],
            "age__boolean": [True],
            "age__missing": [None],
        },
        schema_overrides={"issue_time": pl.Datetime("us", "UTC")},
    )

    alerts = evaluate_alerts(
        make_inputs(tmp_path, manifest=manifest, hourly_matrix=matrix)
    )

    (dropped,) = by_panel(alerts, "provider-dropped")
    (aged,) = by_panel(alerts, "provider-aged-out")
    for source in ("nan", "infinite", "boolean", "missing"):
        assert source in dropped.message
    for source in ("at_cap", "old"):
        assert source in aged.message
    assert "fresh" not in dropped.message.rsplit(": ", 1)[-1].split(", ")
    assert "fresh" not in aged.message.rsplit(": ", 1)[-1].split(", ")


def hourly_coverage(**columns) -> pl.DataFrame:
    """Hourly truth shaped like the real thing: coverage plus its timestamp.

    `build_truth` always emits `valid_hour`, and the trailing-week alert
    windows on it, so a fixture without it is not a truth frame.
    """
    length = len(next(iter(columns.values())))
    return pl.DataFrame(
        {
            "valid_hour": [
                NOW - timedelta(hours=offset) for offset in reversed(range(length))
            ],
            **columns,
        },
        schema_overrides={"valid_hour": pl.Datetime("us", "UTC")},
    )


def test_truth_thinning(tmp_path):
    thin = hourly_coverage(temp_c_cov=[0.5] * 48, pressure_sea_hpa_cov=[0.9] * 48)
    alerts = evaluate_alerts(make_inputs(tmp_path, hourly_truth=thin))
    (alert,) = by_panel(alerts, "truth-thinning")
    assert alert.severity == "amber"
    assert "temp_c=0.50" in alert.message
    assert "pressure" not in alert.message
    assert "min_hour_coverage" in alert.threshold

    healthy = thin.with_columns(pl.lit(0.95).alias("temp_c_cov"))
    alerts = evaluate_alerts(make_inputs(tmp_path, hourly_truth=healthy))
    assert not by_panel(alerts, "truth-thinning")


def test_truth_thinning_includes_daily_temperature_and_rain(tmp_path):
    config = write_config(tmp_path, min_hour_coverage=0.8, min_day_coverage=0.75)
    hourly = hourly_coverage(temp_c_cov=[0.95] * 24)
    daily = pl.DataFrame(
        {
            "date_local": [
                (NOW - timedelta(days=offset)).date() for offset in range(7)
            ],
            "coverage_frac": [0.6] * 7,
            "rain_coverage": [0.5] * 7,
        }
    )

    alerts = by_panel(
        evaluate_alerts(
            make_inputs(
                tmp_path,
                config=config,
                hourly_truth=hourly,
                daily_truth=daily,
            )
        ),
        "truth-thinning",
    )

    assert len(alerts) == 1
    assert "daily.temperature=0.60" in alerts[0].message
    assert "daily.rain=0.50" in alerts[0].message
    assert "min_day_coverage = 0.75" in alerts[0].threshold


def test_truth_thinning_accepts_older_daily_schema(tmp_path):
    config = write_config(tmp_path, min_hour_coverage=0.8, min_day_coverage=0.75)
    hourly = hourly_coverage(temp_c_cov=[0.95] * 24)
    old_daily = pl.DataFrame(
        {
            "date_local": [
                (NOW - timedelta(days=offset)).date() for offset in range(7)
            ],
            "coverage_frac": [0.6] * 7,
        }
    )

    alerts = by_panel(
        evaluate_alerts(
            make_inputs(
                tmp_path,
                config=config,
                hourly_truth=hourly,
                daily_truth=old_daily,
            )
        ),
        "truth-thinning",
    )

    assert len(alerts) == 1
    assert "daily.temperature=0.60" in alerts[0].message
    assert "daily.rain" not in alerts[0].message


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


@pytest.mark.parametrize("age_days", [2, 3])
def test_backend_swap_detects_leader_flip_inside_window(tmp_path, age_days):
    moments = [NOW - timedelta(days=age_days), NOW - timedelta(days=1), NOW]
    history = pl.DataFrame(
        {
            "captured_at": moments,
            "issue_time": moments,
            "method_id": ["boa"] * 3,
            "product": ["hourly"] * 3,
            "variable": ["temp_c"] * 3,
            "dataset_fingerprint": ["f"] * 3,
            "state_json": [
                _weight_state(True),
                _weight_state(False),
                _weight_state(False),
            ],
        }
    )
    alerts = by_panel(
        evaluate_alerts(make_inputs(tmp_path, observability_history=history)),
        "backend-swap",
    )
    assert len(alerts) == 1
    assert "alpha -> beta" in alerts[0].message


def test_backend_swap_ignores_pre_window_state(tmp_path):
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

    assert not any(alert.evaluable for alert in alerts)


def test_backend_swap_reports_latest_flip_back(tmp_path):
    moments = [
        NOW - timedelta(days=2),
        NOW - timedelta(hours=36),
        NOW - timedelta(hours=18),
        NOW,
    ]
    history = pl.DataFrame(
        {
            "captured_at": moments,
            "issue_time": moments,
            "method_id": ["boa"] * 4,
            "product": ["hourly"] * 4,
            "variable": ["temp_c"] * 4,
            "dataset_fingerprint": ["f"] * 4,
            "state_json": [
                _weight_state(True),
                _weight_state(False),
                _weight_state(True),
                _weight_state(True),
            ],
        }
    )

    alerts = by_panel(
        evaluate_alerts(make_inputs(tmp_path, observability_history=history)),
        "backend-swap",
    )

    assert len(alerts) == 1
    assert "beta -> alpha" in alerts[0].message


def test_backend_swap_stable_window_has_no_alert(tmp_path):
    history = pl.DataFrame(
        {
            "captured_at": [NOW - timedelta(days=2), NOW],
            "issue_time": [NOW - timedelta(days=2), NOW],
            "method_id": ["boa", "boa"],
            "product": ["hourly", "hourly"],
            "variable": ["temp_c", "temp_c"],
            "dataset_fingerprint": ["f", "f"],
            "state_json": [_weight_state(True), _weight_state(True)],
        }
    )

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


class TestDegenerateEvidenceIsNeverGreen:
    """Populated-but-degenerate artifacts must never render as healthy.

    Absent inputs were already covered; the gap was evidence that exists but
    carries no usable numbers. ``isinstance(x, (int, float))`` admits NaN and
    every NaN comparison is False, so these all used to return no alert.
    """

    def _hourly_truth(self, values):
        return pl.DataFrame(
            {
                "valid_hour": [NOW - timedelta(hours=index) for index in range(3)],
                "temp_c_cov": values,
            },
            schema_overrides={"valid_hour": pl.Datetime("us", "UTC")},
        )

    @pytest.mark.parametrize(
        ("label", "values"),
        [
            ("all-null", [None, None, None]),
            ("all-nan", [float("nan")] * 3),
        ],
    )
    def test_unusable_truth_coverage_fires_red(self, tmp_path, label, values):
        alerts = evaluate_alerts(
            make_inputs(tmp_path, hourly_truth=self._hourly_truth(values))
        )
        thinning = by_panel(alerts, "truth-thinning")
        assert [alert.severity for alert in thinning] == ["red"], label
        assert "no usable truth-coverage samples" in thinning[0].message

    def test_real_coverage_below_floor_still_fires_amber(self, tmp_path):
        alerts = evaluate_alerts(
            make_inputs(tmp_path, hourly_truth=self._hourly_truth([0.1, 0.1, 0.1]))
        )
        thinning = by_panel(alerts, "truth-thinning")
        assert [alert.severity for alert in thinning] == ["amber"]

    def test_healthy_coverage_is_silent(self, tmp_path):
        alerts = evaluate_alerts(
            make_inputs(tmp_path, hourly_truth=self._hourly_truth([1.0, 1.0, 1.0]))
        )
        assert by_panel(alerts, "truth-thinning") == []

    def test_nan_provider_age_is_reported_not_swallowed(self, tmp_path):
        matrix = pl.DataFrame(
            {
                "issue_time": [NOW],
                "age__alpha": [float("nan")],
                "age__beta": [1.0],
            },
            schema_overrides={"issue_time": pl.Datetime("us", "UTC")},
        )
        alerts = evaluate_alerts(
            make_inputs(tmp_path, manifest=MANIFEST, hourly_matrix=matrix)
        )
        dropped = by_panel(alerts, "provider-dropped")
        assert dropped, "a NaN age must not read as a healthy provider"
        assert "alpha" in dropped[0].message

    def test_non_finite_archive_location_fires_red(self, tmp_path):
        alerts = evaluate_alerts(
            make_inputs(
                tmp_path,
                manifest=MANIFEST,
                archive_location=(float("nan"), float("nan")),
            )
        )
        empties = by_panel(alerts, "silent-empty")
        assert any("non-finite location" in alert.message for alert in empties)
        assert all(alert.severity == "red" for alert in empties)


def test_truth_thinning_window_is_a_week_not_a_row_count(tmp_path):
    """Regression: `tail(24 * 7)` selected 168 ROWS, not seven days.

    On a gappy archive those 168 rows spanned 49 days of healthy history and
    reported a comfortable mean, while the actual trailing week sat far below
    the floor and raised nothing.
    """
    healthy_old = pl.DataFrame(
        {
            "valid_hour": [NOW - timedelta(days=8 + index) for index in range(160)],
            "temp_c_cov": [0.98] * 160,
        },
        schema_overrides={"valid_hour": pl.Datetime("us", "UTC")},
    )
    thin_recent = pl.DataFrame(
        {
            "valid_hour": [NOW - timedelta(hours=index) for index in range(8)],
            "temp_c_cov": [0.10] * 8,
        },
        schema_overrides={"valid_hour": pl.Datetime("us", "UTC")},
    )
    gappy = pl.concat([healthy_old, thin_recent])

    alerts = by_panel(
        evaluate_alerts(make_inputs(tmp_path, hourly_truth=gappy)), "truth-thinning"
    )

    assert len(alerts) == 1
    assert alerts[0].severity == "amber"
    assert "temp_c=0.10" in alerts[0].message


def test_future_truth_cannot_hide_current_thinning(tmp_path):
    recent = pl.DataFrame(
        {
            "valid_hour": [NOW - timedelta(hours=index) for index in range(10)],
            "temp_c_cov": [0.10] * 10,
        },
        schema_overrides={"valid_hour": pl.Datetime("us", "UTC")},
    )
    future = pl.DataFrame(
        {
            "valid_hour": [NOW + timedelta(hours=index + 1) for index in range(100)],
            "temp_c_cov": [1.0] * 100,
        },
        schema_overrides={"valid_hour": pl.Datetime("us", "UTC")},
    )

    alerts = by_panel(
        evaluate_alerts(
            make_inputs(tmp_path, hourly_truth=pl.concat([recent, future]))
        ),
        "truth-thinning",
    )

    assert len(alerts) == 1
    assert "temp_c=0.10" in alerts[0].message


def test_daily_truth_window_uses_station_local_date(tmp_path):
    now = datetime(2026, 7, 18, 1, 0, tzinfo=UTC)
    config = write_config(tmp_path, min_day_coverage=0.75)
    daily = pl.DataFrame(
        {
            "date_local": [datetime(2026, 7, 10).date()],
            "coverage_frac": [0.10],
        }
    )

    alerts = by_panel(
        evaluate_alerts(
            make_inputs(
                tmp_path,
                config=config,
                now=now,
                hourly_truth=pl.DataFrame(),
                daily_truth=daily,
            )
        ),
        "truth-thinning",
    )

    assert len(alerts) == 1
    assert alerts[0].severity == "amber"
    assert "daily.temperature=0.10" in alerts[0].message


def test_truth_thinning_judges_daily_even_without_hourly(tmp_path):
    """Regression: a missing hourly frame discarded valid daily coverage."""
    config = write_config(tmp_path, min_hour_coverage=0.8, min_day_coverage=0.75)
    daily = pl.DataFrame(
        {
            "date_local": [
                (NOW - timedelta(days=offset)).date() for offset in range(5)
            ],
            "coverage_frac": [0.10] * 5,
        }
    )

    alerts = by_panel(
        evaluate_alerts(
            make_inputs(
                tmp_path,
                config=config,
                hourly_truth=pl.DataFrame(),
                daily_truth=daily,
            )
        ),
        "truth-thinning",
    )

    assert len(alerts) == 1
    assert "daily.temperature=0.10" in alerts[0].message


def test_truth_thinning_is_not_evaluable_without_recent_truth(tmp_path):
    """An archive that stopped a month ago has nothing to say about this week."""
    stale = pl.DataFrame(
        {
            "valid_hour": [NOW - timedelta(days=30 + index) for index in range(48)],
            "temp_c_cov": [0.10] * 48,
        },
        schema_overrides={"valid_hour": pl.Datetime("us", "UTC")},
    )

    (alert,) = by_panel(
        evaluate_alerts(make_inputs(tmp_path, hourly_truth=stale)), "truth-thinning"
    )

    assert alert.severity == "info"
    assert "trailing 7 days" in alert.message


def test_unreadable_artifacts_are_named_not_shown_as_absent(tmp_path):
    """Every loader falls back to "absent" on failure.

    That is right for a young archive and wrong for a corrupt or
    permission-denied file: the dashboard rendered "not yet" over a broken
    deployment. The alert names them instead.
    """
    alerts = by_panel(
        evaluate_alerts(
            make_inputs(tmp_path, unreadable_artifacts=("truth_hourly.parquet",))
        ),
        "unreadable-artifacts",
    )

    assert len(alerts) == 1
    assert alerts[0].severity == "red"
    assert "truth_hourly.parquet" in alerts[0].message


def test_no_unreadable_alert_when_everything_loads(tmp_path):
    assert not by_panel(evaluate_alerts(make_inputs(tmp_path)), "unreadable-artifacts")


def _dense_trajectory(leaders):
    """A trajectory at the documented 10-minute `predict` cadence."""
    count = len(leaders)
    moments = [
        NOW - timedelta(minutes=10 * offset) for offset in reversed(range(count))
    ]
    return pl.DataFrame(
        {
            "captured_at": moments,
            "issue_time": moments,
            "method_id": ["boa"] * count,
            "product": ["hourly"] * count,
            "variable": ["temp_c"] * count,
            "dataset_fingerprint": ["f"] * count,
            "state_json": [_weight_state(leader) for leader in leaders],
        }
    )


class TestBackendSwapIgnoresNoise:
    """`_argmax_source` crosses whenever two experts are near-tied.

    A 3-day window holds ~430 samples at the documented cadence, so counting
    every crossing would raise an amber alert off arithmetic noise. A leader
    has to hold `_SWAP_MIN_HOLD_DAYS` before the flip is called a swap.
    """

    def test_a_single_sample_crossing_is_not_a_swap(self, tmp_path):
        leaders = [True] * 432
        leaders[216] = False  # one transient crossing, 10 minutes wide

        alerts = by_panel(
            evaluate_alerts(
                make_inputs(tmp_path, observability_history=_dense_trajectory(leaders))
            ),
            "backend-swap",
        )

        assert alerts == [], "a 10-minute crossing must not read as a regime change"

    def test_a_newest_single_sample_regime_is_still_pending(self, tmp_path):
        history = pl.DataFrame(
            {
                "captured_at": [NOW - timedelta(days=2), NOW],
                "issue_time": [NOW - timedelta(days=2), NOW],
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

        assert alerts == []

    def test_sustained_flapping_is_reported_with_its_count(self, tmp_path):
        # Three day-long regimes: alpha, beta, alpha. Both flips held.
        leaders = [True] * 144 + [False] * 144 + [True] * 144

        alerts = by_panel(
            evaluate_alerts(
                make_inputs(tmp_path, observability_history=_dense_trajectory(leaders))
            ),
            "backend-swap",
        )

        assert len(alerts) == 1
        assert "beta -> alpha" in alerts[0].message, "the latest flip is the headline"
        assert "2 times" in alerts[0].message, "flapping must be distinguishable"

    def test_a_sustained_flip_still_alerts(self, tmp_path):
        leaders = [True] * 216 + [False] * 216

        alerts = by_panel(
            evaluate_alerts(
                make_inputs(tmp_path, observability_history=_dense_trajectory(leaders))
            ),
            "backend-swap",
        )

        assert len(alerts) == 1
        assert "alpha -> beta" in alerts[0].message
        assert "times" not in alerts[0].message, "one flip is not flapping"
