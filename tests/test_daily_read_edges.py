"""Malformed source and freshness boundaries for the daily read contract."""

import json
import sqlite3
from datetime import timedelta

import polars as pl
import pytest
from test_daily_read import NOW, START, provider_db, read, setup, write_json

from grounded_weather_forecast import daily_read as reader


@pytest.mark.parametrize("value", [None, [], {"schema_version": 6, "hourly": []}])
def test_malformed_forecast_keeps_health_result(tmp_path, value):
    config, forecast, _ = setup(tmp_path)
    write_json(forecast, value)
    result = read(config, forecast)
    assert result["forecast"]["status"] == "unknown"
    assert "providers" in result["operational_health"]


@pytest.mark.parametrize("value", [float("inf"), "7", True, 10**400, -1, 1.1])
def test_invalid_probability_is_null(tmp_path, value):
    config, forecast, raw = setup(tmp_path)
    raw["hourly"][0]["values"]["pop"] = value
    write_json(forecast, raw)
    row = read(config, forecast)["forecast"]["hourly"][0]
    assert row["values"]["pop"] is None
    assert row["invalid_variables"] == ["pop"]


def test_replacement_during_byte_read_is_detected(tmp_path, monkeypatch):
    config, forecast, _ = setup(tmp_path)
    original = type(forecast).read_bytes

    def replaced(path):
        data = original(path)
        if path == forecast:
            replacement = tmp_path / "replacement.json"
            replacement.write_bytes(data)
            replacement.replace(path)
        return data

    monkeypatch.setattr(type(forecast), "read_bytes", replaced)
    result = read(config, forecast)
    assert result["limitations"] == ["concurrent_replacement_after_retry"]


def test_invalid_source_columns_and_missing_station_are_precise(tmp_path):
    config, forecast, _ = setup(tmp_path)
    pl.DataFrame({"old_column": [1]}).write_parquet(config.dataset.dir / "runs.parquet")
    result = read(config, forecast)
    assert result["operational_health"]["runs"]["limitations"] == [
        "missing_required_columns"
    ]
    assert result["operational_health"]["station"]["limitations"] == [
        "live_station_and_published_snapshot_unavailable"
    ]


def test_timestamp_freshness_unknown_stale_and_dated_qc(tmp_path):
    config, forecast, _ = setup(tmp_path)
    assert reader.freshness(NOW + timedelta(seconds=1), NOW, 120)["status"] == "unknown"
    assert reader.freshness("bad", NOW, 120)["status"] == "unknown"
    assert reader.freshness(NOW - timedelta(minutes=120), NOW, 120)["status"] == "fresh"
    assert (
        reader.freshness(NOW - timedelta(minutes=120, seconds=1), NOW, 120)["status"]
        == "stale"
    )
    raw = {
        "schema_version": 3,
        "generated_at": START,
        "drift_alert": False,
        "correlation_alert": False,
        "shield_alert": False,
    }
    write_json(config.artifacts_dir / "truth_qc.json", raw)
    result = read(config, forecast)["prediction_quality"]["truth_qc"]
    assert result["status"] == "ok"
    assert result["current_alerts"]["drift_alert"] is False
    write_json(config.artifacts_dir / "truth_qc.json", raw | {"drift_alert": "yes"})
    assert read(config, forecast)["prediction_quality"]["truth_qc"]["limitations"] == [
        "invalid_truth_qc_alerts"
    ]


def test_station_default_tracks_active_home_and_qc_uses_shared_policy(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(reader.Path, "home", classmethod(lambda _cls: tmp_path))
    assert (
        reader.published_station_path()
        == tmp_path
        / "Library/Application Support/grounded-weather-forecast/staged/current.json"
    )
    config, forecast, _ = setup(tmp_path)
    write_json(
        config.artifacts_dir / "truth_qc.json",
        {
            "schema_version": 3,
            "generated_at": START,
            "drift_alert": False,
            "correlation_alert": False,
            "shield_alert": False,
        },
    )
    monkeypatch.setitem(reader.FRESHNESS_THRESHOLDS, "truth_qc_age_minutes", 5.0)
    result = read(config, forecast)["prediction_quality"]["truth_qc"]
    assert result["freshness"]["max_age_minutes"] == 5.0
    assert result["freshness"]["status"] == "stale"
    assert result["current_alerts"] is None


def test_empty_provider_and_old_runs_do_not_invent_health(tmp_path):
    config, forecast, _ = setup(tmp_path)
    provider_db(config.forecasts.db_path)
    result = reader.read_providers(config.forecasts.db_path, NOW + timedelta(days=5))
    assert result["status"] == "unknown"
    assert result["limitations"] == ["no_provider_attempts_in_24h"]
    pl.DataFrame(
        {
            "command": ["predict"],
            "started_at": [START - timedelta(days=2)],
            "ended_at": [START - timedelta(days=2)],
            "exit_code": [1],
        }
    ).write_parquet(config.dataset.dir / "runs.parquet")
    assert (
        read(config, forecast)["operational_health"]["runs"]["latest_by_command"] == []
    )


def test_provider_timestamps_order_by_instant_and_null_run_is_unknown(tmp_path):
    config, forecast, _ = setup(tmp_path)
    provider_db(config.forecasts.db_path)
    connection = sqlite3.connect(config.forecasts.db_path)
    try:
        connection.execute(
            "INSERT INTO forecast_runs VALUES (4, '2026-10-22T07:10:00-07:00')"
        )
        connection.execute(
            "INSERT INTO provider_results VALUES (5,4,'provider','success',NULL,NULL,NULL)"
        )
        connection.commit()
    finally:
        connection.close()
    row = reader.read_providers(config.forecasts.db_path, NOW)["providers"][0]
    assert row["latest_status"] == "success"
    assert row["completed_at"] == "2026-10-22T14:10:00+00:00"
    pl.DataFrame(
        {
            "command": ["predict"],
            "started_at": [START],
            "ended_at": [NOW],
            "exit_code": [None],
        }
    ).write_parquet(config.dataset.dir / "runs.parquet")
    assert (
        read(config, forecast)["operational_health"]["runs"]["latest_by_command"][0][
            "status"
        ]
        == "unknown"
    )


def test_changed_source_field_types_cannot_emit_nonfinite_json(tmp_path):
    config, forecast, raw = setup(tmp_path)
    raw["hourly"][0]["lead_bucket"] = float("nan")
    write_json(forecast, raw)
    pl.DataFrame(
        {
            "command": ["predict"],
            "started_at": [START],
            "ended_at": [NOW],
            "exit_code": [float("nan")],
        }
    ).write_parquet(config.dataset.dir / "runs.parquet")
    path = config.artifacts_dir / "history/pipeline.parquet"
    path.parent.mkdir()
    pl.DataFrame({"recorded_at": [START], "alarms": [float("nan")]}).write_parquet(path)
    result = read(config, forecast)
    assert result["forecast"]["limitations"] == ["invalid_lead_bucket"]
    assert result["operational_health"]["runs"]["limitations"] == [
        "invalid_run_exit_code"
    ]
    assert result["operational_health"]["pipeline"]["limitations"] == [
        "invalid_pipeline_alarms"
    ]
    json.dumps(result, allow_nan=False)


def test_execution_after_window_end_keeps_the_scheduled_window(tmp_path):
    config, forecast, _ = setup(tmp_path)
    result = reader.read_daily_report(
        config,
        forecast_path=forecast,
        start=START,
        as_of=START + timedelta(hours=14),
        published_station=tmp_path / "missing",
    )
    assert result["window"]["local_start"] == "2026-10-22T07:00:00-07:00"
    assert result["window"]["local_end"] == "2026-10-22T19:00:00-07:00"
    assert result["forecast"]["coverage"]["unique_hours"] == 12
    assert result["forecast"]["freshness"]["status"] == "stale"
