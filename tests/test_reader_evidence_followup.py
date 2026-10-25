"""Nullable recency metadata and cached compatibility preserve evidence selection."""

from datetime import timedelta
from unittest.mock import patch

import polars as pl
from test_daily_read import NOW, START, quality, setup, write_json, write_quality
from test_daily_reader_regressions import modern_release, temperature

from grounded_weather_forecast import daily_read as reader


def test_missing_and_invalid_realized_slices_include_null_clock(tmp_path):
    config, forecast, raw = setup(tmp_path)
    write_quality(config, [], realized=True)
    assert (
        temperature(config, forecast, realized=True)["latest_scored_issue_at"] is None
    )
    raw["hourly"][0]["release_ids"]["temp_c"] = ""
    write_json(forecast, raw)
    report = reader.read_forecast(
        reader.Snapshot(),
        forecast,
        START,
        NOW,
        reader.ZoneInfo(config.station.timezone),
    )
    result = reader.read_quality(reader.Snapshot(), config, report, NOW, realized=True)
    assert result["slices"] and all(
        "latest_scored_issue_at" in row for row in result["slices"]
    )
    assert any(
        "slice_has_no_valid_forecast_release" in row["limitations"]
        for row in result["slices"]
    )


def test_incompatible_release_is_checked_once_across_tiers(tmp_path):
    config, forecast, _ = setup(tmp_path)
    modern_release(config)
    path, raw = modern_release(config, "candidate")
    raw["promoted_at"] = (START - timedelta(days=5)).isoformat()
    raw["selections"]["hourly.temp_c.1-6h"]["method_id"] = "different"
    write_json(path, raw)
    rows = [
        quality(
            config,
            realized=True,
            release_id="candidate",
            dataset_fingerprint=dataset,
            recorded_at=START - timedelta(days=days),
        )
        for dataset, days in [("a", 0), ("b", 1), ("c", 2)]
    ]
    write_quality(config, rows, realized=True)
    original = reader._compatible_release
    calls = []

    def checked(*args, **kwargs):
        calls.append((args[2], args[3]))
        return original(*args, **kwargs)

    with patch.object(reader, "_compatible_release", checked):
        result = temperature(config, forecast, realized=True)
    assert result["status"] == "unknown"
    assert calls.count(("candidate", "hourly.temp_c.1-6h")) == 1


def test_daily_run_health_uses_same_last_row_tie_rule(tmp_path):
    rows = [
        {"command": "predict", "started_at": START, "ended_at": NOW, "exit_code": code}
        for code in (0, 1)
    ]
    path = tmp_path / "runs.parquet"
    pl.DataFrame(rows).write_parquet(path)
    assert (
        reader.read_runs(reader.Snapshot(), path, NOW)["latest_by_command"][0]["status"]
        == "failed"
    )


def test_clock_tiers_yield_only_needed_groups_without_repeated_filtering():
    rows = pl.DataFrame(
        [
            {
                "as_of_date": START.date(),
                "recorded_at": START - timedelta(minutes=i // 3),
                "release_id": str(i),
            }
            for i in range(3000)
        ]
    )
    with patch.object(
        pl.DataFrame,
        "filter",
        side_effect=AssertionError("tier traversal must not repeatedly filter"),
    ):
        tiers = reader._quality_tiers(rows)
        assert next(tiers).height == 3
        assert next(tiers).height == 3
    assert list(reader._quality_tiers(rows.clear())) == []
