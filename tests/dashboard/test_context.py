import json
from datetime import UTC, datetime

import polars as pl
from conftest import (
    make_forecast_db,
    make_station_db,
    synthetic_hourly_matrix,
    write_config,
)

from grounded_weather_forecast.dashboard.context import collect_context
from grounded_weather_forecast.dataset.matrix import matrix_path

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def test_bare_config_collects_all_absent_without_raising(tmp_path):
    ctx = collect_context(write_config(tmp_path), now=NOW)
    assert ctx.manifest is None
    assert ctx.truth_minute is None
    assert ctx.truth_hourly is None
    assert ctx.hourly_matrix is None
    assert ctx.score_frames == {}
    assert ctx.history is None
    assert ctx.latest_forecast is None
    assert ctx.releases == ()
    assert ctx.alignment is None
    assert ctx.drift is None
    assert ctx.observability_states == ()
    assert ctx.observability_history.is_empty()
    assert ctx.runs.is_empty()


def test_corrupted_manifest_loads_as_none(tmp_path):
    config = write_config(tmp_path)
    config.dataset.dir.mkdir(parents=True, exist_ok=True)
    (config.dataset.dir / "manifest.json").write_text("{not json", encoding="utf-8")
    assert collect_context(config, now=NOW).manifest is None


def test_populated_context_loads_matrix_and_manifest(tmp_path):
    config = write_config(tmp_path)
    config.dataset.dir.mkdir(parents=True, exist_ok=True)
    matrix = synthetic_hourly_matrix(days=5)
    matrix.write_parquet(matrix_path(config.dataset.dir, "hourly", "live"))
    (config.dataset.dir / "manifest.json").write_text(
        json.dumps({"fingerprint": "abc", "sources": ["alpha", "beta"]}),
        encoding="utf-8",
    )
    ctx = collect_context(config, now=NOW)
    assert ctx.hourly_matrix is not None
    assert ctx.hourly_matrix.height == matrix.height
    assert ctx.manifest is not None
    assert ctx.manifest["fingerprint"] == "abc"


def test_context_reads_actual_archive_location(tmp_path):
    config = write_config(tmp_path)
    make_forecast_db(
        config.forecasts.db_path,
        [
            {
                "completed_at": NOW.isoformat(),
                "latitude": 35.0,
                "longitude": -118.0,
                "results": [],
            }
        ],
    )

    assert collect_context(config, now=NOW).archive_location == (35.0, -118.0)


def test_qc_distinguishes_recovered_flatline_from_active_state(tmp_path):
    config = write_config(
        tmp_path,
        extra_toml="\n[qc.flatline_minutes]\ntemp = 2\n",
    )
    make_station_db(
        config.station.db_path,
        [
            ("2026-07-18 04:00:00", {"outTemp": 70.0}),
            ("2026-07-18 04:01:00", {"outTemp": 70.0}),
            ("2026-07-18 04:02:00", {"outTemp": 70.0}),
            ("2026-07-18 04:03:00", {"outTemp": 71.0}),
        ],
    )

    qc = collect_context(config, now=NOW).qc

    assert qc is not None
    temp = qc.filter(pl.col("channel") == "temp").row(0, named=True)
    assert temp["flatline"] > 0
    assert temp["active_flatline"] is False
