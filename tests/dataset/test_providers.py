import sqlite3

import pytest
from conftest import make_forecast_db, utc, write_config

from grounded_weather_forecast.dataset.providers import (
    read_forecast_archive,
    read_daily_long,
    read_hourly_long,
    read_latest_archive_location,
    read_minutely_long,
    read_run_completions,
    source_slug,
)

FETCH = "2026-03-22T12:00:00+00:00"
VALID = utc(2026, 3, 22, 18, 0)


def one_run_db(tmp_path, results):
    return make_forecast_db(
        tmp_path / "fx.sqlite",
        [{"completed_at": "2026-03-22T12:00:30+00:00", "results": results}],
    )


class TestSourceSlug:
    def test_slugs(self):
        assert source_slug("open_meteo", "best_match") == "open_meteo"
        assert source_slug("nws", "nws") == "nws"
        assert source_slug("open_meteo", "ecmwf_ifs025") == "open_meteo_ecmwf_ifs025"
        assert source_slug("met_norway", "") == "met_norway"


class TestReadHourlyLong:
    def test_lead_recomputed_from_iso_fetched_at(self, tmp_path, la_config):
        one_run_db(
            tmp_path,
            [
                {
                    "provider": "nws",
                    "fetched_at": FETCH,
                    "hourly": [(VALID, {"temperature": 10.0})],
                }
            ],
        )
        frame = read_hourly_long(la_config.forecasts)
        assert frame.height == 1
        row = frame.row(0, named=True)
        assert row["lead_hours"] == pytest.approx(6.0)
        assert row["fetched_at"] == utc(2026, 3, 22, 12)
        assert row["valid_time"] == VALID
        assert row["temp_c"] == 10.0
        assert row["source"] == "nws"
        assert row["source_kind"] == "live"

    def test_error_results_excluded(self, tmp_path, la_config):
        one_run_db(
            tmp_path,
            [
                {
                    "provider": "nws",
                    "status": "error",
                    "fetched_at": FETCH,
                    "hourly": [(VALID, {"temperature": 10.0})],
                }
            ],
        )
        assert read_hourly_long(la_config.forecasts).is_empty()

    def test_null_fetched_at_dropped(self, tmp_path, la_config):
        one_run_db(
            tmp_path,
            [
                {
                    "provider": "nws",
                    "fetched_at": None,
                    "hourly": [(VALID, {"temperature": 10.0})],
                }
            ],
        )
        assert read_hourly_long(la_config.forecasts).is_empty()

    def test_source_allowlist(self, tmp_path):
        one_run_db(
            tmp_path,
            [
                {
                    "provider": "nws",
                    "fetched_at": FETCH,
                    "hourly": [(VALID, {"temperature": 10.0})],
                },
                {
                    "provider": "weatherbit",
                    "fetched_at": FETCH,
                    "hourly": [(VALID, {"temperature": 11.0})],
                },
            ],
        )
        cfg = write_config(tmp_path, sources=["nws"])
        frame = read_hourly_long(cfg.forecasts)
        assert frame["source"].unique().to_list() == ["nws"]

    def test_dedupe_keeps_last(self, tmp_path, la_config):
        one_run_db(
            tmp_path,
            [
                {
                    "provider": "nws",
                    "fetched_at": FETCH,
                    "hourly": [
                        (VALID, {"temperature": 10.0}),
                        (VALID, {"temperature": 99.0}),
                    ],
                }
            ],
        )
        frame = read_hourly_long(la_config.forecasts)
        assert frame.height == 1

    def test_missing_table_is_empty(self, tmp_path, la_config):
        connection = sqlite3.connect(tmp_path / "fx.sqlite")
        connection.execute("CREATE TABLE forecast_runs (id INTEGER)")
        connection.close()
        assert read_hourly_long(la_config.forecasts).is_empty()

    def test_missing_file_raises(self, la_config):
        with pytest.raises(OSError, match="cannot open forecast archive"):
            read_hourly_long(la_config.forecasts)

    def test_explicit_model_gets_composite_slug(self, tmp_path, la_config):
        one_run_db(
            tmp_path,
            [
                {
                    "provider": "open_meteo",
                    "model": "ecmwf_ifs025",
                    "fetched_at": FETCH,
                    "hourly": [(VALID, {"temperature": 8.0})],
                }
            ],
        )
        frame = read_hourly_long(la_config.forecasts)
        assert frame["source"][0] == "open_meteo_ecmwf_ifs025"

    def test_filters_runs_for_other_locations(self, tmp_path, la_config):
        make_forecast_db(
            tmp_path / "fx.sqlite",
            [
                {
                    "completed_at": FETCH,
                    "results": [
                        {
                            "provider": "local",
                            "fetched_at": FETCH,
                            "hourly": [(VALID, {"temperature": 10.0})],
                        }
                    ],
                },
                {
                    "latitude": 40.0,
                    "longitude": -75.0,
                    "completed_at": "2026-03-22T12:01:00+00:00",
                    "results": [
                        {
                            "provider": "other_location",
                            "fetched_at": FETCH,
                            "hourly": [(VALID, {"temperature": 99.0})],
                        }
                    ],
                },
            ],
        )
        archive = read_forecast_archive(la_config.forecasts)
        assert archive.hourly["source"].unique().to_list() == ["local"]
        assert archive.completions.height == 1

    def test_live_wal_rows_are_visible(self, tmp_path, la_config):
        one_run_db(
            tmp_path,
            [
                {
                    "provider": "nws",
                    "fetched_at": FETCH,
                    "hourly": [(VALID, {"temperature": 10.0})],
                }
            ],
        )
        writer = sqlite3.connect(tmp_path / "fx.sqlite")
        try:
            writer.execute("PRAGMA journal_mode=WAL")
            source_id = writer.execute("SELECT id FROM source_forecasts").fetchone()[0]
            writer.execute(
                "INSERT INTO hourly_points (source_forecast_id, timestamp_unix, temperature)"
                " VALUES (?, ?, ?)",
                (source_id, int(utc(2026, 3, 22, 19).timestamp()), 11.0),
            )
            writer.commit()
            assert read_hourly_long(la_config.forecasts).height == 2
        finally:
            writer.close()


class TestReadDailyAndMinutely:
    def test_daily_parse(self, tmp_path, la_config):
        one_run_db(
            tmp_path,
            [
                {
                    "provider": "weatherbit",
                    "fetched_at": FETCH,
                    "daily": [
                        (
                            "2026-03-23",
                            {
                                "temperature_max": 15.0,
                                "temperature_min": 3.0,
                                "precipitation_sum": 1.2,
                                "precipitation_probability_max": 0.4,
                            },
                        )
                    ],
                }
            ],
        )
        frame = read_daily_long(la_config.forecasts)
        row = frame.row(0, named=True)
        assert str(row["forecast_date"]) == "2026-03-23"
        assert row["temp_max_c"] == 15.0
        assert row["pop"] == 0.4

    def test_minutely_parse(self, tmp_path, la_config):
        one_run_db(
            tmp_path,
            [
                {
                    "provider": "pirate_weather",
                    "fetched_at": FETCH,
                    "minutely": [(VALID, 2.5, 0.9)],
                }
            ],
        )
        frame = read_minutely_long(la_config.forecasts)
        row = frame.row(0, named=True)
        assert row["precip_intensity_mmh"] == 2.5
        assert row["pop"] == 0.9


class TestRunCompletions:
    def test_parse(self, tmp_path, la_config):
        one_run_db(tmp_path, [])
        completions = read_run_completions(la_config.forecasts)
        assert completions.height == 1
        assert completions["completed_at"][0] == utc(2026, 3, 22, 12, 0, 30)


def test_latest_archive_location_is_not_station_filtered(tmp_path, la_config):
    make_forecast_db(
        tmp_path / "fx.sqlite",
        [
            {
                "completed_at": "2026-03-22T12:00:30+00:00",
                "latitude": 35.0,
                "longitude": -118.0,
                "results": [],
            }
        ],
    )

    assert read_latest_archive_location(la_config.forecasts) == (35.0, -118.0)
