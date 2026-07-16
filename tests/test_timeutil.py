from datetime import date

import polars as pl

from grounded_weather_forecast.timeutil import (
    local_date_expr,
    local_day_minutes,
    local_day_start_utc,
    parse_iso_utc_expr,
    parse_station_ts_expr,
    utc,
)

LA = "America/Los_Angeles"


class TestStationTsParsing:
    def test_mixed_precision(self):
        frame = pl.DataFrame(
            {"ts": ["2026-07-13 19:21:03.297163", "2025-06-14 18:39:53"]}
        )
        parsed = frame.select(parse_station_ts_expr(pl.col("ts")).alias("dt"))["dt"]
        assert parsed[0] == utc(2026, 7, 13, 19, 21, 3, 297163)
        assert parsed[1] == utc(2025, 6, 14, 18, 39, 53)
        assert parsed.dtype == pl.Datetime("us", "UTC")

    def test_garbage_is_null(self):
        frame = pl.DataFrame({"ts": ["not a timestamp"]})
        parsed = frame.select(parse_station_ts_expr(pl.col("ts")).alias("dt"))["dt"]
        assert parsed[0] is None


class TestIsoParsing:
    def test_utc_offset(self):
        frame = pl.DataFrame({"ts": ["2026-03-22T16:19:40.865310+00:00"]})
        parsed = frame.select(parse_iso_utc_expr(pl.col("ts")).alias("dt"))["dt"]
        assert parsed[0] == utc(2026, 3, 22, 16, 19, 40, 865310)

    def test_nonzero_offset_converts(self):
        frame = pl.DataFrame({"ts": ["2026-03-22T09:00:00-07:00"]})
        parsed = frame.select(parse_iso_utc_expr(pl.col("ts")).alias("dt"))["dt"]
        assert parsed[0] == utc(2026, 3, 22, 16, 0, 0)


class TestLocalDay:
    def test_local_date_crosses_midnight(self):
        frame = pl.DataFrame({"dt": [utc(2026, 7, 14, 5, 30)]})  # 22:30 LA on Jul 13
        got = frame.select(local_date_expr(pl.col("dt"), LA).alias("d"))["d"]
        assert got[0] == date(2026, 7, 13)

    def test_spring_forward_day_has_23_hours(self):
        assert local_day_minutes(date(2026, 3, 8), LA) == 1380

    def test_fall_back_day_has_25_hours(self):
        assert local_day_minutes(date(2026, 11, 1), LA) == 1500

    def test_standard_day(self):
        assert local_day_minutes(date(2026, 7, 13), LA) == 1440

    def test_day_start_utc(self):
        assert local_day_start_utc(date(2026, 7, 13), LA) == utc(2026, 7, 13, 7)
        assert local_day_start_utc(date(2026, 1, 13), LA) == utc(2026, 1, 13, 8)
