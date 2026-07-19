from datetime import timedelta

import polars as pl
import pytest
from conftest import canonical_minute_frame, minute_series, utc, write_config

from grounded_weather_forecast.dataset.qc import apply_qc
from grounded_weather_forecast.dataset.truth import (
    truth_daily,
    truth_hourly,
    truth_minute,
)

NOON = utc(2026, 7, 13, 12, 0)


@pytest.fixture
def config(tmp_path):
    return write_config(tmp_path)


class TestTruthMinute:
    def test_mapping_and_derivation(self, config):
        ts = minute_series(NOON, 3)
        raw = pl.DataFrame(
            {
                "ts": ts,
                "temp": [20.0, 25.0, 99.0],  # 99 exceeds bounds -> flagged
                "humidity": [100.0, 50.0, 50.0],
                "wind_speed": [2.0, None, 3.0],
                "wind_gust": [4.0, 5.0, 6.0],
                "pressure_station": [846.6, 846.0, 845.0],
                "rain_counter": [0.0, 1.0, 1.5],
            },
            schema={
                "ts": pl.Datetime("us", "UTC"),
                **dict.fromkeys(
                    [
                        "temp",
                        "humidity",
                        "wind_speed",
                        "wind_gust",
                        "pressure_station",
                        "rain_counter",
                    ],
                    pl.Float64,
                ),
            },
        )
        flagged = apply_qc(raw, config.qc, sorted(set(config.station.columns.values())))
        minute = truth_minute(flagged, config)
        assert minute["temp_c"].to_list()[:2] == [20.0, 25.0]
        assert minute["temp_c"][2] is None  # out of bounds -> masked
        assert minute["dew_point_c"][0] == pytest.approx(20.0, abs=1e-6)
        assert minute["dew_point_c"][2] is None  # temp flagged -> derived null
        assert minute["pressure_sea_hpa"][0] == pytest.approx(994.2, abs=1.0)
        assert minute["pressure_sea_hpa"][2] is None  # depends on flagged temp
        assert minute["wind_speed_ms"][1] is None
        assert minute["rain_counter_mm"].to_list() == [0.0, 1.0, 1.5]


def hourly_config(tmp_path, **kw):
    return write_config(tmp_path, **kw)


class TestTruthHourlyInstantaneous:
    def test_primary_window(self, tmp_path):
        config = hourly_config(tmp_path)
        ts = [NOON - timedelta(minutes=2), NOON + timedelta(minutes=2)]
        minute = canonical_minute_frame(ts, temp_c=[10.0, 12.0])
        hourly = truth_hourly(minute, config)
        row = hourly.filter(pl.col("valid_hour") == NOON).row(0, named=True)
        assert row["t__temp_c__inst"] == pytest.approx(11.0)

    def test_fallback_window(self, tmp_path):
        config = hourly_config(tmp_path)
        ts = [NOON - timedelta(minutes=8)]
        minute = canonical_minute_frame(ts, temp_c=[10.0])
        hourly = truth_hourly(minute, config)
        row = hourly.filter(pl.col("valid_hour") == NOON).row(0, named=True)
        assert row["t__temp_c__inst"] == pytest.approx(10.0)

    def test_beyond_fallback_is_null(self, tmp_path):
        config = hourly_config(tmp_path)
        ts = [NOON - timedelta(minutes=20)]
        minute = canonical_minute_frame(ts, temp_c=[10.0])
        hourly = truth_hourly(minute, config)
        row = hourly.filter(pl.col("valid_hour") == NOON)
        assert row.is_empty() or row.row(0, named=True)["t__temp_c__inst"] is None


class TestTruthHourlyIntervalMean:
    def test_full_coverage(self, tmp_path):
        config = hourly_config(tmp_path)
        ts = minute_series(NOON, 60)
        minute = canonical_minute_frame(ts, temp_c=[float(i) for i in range(60)])
        hourly = truth_hourly(minute, config)
        row = hourly.filter(pl.col("valid_hour") == NOON).row(0, named=True)
        assert row["t__temp_c__mean"] == pytest.approx(29.5)

    def test_low_coverage_is_null(self, tmp_path):
        config = hourly_config(tmp_path)
        ts = minute_series(NOON, 30)  # 50% < 80%
        minute = canonical_minute_frame(ts, temp_c=[20.0] * 30)
        hourly = truth_hourly(minute, config)
        row = hourly.filter(pl.col("valid_hour") == NOON).row(0, named=True)
        assert row["t__temp_c__mean"] is None

    def test_gust_is_hour_max(self, tmp_path):
        config = hourly_config(tmp_path, min_hour_coverage=0.4)
        ts = minute_series(NOON, 30)
        gusts = [3.0] * 29 + [15.0]
        minute = canonical_minute_frame(ts, wind_gust_ms=gusts)
        hourly = truth_hourly(minute, config)
        row = hourly.filter(pl.col("valid_hour") == NOON).row(0, named=True)
        assert row["t__wind_gust_ms"] == 15.0


class TestTruthHourlyPrecip:
    def test_counter_sum(self, tmp_path):
        config = hourly_config(tmp_path, min_hour_coverage=0.4)
        ts = minute_series(NOON, 40)
        counter = [0.0] * 10 + [0.5] * 10 + [2.0] * 20
        minute = canonical_minute_frame(ts, rain_counter_mm=counter)
        hourly = truth_hourly(minute, config)
        row = hourly.filter(pl.col("valid_hour") == NOON).row(0, named=True)
        assert row["t__precip_mm"] == pytest.approx(2.0)
        assert row["t__pop"] == 1.0

    def test_counter_reset(self, tmp_path):
        config = hourly_config(tmp_path, min_hour_coverage=0.4)
        ts = minute_series(NOON, 40)
        counter = [5.0] * 20 + [0.5] * 20  # reset: delta = new value 0.5
        minute = canonical_minute_frame(ts, rain_counter_mm=counter)
        hourly = truth_hourly(minute, config)
        row = hourly.filter(pl.col("valid_hour") == NOON).row(0, named=True)
        assert row["t__precip_mm"] == pytest.approx(0.5)

    def test_counter_noise_dip_no_phantom_rain(self, tmp_path):
        config = hourly_config(tmp_path, min_hour_coverage=0.4)
        ts = minute_series(NOON, 40)
        # A small sensor jitter (10.0 -> 9.8) is NOT a reset. The old rule treated
        # every decrease as a reset and turned this into 9.8 mm of phantom rain.
        counter = [10.0] * 20 + [9.8] * 20
        minute = canonical_minute_frame(ts, rain_counter_mm=counter)
        hourly = truth_hourly(minute, config)
        row = hourly.filter(pl.col("valid_hour") == NOON).row(0, named=True)
        assert row["t__precip_mm"] == pytest.approx(0.0)

    def test_counter_dip_and_rebound_adds_no_phantom_rain(self, tmp_path):
        config = hourly_config(tmp_path, min_hour_coverage=0.4)
        ts = minute_series(NOON, 40)
        # A dip and rebound (10.0 -> 9.8 -> 10.0) must credit no rain: the running
        # max within the epoch never exceeds 10.0. A fraction-only rule would have
        # credited 0.2 mm on the rebound.
        counter = [10.0] * 10 + [9.8] * 10 + [10.0] * 20
        minute = canonical_minute_frame(ts, rain_counter_mm=counter)
        hourly = truth_hourly(minute, config)
        row = hourly.filter(pl.col("valid_hour") == NOON).row(0, named=True)
        assert row["t__precip_mm"] == pytest.approx(0.0)

    def test_counter_reset_then_reaccumulate(self, tmp_path):
        config = hourly_config(tmp_path, min_hour_coverage=0.4)
        ts = minute_series(NOON, 40)
        # A genuine reset (10.0 -> 0.3, a >50% drop) still counts its new value as
        # the accumulation since the reset.
        counter = [10.0] * 20 + [0.3] * 20
        minute = canonical_minute_frame(ts, rain_counter_mm=counter)
        hourly = truth_hourly(minute, config)
        row = hourly.filter(pl.col("valid_hour") == NOON).row(0, named=True)
        assert row["t__precip_mm"] == pytest.approx(0.3)

    def test_dry_hour_pop_zero(self, tmp_path):
        config = hourly_config(tmp_path, min_hour_coverage=0.4)
        ts = minute_series(NOON, 40)
        minute = canonical_minute_frame(ts, rain_counter_mm=[3.0] * 40)
        hourly = truth_hourly(minute, config)
        row = hourly.filter(pl.col("valid_hour") == NOON).row(0, named=True)
        assert row["t__precip_mm"] == 0.0
        assert row["t__pop"] == 0.0

    def test_long_gap_delta_dropped(self, tmp_path):
        config = hourly_config(tmp_path, min_hour_coverage=0.4)
        # 40 samples but a 15-minute gap in the middle where the counter jumped
        ts = minute_series(NOON, 20) + minute_series(NOON + timedelta(minutes=35), 25)
        counter = [0.0] * 20 + [4.0] * 25
        minute = canonical_minute_frame(ts, rain_counter_mm=counter)
        hourly = truth_hourly(minute, config)
        row = hourly.filter(pl.col("valid_hour") == NOON).row(0, named=True)
        # the 4mm jump spans an unattributable gap -> excluded from the sum
        assert row["t__precip_mm"] == pytest.approx(0.0)

    def test_low_coverage_precip_null(self, tmp_path):
        config = hourly_config(tmp_path)  # 0.8 required
        ts = minute_series(NOON, 10)
        minute = canonical_minute_frame(ts, rain_counter_mm=[0.0] * 10)
        hourly = truth_hourly(minute, config)
        row = hourly.filter(pl.col("valid_hour") == NOON).row(0, named=True)
        assert row["t__precip_mm"] is None
        assert row["t__pop"] is None


class TestTruthDaily:
    def test_extremes_and_coverage_denominator(self, tmp_path):
        # 2026-03-08 is the 23-hour spring-forward day in America/Los_Angeles:
        # 100 clean minutes / 1380 > 0.07, while on a 1440-minute day it fails.
        config = write_config(tmp_path, min_day_coverage=100 / 1400)
        dst_start = utc(2026, 3, 8, 18, 0)  # 10:00 local
        normal_start = utc(2026, 7, 13, 18, 0)
        ts = minute_series(dst_start, 100) + minute_series(normal_start, 100)
        temps = [float(i % 30) for i in range(100)] + [20.0] * 100
        minute = canonical_minute_frame(ts, temp_c=temps)
        daily = truth_daily(minute, config)
        dst_row = daily.filter(pl.col("date_local") == pl.date(2026, 3, 8)).row(
            0, named=True
        )
        normal_row = daily.filter(pl.col("date_local") == pl.date(2026, 7, 13)).row(
            0, named=True
        )
        assert dst_row["t__temp_max_c"] == 29.0  # 100/1380 passes
        assert normal_row["t__temp_max_c"] is None  # 100/1440 fails

    def test_precip_sum_local_day(self, tmp_path):
        config = write_config(tmp_path, min_day_coverage=0.05)
        start = utc(2026, 7, 13, 18, 0)
        ts = minute_series(start, 120)
        counter = [0.0] * 30 + [1.0] * 30 + [2.5] * 60
        # The two coverage columns count different channels, so the fixture
        # gives them different denominators: equal counts would let the test
        # pass even if `rain_coverage` were aliased to `coverage_frac`.
        minute = canonical_minute_frame(
            ts, rain_counter_mm=counter, temp_c=[20.0] * 90 + [None] * 30
        )
        daily = truth_daily(minute, config)
        row = daily.filter(pl.col("date_local") == pl.date(2026, 7, 13)).row(
            0, named=True
        )
        assert row["t__precip_sum_mm"] == pytest.approx(2.5)
        assert row["t__pop"] == 1.0
        assert row["coverage_frac"] == pytest.approx(90 / 1440)
        assert row["rain_coverage"] == pytest.approx(120 / 1440)
