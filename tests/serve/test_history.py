from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, timedelta
from threading import Barrier

import polars as pl
import pytest
from conftest import utc

from grounded_weather_forecast.reports.verification import (
    compare_to_backtest,
    verify_history,
)
from grounded_weather_forecast.serve.history import (
    HISTORY_SCHEMA,
    append_history,
    forecast_to_rows,
    load_archived_forecast,
    load_history,
)
from grounded_weather_forecast.serve.schema import (
    DailyPoint,
    Forecast,
    HourlyPoint,
    MinutelyPoint,
)

ISSUED = utc(2026, 3, 22, 17, 0)


def make_forecast(temp=20.0):
    return Forecast(
        schema_version=1,
        issued_at=ISSUED.isoformat(),
        latitude=34.28,
        longitude=-117.17,
        dataset_fingerprint="abc123",
        sources=["nws"],
        observation_at=ISSUED.isoformat(),
        minutely=[MinutelyPoint(valid_time=ISSUED.isoformat(), minutes_ahead=1)],
        hourly=[
            HourlyPoint(
                valid_time=utc(2026, 3, 22, 18).isoformat(),
                lead_hours=1.0,
                lead_bucket="1-3h",
                values={"temp_c": temp, "wind_speed_ms": None},
                methods={"temp_c": "grounded_equal_weight"},
            )
        ],
        daily=[
            DailyPoint(
                date_local="2026-03-23",
                lead_days=1,
                values={"temp_max_c": 25.0},
                methods={"temp_max_c": "gbm"},
            )
        ],
        timezone="America/Los_Angeles",
    )


class TestHistory:
    def test_flattens_scoreable_rows(self):
        rows = forecast_to_rows(make_forecast())
        assert rows.schema == HISTORY_SCHEMA
        assert rows.height == 2  # null values are not scoreable and are dropped
        hourly = rows.filter(pl.col("product") == "hourly").row(0, named=True)
        assert hourly["variable"] == "temp_c"
        assert hourly["method_id"] == "grounded_equal_weight"
        assert hourly["valid_time"] == utc(2026, 3, 22, 18)
        assert hourly["dataset_fingerprint"] == "abc123"

    def test_append_accumulates(self, tmp_path):
        path = tmp_path / "history.parquet"
        assert append_history(make_forecast(20.0), path) == 2
        assert append_history(make_forecast(21.0), path) == 2
        assert load_history(path).height == 4
        replayed = load_archived_forecast(path, ISSUED.isoformat())
        assert replayed is not None
        assert replayed.to_json() == make_forecast(21.0).to_json()

    def test_concurrent_appends_are_serialized(self, tmp_path):
        path = tmp_path / "history.parquet"
        temperatures = tuple(float(value) for value in range(8))
        barrier = Barrier(len(temperatures))

        def append(temp):
            barrier.wait()
            return append_history(make_forecast(temp), path)

        with ThreadPoolExecutor(max_workers=len(temperatures)) as executor:
            rows_added = tuple(executor.map(append, temperatures))

        assert rows_added == (2,) * len(temperatures)
        assert load_history(path).height == 2 * len(temperatures)

    def test_daily_history_uses_local_midnight(self):
        rows = forecast_to_rows(make_forecast())
        daily = rows.filter(pl.col("product") == "daily").row(0, named=True)
        assert daily["valid_time"] == utc(2026, 3, 23, 7)
        assert str(daily["valid_date"]) == "2026-03-23"

    def test_load_missing_is_empty(self, tmp_path):
        assert load_history(tmp_path / "none.parquet").is_empty()


class TestVerification:
    def truth(self, values):
        return pl.DataFrame(
            {
                "valid_hour": [utc(2026, 3, 22, 18)] * len(values),
                "t__temp_c__inst": values,
            },
            schema={
                "valid_hour": pl.Datetime("us", "UTC"),
                "t__temp_c__inst": pl.Float64,
            },
        )

    def test_scores_served_rows_against_truth(self, tmp_path):
        path = tmp_path / "history.parquet"
        for temp in (18.0, 19.0, 20.0, 21.0, 22.0, 23.0):
            append_history(make_forecast(temp), path)
        # every forecast targeted the same hour; truth for it was 20.0
        live = verify_history(path, self.truth([20.0]))
        row = live.filter(pl.col("product") == "hourly").row(0, named=True)
        assert row["n"] == 6
        assert row["live_mae"] == pytest.approx((2 + 1 + 0 + 1 + 2 + 3) / 6)
        assert row["live_bias"] == pytest.approx(0.5)
        assert row["lead_bucket"] == "1-3h"
        assert row["dataset_fingerprint"] == "abc123"

    def test_compares_to_backtest_expectation(self, tmp_path):
        path = tmp_path / "history.parquet"
        for temp in (19.0, 20.0, 21.0, 22.0, 23.0, 24.0):
            append_history(make_forecast(temp), path)
        live = verify_history(path, self.truth([20.0]))
        board = pl.DataFrame(
            {
                "product": ["hourly"],
                "variable": ["temp_c"],
                "lead_bucket": ["1-3h"],
                "method_id": ["grounded_equal_weight"],
                "n": [100],
                "mae": [1.0],
            }
        )
        compared = compare_to_backtest(live, board)
        row = compared.row(0, named=True)
        assert row["backtest_mae"] == pytest.approx(1.0)
        assert row["mae_gap"] == pytest.approx(row["live_mae"] - 1.0)

    def test_scores_daily_rows_against_daily_truth(self, tmp_path):
        path = tmp_path / "history.parquet"
        for _ in range(6):
            append_history(make_forecast(), path)
        daily_truth = pl.DataFrame(
            {
                "date_local": [date(2026, 3, 23)],
                "t__temp_max_c": [24.0],
            }
        )
        live = verify_history(
            path,
            self.truth([20.0]),
            truth_daily=daily_truth,
        )
        row = live.filter(pl.col("product") == "daily").row(0, named=True)
        assert row["n"] == 6
        assert row["live_mae"] == pytest.approx(1.0)

    def test_scores_daily_when_hourly_truth_is_empty(self, tmp_path):
        path = tmp_path / "history.parquet"
        for _ in range(6):
            append_history(make_forecast(), path)
        daily_truth = pl.DataFrame(
            {
                "date_local": [date(2026, 3, 23)],
                "t__temp_max_c": [24.0],
            }
        )

        live = verify_history(
            path,
            self.truth([]),
            truth_daily=daily_truth,
        )

        row = live.filter(pl.col("product") == "daily").row(0, named=True)
        assert row["n"] == 6
        assert row["live_mae"] == pytest.approx(1.0)

    def test_scores_minutely_rows_at_minute_grain(self, tmp_path):
        path = tmp_path / "history.parquet"
        minutely = MinutelyPoint(
            valid_time=(ISSUED + timedelta(minutes=1)).isoformat(),
            minutes_ahead=1,
            temp_c=20.0,
            methods={"temp_c": "anchored_hourly_blend"},
        )
        for _ in range(6):
            append_history(replace(make_forecast(), minutely=[minutely]), path)
        truth_minute = pl.DataFrame(
            {
                "ts": [ISSUED + timedelta(minutes=1, seconds=30)],
                "temp_c": [19.0],
            },
            schema_overrides={"ts": pl.Datetime("us", "UTC")},
        )
        live = verify_history(
            path,
            self.truth([20.0]),
            truth_minute=truth_minute,
        )
        row = live.filter(pl.col("product") == "minutely").row(0, named=True)
        assert row["n"] == 6
        assert row["live_mae"] == pytest.approx(1.0)

    def test_scores_minutely_when_hourly_truth_is_empty(self, tmp_path):
        path = tmp_path / "history.parquet"
        minutely = MinutelyPoint(
            valid_time=(ISSUED + timedelta(minutes=1)).isoformat(),
            minutes_ahead=1,
            temp_c=20.0,
            methods={"temp_c": "anchored_hourly_blend"},
        )
        for _ in range(6):
            append_history(replace(make_forecast(), minutely=[minutely]), path)
        truth_minute = pl.DataFrame(
            {
                "ts": [ISSUED + timedelta(minutes=1, seconds=30)],
                "temp_c": [19.0],
            },
            schema_overrides={"ts": pl.Datetime("us", "UTC")},
        )

        live = verify_history(
            path,
            self.truth([]),
            truth_minute=truth_minute,
        )

        row = live.filter(pl.col("product") == "minutely").row(0, named=True)
        assert row["n"] == 6
        assert row["live_mae"] == pytest.approx(1.0)

    def test_release_and_bucket_cohorts_are_scored_separately(self, tmp_path):
        path = tmp_path / "history.parquet"
        base = make_forecast()
        long_point = replace(base.hourly[0], lead_hours=30.0)
        for _ in range(5):
            append_history(replace(base, release_ids=["release-short"]), path)
            append_history(
                replace(
                    base,
                    release_ids=["release-long"],
                    hourly=[long_point],
                ),
                path,
            )
        live = verify_history(path, self.truth([20.0]))
        hourly = live.filter(pl.col("product") == "hourly")
        assert hourly.height == 2
        assert set(hourly["lead_bucket"].to_list()) == {"1-3h", "24-48h"}
        assert set(hourly["release_id"].to_list()) == {
            "release-short",
            "release-long",
        }

    def test_empty_history(self, tmp_path):
        assert verify_history(tmp_path / "none.parquet", self.truth([20.0])).is_empty()
