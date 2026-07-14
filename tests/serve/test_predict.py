import json
from datetime import timedelta

import pytest
from conftest import make_forecast_db, make_station_db, minute_series, utc, write_config

from omni_forecast.cli import main
from omni_forecast.dataset.matrix import write_dataset
from omni_forecast.serve.predict import (
    NoForecastDataError,
    build_snapshot,
    predict,
)
from omni_forecast.serve.schema import SCHEMA_VERSION

NOW = utc(2026, 3, 22, 17, 0)
FETCH = "2026-03-22T16:30:00+00:00"


def build_fixture(tmp_path, *, obs_temp_f=68.0):
    """A station with recent obs plus two providers with 48h of hourly data.

    The temperature wobbles: a dead-flat series would (correctly) be caught by
    the flatline QC filter and never become truth.
    """
    samples = minute_series(NOW - timedelta(hours=6), 6 * 12, 300)
    station_rows = [
        (
            ts.strftime("%Y-%m-%d %H:%M:%S"),
            {
                "outTemp": obs_temp_f + 0.1 * (index % 5),
                "outHumi": 40.0 + 0.1 * (index % 3),
            },
        )
        for index, ts in enumerate(samples)
    ]
    make_station_db(tmp_path / "station.db", station_rows)

    results = []
    for provider, offset in (("nws", 0.0), ("open_meteo", 2.0)):
        hourly = [
            (
                NOW + timedelta(hours=lead),
                {
                    "temperature": 15.0 + offset,
                    "humidity": 140.0,
                    "wind_speed": -2.0,
                    "precipitation": -5.0,
                    "precipitation_probability": 2.0,
                },
            )
            for lead in range(49)
        ]
        daily = [
            (
                (NOW + timedelta(days=d)).strftime("%Y-%m-%d"),
                {
                    "temperature_max": 22.0 + offset,
                    "temperature_min": 8.0 + offset,
                    "precipitation_sum": -3.0,
                    "precipitation_probability_max": 2.0,
                },
            )
            for d in range(10)
        ]
        results.append(
            {
                "provider": provider,
                "fetched_at": FETCH,
                "hourly": hourly,
                "daily": daily,
                "minutely": [
                    (NOW + timedelta(minutes=m), 0.5, 0.4) for m in range(1, 31)
                ],
            }
        )
    results.append(
        {
            "provider": "stale_minutely",
            "fetched_at": (NOW - timedelta(hours=13)).isoformat(),
            "minutely": [(NOW + timedelta(minutes=1), 99.0, 1.0)],
        }
    )
    make_forecast_db(
        tmp_path / "fx.sqlite",
        [{"completed_at": FETCH, "results": results}],
    )
    config = write_config(tmp_path, min_hour_coverage=0.05, min_day_coverage=0.02)
    write_dataset(config)
    return config


@pytest.fixture
def config(tmp_path):
    return build_fixture(tmp_path)


class TestSnapshot:
    def test_as_of_view(self, config):
        snapshot = build_snapshot(config, NOW)
        assert snapshot.issue_time == NOW
        assert not snapshot.hourly.is_empty()
        assert not snapshot.daily.is_empty()
        assert snapshot.observation_at is not None
        assert snapshot.observation["temp_c"] == pytest.approx(20.0, abs=0.1)
        assert (snapshot.hourly["lead_hours"] >= 0).all()
        assert "stale_minutely" not in snapshot.minutely["source"].to_list()

    def test_stale_archive_refuses(self, config):
        with pytest.raises(NoForecastDataError, match="no provider forecast"):
            build_snapshot(config, NOW + timedelta(days=30))


class TestPredict:
    def test_three_products(self, config):
        document = predict(config, {}, now=NOW)
        assert document.schema_version == SCHEMA_VERSION
        assert len(document.minutely) == 60
        assert len(document.hourly) == 49
        assert len(document.daily) == 10
        assert set(document.sources) == {"nws", "open_meteo"}

    def test_json_round_trip(self, config):
        payload = json.loads(predict(config, {}, now=NOW).to_json())
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["hourly"][0]["lead_bucket"] == "0-1h"
        assert "temp_c" in payload["hourly"][0]["values"]
        assert "temp_c" in payload["hourly"][0]["methods"]

    def test_minutely_is_anchored_to_the_observation(self, config):
        # providers say 15-17C; the station says 20C. The nowcast must start
        # near the observation and relax toward the blend.
        document = predict(config, {}, now=NOW)
        first = document.minutely[0].temp_c
        last = document.minutely[-1].temp_c
        blend_now = document.hourly[0].values["temp_c"]
        assert first is not None
        assert last is not None
        assert blend_now is not None
        assert abs(first - 20.0) < abs(first - blend_now)  # anchored to obs
        assert abs(last - blend_now) < abs(first - blend_now)  # relaxing back

    def test_minutely_precip_from_native_points(self, config):
        document = predict(config, {}, now=NOW)
        assert document.minutely[0].precip_intensity_mmh == pytest.approx(0.5)
        assert document.minutely[0].pop == pytest.approx(0.4)
        # providers only published 30 minutes of it
        assert document.minutely[-1].precip_intensity_mmh is None

    def test_daily_hi_lo(self, config):
        document = predict(config, {}, now=NOW)
        values = document.daily[0].values
        assert values["temp_max_c"] is not None
        assert values["temp_min_c"] is not None
        assert values["temp_max_c"] > values["temp_min_c"]

    def test_emitted_values_obey_physical_bounds(self, config):
        document = predict(config, {}, now=NOW)
        for point in document.hourly:
            assert point.values["humidity_pct"] == 100.0
            assert point.values["wind_speed_ms"] == 0.0
            assert point.values["precip_mm"] == 0.0
            assert point.values["pop"] == 1.0
        for point in document.daily:
            assert point.values["precip_sum_mm"] == 0.0
            assert point.values["pop"] == 1.0

    def test_force_method(self, config):
        document = predict(config, {}, now=NOW, force_method="equal_weight")
        methods = {m for p in document.hourly for m in p.methods.values()}
        assert methods == {"equal_weight"}

    def test_fallback_method_when_no_scores(self, config):
        document = predict(config, {}, now=NOW)
        methods = {m for p in document.hourly for m in p.methods.values()}
        assert methods == {"equal_weight"}
        assert document.status == "degraded"


class TestPredictCli:
    def test_writes_json_and_history(self, tmp_path, capsys):
        config = build_fixture(tmp_path)
        out = tmp_path / "forecast.json"
        code = main(
            [
                "--config",
                str(tmp_path / "config.toml"),
                "predict",
                "--now",
                NOW.isoformat(),
                "--out",
                str(out),
            ]
        )
        assert code == 0
        printed = capsys.readouterr().out
        assert "wrote" in printed
        assert "appended" in printed
        assert out.exists()
        assert config.predict.history_path.exists()

    def test_stale_archive_exits_nonzero(self, tmp_path, capsys):
        build_fixture(tmp_path)
        code = main(
            [
                "--config",
                str(tmp_path / "config.toml"),
                "predict",
                "--now",
                (NOW + timedelta(days=30)).isoformat(),
                "--no-history",
            ]
        )
        assert code == 1
        assert "cannot predict" in capsys.readouterr().out

    def test_stdout_is_json_only(self, tmp_path, capsys):
        build_fixture(tmp_path)
        code = main(
            [
                "--config",
                str(tmp_path / "config.toml"),
                "predict",
                "--now",
                NOW.isoformat(),
            ]
        )
        assert code == 0
        captured = capsys.readouterr()
        assert json.loads(captured.out)["issued_at"] == NOW.isoformat()
        assert "appended" in captured.err
