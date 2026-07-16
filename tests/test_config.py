from datetime import date
from pathlib import Path

import pytest

from grounded_weather_forecast.config import ConfigError, load_config

MINIMAL = """
[station]
db_path = "obs.db"
latitude = 34.28
longitude = -117.17
elevation_m = 1400.0

[forecasts]
db_path = "fx.sqlite"
"""

FULL = """
[station]
db_path = "obs.db"
timezone = "America/Los_Angeles"
latitude = 34.28
longitude = -117.17
elevation_m = 1400.0
immutable = true

[station.columns]
solarrad = "solar"

[station.units]
solar = "wm2"

[forecasts]
db_path = "fx.sqlite"
sources = ["open_meteo", "nws"]
max_forecast_age_hours = 6.0

[dataset]
dir = "mydata"
min_hour_coverage = 0.9

[qc.bounds]
solar = [0.0, 1500.0]

[qc.max_step]
temp = 4.0

[backfill.open_meteo]
models = ["gfs_seamless"]
start_date = 2024-06-01

[backtest]
initial_train_days = 60

[predict]
selection = "pinned"
[predict.methods]
"hourly.temp_c" = "gbm"

[reports]
dir = "out/reports"

[artifacts]
dir = "out/artifacts"
"""


def write(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


class TestMinimalConfig:
    def test_defaults(self, tmp_path):
        cfg = load_config(write(tmp_path, MINIMAL))
        assert cfg.station.db_path == Path("obs.db")
        assert cfg.station.timezone == "UTC"
        assert cfg.station.columns["outTemp"] == "temp"
        assert cfg.station.units["temp"] == "degF"
        assert cfg.forecasts.sources == ()
        assert cfg.forecasts.max_forecast_age_hours == 12.0
        assert cfg.forecasts.immutable is False
        assert cfg.dataset.dir == Path("data")
        assert cfg.dataset.pop_threshold_mm == 0.254
        assert cfg.dataset.precip_reset_fraction == 0.5
        assert cfg.provider_qc.enabled is True
        assert cfg.provider_qc.mad_k == 5.0
        assert cfg.provider_qc.min_sources == 4
        assert cfg.provider_qc.bounds["pressure_sea_hpa"] == (850.0, 1090.0)
        assert "pressure_sea_hpa" in cfg.provider_qc.cross_source_variables
        assert cfg.qc.bounds["temp"] == (-40.0, 55.0)
        assert cfg.qc.max_step["temp"] == 5.0
        assert cfg.qc.flatline_minutes["temp"] == 180
        assert cfg.backfill.models == ()
        assert cfg.backfill.start_date is None
        assert cfg.backtest.initial_train_days == 90
        assert cfg.predict.selection == "skill_per_slice"
        assert cfg.predict.history_path == Path("data/predict_history.parquet")
        assert cfg.predict.minutely_tau_hours == 3.0
        assert cfg.reports_dir == Path("reports")
        assert cfg.artifacts_dir == Path("artifacts")


class TestFullConfig:
    def test_overrides(self, tmp_path):
        cfg = load_config(write(tmp_path, FULL))
        assert cfg.station.immutable is True
        assert cfg.station.elevation_m == 1400.0
        assert cfg.station.columns["solarrad"] == "solar"
        assert cfg.station.columns["outTemp"] == "temp"  # defaults preserved
        assert cfg.station.units["solar"] == "wm2"
        assert cfg.forecasts.sources == ("open_meteo", "nws")
        assert cfg.dataset.dir == Path("mydata")
        assert cfg.dataset.min_hour_coverage == 0.9
        assert cfg.qc.bounds["solar"] == (0.0, 1500.0)
        assert cfg.qc.max_step["temp"] == 4.0
        assert cfg.backfill.models == ("gfs_seamless",)
        assert cfg.backfill.start_date == date(2024, 6, 1)
        assert cfg.backtest.initial_train_days == 60
        assert cfg.backtest.step_days == 7  # default preserved
        assert cfg.predict.selection == "pinned"
        assert cfg.predict.methods["hourly.temp_c"] == "gbm"
        assert cfg.reports_dir == Path("out/reports")


class TestErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="cannot load"):
            load_config(tmp_path / "nope.toml")

    def test_invalid_toml(self, tmp_path):
        with pytest.raises(ConfigError, match="cannot load"):
            load_config(write(tmp_path, "[station\n"))

    def test_missing_required_key(self, tmp_path):
        with pytest.raises(ConfigError, match="missing required key 'db_path'"):
            load_config(write(tmp_path, "[station]\nlatitude=1.0\nlongitude=2.0\n"))

    def test_missing_elevation_rejected(self, tmp_path):
        text = MINIMAL.replace("elevation_m = 1400.0\n", "")
        with pytest.raises(ConfigError, match="missing required key 'elevation_m'"):
            load_config(write(tmp_path, text))

    def test_bad_precip_reset_fraction(self, tmp_path):
        text = MINIMAL + "\n[dataset]\nprecip_reset_fraction = 1.5\n"
        with pytest.raises(ConfigError, match="between 0 and 1"):
            load_config(write(tmp_path, text))

    def test_provider_qc_overrides(self, tmp_path):
        text = (
            MINIMAL
            + "\n[provider_qc]\nenabled = false\nmad_k = 3.0\n"
            + "[provider_qc.bounds]\npressure_sea_hpa = [900.0, 1050.0]\n"
        )
        cfg = load_config(write(tmp_path, text))
        assert cfg.provider_qc.enabled is False
        assert cfg.provider_qc.mad_k == 3.0
        assert cfg.provider_qc.bounds["pressure_sea_hpa"] == (900.0, 1050.0)
        # defaults for un-overridden variables are preserved
        assert cfg.provider_qc.bounds["temp_c"] == (-90.0, 60.0)

    def test_bad_number(self, tmp_path):
        text = MINIMAL.replace("latitude = 34.28", 'latitude = "north"')
        with pytest.raises(ConfigError, match="must be a number"):
            load_config(write(tmp_path, text))

    def test_bad_bounds(self, tmp_path):
        text = MINIMAL + "\n[qc.bounds]\ntemp = [1.0]\n"
        with pytest.raises(ConfigError, match="must be \\[low, high\\]"):
            load_config(write(tmp_path, text))

    def test_bad_sources(self, tmp_path):
        text = MINIMAL.replace(
            'db_path = "fx.sqlite"', 'db_path = "fx.sqlite"\nsources = [1, 2]'
        )
        with pytest.raises(ConfigError, match="list of strings"):
            load_config(write(tmp_path, text))

    def test_bad_columns_map(self, tmp_path):
        text = MINIMAL + "\n[station.columns]\noutTemp = 5\n"
        with pytest.raises(ConfigError, match="table of strings"):
            load_config(write(tmp_path, text))

    def test_start_date_string_accepted(self, tmp_path):
        text = (
            MINIMAL
            + '\n[backfill.open_meteo]\nmodels = []\nstart_date = "2024-01-02"\n'
        )
        cfg = load_config(write(tmp_path, text))
        assert cfg.backfill.start_date == date(2024, 1, 2)

    @pytest.mark.parametrize(
        "raw",
        ('"not-a-date"', "2024-01-02T03:04:05Z"),
    )
    def test_malformed_or_datetime_start_rejected(self, tmp_path, raw):
        text = MINIMAL + f"\n[backfill.open_meteo]\nstart_date = {raw}\n"
        with pytest.raises(ConfigError, match="start_date"):
            load_config(write(tmp_path, text))

    def test_duplicate_canonical_station_alias_rejected(self, tmp_path):
        text = MINIMAL + '\n[station.columns]\nsecond_temp = "temp"\n'
        with pytest.raises(ConfigError, match="multiple database columns"):
            load_config(write(tmp_path, text))

    def test_nonpositive_backtest_step_rejected(self, tmp_path):
        text = MINIMAL + "\n[backtest]\nstep_days = 0\n"
        with pytest.raises(ConfigError, match="positive integer"):
            load_config(write(tmp_path, text))
