import sqlite3

import pytest
from conftest import make_station_db, write_config

from grounded_weather_forecast.dataset.station import read_observations, sqlite_uri
from grounded_weather_forecast.timeutil import utc


@pytest.fixture
def config(tmp_path):
    return write_config(tmp_path)


class TestReadObservations:
    def test_mixed_precision_and_units(self, tmp_path, config):
        make_station_db(
            tmp_path / "station.db",
            [
                ("2026-07-13 19:21:03.297163", {"outTemp": 32.0, "outHumi": 55.0}),
                ("2026-07-13 19:22:03", {"outTemp": 212.0, "AbsPress": 29.92}),
            ],
        )
        frame = read_observations(config.station)
        assert frame.height == 2
        assert frame["ts"][0] == utc(2026, 7, 13, 19, 21, 3, 297163)
        assert frame["ts"][1] == utc(2026, 7, 13, 19, 22, 3)
        assert frame["temp"][0] == pytest.approx(0.0)
        assert frame["temp"][1] == pytest.approx(100.0)
        assert frame["humidity"][0] == 55.0
        assert frame["pressure_station"][1] == pytest.approx(1013.2, abs=0.1)

    def test_sorts_and_dedupes(self, tmp_path, config):
        make_station_db(
            tmp_path / "station.db",
            [
                ("2026-07-13 19:22:03", {"outTemp": 50.0}),
                ("2026-07-13 19:21:03", {"outTemp": 32.0}),
                # same instant, different precision spelling -> duplicate after parse
                ("2026-07-13 19:21:03.000000", {"outTemp": 99.0}),
            ],
        )
        frame = read_observations(config.station)
        assert frame.height == 2
        assert frame["ts"].is_sorted()
        assert frame["temp"][0] == pytest.approx(0.0)  # first kept

    def test_missing_table_is_empty_frame(self, tmp_path, config):
        sqlite3.connect(tmp_path / "station.db").close()  # empty db file
        frame = read_observations(config.station)
        assert frame.is_empty()
        assert "temp" in frame.columns

    def test_missing_configured_column_is_all_null(self, tmp_path):
        make_station_db(
            tmp_path / "station.db", [("2026-07-13 19:21:03", {"outTemp": 32.0})]
        )
        cfg = write_config(
            tmp_path,
            extra_toml='[station.columns]\nsoilmoist1 = "soil"\n[station.units]\nsoil = "pct"\n',
        )
        frame = read_observations(cfg.station)
        assert frame["soil"].null_count() == frame.height

    def test_unopenable_db_raises(self, config):
        with pytest.raises(OSError, match="cannot open station database"):
            read_observations(config.station)  # file does not exist

    def test_null_ts_rows_dropped(self, tmp_path, config):
        make_station_db(
            tmp_path / "station.db", [("2026-07-13 19:21:03", {"outTemp": 32.0})]
        )
        conn = sqlite3.connect(tmp_path / "station.db")
        conn.execute("INSERT INTO observations (ts, outTemp) VALUES (NULL, 5.0)")
        conn.commit()
        conn.close()
        frame = read_observations(config.station)
        assert frame.height == 1


class TestSqliteUri:
    def test_modes(self, tmp_path):
        assert sqlite_uri(tmp_path / "x.db", immutable=True).endswith("?immutable=1")
        assert sqlite_uri(tmp_path / "x.db", immutable=False).endswith("?mode=ro")
