import sqlite3

from conftest import make_station_db, write_config

from grounded_weather_forecast.cli import main


class TestQcCommand:
    def test_qc_happy_path(self, tmp_path, capsys):
        write_config(tmp_path)
        make_station_db(
            tmp_path / "station.db",
            [
                ("2026-07-13 19:21:03", {"outTemp": 70.0, "outHumi": 50.0}),
                ("2026-07-13 19:22:03", {"outTemp": 71.0, "outHumi": 51.0}),
            ],
        )
        code = main(["--config", str(tmp_path / "config.toml"), "qc"])
        out = capsys.readouterr().out
        assert code == 0
        assert "observations: 2 samples" in out
        assert "hourly truth" in out

    def test_qc_empty_db(self, tmp_path, capsys):
        write_config(tmp_path)
        sqlite3.connect(tmp_path / "station.db").close()
        code = main(["--config", str(tmp_path / "config.toml"), "qc"])
        assert code == 1
        assert "no observations" in capsys.readouterr().out

    def test_bad_config(self, tmp_path, capsys):
        (tmp_path / "config.toml").write_text("[station]\n", encoding="utf-8")
        code = main(["--config", str(tmp_path / "config.toml"), "qc"])
        assert code == 2
        assert "config error" in capsys.readouterr().out
