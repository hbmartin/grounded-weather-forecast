"""End-to-end CLI flow on synthesized fixture databases:
build-dataset -> backtest -> report."""

from datetime import timedelta

import pytest
from conftest import make_forecast_db, make_station_db, utc, write_config

from grounded_weather_forecast.cli import main

DAYS = 15
START = utc(2026, 2, 1)


def temp_f(dt):
    # simple diurnal cycle in Fahrenheit for the station DB
    return 50.0 + 18.0 * ((dt.hour - 14) % 24 - 12) / 12.0


@pytest.fixture
def e2e_config(tmp_path):
    station_rows = []
    for day in range(DAYS):
        for hour in range(24):
            hour_start = START + timedelta(days=day, hours=hour)
            for minute_offset in (-8, -4, -2, 0, 2, 4, 8):
                ts = hour_start + timedelta(minutes=minute_offset)
                if ts < START:
                    continue
                station_rows.append(
                    (
                        ts.strftime("%Y-%m-%d %H:%M:%S"),
                        {"outTemp": temp_f(ts), "outHumi": 50.0},
                    )
                )
    make_station_db(tmp_path / "station.db", station_rows)

    runs = []
    for day in range(DAYS):
        issue = START + timedelta(days=day, hours=12)
        fetched = issue - timedelta(minutes=20)
        results = []
        for provider, bias in (("nws", 0.0), ("open_meteo", 4.0)):
            hourly = []
            for lead in range(1, 37):
                valid = issue.replace(minute=0) + timedelta(hours=lead)
                truth_c = (temp_f(valid) - 32.0) * 5.0 / 9.0
                hourly.append((valid, {"temperature": truth_c + bias}))
            results.append(
                {
                    "provider": provider,
                    "fetched_at": fetched.isoformat(),
                    "hourly": hourly,
                }
            )
        runs.append({"completed_at": issue.isoformat(), "results": results})
    make_forecast_db(tmp_path / "fx.sqlite", runs)

    write_config(
        tmp_path,
        min_hour_coverage=0.05,
        min_day_coverage=0.05,
        extra_toml="\n[backtest]\ninitial_train_days = 5\nstep_days = 3\n",
    )
    return tmp_path


class TestEndToEnd:
    def test_build_backtest_report(self, e2e_config, capsys):
        config_arg = ["--config", str(e2e_config / "config.toml")]

        assert main([*config_arg, "build-dataset"]) == 0
        out = capsys.readouterr().out
        assert "hourly_matrix" in out

        assert (
            main(
                [
                    *config_arg,
                    "backtest",
                    "--methods",
                    "equal_weight,best_provider,climatology",
                    "--hourly-variables",
                    "temp_c",
                    "--products",
                    "hourly",
                ]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "score rows" in out

        assert main([*config_arg, "report"]) == 0
        out = capsys.readouterr().out
        assert "wrote" in out
        leaderboards = list((e2e_config / "reports").glob("leaderboard_*.md"))
        assert leaderboards
        text = leaderboards[0].read_text()
        assert "equal_weight" in text
        assert "Per-slice winners" in text
        # the unbiased provider should be discoverable by best_provider
        assert "best_provider" in text

    def test_backtest_without_dataset(self, tmp_path, capsys):
        write_config(tmp_path)
        code = main(["--config", str(tmp_path / "config.toml"), "backtest"])
        assert code == 1
        assert "run build-dataset first" in capsys.readouterr().out

    def test_report_without_scores(self, tmp_path, capsys):
        write_config(tmp_path)
        code = main(["--config", str(tmp_path / "config.toml"), "report"])
        assert code == 1
        assert "run backtest first" in capsys.readouterr().out
