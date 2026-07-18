import json
import sqlite3
from datetime import date

import polars as pl
from conftest import make_station_db, write_config

from grounded_weather_forecast.cli import build_parser, main
from grounded_weather_forecast.dataset.neighbors import NeighborChecks
from grounded_weather_forecast.serve.predict import UnsupportedMethodError


def test_backfill_start_and_truth_qc_days_are_parsed():
    parser = build_parser()
    backfill = parser.parse_args(
        ["backfill", "--provider", "open_meteo", "--start", "2026-02-01"]
    )
    truth_qc = parser.parse_args(["truth-qc", "--days", "45"])

    assert backfill.start == date(2026, 2, 1)
    assert truth_qc.days == 45


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

    def test_truth_qc_writes_v2_unknown_artifact_and_exits_two(
        self, tmp_path, capsys, monkeypatch
    ):
        config = write_config(
            tmp_path,
            extra_toml='[truth_qc]\nsynoptic_token = "test-token"\n',
        )
        make_station_db(
            tmp_path / "station.db",
            [("2026-07-13 19:21:03", {"outTemp": 70.0, "avgwind": 1.0})],
        )
        empty_daily = pl.DataFrame(
            schema={"date": pl.Date(), "station_minus_consensus_c": pl.Float64()}
        )
        empty_correlation = pl.DataFrame(
            schema={
                "valid_hour": pl.Datetime("us", "UTC"),
                "correlation": pl.Float64(),
            }
        )
        empty_comparison = pl.DataFrame(
            schema={
                "valid_hour": pl.Datetime("us", "UTC"),
                "t__temp_c__inst": pl.Float64(),
                "consensus_c": pl.Float64(),
                "difference": pl.Float64(),
            }
        )
        checks = NeighborChecks(
            daily_drift=empty_daily,
            rolling_correlation=empty_correlation,
            comparison=empty_comparison,
            drift_alert=None,
            correlation_alert=None,
            n_neighbors=0,
            overlap_hours=0,
            drift_reason="no overlap",
            correlation_reason="no overlap",
        )
        monkeypatch.setattr(
            "grounded_weather_forecast.dataset.neighbors.fetch_neighbor_checks",
            lambda *_args, **_kwargs: checks,
        )

        code = main(
            ["--config", str(tmp_path / "config.toml"), "truth-qc", "--days", "30"]
        )

        artifact = json.loads(
            (config.artifacts_dir / "truth_qc.json").read_text(encoding="utf-8")
        )
        assert code == 2
        assert artifact["schema_version"] == 2
        assert artifact["drift_alert"] is None
        assert artifact["correlation_alert"] is None
        assert artifact["shield_alert"] is None
        assert "overlap: 0 hours" in capsys.readouterr().out


def test_predict_reports_unsupported_method_without_traceback(
    tmp_path, capsys, monkeypatch
):
    write_config(tmp_path)

    def unsupported(*_args, **_kwargs):
        raise UnsupportedMethodError(
            "method 'persistence' does not support daily.temp_max_c"
        )

    monkeypatch.setattr(
        "grounded_weather_forecast.serve.predict.predict",
        unsupported,
    )

    code = main(
        [
            "--config",
            str(tmp_path / "config.toml"),
            "predict",
            "--method",
            "persistence",
            "--no-history",
        ]
    )

    assert code == 1
    assert "cannot predict: method 'persistence'" in capsys.readouterr().out


def test_ensemble_store_failure_is_an_actionable_cli_error(
    tmp_path, capsys, monkeypatch
):
    write_config(tmp_path)
    monkeypatch.setattr(
        "grounded_weather_forecast.dataset.ensembles.ingest_ensembles",
        lambda _config: pl.DataFrame({"model": ["gefs"]}),
    )

    def fail_append(_path, _fresh):
        raise OSError("disk full")

    monkeypatch.setattr(
        "grounded_weather_forecast.dataset.ensembles.append_ensembles",
        fail_append,
    )

    code = main(["--config", str(tmp_path / "config.toml"), "ingest-ensembles"])

    assert code == 1
    assert "ensemble ingest failed: disk full" in capsys.readouterr().out
