"""Daily read contract: synthetic files only, no collectors or live APIs."""

import json
import sqlite3
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from conftest import make_station_db, write_config

from grounded_weather_forecast import cli
from grounded_weather_forecast import daily_read as reader
from grounded_weather_forecast.contracts import HOURLY_VARIABLES
from grounded_weather_forecast.evaluation import config_fingerprint
from grounded_weather_forecast.reports.evidence import (
    QUALITY_SCHEMA,
    SERVED_QUALITY_SCHEMA,
)

START = datetime(2026, 10, 22, 14, tzinfo=UTC)
NOW = START + timedelta(minutes=20)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=reader.json_default))


def document(start=START):
    values = {s.name: 0.5 for s in HOURLY_VARIABLES}
    return {
        "schema_version": 6,
        "issued_at": start - timedelta(minutes=10),
        "observation_at": start - timedelta(minutes=1),
        "dataset_fingerprint": "dataset1",
        "status": "ready",
        "release_ids": ["release1"],
        "hourly": [
            {
                "valid_time": start + timedelta(hours=i),
                "lead_bucket": "1-6h",
                "values": values.copy(),
                "methods": dict.fromkeys(values, "equal_weight"),
                "release_ids": dict.fromkeys(values, "release1"),
                "truth_semantics": dict.fromkeys(values, "inst"),
                "selection_reasons": {},
            }
            for i in range(13)
        ],
    }


def setup(tmp_path, start=START):
    config = write_config(tmp_path)
    forecast = tmp_path / "forecast.json"
    raw = document(start)
    write_json(forecast, raw)
    write_json(config.dataset.dir / "manifest.json", {"fingerprint": "dataset1"})
    configuration = config_fingerprint(config)
    selections = {
        f"hourly.{v.name}.1-6h": {
            "method_id": "equal_weight",
            "truth_semantics": "inst",
            "evaluation_id": "eval1",
        }
        for v in HOURLY_VARIABLES
    }
    release = {
        "release_id": "release1",
        "dataset_fingerprint": "dataset1",
        "config_fingerprint": configuration,
        "selections": selections,
    }
    write_json(config.artifacts_dir / "releases/release1.json", release)
    return config, forecast, raw


def read(config, forecast, **kwargs):
    return reader.read_daily_report(
        config,
        forecast_path=forecast,
        start=START,
        as_of=NOW,
        published_station=forecast.parent / "published.json",
        **kwargs,
    )


def quality(config, *, realized=False, **updates):
    row = {
        "recorded_at": START,
        "evaluation_id": "eval1",
        "evaluation_created_at": START - timedelta(hours=1),
        "as_of_date": START.date(),
        "product": "hourly",
        "source_kind": "live",
        "variable": "temp_c",
        "truth_semantics": "inst",
        "lead_bucket": "1-6h",
        "method_id": "equal_weight",
        "release_id": "release1",
        "n": 100,
        "mae": 1.2,
        "live_mae": 2.0,
        "dataset_fingerprint": "dataset1",
        "config_fingerprint": config_fingerprint(config),
    }
    row.update(updates)
    schema = SERVED_QUALITY_SCHEMA if realized else QUALITY_SCHEMA
    return {key: row.get(key) for key in schema}


def write_quality(config, rows, *, realized=False):
    name = "served_quality" if realized else "quality"
    schema = SERVED_QUALITY_SCHEMA if realized else QUALITY_SCHEMA
    path = config.artifacts_dir / "history" / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=schema).write_parquet(path)


def provider_db(path):
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE forecast_runs(id INTEGER PRIMARY KEY, completed_at TEXT);
        CREATE TABLE provider_results(id INTEGER PRIMARY KEY, run_id INTEGER, provider TEXT,
          status TEXT, fetched_at TEXT, raw_json TEXT, error_message TEXT);
    """)
    connection.executemany(
        "INSERT INTO forecast_runs VALUES (?, ?)",
        [
            (1, (START - timedelta(hours=25)).isoformat()),
            (2, START.isoformat()),
            (3, (NOW + timedelta(hours=1)).isoformat()),
        ],
    )
    connection.executemany(
        "INSERT INTO provider_results VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 1, "old", "error", None, "credential=SECRET", "token=SECRET"),
            (2, 2, "provider", "success", START.isoformat(), "credential=SECRET", None),
            (3, 2, "provider", "error", None, "credential=SECRET", "token=SECRET"),
            (4, 3, "future", "error", None, "SECRET", "SECRET"),
        ],
    )
    connection.commit()
    connection.close()


def test_daily_window_is_fixed_and_end_exclusive(tmp_path):
    config, forecast, _ = setup(tmp_path)
    result = read(config, forecast)
    assert result["schema_version"] == 1
    assert result["window"]["local_start"] == "2026-10-22T07:00:00-07:00"
    assert result["window"]["local_end"] == "2026-10-22T19:00:00-07:00"
    assert result["forecast"]["coverage"]["unique_hours"] == 12
    assert result["forecast"]["hourly"][-1]["local_time"] == "2026-10-22T18:00:00-07:00"
    assert result["forecast"]["units"]["pop"] == "1 (probability)"
    assert result["forecast"]["issued_at"] == "2026-10-22T13:50:00+00:00"
    assert result["forecast"]["observation_freshness"]["status"] == "fresh"


@pytest.mark.parametrize(
    ("day", "offset"), [("2026-03-08", "-07:00"), ("2026-11-01", "-08:00")]
)
def test_dst_daily_seven_to_nineteen(tmp_path, day, offset):
    start = datetime.fromisoformat(f"{day}T07:00:00{offset}")
    config, forecast, _ = setup(tmp_path, start)
    result = reader.read_daily_report(
        config,
        forecast_path=forecast,
        start=start,
        as_of=start + timedelta(minutes=40),
        published_station=tmp_path / "missing",
    )
    assert result["window"]["local_end"] == f"{day}T19:00:00{offset}"
    assert len(result["forecast"]["hourly"]) == 12


@pytest.mark.parametrize(
    "start", ["2026-03-08T00:00:00-08:00", "2026-11-01T00:00:00-07:00"]
)
def test_dst_crossing_counts_absolute_hours(tmp_path, start):
    start = datetime.fromisoformat(start)
    config, forecast, _ = setup(tmp_path, start)
    result = reader.read_daily_report(
        config,
        forecast_path=forecast,
        start=start,
        as_of=start,
        published_station=tmp_path / "missing",
    )
    rows = result["forecast"]["hourly"]
    assert len({r["valid_time"] for r in rows}) == 12
    assert len({r["local_time"] for r in rows}) == 12
    assert (
        result["window"]["end"]
        == (reader.timestamp(start) + timedelta(hours=12)).isoformat()
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": START.replace(tzinfo=None)},
        {"start": START + timedelta(minutes=1)},
        {"as_of": START - timedelta(seconds=1)},
        {"hours": 0},
        {"hours": 13},
    ],
)
def test_invalid_time_ranges(tmp_path, kwargs):
    config, forecast, _ = setup(tmp_path)
    args = {"forecast_path": forecast, "start": START, "as_of": NOW, **kwargs}
    with pytest.raises(ValueError, match=r"timezone_required|start must|as-of must"):
        reader.read_daily_report(config, **args)


def test_partial_coverage_nulls_duplicates_and_invalid_values(tmp_path):
    config, forecast, raw = setup(tmp_path)
    raw["hourly"].pop(1)
    raw["hourly"].append(deepcopy(raw["hourly"][0]))
    raw["hourly"][1]["values"]["temp_c"] = None
    del raw["hourly"][1]["values"]["humidity_pct"]
    raw["hourly"][1]["values"]["pop"] = float("nan")
    raw["hourly"][2]["values"]["temp_c"] = True
    write_json(forecast, raw)
    result = read(config, forecast)["forecast"]
    assert result["status"] == "partial"
    assert result["coverage"]["unique_hours"] == 10
    assert len(result["coverage"]["duplicate_hours"]) == 1
    assert result["hourly"][0]["values"]["temp_c"] is None
    assert result["hourly"][2]["null_variables"] == ["temp_c"]
    assert result["hourly"][2]["missing_variables"] == ["humidity_pct"]
    assert result["hourly"][2]["invalid_variables"] == ["pop"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"schema_version": 7}, "unsupported_source_schema"),
        ({"schema_version": True}, "unsupported_source_schema"),
        ({"issued_at": NOW + timedelta(minutes=1)}, "forecast_issued_after_as_of"),
        ({"hourly": None}, "missing_hourly_array"),
        ({"release_ids": None}, "invalid_release_ids"),
    ],
)
def test_forecast_schema_and_future_sources(tmp_path, change, code):
    config, forecast, raw = setup(tmp_path)
    raw.update(change)
    write_json(forecast, raw)
    result = read(config, forecast)
    assert result["forecast"]["limitations"] == [code]
    assert "operational_health" in result


def test_missing_corrupt_and_stale_forecast(tmp_path):
    config, forecast, raw = setup(tmp_path)
    raw["issued_at"] = NOW - timedelta(hours=4)
    raw["observation_at"] = None
    write_json(forecast, raw)
    result = read(config, forecast)["forecast"]
    assert result["freshness"]["status"] == "stale"
    assert result["observation_freshness"]["status"] == "unknown"
    forecast.write_text("invalid")
    assert read(config, forecast)["forecast"]["status"] == "unknown"
    forecast.unlink()
    assert read(config, forecast)["forecast"]["limitations"] == ["missing_source"]


def test_rebuild_without_releases_cannot_reuse_historical_quality(tmp_path):
    config, forecast, raw = setup(tmp_path)
    write_quality(config, [quality(config)])
    raw.update(
        status="degraded",
        status_reason="rebuild invalidates promoted evidence",
        release_ids=[],
    )
    write_json(forecast, raw)
    result = read(config, forecast)
    assert result["forecast"]["forecast_status"] == "degraded"
    for key in ("backtest", "realized_live"):
        assert result["prediction_quality"][key]["limitations"] == [
            "forecast_has_no_release_ids"
        ]


@pytest.mark.parametrize("realized", [False, True])
def test_scores_match_current_dataset_release_slice_and_latest_cohort(
    tmp_path, realized
):
    config, forecast, _ = setup(tmp_path)
    matching = quality(config, realized=realized)
    wrong = [
        quality(config, realized=realized, **changes)
        for changes in [
            {"dataset_fingerprint": "old"},
            {"config_fingerprint": "old"},
            {"product": "minutely"},
            {"method_id": "other"},
            {"source_kind": "synthetic", "release_id": "old", "evaluation_id": "old"},
            {"recorded_at": NOW + timedelta(days=1)},
        ]
    ]
    old = quality(
        config,
        realized=realized,
        recorded_at=NOW,
        as_of_date=(START - timedelta(days=1)).date(),
        evaluation_created_at=START - timedelta(days=3),
        mae=100,
        live_mae=100,
    )
    write_quality(config, [matching, old, *wrong], realized=realized)
    result = read(config, forecast)["prediction_quality"][
        "realized_live" if realized else "backtest"
    ]
    score = next(r for r in result["slices"] if r["variable"] == "temp_c")
    assert result["kind"] == ("realized_live" if realized else "backtest")
    assert score["n"] == 100
    assert score["mae"] == (2 if realized else 1.2)
    assert score["status"] == "available"
    assert (
        next(r for r in result["slices"] if r["variable"] == "pop")["status"]
        == "unknown"
    )


@pytest.mark.parametrize(
    ("changes", "status"),
    [
        ({"n": 0}, "insufficient"),
        ({"mae": None}, "insufficient"),
        ({"evaluation_created_at": START - timedelta(days=3)}, "stale"),
        ({"dataset_fingerprint": "old"}, "unknown"),
        ({"evaluation_id": "old"}, "unknown"),
    ],
)
def test_unknown_stale_and_insufficient_quality(tmp_path, changes, status):
    config, forecast, _ = setup(tmp_path)
    write_quality(config, [quality(config, **changes)])
    scores = read(config, forecast)["prediction_quality"]["backtest"]["slices"]
    assert next(r for r in scores if r["variable"] == "temp_c")["status"] == status


@pytest.mark.parametrize(
    "target", ["dataset", "release", "selection", "slice_release", "release_file"]
)
def test_identity_gaps_are_explicit(tmp_path, target):
    config, forecast, raw = setup(tmp_path)
    write_quality(config, [quality(config)])
    release_path = config.artifacts_dir / "releases/release1.json"
    if target == "dataset":
        write_json(config.dataset.dir / "manifest.json", {"fingerprint": "new"})
    elif target == "release_file":
        release_path.unlink()
    elif target == "slice_release":
        raw["hourly"][0]["release_ids"] = {}
        write_json(forecast, raw)
    else:
        release = json.loads(release_path.read_text())
        if target == "release":
            release["dataset_fingerprint"] = "old"
        else:
            release["selections"] = {}
        write_json(release_path, release)
    result = read(config, forecast)["prediction_quality"]["backtest"]
    assert result["status"] in ("partial", "unknown")
    if target == "dataset":
        assert result["limitations"] == ["forecast_current_dataset_mismatch"]
    else:
        assert any(r["status"] == "unknown" for r in result["slices"])


def test_provider_errors_use_parent_completion_and_exclude_secrets(tmp_path):
    config, forecast, _ = setup(tmp_path)
    provider_db(config.forecasts.db_path)
    result = read(config, forecast)["operational_health"]["providers"]
    assert len(result["providers"]) == 1
    row = result["providers"][0]
    assert row["attempts"] == 2
    assert row["failures"] == 1
    assert row["latest_status"] == "error"
    assert row["success_rate"] == 0.5
    assert row["completed_at"] == START.isoformat()
    assert "SECRET" not in json.dumps(result)


def test_station_fallback_is_labeled_and_never_immutable(tmp_path, monkeypatch):
    config, forecast, _ = setup(tmp_path)
    write_json(
        tmp_path / "published.json",
        {"metadata": {"observed_at": START}, "data": {"secret": "SECRET"}},
    )
    original = sqlite3.connect
    uris = []

    def connect(database, **kwargs):
        uris.append(database)
        return original(database, **kwargs)

    monkeypatch.setattr(reader.sqlite3, "connect", connect)
    result = read(config, forecast)["operational_health"]["station"]
    assert result.get("source") == "published_station_snapshot", result
    assert result["limitations"] == ["live_station_unavailable"]
    assert result["freshness"]["status"] == "fresh"
    assert all("mode=ro" in uri and "immutable" not in uri for uri in uris)
    assert "SECRET" not in json.dumps(result)


def test_live_wal_reads_uncheckpointed_observations(tmp_path):
    config, forecast, _ = setup(tmp_path)
    connection = sqlite3.connect(config.station.db_path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE observations(ts TEXT)")
        connection.execute(
            "INSERT INTO observations VALUES (?)",
            (START.replace(tzinfo=None).isoformat(),),
        )
        connection.commit()
        result = read(config, forecast)["operational_health"]["station"]
        assert result["source"] == "live_station_sqlite"
        assert result["freshness"]["at"] == START.isoformat()
    finally:
        connection.close()
    wal = config.station.db_path.with_name(config.station.db_path.name + "-wal")
    wal.touch()
    with pytest.raises(reader.SourceError, match="wal_shared_memory_unavailable"):
        reader._open_readonly(config.station.db_path)


def test_historical_alarms_and_latest_completed_runs(tmp_path):
    config, forecast, _ = setup(tmp_path)
    history = config.artifacts_dir / "history"
    history.mkdir()
    pl.DataFrame(
        {"recorded_at": [START - timedelta(days=5)], "alarms": ["old failure"]}
    ).write_parquet(history / "pipeline.parquet")
    write_json(
        config.artifacts_dir / "drift.json",
        {"schema_version": 2, "alarms": ["old alarm"]},
    )
    write_json(
        config.artifacts_dir / "truth_qc.json",
        {
            "schema_version": 3,
            "drift_verdict": {"as_of_date": "2026-10-21"},
            "drift_alert": None,
        },
    )
    pl.DataFrame(
        [
            {
                "command": "predict",
                "started_at": START - timedelta(hours=1),
                "ended_at": START - timedelta(minutes=59),
                "exit_code": 1,
                "error": "Error",
                "failure_kind": None,
            },
            {
                "command": "predict",
                "started_at": START,
                "ended_at": NOW,
                "exit_code": 0,
                "error": None,
                "failure_kind": None,
            },
            {
                "command": "backtest",
                "started_at": START,
                "ended_at": NOW,
                "exit_code": 3,
                "error": None,
                "failure_kind": "no_folds",
            },
        ]
    ).write_parquet(config.dataset.dir / "runs.parquet")
    result = read(config, forecast)
    assert result["operational_health"]["pipeline"]["status"] == "historical"
    assert result["operational_health"]["pipeline"]["freshness"]["status"] == "stale"
    assert result["prediction_quality"]["drift"]["current_alarm_count"] is None
    assert result["prediction_quality"]["truth_qc"]["current_alerts"] is None
    assert [
        r["status"] for r in result["operational_health"]["runs"]["latest_by_command"]
    ] == ["ok", "pending"]


def test_concurrent_file_replacement_retries_and_discards_mixed_result(
    tmp_path, monkeypatch
):
    config, forecast, raw = setup(tmp_path)
    original = reader.Snapshot.changed
    calls = []

    def changed(snapshot):
        calls.append(1)
        if len(calls) == 1:
            write_json(forecast, raw | {"status": "degraded"})
        return original(snapshot)

    monkeypatch.setattr(reader.Snapshot, "changed", changed)
    assert read(config, forecast)["forecast"]["forecast_status"] == "degraded"
    assert len(calls) == 2
    monkeypatch.setattr(reader.Snapshot, "changed", lambda _: True)
    result = read(config, forecast)
    assert "window" in result
    assert "prediction_quality" in result
    assert result["forecast"]["limitations"] == ["concurrent_replacement_after_retry"]
    assert "hourly" not in result["forecast"]


def test_source_schema_changes_are_partial(tmp_path):
    config, forecast, _ = setup(tmp_path)
    path = config.artifacts_dir / "history/pipeline.parquet"
    path.parent.mkdir()
    pl.DataFrame({"renamed": [1]}).write_parquet(path)
    write_json(config.artifacts_dir / "truth_qc.json", {"schema_version": 99})
    write_json(config.artifacts_dir / "drift.json", {"schema_version": 99})
    result = read(config, forecast)
    assert result["forecast"]["status"] == "ok"
    assert result["operational_health"]["pipeline"]["limitations"] == [
        "missing_required_columns"
    ]
    assert result["prediction_quality"]["truth_qc"]["limitations"] == [
        "unsupported_source_schema"
    ]


def file_state(path):
    return {
        p.relative_to(path): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in path.rglob("*")
        if p.is_file()
    }


def test_cli_has_no_writes_network_processes_or_invocation_record(
    tmp_path, monkeypatch, capsys
):
    config, forecast, _ = setup(tmp_path)
    make_station_db(
        config.station.db_path, [(START.replace(tzinfo=None).isoformat(), {})]
    )
    provider_db(config.forecasts.db_path)
    write_quality(config, [quality(config)])

    def forbidden(*_args, **_kwargs):
        pytest.fail("read command reached a mutating dispatch or invocation recorder")

    monkeypatch.setattr(cli, "_record_run", forbidden)
    monkeypatch.setattr(cli, "_dispatch", forbidden)
    before = file_state(tmp_path)
    args = [
        "--config",
        str(tmp_path / "config.toml"),
        "read-daily-report",
        "--forecast",
        str(forecast),
        "--published-station",
        str(tmp_path / "missing"),
        "--start",
        START.isoformat(),
        "--as-of",
        NOW.isoformat(),
    ]
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema_version"] == 1
    assert before == file_state(tmp_path)
    # A fresh interpreter also catches accidental imports, bytecode, locks,
    # subprocesses and sockets. The audited invocation uses the documented -B.
    script = """
import sys
def audit(event, args):
    if event == "open" and ((isinstance(args[1], str) and any(c in args[1] for c in "wax+")) or (isinstance(args[2], int) and args[2] & 3)):
        raise RuntimeError("write forbidden")
    if event.startswith(("socket.", "subprocess.", "os.mkdir", "os.rename", "os.remove")):
        raise RuntimeError("side effect forbidden")
sys.addaudithook(audit)
from grounded_weather_forecast.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and synthetic arguments
        [sys.executable, "-B", "-c", script, *args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["forecast"]["status"] == "ok"
    assert before == file_state(tmp_path)


def test_cli_errors_are_json_without_recording(tmp_path, capsys, monkeypatch):
    _, forecast, _ = setup(tmp_path)
    monkeypatch.setattr(
        cli, "_record_run", lambda *_args, **_kwargs: pytest.fail("recorded")
    )
    args = [
        "--config",
        str(tmp_path / "config.toml"),
        "read-daily-report",
        "--forecast",
        str(forecast),
        "--start",
        "invalid",
        "--as-of",
        NOW.isoformat(),
    ]
    assert cli.main(args) == 2
    assert json.loads(capsys.readouterr().out)["limitations"] == ["invalid_time_range"]
    (tmp_path / "config.toml").write_text("[station]\n")
    assert cli.main(args) == 2
    assert json.loads(capsys.readouterr().out)["limitations"] == [
        "invalid_configuration"
    ]
