from datetime import UTC, datetime, timedelta

import polars as pl
from filelock import FileLock

from grounded_weather_forecast import runs as runs_module
from grounded_weather_forecast.runs import (
    RUNS_SCHEMA,
    RunRecord,
    append_run,
    load_runs,
    run_id_for,
)


def _record(command="qc", exit_code=0, error=None, offset_s=0):
    started = datetime(2026, 7, 18, 12, 0, tzinfo=UTC) + timedelta(seconds=offset_s)
    ended = started + timedelta(milliseconds=1500)
    return RunRecord(
        run_id=run_id_for(command, started),
        command=command,
        args_json="{}",
        started_at=started,
        ended_at=ended,
        exit_code=exit_code,
        error=error,
        dataset_fingerprint="unknown",
        config_fingerprint="abc123",
        code_version="0.0.0-test",
    )


def test_append_and_load_round_trip(tmp_path):
    path = tmp_path / "runs.parquet"
    append_run(_record(), path)
    frame = load_runs(path)
    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["command"] == "qc"
    assert row["exit_code"] == 0
    assert row["error"] is None
    assert row["duration_ms"] == 1500


def test_append_is_additive_and_ordered(tmp_path):
    path = tmp_path / "runs.parquet"
    append_run(_record(command="qc"), path)
    append_run(_record(command="report", offset_s=60), path)
    assert load_runs(path)["command"].to_list() == ["qc", "report"]


def test_load_missing_file_returns_empty_schema_frame(tmp_path):
    frame = load_runs(tmp_path / "absent.parquet")
    assert frame.is_empty()
    assert frame.schema == RUNS_SCHEMA


def test_load_null_fills_missing_columns(tmp_path):
    path = tmp_path / "runs.parquet"
    pl.DataFrame({"run_id": ["x"], "command": ["qc"]}).write_parquet(path)
    frame = load_runs(path)
    assert frame["exit_code"].to_list() == [None]
    assert frame.schema == RUNS_SCHEMA


def test_write_failure_is_swallowed(tmp_path, monkeypatch):
    def boom(frame, path):
        raise OSError("disk full")

    monkeypatch.setattr(runs_module, "atomic_write_parquet", boom)
    append_run(_record(), tmp_path / "runs.parquet")
    assert not (tmp_path / "runs.parquet").exists()


def test_lock_timeout_is_swallowed(tmp_path, monkeypatch):
    monkeypatch.setattr(runs_module, "_LOCK_TIMEOUT_SECONDS", 0.05)
    path = tmp_path / "runs.parquet"
    with FileLock(path.with_suffix(".parquet.lock")):
        append_run(_record(), path)
    assert not path.exists()
