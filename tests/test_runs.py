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

TEST_NOW = datetime(2026, 7, 19, tzinfo=UTC)


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
    append_run(_record(), path, now=TEST_NOW)
    frame = load_runs(path)
    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["command"] == "qc"
    assert row["exit_code"] == 0
    assert row["error"] is None
    assert row["duration_ms"] == 1500


def test_append_is_additive_and_ordered(tmp_path):
    path = tmp_path / "runs.parquet"
    append_run(_record(command="qc"), path, now=TEST_NOW)
    append_run(_record(command="report", offset_s=60), path, now=TEST_NOW)
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


def test_load_corrupt_parquet_returns_empty_schema_frame(tmp_path):
    path = tmp_path / "runs.parquet"
    path.write_bytes(b"not a parquet file")

    frame = load_runs(path)

    assert frame.is_empty()
    assert frame.schema == RUNS_SCHEMA


def test_load_normalization_failure_returns_empty_schema_frame(tmp_path, monkeypatch):
    class BrokenFrame:
        columns = RUNS_SCHEMA.names()

        @staticmethod
        def with_columns(*_expressions):
            raise pl.exceptions.ComputeError("corrupt schema")

    path = tmp_path / "runs.parquet"
    path.touch()
    monkeypatch.setattr(runs_module.pl, "read_parquet", lambda _path: BrokenFrame())

    frame = load_runs(path)

    assert frame.is_empty()
    assert frame.schema == RUNS_SCHEMA


def test_write_failure_is_swallowed(tmp_path, monkeypatch):
    def boom(frame, path):
        raise OSError("disk full")

    monkeypatch.setattr(runs_module, "atomic_write_parquet", boom)
    append_run(_record(), tmp_path / "runs.parquet", now=TEST_NOW)
    assert not (tmp_path / "runs.parquet").exists()


def test_lock_timeout_is_swallowed(tmp_path, monkeypatch):
    monkeypatch.setattr(runs_module, "_LOCK_TIMEOUT_SECONDS", 0.05)
    path = tmp_path / "runs.parquet"
    with FileLock(path.with_suffix(".parquet.lock")):
        append_run(_record(), path, now=TEST_NOW)
    assert not path.exists()


class TestRetention:
    """The ledger is rewritten in full on every append, so it must stay bounded."""

    def test_rows_past_the_retention_horizon_are_dropped(self):
        from grounded_weather_forecast.runs import RUNS_SCHEMA, prune_runs

        now = datetime(2026, 7, 19, tzinfo=UTC)
        frame = pl.DataFrame(
            {
                "run_id": ["old", "recent"],
                "command": ["predict", "predict"],
                "args_json": ["{}", "{}"],
                "started_at": [now - timedelta(days=200), now - timedelta(days=1)],
                "ended_at": [now - timedelta(days=200), now - timedelta(days=1)],
                "duration_ms": [1, 1],
                "exit_code": [0, 0],
                "error": [None, None],
                "dataset_fingerprint": ["f", "f"],
                "config_fingerprint": ["c", "c"],
                "code_version": ["0.4.0", "0.4.0"],
            },
            schema=RUNS_SCHEMA,
        )
        assert prune_runs(frame, now=now)["run_id"].to_list() == ["recent"]

    def test_appending_does_not_grow_without_bound(self, tmp_path):
        from grounded_weather_forecast import runs

        path = tmp_path / "runs.parquet"
        base = datetime(2026, 7, 19, tzinfo=UTC)
        for index in range(5):
            stamp = base - timedelta(days=365 if index < 3 else 0)
            runs.append_run(
                runs.RunRecord(
                    run_id=f"r{index}",
                    command="predict",
                    args_json="{}",
                    started_at=stamp,
                    ended_at=stamp,
                    exit_code=0,
                    error=None,
                    dataset_fingerprint="f",
                    config_fingerprint="c",
                    code_version="0.4.0",
                ),
                path,
                now=base,
            )
        # The three year-old rows are pruned; only the two recent ones survive.
        assert runs.load_runs(path)["run_id"].to_list() == ["r3", "r4"]


def test_prune_keeps_rows_whose_started_at_is_null():
    """`load_runs` promises older schemas load null-filled; pruning must agree."""
    from datetime import UTC, datetime

    import polars as pl

    from grounded_weather_forecast.runs import RUNS_SCHEMA, prune_runs

    now = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)
    frame = pl.DataFrame({column: [None] for column in RUNS_SCHEMA}, schema=RUNS_SCHEMA)
    assert prune_runs(frame, now=now).height == 1


def test_a_naive_timestamp_still_writes_a_row(tmp_path):
    """Regression: a naive started_at silently wrote nothing at all.

    `_to_frame` reinterpreted it as UTC, `prune_runs` then raised SchemaError
    comparing the tz-aware column to a tz-naive literal, and `append_run`
    swallowed it -- so the ledger never appeared and nothing said why.
    """
    from datetime import datetime

    from grounded_weather_forecast.runs import RunRecord, append_run, load_runs

    naive = datetime(2026, 7, 19, 12, 0)
    path = tmp_path / "runs.parquet"
    append_run(
        RunRecord(
            run_id="r1",
            command="predict",
            args_json="{}",
            started_at=naive,
            ended_at=naive,
            exit_code=0,
            error=None,
            dataset_fingerprint="fp",
            config_fingerprint="cfg",
            code_version="0.4.0",
        ),
        path,
        now=datetime(2026, 7, 19, 12, 0, tzinfo=UTC),
    )

    assert path.exists(), "a naive timestamp must not silently drop the row"
    assert load_runs(path).height == 1


def test_mixed_timestamp_awareness_is_normalized_before_duration(tmp_path):
    path = tmp_path / "runs.parquet"
    started = datetime(2026, 7, 19, 12, 0)
    ended = datetime(2026, 7, 19, 12, 0, 1, 500_000, tzinfo=UTC)

    append_run(
        RunRecord(
            run_id="mixed",
            command="predict",
            args_json="{}",
            started_at=started,
            ended_at=ended,
            exit_code=0,
            error=None,
            dataset_fingerprint="fp",
            config_fingerprint="cfg",
            code_version="0.4.0",
        ),
        path,
        now=ended,
    )

    row = load_runs(path).row(0, named=True)
    assert row["started_at"] == started.replace(tzinfo=UTC)
    assert row["ended_at"] == ended
    assert row["duration_ms"] == 1500


def test_future_timestamp_does_not_move_the_retention_horizon(tmp_path):
    path = tmp_path / "runs.parquet"
    recent = _record(command="recent")
    future = RunRecord(
        run_id="future",
        command="predict",
        args_json="{}",
        started_at=TEST_NOW + timedelta(days=365),
        ended_at=TEST_NOW + timedelta(days=365),
        exit_code=0,
        error=None,
        dataset_fingerprint="f",
        config_fingerprint="c",
        code_version="0.4.0",
    )

    append_run(recent, path, now=TEST_NOW)
    append_run(future, path, now=TEST_NOW)

    assert load_runs(path)["run_id"].to_list() == [recent.run_id, "future"]
