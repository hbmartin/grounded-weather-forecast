"""Malformed run rows affect only relevant commands with unresolved chronology."""

from datetime import timedelta

import polars as pl
import pytest
from test_daily_read import NOW, START

from grounded_weather_forecast import daily_read as reader
from grounded_weather_forecast.health_policy import _latest_attempt
from grounded_weather_forecast.run_health import aware_moment, latest_attempt
from grounded_weather_forecast.serve.provenance import (
    selection_code_version,
    selection_truth_semantics,
)


def run(**changes):
    return {
        "command": "predict",
        "started_at": START.isoformat(),
        "ended_at": NOW.isoformat(),
        "exit_code": 0,
        "failure_kind": None,
        "record_kind": "invocation",
    } | changes


def read_rows(tmp_path, rows):
    path = tmp_path / "runs.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return reader.read_runs(reader.Snapshot(), path, NOW)


def test_record_kind_is_filtered_before_field_validation(tmp_path):
    result = read_rows(tmp_path, [run(), run(record_kind="unrelated", failure_kind=42)])
    assert result["status"] == "ok" and result["limitations"] == []
    assert result["latest_by_command"][0]["status"] == "ok"


def test_superseded_damage_is_information_without_degrading_health(tmp_path):
    bad = run(started_at="bad", ended_at=(START - timedelta(days=80)).isoformat())
    result = read_rows(tmp_path, [run(), bad])
    assert result["status"] == "ok" and result["limitations"] == [
        "superseded_invalid_run_rows"
    ]


@pytest.mark.parametrize(
    "clock", [None, "bad", (NOW + timedelta(seconds=0)).isoformat()]
)
def test_unorderable_or_equally_recent_damage_preserves_unknown_command(
    tmp_path, clock
):
    result = read_rows(
        tmp_path, [run(), run(failure_kind=42, ended_at=clock), run(command="publish")]
    )
    assert {r["command"]: r["status"] for r in result["latest_by_command"]} == {
        "predict": "unknown",
        "publish": "ok",
    }
    assert result["status"] == "partial" and "invalid_run_rows" in result["limitations"]


def test_otherwise_valid_incomplete_rows_are_skipped(tmp_path):
    result = read_rows(tmp_path, [run(), run(started_at=None, ended_at=None)])
    assert result["status"] == "ok"


@pytest.mark.parametrize(
    "clock", [None, "bad", "2026-01-01T00:00:00", "0001-01-01T00:00:00+01:00"]
)
def test_shared_clocks_fail_closed(clock):
    assert aware_moment(clock) is None


def test_shared_truth_qc_selection_ignores_invalid_or_unrelated_attempts():
    good = run(command="truth-qc", started_at=START, ended_at=NOW)
    rows = [
        good,
        good | {"record_kind": "other"},
        good | {"started_at": NOW + timedelta(hours=1)},
    ]
    assert (
        _latest_attempt(pl.DataFrame(rows), NOW)
        == latest_attempt(pl.DataFrame(rows), "truth-qc", NOW)
        == good
    )


@pytest.mark.parametrize("context_id", ["unknown", ""])
def test_legacy_invalid_context_ids_remain_rejected(context_id):
    release = {
        "evaluation_contexts": [
            {
                "evaluation_id": context_id,
                "code_version": "code",
                "semantics": {"temp_c": "inst"},
            }
        ]
    }
    assert selection_code_version(release, {}) is None
    assert selection_truth_semantics(release, {}, "temp_c") is None
    assert selection_code_version(release, {"code_version": "direct"}) == "direct"
