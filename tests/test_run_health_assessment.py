"""Uncertain run evidence must not turn an older outcome into current success."""

from datetime import UTC, datetime, timedelta, tzinfo

import polars as pl
import pytest
from test_daily_read import NOW, START
from test_retention_verification import run_row

from grounded_weather_forecast import daily_read as reader
from grounded_weather_forecast.clocks import aware_utc
from grounded_weather_forecast.health_policy import truth_qc_freshness
from grounded_weather_forecast.retention_status import (
    RetentionVerification,
    retention_status,
)
from grounded_weather_forecast.run_health import assess_attempts, latest_attempt
from grounded_weather_forecast.serve.approvals import aware_timestamp
from grounded_weather_forecast.status import failure_details_json


class NoOffset(tzinfo):
    def utcoffset(self, dt):
        return None

    def dst(self, dt):
        return None


def test_offsetless_zone_never_uses_local_time():
    moment = datetime(2026, 7, 10, 12, tzinfo=NoOffset())
    assert aware_utc(moment) is None
    with pytest.raises(reader.SourceError, match="timezone_required"):
        reader.timestamp(moment)
    assert aware_timestamp(moment) is None  # string-only approval compatibility API
    assert aware_timestamp(moment.isoformat()) is None


@pytest.mark.parametrize(
    "value", [None, 42, "bad", "2026-10-22T14:00:00", "0001-01-01T00:00:00+01:00"]
)
def test_clock_adapters_reject_invalid_input(value):
    assert aware_utc(value) is None and aware_timestamp(value) is None


@pytest.mark.parametrize(
    "damage",
    [
        {"started_at": None},
        {"started_at": NOW + timedelta(hours=1)},
        {"failure_kind": 42},
        {"ended_at": "bad"},
    ],
)
def test_newer_invalid_attempt_preserves_check_and_shows_unknown(damage):
    older = run_row("clear", when=START)
    newer = run_row() | {"exit_code": 1, "failure_details_json": None} | damage
    if damage.get("ended_at") == "bad":
        older = older | {
            "started_at": older["started_at"].isoformat(),
            "ended_at": older["ended_at"].isoformat(),
        }
        newer = newer | {"started_at": newer["started_at"].isoformat()}
    frame = pl.DataFrame([older, newer], strict=False)
    status = retention_status(frame, as_of=NOW)
    assert status.verification.state == "clear"
    assert status.attempt_state == "unknown" and status.severity == "amber"
    assert latest_attempt(frame, "prune-scores", NOW) is None


@pytest.mark.parametrize("exit_code", [0, 1])
def test_exact_clock_ties_use_last_persisted_row(exit_code):
    rows = [
        run_row() | {"exit_code": 1 - exit_code},
        run_row() | {"exit_code": exit_code},
    ]
    frame = pl.DataFrame(rows)
    assert latest_attempt(frame, "prune-scores", NOW)["exit_code"] == exit_code
    assert retention_status(frame, as_of=NOW).attempt_state == (
        "failed" if exit_code else "completed"
    )


def test_contention_neither_creates_nor_clears_warning():
    clear = run_row("clear", when=START)
    wait = run_row() | {"exit_code": 75, "failure_details_json": None}
    status = retention_status(pl.DataFrame([clear, wait]), as_of=NOW)
    assert status.attempt_state == "contended" and status.severity == "info"
    failed = run_row(when=START + timedelta(seconds=1)) | {"exit_code": 1}
    status = retention_status(pl.DataFrame([clear, failed, wait]), as_of=NOW)
    assert (
        status.attempt_state == "contended"
        and status.outstanding_failure
        and status.severity == "amber"
    )
    clean = run_row("clear", when=NOW + timedelta(seconds=1))
    assert (
        retention_status(
            pl.DataFrame([clear, failed, wait, clean]), as_of=NOW + timedelta(seconds=2)
        ).severity
        == "ok"
    )


def test_completed_batches_in_one_row_select_latest_clock_and_id():
    batches = (
        RetentionVerification("suspended", START, check_id="first").diagnostics()
        + RetentionVerification(
            "clear", NOW - timedelta(seconds=1), check_id="second"
        ).diagnostics()
    )
    status = retention_status(
        pl.DataFrame(
            [run_row() | {"failure_details_json": failure_details_json(batches)}]
        ),
        as_of=NOW,
    )
    assert (
        status.verification.state == "clear"
        and status.verification.check_id == "second"
    )
    same_clock = (
        RetentionVerification("suspended", START, check_id="first").diagnostics()
        + RetentionVerification("clear", START, check_id="second").diagnostics()
    )
    status = retention_status(
        pl.DataFrame(
            [run_row() | {"failure_details_json": failure_details_json(same_clock)}]
        ),
        as_of=NOW,
    )
    assert status.verification.check_id == "second"


def test_legacy_distinct_clocks_and_report_age():
    items = (
        RetentionVerification("suspended", START).diagnostics()
        + RetentionVerification("clear", NOW - timedelta(minutes=1)).diagnostics()
    )
    status = retention_status(
        pl.DataFrame(
            [run_row() | {"failure_details_json": failure_details_json(items)}]
        ),
        as_of=NOW,
    )
    assert status.verification.state == "clear"
    assert status.frame()["check_age_minutes"][0] == 1


def test_superseded_damage_and_ineligible_future_rows():
    good = run_row()
    old = run_row(when=START) | {"started_at": None}
    future = run_row(when=NOW + timedelta(days=1)) | {"failure_kind": 42}
    assessment = assess_attempts(
        pl.DataFrame([good, old, future], strict=False), "prune-scores", NOW
    )
    assert assessment.superseded and not assessment.unresolved
    with pytest.raises(ValueError, match="timezone-aware"):
        assess_attempts(pl.DataFrame(), "prune-scores", NOW.replace(tzinfo=None))


def test_truth_qc_unknown_run_is_warning_without_inventing_failed_refresh():
    rows = [run_row() | {"command": "truth-qc", "started_at": None, "exit_code": 1}]
    result = truth_qc_freshness(
        {
            "generated_at": START.isoformat(),
            "drift_verdict": {"as_of_date": START.date().isoformat()},
        },
        pl.DataFrame(rows),
        NOW,
    )
    assert result.attempt_unknown and result.failed_attempt is None
    assert any("outcome is unknown" in warning for warning in result.warnings)
