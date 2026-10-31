"""Trusted command policy and independently visible cleanup evidence."""

import argparse
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import IntEnum, StrEnum
from typing import Literal
from unittest.mock import Mock

import polars as pl
import pytest
from test_daily_read import setup
from test_retention_diagnostic_boundaries import NOW, check, payload, row

from grounded_weather_forecast import cli, runs, status
from grounded_weather_forecast.cleanup_status import assess_cleanup
from grounded_weather_forecast.dashboard.context import collect_context
from grounded_weather_forecast.dashboard.zones.operations import _retention_panel
from grounded_weather_forecast.diagnostic_values import actual_integer, actual_string
from grounded_weather_forecast.retention_audit import AuditCleanupResult
from grounded_weather_forecast.retention_status import (
    RetentionVerification,
    retention_status,
)


class PretendString:
    @property
    def __class__(self):
        return str


class PretendPending:
    def __eq__(self, other):
        return other == "no_folds"


class HostileString(str):
    def __str__(self):
        raise RuntimeError("must not call overridden str")


class HostileInteger(int):
    def __int__(self):
        raise RuntimeError("must not call overridden int")


class Kind(StrEnum):
    PENDING = "no_folds"


class Exit(IntEnum):
    PENDING = 3


def saved(config):
    frame = runs.read_runs(runs.runs_path(config))
    assert frame.height == 1
    return frame.row(0, named=True)


@pytest.mark.parametrize(
    "fake", [Mock(spec=str), PretendString()], ids=["mock", "proxy"]
)
@pytest.mark.parametrize("field", ["stage", "check_id", "progress_id", "detail"])
def test_spoofed_strings_preserve_independent_evidence_and_never_throw(fake, field):
    assert isinstance(fake, str) and actual_string(fake) is None
    valid = check("suspended", name="valid", count=2).diagnostics()
    bad = replace(status.ComponentDiagnostic("other", 0), **{field: fake})
    with status.diagnostic_progress() as progress:
        status.record_progress(valid + (bad,))
    assert tuple(progress)[: len(valid)] == valid
    wire = json.loads(
        status.failure_details_json(status.merge_diagnostics(tuple(progress), (bad,)))
    )
    assert wire[0]["check_id"] == "valid" and wire[0]["issue_count"] == 2
    assert wire[-1]["metadata_invalid"] is True
    assert wire[-1][field]
    with pytest.raises(ValueError, match="identity"):
        status.effective_progress_id(fake, None)


@pytest.mark.parametrize(
    "bad_kind",
    [Mock(spec=str), PretendString(), PretendPending(), [], True],
    ids=["mock", "proxy", "equality", "list", "bool"],
)
def test_invalid_classification_keeps_exit_without_granting_exemption(
    tmp_path, monkeypatch, bad_kind
):
    config, _, _ = setup(tmp_path)
    outcome = status.CommandOutcome.normalize(status.CommandOutcome(3, bad_kind))
    assert outcome.exit_code == 3 and outcome.failure_kind is None
    assert status.CommandOutcome.normalize(outcome) is outcome
    later = []
    result = cli._run_pipeline_steps(
        config,
        argparse.Namespace(command="maintain"),
        (
            ("backtest", lambda: status.CommandOutcome(3, bad_kind)),
            ("report", lambda: later.append(True) or 0),
        ),
        tolerate_pending=True,
        continue_on_failure_kinds=status.PROMOTION_ONLY_FAILURE_KINDS,
    )
    assert result.exit_code == 3 and not later
    record = saved(config)
    assert record["exit_code"] == 3 and record["failure_kind"] is None
    assert "invalid_diagnostic_metadata" in record["failure_details_json"]


@pytest.mark.parametrize(
    "code",
    [0.5, 1.5, "3", True, False, 2**63, -(2**63) - 1, Mock(spec=int)],
    ids=["fraction", "float", "str", "true", "false", "overflow", "underflow", "mock"],
)
def test_invalid_exits_fail_closed_in_cli_hint_and_ledger(tmp_path, monkeypatch, code):
    config, _, _ = setup(tmp_path)
    hint = tmp_path / "outcome.json"
    monkeypatch.setenv(status.COMMAND_OUTCOME_PATH, str(hint))
    monkeypatch.setattr(
        cli, "_dispatch", lambda *args: status.CommandOutcome(code, "no_folds")
    )
    assert cli.main(["--config", str(tmp_path / "config.toml"), "backtest"]) == 1
    record = saved(config)
    assert (
        record["exit_code"] == 1 and record["failure_kind"] == "invalid_command_outcome"
    )
    assert "exit_code" in record["failure_details_json"]
    assert json.loads(hint.read_text()) == {
        "exit_code": 1,
        "failure_kind": "invalid_command_outcome",
    }


@pytest.mark.parametrize(
    "code", [0, 1, 2, 3, 75, -9, -(2**63), 2**63 - 1, Exit.PENDING, HostileInteger(3)]
)
def test_real_integer_outcomes_and_subclasses_remain_exact(code):
    value = status.CommandOutcome.normalize(status.CommandOutcome(code, Kind.PENDING))
    assert type(value.exit_code) is int and value.exit_code == code
    assert type(value.failure_kind) is str and value.failure_kind == "no_folds"
    assert not value.diagnostics
    assert status.CommandOutcome.normalize(value) is value


def test_actual_primitives_bypass_overrides_and_reject_spoofed_int():
    assert actual_string(HostileString("value")) == "value"
    assert actual_integer(HostileInteger(7)) == 7
    assert actual_integer(Mock(spec=int)) is None
    assert actual_integer(True) is None


def test_error_detail_subclass_is_normalized_before_recorder_storage(tmp_path):
    config, _, _ = setup(tmp_path)
    outcome = status.CommandOutcome.normalize(
        status.CommandOutcome(1, error_detail=HostileString("original detail"))
    )
    assert (
        type(outcome.error_detail) is str and outcome.error_detail == "original detail"
    )
    assert status.CommandOutcome.normalize(outcome) is outcome
    cli._record_run(
        config,
        argparse.Namespace(command="backtest"),
        started_at=datetime.now(UTC),
        exit_code=1,
        error_detail=HostileString("original detail"),
    )
    assert saved(config)["error_detail"] == "original detail"


def test_exception_rendered_string_subclass_is_normalized_too(tmp_path):
    class OriginalError(RuntimeError):
        def __str__(self):
            return HostileString("original error")

    config, _, _ = setup(tmp_path)
    cli._record_run(
        config,
        argparse.Namespace(command="backtest"),
        started_at=datetime.now(UTC),
        exit_code=None,
        error=OriginalError(),
    )
    record = saved(config)
    assert (
        record["error"] == "OriginalError"
        and record["error_detail"] == "original error"
    )


@pytest.mark.parametrize("before", [False, True])
def test_invalid_outcome_stops_even_named_continuation_and_prior_promotion_error(
    tmp_path, before
):
    config, _, _ = setup(tmp_path)
    later = []
    stages = (
        (("promotion", lambda: status.CommandOutcome(1, "promotion_evidence_invalid")),)
        if before
        else ()
    )
    outcome = cli._run_pipeline_steps(
        config,
        argparse.Namespace(command="maintain"),
        (
            *stages,
            ("backtest", lambda: status.CommandOutcome(0.5)),
            ("report", lambda: later.append(True) or 0),
        ),
        continue_on_failure=frozenset({"backtest"}),
        continue_on_failure_kinds=status.PROMOTION_ONLY_FAILURE_KINDS,
    )
    assert outcome.exit_code == 1 and outcome.failure_kind == "invalid_command_outcome"
    assert not later


@pytest.mark.parametrize("pipeline", [False, True])
def test_successful_nested_call_ignores_caller_exception(
    tmp_path, monkeypatch, pipeline
):
    config, _, _ = setup(tmp_path)
    monkeypatch.setattr(cli, "_dispatch", lambda *args: status.CommandOutcome(0))
    try:
        raise ValueError("outside this invocation")
    except ValueError:
        if pipeline:
            outcome = cli._run_pipeline_steps(
                config,
                argparse.Namespace(command="maintain"),
                (("backtest", lambda: 0),),
            )
            assert outcome.exit_code == 0
        else:
            assert (
                cli.main(["--config", str(tmp_path / "config.toml"), "backtest"]) == 0
            )
    record = saved(config)
    assert record["exit_code"] == 0 and record["error"] is None


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("pipeline", [False, True])
def test_this_invocation_exception_retains_own_identity(
    tmp_path, monkeypatch, failure, pipeline
):
    config, _, _ = setup(tmp_path)

    def fail(*args):
        raise failure("this invocation failed")

    monkeypatch.setattr(cli, "_dispatch", fail)
    try:
        raise ValueError("caller exception")
    except ValueError:
        with pytest.raises(failure, match="this invocation failed"):
            if pipeline:
                cli._run_pipeline_steps(
                    config,
                    argparse.Namespace(command="maintain"),
                    (("backtest", fail),),
                )
            else:
                cli.main(["--config", str(tmp_path / "config.toml"), "backtest"])
    record = saved(config)
    assert record["exit_code"] is None and record["error"] == failure.__name__


@pytest.mark.parametrize("field", ["failure_kind", "error_detail"])
def test_recorder_optional_projection_keeps_core_and_valid_checks(tmp_path, field):
    config, _, _ = setup(tmp_path)
    cli._record_run(
        config,
        argparse.Namespace(command="backtest"),
        started_at=datetime.now(UTC),
        exit_code=0,
        diagnostics=check(name="preserved").diagnostics(),
        **{field: Mock(spec=str)},
    )
    record = saved(config)
    assert record["exit_code"] == 0 and record[field] is None
    wire = json.loads(record["failure_details_json"])
    assert (
        wire[0]["check_id"] == "preserved"
        and wire[-1]["failure_kind"] == "invalid_diagnostic_metadata"
    )


@dataclass(frozen=True, slots=True)
class ExtendedDiagnostic(status.ComponentDiagnostic):
    number: int = 4
    text: str | None = "extra"


@dataclass(frozen=True, slots=True)
class UnsupportedDiagnostic(ExtendedDiagnostic):
    literal: Literal["x"] = "x"
    sequence: tuple[str, ...] = ("private",)


def test_primitive_extension_schema_keeps_stage_identity_and_subclasses():
    item = ExtendedDiagnostic(
        "retention_verification",
        0,
        "retention_clear",
        check_id="extended",
        issue_count=0,
        number=Exit.PENDING,
        text=HostileString("literal"),
    )
    wire = json.loads(status.failure_details_json((item,)))[0]
    assert (
        wire["check_id"] == "extended"
        and wire["number"] == 3
        and wire["text"] == "literal"
    )
    assert "metadata_invalid" not in wire


def test_unsupported_extensions_only_damage_their_fields():
    item = UnsupportedDiagnostic(
        "retention_verification",
        0,
        "retention_clear",
        check_id="retained",
        issue_count=0,
    )
    wire = json.loads(status.failure_details_json((item,)))[0]
    assert wire["stage"] == "retention_verification" and wire["check_id"] == "retained"
    assert wire["number"] == 4 and wire["literal"]["invalid_field_type"] == "str"
    assert (
        wire["sequence"]["invalid_field_type"] == "tuple"
        and wire["metadata_invalid"] is True
    )


@pytest.mark.parametrize(
    "state",
    ["invalid", "", [], {}, None, Mock(spec=str)],
    ids=["text", "empty", "list", "dict", "null", "mock"],
)
def test_invalid_verification_states_fail_at_construction(state):
    with pytest.raises(ValueError, match="verification state"):
        RetentionVerification(state)


def cleanup_wire(when=NOW, *, complete=True, dry_run=False, **updates):
    result = AuditCleanupResult(
        recorded_at=when,
        dry_run=dry_run,
        completed=complete,
        aborted=not complete,
        deleted_audit_count=0,
        deleted_staging_count=0,
        eligible_audit_count=0,
        eligible_staging_count=0,
    )
    items = json.loads(
        status.failure_details_json(result.diagnostics(check_id="shared"))
    )
    items[0].update(updates)
    return items


@pytest.mark.parametrize(
    "field,value",
    [
        ("deleted_audit_count", "0"),
        ("cleanup_completed", "yes"),
        ("recorded_at", "naive"),
        ("issue_count", -1),
        ("check_id", []),
        ("failure_kind", {}),
    ],
)
def test_cleanup_damage_has_own_amber_status_and_keeps_apply_recovery(field, value):
    initial = row([], when=NOW - timedelta(hours=1), exit_code=1)
    good = payload(check(name="shared"))
    current = row(good + cleanup_wire(**{field: value}))
    result = retention_status(pl.DataFrame([initial, current]), as_of=NOW)
    assert result.cleanup_evidence_unknown and result.severity == "amber"
    assert result.verification.state == "clear" and not result.verification_unknown
    assert not result.outstanding_failure and result.cleanup_outcome == "not_recorded"
    frame = result.frame().row(0, named=True)
    assert frame["cleanup_evidence_unknown"] and frame["cleanup_recorded_at"] is None
    assert "damaged or inconsistent" in frame["cleanup_warning"]


@pytest.mark.parametrize(
    "complete,dry_run", [(True, False), (True, True), (False, False), (False, True)]
)
def test_strictly_later_valid_cleanup_resolves_damage_independently(complete, dry_run):
    earlier = NOW - timedelta(hours=2)
    damaged = row(payload(check(name="shared")) + cleanup_wire(deleted_audit_count="0"))
    prior = row(cleanup_wire(earlier), when=earlier)
    when = NOW + timedelta(minutes=1)
    later = row(
        payload(check(name="later", when=when, dry_run=dry_run))
        + cleanup_wire(when, complete=complete, dry_run=dry_run),
        when=when,
    )
    before = retention_status(pl.DataFrame([prior, damaged, later]), as_of=NOW)
    assert before.cleanup_evidence_unknown and before.cleanup_recorded_at == earlier
    result = retention_status(pl.DataFrame([prior, damaged, later]), as_of=when)
    assert not result.cleanup_evidence_unknown and result.cleanup_recorded_at == when
    assert result.cleanup_outcome == ("completed" if complete else "incomplete")
    assert result.severity == "ok"


def test_equal_clock_and_summaryless_legacy_cleanup_do_not_clear_damage():
    initial = row(cleanup_wire(deleted_audit_count="0"))
    tied = row(cleanup_wire())
    unknown = row(
        [
            {
                "stage": "retention_audit_cleanup",
                "exit_code": 0,
                "detail": "legacy sample",
            }
        ],
        when=NOW + timedelta(minutes=1),
    )
    result = retention_status(
        pl.DataFrame([initial, tied, unknown]), as_of=NOW + timedelta(minutes=1)
    )
    assert result.cleanup_evidence_unknown


def test_legacy_absence_is_not_verified_cleanup():
    result = retention_status(pl.DataFrame([row(payload(check()))]), as_of=NOW)
    assert (
        not result.cleanup_evidence_unknown and result.cleanup_outcome == "not_recorded"
    )
    assert result.cleanup_recorded_at is None and result.severity == "ok"


def test_cleanup_recovery_does_not_clear_execution_failure_or_verification_damage():
    initial = row([], when=NOW - timedelta(hours=1), exit_code=1)
    bad = payload(check(name="bad"))
    bad[0]["issue_count"] = -1
    damaged = row(bad + cleanup_wire(deleted_audit_count="0"))
    when = NOW + timedelta(minutes=1)
    later = row(cleanup_wire(when, dry_run=True), when=when)
    result = retention_status(pl.DataFrame([initial, damaged, later]), as_of=when)
    assert (
        not result.cleanup_evidence_unknown
        and result.verification_unknown
        and result.outstanding_failure
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"cleanup_completed": False},
        {"issue_count": 1},
        {"omitted_count": -1},
        {"recorded_at": (NOW + timedelta(seconds=1)).isoformat()},
        {"dry_run": True, "deleted_audit_count": 1},
    ],
)
def test_cleanup_contradictions_are_not_trusted_summaries(updates):
    assessment = assess_cleanup(row(cleanup_wire(**updates)), NOW)
    assert assessment.damaged and not assessment.summaries


def test_dashboard_exposes_cleanup_damage_without_raw_private_prose(tmp_path):
    config, _, _ = setup(tmp_path)
    data = row(
        payload(check()) + cleanup_wire(deleted_audit_count="/private/user/secret")
    )
    pl.DataFrame(
        [{key: data.get(key) for key in runs.RUNS_SCHEMA}], schema=runs.RUNS_SCHEMA
    ).write_parquet(runs.runs_path(config))
    panel = _retention_panel(collect_context(config, now=NOW))
    assert panel.status == "amber" and "audit-cleanup evidence" in panel.intro
    assert "/private" not in panel.intro
    stats = {s.label: s.value for s in panel.stats}
    assert (
        stats["last recorded check"] == "clear"
        and stats["cleanup evidence"] == "damaged"
    )
    assert stats["verification uncertainty"] == "no"


def test_unresolved_extension_annotation_does_not_erase_base_fields():
    @dataclass(frozen=True, slots=True)
    class Unresolved(status.ComponentDiagnostic):
        extension: "UnavailableType" = None

    item = Unresolved("other", 0, check_id="preserved")
    wire = json.loads(status.failure_details_json((item,)))[0]
    assert wire["stage"] == "other" and wire["check_id"] == "preserved"
    assert wire["metadata_invalid"] is True


def test_normalization_preserves_both_bad_fields_with_broken_optional_container():
    outcome = status.CommandOutcome.normalize(
        status.CommandOutcome(0.5, [], diagnostics=None)
    )
    assert outcome.exit_code == 1 and outcome.failure_kind == "invalid_command_outcome"
    details = [item.detail for item in outcome.diagnostics]
    assert any("exit_code" in detail for detail in details)
    assert any("failure_kind" in detail for detail in details)
    assert status.CommandOutcome.normalize(outcome) is outcome


def test_unrenderable_original_exception_still_gets_a_core_row(tmp_path):
    class OriginalError(RuntimeError):
        def __str__(self):
            raise ValueError("cannot render original")

    config, _, _ = setup(tmp_path)
    cli._record_run(
        config,
        argparse.Namespace(command="backtest"),
        started_at=datetime.now(UTC),
        exit_code=None,
        error=OriginalError(),
    )
    record = saved(config)
    assert record["error"] == "OriginalError" and record["exit_code"] is None
    assert (
        record["error_detail"] is None
        and "invalid_diagnostic_metadata" in record["failure_details_json"]
    )


def test_known_summary_remains_after_unidentified_cleanup_damage():
    items = cleanup_wire()
    damaged = {
        "stage": "retention_audit_cleanup",
        "exit_code": 0,
        "detail": {},
        "metadata_invalid": True,
    }
    result = retention_status(
        pl.DataFrame([row(payload(check()) + items + [damaged])]), as_of=NOW
    )
    assert result.cleanup_evidence_unknown and result.cleanup_outcome == "completed"
    assert result.cleanup_recorded_at == NOW and not result.verification_unknown


def test_identified_cleanup_damage_cannot_lend_its_summary_trust():
    items = cleanup_wire()
    items.append(
        {
            "stage": "retention_audit_cleanup",
            "exit_code": 0,
            "check_id": "shared",
            "detail": {},
            "metadata_invalid": True,
        }
    )
    result = assess_cleanup(row(items), NOW)
    assert result.damaged and not result.summaries


def test_legacy_missing_cleanup_clock_is_not_invented_and_later_clock_can_replace_it():
    missing = cleanup_wire()
    missing[0].pop("recorded_at")
    first = row(missing)
    before = retention_status(pl.DataFrame([first]), as_of=NOW)
    assert before.cleanup_outcome == "completed" and before.cleanup_recorded_at is None
    when = NOW + timedelta(minutes=1)
    later = row(cleanup_wire(when), when=when)
    result = retention_status(pl.DataFrame([first, later]), as_of=when)
    assert result.cleanup_recorded_at == when


@pytest.mark.parametrize("raw", ["not json", "{}", "null", "[]"])
def test_unattributable_payload_never_fabricates_cleanup_state(raw):
    data = row([])
    data["failure_details_json"] = raw
    assessment = assess_cleanup(data, NOW)
    assert not assessment.damaged and not assessment.summaries
