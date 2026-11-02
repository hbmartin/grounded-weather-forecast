"""Regression boundaries from the coordinated review backlog."""

import json
import os
import signal
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, NewType

import pytest
from test_launchd import SCRIPT, load_script
from test_trusted_outcome_cleanup import NOW, cleanup_wire, row

from grounded_weather_forecast import (
    diagnostic_groups,
    diagnostic_values,
    shadow_artifacts,
    status,
)
from grounded_weather_forecast import (
    retention_status as retained_status,
)
from grounded_weather_forecast.cleanup_status import assess_cleanup
from grounded_weather_forecast.selection_decisions import argmin_selected


@dataclass(frozen=True, slots=True)
class BrokenAnnotation(status.ComponentDiagnostic):
    bad: "status.NoSuchThing" = None
    valid: "int | None" = 4


Identifier = NewType("Identifier", str)


@dataclass(frozen=True, slots=True)
class NarrowedDiagnostic(status.ComponentDiagnostic):
    stage: Literal["retention_verification"]
    check_id: Identifier | None = None


def test_bad_annotation_preserves_siblings_and_wire_identity():
    wire = json.loads(
        status.failure_details_json((BrokenAnnotation("other", 0, check_id="kept"),))
    )[0]
    assert wire["stage"] == "other"
    assert wire["check_id"] == "kept"
    assert wire["valid"] == 4
    assert wire["metadata_invalid"]


def test_narrowed_annotations_preserve_inherited_wire_fields():
    wire = json.loads(
        status.failure_details_json(
            (
                NarrowedDiagnostic(
                    "retention_verification", 0, check_id=Identifier("check")
                ),
            )
        )
    )[0]
    assert wire["stage"] == "retention_verification"
    assert wire["check_id"] == "check"
    assert "metadata_invalid" not in wire


def test_incomplete_resolution_is_not_cached(monkeypatch):
    @dataclass(frozen=True, slots=True)
    class ResolvesLater(status.ComponentDiagnostic):
        extra: "LaterPrimitive" = 4  # noqa: F821 -- resolves only after first assessment

    assert diagnostic_values.declared_types(ResolvesLater)["extra"] == ()
    monkeypatch.setitem(globals(), "LaterPrimitive", int)
    assert diagnostic_values.declared_types(ResolvesLater)["extra"] == (int,)


@pytest.mark.parametrize("exit_code", [256, 2**32, -256, 2**63 - 1, -(2**63)])
def test_nonportable_exit_preserves_original_kind_and_fails_closed(exit_code):
    outcome = status.CommandOutcome.normalize(
        status.CommandOutcome(exit_code, "promotion_evidence_invalid")
    )
    assert outcome.exit_code == 1
    assert outcome.invalid
    wire = status.failure_details_json(outcome.diagnostics)
    assert "promotion_evidence_invalid" in wire
    assert str(exit_code) in wire


def test_subprocess_signal_is_normalized_once():
    outcome = status.CommandOutcome.normalize(-int(signal.SIGTERM))
    assert outcome.exit_code == 128 + signal.SIGTERM
    assert status.CommandOutcome.normalize(outcome) is outcome


@pytest.mark.parametrize("kind", ["contradiction", "multiple-owners", "check-only"])
def test_cleanup_quarantines_conflicting_identity(kind):
    first = cleanup_wire(progress_id="one")
    other = dict(first[0])
    if kind == "contradiction":
        other["failure_kind"] = "retention_clear"
    elif kind == "multiple-owners":
        other["progress_id"] = "two"
    else:
        other.pop("progress_id", None)
        other["issue_count"] = -1
    result = assess_cleanup(row([*first, other]), NOW)
    assert result.damaged
    assert not result.summaries


@pytest.mark.parametrize(
    "decision", ["dm_not_significant", "seq_no_matrix", "mcs_not_separated"]
)
def test_statistical_rejection_is_never_an_argmin(decision):
    assert not argmin_selected({"gate": "eligibility", "decision_gate": decision})
    assert argmin_selected({"gate": "eligibility", "decision_gate": None})


@pytest.mark.parametrize("value", [False, 0, "", [], {}])
def test_falsy_invalid_decisions_never_receive_argmin_exemptions(value):
    assert not argmin_selected({"decision_gate": value, "gate": None})


@pytest.mark.parametrize(
    "value", ["0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00"]
)
def test_extreme_timestamp_offsets_are_named_refusals(value):
    with pytest.raises(
        shadow_artifacts.EvidenceError, match="timestamp must include timezone"
    ):
        shadow_artifacts.timestamp(value)


@pytest.mark.parametrize("kind", ["promotion_evidence_invalid", "long_kind" * 100])
def test_invalid_exit_retains_valid_original_kind_without_misclassifying_it(kind):
    result = status.CommandOutcome.normalize(status.CommandOutcome(256, kind))
    original = [
        d for d in result.diagnostics if "original_failure_kind" in (d.detail or "")
    ]
    assert len(original) == 1
    assert original[0].failure_kind == status.INVALID_COMMAND_OUTCOME
    assert len(original[0].detail) <= 400
    if len(kind) > 400:
        assert original[0].detail.endswith("...")


def test_grouped_reader_reuses_input_identity_assessments(monkeypatch):
    items = [
        {
            "stage": "retention_verification",
            "exit_code": 0,
            "failure_kind": "retention_clear",
            "progress_id": "owner",
            "check_id": "check",
        }
    ] * 2
    document = diagnostic_groups.decode_diagnostics(row(items))
    assess = retained_status.effective_progress_id
    calls = []

    def counted(progress, check):
        calls.append((progress, check))
        return assess(progress, check)

    monkeypatch.setattr(retained_status, "effective_progress_id", counted)
    result = retained_status._recorded_check_assessment(
        row(items), NOW, document=document
    )
    assert not result.invalid
    assert len(result.checks) == 1
    # Only the new summary's public constructor validates its own identity.
    # Input items were assessed during decoding, never again per item.
    assert calls.count(("owner", "check")) == 1


@pytest.fixture
def local_calendar(monkeypatch):
    prior = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    yield
    if prior is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = prior
    time.tzset()


@pytest.mark.usefixtures("local_calendar")
def test_slot_calendar_handles_both_dst_transitions():
    module = load_script(SCRIPT)
    spring = datetime.fromtimestamp(module.local_slot(date(2026, 3, 8), (2, 15)))  # noqa: DTZ006 -- tests host-local calendar
    assert (spring.hour, spring.minute) == (3, 0)
    fall = module.local_slot(date(2026, 11, 1), (1, 15))
    assert time.localtime(fall).tm_isdst == 1
    assert module.latest_scheduled_slot(fall + 3600, (1, 15)) == fall


def test_stamp_preserves_symlink_mode_and_cleans_interrupt(tmp_path, monkeypatch):
    module = load_script(SCRIPT)
    target = tmp_path / "target"
    target.write_text("")
    target.chmod(0o640)
    stamp = tmp_path / "last-maintenance"
    stamp.symlink_to(target)
    module.stamp_success(stamp, 1000)
    assert stamp.is_symlink()
    assert target.stat().st_mode & 0o777 == 0o640
    before = set(tmp_path.iterdir())

    def interrupted(*_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(module.os, "utime", interrupted)
    with pytest.raises(KeyboardInterrupt):
        module.stamp_success(stamp, 2000)
    assert set(tmp_path.iterdir()) == before
    assert target.stat().st_mtime == 1000


def test_evidence_writer_syncs_before_link_and_cleans_failed_write(
    tmp_path, monkeypatch
):
    events = []
    fsync, link = shadow_artifacts.os.fsync, shadow_artifacts.os.link

    def synchronized(descriptor):
        events.append("sync")
        fsync(descriptor)

    def linked(*args):
        events.append("link")
        link(*args)

    monkeypatch.setattr(shadow_artifacts.os, "fsync", synchronized)
    monkeypatch.setattr(shadow_artifacts.os, "link", linked)
    target = tmp_path / "record.json"
    shadow_artifacts.write_bytes_new(target, b"complete")
    assert events == ["sync", "link", "sync"]
    assert target.read_bytes() == b"complete"

    def unavailable(_descriptor):
        raise OSError("sync unavailable")

    monkeypatch.setattr(shadow_artifacts.os, "fsync", unavailable)
    with pytest.raises(OSError, match="sync unavailable"):
        shadow_artifacts.write_bytes_new(tmp_path / "failed.json", b"payload")
    assert set(tmp_path.iterdir()) == {target}
