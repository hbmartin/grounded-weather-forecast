"""Raw evidence is lossless; immutable cleanup batches stay bound to their source."""

import argparse
import copy
import json
import pickle
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from test_daily_read import NOW, setup
from test_retention_verification import run_row

from grounded_weather_forecast import cli, runs
from grounded_weather_forecast import retention_audit as audit
from grounded_weather_forecast.reports import operations
from grounded_weather_forecast.retention_status import (
    RetentionVerification,
    retention_status,
)
from grounded_weather_forecast.status import (
    ComponentDiagnostic,
    diagnostic_progress,
    failure_details_json,
    merge_diagnostics,
    record_progress,
)
from grounded_weather_forecast.verification_text import VerificationIssue


@pytest.mark.parametrize(
    "raw",
    [
        "unreadable: x",
        "/p: unreadable: x",
        "/p: unreadable: /p: unreadable: actual prose",
        r"C:\Users\private\x: unreadable: x",
        r"\\server\share\x: unreadable: x",
        "https://private.example/path: unreadable: x",
        "",
        "first\nsecond",
    ],
)
def test_raw_views_constructor_and_text_replacement_roundtrip(raw):
    check = RetentionVerification(
        "suspended",
        NOW,
        issues=(VerificationIssue(Path("/p"), "unreadable", raw, "future-code"),),
        check_id="check",
    )
    for _ in range(3):
        rebuilt = RetentionVerification(
            check.state,
            check.recorded_at,
            check.details,
            check.paths,
            categories=check.categories,
            reason_codes=check.reason_codes,
            check_id=check.check_id,
        )
        assert rebuilt == check and rebuilt.details == (raw,)
        assert replace(check, details=check.details) == check
        assert replace(check, recorded_at=NOW).issues is check.issues
        check = rebuilt
    assert copy.copy(check) == pickle.loads(pickle.dumps(check)) == check
    assert replace(check, details=()).issues == ()


def test_explicit_legacy_adapter_strips_only_recorded_prefixes_once():
    value = "/p: unreadable: /p: unreadable: genuine evidence"
    legacy = RetentionVerification.from_legacy_details(
        "suspended",
        NOW,
        (value, "unreadable: no path", "unknown prefix"),
        ("/p", ""),
        categories=("unreadable", "unreadable"),
        reason_codes=("", "future-code"),
        check_id="legacy",
        dry_run=True,
    )
    assert legacy.details == (
        "/p: unreadable: genuine evidence",
        "no path",
        "unknown prefix",
    )
    assert legacy.paths == ("/p", None, None)
    assert legacy.categories == ("unreadable", "unreadable", "verification_invalid")
    assert legacy.reason_codes == (None, "future-code", None)
    assert legacy.dry_run is True and legacy.check_id == "legacy"
    for _ in range(3):
        row = run_row() | {
            "failure_details_json": failure_details_json(legacy.diagnostics())
        }
        loaded = retention_status(pl.DataFrame([row]), as_of=NOW).verification
        assert (
            loaded.details == legacy.details
            and loaded.sample_issues() == legacy.sample_issues()
        )
        legacy = loaded
    assert RetentionVerification(
        details=(value,), paths=("/p",), categories=("unreadable",)
    ).details == (value,)


def failures():
    return tuple(
        audit.AuditCleanupFailure(
            "delete", Path(f"/p/{i}"), f"raw-{i}", error_type="OSError"
        )
        for i in range(3)
    )


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_partial_legacy_failure_metadata_retains_every_issue(count):
    raw = failures()
    issues = tuple(failure.issue() for failure in raw)
    result = audit.AuditCleanupResult(
        issues=issues, raw_failures=raw[:count], completed=True
    )
    assert result.issues == issues and len(result.entries) == 3
    for index, sample in enumerate(result.diagnostics()[1:]):
        assert sample.path == str(issues[index].path)
        assert sample.operation == ("delete" if index < count else None)
        assert sample.error_type == ("OSError" if index < count else None)
    assert result.raw_failures == raw[:count]
    assert pickle.loads(pickle.dumps(result)) == result


def test_canonical_entries_and_empty_legacy_metadata_are_not_clearing():
    raw = failures()
    entries = tuple(audit.AuditCleanupEntry(value.issue(), value) for value in raw)
    result = audit.AuditCleanupResult(entries=entries, raw_failures=(), completed=True)
    assert result.entries is entries and result.raw_failures == raw
    assert replace(result, raw_failures=()).entries is entries
    overlay = audit.AuditCleanupResult(
        entries=entries, raw_failures=raw[:1], completed=True
    )
    assert overlay.entries == entries
    cleared = replace(result, issues=(), raw_failures=())
    assert cleared.entries == () and cleared.raw_failures == ()
    detached = replace(result, issues=result.issues)
    assert detached.issues == result.issues and detached.raw_failures == ()
    derived = audit.AuditCleanupResult(raw_failures=raw, completed=True)
    assert derived.entries == entries


def test_mismatched_or_excess_metadata_still_fails():
    raw = failures()
    with pytest.raises(ValueError, match="must match"):
        audit.AuditCleanupResult(issues=(raw[0].issue(),), raw_failures=raw)
    with pytest.raises(ValueError, match="does not match"):
        audit.AuditCleanupResult(
            issues=(raw[1].issue(), raw[2].issue()), raw_failures=raw[:1]
        )
    with pytest.raises(ValueError, match="does not match"):
        audit.AuditCleanupResult(
            entries=(audit.AuditCleanupEntry(raw[1].issue()),), raw_failures=raw[:1]
        )


@pytest.mark.parametrize(
    "aborted,completed,outcome",
    [
        (False, None, audit.CleanupOutcome.UNKNOWN),
        (False, False, audit.CleanupOutcome.INCOMPLETE),
        (False, True, audit.CleanupOutcome.COMPLETED),
        (True, None, audit.CleanupOutcome.INCOMPLETE),
        (True, False, audit.CleanupOutcome.INCOMPLETE),
    ],
)
def test_one_outcome_drives_summary_and_emitted_flag(aborted, completed, outcome):
    result = audit.AuditCleanupResult(
        aborted=aborted,
        completed=completed,
        recorded_at=NOW,
        deleted_audit_count=1,
        deleted_staging_count=0,
    )
    assert result.outcome is outcome
    summary = result.diagnostics()[0]
    expected = {
        audit.CleanupOutcome.UNKNOWN: (
            "final status is unknown",
            "audit_cleanup_unverified",
            None,
        ),
        audit.CleanupOutcome.INCOMPLETE: (
            "did not complete",
            "audit_cleanup_aborted",
            False,
        ),
        audit.CleanupOutcome.COMPLETED: ("completed", "audit_cleanup_complete", True),
    }[outcome]
    assert expected[0] in result.summary() and summary.failure_kind == expected[1]
    assert summary.cleanup_completed is expected[2] and summary.deleted_audit_count == 1
    assert (
        "Recorded deletions: 1 audits" in result.summary()
        if outcome is audit.CleanupOutcome.UNKNOWN
        else "Deleted 1 audits" in result.summary()
    )
    assert summary.exit_code == 0


def test_contradictory_outcome_flags_are_rejected():
    with pytest.raises(ValueError, match="aborted cleanup cannot be completed"):
        audit.AuditCleanupResult(aborted=True, completed=True)
    incomplete = audit.AuditCleanupResult(aborted=True, completed=False)
    with pytest.raises(ValueError, match="aborted cleanup cannot be completed"):
        replace(incomplete, completed=True)
    assert (
        audit.AuditCleanupResult(deleted_audit_count=1).outcome
        is audit.CleanupOutcome.UNKNOWN
    )


def cleanup(completed):
    return audit.AuditCleanupResult(
        recorded_at=NOW,
        completed=completed,
        aborted=not completed,
        deleted_audit_count=0,
        deleted_staging_count=0,
        dry_run=True,
    )


@pytest.mark.parametrize("initial_complete", [False, True])
def test_result_copies_reuse_or_rebind_atomic_diagnostic_source(
    monkeypatch, initial_complete
):
    original, calls = audit.AuditCleanupResult.diagnostics, []

    def counted(self, **kwargs):
        calls.append((self, kwargs.get("check_id")))
        return original(self, **kwargs)

    monkeypatch.setattr(audit.AuditCleanupResult, "diagnostics", counted)
    check = RetentionVerification("clear", NOW, check_id="first")
    result = operations.PruneResult(
        (), (), (), 0.0, verification=check, audit_cleanup=cleanup(initial_complete)
    )
    assert len(calls) == 1
    metadata = replace(result, freed_mb=2.0)
    assert metadata.cleanup_record is result.cleanup_record and len(calls) == 1
    opposite = replace(result, audit_cleanup=cleanup(not initial_complete))
    assert len(calls) == 2 and opposite.cleanup_record is not result.cleanup_record
    assert opposite.audit_cleanup_diagnostics[0].cleanup_completed is (
        not initial_complete
    )
    assert result.audit_cleanup_diagnostics[0].cleanup_completed is initial_complete
    new_check = replace(opposite, verification=replace(check, check_id="second"))
    assert len(calls) == 3
    assert all(
        item.check_id == "second" for item in new_check.audit_cleanup_diagnostics
    )
    for value in (copy.copy(new_check), pickle.loads(pickle.dumps(new_check))):
        assert (
            value == new_check
            and value.audit_cleanup_diagnostics == new_check.audit_cleanup_diagnostics
        )
    assert len(calls) == 3


def test_explicit_batch_must_match_bound_source_and_check():
    check = RetentionVerification("clear", NOW, check_id="same")
    record = audit.AuditCleanupRecord(cleanup(True), "same")
    compatible = operations.PruneResult(
        (),
        (),
        (),
        0.0,
        verification=check,
        audit_cleanup=record.result,
        cleanup_record=record,
        audit_cleanup_diagnostics=record.diagnostics,
    )
    assert compatible.cleanup_record is record
    with pytest.raises(ValueError, match="diagnostics do not match"):
        operations.PruneResult(
            (),
            (),
            (),
            0.0,
            verification=check,
            audit_cleanup=cleanup(False),
            audit_cleanup_diagnostics=record.diagnostics,
        )
    with pytest.raises(ValueError, match="diagnostics do not match"):
        replace(
            compatible,
            verification=replace(check, check_id="other"),
            audit_cleanup_diagnostics=record.diagnostics,
        )
    with pytest.raises(ValueError, match="diagnostics do not match"):
        operations.PruneResult(
            (), (), (), 0.0, audit_cleanup=cleanup(True), audit_cleanup_diagnostics=()
        )
    rebound = replace(record, result=cleanup(False))
    assert rebound.diagnostics[0].cleanup_completed is False


def test_identified_batch_replacement_preserves_other_checks_and_diagnostics():
    old = audit.AuditCleanupRecord(cleanup(True), "same").diagnostics
    new = audit.AuditCleanupRecord(cleanup(False), "same").diagnostics
    other = audit.AuditCleanupRecord(cleanup(True), "other").diagnostics
    unrelated = (ComponentDiagnostic("backtest", 1, "actual_failure", "kept"),)
    with diagnostic_progress() as progress:
        record_progress(old + unrelated + other)
        record_progress(new)
        assert tuple(progress) == unrelated + other + new
        record_progress(new)
        assert tuple(progress) == unrelated + other + new
    merged = merge_diagnostics(old + unrelated + other, new)
    assert merged == unrelated + other + new
    assert merge_diagnostics(merged, ()) == merged
    legacy_old = audit.AuditCleanupRecord(cleanup(True)).diagnostics
    legacy_new = audit.AuditCleanupRecord(cleanup(False)).diagnostics
    assert len(merge_diagnostics(legacy_old, legacy_new)) == 2


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("late_failure", [False, True])
def test_copied_cleanup_replaces_recorded_batch_in_one_journal_row(
    tmp_path, monkeypatch, capsys, embedded, late_failure
):
    config, _, _ = setup(tmp_path)
    prune = operations.prune_scores_files

    def updated(*args, **kwargs):
        result = prune(*args, **kwargs)
        changed = replace(
            result,
            audit_cleanup=replace(result.audit_cleanup, completed=False, aborted=True),
        )
        if late_failure:
            record_progress(changed.audit_cleanup_diagnostics)
        return changed

    monkeypatch.setattr(operations, "prune_scores_files", updated)
    if late_failure:

        def fail(*args, **kwargs):
            raise OSError("later output failure")

        monkeypatch.setattr(cli, "_print_score_temporaries", fail)

    def invoke():
        if embedded:
            args = argparse.Namespace(command="maintain", dry_run=True)
            return cli._run_pipeline_steps(
                config,
                args,
                (("prune-scores", lambda: cli._cmd_prune_scores(config, args)),),
            ).exit_code
        return cli.main(
            ["--config", str(tmp_path / "config.toml"), "prune-scores", "--dry-run"]
        )

    if late_failure:
        with pytest.raises(OSError, match="later output failure"):
            invoke()
    else:
        assert invoke() == 0
    ledger = runs.read_runs(runs.runs_path(config))
    assert ledger.height == 1
    items = json.loads(ledger["failure_details_json"][0])
    (summary,) = [
        item
        for item in items
        if item["stage"] == "retention_audit_cleanup" and "issue_count" in item
    ]
    assert summary["failure_kind"] == "audit_cleanup_aborted"
    assert summary["cleanup_completed"] is False and summary["dry_run"] is True
    assert summary["exit_code"] == 0
    assert "did not complete" in capsys.readouterr().out
    if not late_failure:
        assert operations._run_counts(ledger, ledger["ended_at"][0]) == (0, 0, 0)


def test_standalone_bound_record_retains_its_known_identity():
    record = audit.AuditCleanupRecord(cleanup(True), "known")
    result = operations.PruneResult((), (), (), 0.0, cleanup_record=record)
    assert result.cleanup_record is record and result.verification.check_id == "known"
    assert result.verification.state == "not_recorded"
    assert all(item.check_id == "known" for item in result.audit_cleanup_diagnostics)
    assert replace(result, kept=(Path("/score"),)).cleanup_record is record
    cleared = replace(result, verification=RetentionVerification())
    assert cleared.cleanup_record.check_id is None
    assert all(item.check_id is None for item in cleared.audit_cleanup_diagnostics)
