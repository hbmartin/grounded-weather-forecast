"""Canonical compatibility boundaries and verified audit-cleanup progress."""

import argparse
import copy
import json
import pickle
from dataclasses import dataclass, replace
from pathlib import Path

import polars as pl
import pytest
from filelock import FileLock
from test_audit_outcomes import staging_file
from test_daily_read import NOW, setup
from test_retention_verification import run_row
from test_safe_retention_diagnostics import old_audit

from grounded_weather_forecast import cli
from grounded_weather_forecast import retention_audit as audit
from grounded_weather_forecast.dashboard.context import DashboardContext
from grounded_weather_forecast.dashboard.zones.operations import _retention_panel
from grounded_weather_forecast.reports import operations
from grounded_weather_forecast.retention_status import (
    RetentionVerification,
    retention_status,
)
from grounded_weather_forecast.serve.approvals import ApprovalIssue
from grounded_weather_forecast.status import failure_details_json
from grounded_weather_forecast.verification_text import VerificationIssue


@dataclass
class MutableIssue:
    path: Path | None
    category: str
    detail: str
    reason_code: str | None = None


@pytest.mark.parametrize("source", ["list", "approval", "mutable"])
def test_verification_freezes_issue_sources_and_raw_views(source):
    issue = VerificationIssue(Path("/p"), "unreadable", "/p: unreadable: real prose")
    if source == "approval":
        supplied = [ApprovalIssue(issue.path, issue.category, issue.detail)]
    elif source == "mutable":
        supplied = [MutableIssue(issue.path, issue.category, issue.detail)]
    else:
        supplied = [issue]
    check = RetentionVerification("suspended", NOW, issues=supplied)
    if source == "mutable":
        supplied[0].detail = "changed externally"
    supplied.clear()
    assert check.issues == (issue,) and isinstance(check.issues, tuple)
    assert check.details == (issue.detail,)
    assert check.formatted_details == (issue.diagnostic(),)
    assert replace(check, check_id="x").issues is check.issues
    assert RetentionVerification(issues=check.issues).issues is check.issues
    assert copy.copy(check) == pickle.loads(pickle.dumps(check)) == check


def test_legacy_edits_have_explicit_precedence_and_clear_without_hidden_state():
    original = RetentionVerification(
        "suspended",
        NOW,
        issues=(VerificationIssue(Path("/a"), "unreadable", "a"),),
    )
    edited = replace(
        original,
        details=("/a: unreadable: changed", "new"),
        categories=("unreadable", ""),
    )
    assert edited.issues == (
        VerificationIssue(Path("/a"), "unreadable", "changed"),
        VerificationIssue(None, "verification_invalid", "new"),
    )
    assert replace(original, details=()).issues == ()
    assert replace(original, paths=("",), reason_codes=("",)).issues == (
        VerificationIssue(None, "unreadable", "a"),
    )
    fresh = VerificationIssue(Path("/fresh"), "unknown-code", "/a: genuine text")
    assert replace(original, issues=[fresh]).issues == (fresh,)
    assert replace(original, issues=[fresh]).details == (fresh.detail,)


def test_journal_raw_details_roundtrip_without_reattaching_or_stripping_prefixes():
    raw = "/p: unreadable: actual evidence"
    check = RetentionVerification(
        "suspended",
        NOW,
        check_id="check",
        issues=(VerificationIssue(Path("/p"), "unreadable", raw),),
    )
    for _ in range(3):
        row = run_row() | {
            "failure_details_json": failure_details_json(check.diagnostics())
        }
        check = retention_status(pl.DataFrame([row]), as_of=NOW).verification
        assert check.details == (raw,)
        assert check.formatted_details == ("/p: unreadable: " + raw,)


@pytest.mark.parametrize("supplied", [False, True])
def test_typed_failure_context_distinguishes_omission_from_unknown(supplied):
    cause = TimeoutError("private raw timeout")
    error = (
        audit._AuditPublicationError("audit_io_failed", cause, None, None)
        if supplied
        else audit._AuditPublicationError("audit_io_failed", cause)
    )
    for value in (error, copy.copy(error), pickle.loads(pickle.dumps(error))):
        failure = audit.AuditCleanupFailure.capture(
            value, operation="references", path=Path("/verified")
        )
        assert failure.operation == "references"
        assert failure.path == (None if supplied else Path("/verified"))
        assert failure.raw_detail == str(cause) and failure.error_type == "TimeoutError"
        assert value.filename is None


def test_cleanup_entries_replace_atomically_and_preserve_legacy_adapters():
    failure = audit.AuditCleanupFailure(
        "delete",
        Path("/old"),
        "denied",
        "audit_io_failed",
        "permission_denied",
        "PermissionError",
    )
    entry = audit.AuditCleanupEntry(failure.issue(), failure)
    supplied = [entry]
    result = audit.AuditCleanupResult(entries=supplied, completed=True, recorded_at=NOW)
    supplied.clear()
    assert result.entries == (entry,) and result.raw_failures == (failure,)
    assert replace(result, dry_run=True).entries is result.entries
    for value in (copy.copy(result), pickle.loads(pickle.dumps(result))):
        assert value == result and value.diagnostics() == result.diagnostics()
    fresh = VerificationIssue(Path("/new"), "audit_cleanup_file_failed", "different")
    changed = replace(result, issues=[fresh])
    assert changed.issues == (fresh,) and changed.raw_failures == ()
    assert changed.diagnostics()[1].operation is None
    assert changed.diagnostics()[1].audit_cause_code is None
    reconstructed = audit.AuditCleanupResult(
        result.issues,
        False,
        0,
        NOW,
        None,
        None,
        None,
        result.raw_failures,
        completed=True,
    )
    assert reconstructed == result
    with pytest.raises(ValueError, match="does not match"):
        replace(result, issues=[fresh], raw_failures=result.raw_failures)
    with pytest.raises(ValueError, match="must match"):
        audit.AuditCleanupResult(issues=(), raw_failures=result.raw_failures)
    assert replace(result, issues=()).entries == ()


def test_entry_failure_metadata_is_local_to_its_own_issue():
    missing = audit.AuditCleanupFailure(
        "revalidate", Path("/missing"), "gone", cause_code="not_found"
    )
    legacy = VerificationIssue(Path("/legacy"), "audit_cleanup_file_failed", "old")
    result = audit.AuditCleanupResult(
        entries=(
            audit.AuditCleanupEntry(legacy),
            audit.AuditCleanupEntry(missing.issue(), missing),
        ),
        completed=True,
    )
    first, second = result.diagnostics()[1:]
    assert first.operation is None and first.audit_cause_code is None
    assert second.path == "/missing" and second.operation == "revalidate"
    assert second.audit_cause_code == "not_found"


@pytest.mark.parametrize("recorded", [False, True])
def test_unknown_cleanup_never_claims_success(tmp_path, monkeypatch, capsys, recorded):
    config, _, _ = setup(tmp_path)
    result = audit.AuditCleanupResult(recorded_at=NOW if recorded else None)
    assert "outcome was not recorded" in result.summary()
    assert "Audit cleanup completed" not in result.summary()
    if recorded:
        (summary,) = result.diagnostics()
        assert summary.failure_kind == "audit_cleanup_unverified"
        assert summary.cleanup_completed is None
        assert summary.deleted_audit_count is summary.eligible_audit_count is None
    else:
        assert result.diagnostics() == ()
    wrapped = operations.PruneResult((), (), (), 0.0, audit_cleanup=result)
    monkeypatch.setattr(operations, "prune_scores_files", lambda *a, **k: wrapped)
    outcome = cli._cmd_prune_scores(config, argparse.Namespace(dry_run=True))
    assert "outcome was not recorded" in capsys.readouterr().out
    assert not any(
        i.failure_kind == "audit_cleanup_complete" for i in outcome.diagnostics
    )


@pytest.mark.parametrize("dry_run", [False, True])
def test_preview_counts_require_reference_and_signature_validation(
    tmp_path, monkeypatch, dry_run
):
    config, _, _ = setup(tmp_path)
    eligible, referenced, changed = [old_audit(config) for _ in range(3)]
    staging = staging_file(config)
    fresh = audit.publish_verification(
        config, RetentionVerification("suspended", NOW, ("fresh",)), dry_run=False
    )
    monkeypatch.setattr(audit, "_referenced_audits", lambda _c: {referenced.name})
    signature = audit._signature
    calls = 0

    def changing(path):
        nonlocal calls
        value = signature(path)
        if path == changed:
            calls += 1
            if calls == 3:
                return (*value[:-1], value[-1] + 1)
        return value

    monkeypatch.setattr(audit, "_signature", changing)
    result = audit.cleanup_audits_result(config, now=NOW, dry_run=dry_run)
    assert result.completed is True and not result.aborted
    assert result.eligible_audit_count == result.eligible_staging_count == 1
    assert (
        result.deleted_audit_count
        == result.deleted_staging_count
        == (0 if dry_run else 1)
    )
    assert eligible.exists() is dry_run and staging.exists() is dry_run
    assert (
        referenced.exists()
        and changed.exists()
        and (config.artifacts_dir / fresh.audit_path).exists()
    )
    assert ("Would delete 1 audits and 1 staging files." in result.summary()) is dry_run
    summary = json.loads(failure_details_json(result.diagnostics()))[0]
    assert summary["eligible_audit_count"] == summary["eligible_staging_count"] == 1
    assert summary["cleanup_completed"] is True and summary["dry_run"] is dry_run


def test_failed_unlink_counts_eligible_separately_and_preserves_error_type(
    tmp_path, monkeypatch
):
    config, _, _ = setup(tmp_path)
    first, second = sorted([old_audit(config), old_audit(config)])
    unlink = Path.unlink
    raw = "denied\n" + "/Users/private/" * 100

    def denied(path, *args, **kwargs):
        if path == first:
            raise PermissionError(raw)
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", denied)
    result = audit.cleanup_audits_result(config, now=NOW, dry_run=False)
    assert result.eligible_audit_count == 2 and result.deleted_audit_count == 1
    assert first.exists() and not second.exists() and result.completed is True
    issue = result.issues[0]
    assert "\n" not in issue.detail and len(issue.detail) == 400
    assert result.raw_failures[0].raw_detail == raw
    assert result.diagnostics()[1].error_type == "PermissionError"
    assert "private" not in " ".join(result.public_notes(config))


@pytest.mark.parametrize("dry_run", [False, True])
def test_unlock_abort_reports_verified_progress_without_complete_preview(
    tmp_path, monkeypatch, dry_run
):
    config, _, _ = setup(tmp_path)
    committed, staging = old_audit(config), staging_file(config)
    release = FileLock.release

    def failed_release(lock, *args, **kwargs):
        release(lock, *args, **kwargs)
        raise OSError("lock release failed")

    with monkeypatch.context() as patch:
        patch.setattr(FileLock, "release", failed_release)
        result = audit.cleanup_audits_result(config, now=NOW, dry_run=dry_run)
    assert result.completed is False and result.aborted
    assert result.eligible_audit_count == result.eligible_staging_count == 1
    assert (
        result.deleted_audit_count
        == result.deleted_staging_count
        == (0 if dry_run else 1)
    )
    assert committed.exists() is dry_run and staging.exists() is dry_run
    assert "Verified eligible so far:" in result.summary()
    assert "Would delete" not in result.summary()
    assert "earlier verified deletions may" not in " ".join(result.public_notes(config))


def test_preacquisition_abort_preserves_shared_code_and_known_zero_progress(tmp_path):
    config, _, _ = setup(tmp_path)
    committed = old_audit(config)
    with FileLock(committed.parent / ".audit.lock", timeout=0):
        result = audit.cleanup_audits_result(config, now=NOW, dry_run=True)
    assert result.completed is False and result.deleted_audit_count == 0
    assert result.eligible_audit_count == 0 and committed.exists()
    assert result.issues[0].reason_code == "audit_lock_contended"
    assert result.diagnostics()[1].operation == "lock_acquire"
    assert "may have completed" not in result.public_notes(config)[0]


@pytest.mark.parametrize(
    "reason", ["audit_lock_contended", "audit_link_failed", "audit_sync_failed"]
)
def test_legacy_publication_reasons_preserve_only_proven_outcomes(
    tmp_path, monkeypatch, capsys, reason
):
    config, _, _ = setup(tmp_path)
    check = RetentionVerification(
        "suspended",
        NOW,
        check_id="old",
        audit_error="private /Users/private",
        audit_reason_code=reason,
    )
    proven = reason != "audit_sync_failed"
    assert ("No audit was published" in check.audit_explanation()) is proven
    assert "private" not in check.audit_explanation()
    assert (
        "linked audit remains unconfirmed"
        in replace(
            check, audit_candidate_path="history/pruning_verifications/x.jsonl"
        ).audit_explanation()
    )
    row = run_row() | {
        "failure_details_json": failure_details_json(check.diagnostics())
    }
    loaded = retention_status(pl.DataFrame([row]), as_of=NOW).verification
    assert loaded.audit_explanation() == check.audit_explanation()
    assert loaded.frame()["audit_error"][0] == check.audit_explanation()
    panel = _retention_panel(DashboardContext(config, NOW, runs=pl.DataFrame([row])))
    assert check.audit_explanation() in panel.intro
    monkeypatch.setattr(
        operations,
        "prune_scores_files",
        lambda *a, **k: operations.PruneResult((), (), (), 0.0, verification=check),
    )
    assert (
        cli._cmd_prune_scores(config, argparse.Namespace(dry_run=True)).exit_code == 0
    )
    assert check.audit_explanation() in capsys.readouterr().out


def test_compatibility_result_initializes_cleanup_batch_once(
    tmp_path, monkeypatch, capsys
):
    config, _, _ = setup(tmp_path)
    failure = audit.AuditCleanupFailure(
        "delete", Path("/private/path"), "private", error_type="OSError"
    )
    cleanup = audit.AuditCleanupResult(
        entries=(audit.AuditCleanupEntry(failure.issue(), failure),), recorded_at=NOW
    )
    original, calls = audit.AuditCleanupResult.diagnostics, []

    def counted(self, **kwargs):
        calls.append(self)
        return original(self, **kwargs)

    monkeypatch.setattr(audit.AuditCleanupResult, "diagnostics", counted)
    check = RetentionVerification("clear", NOW, check_id="same")
    result = operations.PruneResult(
        (), (), (), 0.0, verification=check, audit_cleanup=cleanup
    )
    assert len(calls) == 1 and all(
        i.check_id == "same" for i in result.audit_cleanup_diagnostics
    )
    monkeypatch.setattr(operations, "prune_scores_files", lambda *a, **k: result)
    outcome = cli._cmd_prune_scores(config, argparse.Namespace(dry_run=False))
    assert (
        len(calls) == 1
        and outcome.diagnostics[-1] is result.audit_cleanup_diagnostics[-1]
    )
    assert "private" not in capsys.readouterr().out


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("late_failure", [False, True])
def test_preview_batch_is_recorded_once_through_late_failures(
    tmp_path, monkeypatch, capsys, embedded, late_failure
):
    from grounded_weather_forecast import runs

    config, _, _ = setup(tmp_path)
    committed, staging = old_audit(config), staging_file(config)
    original, calls = audit.AuditCleanupResult.diagnostics, []

    def counted(self, **kwargs):
        calls.append(self)
        return original(self, **kwargs)

    monkeypatch.setattr(audit.AuditCleanupResult, "diagnostics", counted)
    if late_failure:

        def fail(*args, **kwargs):
            raise OSError("later output failed")

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
        with pytest.raises(OSError, match="later output"):
            invoke()
    else:
        assert invoke() == 0
    assert len(calls) == 1 and committed.exists() and staging.exists()
    ledger = runs.read_runs(runs.runs_path(config))
    assert ledger.height == 1
    diagnostics = json.loads(ledger["failure_details_json"][0])
    (summary,) = [
        i
        for i in diagnostics
        if i["stage"] == "retention_audit_cleanup" and "issue_count" in i
    ]
    assert summary["eligible_audit_count"] == summary["eligible_staging_count"] == 1
    assert summary["deleted_audit_count"] == summary["deleted_staging_count"] == 0
    assert summary["cleanup_completed"] is summary["dry_run"] is True
    assert summary["exit_code"] == 0
    if not late_failure:
        assert operations._run_counts(ledger, ledger["ended_at"][0]) == (0, 0, 0)
    capsys.readouterr()


def test_explicit_incomplete_cleanup_is_distinct_from_unknown_legacy_outcome():
    result = audit.AuditCleanupResult(completed=False, recorded_at=NOW)
    assert "did not complete" in result.summary()
    assert "outcome was not recorded" not in result.summary()
    assert result.diagnostics()[0].failure_kind == "audit_cleanup_aborted"
    assert result.diagnostics()[0].cleanup_completed is False


@pytest.mark.parametrize("explicit_unknown", [False, True])
def test_operation_boundary_supplies_only_missing_legacy_context(explicit_unknown):
    cause = TimeoutError("private source")
    original = (
        audit._AuditPublicationError("audit_io_failed", cause, None, None)
        if explicit_unknown
        else audit._AuditPublicationError("audit_io_failed", cause)
    )
    with pytest.raises(audit._AuditPublicationError) as caught:
        with audit._audit_operation("revalidate", Path("/verified")):
            raise original
    failure = audit.AuditCleanupFailure.capture(
        caught.value, operation="delete", path=Path("/other")
    )
    assert failure.operation == "revalidate"
    assert failure.path == (None if explicit_unknown else Path("/verified"))
    assert failure.raw_detail == str(cause) and failure.error_type == "TimeoutError"
    complete = audit._AuditPublicationError(
        "audit_cleanup_failed", cause, "lock_release", None
    )
    with pytest.raises(audit._AuditPublicationError) as preserved:
        with audit._audit_operation("staging_write", Path("/unrelated")):
            raise complete
    assert preserved.value is complete
