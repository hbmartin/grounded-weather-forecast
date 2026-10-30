"""Abort causes are recorded facts; authoritative batches preserve occurrences."""

import argparse
import copy
import json
import pickle
from dataclasses import replace
from datetime import timedelta
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
    _recorded_checks,
    retention_status,
)
from grounded_weather_forecast.status import (
    ComponentDiagnostic,
    diagnostic_progress,
    effective_progress_id,
    failure_details_json,
    merge_diagnostics,
    record_progress,
    record_verification_progress,
)
from grounded_weather_forecast.verification_text import (
    VerificationIssue,
    public_explanation,
)


def file_failure():
    return audit.AuditCleanupFailure(
        "delete",
        Path("/private/file"),
        "private: original\n" + "x" * 500,
        error_type="OSError",
        failure_role="file_local",
    )


def abort_failure():
    return audit.AuditCleanupFailure(
        "references",
        None,
        "uncertain references",
        error_type="ValueError",
        failure_role="abort_cause",
    )


def recorded(check):
    row = run_row() | {
        "failure_details_json": failure_details_json(check.diagnostics())
    }
    return retention_status(pl.DataFrame([row]), as_of=NOW).verification


def test_filtering_abort_cause_never_transfers_role_to_another_failure():
    cause, local = abort_failure(), file_failure()
    original = audit.AuditCleanupResult(
        raw_failures=(cause, local),
        aborted=True,
        completed=False,
        deleted_audit_count=1,
        deleted_staging_count=2,
    )
    filtered = replace(original, raw_failures=original.raw_failures[1:])
    assert (
        filtered.aborted is True and filtered.outcome is audit.CleanupOutcome.INCOMPLETE
    )
    assert filtered.raw_failures == (local,)
    assert filtered.issues[0].category == "audit_cleanup_file_failed"
    assert filtered.entries[0].failure.failure_role == "file_local"
    assert filtered.deleted_audit_count == 1 and filtered.deleted_staging_count == 2
    assert original.raw_failures == (cause, local)
    assert original.issues[0].category == "audit_cleanup_aborted"


def test_outcome_changes_require_explicit_correction_of_retained_abort_evidence():
    cause, local = abort_failure(), file_failure()
    original = audit.AuditCleanupResult(
        raw_failures=(cause, local), aborted=True, completed=False
    )
    with pytest.raises(ValueError, match="abort-cause evidence"):
        replace(original, aborted=False, completed=True)
    with pytest.raises(ValueError, match="abort-cause evidence"):
        replace(original, aborted=False)
    changed = replace(original, raw_failures=(local,), aborted=False, completed=True)
    assert changed.outcome is audit.CleanupOutcome.COMPLETED
    assert changed.raw_failures == (local,)
    corrected = replace(cause, failure_role="file_local")
    explicit = replace(
        original, raw_failures=(corrected, local), aborted=False, completed=True
    )
    assert explicit.raw_failures[0].raw_detail == cause.raw_detail
    assert explicit.issues[0].category == "audit_cleanup_file_failed"
    assert original.entries[0].failure is cause
    with pytest.raises(ValueError, match="cannot be completed"):
        replace(original, completed=True)


@pytest.mark.parametrize("operation", ["delete", "references", "lock_release"])
@pytest.mark.parametrize("aborted", [False, True])
def test_unclassified_legacy_failures_never_infer_roles(operation, aborted):
    failure = audit.AuditCleanupFailure(operation, None, "abort: denied /private/path")
    result = audit.AuditCleanupResult(
        raw_failures=(failure,), aborted=aborted, completed=False
    )
    assert result.raw_failures[0].failure_role is None
    assert result.issues[0].category == "audit_cleanup_failure"
    text = public_explanation(category=result.issues[0].category)
    assert "cleanup operation failed" in text and "private" not in text
    assert result.raw_failures[0].raw_detail == failure.raw_detail


@pytest.mark.parametrize(
    "role,category",
    [
        ("abort_cause", "audit_cleanup_aborted"),
        ("file_local", "audit_cleanup_file_failed"),
    ],
)
def test_explicit_legacy_categories_establish_roles_without_parsing_prose(
    role, category
):
    unknown = replace(file_failure(), failure_role=None)
    issue = VerificationIssue(unknown.path, category, unknown.issue().detail)
    entry = audit.AuditCleanupEntry(issue, unknown)
    assert entry.failure.failure_role == role
    assert entry.failure.raw_detail == unknown.raw_detail
    result = audit.AuditCleanupResult(entries=(entry,), aborted=role == "abort_cause")
    assert result.raw_failures[0].failure_role == role
    assert (
        audit.AuditCleanupResult(
            issues=result.issues,
            raw_failures=result.raw_failures,
            aborted=result.aborted,
        )
        == result
    )
    assert copy.copy(result) == pickle.loads(pickle.dumps(result)) == result
    assert replace(result, dry_run=True).entries is result.entries


def test_explicit_issue_adapter_cannot_override_a_known_role():
    cause, local = abort_failure(), file_failure()
    assert cause.issue().category == "audit_cleanup_aborted"
    assert local.issue().category == "audit_cleanup_file_failed"
    for failure, value in [(cause, False), (local, True)]:
        with pytest.raises(ValueError, match="conflicts"):
            failure.issue(aborted=value)
    unknown = replace(local, failure_role=None)
    legacy_abort = unknown.issue(aborted=True)
    entry = audit.AuditCleanupEntry(legacy_abort, unknown)
    assert entry.failure.failure_role == "abort_cause"
    with pytest.raises(ValueError, match="abort-cause evidence"):
        audit.AuditCleanupResult(entries=(entry,), completed=True)
    with pytest.raises(ValueError, match="abort-cause evidence"):
        audit.AuditCleanupResult(issues=(legacy_abort,))
    with pytest.raises(ValueError, match="does not match"):
        audit.AuditCleanupEntry(legacy_abort, local)
    with pytest.raises(ValueError, match="unknown.*role"):
        replace(local, failure_role="invalid")


def test_alignment_and_compact_annotation_keep_explicit_roles():
    cause, local = abort_failure(), file_failure()
    issues = (cause.issue(), local.issue(), local.issue())
    result = audit.AuditCleanupResult(
        issues=issues, aligned_failures=(cause, None, local), aborted=True
    )
    assert result.aligned_failures == (cause, None, local)
    with pytest.raises(ValueError, match="ambiguous"):
        audit.AuditCleanupResult(
            issues=issues, raw_failures=result.raw_failures, aborted=True
        )
    with pytest.raises(ValueError, match="does not match"):
        audit.AuditCleanupResult(issues=(local.issue(),), raw_failures=(cause,))
    assert (
        audit.AuditCleanupResult(
            issues=result.issues, aligned_failures=result.aligned_failures, aborted=True
        )
        == result
    )
    assert len(result.issues[2].detail) <= 400 and "\n" not in result.issues[2].detail
    assert result.raw_failures[1].raw_detail == local.raw_detail


@pytest.mark.parametrize("global_failure", [False, True])
def test_real_cleanup_producers_capture_roles_at_the_failure_site(
    tmp_path, monkeypatch, global_failure
):
    config, _, _ = setup(tmp_path)
    check = RetentionVerification(
        "suspended",
        NOW - timedelta(days=91),
        issues=(VerificationIssue(None, "unreadable", "old"),),
    )
    published = audit.publish_verification(config, check, dry_run=False)
    path = config.artifacts_dir / published.audit_path
    if global_failure:

        def fail(*args, **kwargs):
            raise ValueError("uncertain /private/reference")

        monkeypatch.setattr(audit, "_referenced_audits", fail)
    else:
        original = Path.unlink

        def unlink(target, *args, **kwargs):
            if target == path:
                raise PermissionError("denied /private/file")
            return original(target, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", unlink)
    result = audit.cleanup_audits_result(config, now=NOW, dry_run=False)
    assert result.aborted is global_failure
    assert result.completed is (not global_failure)
    assert result.raw_failures[0].failure_role == (
        "abort_cause" if global_failure else "file_local"
    )
    assert result.issues[0].category == (
        "audit_cleanup_aborted" if global_failure else "audit_cleanup_file_failed"
    )
    assert path.exists() and result.deleted_audit_count == 0


@pytest.mark.parametrize("stage", ["retention_verification", "retention_audit_cleanup"])
@pytest.mark.parametrize("count", [2, 23])
def test_authoritative_batches_preserve_occurrences_but_not_repeated_delivery(
    stage, count
):
    if stage == "retention_verification":
        check = RetentionVerification(
            "suspended",
            NOW,
            check_id="check",
            issues=(VerificationIssue(Path("/same"), "unreadable", "same"),) * count,
        )
        batch = check.diagnostics()
    else:
        failure = file_failure()
        entry = audit.AuditCleanupEntry(failure.issue(), failure)
        result = audit.AuditCleanupResult(
            entries=(entry,) * min(20, count),
            omitted_count=max(0, count - 20),
            completed=True,
            recorded_at=NOW,
        )
        batch = audit.AuditCleanupRecord(result, "check").diagnostics
    with diagnostic_progress() as progress:
        record_progress(batch)
        record_progress(batch)
        assert tuple(progress) == batch
    merged = merge_diagnostics(batch, batch, tuple(progress))
    assert merged == batch
    rows = json.loads(failure_details_json(merged))
    assert rows[0]["issue_count"] == count
    assert rows[0]["omitted_count"] == max(0, count - 20)
    assert len(rows[1:]) == min(20, count)
    assert all(row == rows[1] for row in rows[1:])
    if stage == "retention_verification":
        loaded = recorded(check)
        assert loaded.total == count and loaded.omitted == max(0, count - 20)
        assert len(loaded.issues) == min(20, count)


def test_full_audit_and_bounded_duplicate_samples_have_consistent_counts(tmp_path):
    config, _, _ = setup(tmp_path)
    issue = VerificationIssue(
        Path("/same"), "unreadable", "unreadable: genuine evidence"
    )
    check = audit.publish_verification(
        config,
        RetentionVerification("suspended", NOW, issues=(issue,) * 23),
        dry_run=True,
    )
    rows = [
        json.loads(line)
        for line in (config.artifacts_dir / check.audit_path).read_text().splitlines()
    ]
    assert rows[-1] == {"kind": "complete", "issue_count": 23}
    assert len(rows[1:-1]) == 23 and all(row == rows[1] for row in rows[1:-1])
    loaded = recorded(check)
    assert loaded.total == 23 and loaded.omitted == 3 and loaded.issues == (issue,) * 20


@pytest.mark.parametrize("state", ["clear", "suspended"])
@pytest.mark.parametrize("count", [0, 2, 23])
def test_progress_only_checks_roundtrip_summaries_and_duplicate_samples(state, count):
    check = RetentionVerification(
        state,
        NOW,
        progress_id="owner",
        dry_run=False,
        issues=(VerificationIssue(None, "unreadable", "same"),) * count,
    )
    for _ in range(3):
        rows = json.loads(failure_details_json(check.diagnostics()))
        assert rows[0]["issue_count"] == count
        assert all(
            row["progress_id"] == "owner" and "check_id" not in row for row in rows
        )
        loaded = recorded(check)
        assert loaded.progress_id == "owner" and loaded.check_id is None
        assert loaded.state == state and loaded.total == count
        assert len(loaded.issues) == min(20, count)
        assert loaded.omitted == max(0, count - 20)
        check = loaded


def test_distinct_progress_only_checks_at_equal_clocks_never_coalesce():
    first = RetentionVerification("clear", NOW, progress_id="A")
    second = RetentionVerification("suspended", NOW, progress_id="B")
    rows = {
        "failure_details_json": failure_details_json(
            first.diagnostics() + second.diagnostics()
        )
    }
    checks = _recorded_checks(rows, NOW)
    assert [(item.progress_id, item.state) for item in checks] == [
        ("A", "clear"),
        ("B", "suspended"),
    ]


@pytest.mark.parametrize("state", ["clear", "suspended"])
def test_old_summaryless_progress_samples_recover_identity_and_occurrences(state):
    check = RetentionVerification(
        state,
        NOW,
        progress_id="old",
        dry_run=False,
        issues=(VerificationIssue(None, "unreadable", "same"),) * 2,
    )
    old = tuple(item for item in check.diagnostics() if item.issue_count is None)
    row = run_row() | {"failure_details_json": failure_details_json(old)}
    loaded = retention_status(pl.DataFrame([row]), as_of=NOW).verification
    assert loaded.progress_id == "old" and loaded.check_id is None
    assert loaded.issues == check.issues and loaded.state == state
    assert loaded.total == 2 and loaded.omitted == 0


@pytest.mark.parametrize("key", ["progress_id", "check_id"])
@pytest.mark.parametrize("value", ["", " ", 7, [], {}])
def test_malformed_recorded_identity_cannot_establish_a_clear_check(key, value):
    rows = json.loads(
        failure_details_json(
            RetentionVerification("clear", NOW, check_id="valid").diagnostics()
        )
    )
    rows[0][key] = value
    row = run_row() | {"failure_details_json": json.dumps(rows)}
    assert (
        retention_status(pl.DataFrame([row]), as_of=NOW).verification.state
        == "not_recorded"
    )
    with pytest.raises(ValueError, match="identity"):
        effective_progress_id(
            value if key == "progress_id" else None,
            value if key == "check_id" else None,
        )


@pytest.mark.parametrize("conflict", ["different_owner", "different_check"])
def test_conflicting_identity_rows_cannot_establish_a_clear_check(conflict):
    check = RetentionVerification(
        "clear",
        NOW,
        check_id="check",
        progress_id="owner",
        issues=(VerificationIssue(None, "unreadable", "same"),),
    )
    rows = json.loads(failure_details_json(check.diagnostics()))
    rows[1]["progress_id" if conflict == "different_owner" else "check_id"] = "conflict"
    row = run_row() | {"failure_details_json": json.dumps(rows)}
    assert (
        retention_status(pl.DataFrame([row]), as_of=NOW).verification.state
        == "not_recorded"
    )


def test_conditional_identity_compression_preserves_renamed_batch_ownership():
    original = RetentionVerification(
        "suspended",
        NOW,
        check_id="A",
        issues=(VerificationIssue(None, "unreadable", "old"),) * 2,
    )
    compact = json.loads(failure_details_json(original.diagnostics()))
    assert all("progress_id" not in item for item in compact)
    restored = tuple(ComponentDiagnostic(**item) for item in compact)
    middle = replace(original, check_id="B")
    final = replace(
        middle, check_id="C", issues=(VerificationIssue(None, "unreadable", "new"),) * 2
    )
    independent = RetentionVerification("clear", NOW, check_id="other").diagnostics()
    with diagnostic_progress() as progress:
        record_progress(restored + independent)
        record_verification_progress(middle.diagnostics())
        record_progress(final.diagnostics())
    merged = merge_diagnostics(
        restored + independent, middle.diagnostics(), final.diagnostics()
    )
    assert merged == tuple(progress)
    assert [item.check_id for item in merged] == ["other", "C", "C", "C"]
    rows = json.loads(failure_details_json(merged))
    assert "progress_id" not in rows[0]
    assert all(item["progress_id"] == "A" for item in rows[1:])
    assert all(item.detail == "new" for item in merged[2:])
    loaded = recorded(final)
    assert (
        loaded.progress_id == "A" and loaded.check_id == "C" and len(loaded.issues) == 2
    )
    legacy = (ComponentDiagnostic("other", 0, detail="same"),) * 2
    assert merge_diagnostics(legacy, legacy) == legacy[:1]


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("late_failure", [False, True])
def test_duplicate_progress_only_batches_journal_once_through_late_failures(
    tmp_path, monkeypatch, capsys, embedded, late_failure
):
    config, _, _ = setup(tmp_path)
    prune = operations.prune_scores_files
    expected = []

    def changed(*args, **kwargs):
        result = prune(*args, **kwargs)
        check = replace(
            result.verification,
            check_id=None,
            state="suspended",
            issue_count=2,
            issues=(
                VerificationIssue(
                    Path("/private/check"), "unreadable", "private check evidence"
                ),
            )
            * 2,
        )
        failure = file_failure()
        cleanup = replace(
            result.audit_cleanup,
            issues=(failure.issue(),) * 2,
            aligned_failures=(failure, failure),
        )
        updated = replace(result, verification=check, audit_cleanup=cleanup)
        expected.append(updated)
        if late_failure:
            record_progress(check.diagnostics() + updated.audit_cleanup_diagnostics)
        return updated

    monkeypatch.setattr(operations, "prune_scores_files", changed)
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
    assert len(items) == 6
    assert all("check_id" not in item for item in items)
    assert {item["progress_id"] for item in items} == {
        expected[0].verification.progress_id
    }
    for stage in ("retention_verification", "retention_audit_cleanup"):
        batch = [item for item in items if item["stage"] == stage]
        assert len(batch) == 3 and batch[0]["issue_count"] == 2
        assert batch[1] == batch[2]
    assert ledger["exit_code"][0] == (None if late_failure else 0)
    loaded = retention_status(ledger, as_of=ledger["ended_at"][0]).verification
    assert loaded.progress_id == expected[0].verification.progress_id
    assert loaded.total == 2 and len(loaded.issues) == 2
    assert "private check evidence" not in capsys.readouterr().out
    if not late_failure:
        assert operations._run_counts(ledger, ledger["ended_at"][0]) == (0, 0, 0)
