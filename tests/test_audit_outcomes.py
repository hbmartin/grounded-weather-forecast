"""Audit checkpoints and immutable evidence keep partial work accurately attributed."""

import argparse
import copy
import json
import os
import pickle
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest
from filelock import FileLock
from test_daily_read import NOW, setup
from test_retention_audits import verification
from test_retention_verification import run_row
from test_safe_retention_diagnostics import old_audit

from grounded_weather_forecast import cli, retention_audit as audit, runs
from grounded_weather_forecast.artifact_labels import CompatibilityNotice
from grounded_weather_forecast.dashboard.context import (
    DashboardContext,
)
from grounded_weather_forecast.dashboard.zones.operations import _retention_panel
from grounded_weather_forecast.reports import operations
from grounded_weather_forecast.retention_status import (
    RetentionVerification,
    retention_status,
)
from grounded_weather_forecast.status import (
    failure_details_json,
    merge_diagnostics,
)
from grounded_weather_forecast.verification_text import VerificationIssue


def staging_file(config):
    path = audit.audit_directory(config) / (
        "." + "e" * 32 + ".jsonl." + "f" * 16 + ".tmp"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("interrupted writer")
    old = (NOW - timedelta(days=91)).timestamp()
    os.utime(path, (old, old))
    return path


@pytest.mark.parametrize("dry_run", [False, True])
def test_unlock_failure_preserves_completed_deletions_and_operation(
    tmp_path, monkeypatch, dry_run
):
    config, _, _ = setup(tmp_path)
    committed, staging = old_audit(config), staging_file(config)
    release = FileLock.release

    def release_then_fail(lock, *args, **kwargs):
        release(lock, *args, **kwargs)
        raise PermissionError("private lock release error")

    with monkeypatch.context() as patch:
        patch.setattr(FileLock, "release", release_then_fail)
        result = audit.cleanup_audits_result(config, now=NOW, dry_run=dry_run)
    assert result.aborted
    assert committed.exists() is dry_run and staging.exists() is dry_run
    expected = 0 if dry_run else 1
    assert result.deleted_audit_count == result.deleted_staging_count == expected
    assert result.issues[0].reason_code == "audit_cleanup_failed"
    assert result.issues[0].path == committed.parent
    items = result.diagnostics(check_id="check")
    assert items[0].dry_run is dry_run
    assert items[0].deleted_audit_count == items[0].deleted_staging_count == expected
    assert (
        items[1].operation == "lock_release"
        and items[1].audit_cause_code == "permission_denied"
    )
    assert (
        f"Deleted {expected} audits and {expected} staging files." in result.summary()
    )
    assert "publication is unconfirmed" not in " ".join(result.public_notes(config))
    assert "private lock" not in " ".join(result.public_notes(config))
    assert all(item.exit_code == 0 for item in items)


@pytest.mark.parametrize("dry_run", [False, True])
def test_clean_cleanup_has_one_summary_with_known_mode_and_counts(tmp_path, dry_run):
    config, _, _ = setup(tmp_path)
    result = audit.cleanup_audits_result(config, now=NOW, dry_run=dry_run)
    assert not result.issues and result.public_notes(config) == ()
    (summary,) = result.diagnostics(check_id="check")
    assert summary.failure_kind == "audit_cleanup_complete"
    assert summary.issue_count == 0 and summary.dry_run is dry_run
    assert summary.deleted_audit_count == summary.deleted_staging_count == 0
    legacy = audit.AuditCleanupResult()
    assert legacy.diagnostics()[0].dry_run is None
    assert legacy.diagnostics()[0].deleted_audit_count is None


@pytest.mark.parametrize(
    "phase,operation,published",
    [
        ("file_sync", "staging_sync", False),
        ("directory_sync", "directory_sync", True),
        ("link", "link", False),
        ("write", "staging_write", False),
    ],
)
def test_publication_phase_roundtrips_and_all_renderers_agree(
    tmp_path, monkeypatch, capsys, phase, operation, published
):
    config, _, _ = setup(tmp_path)

    def fail(*args, **kwargs):
        raise TimeoutError("private timeout [/Users/private/location]")

    targets = {
        "file_sync": (audit.os, "fsync"),
        "directory_sync": (audit, "_sync_directory"),
        "link": (audit.os, "link"),
        "write": (audit, "_publish_audit"),
    }
    target, attribute = targets[phase]
    monkeypatch.setattr(target, attribute, fail)
    check = audit.publish_verification(config, verification(), dry_run=False)
    assert check.audit_operation == operation
    assert bool(check.audit_candidate_path) is published
    assert check.audit_path is None
    assert len(list(audit.audit_directory(config).glob("*.jsonl"))) == int(published)
    assert not list(audit.audit_directory(config).glob("*.tmp"))
    explanation = check.audit_explanation()
    expected = (
        "linked audit remains unconfirmed" if published else "No audit was published"
    )
    assert expected in explanation and "private" not in explanation
    assert "lock" not in check.audit_reason_code
    row = run_row() | {
        "failure_details_json": failure_details_json(check.diagnostics())
    }
    loaded = retention_status(pl.DataFrame([row]), as_of=NOW).verification
    assert (
        loaded.audit_explanation() == explanation
        and loaded.audit_operation == operation
    )
    assert loaded.frame()["audit_error"][0] == explanation
    panel = _retention_panel(DashboardContext(config, NOW, runs=pl.DataFrame([row])))
    assert explanation in panel.intro
    result = operations.PruneResult((), (), (), 0.0, verification=check)
    monkeypatch.setattr(operations, "prune_scores_files", lambda *_a, **_k: result)
    cli._cmd_prune_scores(config, argparse.Namespace(dry_run=False))
    assert explanation in capsys.readouterr().out


def test_legacy_sync_failure_does_not_invent_publication_phase():
    check = RetentionVerification(
        "suspended",
        NOW,
        audit_error="legacy raw error",
        audit_reason_code="audit_sync_failed",
    )
    assert "No confirmed audit reference is available" in check.audit_explanation()
    assert "No audit was published" not in check.audit_explanation()
    assert "linked audit" not in check.audit_explanation()


@pytest.mark.parametrize("phase", ["references", "enumerate"])
def test_cleanup_failure_path_comes_from_operation_not_mutable_state(
    tmp_path, monkeypatch, phase
):
    config, _, _ = setup(tmp_path)
    old = old_audit(config)
    target, attribute = (
        (audit, "_referenced_audits")
        if phase == "references"
        else (audit.os, "scandir")
    )

    def fail(*args, **kwargs):
        raise ValueError("unknown private source")

    monkeypatch.setattr(target, attribute, fail)
    result = audit.cleanup_audits_result(config, now=NOW, dry_run=False)
    assert old.exists() and result.aborted
    assert result.issues[0].path == (old.parent if phase == "enumerate" else None)
    assert result.diagnostics()[1].operation == phase


def test_unrelated_file_failure_keeps_bounded_view_and_full_raw_evidence(
    tmp_path, monkeypatch
):
    config, _, _ = setup(tmp_path)
    first, second = sorted([old_audit(config), old_audit(config)])
    unlink = Path.unlink
    raw = "denied\n" + r"C:\Users\private\file" * 200

    def fail(path, *args, **kwargs):
        if path == first:
            raise PermissionError(raw)
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail)
    result = audit.cleanup_audits_result(config, now=NOW, dry_run=False)
    assert first.exists() and not second.exists() and not result.aborted
    assert result.deleted_audit_count == 1
    assert result.raw_failures[0].raw_detail == raw
    issue = result.issues[0]
    assert len(issue.detail) == 400 and "\n" not in issue.detail
    assert issue.path == first and result.diagnostics()[1].operation == "delete"
    assert "private" not in " ".join(result.public_notes(config))


def test_canonical_issues_equal_replace_and_reconstruct_without_flags():
    path = Path("/p")
    legacy = RetentionVerification(
        "suspended",
        NOW,
        ("/p: unreadable: evidence",),
        ("/p",),
        categories=("unreadable",),
    )
    typed = RetentionVerification(
        "suspended", NOW, issues=(VerificationIssue(path, "unreadable", "evidence"),)
    )
    assert legacy == typed and legacy.sample_issues() == typed.sample_issues()
    assert not hasattr(typed, "_details_normalized")
    metadata = replace(typed, check_id="check")
    assert metadata.issues is typed.issues
    changed = replace(metadata, details=("/p: unreadable: replacement",))
    assert changed.sample_issues() == (
        VerificationIssue(path, "unreadable", "replacement"),
    )
    raw = "/p: unreadable: actual raw evidence"
    replacement = replace(typed, issues=(VerificationIssue(path, "unreadable", raw),))
    assert replacement.sample_issues()[0].detail == raw
    for reconstructed in (
        copy.copy(replacement),
        pickle.loads(pickle.dumps(replacement)),
    ):
        assert (
            reconstructed == replacement
            and reconstructed.sample_issues() == replacement.sample_issues()
        )
    assert replace(typed, reason_codes=("",)).reason_codes == (None,)
    assert replace(typed, paths=("",)).paths == (None,)
    assert replace(typed, categories=("",)).categories == ("verification_invalid",)


@pytest.mark.parametrize("reason", ["", None, "future-code"])
def test_journal_reason_ingestion_is_canonical(reason):
    item = {
        "stage": "retention_verification",
        "exit_code": 0,
        "failure_kind": "retention_suspended",
        "detail": "raw",
        "category": "",
        "path": "",
        "reason_code": reason,
        "recorded_at": NOW.isoformat(),
    }
    row = run_row() | {"failure_details_json": json.dumps([item])}
    check = retention_status(pl.DataFrame([row]), as_of=NOW).verification
    assert check.reason_codes == (reason or None,)
    assert check.categories == ("verification_invalid",) and check.paths == (None,)


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("late_failure", [False, True])
def test_cleanup_batch_constructed_once_and_survives_output_failure(
    tmp_path, monkeypatch, capsys, embedded, late_failure
):
    config, _, _ = setup(tmp_path)
    counts = []
    original = audit.AuditCleanupResult.diagnostics

    def counted(self, **kwargs):
        counts.append(self)
        return original(self, **kwargs)

    monkeypatch.setattr(audit.AuditCleanupResult, "diagnostics", counted)
    if late_failure:

        def fail(*args, **kwargs):
            raise OSError("later output interruption")

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
    assert len(counts) == 1
    ledger = runs.read_runs(runs.runs_path(config))
    assert ledger.height == 1
    items = json.loads(ledger["failure_details_json"][0])
    summaries = [
        i
        for i in items
        if i["stage"] == "retention_audit_cleanup" and "issue_count" in i
    ]
    assert len(summaries) == 1 and summaries[0]["dry_run"] is True
    assert (
        summaries[0]["deleted_audit_count"]
        == summaries[0]["deleted_staging_count"]
        == 0
    )
    assert summaries[0]["exit_code"] == 0
    if not late_failure:
        assert operations._run_counts(ledger, ledger["ended_at"][0]) == (0, 0, 0)
    capsys.readouterr()


def test_shared_batch_is_reused_by_outcome_and_progress(tmp_path, monkeypatch):
    config, _, _ = setup(tmp_path)
    result = operations.prune_scores_files(config, now=NOW, dry_run=True)
    monkeypatch.setattr(operations, "prune_scores_files", lambda *_a, **_k: result)
    outcome = cli._cmd_prune_scores(config, argparse.Namespace(dry_run=True))
    assert outcome.diagnostics[-1] is result.audit_cleanup_diagnostics[0]
    assert (
        merge_diagnostics(result.audit_cleanup_diagnostics, outcome.diagnostics).count(
            result.audit_cleanup_diagnostics[0]
        )
        == 1
    )


def test_structured_notices_derive_legacy_view_and_keep_separate_limits(tmp_path):
    config, _, _ = setup(tmp_path)
    notices = tuple(
        CompatibilityNotice(config.artifacts_dir / f"releases/{i}.json")
        for i in range(100)
    )
    ctx = DashboardContext(
        config,
        NOW,
        compatibility_notes=("stale string",),
        compatibility_notices=notices,
    )
    assert ctx.compatibility_notes == tuple(notice.label(config) for notice in notices)
    updated = replace(ctx, compatibility_notices=notices[:1])
    assert updated.compatibility_notes == (notices[0].label(config),)
    modern = _retention_panel(ctx)
    assert "80 additional compatibility notes omitted" in modern.intro
    legacy = _retention_panel(
        DashboardContext(config, NOW, compatibility_notes=("private raw",) * 100)
    )
    assert "100 legacy compatibility notes" in legacy.intro
    assert "omitted" not in legacy.intro and "private" not in legacy.intro
    assert (
        CompatibilityNotice(None, "unknown")
        .label(config)
        .startswith("Legacy compatibility information")
    )


def test_operation_error_copy_preserves_context_without_fake_filename():
    error = audit._AuditPublicationError(
        "audit_cleanup_failed", PermissionError("denied"), "lock_release", Path("/p")
    )
    assert error.filename is None
    for reconstructed in (copy.copy(error), pickle.loads(pickle.dumps(error))):
        assert reconstructed.operation == "lock_release" and reconstructed.path == Path(
            "/p"
        )
        assert reconstructed.reason_code == "audit_cleanup_failed"
