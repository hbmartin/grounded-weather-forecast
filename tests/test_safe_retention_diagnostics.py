"""Safe public diagnostics preserve actionable operation and private raw evidence."""

import argparse
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from filelock import FileLock
from test_daily_read import NOW, START, setup, write_json
from test_retention_audits import verification
from test_retention_verification import run_row

from grounded_weather_forecast import cli, retention_audit as audit, runs
from grounded_weather_forecast.artifact_labels import CompatibilityNotice
from grounded_weather_forecast.dashboard.zones.operations import _retention_panel
from grounded_weather_forecast.reports import operations
from grounded_weather_forecast.retention_status import (
    RetentionVerification,
    retention_status,
)
from grounded_weather_forecast.serve.approvals import ApprovalIssue
from grounded_weather_forecast.status import diagnostic_progress, failure_details_json
from grounded_weather_forecast.verification_text import (
    VerificationIssue,
    public_detail,
    public_explanation,
)


PRIVATE_DETAILS = [
    "denied [/Users/private-user/Library/runs.parquet]",
    "denied,path=/Users/private-user/runs.parquet;",
    r"denied [C:\Users\private-user\runs.parquet]",
    r"denied [\\server\private-user\runs.parquet]",
    "denied [file:///Users/private-user/runs.parquet]",
    "denied [https://private-user.example/runs.parquet]",
]


def old_audit(config):
    check = audit.publish_verification(
        config, verification(NOW - timedelta(days=91)), dry_run=False
    )
    return config.artifacts_dir / check.audit_path


def test_sample_projection_normalizes_uneven_legacy_metadata_and_bounds():
    check = RetentionVerification(
        "suspended",
        NOW,
        (
            "/p: old: evidence includes /p: old: later",
            "legacy",
            "other",
            *("extra" for _ in range(22)),
        ),
        ("/p", "", None),
        categories=("old", ""),
        reason_codes=("future-code", ""),
    )
    samples = check.sample_issues()
    assert len(samples) == 20 and len(tuple(check.iter_issues())) == 25
    assert samples[0] == VerificationIssue(
        Path("/p"), "old", "evidence includes /p: old: later", "future-code"
    )
    assert samples[1] == VerificationIssue(None, "verification_invalid", "legacy")
    assert all(
        issue.path is None and issue.category == "verification_invalid"
        for issue in samples[2:]
    )
    data = json.loads(failure_details_json(check.diagnostics()))
    assert len(data) == 21 and data[2]["category"] == "verification_invalid"
    restored = retention_status(
        pl.DataFrame([run_row() | {"failure_details_json": json.dumps(data)}]),
        as_of=NOW,
    ).verification
    assert restored.sample_issues() == samples


def test_issue_conversion_preserves_normalized_identity_and_required_path():
    issue = VerificationIssue(None, "future-category", "raw")
    assert VerificationIssue.from_issue(issue) is issue
    assert VerificationIssue.from_issue(
        ApprovalIssue(Path("/p"), "unreadable", "raw")
    ) == replace(issue, path=Path("/p"), category="unreadable")


def test_report_deduplicates_explanations_without_losing_paths_or_samples():
    check = RetentionVerification(
        "suspended",
        NOW,
        ("raw",) * 4,
        ("/a", "/b", "/c", "/d"),
        categories=(
            "unreadable",
            "unreadable",
            "future-category",
            "another-future-category",
        ),
    )
    detail = check.frame()["detail"][0]
    assert detail.count("Artifact could not be read") == 1
    assert detail.count("Protection could not be verified") == 1
    assert (
        len(check.diagnostics()) == 4
        and check.frame()["affected_paths"][0] == "/a; /b; /c; /d"
    )


@pytest.mark.parametrize(
    "phase,expected",
    [
        ("write", "audit_io_failed"),
        ("link", "audit_link_failed"),
        ("sync", "audit_sync_failed"),
    ],
)
def test_io_timeouts_are_not_lock_contention(tmp_path, monkeypatch, phase, expected):
    config, _, _ = setup(tmp_path)

    def fail(*_a, **_k):
        raise TimeoutError("I/O stalled /Users/private-user/audit")

    if phase == "write":
        monkeypatch.setattr(audit, "_publish_audit", fail)
    elif phase == "link":
        monkeypatch.setattr(audit.os, "link", fail)
    else:
        monkeypatch.setattr(audit.os, "fsync", fail)
    check = audit.publish_verification(config, verification(), dry_run=True)
    assert check.audit_reason_code == expected and check.audit_candidate_path is None
    assert "private-user" not in check.frame()["audit_error"][0]


def test_actual_lock_contention_has_no_retry_claim_or_command_contention(tmp_path):
    config, _, _ = setup(tmp_path)
    directory = audit.audit_directory(config)
    directory.mkdir(parents=True)
    with FileLock(directory / ".audit.lock", timeout=0):
        check = audit.publish_verification(config, verification(), dry_run=False)
        cleanup = audit.cleanup_audits_result(config, now=NOW, dry_run=False)
    assert check.audit_reason_code == "audit_lock_contended"
    text = check.frame()["audit_error"][0]
    assert "no audit was published for this check" in text and "deferred" not in text
    assert "busy" in cleanup.public_notes(config)[0]
    assert all(item.exit_code == 0 for item in cleanup.diagnostics())


@pytest.mark.parametrize(
    "error,cause",
    [
        (PermissionError(1, "private EPERM"), "permission_denied"),
        (FileNotFoundError(2, "private ENOENT"), "not_found"),
        (OSError(95, "private unsupported links"), None),
    ],
)
def test_link_operation_survives_known_and_unknown_causes(
    tmp_path, monkeypatch, error, cause
):
    config, _, _ = setup(tmp_path)

    def fail(*_a, **_k):
        raise error

    monkeypatch.setattr(audit.os, "link", fail)
    check = audit.publish_verification(config, verification(), dry_run=False)
    assert (
        check.audit_reason_code == "audit_link_failed"
        and check.audit_cause_code == cause
    )
    assert "Exclusive audit publication failed" in check.frame()["audit_error"][0]
    data = json.loads(failure_details_json(check.diagnostics()))
    restored = retention_status(
        pl.DataFrame([run_row() | {"failure_details_json": json.dumps(data)}]),
        as_of=NOW,
    ).verification
    assert (
        restored.audit_reason_code == "audit_link_failed"
        and restored.audit_cause_code == cause
    )
    assert "private" not in restored.frame()["audit_error"][0]


@pytest.mark.parametrize("cause", [123, [], {}, "unknown-cause"])
def test_untrusted_audit_cause_metadata_is_not_public(cause):
    check = RetentionVerification(
        "suspended",
        NOW,
        ("raw",),
        check_id="check",
        audit_error="raw private error",
        audit_reason_code="audit_link_failed",
    )
    items = json.loads(failure_details_json(check.diagnostics()))
    items[0]["audit_cause_code"] = cause
    restored = retention_status(
        pl.DataFrame([run_row() | {"failure_details_json": json.dumps(items)}]),
        as_of=NOW,
    ).verification
    assert restored.frame()["audit_error"][0].startswith(
        "Audit publication: Exclusive audit"
    )
    assert "unknown-cause" not in restored.frame()["audit_error"][0]


@pytest.mark.parametrize(
    "case,reason",
    [
        ("missing", "release_missing"),
        ("unapproved", "release_unapproved"),
        ("malformed", "release_evidence_invalid"),
    ],
)
def test_reference_causes_remain_distinct_with_real_paths(tmp_path, case, reason):
    config, _, _ = setup(tmp_path)
    release = config.artifacts_dir / "releases/release1.json"
    raw = {
        "release_id": "release1",
        "promoted_at": START.isoformat(),
        "evaluation_ids": [],
    }
    if case == "missing":
        release.unlink()
    else:
        raw.update(
            {"approval_required": True}
            if case == "unapproved"
            else {"evaluation_ids": "bad"}
        )
        write_json(release, raw)
    write_json(config.artifacts_dir / "active_release.json", {"release_id": "release1"})
    result = operations.prune_scores_files(config, now=NOW, dry_run=True)
    (issue,) = result.verification.sample_issues()
    assert issue.reason_code == reason and issue.path == release
    assert result.verification.state == "suspended" and not result.deleted
    assert "private" not in public_explanation(
        category=issue.category, reason_code=reason
    )
    audit_rows = [
        json.loads(line)
        for line in (config.artifacts_dir / result.verification.audit_path)
        .read_text()
        .splitlines()
    ]
    assert audit_rows[1]["reason_code"] == reason and audit_rows[1]["path"] == str(
        release
    )


@pytest.mark.parametrize("raw", PRIVATE_DETAILS)
def test_cleanup_abort_keeps_raw_evidence_but_cli_is_safe(
    tmp_path, monkeypatch, capsys, raw
):
    config, _, _ = setup(tmp_path)
    old = old_audit(config)

    def fail(*_a):
        raise PermissionError(raw)

    monkeypatch.setattr(audit, "_referenced_audits", fail)
    result = operations.prune_scores_files(config, now=NOW, dry_run=True)
    assert result.audit_cleanup.aborted and old.exists()
    assert (
        result.audit_cleanup.issues[0].detail == raw
        and result.audit_cleanup.issues[0].path is None
    )
    assert raw not in str(result.skipped) and "Access denied" in str(result.skipped)
    outcome = cli._cmd_prune_scores(config, argparse.Namespace(dry_run=True))
    output = capsys.readouterr().out
    assert "private-user" not in output and str(tmp_path) not in output
    assert "Access denied" in output and outcome.exit_code == 0
    assert any(
        item.detail == raw and item.category == "audit_cleanup_aborted"
        for item in outcome.diagnostics
    )


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("late_failure", [False, True])
def test_cleanup_journals_once_without_failure_counter_or_lost_progress(
    tmp_path, monkeypatch, capsys, embedded, late_failure
):
    config, _, _ = setup(tmp_path)
    old_audit(config)

    def fail(*_a):
        raise PermissionError("private cleanup cause")

    monkeypatch.setattr(audit, "_referenced_audits", fail)
    if late_failure:

        def output_failure(*_a, **_k):
            raise OSError("later output failure")

        monkeypatch.setattr(cli, "_print_score_temporaries", output_failure)

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
    summaries = [
        i
        for i in items
        if i.get("stage") == "retention_audit_cleanup" and "issue_count" in i
    ]
    assert len(summaries) == 1 and summaries[0]["exit_code"] == 0
    assert any(i.get("detail") == "private cleanup cause" for i in items)
    if not late_failure:
        assert operations._run_counts(ledger, ledger["ended_at"][0]) == (0, 0, 0)
    assert "private cleanup cause" not in capsys.readouterr().out


@pytest.mark.parametrize("dry_run", [False, True])
def test_cleanup_retains_abort_and_omitted_count_with_full_sample_table(
    tmp_path, monkeypatch, dry_run
):
    config, _, _ = setup(tmp_path)
    old = old_audit(config)
    for index in range(25):
        (old.parent / f"{index:032x}.jsonl").write_text("{bad")

    def uncertain(*_a):
        raise ValueError("cannot establish private reference")

    monkeypatch.setattr(audit, "_referenced_audits", uncertain)
    result = audit.cleanup_audits_result(config, now=NOW, dry_run=dry_run)
    notes = result.public_notes(config)
    assert len(notes) == 20 and result.aborted and result.omitted_count == 7
    assert result.issues[0].category == "audit_cleanup_aborted"
    assert notes[-1] == "7 additional audit cleanup issues omitted."
    assert all("private" not in note for note in notes) and old.exists()


def test_compatibility_notices_are_structured_and_legacy_notes_are_informational(
    tmp_path,
):
    config, _, _ = setup(tmp_path)
    note = CompatibilityNotice(config.artifacts_dir / "releases/legacy.json")
    common = {
        "config": config,
        "runs": pl.DataFrame([run_row()]),
        "now": NOW,
        "compatibility_notes": ("raw /Users/private-user/path",) * 25,
    }
    legacy = _retention_panel(SimpleNamespace(**common))
    modern = _retention_panel(SimpleNamespace(**common, compatibility_notices=(note,)))
    assert legacy.status == modern.status == "ok"
    assert (
        "20 legacy compatibility notes" in legacy.intro
        and "5 additional" in legacy.intro
    )
    assert "Protection could not be verified" not in legacy.intro
    assert (
        "artifacts/releases/legacy.json" in modern.intro
        and "retention only" in modern.intro
    )
    assert "private-user" not in str(legacy) + str(modern)
    assert public_detail("raw /Users/private-user/path") == public_explanation()


def test_legacy_full_audit_is_not_truncated_to_public_sample_limit(tmp_path):
    config, _, _ = setup(tmp_path)
    check = audit.publish_verification(config, verification(count=25), dry_run=True)
    assert check.total == 25 and len(check.sample_issues()) == 20 and check.omitted == 5
    lines = (config.artifacts_dir / check.audit_path).read_text().splitlines()
    assert len(lines) == 27 and json.loads(lines[-1])["issue_count"] == 25


def test_normalized_evidence_prefix_is_not_removed_again_on_journal_read(tmp_path):
    config, _, _ = setup(tmp_path)
    path = config.artifacts_dir / "releases/example.json"
    raw = f"{path}: unreadable: actual recorded evidence"
    check = audit.publish_verification(
        config,
        RetentionVerification("suspended", NOW),
        dry_run=True,
        issues=(VerificationIssue(path, "unreadable", raw),),
    )
    data = failure_details_json(check.diagnostics())
    for _ in range(3):
        loaded = retention_status(
            pl.DataFrame([run_row() | {"failure_details_json": data}]), as_of=NOW
        ).verification
        assert loaded.sample_issues()[0].detail == raw
        data = failure_details_json(loaded.diagnostics())
    republished = audit.publish_verification(config, loaded, dry_run=True)
    lines = [
        json.loads(line)
        for line in (config.artifacts_dir / republished.audit_path)
        .read_text()
        .splitlines()
    ]
    assert lines[1]["detail"] == raw
