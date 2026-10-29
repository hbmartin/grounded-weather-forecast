"""Failure compatibility preserves attribution; copied checks own one journal batch."""

import argparse
import copy
import json
import pickle
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from test_bound_retention_diagnostics import cleanup, failures
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
    record_verification_progress,
)
from grounded_weather_forecast.verification_text import VerificationIssue


@pytest.mark.parametrize("replacement_indexes", [(0,), (2,), (1, 2)])
def test_failure_only_replacement_rebuilds_and_truncates(replacement_indexes):
    raw = failures()
    result = audit.AuditCleanupResult(raw_failures=raw, completed=True)
    selected = tuple(raw[index] for index in replacement_indexes)
    changed = replace(result, raw_failures=selected)
    assert changed.raw_failures == selected
    assert changed.issues == tuple(item.issue() for item in selected)
    assert len(changed.entries) == len(selected)
    assert result.raw_failures == raw


def test_failure_reconstruction_preserves_roles_independently_of_outcome():
    raw = tuple(replace(item, failure_role="file_local") for item in failures())
    result = audit.AuditCleanupResult(raw_failures=raw, aborted=True, completed=False)
    changed = replace(result, aborted=False, completed=True, raw_failures=raw[:1])
    assert changed.outcome is audit.CleanupOutcome.COMPLETED
    assert changed.issues[0].category == "audit_cleanup_file_failed"
    again = replace(changed, aborted=True, completed=False, raw_failures=raw[1:])
    assert again.outcome is audit.CleanupOutcome.INCOMPLETE
    assert again.issues[0].category == "audit_cleanup_file_failed"
    assert again.issues[1].category == "audit_cleanup_file_failed"
    with pytest.raises(ValueError, match="cannot be completed"):
        replace(changed, aborted=True)


@pytest.mark.parametrize("positions", [(1,), (2,), (0, 2), (1, 2), ()])
def test_sparse_compact_views_roundtrip_when_association_is_unique(positions):
    raw = failures()
    entries = tuple(
        audit.AuditCleanupEntry(item.issue(), item if index in positions else None)
        for index, item in enumerate(raw)
    )
    result = audit.AuditCleanupResult(entries=entries, completed=True)
    rebuilt = audit.AuditCleanupResult(
        issues=result.issues, raw_failures=result.raw_failures, completed=True
    )
    assert rebuilt == result
    assert rebuilt.aligned_failures == result.aligned_failures
    assert result.raw_failures == tuple(raw[index] for index in positions)


def test_compact_association_requires_unique_order_preserving_match():
    first, second, _ = failures()
    with pytest.raises(ValueError, match="ambiguous"):
        audit.AuditCleanupResult(
            issues=(first.issue(), first.issue()), raw_failures=(first,)
        )
    with pytest.raises(ValueError, match="does not match"):
        audit.AuditCleanupResult(
            issues=(first.issue(), second.issue()), raw_failures=(second, first)
        )
    # Matching all duplicate positions in order is unambiguous.
    both = audit.AuditCleanupResult(
        issues=(first.issue(), first.issue()), raw_failures=(first, first)
    )
    assert both.aligned_failures == (first, first)


@pytest.mark.parametrize("slots", [(None, 0), (0, None), (None, None)])
def test_aligned_view_handles_duplicate_issues_and_immutable_ingestion(slots):
    first = failures()[0]
    supplied = [first if value == 0 else None for value in slots]
    result = audit.AuditCleanupResult(
        issues=[first.issue(), first.issue()], aligned_failures=supplied, completed=True
    )
    original = tuple(supplied)
    supplied.clear()
    assert result.aligned_failures == original
    rebuilt = audit.AuditCleanupResult(
        issues=result.issues, aligned_failures=result.aligned_failures, completed=True
    )
    assert rebuilt == result == pickle.loads(pickle.dumps(result)) == copy.copy(result)
    assert replace(result, dry_run=True).entries is result.entries
    assert replace(result, issues=()).raw_failures == ()
    assert replace(result, aligned_failures=(None, None)).raw_failures == ()


def test_aligned_annotation_rejects_conflicts_and_incorrect_lengths():
    first, second, _ = failures()
    result = audit.AuditCleanupResult(raw_failures=(first, second), completed=True)
    annotated = replace(result, aligned_failures=(None, second), raw_failures=())
    assert annotated.issues == result.issues
    assert annotated.aligned_failures == (None, second)
    for kwargs, message in [
        ({"aligned_failures": ()}, "issue count"),
        ({"aligned_failures": (second, None)}, "does not match"),
        (
            {"aligned_failures": (None, second), "raw_failures": (second,)},
            "cannot be combined",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            replace(result, **kwargs)


@pytest.mark.parametrize("raw", [None, ()])
def test_empty_failure_input_preserves_canonical_entries(raw):
    first = failures()[0]
    entries = (audit.AuditCleanupEntry(first.issue(), first),)
    result = audit.AuditCleanupResult(entries=entries, raw_failures=raw)
    assert result.entries is entries
    assert replace(result, raw_failures=raw).entries is entries
    assert replace(result, entries=()).entries == ()


@pytest.mark.parametrize("key", ["paths", "categories", "reason_codes"])
def test_legacy_adapter_accepts_nullable_metadata(key):
    result = RetentionVerification.from_legacy_details(
        "suspended", NOW, ("genuine: evidence",), **{key: None}
    )
    assert result.details == ("genuine: evidence",)
    assert result.paths == (None,) and result.categories == ("verification_invalid",)


def test_legacy_adapter_has_typed_constructor_compatible_arguments():
    typed = (VerificationIssue(Path("/p"), "unreadable", "/p: unreadable: raw"),)
    retained = RetentionVerification.from_legacy_details(
        "suspended", NOW, None, None, "check", issues=typed
    )
    assert retained.issues is typed and retained.details == ("/p: unreadable: raw",)
    normalized = RetentionVerification.from_legacy_details(
        "suspended",
        NOW,
        ("/p: unreadable: /p: unreadable: genuine",),
        None,
        "check",
        "confirmed.jsonl",
        1,
        "audit evidence",
        None,
        True,
        None,
        "candidate.jsonl",
        "future-reason",
        "future-cause",
        issues=typed,
        audit_operation="sync",
        progress_id="origin",
    )
    assert normalized.details == ("/p: unreadable: genuine",)
    assert normalized.paths == ("/p",) and normalized.categories == ("unreadable",)
    assert normalized.check_id == "check" and normalized.progress_id == "origin"
    assert normalized.audit_path == "confirmed.jsonl" and normalized.dry_run is True
    assert normalized.audit_candidate_path == "candidate.jsonl"
    assert normalized.audit_reason_code == "future-reason"
    assert (
        normalized.audit_cause_code == "future-cause"
        and normalized.audit_operation == "sync"
    )
    assert replace(normalized, details=normalized.details) == normalized
    assert RetentionVerification.from_legacy_details(details=()).issues == ()


@pytest.mark.parametrize(
    "before_state,after_state", [("clear", "suspended"), ("suspended", "clear")]
)
def test_authoritative_verification_batch_replaces_summary_and_samples(
    before_state, after_state
):
    before = RetentionVerification(
        before_state,
        NOW,
        check_id="same",
        issue_count=1,
        audit_error="old",
        issues=(VerificationIssue(Path("/old"), "unreadable", "old evidence"),),
    )
    after = replace(
        before,
        state=after_state,
        issue_count=2,
        audit_error="new",
        issues=(VerificationIssue(Path("/new"), "unreadable", "new evidence"),),
    )
    unrelated = (ComponentDiagnostic("backtest", 1, "actual_error"),)
    other = RetentionVerification("clear", NOW, check_id="other").diagnostics()
    with diagnostic_progress() as progress:
        record_verification_progress(before.diagnostics() + unrelated + other)
        record_progress(after.diagnostics())
        assert tuple(progress) == unrelated + other + after.diagnostics()
    merged = merge_diagnostics(
        before.diagnostics() + unrelated + other, after.diagnostics()
    )
    assert merged == tuple(progress)
    row = run_row() | {"failure_details_json": failure_details_json(merged)}
    loaded = retention_status(pl.DataFrame([row]), as_of=NOW).verification
    assert loaded.state == after_state and loaded.total == 2
    assert loaded.paths == ("/new",) and loaded.audit_error == "new"


def test_copied_identity_supersedes_its_lineage_but_preserves_independent_checks():
    check = RetentionVerification("clear", NOW, check_id="original")
    result = operations.PruneResult(
        (), (), (), 0.0, verification=check, audit_cleanup=cleanup(True)
    )
    final = replace(
        result, verification=replace(result.verification, check_id="renamed")
    )
    final = replace(final, verification=replace(final.verification, check_id="last"))
    assert (
        final.verification.progress_id == final.cleanup_record.progress_id == "original"
    )
    assert result.verification.check_id == "original"
    independent = RetentionVerification(
        "clear", NOW, check_id="independent"
    ).diagnostics()
    with diagnostic_progress() as progress:
        record_progress(
            result.verification.diagnostics()
            + result.audit_cleanup_diagnostics
            + independent
        )
        record_verification_progress(final.verification.diagnostics())
        record_progress(final.audit_cleanup_diagnostics)
        assert {item.check_id for item in progress} == {"last", "independent"}
    merged = merge_diagnostics(
        result.verification.diagnostics()
        + result.audit_cleanup_diagnostics
        + independent,
        final.verification.diagnostics() + final.audit_cleanup_diagnostics,
    )
    assert merged == tuple(progress)
    assert (
        sum(
            item.issue_count is not None and item.stage == "retention_audit_cleanup"
            for item in merged
        )
        == 1
    )
    assert replace(final, kept=(Path("/score"),)).cleanup_record is final.cleanup_record
    assert pickle.loads(pickle.dumps(final)) == copy.copy(final) == final


def test_shared_replacement_accepts_legacy_check_ids_and_preserves_unknowns():
    old = (
        ComponentDiagnostic(
            "retention_verification", 0, "retention_clear", check_id="legacy"
        ),
    )
    new = RetentionVerification("suspended", NOW, check_id="legacy").diagnostics()
    assert merge_diagnostics(old, new) == new
    unknown = (ComponentDiagnostic("retention_verification", 0, "retention_clear"),)
    changed_unknown = (replace(unknown[0], failure_kind="retention_suspended"),)
    assert (
        merge_diagnostics(unknown, changed_unknown, unknown)
        == unknown + changed_unknown
    )


def test_progress_identity_journal_roundtrip_and_independent_publications(tmp_path):
    config, _, _ = setup(tmp_path)
    seed = RetentionVerification("clear", NOW, check_id="seed")
    with diagnostic_progress() as progress:
        first = audit.publish_verification(config, seed, dry_run=True)
        second = audit.publish_verification(config, seed, dry_run=True)
    assert first.progress_id == first.check_id != second.check_id == second.progress_id
    assert {item.check_id for item in progress} == {first.check_id, second.check_id}
    renamed = replace(second, check_id="changed")
    for _ in range(3):
        serialized = failure_details_json(renamed.diagnostics())
        assert json.loads(serialized)[0]["progress_id"] == second.progress_id
        row = run_row() | {"failure_details_json": serialized}
        loaded = retention_status(pl.DataFrame([row]), as_of=NOW).verification
        assert loaded == renamed
        renamed = loaded
    assert "progress_id" not in runs.RUNS_SCHEMA


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("late_failure", [False, True])
def test_rebound_verification_and_cleanup_journal_once(
    tmp_path, monkeypatch, capsys, embedded, late_failure
):
    config, _, _ = setup(tmp_path)
    prune = operations.prune_scores_files
    expected = []

    def updated(*args, **kwargs):
        original = prune(*args, **kwargs)
        check = replace(
            original.verification,
            check_id="renamed",
            state="suspended",
            issue_count=1,
            issues=(
                VerificationIssue(
                    Path("/private/new"), "unreadable", "unreadable: genuine evidence"
                ),
            ),
        )
        changed = replace(
            original,
            verification=check,
            audit_cleanup=replace(
                original.audit_cleanup, aborted=True, completed=False
            ),
        )
        expected.append(changed)
        if late_failure:
            record_progress(
                changed.verification.diagnostics() + changed.audit_cleanup_diagnostics
            )
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
    assert {item["check_id"] for item in items} == {"renamed"}
    assert {item["progress_id"] for item in items} == {
        expected[0].verification.progress_id
    }
    summaries = [item for item in items if "issue_count" in item]
    assert len(summaries) == 2
    assert {item["failure_kind"] for item in summaries} == {
        "retention_suspended",
        "audit_cleanup_aborted",
    }
    assert all(item["exit_code"] == 0 and item["dry_run"] is True for item in items)
    assert ledger["exit_code"][0] is (None if late_failure else 0)
    assert "unreadable: genuine evidence" not in capsys.readouterr().out


def test_empty_legacy_batches_are_omitted_without_reconstructing_bound_diagnostics(
    monkeypatch,
):
    calls = []
    original = audit.AuditCleanupResult.diagnostics

    def counted(self, **kwargs):
        calls.append(self)
        return original(self, **kwargs)

    monkeypatch.setattr(audit.AuditCleanupResult, "diagnostics", counted)
    record = audit.AuditCleanupRecord(cleanup(True), "known")
    result = operations.PruneResult(
        (), (), (), 0.0, cleanup_record=record, audit_cleanup_diagnostics=()
    )
    assert result.cleanup_record is record and len(calls) == 1
    assert (
        replace(result, skipped=("kept",), audit_cleanup_diagnostics=()).cleanup_record
        is record
    )
    assert len(calls) == 1
