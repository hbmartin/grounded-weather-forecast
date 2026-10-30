"""Optional journal damage cannot erase commands or unrelated protection checks."""

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest
from test_daily_read import NOW, setup
from test_retention_verification import run_row

from grounded_weather_forecast import cli, runs, status
from grounded_weather_forecast.dashboard.context import collect_context
from grounded_weather_forecast.dashboard.zones.operations import _retention_panel
from grounded_weather_forecast.retention_audit import (
    AuditCleanupEntry,
    AuditCleanupFailure,
    AuditCleanupResult,
)
from grounded_weather_forecast.retention_status import (
    RetentionVerification,
    _recorded_check_assessment,
    _recorded_checks,
    retention_status,
)
from grounded_weather_forecast.status import (
    CommandOutcome,
    ComponentDiagnostic,
    diagnostic_progress,
    failure_details_json,
    merge_diagnostics,
    record_progress,
)
from grounded_weather_forecast.verification_text import VerificationIssue


def check(state="clear", *, name="check", when=NOW, count=0, dry_run=False):
    return RetentionVerification(
        state,
        when,
        check_id=name,
        dry_run=dry_run,
        issues=(VerificationIssue(None, "unreadable", "literal: private evidence"),)
        * count,
    )


def payload(value):
    return json.loads(failure_details_json(value.diagnostics()))


def row(items, *, when=NOW, exit_code=0):
    return run_row(when=when) | {
        "failure_details_json": json.dumps(items),
        "exit_code": exit_code,
    }


def damaged_items():
    items = payload(check(name="damaged"))
    items[0]["progress_id"] = []
    return items


@pytest.mark.parametrize("key", ["check_id", "progress_id"])
@pytest.mark.parametrize("value", ["", " ", 7, False, [], {}, ["x" * 1000]])
def test_invalid_identity_remains_untrusted_without_breaking_journal(key, value):
    invalid = replace(
        ComponentDiagnostic("retention_verification", 0, "retention_clear"),
        **{key: value},
    )
    valid = check("suspended", name="independent", count=2).diagnostics()
    with diagnostic_progress() as progress:
        record_progress(valid + (invalid,))
    merged = merge_diagnostics(tuple(progress), (invalid,))
    serialized = failure_details_json(merged)
    decoded = json.loads(serialized)
    damaged = [item for item in decoded if isinstance(item.get(key), dict)]
    assert damaged and len(damaged[0][key]["detail"]) <= 400
    recovered = _recorded_check_assessment(row(decoded), NOW)
    assert [item.check_id for item in recovered.checks] == ["independent"]
    assert recovered.invalid
    with pytest.raises(ValueError, match="identity"):
        status.effective_progress_id(
            value if key == "progress_id" else None,
            value if key == "check_id" else None,
        )


class Unrepresentable:
    def __repr__(self):
        raise ValueError("private repr error")


def test_unrepresentable_identity_is_bounded_and_keeps_run_evidence():
    value = ComponentDiagnostic("other", 0, check_id=Unrepresentable())
    assert merge_diagnostics((value,)) == (value,)
    encoded = json.loads(failure_details_json((value,)))[0]
    assert encoded["check_id"]["invalid_identity_type"] == "Unrepresentable"
    assert "private" not in encoded["check_id"]["detail"]


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("outcome", [0, 1, OSError, KeyboardInterrupt])
def test_bad_metadata_preserves_return_exception_and_one_journal_row(
    tmp_path, monkeypatch, embedded, outcome
):
    config, _, _ = setup(tmp_path)
    invalid = ComponentDiagnostic(
        "retention_verification", 0, "retention_clear", check_id=[], progress_id="owner"
    )
    independent = check("suspended", name="independent", count=2).diagnostics()

    def invoke_stage(*args, **kwargs):
        record_progress(independent + (invalid,))
        if isinstance(outcome, type):
            raise outcome("original command outcome")
        return CommandOutcome(outcome, diagnostics=independent + (invalid,))

    monkeypatch.setattr(cli, "_dispatch", invoke_stage)

    def invoke():
        if embedded:
            return cli._run_pipeline_steps(
                config,
                cli.argparse.Namespace(command="maintain", dry_run=False),
                (("prune-scores", invoke_stage),),
            ).exit_code
        return cli.main(["--config", str(tmp_path / "config.toml"), "prune-scores"])

    if isinstance(outcome, type):
        with pytest.raises(outcome, match="original command outcome"):
            invoke()
    else:
        assert invoke() == outcome
    ledger = runs.read_runs(runs.runs_path(config))
    assert ledger.height == 1
    assert ledger["record_kind"][0] == ("stage" if embedded else "invocation")
    assert ledger["exit_code"][0] == (None if isinstance(outcome, type) else outcome)
    assert ledger["error"][0] == (
        outcome.__name__ if isinstance(outcome, type) else None
    )
    decoded = json.loads(ledger["failure_details_json"][0])
    assert sum(item.get("issue_count") == 2 for item in decoded) == 1
    assert _recorded_check_assessment(
        ledger.row(0, named=True), ledger["ended_at"][0]
    ).invalid


@pytest.mark.parametrize("stage", ["retention_verification", "retention_audit_cleanup"])
def test_repeated_batch_inside_one_argument_replaces_summary_without_collapsing_samples(
    stage,
):
    original = tuple(
        replace(item, stage=stage) for item in check("suspended", count=2).diagnostics()
    )
    changed = tuple(
        replace(item, detail="new" if item.issue_count is None else "summary")
        for item in original
    )
    other = check(name="other").diagnostics()
    expected = other + changed
    assert merge_diagnostics(original + other + original + changed) == expected
    assert merge_diagnostics(original, original + other + changed) == expected
    with diagnostic_progress() as progress:
        record_progress(original + other + original + changed)
    assert tuple(progress) == expected
    assert len(expected) == 4 and expected[-1] == expected[-2]


def test_summaryless_samples_are_not_guessed_to_be_repeated_deliveries():
    batch = check("suspended", count=2).diagnostics()[1:]
    assert merge_diagnostics(batch + batch) == batch * 2
    assert merge_diagnostics(batch + batch, batch) == batch
    unrelated = (ComponentDiagnostic("other", 0, detail="same"),) * 2
    assert merge_diagnostics(unrelated, unrelated) == unrelated[:1]


def test_merge_resolves_each_object_once_and_retains_compression(monkeypatch):
    original = status.assess_progress_identity
    identities = []

    def assess(*args):
        identities.append(args)
        return original(*args)

    batch = check("suspended", count=2).diagnostics()
    renamed = tuple(replace(item, check_id="renamed") for item in batch)
    monkeypatch.setattr(status, "assess_progress_identity", assess)
    assert merge_diagnostics(batch * 2, renamed * 2) == renamed
    assert len(identities) == len(batch) + len(renamed)
    equal = json.loads(failure_details_json(batch))
    different = json.loads(failure_details_json(renamed))
    assert all("progress_id" not in item for item in equal)
    assert all(item["progress_id"] == "check" for item in different)


@pytest.mark.parametrize(
    "kind", ["invalid_identity", "conflicting_owner", "conflicting_check", "bad_clock"]
)
def test_invalid_group_quarantine_keeps_independent_checks(kind):
    independent = payload(check("suspended", name="independent", count=2))
    bad = payload(check(name="bad", count=1))
    if kind == "invalid_identity":
        bad[0]["progress_id"] = {}
    elif kind == "conflicting_owner":
        bad[1]["progress_id"] = "other-owner"
    elif kind == "conflicting_check":
        for item in bad:
            item["progress_id"] = "owner"
        bad[1]["check_id"] = "other-check"
    else:
        bad[0]["recorded_at"] = "not-a-clock"
    recorded = _recorded_check_assessment(row(independent + bad), NOW)
    assert [item.check_id for item in recorded.checks] == ["independent"]
    assert recorded.invalid
    assert _recorded_checks(row(independent + bad), NOW) == recorded.checks
    visible = retention_status(pl.DataFrame([row(independent + bad)]), as_of=NOW)
    assert visible.verification.check_id == "independent"
    assert visible.verification_unknown and visible.severity == "amber"
    assert visible.outstanding_failure is False
    assert "strictly after" in visible.frame()["detail"][0]


@pytest.mark.parametrize("later_state", ["clear", "suspended"])
@pytest.mark.parametrize("later_clock", [NOW, NOW + timedelta(seconds=1)])
@pytest.mark.parametrize("dry_run", [False, True])
def test_uncertainty_requires_strictly_later_clear_but_dry_run_can_verify(
    later_state, later_clock, dry_run
):
    initial = row(payload(check(name="old")) + damaged_items())
    later = row(
        payload(check(later_state, name="later", when=later_clock, dry_run=dry_run)),
        when=NOW + timedelta(seconds=2),
    )
    result = retention_status(
        pl.DataFrame([initial, later]), as_of=NOW + timedelta(seconds=3)
    )
    resolved = later_state == "clear" and later_clock > NOW
    assert result.verification.check_id == "later"
    assert result.verification_unknown is not resolved
    assert result.severity == ("ok" if resolved else "amber")
    assert result.frame()["verification_unknown"][0] is not resolved


def test_dry_run_does_not_clear_execution_failure_after_resolving_check_damage():
    initial = row(damaged_items(), exit_code=1)
    later_clock = NOW + timedelta(seconds=1)
    later = row(
        payload(check(name="later", when=later_clock, dry_run=True)), when=later_clock
    )
    result = retention_status(pl.DataFrame([initial, later]), as_of=later_clock)
    assert result.verification_unknown is False
    assert result.outstanding_failure and result.severity == "amber"
    real_clock = NOW + timedelta(seconds=2)
    real = row(payload(check(name="real", when=real_clock)), when=real_clock)
    result = retention_status(pl.DataFrame([initial, later, real]), as_of=real_clock)
    assert not result.verification_unknown and not result.outstanding_failure
    assert result.severity == "ok"


def test_invalid_new_evidence_does_not_erase_older_check_and_honors_cutoffs():
    old_clock = NOW - timedelta(hours=1)
    old = row(payload(check(name="old", when=old_clock)), when=old_clock)
    frame = pl.DataFrame([old, row(damaged_items())])
    current = retention_status(frame, as_of=NOW)
    assert current.verification.check_id == "old" and current.verification_unknown
    assert current.severity == "amber"
    historical = retention_status(frame, as_of=old_clock)
    assert historical.verification.check_id == "old" and historical.severity == "ok"
    assert historical.verification_unknown is False


def test_dashboard_explains_uncertainty_without_exposing_identity_prose(tmp_path):
    config, _, _ = setup(tmp_path)
    items = payload(check(name="valid")) + damaged_items()
    items[-1]["progress_id"] = ["/private/secret"]
    data = row(items)
    pl.DataFrame(
        [{key: data.get(key) for key in runs.RUNS_SCHEMA}], schema=runs.RUNS_SCHEMA
    ).write_parquet(runs.runs_path(config))
    ctx = collect_context(config, now=NOW)
    panel = _retention_panel(ctx)
    assert panel.status == "amber"
    assert "strictly after" in panel.intro and "private" not in panel.intro
    assert runs.read_runs(runs.runs_path(config)).height == 1


def test_legacy_verdict_only_clear_rows_never_create_phantom_samples():
    legacy = {
        "stage": "retention_verification",
        "exit_code": 0,
        "failure_kind": "retention_clear",
        "recorded_at": NOW.isoformat(),
        "detail": "published-score protection verified",
    }
    clear = _recorded_checks(row([legacy]), NOW)[0]
    assert clear.total == 0 and clear.issues == ()
    sample = legacy | {
        "category": "unreadable",
        "detail": "unreadable: genuine evidence",
    }
    mixed = _recorded_checks(row([legacy, sample, sample]), NOW)[0]
    assert mixed.total == 2 and mixed.omitted == 0
    assert mixed.details == ("unreadable: genuine evidence",) * 2


@pytest.mark.parametrize("count", [-1, "2", None, True, 1.0])
def test_invalid_summary_never_inflates_samples_or_authorizes_clear(count):
    samples = payload(check(name="damaged", count=1))
    samples[0]["issue_count"] = count
    recorded = _recorded_check_assessment(row(samples), NOW)
    assert recorded.invalid and not recorded.checks
    result = retention_status(pl.DataFrame([row(samples)]), as_of=NOW)
    assert result.verification_unknown and result.severity == "amber"


@pytest.mark.parametrize(
    "edit", [{"issue_count": 0}, {"omitted_count": -1}, {"omitted_count": 3}]
)
def test_inconsistent_recorded_counts_remain_unverified(edit):
    samples = payload(check(count=2))
    samples[0].update(edit)
    assert _recorded_check_assessment(row(samples), NOW).invalid


@pytest.mark.parametrize("state", ["clear", "suspended"])
def test_bounded_counts_and_progress_only_recovery_remain_lossless(state):
    value = check(state, name=None, count=23)
    value = replace(value, progress_id="owner")
    for _ in range(3):
        assessment = _recorded_check_assessment(row(payload(value)), NOW)
        assert not assessment.invalid
        value = assessment.checks[0]
        assert value.progress_id == "owner" and value.check_id is None
        assert value.total == 23 and value.omitted == 3 and len(value.issues) == 20


@pytest.mark.parametrize("role", [[], {}, set(), 7, "invalid"])
def test_unhashable_and_other_invalid_cleanup_roles_raise_value_error(role):
    with pytest.raises(ValueError, match="unknown.*role"):
        AuditCleanupFailure("delete", Path("/private/file"), "raw", failure_role=role)


@pytest.mark.parametrize(
    "role,aborted", [(None, False), ("file_local", False), ("abort_cause", True)]
)
def test_shared_role_projection_preserves_legacy_and_modern_attribution(role, aborted):
    failure = AuditCleanupFailure(
        "delete", Path("/private/file"), "original", failure_role=role
    )
    issue = failure.issue()
    result = AuditCleanupResult(
        issues=(issue,), raw_failures=(failure,), aborted=aborted
    )
    assert result.raw_failures[0].failure_role == role
    assert result.raw_failures[0].raw_detail == "original"
    if role is not None:
        legacy = replace(failure, failure_role=None)
        assert AuditCleanupEntry(issue, legacy).failure.failure_role == role
        with pytest.raises(ValueError, match="conflicts"):
            failure.issue(aborted=not aborted)


@pytest.mark.parametrize(
    "state_order", [("clear", "suspended"), ("suspended", "clear")]
)
def test_independent_valid_equal_clock_checks_keep_persisted_order(state_order):
    items = []
    for index, state in enumerate(state_order):
        items.extend(payload(check(state, name=f"check-{index}")))
    result = retention_status(pl.DataFrame([row(items)]), as_of=NOW)
    assert result.verification.state == state_order[-1]
    assert result.verification_unknown is False


@pytest.mark.parametrize("kind", ["path", "category", "reason_code"])
@pytest.mark.parametrize("value", [None, ""])
def test_empty_legacy_metadata_stays_verdict_only(kind, value):
    items = payload(check())
    legacy = {
        key: value
        for key, value in items[0].items()
        if key not in {"issue_count", "omitted_count"}
    }
    legacy[kind] = value
    recovered = _recorded_check_assessment(row([legacy]), NOW)
    assert not recovered.invalid and recovered.checks[0].issues == ()


@pytest.mark.parametrize("kind", ["path", "category", "reason_code"])
def test_bad_additive_metadata_retains_a_safe_sample_instead_of_a_verdict_only_row(
    kind,
):
    items = payload(check())
    items[0].pop("issue_count")
    items[0].pop("omitted_count")
    items[0][kind] = []
    recorded = _recorded_check_assessment(row(items), NOW)
    assert not recorded.invalid
    assert recorded.checks[0].total == 1
    assert recorded.checks[0].issues[0].category == "verification_invalid"
    assert recorded.checks[0].issues[0].path is None


@pytest.mark.parametrize("exit_code", [1, 3, 75])
def test_noncompleted_verification_diagnostics_do_not_create_new_check_damage(
    exit_code,
):
    items = payload(check())
    items[0]["exit_code"] = exit_code
    result = retention_status(
        pl.DataFrame([row(items, exit_code=exit_code)]), as_of=NOW
    )
    assert result.verification_unknown is False
    assert result.verification.state == "not_recorded"
    assert result.attempt_state == ("contended" if exit_code == 75 else "failed")
