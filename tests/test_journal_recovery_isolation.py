"""Optional diagnostic corruption cannot erase runs or independent recovery."""

import json
from dataclasses import fields, replace
from datetime import timedelta

import polars as pl
import pytest
from test_daily_read import NOW, setup
from test_retention_diagnostic_boundaries import check, payload, row

from grounded_weather_forecast import cli, runs, status
from grounded_weather_forecast.dashboard.context import collect_context
from grounded_weather_forecast.dashboard.zones.operations import _retention_panel
from grounded_weather_forecast.retention_status import (
    RetentionVerification,
    _CheckGroups,
    _recorded_check_assessment,
    retention_status,
)


class Unserializable:
    def __repr__(self):
        raise ValueError("private representation failure")

    def __str__(self):
        raise ValueError("private string failure")


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("outcome", [0, 1, OSError, KeyboardInterrupt])
@pytest.mark.parametrize(
    "field,value",
    [
        ("stage", ["s"]),
        ("detail", ["x"]),
        ("exit_code", {}),
        ("issue_count", []),
        ("audit_path", {"path"}),
        ("recorded_at", Unserializable()),
    ],
)
def test_malformed_fields_keep_command_result_and_one_row(
    tmp_path, monkeypatch, embedded, outcome, field, value
):
    config, _, _ = setup(tmp_path)
    independent = check("suspended", name="independent", count=2).diagnostics()
    bad = replace(
        status.ComponentDiagnostic("retention_verification", 0, "retention_clear"),
        **{field: value},
    )

    def step(*args, **kwargs):
        status.record_progress(independent + (bad,))
        if isinstance(outcome, type):
            raise outcome("original result")
        return status.CommandOutcome(outcome, diagnostics=independent + (bad,))

    monkeypatch.setattr(cli, "_dispatch", step)

    def invoke():
        if embedded:
            return cli._run_pipeline_steps(
                config,
                cli.argparse.Namespace(command="maintain"),
                (("prune-scores", step),),
            ).exit_code
        return cli.main(["--config", str(tmp_path / "config.toml"), "prune-scores"])

    if isinstance(outcome, type):
        with pytest.raises(outcome, match="original result"):
            invoke()
    else:
        assert invoke() == outcome
    ledger = runs.read_runs(runs.runs_path(config))
    assert ledger.height == 1
    saved = ledger.row(0, named=True)
    assert saved["error"] == (outcome.__name__ if isinstance(outcome, type) else None)
    assert saved["exit_code"] == (None if isinstance(outcome, type) else outcome)
    items = json.loads(saved["failure_details_json"])
    assert len(items) == 4  # one summary, two genuine samples, one damaged item
    assert sum(item.get("metadata_invalid") is True for item in items) == 1
    recovered = _recorded_check_assessment(saved, saved["ended_at"])
    assert recovered.invalid
    assert recovered.checks[0].check_id == "independent"


@pytest.mark.parametrize("broken", ["merge_diagnostics", "failure_details_json"])
@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("outcome", [0, OSError, KeyboardInterrupt])
def test_recorder_fallback_attempts_core_row_once(
    tmp_path, monkeypatch, broken, embedded, outcome
):
    config, _, _ = setup(tmp_path)

    def step(*args):
        if isinstance(outcome, type):
            raise outcome("original")
        return status.CommandOutcome(outcome)

    def broken_optional(*args):
        raise TypeError("optional journal failure")

    monkeypatch.setattr(cli, broken, broken_optional)
    monkeypatch.setattr(cli, "_dispatch", step)
    original = runs.append_run
    writes = []

    def append(record, path):
        writes.append(record)
        original(record, path)

    monkeypatch.setattr(runs, "append_run", append)

    def invoke():
        if embedded:
            return cli._run_pipeline_steps(
                config,
                cli.argparse.Namespace(command="maintain"),
                (("prune-scores", step),),
            ).exit_code
        return cli.main(["--config", str(tmp_path / "config.toml"), "prune-scores"])

    if isinstance(outcome, type):
        with pytest.raises(outcome, match="original"):
            invoke()
    else:
        assert invoke() == outcome
    assert len(writes) == 1
    saved = runs.read_runs(runs.runs_path(config))
    assert saved.height == 1
    assert "invalid_diagnostic_metadata" in saved["failure_details_json"][0]


def test_capture_failure_retains_previous_progress(monkeypatch):
    valid = check().diagnostics()
    with status.diagnostic_progress() as progress:
        status.record_progress(valid)
        monkeypatch.setattr(
            status,
            "merge_diagnostics",
            lambda *args: (_ for _ in ()).throw(TypeError("bad merge")),
        )
        status.record_progress(valid)
    assert tuple(progress[:-1]) == valid
    assert progress[-1].failure_kind == "invalid_diagnostic_metadata"


@pytest.mark.parametrize("value", ["", " ", 7, [], {}, Unserializable()])
def test_invalid_identity_delivery_deduplicates_without_ownership(value):
    bad = status.ComponentDiagnostic(
        "retention_verification", 0, "retention_clear", check_id=value
    )
    with status.diagnostic_progress() as progress:
        status.record_progress((bad,))
    assert status.merge_diagnostics(tuple(progress), (bad,)) == (bad,)
    assert len(json.loads(status.failure_details_json((bad,)))) == 1


def test_distinct_evidence_does_not_deduplicate_by_bounded_preview():
    first = status.ComponentDiagnostic("other", 0, detail=["x" * 500, "first"])
    second = replace(first, detail=["x" * 500, "second"])
    assert status.merge_diagnostics((first, second), (first,)) == (first, second)
    assert len(json.loads(status.failure_details_json((first, second)))) == 2
    cyclic = []
    cyclic.append(cyclic)
    cycle = replace(first, detail=cyclic)
    assert status.merge_diagnostics((cycle,), (cycle,)) == (cycle,)


def test_all_fields_are_json_safe_individually():
    base = status.ComponentDiagnostic("other", 0)
    for field in fields(base):
        bad = replace(base, **{field.name: Unserializable()})
        result = json.loads(status.failure_details_json((bad, base)))
        assert result[0]["metadata_invalid"] is True
        assert result[1]["stage"] == "other"
        assert "private" not in json.dumps(result)


def test_interleaved_survivors_keep_order_and_sample_multiplicity():
    a = check("suspended", name="A", count=2).diagnostics()
    b = check(name="B").diagnostics()
    incoming = (a[0], a[1], b[0], a[2])
    assert status.merge_diagnostics(incoming) == incoming
    assert status.merge_diagnostics(incoming, a) == b + a
    assert status.merge_diagnostics(incoming + a) == b + a


def test_execution_recovery_is_independent_of_unrelated_damage():
    old = row([], when=NOW - timedelta(hours=1), exit_code=1)
    bad = payload(check(name="bad"))
    bad[0]["progress_id"] = []
    apply = row(payload(check()) + bad)
    first = retention_status(pl.DataFrame([old, apply]), as_of=NOW)
    assert not first.outstanding_failure and first.verification_unknown
    assert first.severity == "amber"
    clock = NOW + timedelta(seconds=1)
    dry = row(payload(check(name="dry", when=clock, dry_run=True)), when=clock)
    final = retention_status(pl.DataFrame([old, apply, dry]), as_of=clock)
    assert not final.outstanding_failure and not final.verification_unknown
    assert final.severity == "ok"


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "bad",
        {},
        {"stage": None},
        {"stage": []},
        {"stage": ""},
        {"stage": "other", "failure_kind": "retention_suspended"},
    ],
)
def test_unattributable_damage_preserves_valid_check_but_marks_uncertain(bad):
    value = row(payload(check()) + [bad])
    result = retention_status(pl.DataFrame([value]), as_of=NOW)
    assert result.verification.check_id == "check"
    assert result.verification_unknown and result.severity == "amber"
    assert not result.outstanding_failure
    assert "private" not in str(result.frame())


def test_valid_unrelated_stages_do_not_contaminate_verification():
    other = {"stage": "truth-qc", "exit_code": 1, "failure_kind": "other"}
    result = retention_status(
        pl.DataFrame([row(payload(check()) + [other])]), as_of=NOW
    )
    assert result.severity == "ok" and not result.verification_unknown


@pytest.mark.parametrize(
    "clocks",
    [
        (None, NOW.isoformat()),
        (NOW.isoformat(), NOW.isoformat().replace("+00:00", "Z")),
        (NOW.isoformat(), "2026-10-22T07:20:00-07:00"),
    ],
)
def test_raw_legacy_clock_boundaries_survive(clocks):
    items = [
        {
            "stage": "retention_verification",
            "exit_code": 0,
            "failure_kind": "retention_clear",
            "issue_count": 0,
            "omitted_count": 0,
            "recorded_at": clock,
        }
        for clock in clocks
    ]
    recovered = _recorded_check_assessment(row(items), NOW)
    assert not recovered.invalid and len(recovered.checks) == 2
    assert recovered.checks[0].recorded_at == recovered.checks[1].recorded_at == NOW


def test_legacy_suspended_verdict_has_unknown_count_and_no_invented_samples(tmp_path):
    config, _, _ = setup(tmp_path)
    item = {
        "stage": "retention_verification",
        "exit_code": 0,
        "failure_kind": "retention_suspended",
        "recorded_at": NOW.isoformat(),
        "detail": "published-score deletion suspended: protection could not be verified",
    }
    value = _recorded_check_assessment(row([item]), NOW).checks[0]
    for _ in range(3):
        assert value.state == "suspended" and value.issues == ()
        assert value.total == 0 and value.issue_count_known is False
        assert value.frame()["issue_count"][0] is None
        value = replace(value, progress_id="owner")
        value = _recorded_check_assessment(row(payload(value)), NOW).checks[0]
    data = row(payload(value))
    pl.DataFrame(
        [{key: data.get(key) for key in runs.RUNS_SCHEMA}], schema=runs.RUNS_SCHEMA
    ).write_parquet(runs.runs_path(config))
    panel = _retention_panel(collect_context(config, now=NOW))
    assert panel.status == "amber"
    assert (
        next(stat.value for stat in panel.stats if stat.label == "verification issues")
        == "—"
    )


@pytest.mark.parametrize("knowledge", ["false", [], 0])
def test_invalid_count_knowledge_cannot_establish_clear(knowledge):
    items = payload(check())
    items[0]["issue_count_known"] = knowledge
    assert _recorded_check_assessment(row(items), NOW).invalid


def test_count_knowledge_requires_sample_free_suspension():
    with pytest.raises(ValueError, match="sample-free suspension"):
        RetentionVerification("clear", NOW, issue_count_known=False)
    with pytest.raises(ValueError, match="boolean"):
        RetentionVerification("suspended", NOW, issue_count_known="false")
    with pytest.raises(ValueError, match="sample-free suspension"):
        replace(check("suspended", count=1), issue_count_known=False)
    adapted = RetentionVerification.from_legacy_details(
        "suspended", NOW, issue_count_known=False
    )
    assert adapted.issue_count_known is False


def test_assessment_is_repeatable_and_does_not_mutate_group_state():
    groups = _CheckGroups()
    groups.add(payload(check())[0] | {"recorded_at": "bad"})
    before = (dict(groups.check_owners), set(groups.bad_checks), tuple(groups.invalid))
    first = groups.assess(NOW, mode=False)
    second = groups.assess(NOW, mode=False)
    assert first == second and len(first.invalid) == 1
    assert before == (
        dict(groups.check_owners),
        set(groups.bad_checks),
        tuple(groups.invalid),
    )
