"""Historical wire compatibility and independent diagnostic failure domains."""

import ast
import json
from collections import Counter
from dataclasses import fields, replace
from datetime import timedelta
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import get_args, get_type_hints

import polars as pl
import pytest
from test_daily_read import NOW, setup
from test_retention_diagnostic_boundaries import check, payload, row

from grounded_weather_forecast import cli, runs, status
from grounded_weather_forecast.dashboard.context import collect_context
from grounded_weather_forecast.dashboard.zones.operations import _retention_panel
from grounded_weather_forecast.retention_status import (
    _recorded_check_assessment,
    retention_status,
)

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures/retention_legacy_wire.json").read_text()
)
FALLBACK = "published-score deletion suspended: protection could not be verified"


@pytest.mark.parametrize(
    "fixture",
    FIXTURES,
    ids=lambda f: f"{f['writer_commit']}-{f['identified']}-{f['samples']}",
)
def test_historical_writer_rows_round_trip_without_git_history(fixture):
    value = _recorded_check_assessment(row(fixture["diagnostics"]), NOW)
    assert not value.invalid
    recovered = value.checks[0]
    expected = 2 if fixture["samples"] else 0
    for _ in range(3):
        assert recovered.state == "suspended" and len(recovered.issues) == expected
        assert recovered.total == expected
        assert recovered.check_id == ("old" if fixture["identified"] else None)
        if fixture["samples"]:
            assert recovered.details == (
                "unreadable: genuine raw evidence",
                "second genuine legacy detail",
            )
        else:
            assert recovered.issue_count_known is (
                None if fixture["identified"] else False
            )
        recovered = _recorded_check_assessment(row(payload(recovered)), NOW).checks[0]


def historical_rows():
    return next(f["diagnostics"] for f in FIXTURES if f["identified"] and f["samples"])


def test_legacy_duplicate_samples_obey_original_summary_and_cap():
    summary, first, _ = historical_rows()
    items = [summary | {"issue_count": 23, "omitted_count": 3}] + [
        first.copy() for _ in range(20)
    ]
    value = _recorded_check_assessment(row(items), NOW)
    assert not value.invalid
    recovered = value.checks[0]
    assert (
        recovered.total == 23 and recovered.omitted == 3 and len(recovered.issues) == 20
    )
    assert recovered.details == (first["detail"],) * 20
    assert (
        _recorded_check_assessment(row(payload(recovered)), NOW).checks[0].details
        == recovered.details
    )


@pytest.mark.parametrize("count,expected", [(0, 0), (2, 2)])
def test_summary_resolves_exact_fallback_text_used_as_real_samples(count, expected):
    summary, first, _ = historical_rows()
    items = [summary | {"issue_count": count, "omitted_count": 0}] + [
        first | {"detail": FALLBACK} for _ in range(2)
    ]
    value = _recorded_check_assessment(row(items), NOW)
    assert not value.invalid and value.checks[0].total == count
    assert len(value.checks[0].issues) == expected


@pytest.mark.parametrize("omitted", [0, None, -1, "0"])
def test_ambiguous_or_invalid_legacy_count_context_remains_uncertain(omitted):
    summary, first, _ = historical_rows()
    current = summary | {"issue_count": 1}
    if omitted is not None:
        current["omitted_count"] = omitted
    else:
        current.pop("omitted_count")
    items = [current] + [first | {"detail": FALLBACK} for _ in range(2)]
    result = retention_status(pl.DataFrame([row(items)]), as_of=NOW)
    assert result.verification_unknown and result.verification.state == "not_recorded"


def test_plain_detail_is_evidence_not_an_inferred_verdict():
    _, first, _ = historical_rows()
    first = {key: value for key, value in first.items() if key != "check_id"}
    first["detail"] = "legacy verdict"
    result = _recorded_check_assessment(row([first]), NOW)
    assert not result.invalid and result.checks[0].details == ("legacy verdict",)
    assert result.checks[0].total == 1


@pytest.mark.parametrize("stage", ["retention_verification", "retention_audit_cleanup"])
@pytest.mark.parametrize(
    "field,value",
    [("omitted_count", "0"), ("detail", []), ("audit_error", {}), ("issue_count", "0")],
)
def test_valid_final_batch_supersedes_damaged_progress(stage, field, value):
    valid = tuple(
        replace(item, stage=stage)
        for item in check("suspended", name="same", count=2).diagnostics()
    )
    damaged = (replace(valid[0], **{field: value}),)
    assert status.merge_diagnostics(damaged, valid) == valid
    assert status.merge_diagnostics(damaged + valid) == valid
    with status.diagnostic_progress() as progress:
        status.record_progress(damaged)
        status.record_progress(valid)
    assert tuple(progress) == valid
    if stage == "retention_verification":
        result = _recorded_check_assessment(
            row(json.loads(status.failure_details_json(tuple(progress)))), NOW
        )
        assert not result.invalid and result.checks[0].total == 2


def test_bad_final_payload_remains_damage_not_valid_evidence():
    valid = check(name="same").diagnostics()
    invalid = (replace(valid[0], omitted_count="0"),)
    merged = status.merge_diagnostics(valid, invalid)
    assert merged == invalid
    result = _recorded_check_assessment(
        row(json.loads(status.failure_details_json(merged))), NOW
    )
    assert result.invalid and not result.checks


@pytest.mark.parametrize(
    "field,value",
    [
        ("deleted_audit_count", "0"),
        ("check_id", []),
        ("detail", object()),
        ("issue_count", {}),
    ],
)
def test_cleanup_damage_does_not_taint_shared_verification_or_recovery(field, value):
    verification = check(name="same").diagnostics()
    cleanup = status.ComponentDiagnostic(
        "retention_audit_cleanup",
        0,
        "audit_cleanup_complete",
        check_id="same",
        progress_id="same",
        issue_count=0,
    )
    cleanup = replace(cleanup, **{field: value})
    prior = row([], when=NOW - timedelta(hours=1), exit_code=1)
    current = row(json.loads(status.failure_details_json(verification + (cleanup,))))
    result = retention_status(pl.DataFrame([prior, current]), as_of=NOW)
    assert (
        result.verification.state == "clear" and result.verification.check_id == "same"
    )
    assert not result.verification_unknown and not result.outstanding_failure
    assert result.cleanup_evidence_unknown and result.severity == "amber"
    assert json.loads(current["failure_details_json"])[-1]["metadata_invalid"] is True


def test_contradictory_cleanup_stage_and_retention_verdict_still_quarantines():
    items = payload(check(name="same"))
    items.append(items[0] | {"stage": "retention_audit_cleanup"})
    result = _recorded_check_assessment(row(items), NOW)
    assert result.invalid and not result.checks


class Stage(StrEnum):
    VERIFICATION = "retention_verification"


class Code(IntEnum):
    ZERO = 0


def test_declared_primitive_subclasses_are_normalized_without_metadata_damage():
    value = status.ComponentDiagnostic(
        Stage.VERIFICATION,
        Code.ZERO,
        "retention_clear",
        check_id=Stage.VERIFICATION,
        issue_count=Code.ZERO,
        omitted_count=Code.ZERO,
    )
    decoded = json.loads(status.failure_details_json((value,)))[0]
    assert type(decoded["stage"]) is str and type(decoded["exit_code"]) is int
    assert "metadata_invalid" not in decoded
    result = _recorded_check_assessment(row([decoded]), NOW)
    assert not result.invalid and result.checks[0].state == "clear"


def test_every_declared_field_uses_its_annotation_not_name():
    hints = get_type_hints(status.ComponentDiagnostic)
    for name, annotation in hints.items():
        allowed = get_args(annotation) or (annotation,)
        assert status._field_valid(name, None) == (type(None) in allowed)
        assert status._field_valid(name, True) == (bool in allowed)
        assert status._field_valid(name, Code.ZERO) == (int in allowed)
        assert status._field_valid(name, Stage.VERIFICATION) == (str in allowed)
    # A future annotated integer name need not follow the old suffix convention.
    from unittest.mock import patch

    with patch.dict(status._FIELD_TYPES, {"attempt_number": (int,)}):
        assert status._field_valid("attempt_number", 7)
        assert not status._field_valid("attempt_number", True)


def test_serialization_assesses_each_field_once(monkeypatch):
    seen = Counter()
    original = status._field_valid

    def counted(name, value):
        seen[name] += 1
        return original(name, value)

    monkeypatch.setattr(status, "_field_valid", counted)
    value = status.ComponentDiagnostic("other", 0, detail=[])
    saved = json.loads(status.failure_details_json((value,)))[0]
    assert saved["metadata_invalid"] is True
    assert seen == Counter({field.name: 1 for field in fields(value)})


@pytest.mark.parametrize(
    "outcome",
    [
        status.CommandOutcome(0),
        status.CommandOutcome(1, object()),
        status.CommandOutcome(0, []),
    ],
)
def test_hint_and_parent_optional_metadata_never_erase_the_run(
    tmp_path, monkeypatch, outcome
):
    config, _, _ = setup(tmp_path)
    hint = tmp_path / "hint.json"
    monkeypatch.setenv(status.COMMAND_OUTCOME_PATH, str(hint))
    monkeypatch.setattr(cli, "_dispatch", lambda *args: outcome)
    assert (
        cli.main(["--config", str(tmp_path / "config.toml"), "backtest"])
        == outcome.exit_code
    )
    ledger = runs.read_runs(runs.runs_path(config))
    assert ledger.height == 1 and ledger["exit_code"][0] == outcome.exit_code
    if outcome.failure_kind is not None:
        assert json.loads(hint.read_text())["failure_kind"] is None
        assert ledger["failure_kind"][0] is None
        assert "invalid_diagnostic_metadata" in ledger["failure_details_json"][0]
        assert status.read_outcome_hint(hint, outcome.exit_code).failure_kind is None


@pytest.mark.parametrize("failure", [OSError, TypeError, ValueError, RuntimeError])
@pytest.mark.parametrize("where", ["encoding", "writing", "helper"])
def test_hint_failures_preserve_valid_command_and_single_record(
    tmp_path, monkeypatch, failure, where
):
    config, _, _ = setup(tmp_path)
    hint = tmp_path / "hint.json"
    monkeypatch.setenv(status.COMMAND_OUTCOME_PATH, str(hint))
    monkeypatch.setattr(cli, "_dispatch", lambda *args: status.CommandOutcome(0))

    def fail(*args, **kwargs):
        raise failure("private hint fault")

    if where == "helper":
        monkeypatch.setattr(cli, "write_outcome_hint", fail)
    elif where == "writing":
        monkeypatch.setattr(Path, "write_text", fail)
    else:
        original = status.json.dumps

        def dumps(value, *args, **kwargs):
            if isinstance(value, dict) and set(value) == {"exit_code", "failure_kind"}:
                return fail()
            return original(value, *args, **kwargs)

        monkeypatch.setattr(status.json, "dumps", dumps)
    assert cli.main(["--config", str(tmp_path / "config.toml"), "backtest"]) == 0
    ledger = runs.read_runs(runs.runs_path(config))
    assert ledger.height == 1 and ledger["exit_code"][0] == 0
    assert "invalid_diagnostic_metadata" in ledger["failure_details_json"][0]


@pytest.mark.parametrize("cancel", [KeyboardInterrupt, SystemExit])
def test_hint_cancellation_propagates_after_null_exit_journal(
    tmp_path, monkeypatch, cancel
):
    config, _, _ = setup(tmp_path)
    monkeypatch.setattr(cli, "_dispatch", lambda *args: status.CommandOutcome(0))

    def interrupted(*args):
        raise cancel("original cancellation")

    monkeypatch.setattr(cli, "write_outcome_hint", interrupted)
    with pytest.raises(cancel, match="original cancellation"):
        cli.main(["--config", str(tmp_path / "config.toml"), "backtest"])
    ledger = runs.read_runs(runs.runs_path(config))
    assert ledger.height == 1 and ledger["exit_code"][0] is None
    assert ledger["error"][0] == cancel.__name__


def test_invalid_optional_stage_classification_grants_no_continuation(tmp_path):
    config, _, _ = setup(tmp_path)
    later = []
    result = cli._run_pipeline_steps(
        config,
        cli.argparse.Namespace(command="maintain"),
        (
            ("backtest", lambda: status.CommandOutcome(1, [])),
            ("report", lambda: later.append(True) or 0),
        ),
        continue_on_failure_kinds=status.PROMOTION_ONLY_FAILURE_KINDS,
    )
    assert result.exit_code == 1 and not later
    ledger = runs.read_runs(runs.runs_path(config))
    assert ledger.height == 1 and ledger["failure_kind"][0] is None


def test_unknown_omitted_aggregate_is_unavailable_but_integer_views_remain(tmp_path):
    config, _, _ = setup(tmp_path)
    value = replace(check("suspended", name="legacy"), issue_count_known=False)
    assert value.total == 0 and value.omitted == 0
    assert value.frame().select("issue_count", "omitted_count").row(0) == (None, None)
    items = payload(value)
    assert "omitted_count" not in items[0]
    for legacy_zero in (False, True):
        data = [item | ({"omitted_count": 0} if legacy_zero else {}) for item in items]
        recovered = _recorded_check_assessment(row(data), NOW)
        assert not recovered.invalid and recovered.checks[0].issue_count_known is False
    data = row(items)
    pl.DataFrame(
        [{key: data.get(key) for key in runs.RUNS_SCHEMA}], schema=runs.RUNS_SCHEMA
    ).write_parquet(runs.runs_path(config))
    panel = _retention_panel(collect_context(config, now=NOW))
    assert (
        next(stat.value for stat in panel.stats if stat.label == "omitted from samples")
        == "—"
    )


def test_independent_final_fallback_matches_normal_damage_helper():
    source = ast.parse(Path(cli.__file__).read_text())
    recorder = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "_record_run"
    )
    literals = [
        node.value
        for node in ast.walk(recorder)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and '"invalid_diagnostic_metadata"' in node.value
    ]
    assert len(literals) == 1
    assert json.loads(literals[0]) == json.loads(
        status.failure_details_json((status.diagnostic_damage(),))
    )
