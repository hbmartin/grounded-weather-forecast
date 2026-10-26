"""Public causes survive without exposing raw evidence; unconfirmed audits survive too."""

import argparse
import copy
import pickle
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from test_daily_read import NOW, START, setup, write_json
from test_retention_audits import verification
from test_retention_verification import run_row

from grounded_weather_forecast import cli, retention_audit, runs
from grounded_weather_forecast.artifact_labels import (
    CompatibilityNote,
    CompatibilityNotice,
    artifact_label,
    audit_label,
)
from grounded_weather_forecast.dashboard.context import collect_context
from grounded_weather_forecast.dashboard.zones.operations import _retention_panel
from grounded_weather_forecast.reports import operations
from grounded_weather_forecast.retention_status import (
    RetentionVerification,
    retention_status,
)
from grounded_weather_forecast.status import diagnostic_progress, failure_details_json
from grounded_weather_forecast.verification_text import (
    VerificationIssue,
    normalize_detail,
    public_detail,
    reason_for_error,
)


def record_check(config, check):
    row = {key: None for key in runs.RUNS_SCHEMA}
    row.update(
        command="prune-scores",
        started_at=check.recorded_at,
        ended_at=check.recorded_at,
        exit_code=0,
        failure_details_json=failure_details_json(check.diagnostics()),
    )
    pl.DataFrame([row], schema=runs.RUNS_SCHEMA).write_parquet(runs.runs_path(config))


@pytest.mark.parametrize(
    "raw",
    [
        "conflicting outputs [/Users/private/a.json]",
        "paths=a,/Users/private/b",
        "source </Users/private/c>",
        "{/Users/private/d}",
        r"source [C:\Users\private\a]",
        r"source [\\server\private\a]",
        "source [file:///Users/private/a]",
        "source [https://private.example/secret]",
    ],
)
def test_raw_prose_never_supplies_public_explanation(raw):
    assert (
        public_detail(raw)
        == "Protection could not be verified; see the stored diagnostic."
    )
    assert public_detail(raw, category=raw, reason_code=raw) == public_detail(raw)
    value = public_detail(
        raw, category="active_reference_invalid", reason_code="permission_denied"
    )
    assert "Active pointer" in value and "Access denied" in value
    assert raw not in value


def test_typed_causes_and_raw_evidence_round_trip(tmp_path):
    config, _, _ = setup(tmp_path)
    source = config.artifacts_dir / "active_release.json"
    raw = f"conflicting outputs [/Users/private/a.json]; {source}"
    check = retention_audit.publish_verification(
        config,
        RetentionVerification("suspended", NOW),
        dry_run=True,
        issues=(
            VerificationIssue(
                source, "active_reference_invalid", raw, "permission_denied"
            ),
        ),
    )
    stored = json.loads(failure_details_json(check.diagnostics()))
    assert stored[1]["category"] == "active_reference_invalid"
    assert (
        stored[1]["reason_code"] == "permission_denied" and stored[1]["detail"] == raw
    )
    restored = retention_status(
        pl.DataFrame([run_row() | {"failure_details_json": json.dumps(stored)}]),
        as_of=NOW,
    ).verification
    frame = restored.frame(
        path_label=lambda p: artifact_label(Path(p), config),
        audit_path_label=lambda p: audit_label(p, config),
    )
    assert "Access denied" in frame["detail"][0] and str(tmp_path) not in str(frame)
    assert frame["affected_paths"][0] == "artifacts/active_release.json"
    audit = [
        json.loads(line)
        for line in (config.artifacts_dir / check.audit_path).read_text().splitlines()
    ]
    assert audit[1]["path"] == str(source) and audit[1]["detail"] == raw


@pytest.mark.parametrize(
    "category,reason", [(123, []), ([], {}), ("unknown", "unknown")]
)
def test_malformed_additive_codes_are_not_rendered(tmp_path, category, reason):
    check = RetentionVerification(
        "suspended", NOW, ("[/Users/private/a]",), check_id="a"
    )
    data = json.loads(failure_details_json(check.diagnostics()))
    data[1].update(category=category, reason_code=reason)
    status = retention_status(
        pl.DataFrame([run_row() | {"failure_details_json": json.dumps(data)}]),
        as_of=NOW,
    )
    assert status.verification.state == "suspended"
    assert "/Users" not in str(status.frame())


@pytest.mark.parametrize("case", ["pointer", "permission", "archive"])
def test_real_verification_issues_have_safe_actionable_causes(
    tmp_path, monkeypatch, capsys, case
):
    config, _, _ = setup(tmp_path)
    path = config.artifacts_dir / "active_release.json"
    if case == "pointer":
        write_json(path, {"release_ids": [None]})
        expected = "Active pointer"
    elif case == "archive":
        path = config.predict.history_path.parent / "served_forecasts/bad.json"
        write_json(path, {})
        expected = "Archived served document"
    else:
        read = Path.read_text

        def denied(current, *args, **kwargs):
            if current == path:
                raise PermissionError(f"denied [{path}]")
            return read(current, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", denied)
        expected = "Access denied"
    outcome = cli._cmd_prune_scores(config, argparse.Namespace(dry_run=True))
    output = capsys.readouterr().out
    assert outcome.exit_code == 0 and expected in output
    assert str(tmp_path) not in output
    assert any(item.category is not None for item in outcome.diagnostics)


def test_unknown_reference_path_stays_missing(tmp_path, monkeypatch):
    config, _, _ = setup(tmp_path)

    class BrokenSnapshot:
        diagnostics = ()
        issues = ()

        def records(self):
            raise ValueError("unlocated verification failure")

    monkeypatch.setattr(
        operations, "read_approval_snapshot", lambda *a, **k: BrokenSnapshot()
    )
    captured = []
    assert operations._protected_evaluations(config, now=NOW, issues=captured) is None
    assert len(captured) == 1 and captured[0].path is None
    assert captured[0].category == "release_reference_invalid"


def test_legacy_adapter_normalizes_exact_prefixes_and_empty_paths(tmp_path):
    config, _, _ = setup(tmp_path)
    check = RetentionVerification(
        "suspended",
        NOW,
        ("/p: schema_invalid: bad /p: content", "schema_invalid: empty path"),
        ("/p", ""),
        categories=("schema_invalid", "schema_invalid"),
    )
    result = retention_audit.publish_verification(config, check, dry_run=True)
    assert result.details == (
        "/p: schema_invalid: bad /p: content",
        "schema_invalid: empty path",
    )
    assert result.paths == ("/p", None)
    assert (
        normalize_detail("/p: schema_invalid: bad /p: content", "/p", "schema_invalid")
        == "bad /p: content"
    )
    rows = [
        json.loads(line)
        for line in (config.artifacts_dir / result.audit_path).read_text().splitlines()
    ]
    assert rows[1]["detail"] == "bad /p: content" and rows[2]["path"] is None


@pytest.mark.parametrize("point", ["sync", "staging", "lock"])
def test_post_link_failures_retain_unconfirmed_reference(tmp_path, monkeypatch, point):
    config, _, _ = setup(tmp_path)
    if point == "sync":
        monkeypatch.setattr(
            retention_audit,
            "_sync_directory",
            lambda *_: (_ for _ in ()).throw(OSError("sync failed [/private/path]")),
        )
    elif point == "staging":
        unlink = Path.unlink

        def denied(path, *args, **kwargs):
            if path.name.endswith(".tmp"):
                raise PermissionError("cleanup denied")
            return unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", denied)
    else:
        from filelock import FileLock

        release = FileLock.release

        def failed(lock, *a, **k):
            release(lock, *a, **k)
            raise PermissionError("release denied")

        monkeypatch.setattr(FileLock, "release", failed)
    with diagnostic_progress() as progress:
        check = retention_audit.publish_verification(
            config, verification(), dry_run=True
        )
    assert check.audit_path is None and check.audit_candidate_path
    path = config.artifacts_dir / check.audit_candidate_path
    assert json.loads(path.read_text().splitlines()[-1])["kind"] == "complete"
    summaries = [item for item in progress if item.issue_count is not None]
    assert (
        len(summaries) == 1
        and summaries[0].audit_candidate_path == check.audit_candidate_path
    )
    row = run_row() | {
        "failure_details_json": failure_details_json(check.diagnostics())
    }
    ctx = SimpleNamespace(
        config=config, runs=pl.DataFrame([row]), now=NOW, compatibility_notes=()
    )
    panel = _retention_panel(ctx)
    assert any(
        stat.label == "audit state" and stat.value == "unconfirmed"
        for stat in panel.stats
    )
    assert "unconfirmed" in panel.intro and str(tmp_path) not in str(panel)
    assert check.frame()["audit_candidate_path"][0] == check.audit_candidate_path


@pytest.mark.parametrize("embedded", [False, True])
def test_cancellation_after_link_journals_candidate_once(
    tmp_path, monkeypatch, embedded
):
    config, _, _ = setup(tmp_path)
    monkeypatch.setattr(
        retention_audit,
        "_sync_directory",
        lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    if embedded:
        monkeypatch.setattr(
            cli,
            "_dispatch",
            lambda c, a, p: cli._run_pipeline_steps(
                c,
                a,
                (
                    (
                        "prune-scores",
                        lambda: cli._cmd_prune_scores(
                            c, argparse.Namespace(dry_run=True)
                        ),
                    ),
                ),
                stage_args={"prune-scores": argparse.Namespace(dry_run=True)},
            ),
        )
    with pytest.raises(KeyboardInterrupt):
        cli.main(
            [
                "--config",
                str(tmp_path / "config.toml"),
                "maintain" if embedded else "prune-scores",
            ]
        )
    ledger = runs.load_runs(runs.runs_path(config)).filter(
        pl.col("command") == "prune-scores"
    )
    assert ledger.height == 1
    summaries = [
        item
        for item in json.loads(ledger["failure_details_json"][0])
        if item.get("issue_count") is not None
    ]
    assert len(summaries) == 1 and summaries[0]["audit_candidate_path"]
    assert "audit_path" not in summaries[0]


@pytest.mark.parametrize("legacy", [False, True])
def test_unconfirmed_and_legacy_orphan_retention(tmp_path, monkeypatch, legacy):
    config, _, _ = setup(tmp_path)
    old = NOW - timedelta(days=91)
    with monkeypatch.context() as patch:
        patch.setattr(
            retention_audit,
            "_sync_directory",
            lambda *_: (_ for _ in ()).throw(OSError("failed")),
        )
        check = retention_audit.publish_verification(
            config, verification(old), dry_run=False
        )
    path = config.artifacts_dir / check.audit_candidate_path
    record_check(config, replace(check, audit_candidate_path=None) if legacy else check)
    content = path.read_bytes()
    assert retention_audit.cleanup_audits(config, now=NOW, dry_run=False) == ()
    assert path.read_bytes() == content
    runs.runs_path(config).unlink()
    retention_audit.cleanup_audits(config, now=NOW, dry_run=True)
    assert path.read_bytes() == content
    retention_audit.cleanup_audits(config, now=NOW, dry_run=False)
    assert not path.exists()


def test_collision_is_not_an_unconfirmed_candidate(tmp_path, monkeypatch):
    config, _, _ = setup(tmp_path)
    monkeypatch.setattr(retention_audit, "uuid4", lambda: SimpleNamespace(hex="a" * 32))
    first = retention_audit.publish_verification(config, verification(), dry_run=False)
    path = config.artifacts_dir / first.audit_path
    before = path.read_bytes()
    second = retention_audit.publish_verification(
        config, verification(count=2), dry_run=False
    )
    assert first.audit_candidate_path is None
    assert second.audit_path is None and second.audit_candidate_path is None
    assert path.read_bytes() == before


@pytest.mark.parametrize("dry_run", [False, True])
def test_cleanup_abort_survives_full_sample_budget(tmp_path, monkeypatch, dry_run):
    config, _, _ = setup(tmp_path)
    check = retention_audit.publish_verification(
        config, verification(NOW - timedelta(days=91)), dry_run=False
    )
    directory = retention_audit.audit_directory(config)
    for index in range(25):
        (directory / f"{index:032x}.jsonl").write_text("{broken")
    monkeypatch.setattr(
        retention_audit,
        "_referenced_audits",
        lambda *_: (_ for _ in ()).throw(ValueError("uncertain references")),
    )
    before = (config.artifacts_dir / check.audit_path).read_bytes()
    notes = retention_audit.cleanup_audits(config, now=NOW, dry_run=dry_run)
    assert len(notes) == 20 and "not completed" in notes[0]
    assert notes[-1] == "7 additional audit cleanup issues omitted."
    assert (config.artifacts_dir / check.audit_path).read_bytes() == before


@pytest.mark.parametrize("operation", ["unlink", "restat"])
def test_file_local_cleanup_failure_does_not_stop_others(
    tmp_path, monkeypatch, operation
):
    config, _, _ = setup(tmp_path)
    checks = [
        retention_audit.publish_verification(
            config, verification(NOW - timedelta(days=91)), dry_run=False
        )
        for _ in range(3)
    ]
    paths = sorted(config.artifacts_dir / check.audit_path for check in checks)
    if operation == "unlink":
        unlink = Path.unlink

        def fail(path, *a, **k):
            if path == paths[0]:
                raise PermissionError("denied")
            return unlink(path, *a, **k)

        monkeypatch.setattr(Path, "unlink", fail)
    else:
        signature = retention_audit._signature
        count = 0

        def fail(path):
            nonlocal count
            if path == paths[0]:
                count += 1
                if count == 3:
                    raise PermissionError("denied")
            return signature(path)

        monkeypatch.setattr(retention_audit, "_signature", fail)
    notes = retention_audit.cleanup_audits(config, now=NOW, dry_run=False)
    assert paths[0].exists() and all(not path.exists() for path in paths[1:])
    assert len(notes) == 1 and paths[0].name in notes[0]


def test_compatibility_note_retains_formatted_path(tmp_path):
    config, _, _ = setup(tmp_path)
    note = CompatibilityNote(config.artifacts_dir / "releases/legacy.json", config)
    panel = _retention_panel(
        SimpleNamespace(
            config=config,
            runs=pl.DataFrame(),
            now=NOW,
            compatibility_notes=(note,),
            compatibility_notices=(
                CompatibilityNotice(config.artifacts_dir / "releases/legacy.json"),
            ),
        )
    )
    assert "artifacts/releases/legacy.json" in panel.intro
    assert "retention only" in panel.intro and str(tmp_path) not in panel.intro


def test_error_reasons_follow_causes_without_parsing_text():
    outer = ValueError("PermissionError /private/file")
    assert reason_for_error(outer) is None
    outer.__cause__ = PermissionError("denied")
    assert reason_for_error(outer) == "permission_denied"
    outer.__cause__ = outer
    assert reason_for_error(outer) is None


@pytest.mark.parametrize(
    "reference",
    ["/private/a.jsonl", ["invalid"], "history/pruning_verifications/../a.jsonl"],
)
def test_invalid_candidate_reference_stops_all_cleanup(tmp_path, reference):
    config, _, _ = setup(tmp_path)
    old = NOW - timedelta(days=91)
    check = retention_audit.publish_verification(
        config, verification(old), dry_run=False
    )
    path = config.artifacts_dir / check.audit_path
    record_check(config, check)
    frame = runs.load_runs(runs.runs_path(config))
    payload = json.loads(frame["failure_details_json"][0])
    payload[0]["audit_candidate_path"] = reference
    frame.with_columns(
        pl.lit(json.dumps(payload)).alias("failure_details_json")
    ).write_parquet(runs.runs_path(config))
    before = path.read_bytes()
    notes = retention_audit.cleanup_audits(config, now=NOW, dry_run=False)
    assert "not completed" in notes[0] and path.read_bytes() == before


def test_compatibility_note_copy_preserves_trusted_value(tmp_path):
    config, _, _ = setup(tmp_path)
    note = CompatibilityNote(config.artifacts_dir / "releases/legacy.json", config)
    for restored in (
        copy.copy(note),
        copy.deepcopy(note),
        pickle.loads(pickle.dumps(note)),
    ):
        assert isinstance(restored, CompatibilityNote) and restored == note
        assert str(tmp_path) not in restored


def test_audit_publication_error_copy_preserves_reason_and_cause():
    error = retention_audit._AuditPublicationError(
        "audit_sync_failed", OSError("sync failed")
    )
    for restored in (copy.copy(error), pickle.loads(pickle.dumps(error))):
        assert restored.reason_code == "audit_sync_failed"
        assert str(restored) == str(error)


@pytest.mark.parametrize("referenced_output", [False, True])
def test_direct_dashboard_pointer_warnings_only_publish_artifact_labels(
    tmp_path, referenced_output
):
    config, _, _ = setup(tmp_path)
    pointer = config.artifacts_dir / "active_release.json"
    missing = config.predict.history_path.parent / "missing-output.json"
    write_json(
        pointer,
        {"output_path": str(missing)} if referenced_output else {"release_ids": [None]},
    )
    context = collect_context(config, now=NOW)
    expected = artifact_label(missing if referenced_output else pointer, config)
    assert expected in context.unreadable_artifacts
    assert all(str(tmp_path) not in item for item in context.unreadable_artifacts)
