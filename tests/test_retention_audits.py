"""Full audit evidence stays separate from bounded journal/display samples."""

import copy
import json
import os
import pickle
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest
from test_daily_read import NOW, START, setup, write_json

from grounded_weather_forecast import runs
from grounded_weather_forecast.reports import operations
from grounded_weather_forecast.retention_audit import (
    audit_directory,
    cleanup_audits,
    publish_verification,
)
from grounded_weather_forecast.retention_status import (
    RetentionVerification,
    retention_status,
)
from grounded_weather_forecast.status import failure_details_json


def verification(when=NOW, count=1):
    return RetentionVerification.from_legacy_details(
        "suspended",
        when,
        tuple(f"source-{i}: invalid schema" for i in range(count)),
        tuple(f"source-{i}" for i in range(count)),
        categories=("schema_invalid",) * count,
    )


@pytest.mark.parametrize("dry_run", [False, True])
def test_many_archives_full_audit_bounded_journal_and_unchanged_scores(
    tmp_path, dry_run
):
    config, _, _ = setup(tmp_path)
    write_json(
        config.artifacts_dir / "releases/release1.json",
        {
            "release_id": "release1",
            "promoted_at": START.isoformat(),
            "evaluation_ids": [],
        },
    )
    archive = config.predict.history_path.parent / "served_forecasts"
    archive.mkdir()
    paths = [archive / f"broken-{i:03}.json" for i in range(125)]
    for path in paths:
        path.write_text("{broken")
    scores = config.dataset.dir / "scores"
    scores.mkdir()
    score = scores / "scores_hourly_live_expanding_e1.parquet"
    score.write_bytes(b"irreplaceable score")
    result = operations.prune_scores_files(config, dry_run=dry_run, now=NOW)
    check = result.verification
    stored = json.loads(failure_details_json(check.diagnostics()))
    assert check.total == 125 and check.omitted == 105 and len(stored) == 21
    assert len(result.skipped) == 21
    assert score.read_bytes() == b"irreplaceable score" and not result.deleted
    audit = config.artifacts_dir / check.audit_path
    lines = [json.loads(line) for line in audit.read_text().splitlines()]
    assert lines[0]["dry_run"] is dry_run
    assert lines[-1] == {"kind": "complete", "issue_count": 125}
    assert {item["path"] for item in lines[1:-1]} == {str(path) for path in paths}
    assert {item["category"] for item in lines[1:-1]} == {"archived_document_invalid"}
    assert all(len(item["detail"]) <= 400 for item in stored)
    assert not list(audit.parent.glob("*.tmp"))
    for path in paths:
        path.unlink()
    resumed = operations.prune_scores_files(config, dry_run=dry_run, now=NOW)
    assert (
        resumed.verification.state == "clear"
        and resumed.verification.audit_path is None
    )


def test_audit_publish_failure_is_explicit_and_nonfatal(tmp_path, monkeypatch):
    config, _, _ = setup(tmp_path)
    monkeypatch.setattr(
        os, "link", lambda *a, **k: (_ for _ in ()).throw(PermissionError("denied"))
    )
    result = publish_verification(config, verification(count=23), dry_run=False)
    assert (
        result.state == "suspended"
        and result.audit_path is None
        and "audit unavailable" in result.audit_error
    )
    assert result.total == 23 and result.omitted == 3
    assert not list(audit_directory(config).glob("*.jsonl"))
    assert not list(audit_directory(config).glob("*.tmp"))
    row = {
        "command": "prune-scores",
        "started_at": START,
        "ended_at": NOW,
        "exit_code": 0,
        "failure_details_json": failure_details_json(result.diagnostics()),
    }
    assert retention_status(pl.DataFrame([row]), as_of=NOW).severity == "amber"


def test_cancellation_cleans_staging_and_propagates(tmp_path, monkeypatch):
    config, _, _ = setup(tmp_path)
    monkeypatch.setattr(
        os, "link", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        publish_verification(config, verification(), dry_run=False)
    assert not list(audit_directory(config).glob("*.jsonl"))
    assert not list(audit_directory(config).glob("*.tmp"))


def test_exclusive_publication_does_not_rewrite_existing_audit(tmp_path, monkeypatch):
    config, _, _ = setup(tmp_path)
    from grounded_weather_forecast import retention_audit
    from types import SimpleNamespace

    monkeypatch.setattr(retention_audit, "uuid4", lambda: SimpleNamespace(hex="a" * 32))
    first = publish_verification(config, verification(), dry_run=False)
    path = config.artifacts_dir / first.audit_path
    content = path.read_bytes()
    second = publish_verification(config, verification(count=2), dry_run=False)
    assert second.audit_error and path.read_bytes() == content


def test_expiry_preserves_referenced_audits_and_dry_runs(tmp_path):
    config, _, _ = setup(tmp_path)
    old = NOW - timedelta(days=91)
    kept = publish_verification(config, verification(old), dry_run=False)
    removed = publish_verification(config, verification(old), dry_run=False)
    fresh = publish_verification(config, verification(), dry_run=False)
    row = {key: None for key in runs.RUNS_SCHEMA}
    row.update(
        command="prune-scores",
        started_at=old,
        ended_at=old,
        exit_code=0,
        failure_details_json=failure_details_json(kept.diagnostics()),
    )
    pl.DataFrame([row], schema=runs.RUNS_SCHEMA).write_parquet(runs.runs_path(config))
    assert cleanup_audits(config, now=NOW, dry_run=True) == ()
    assert (config.artifacts_dir / removed.audit_path).exists()
    assert cleanup_audits(config, now=NOW, dry_run=False) == ()
    assert not (config.artifacts_dir / removed.audit_path).exists()
    assert all(
        (config.artifacts_dir / check.audit_path).exists() for check in [kept, fresh]
    )


@pytest.mark.parametrize(
    "damage", ["ledger", "references", "audit", "future", "symlink"]
)
def test_uncertain_cleanup_preserves_files(tmp_path, damage):
    config, _, _ = setup(tmp_path)
    old = publish_verification(
        config, verification(NOW - timedelta(days=91)), dry_run=False
    )
    path = config.artifacts_dir / old.audit_path
    if damage == "ledger":
        runs.runs_path(config).write_bytes(b"broken parquet")
    elif damage == "references":
        row = {key: None for key in runs.RUNS_SCHEMA}
        row["failure_details_json"] = "{broken"
        pl.DataFrame([row], schema=runs.RUNS_SCHEMA).write_parquet(
            runs.runs_path(config)
        )
    elif damage == "audit":
        path.write_text("{broken")
    elif damage == "future":
        lines = path.read_text().splitlines()
        header = json.loads(lines[0])
        header["recorded_at"] = (NOW + timedelta(days=1)).isoformat()
        path.write_text(json.dumps(header) + "\n" + "\n".join(lines[1:]) + "\n")
    else:
        content = path.read_bytes()
        path.unlink()
        destination = tmp_path / "external.jsonl"
        destination.write_bytes(content)
        path.symlink_to(destination)
    before = path.read_bytes()
    cleanup_audits(config, now=NOW, dry_run=False)
    assert path.read_bytes() == before


def test_error_copy_and_pickle_preserve_diagnostics():
    error = operations._ServedDocumentError(
        Path("served.json"), "archived", ValueError("invalid")
    )
    for restored in (copy.copy(error), pickle.loads(pickle.dumps(error))):
        assert str(restored) == str(error) and restored.issue == error.issue


def test_denied_ledger_stat_cannot_be_mistaken_for_absence(tmp_path, monkeypatch):
    config, _, _ = setup(tmp_path)
    old = publish_verification(
        config, verification(NOW - timedelta(days=91)), dry_run=False
    )
    audit = config.artifacts_dir / old.audit_path
    ledger = runs.runs_path(config)
    stat = Path.stat

    def denied(path, *args, **kwargs):
        if path == ledger:
            raise PermissionError("ledger permission denied")
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied)
    notes = cleanup_audits(config, now=NOW, dry_run=False)
    assert notes and "Access denied" in notes[0]
    assert audit.exists()


def test_dangling_ledger_and_unrecognized_audits_are_preserved(tmp_path):
    config, _, _ = setup(tmp_path)
    old = publish_verification(
        config, verification(NOW - timedelta(days=91)), dry_run=False
    )
    audit = config.artifacts_dir / old.audit_path
    unknown = audit.parent / "operator-notes.jsonl"
    unknown.write_text("manual notes")
    runs.runs_path(config).symlink_to(tmp_path / "missing-ledger.parquet")
    notes = cleanup_audits(config, now=NOW, dry_run=False)
    assert notes and audit.exists() and unknown.read_text() == "manual notes"


def test_legacy_rendering_is_bounded_without_modifying_old_rows(tmp_path):
    check = verification(count=125)
    raw = []
    for index, detail in enumerate(check.details):
        raw.append(
            {
                "stage": "retention_verification",
                "exit_code": 0,
                "failure_kind": "retention_suspended",
                "detail": detail,
                "path": check.paths[index],
                "recorded_at": NOW.isoformat(),
            }
        )
    encoded = json.dumps(raw)
    row = {
        "command": "prune-scores",
        "started_at": START,
        "ended_at": NOW,
        "exit_code": 0,
        "failure_details_json": encoded,
    }
    status = retention_status(pl.DataFrame([row]), as_of=NOW)
    assert status.verification.total == 125 and status.verification.omitted == 105
    assert len(status.verification.details) == 20
    assert row["failure_details_json"] == encoded
