"""No score deletion may follow unverifiable release or served-document metadata."""

import json
import os
from datetime import timedelta
from pathlib import Path

import pytest
from test_operations import NOW, TestPruneScores as SeededScores

from grounded_weather_forecast.reports import operations


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize(
    "damage",
    [
        "pointer",
        "event",
        "release_clock",
        "release_id",
        "future_legacy",
        "archive",
        "listing",
        "event_listing",
        "pointer_permission",
        "archive_listing",
    ],
)
def test_uncertainty_keeps_all_published_bytes(tmp_path, monkeypatch, dry_run, damage):
    config, scores = SeededScores().seeded(tmp_path)
    releases = config.artifacts_dir / "releases"
    target = releases / "uncertain.json"
    target.write_text(
        json.dumps(
            {
                "release_id": "uncertain",
                "promoted_at": (NOW - timedelta(days=1)).isoformat(),
                "evaluation_ids": ["e2"],
            }
        )
    )
    if damage == "pointer":
        target = config.artifacts_dir / "active_release.json"
        target.write_text("{bad")
    elif damage == "event":
        target = releases / "approvals/00000000000000000001-uncertain.json"
        target.parent.mkdir()
        target.write_text("{bad")
    elif damage in {"release_clock", "release_id", "future_legacy"}:
        raw = json.loads(target.read_text())
        raw.update(
            {"promoted_at": "2026-01-01"}
            if damage == "release_clock"
            else (
                {"release_id": 4}
                if damage == "release_id"
                else {"promoted_at": (NOW + timedelta(days=1)).isoformat()}
            )
        )
        target.write_text(json.dumps(raw))
    elif damage == "archive":
        target = config.predict.history_path.parent / "served_forecasts/recent.json"
        target.parent.mkdir(parents=True)
        target.write_text("{bad")
    elif damage in {"listing", "event_listing", "archive_listing"}:
        target = (
            releases
            if damage == "listing"
            else (
                releases / "approvals"
                if damage == "event_listing"
                else config.predict.history_path.parent / "served_forecasts"
            )
        )
        original = os.scandir

        def scan(path):
            if Path(path) == target:
                raise PermissionError("denied listing")
            return original(path)

        monkeypatch.setattr(os, "scandir", scan)
    else:
        target = config.artifacts_dir / "active_release.json"
        target.write_text("{}")
        original = Path.read_text

        def denied(path, *args, **kwargs):
            if path == target:
                raise PermissionError("denied pointer")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", denied)
    before = {p: p.read_bytes() for p in scores.glob("*.parquet")}
    result = operations.prune_scores_files(config, dry_run=dry_run, now=NOW)
    assert not result.deleted and set(result.kept) == set(before)
    assert before == {p: p.read_bytes() for p in before}
    assert any(str(target) in reason for reason in result.skipped)


def test_missing_referenced_release_has_specific_diagnostic(tmp_path):
    config, scores = SeededScores().seeded(tmp_path)
    (config.artifacts_dir / "active_release.json").write_text(
        json.dumps({"release_id": "missing-release"})
    )
    result = operations.prune_scores_files(config, dry_run=False, now=NOW)
    assert not result.deleted
    assert any(
        "referenced release missing-release is missing" in note
        for note in result.skipped
    )
    assert (scores / "scores_hourly_live_expanding_e2.parquet").exists()


def test_repaired_pointer_resumes_pruning_and_exit_semantics(tmp_path):
    config, _ = SeededScores().seeded(tmp_path)
    pointer = config.artifacts_dir / "active_release.json"
    pointer.write_text("{bad")
    assert not operations.prune_scores_files(config, dry_run=True, now=NOW).deleted
    pointer.write_text(json.dumps({"release_id": "r1"}))
    result = operations.prune_scores_files(config, dry_run=True, now=NOW)
    assert [p.name for p in result.deleted] == [
        "scores_hourly_live_expanding_e2.parquet"
    ]


@pytest.mark.parametrize(
    "pointer",
    [
        {"release_ids": ["r1", 3], "release_id": "r1"},
        {"release_ids": [], "output_path": 4},
        {"release_ids": "r1"},
        [],
    ],
)
def test_parseable_malformed_pointer_never_loses_protection(tmp_path, pointer):
    config, scores = SeededScores().seeded(tmp_path)
    (config.artifacts_dir / "active_release.json").write_text(json.dumps(pointer))
    before = {p: p.read_bytes() for p in scores.glob("*.parquet")}
    result = operations.prune_scores_files(config, dry_run=False, now=NOW)
    assert not result.deleted and any(
        "could not be verified" in note for note in result.skipped
    )
    assert before == {p: p.read_bytes() for p in before}


def test_recent_parseable_archive_without_references_suspends_pruning(tmp_path):
    config, _ = SeededScores().seeded(tmp_path)
    path = config.predict.history_path.parent / "served_forecasts/recent.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"issued_at": NOW.isoformat(), "status": "ready"}))
    result = operations.prune_scores_files(config, dry_run=True, now=NOW)
    assert not result.deleted and any(str(path) in reason for reason in result.skipped)
    # A verifiably old document does not belong to the six-hour safeguard.
    path.write_text(json.dumps({"issued_at": (NOW - timedelta(days=1)).isoformat()}))
    assert operations.prune_scores_files(config, dry_run=True, now=NOW).deleted
