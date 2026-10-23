"""Uncertain ledger artifacts remain visible without authorizing them."""

import json
import os
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import pytest
from test_approvals import PAST, commit, release

from grounded_weather_forecast.serve import approvals


@pytest.mark.parametrize(
    "damage", ["json", "array", "clock", "naive", "overflow", "identity"]
)
def test_event_verification_diagnostics_preserve_filename_reservations(
    tmp_path, damage
):
    result = commit(tmp_path, release("a"))
    path = tmp_path / "approvals" / f"{result.approval_id}.json"
    raw = json.loads(path.read_text())
    values = {
        "clock": {"approved_at": "bad"},
        "naive": {"approved_at": "2026-01-01"},
        "overflow": {"visible_at": "0001-01-01T00:00:00+01:00"},
        "identity": {"release_id": "other"},
    }
    path.write_text(
        "{bad"
        if damage == "json"
        else json.dumps([] if damage == "array" else raw | values[damage])
    )
    snapshot = approvals.read_approval_snapshot(tmp_path)
    assert str(path) in snapshot.issues
    assert snapshot.diagnostics[0].path == path
    assert len(snapshot.diagnostics[0].detail) <= 400
    assert snapshot.records() == []
    assert snapshot.reserved_sequence == 1


def test_additive_event_fields_are_forward_readable(tmp_path):
    result = commit(tmp_path, release("a"))
    path = tmp_path / "approvals" / f"{result.approval_id}.json"
    raw = json.loads(path.read_text()) | {"future_additive_field": {"value": 1}}
    path.write_text(json.dumps(raw))
    snapshot = approvals.read_approval_snapshot(tmp_path)
    assert (
        not snapshot.issues and snapshot.records()[0].approval_id == result.approval_id
    )


@pytest.mark.parametrize("child", ["", "approvals"])
def test_listing_permission_errors_are_recorded(tmp_path, monkeypatch, child):
    commit(tmp_path, release("a"))
    target = tmp_path / child
    original = os.scandir

    def scan(path):
        if Path(path) == target:
            raise PermissionError("injected denied listing")
        return original(path)

    monkeypatch.setattr(os, "scandir", scan)
    snapshot = approvals.read_approval_snapshot(tmp_path)
    assert str(target) in snapshot.issues
    assert any(i.category == "directory_unreadable" for i in snapshot.diagnostics)


@pytest.mark.parametrize(
    "clock",
    [None, "bad", "2026-01-01", "0001-01-01T00:00:00+01:00", "2999-01-01T00:00:00Z"],
)
def test_uncertain_legacy_publication_is_diagnostic(tmp_path, clock):
    model = release("a")
    raw = asdict(model) | {"promoted_at": clock, "approval_required": False}
    path = tmp_path / f"{model.release_id}.json"
    path.write_text(json.dumps(raw))
    snapshot = approvals.read_approval_snapshot(tmp_path)
    assert snapshot.records() == [] and str(path) in snapshot.issues


def test_historical_external_payload_cannot_bypass_its_publication_clock(tmp_path):
    commit(tmp_path, release("a"))
    snapshot = approvals.read_approval_snapshot(tmp_path)
    cutoff = snapshot.cutoff
    historical = approvals.read_approval_snapshot(tmp_path, as_of=cutoff)
    external = dict(
        historical.releases[0], promoted_at=(cutoff + timedelta(days=1)).isoformat()
    )
    assert historical.records([external]) == []
    assert snapshot.records([external])


def test_snapshot_derives_visibility_only_once(tmp_path, monkeypatch):
    commit(tmp_path, release("a"))
    original = approvals._visibility_floors
    calls = []

    def floors(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(approvals, "_visibility_floors", floors)
    snapshot = approvals.read_approval_snapshot(tmp_path)
    for _ in range(6):
        snapshot.records()
        snapshot.records(mode="historical")
        snapshot.visibility()
    assert len(calls) == 1


def test_cutoff_requires_valid_aware_clock(tmp_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        approvals.read_approval_snapshot(tmp_path, as_of=PAST.replace(tzinfo=None))


@pytest.mark.parametrize("failure", [OSError("denied"), ValueError("bad content")])
def test_publication_failures_are_typed_and_do_not_commit_approval(
    tmp_path, monkeypatch, failure
):
    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(approvals.ModelRelease, "write_once", fail)
    with pytest.raises(approvals.ApprovalPublicationError) as caught:
        commit(tmp_path, release("a"))
    assert caught.value.failure_kind == "promotion_publication_failed"
    assert not (tmp_path / "approvals").exists()
