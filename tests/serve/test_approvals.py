"""Approval transitions, atomic visibility, and shared historical ordering."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from conftest import write_config

from grounded_weather_forecast.evaluation import ModelRelease
from grounded_weather_forecast.serve import approvals
from grounded_weather_forecast.storage import locked_path

PAST = datetime(2026, 8, 1, tzinfo=UTC)


def release(method):
    return ModelRelease.create(
        dataset="data",
        configuration="config",
        evaluation_ids=(method,),
        evaluation_contexts=(),
        training_cutoff=None,
        selections={"hourly.temp_c.0-1h": {"method_id": method}},
        promoted_at=PAST,
    )


def records(directory, as_of=None):
    if as_of is None:
        return approvals.read_approval_snapshot(directory).records()
    return approvals.ordered_approvals(
        [json.loads(path.read_text()) for path in directory.glob("*.json")],
        directory,
        as_of or datetime.now(UTC),
    )


def commit(directory, model):
    with locked_path(directory / ".approval-ledger"):
        existing = records(directory)
        return approvals.publish_approval(
            model, directory, existing[0] if existing else None
        )


def clock(monkeypatch, moment):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment

    monkeypatch.setattr(approvals, "datetime", Clock)


def test_a_b_a_reapproval_and_unchanged_reruns(tmp_path, monkeypatch):
    a, b = release("a"), release("b")
    clock(monkeypatch, PAST + timedelta(days=1))
    first = commit(tmp_path, a)
    original = (tmp_path / f"{a.release_id}.json").read_bytes()
    assert first.created and first.changed
    assert not commit(tmp_path, a).changed
    clock(monkeypatch, PAST + timedelta(days=2))
    commit(tmp_path, b)
    clock(monkeypatch, PAST + timedelta(days=3))
    again = commit(tmp_path, a)
    assert again.changed and not again.created
    assert first.approval_id != again.approval_id
    assert records(tmp_path)[0].release["release_id"] == a.release_id
    assert (
        records(tmp_path, PAST + timedelta(days=2))[0].release["release_id"]
        == b.release_id
    )
    assert (
        records(tmp_path, PAST + timedelta(days=1))[0].release["release_id"]
        == a.release_id
    )
    assert (tmp_path / f"{a.release_id}.json").read_bytes() == original
    assert len(list((tmp_path / "approvals").glob("*.json"))) == 3


@pytest.mark.parametrize("repeat", range(3))
def test_concurrent_unchanged_approvals_commit_once(tmp_path, repeat):
    barrier = threading.Barrier(6, timeout=10)
    model = release("a")

    def approve():
        barrier.wait()
        return commit(tmp_path, model)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(approve) for _ in range(6)]
        results = [future.result(timeout=15) for future in futures]
    assert sum(result.created for result in results) == 1
    assert sum(result.changed for result in results) == 1
    assert len({result.approval_id for result in results}) == 1
    assert len(list((tmp_path / "approvals").glob("*.json"))) == 1


def test_interrupted_event_leaves_unapproved_content_and_can_retry(
    tmp_path, monkeypatch
):
    model = release("a")
    publish = approvals.atomic_write_text

    def fail(*_args):
        raise OSError("event publication fixture")

    monkeypatch.setattr(approvals, "atomic_write_text", fail)
    with pytest.raises(OSError, match="event publication"):
        commit(tmp_path, model)
    assert (tmp_path / f"{model.release_id}.json").exists()
    assert records(tmp_path) == []
    monkeypatch.setattr(approvals, "atomic_write_text", publish)
    retried = commit(tmp_path, model)
    assert retried.changed and not retried.created
    assert records(tmp_path)[0].release["release_id"] == model.release_id


def test_failed_content_publication_cannot_commit_event(tmp_path, monkeypatch):
    def fail(*_args):
        raise OSError("content fixture")

    monkeypatch.setattr(ModelRelease, "write_once", fail)
    with pytest.raises(OSError, match="content fixture"):
        commit(tmp_path, release("a"))
    assert records(tmp_path) == []
    assert not (tmp_path / "approvals").exists()


def test_legacy_implicit_approval_is_unchanged_without_migration(tmp_path):
    model = release("a")
    model.write(tmp_path)
    path = tmp_path / f"{model.release_id}.json"
    raw = json.loads(path.read_text())
    raw.pop("approval_required")
    path.write_text(json.dumps(raw))
    before = path.read_bytes()
    assert records(tmp_path)[0].approval_id is None
    assert not commit(tmp_path, model).changed
    assert path.read_bytes() == before
    assert not (tmp_path / "approvals").exists()


@pytest.mark.parametrize(
    "timestamp", ["broken", "2026-08-01T00:00:00", "2999-01-01T00:00:00+00:00", None]
)
def test_invalid_or_future_legacy_clocks_are_not_approvals(tmp_path, timestamp):
    raw = asdict(release("a")) | {"promoted_at": timestamp, "approval_required": False}
    (tmp_path / "legacy.json").write_text(json.dumps(raw))
    assert records(tmp_path) == []


@pytest.mark.parametrize("timestamp", ["broken", "2026-08-01T00:00:00"])
def test_invalid_event_clocks_do_not_approve_managed_content(tmp_path, timestamp):
    model = release("a")
    result = commit(tmp_path, model)
    path = tmp_path / "approvals" / f"{result.approval_id}.json"
    raw = json.loads(path.read_text()) | {"approved_at": timestamp}
    path.write_text(json.dumps(raw))
    assert records(tmp_path) == []


def test_actual_instants_and_sequence_order_equal_clocks(tmp_path, monkeypatch):
    a, b = release("a"), release("b")
    moment = PAST + timedelta(days=1)
    clock(monkeypatch, moment)
    commit(tmp_path, a)
    commit(tmp_path, b)
    assert records(tmp_path)[0].release["release_id"] == b.release_id
    assert records(tmp_path)[0].sequence == 2
    assert approvals.aware_timestamp("2026-08-02T01:00:00+02:00") < moment


def test_transport_marker_is_outside_content_identity():
    model = release("a")
    assert replace(model, approval_required=True).release_id == model.release_id


def test_damaged_event_is_not_overwritten_by_retry(tmp_path):
    model = release("a")
    first = commit(tmp_path, model)
    path = tmp_path / "approvals" / f"{first.approval_id}.json"
    path.write_text("{broken")
    assert records(tmp_path) == []
    again = commit(tmp_path, model)
    assert again.changed and again.approval_id != first.approval_id
    assert path.read_text() == "{broken"
    assert records(tmp_path)[0].sequence == 2


def test_reporting_and_pruning_share_approval_visibility_and_order(
    tmp_path, monkeypatch
):
    from grounded_weather_forecast.dashboard.context import _releases
    from grounded_weather_forecast.reports import evidence, operations

    config = write_config(tmp_path)
    directory = config.artifacts_dir / "releases"
    a, b = release("a"), release("b")
    clock(monkeypatch, PAST + timedelta(days=1))
    commit(directory, a)
    clock(monkeypatch, PAST + timedelta(days=20))
    commit(directory, b)
    clock(monkeypatch, PAST + timedelta(days=50))
    commit(directory, a)
    replace(release("uncommitted"), approval_required=True).write(directory)
    clock(monkeypatch, PAST + timedelta(days=51))
    report = evidence._readable_releases(directory)
    dashboard = _releases(config, [])
    assert report[-1]["release_id"] == dashboard[0]["release_id"] == a.release_id
    assert {raw["release_id"] for raw in report} == {a.release_id, b.release_id}
    assert dashboard[0]["promoted_at"] == a.promoted_at
    assert approvals.aware_timestamp(dashboard[0]["approved_at"]) == PAST + timedelta(
        days=50
    )
    assert operations._protected_evaluations(config, now=PAST + timedelta(days=51)) == {
        "a"
    }


def test_repeated_transitions_are_distinct_and_reports_remain_idempotent(
    tmp_path, monkeypatch
):
    from grounded_weather_forecast.reports import evidence

    config = write_config(tmp_path)
    directory = config.artifacts_dir / "releases"
    moment = PAST + timedelta(days=1)
    clock(monkeypatch, moment)
    a, b = release("a"), release("b")
    for model in (a, b, a, b, a):
        commit(directory, model)
        evidence.record_selection_churn(config, now=moment)
        evidence.record_selection_churn(config, now=moment)
    ledger = evidence.read_ledger(
        evidence.ledger_path(config, evidence.CHURN_LEDGER), evidence.CHURN_SCHEMA
    )
    assert ledger.height == 4
    assert ledger["to_approval_id"].n_unique() == 4


def test_clock_correction_never_resurrects_superseded_release(tmp_path, monkeypatch):
    a, b = release("a"), release("b")
    corrected = PAST + timedelta(days=1)
    clock(monkeypatch, corrected + timedelta(hours=1))
    commit(tmp_path, a)
    clock(monkeypatch, corrected)
    commit(tmp_path, b)
    assert records(tmp_path)[0].release["release_id"] == b.release_id
    clock(monkeypatch, corrected + timedelta(hours=2))
    assert records(tmp_path)[0].release["release_id"] == b.release_id
    assert records(tmp_path, corrected) == []
    assert (
        records(tmp_path, corrected + timedelta(hours=1))[0].release["release_id"]
        == b.release_id
    )


def test_managed_approval_precedes_newer_implicit_legacy(tmp_path, monkeypatch):
    a, b = release("a"), release("b")
    clock(monkeypatch, PAST + timedelta(days=1))
    commit(tmp_path, a)
    replace(
        b, approval_required=False, promoted_at=(PAST + timedelta(days=2)).isoformat()
    ).write(tmp_path)
    clock(monkeypatch, PAST + timedelta(days=3))
    assert records(tmp_path)[0].release["release_id"] == a.release_id


def test_new_content_requires_event_and_orphan_retry_is_first_approval(
    tmp_path, monkeypatch
):
    model = release("a")
    assert model.approval_required
    model.write(tmp_path)
    assert records(tmp_path) == []
    first = commit(tmp_path, model)
    assert first.disposition == "first" and not first.created and first.changed
    assert commit(tmp_path, model).disposition == "unchanged"
    commit(tmp_path, release("b"))
    assert commit(tmp_path, model).disposition == "reapproval"


@pytest.mark.parametrize(
    "timestamp", ["0001-01-01T00:00:00+05:00", "9999-12-31T23:59:59-05:00"]
)
def test_overflowing_clocks_are_excluded(tmp_path, timestamp):
    assert approvals.aware_timestamp(timestamp) is None
    model = release("a")
    result = commit(tmp_path, model)
    path = tmp_path / "approvals" / f"{result.approval_id}.json"
    raw = json.loads(path.read_text()) | {"approved_at": timestamp}
    path.write_text(json.dumps(raw))
    assert records(tmp_path) == []


def test_sequence_reservation_ignores_unicode_but_keeps_damaged_ascii_names(tmp_path):
    directory = tmp_path / "approvals"
    directory.mkdir()
    (directory / "²-stray.json").write_text("{}")
    (directory / "１２-stray.json").write_text("{}")
    (directory / "00000000000000000042-damaged.json").write_text("broken")
    result = commit(tmp_path, release("a"))
    assert result.approval_id.startswith("00000000000000000043-")
    assert (directory / "00000000000000000042-damaged.json").read_text() == "broken"


def test_readers_and_retention_use_sequence_after_recent_window(tmp_path, monkeypatch):
    from grounded_weather_forecast.reports import evidence, operations
    from grounded_weather_forecast.dashboard.context import _releases

    config = write_config(tmp_path)
    directory = config.artifacts_dir / "releases"
    high, low = sorted(
        [release("a"), release("b")], key=lambda r: r.release_id, reverse=True
    )
    when = PAST + timedelta(days=1)
    clock(monkeypatch, when)
    commit(directory, high)
    commit(directory, low)
    clock(monkeypatch, when + timedelta(days=10))
    assert _releases(config, [])[0]["release_id"] == low.release_id
    assert evidence._readable_releases(directory)[-1]["release_id"] == low.release_id
    assert operations._protected_evaluations(
        config, now=when + timedelta(days=10)
    ) == set(low.evaluation_ids)


def test_past_report_excludes_later_approvals_but_retention_protects_current(
    tmp_path, monkeypatch
):
    from grounded_weather_forecast.reports import evidence, operations

    config = write_config(tmp_path)
    directory = config.artifacts_dir / "releases"
    before = PAST + timedelta(days=1)
    clock(monkeypatch, before)
    commit(directory, release("a"))
    clock(monkeypatch, before + timedelta(days=1))
    commit(directory, release("b"))
    assert evidence.record_selection_churn(config, now=before).is_empty()
    assert operations._protected_evaluations(config, now=before) == {"a", "b"}
    assert approvals.read_approved_releases(directory, as_of=before)[0][
        "evaluation_ids"
    ] == ["a"]


def test_commit_clock_is_not_backdated_to_promotion_snapshot(tmp_path, monkeypatch):
    model = release("a")
    clock(monkeypatch, PAST + timedelta(days=1))
    snapshot = approvals.read_approval_snapshot(tmp_path)
    clock(monkeypatch, PAST + timedelta(days=2))
    with locked_path(tmp_path / ".approval-ledger"):
        approvals.publish_approval(model, tmp_path, None, snapshot=snapshot)
    assert records(tmp_path, PAST + timedelta(days=1, hours=1)) == []
    assert records(tmp_path)[0].approved_at == PAST + timedelta(days=2)


def test_snapshot_remains_fixed_when_later_approval_commits(tmp_path):
    a, b = release("a"), release("b")
    commit(tmp_path, a)
    snapshot = approvals.read_approval_snapshot(tmp_path)
    commit(tmp_path, b)
    assert snapshot.records()[0].release["release_id"] == a.release_id
    assert records(tmp_path)[0].release["release_id"] == b.release_id


def test_snapshot_preserves_unreadable_and_wrong_shape_diagnostics(tmp_path):
    (tmp_path / "broken.json").write_text("broken")
    (tmp_path / "array.json").write_text("[]")
    snapshot = approvals.read_approval_snapshot(tmp_path)
    assert {Path(p).name for p in snapshot.issues} == {"broken.json", "array.json"}


@pytest.mark.parametrize("legacy_events", [False, True])
def test_corrected_clock_reapproval_uses_stored_publication_and_visibility_floor(
    tmp_path, monkeypatch, legacy_events
):
    corrected = PAST + timedelta(days=1)
    fast = corrected + timedelta(hours=1)
    b = replace(release("b"), promoted_at=fast.isoformat())
    a = release("a")
    clock(monkeypatch, fast)
    first = commit(tmp_path, b)
    clock(monkeypatch, corrected)
    commit(tmp_path, a)
    clock(monkeypatch, corrected + timedelta(minutes=1))
    # The newly constructed content has an earlier clock; write_once must use
    # the publication time in the already stored B, not this caller's value.
    again = commit(tmp_path, release("b"))
    assert again.disposition == "reapproval" and not again.created
    for path in (tmp_path / "approvals").glob("*.json"):
        raw = json.loads(path.read_text())
        assert approvals.aware_timestamp(raw["visible_at"]) == fast
        if legacy_events:
            raw.pop("visible_at")
            path.write_text(json.dumps(raw))
    assert records(tmp_path)[0].approval_id == again.approval_id
    assert records(tmp_path, corrected + timedelta(minutes=2)) == []
    clock(monkeypatch, fast + timedelta(hours=1))
    assert records(tmp_path)[0].approval_id == again.approval_id
    assert records(tmp_path, fast)[0].approval_id == again.approval_id
    assert first.approval_id != again.approval_id


def test_forward_clock_correction_cannot_revert_current_serving(tmp_path, monkeypatch):
    corrected = PAST + timedelta(days=1)
    clock(monkeypatch, corrected)
    first = commit(tmp_path, release("a"))
    clock(monkeypatch, corrected + timedelta(hours=1))
    latest = commit(tmp_path, release("b"))
    clock(monkeypatch, corrected + timedelta(minutes=1))
    assert records(tmp_path)[0].approval_id == latest.approval_id
    assert (
        records(tmp_path, corrected + timedelta(minutes=1))[0].approval_id
        == first.approval_id
    )


def test_stale_snapshot_reserves_fresh_sequence_and_rechecks_current(tmp_path):
    a, b = release("a"), release("b")
    commit(tmp_path, a)
    snapshot = approvals.read_approval_snapshot(tmp_path)
    current = snapshot.records()[0]
    commit(tmp_path, b)
    before = {p: p.read_bytes() for p in (tmp_path / "approvals").glob("*.json")}
    with locked_path(tmp_path / ".approval-ledger"):
        result = approvals.publish_approval(a, tmp_path, current, snapshot=snapshot)
    assert result.disposition == "reapproval"
    assert records(tmp_path)[0].sequence == 3
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize(
    "clock_value", ["bad", "2026-08-01T00:00:00", "9999-12-31T23:59:59-05:00"]
)
def test_invalid_visibility_clock_cannot_authorize_content(tmp_path, clock_value):
    first = commit(tmp_path, release("a"))
    path = tmp_path / "approvals" / f"{first.approval_id}.json"
    raw = json.loads(path.read_text()) | {"visible_at": clock_value}
    path.write_text(json.dumps(raw))
    assert records(tmp_path) == []
