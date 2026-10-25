"""Recorded pruning verification is visible without rescanning archives in renders."""

import json
from datetime import timedelta

import polars as pl
import pytest
from conftest import write_config
from test_daily_read import NOW, START, setup, write_json

from grounded_weather_forecast import cli, runs
from grounded_weather_forecast.dashboard.context import collect_context
from grounded_weather_forecast.dashboard.zones.operations import _retention_panel
from grounded_weather_forecast.reports import operations
from grounded_weather_forecast.retention_status import (
    RetentionVerification,
    latest_retention_verification,
    retention_status,
)
from grounded_weather_forecast.serve.approvals import read_approval_snapshot
from grounded_weather_forecast.status import failure_details_json


def run_row(state="clear", *, when=NOW):
    return {
        "command": "prune-scores",
        "started_at": when - timedelta(seconds=1),
        "ended_at": when,
        "exit_code": 0,
        "failure_details_json": failure_details_json(
            RetentionVerification(
                state, when, ("affected/path.json",) if state == "suspended" else ()
            ).diagnostics()
        ),
    }


@pytest.mark.parametrize(
    "state,status", [("clear", "ok"), ("suspended", "amber"), ("not_recorded", "info")]
)
def test_verification_states_and_timestamps_render_without_failure_alarm(
    tmp_path, state, status
):
    config = write_config(tmp_path)
    config.dataset.dir.mkdir(parents=True, exist_ok=True)
    row = run_row(state)
    pl.DataFrame(
        [{key: row.get(key) for key in runs.RUNS_SCHEMA}], schema=runs.RUNS_SCHEMA
    ).write_parquet(runs.runs_path(config))
    ctx = collect_context(config, now=NOW)
    panel = _retention_panel(ctx)
    assert panel.status == status
    assert panel.stats[0].value == state.replace("_", " ")
    assert panel.stats[1].value == (NOW.isoformat() if state != "not_recorded" else "—")
    assert panel.stats[4].value == NOW.isoformat()
    assert operations._run_counts(ctx.runs, NOW) == (0, 0, 0)


@pytest.mark.parametrize("raw", [None, "invalid", "null", "{}", '[{"stage":"other"}]'])
def test_legacy_or_malformed_diagnostics_are_not_verified_clean(raw):
    row = run_row() | {"failure_details_json": raw}
    result = latest_retention_verification(pl.DataFrame([row]), as_of=NOW)
    assert result.state == "not_recorded" and result.recorded_at is None


def test_latest_clear_verification_replaces_prior_suspension_and_respects_cutoff():
    frame = pl.DataFrame([run_row("suspended", when=START), run_row("clear")])
    assert latest_retention_verification(frame, as_of=START).state == "suspended"
    assert latest_retention_verification(frame, as_of=NOW).state == "clear"
    assert (
        latest_retention_verification(frame, as_of=START - timedelta(seconds=1)).state
        == "not_recorded"
    )


def test_legacy_latest_attempt_does_not_relabel_the_older_check():
    row = run_row() | {
        "ended_at": NOW + timedelta(seconds=1),
        "failure_details_json": None,
    }
    status = retention_status(
        pl.DataFrame([run_row(), row]), as_of=NOW + timedelta(seconds=2)
    )
    assert status.verification.state == "clear"
    assert status.verification.recorded_at == NOW
    assert status.attempt_state == "unverified" and status.severity == "info"


def test_dashboard_does_not_open_full_served_archive_but_pruning_does(
    tmp_path, monkeypatch
):
    config = write_config(tmp_path)
    archived = config.predict.history_path.parent / "served_forecasts"
    archived.mkdir(parents=True)
    (archived / "ancient.json").write_text("{broken")
    (archived / "newest.json").write_text("{broken")

    def full_scan(*_a, **_k):
        raise AssertionError("dashboard must not perform pruning verification")

    original = operations._active_reference_ids
    monkeypatch.setattr(operations, "_active_reference_ids", full_scan)
    ctx = collect_context(config, now=NOW)
    assert any("newest.json" in note for note in ctx.unreadable_artifacts)
    assert all("ancient.json" not in note for note in ctx.unreadable_artifacts)
    monkeypatch.setattr(operations, "_active_reference_ids", original)
    result = operations.prune_scores_files(config, dry_run=True, now=NOW)
    assert result.verification.state == "suspended"
    assert any("ancient.json" in detail for detail in result.verification.details)


def test_legacy_missing_id_is_information_without_serving_authorization(tmp_path):
    config = write_config(tmp_path)
    path = config.artifacts_dir / "releases/legacy.json"
    write_json(
        path,
        {
            "promoted_at": START.isoformat(),
            "evaluation_ids": ["known"],
            "selections": {},
        },
    )
    ctx = collect_context(config, now=NOW)
    assert not any("legacy.json" in note for note in ctx.unreadable_artifacts)
    assert any("legacy.json" in note for note in ctx.compatibility_notes)
    assert read_approval_snapshot(path.parent).records() == []
    assert read_approval_snapshot(path.parent, legacy_references=True).records()


@pytest.mark.parametrize(
    "pointer, expected",
    [
        ({"release_ids": 2}, "active pointer fields"),
        ({"output_path": "missing.json"}, "active served document"),
    ],
)
def test_pointer_and_output_diagnostics_are_distinct_and_not_nested(
    tmp_path, pointer, expected
):
    config = write_config(tmp_path)
    if "output_path" in pointer:
        pointer["output_path"] = str(tmp_path / pointer["output_path"])
    write_json(config.artifacts_dir / "active_release.json", pointer)
    with pytest.raises(ValueError, match=expected) as error:
        operations._pointer_reference_ids(config)
    assert str(error.value).count("could not be verified") == 1


def test_cli_records_clear_and_suspended_results_without_exit_change(tmp_path):
    config, _, _ = setup(tmp_path)
    # Setup's minimal release deliberately has no verifiable clock.
    args = ["--config", str(tmp_path / "config.toml"), "prune-scores", "--dry-run"]
    assert cli.main(args) == 0
    frame = runs.load_runs(runs.runs_path(config))
    assert (
        latest_retention_verification(frame, as_of=frame["ended_at"].max()).state
        == "suspended"
    )
    write_json(
        config.artifacts_dir / "releases/release1.json",
        {
            "release_id": "release1",
            "promoted_at": START.isoformat(),
            "evaluation_ids": [],
        },
    )
    assert cli.main(args) == 0
    frame = runs.load_runs(runs.runs_path(config))
    latest = latest_retention_verification(frame, as_of=frame["ended_at"].max())
    assert latest.state == "clear"
    assert all(
        len(item["detail"]) <= 400
        for value in frame["failure_details_json"].drop_nulls()
        for item in json.loads(value)
    )


def test_pipeline_report_includes_latest_recorded_verification(tmp_path, monkeypatch):
    config = write_config(tmp_path)
    config.dataset.dir.mkdir(parents=True, exist_ok=True)
    row = run_row("suspended")
    pl.DataFrame(
        [{key: row.get(key) for key in runs.RUNS_SCHEMA}], schema=runs.RUNS_SCHEMA
    ).write_parquet(runs.runs_path(config))
    from grounded_weather_forecast.reports import render

    sections = []

    def write(_directory, _name, _title, entries):
        sections.extend(entries)
        return tmp_path / "health.md"

    monkeypatch.setattr(render, "write_markdown_report", write)
    cli._report_pipeline_health(config, [], [])
    frame = next(frame for title, frame in sections if "pruning verification" in title)
    assert frame["verification"].to_list() == ["suspended"]


def test_each_unreadable_path_has_its_own_bounded_diagnostic(tmp_path):
    config = write_config(tmp_path)
    paths = [
        config.artifacts_dir / "releases" / (letter * 90 + ".json")
        for letter in ("a", "b", "c")
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{bad")
    result = operations.prune_scores_files(config, dry_run=True, now=NOW)
    recorded = json.loads(failure_details_json(result.verification.diagnostics()))
    assert len(recorded) == 4
    recorded = [item for item in recorded if item.get("path")]
    for path, diagnostic in zip(paths, recorded, strict=True):
        assert str(path) == diagnostic["path"]
        assert len(diagnostic["detail"]) <= 400


@pytest.mark.parametrize(
    "clock", [None, "bad", "2026-10-22T14:00:00", (NOW + timedelta(days=1)).isoformat()]
)
def test_unverifiable_legacy_identity_still_warns(tmp_path, clock):
    config = write_config(tmp_path)
    write_json(
        config.artifacts_dir / "releases/legacy.json",
        {"promoted_at": clock, "selections": {}},
    )
    context = collect_context(config, now=NOW)
    assert any("legacy.json" in note for note in context.unreadable_artifacts)
    assert context.compatibility_notes == ()
