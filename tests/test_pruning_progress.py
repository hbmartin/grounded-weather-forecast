"""Completed checks survive late exceptions, and remain distinct from attempts."""

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from test_daily_read import NOW, START, setup, write_json
from test_retention_verification import run_row

from grounded_weather_forecast import cli, runs
from grounded_weather_forecast.dashboard.context import collect_context
from grounded_weather_forecast.dashboard.zones.operations import _retention_panel
from grounded_weather_forecast.reports import operations
from grounded_weather_forecast.retention_status import (
    RetentionVerification,
    retention_status,
)
from grounded_weather_forecast.status import (
    diagnostic_progress,
    failure_details_json,
    record_progress,
)


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("phase", ["before_check", "after_check", "output", "deletion"])
def test_late_failures_preserve_checks_without_duplicate_journal_rows(
    tmp_path, monkeypatch, embedded, phase
):
    config, _, _ = setup(tmp_path)
    # This release is intentionally uncertain, so ordinary completion records suspension.
    original = operations.prune_scores_files
    if phase in {"before_check", "after_check"}:

        def prune(*args, **kwargs):
            if phase == "after_check":
                original(*args, **kwargs)
            raise OSError("injected failure")

        monkeypatch.setattr(operations, "prune_scores_files", prune)
    elif phase == "output":
        monkeypatch.setattr(
            cli,
            "_print_score_temporaries",
            lambda *a, **k: (_ for _ in ()).throw(OSError("output failure")),
        )
    else:
        # Prove a real deletion fault carries the completed clear check.
        write_json(
            config.artifacts_dir / "releases/release1.json",
            {
                "release_id": "release1",
                "promoted_at": START.isoformat(),
                "evaluation_ids": [],
            },
        )
        directory = config.dataset.dir / "scores"
        directory.mkdir()
        files = [
            directory / f"scores_hourly_live_expanding_e{i}.parquet" for i in range(5)
        ]
        for file in files:
            file.write_bytes(b"score")
        from grounded_weather_forecast.reports import evidence

        evidence.append_ledger(
            pl.DataFrame(
                [{"evaluation_id": f"e{i}"} for i in range(5)],
                schema=evidence.EVALUATIONS_SCHEMA,
            ),
            evidence.ledger_path(config, evidence.EVALUATIONS_LEDGER),
            evidence.EVALUATIONS_LEDGER,
        )
        unlink = Path.unlink

        def failed_unlink(path, *a, **k):
            if path in files:
                raise PermissionError("delete failure")
            return unlink(path, *a, **k)

        monkeypatch.setattr(Path, "unlink", failed_unlink)
    args = argparse.Namespace(command="maintain", dry_run=phase != "deletion")
    with pytest.raises(OSError):
        if embedded:
            cli._run_pipeline_steps(
                config,
                args,
                (("prune-scores", lambda: cli._cmd_prune_scores(config, args)),),
            )
        else:
            cli.main(
                [
                    "--config",
                    str(tmp_path / "config.toml"),
                    "prune-scores",
                    *(["--dry-run"] if phase != "deletion" else []),
                ]
            )
    frame = runs.load_runs(runs.runs_path(config))
    assert frame.height == 1 and frame["error"][0] in {"OSError", "PermissionError"}
    status = retention_status(frame, as_of=datetime.now(UTC))
    assert status.attempt_state == "failed" and status.severity == "amber"
    assert status.verification.state == (
        "not_recorded"
        if phase == "before_check"
        else "clear"
        if phase == "deletion"
        else "suspended"
    )
    if phase != "before_check":
        diagnostic = json.loads(frame["failure_details_json"][0])
        checks = [
            item for item in diagnostic if item["stage"] == "retention_verification"
        ]
        cleanup = [
            item for item in diagnostic if item["stage"] == "retention_audit_cleanup"
        ]
        assert len(checks) == (1 if phase == "deletion" else 2)
        assert (
            len(cleanup) == 1 and cleanup[0]["failure_kind"] == "audit_cleanup_complete"
        )
        assert cleanup[0]["dry_run"] is (phase != "deletion")
        assert (
            cleanup[0]["deleted_audit_count"]
            == cleanup[0]["deleted_staging_count"]
            == 0
        )
        assert len({item["check_id"] for item in diagnostic}) == 1
        assert diagnostic[0]["recorded_at"] is not None


def test_scopes_reset_and_success_diagnostics_are_not_duplicated(tmp_path):
    config, _, _ = setup(tmp_path)
    for _ in range(2):
        assert (
            cli.main(
                ["--config", str(tmp_path / "config.toml"), "prune-scores", "--dry-run"]
            )
            == 0
        )
    frame = runs.load_runs(runs.runs_path(config))
    assert frame.height == 2
    for raw in frame["failure_details_json"]:
        diagnostic = json.loads(raw)
        assert len(diagnostic) == 3
        assert len({json.dumps(item, sort_keys=True) for item in diagnostic}) == 3
        assert (
            sum(item["stage"] == "retention_audit_cleanup" for item in diagnostic) == 1
        )
    with diagnostic_progress() as outer:
        with diagnostic_progress() as inner:
            record_progress(RetentionVerification("clear", NOW).diagnostics())
        assert len(inner) == 1 and outer == []


@pytest.mark.parametrize("kind", [None, "invocation", "stage"])
def test_failed_or_unverified_attempt_does_not_erase_check(kind):
    previous = run_row("suspended", when=START)
    later = run_row() | {
        "failure_details_json": None,
        "record_kind": kind,
        "exit_code": None,
        "error": "OSError",
    }
    status = retention_status(pl.DataFrame([previous, later]), as_of=NOW)
    assert (
        status.verification.state == "suspended"
        and status.verification.recorded_at == START
    )
    assert status.attempt_at == NOW and status.severity == "amber"
    clear = run_row("clear", when=NOW + timedelta(seconds=1))
    assert (
        retention_status(
            pl.DataFrame([previous, later, clear]), as_of=NOW + timedelta(seconds=2)
        ).severity
        == "ok"
    )


@pytest.mark.parametrize(
    "damage",
    [
        {"record_kind": "other"},
        {"started_at": NOW + timedelta(hours=1)},
        {"started_at": None},
    ],
)
def test_ineligible_attempts_cannot_override_a_check(damage):
    previous = run_row("suspended", when=START)
    status = retention_status(
        pl.DataFrame([previous, run_row("clear") | damage]), as_of=NOW
    )
    assert status.verification.state == "suspended"
    assert status.attempt_at == (START if damage.get("record_kind") == "other" else NOW)
    if "started_at" in damage:
        assert status.attempt_state == "unknown" and status.severity == "amber"


def test_multiple_archive_failures_keep_paths_independently_in_bounded_diagnostics(
    tmp_path,
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
    directory = config.predict.history_path.parent / "served_forecasts"
    # Full paths exceed the detail budget; the structured path must still survive.
    for index in range(3):
        directory /= f"segment-{index}-" + "x" * 90
    directory.mkdir(parents=True)
    paths = [directory / (letter * 90 + ".json") for letter in "abc"]
    # Point the archive location at the long directory without changing stored artifacts.
    from dataclasses import replace

    config = replace(
        config,
        predict=replace(
            config.predict, history_path=directory.parent / "history.parquet"
        ),
    )
    archive = directory.parent / "served_forecasts"
    archive.mkdir()
    paths = [archive / path.name for path in paths]
    for path in paths:
        path.write_text("{broken")
    result = operations.prune_scores_files(config, dry_run=True, now=NOW)
    stored = json.loads(failure_details_json(result.verification.diagnostics()))
    assert result.verification.state == "suspended" and len(stored) == 4
    stored = [item for item in stored if item.get("path")]
    assert {item["path"] for item in stored} == {str(path) for path in paths}
    assert all(len(item["detail"]) <= 400 for item in stored)
    for path in paths:
        path.unlink()
    assert (
        operations.prune_scores_files(config, dry_run=True, now=NOW).verification.state
        == "clear"
    )


def test_empty_suspension_and_public_labels_are_explicit(tmp_path):
    assert "suspended" in RetentionVerification("suspended").diagnostics()[0].detail
    config, _, _ = setup(tmp_path)
    write_json(
        config.artifacts_dir / "releases/legacy.json",
        {"promoted_at": START.isoformat(), "selections": {}},
    )
    ctx = collect_context(config, now=NOW)
    assert all(str(tmp_path) not in note for note in ctx.compatibility_notes)
    panel = _retention_panel(ctx)
    assert panel.empty_reason is not None and panel.stats[1].value == "—"
