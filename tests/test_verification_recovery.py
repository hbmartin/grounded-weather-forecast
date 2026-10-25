"""Verification can recover independently of execution and optional audit writes."""

import argparse
import json
import os
from datetime import timedelta
from types import SimpleNamespace

import polars as pl
import pytest
from filelock import FileLock
from test_daily_read import NOW, START, setup
from test_retention_audits import verification
from test_retention_verification import run_row

from grounded_weather_forecast import cli, health_policy, retention_audit, runs
from grounded_weather_forecast.artifact_labels import audit_label
from grounded_weather_forecast.dashboard.zones.operations import _retention_panel
from grounded_weather_forecast.reports import operations
from grounded_weather_forecast.retention_status import (
    RetentionVerification,
    retention_status,
)
from grounded_weather_forecast.status import (
    ComponentDiagnostic,
    diagnostic_progress,
    failure_details_json,
    record_progress,
)
from grounded_weather_forecast.verification_text import normalize_detail, public_detail


@pytest.mark.parametrize(
    "explicit,args,expected",
    [
        (True, '{"dry_run":false}', True),
        (False, '{"dry_run":true}', False),
        (None, '{"dry_run":false}', False),
        (None, '{"dry_run":true}', True),
        (None, "{}", None),
        (None, "{bad", None),
        (None, "[]", None),
        (None, '{"dry_run":0}', None),
    ],
)
def test_recovery_requires_explicit_successful_apply(explicit, args, expected):
    failed = run_row(when=START) | {"exit_code": 1}
    check = RetentionVerification("clear", NOW, dry_run=explicit)
    clear = run_row() | {
        "args_json": args,
        "failure_details_json": failure_details_json(check.diagnostics()),
    }
    status = retention_status(pl.DataFrame([failed, clear]), as_of=NOW)
    assert status.verification.state == "clear"
    assert status.verification.dry_run is expected
    assert status.outstanding_failure is (expected is not False)
    assert status.severity == ("ok" if expected is False else "amber")
    assert status.frame()["latest_attempt_dry_run"][0] is expected


def test_dry_run_clears_suspension_without_erasing_execution_failure():
    failed = run_row("suspended", when=START) | {"exit_code": 1}
    dry = run_row() | {"args_json": '{"dry_run":true}'}
    real = run_row(when=NOW + timedelta(seconds=1))
    rows = pl.DataFrame([failed, dry, real])
    pending = retention_status(rows, as_of=NOW)
    assert pending.verification.state == "clear" and pending.outstanding_failure
    complete = retention_status(rows, as_of=NOW + timedelta(seconds=2))
    assert complete.severity == "ok" and not complete.outstanding_failure


@pytest.mark.parametrize("clock", [START, NOW, None])
def test_qc_refresh_supersedes_only_provably_older_damage(clock):
    bad = run_row() | {
        "command": "truth-qc",
        "started_at": None,
        "failure_kind": 123,
        "ended_at": clock,
    }
    refresh = START + timedelta(minutes=10)
    result = health_policy.truth_qc_freshness(
        {
            "generated_at": refresh.isoformat(),
            "drift_verdict": {"as_of_date": str(START.date())},
        },
        pl.DataFrame([bad]),
        NOW,
    )
    assert result.attempt_unknown is (clock is None or clock >= refresh)


def test_date_only_qc_refresh_uses_earliest_bound_and_one_assessment(monkeypatch):
    midnight = START.replace(hour=0, minute=0)
    rows = pl.DataFrame(
        [
            run_row()
            | {"command": "truth-qc", "started_at": None, "ended_at": midnight},
        ]
    )
    calls = []
    assess = health_policy.assess_attempts
    monkeypatch.setattr(
        health_policy, "assess_attempts", lambda *a: (calls.append(1), assess(*a))[1]
    )
    result = health_policy.truth_qc_freshness(
        {"drift_verdict": {"as_of_date": str(START.date())}}, rows, NOW
    )
    assert result.attempt_unknown and len(calls) == 1
    rows = rows.with_columns(pl.lit(midnight - timedelta(seconds=1)).alias("ended_at"))
    assert not health_policy.truth_qc_freshness(
        {"drift_verdict": {"as_of_date": str(START.date())}}, rows, NOW
    ).attempt_unknown


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize(
    "point", ["before_write", "link", "directory_sync", "after_publish"]
)
def test_interrupted_audit_preserves_one_journal_check(
    tmp_path, monkeypatch, embedded, point
):
    config, _, _ = setup(tmp_path)
    if point == "before_write":
        monkeypatch.setattr(
            retention_audit,
            "temporary_sibling",
            lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
    elif point == "link":
        monkeypatch.setattr(
            os, "link", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt())
        )
    elif point == "directory_sync":
        monkeypatch.setattr(
            retention_audit,
            "_sync_directory",
            lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
    else:
        original = retention_audit._publish_audit

        def interrupt(*a, **k):
            original(*a, **k)
            raise KeyboardInterrupt()

        monkeypatch.setattr(retention_audit, "_publish_audit", interrupt)
    if embedded:

        def dispatch(config, args, parser):
            return cli._run_pipeline_steps(
                config,
                args,
                (
                    (
                        "prune-scores",
                        lambda: cli._cmd_prune_scores(
                            config, argparse.Namespace(dry_run=False)
                        ),
                    ),
                ),
                stage_args={"prune-scores": argparse.Namespace(dry_run=False)},
            )

        monkeypatch.setattr(cli, "_dispatch", dispatch)
        argv = ["--config", str(tmp_path / "config.toml"), "maintain"]
    else:
        argv = ["--config", str(tmp_path / "config.toml"), "prune-scores"]
    with pytest.raises(KeyboardInterrupt):
        cli.main(argv)
    ledger = runs.load_runs(runs.runs_path(config))
    pruning = ledger.filter(pl.col("command") == "prune-scores")
    assert pruning.height == 1
    diagnostics = json.loads(pruning["failure_details_json"][0])
    summaries = [item for item in diagnostics if item.get("issue_count") is not None]
    assert (
        len(summaries) == 1
        and summaries[0]["audit_error"] == "audit publication not completed"
    )
    assert summaries[0]["dry_run"] is False
    assert len({item["check_id"] for item in diagnostics}) == 1
    assert ledger.height == (2 if embedded else 1)


def test_successful_publication_replaces_only_its_own_progress(tmp_path):
    config, _, _ = setup(tmp_path)
    with diagnostic_progress() as progress:
        record_progress((ComponentDiagnostic("other", 1, detail="retain me"),))
        first = retention_audit.publish_verification(
            config, verification(), dry_run=True
        )
        second = retention_audit.publish_verification(
            config, verification(count=2), dry_run=False
        )
        assert progress[0].stage == "other"
        assert len(progress) == 6  # unrelated + two complete bounded batches
        assert not any(item.audit_error for item in progress)
        assert sum(item.check_id == first.check_id for item in progress) == 2
        assert sum(item.check_id == second.check_id for item in progress) == 3


@pytest.mark.parametrize("operation", ["file_sync", "directory_sync", "hard_link"])
def test_sync_or_capability_failure_keeps_check_without_confirming_audit(
    tmp_path, monkeypatch, operation
):
    config, _, _ = setup(tmp_path)
    fail = lambda *_: (_ for _ in ()).throw(OSError("not supported"))
    if operation == "file_sync":
        monkeypatch.setattr(os, "fsync", fail)
    elif operation == "directory_sync":
        monkeypatch.setattr(retention_audit, "_sync_directory", fail)
    else:
        monkeypatch.setattr(os, "link", fail)
    with diagnostic_progress() as progress:
        result = operations.prune_scores_files(config, dry_run=False, now=NOW)
    assert result.verification.state == "suspended" and not result.deleted
    assert result.verification.audit_path is None and result.verification.audit_error
    assert tuple(progress) == result.verification.diagnostics()


def test_sync_precedes_link_and_directory_sync_follows(tmp_path, monkeypatch):
    config, _, _ = setup(tmp_path)
    actions = []
    sync, link = os.fsync, os.link
    monkeypatch.setattr(os, "fsync", lambda fd: (actions.append("sync"), sync(fd))[1])
    monkeypatch.setattr(os, "link", lambda *a: (actions.append("link"), link(*a))[1])
    result = retention_audit.publish_verification(config, verification(), dry_run=False)
    assert result.audit_path and actions == ["sync", "link", "sync"]


def test_audit_lock_defers_publish_and_cleanup_without_command_contention(tmp_path):
    config, _, _ = setup(tmp_path)
    directory = retention_audit.audit_directory(config)
    directory.mkdir(parents=True)
    with FileLock(directory / ".audit.lock", timeout=0):
        result = retention_audit.publish_verification(
            config, verification(), dry_run=False
        )
        notes = retention_audit.cleanup_audits(config, now=NOW, dry_run=False)
    assert result.audit_error and "Timeout" in result.audit_error
    assert notes and "Timeout" in notes[0]


@pytest.mark.parametrize(
    "contents", ["absent", "empty", "fresh", "unrecognized", "corrupt"]
)
def test_no_reference_scan_without_expired_audit_candidates(
    tmp_path, monkeypatch, contents
):
    config, _, _ = setup(tmp_path)
    directory = retention_audit.audit_directory(config)
    if contents != "absent":
        directory.mkdir(parents=True)
    if contents == "fresh":
        retention_audit.publish_verification(config, verification(), dry_run=False)
    elif contents in {"unrecognized", "corrupt"}:
        (
            directory
            / ("notes.txt" if contents == "unrecognized" else "a" * 32 + ".jsonl")
        ).write_text("{bad")
    monkeypatch.setattr(
        retention_audit,
        "_referenced_audits",
        lambda *_: pytest.fail("no eligible audit needs a ledger scan"),
    )
    retention_audit.cleanup_audits(config, now=NOW, dry_run=False)


@pytest.mark.parametrize("dry_run", [False, True])
def test_staging_cleanup_requires_exact_old_regular_unchanged_file(
    tmp_path, monkeypatch, dry_run
):
    config, _, _ = setup(tmp_path)
    directory = retention_audit.audit_directory(config)
    directory.mkdir(parents=True)
    name = "." + "a" * 32 + ".jsonl." + "b" * 16 + ".tmp"
    old = directory / name
    old.write_text("incomplete stream")
    stamp = (NOW - timedelta(days=91)).timestamp()
    os.utime(old, (stamp, stamp))
    fresh = directory / name.replace("a", "c")
    fresh.write_text("live stream")
    link = directory / name.replace("a", "d")
    link.symlink_to(old)
    unknown = directory / "unrecognized.tmp"
    unknown.write_text("manual")
    monkeypatch.setattr(
        retention_audit,
        "_referenced_audits",
        lambda *_: pytest.fail("staging is not a published reference"),
    )
    assert retention_audit.cleanup_audits(config, now=NOW, dry_run=dry_run) == ()
    assert old.exists() is dry_run
    assert fresh.exists() and link.is_symlink() and unknown.exists()


def test_panel_notes_and_public_paths_do_not_consume_issue_samples(tmp_path):
    config, _, _ = setup(tmp_path)
    path = str(config.artifacts_dir / "releases/bad.json")
    check = RetentionVerification(
        "suspended",
        START,
        (f"nested failure: {path}",) * 20,
        (path,) * 20,
        check_id="check",
        issue_count=30,
        audit_error="failed",
    )
    row = run_row() | {
        "failure_details_json": failure_details_json(check.diagnostics())
    }
    panel = _retention_panel(
        SimpleNamespace(
            config=config,
            runs=pl.DataFrame([row]),
            now=NOW,
            compatibility_notes=tuple(f"legacy {i}" for i in range(25)),
        )
    )
    assert len(panel.table.rows) == 20 and str(tmp_path) not in str(panel)
    assert "audit is unavailable" in panel.intro and "5 additional" in panel.intro
    assert "legacy 19" in panel.intro and "legacy 20;" not in panel.intro
    assert normalize_detail(f"{path}: metadata.", path) == "metadata."


def test_cli_and_report_use_same_audit_label(tmp_path, capsys):
    config, _, _ = setup(tmp_path)
    outcome = cli._cmd_prune_scores(config, argparse.Namespace(dry_run=True))
    summary = next(item for item in outcome.diagnostics if item.audit_path)
    label = audit_label(summary.audit_path, config)
    assert label in capsys.readouterr().out
    # Actual command uses wall time; report formatting is also available directly.
    frame = RetentionVerification(
        "suspended", NOW, audit_path=summary.audit_path
    ).frame(audit_path_label=lambda p: audit_label(p, config))
    assert frame["audit_path"][0] == label


def test_cancellation_is_not_replaced_by_staging_cleanup_failure(tmp_path, monkeypatch):
    config, _, _ = setup(tmp_path)
    from pathlib import Path

    unlink = Path.unlink

    def remove(path, *args, **kwargs):
        if path.name.endswith(".tmp"):
            raise PermissionError("cannot remove stage")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", remove)
    monkeypatch.setattr(
        os, "link", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with diagnostic_progress() as progress, pytest.raises(KeyboardInterrupt):
        retention_audit.publish_verification(config, verification(), dry_run=False)
    assert progress[0].audit_error == "audit publication not completed"


def test_changed_staging_file_is_not_deleted(tmp_path, monkeypatch):
    config, _, _ = setup(tmp_path)
    directory = retention_audit.audit_directory(config)
    directory.mkdir(parents=True)
    stage = directory / ("." + "a" * 32 + ".jsonl." + "b" * 16 + ".tmp")
    stage.write_text("old")
    clock = (NOW - timedelta(days=91)).timestamp()
    os.utime(stage, (clock, clock))
    signature = retention_audit._signature
    calls = []

    def changing(path):
        calls.append(path)
        if len(calls) == 3:
            stage.write_text("changed")
        return signature(path)

    monkeypatch.setattr(retention_audit, "_signature", changing)
    retention_audit.cleanup_audits(config, now=NOW, dry_run=False)
    assert stage.read_text() == "changed"


def test_malformed_explicit_check_mode_cannot_borrow_apply_mode():
    failed = run_row(when=START) | {"exit_code": 1}
    clear = run_row()
    details = json.loads(clear["failure_details_json"])
    details[0]["dry_run"] = "false"
    clear["failure_details_json"] = json.dumps(details)
    status = retention_status(pl.DataFrame([failed, clear]), as_of=NOW)
    assert status.attempt_dry_run is None and status.outstanding_failure


@pytest.mark.parametrize("repetition", range(3))
def test_concurrent_colliding_publishers_never_overwrite(
    tmp_path, monkeypatch, repetition
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    config, _, _ = setup(tmp_path)
    monkeypatch.setattr(retention_audit, "uuid4", lambda: SimpleNamespace(hex="e" * 32))
    barrier = Barrier(3, timeout=5)

    def publish(index):
        with diagnostic_progress() as progress:
            barrier.wait()
            result = retention_audit.publish_verification(
                config, verification(count=index + 1), dry_run=False
            )
            assert tuple(progress) == result.diagnostics()
            return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(publish, index) for index in range(3)]
        results = [future.result(timeout=10) for future in futures]
    winners = [result for result in results if result.audit_path]
    assert len(winners) == 1
    path = config.artifacts_dir / winners[0].audit_path
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[-1]["issue_count"] == winners[0].total
    assert len(list(path.parent.glob("*.jsonl"))) == 1


@pytest.mark.parametrize(
    "text", ["metadata invalid. A . B", "3 / 4 cases", "see https://example.org/help"]
)
def test_safe_diagnostic_prose_is_preserved(text):
    assert public_detail(text) == text


@pytest.mark.parametrize(
    "path",
    [
        "/Users/private/file",
        "//server/private/file",
        "C:\\Users\\private",
        "file:///Users/private/file",
    ],
)
def test_unsafe_embedded_path_is_not_public(path):
    assert path not in public_detail(f"source failed: {path}")


@pytest.mark.parametrize("failure", ["staging_cleanup", "lock_release"])
def test_handled_outer_exception_cannot_hide_audit_cleanup_failure(
    tmp_path, monkeypatch, failure
):
    from pathlib import Path

    config, _, _ = setup(tmp_path)
    if failure == "staging_cleanup":
        unlink = Path.unlink

        def remove(path, *a, **k):
            if path.name.endswith(".tmp"):
                raise PermissionError("staging cleanup failed")
            return unlink(path, *a, **k)

        monkeypatch.setattr(Path, "unlink", remove)
    else:
        release = FileLock.release

        def unlock(lock, *a, **k):
            held = lock.is_locked
            release(lock, *a, **k)
            if held:
                raise PermissionError("lock release failed")

        monkeypatch.setattr(FileLock, "release", unlock)
    try:
        raise RuntimeError("caller is already handling this unrelated exception")
    except RuntimeError:
        result = retention_audit.publish_verification(
            config, verification(), dry_run=False
        )
    assert result.audit_path is None and "PermissionError" in result.audit_error
