"""Render path fields, never perform replacement inside diagnostic prose."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import polars as pl
from test_daily_read import NOW, START, setup
from test_retention_verification import run_row

from grounded_weather_forecast.artifact_labels import artifact_label
from grounded_weather_forecast.dashboard.zones.operations import _retention_panel
from grounded_weather_forecast.retention_status import RetentionVerification
from grounded_weather_forecast.status import failure_details_json


def panel(config, check, notes=()):
    row = run_row() | {
        "failure_details_json": failure_details_json(check.diagnostics())
    }
    return _retention_panel(
        SimpleNamespace(
            config=config, runs=pl.DataFrame([row]), now=NOW, compatibility_notes=notes
        )
    )


def test_shipped_relative_roots_and_prose_are_not_rewritten(tmp_path):
    config, _, _ = setup(tmp_path)
    config = replace(
        config,
        artifacts_dir=Path("artifacts"),
        dataset=replace(config.dataset, dir=Path("data")),
        predict=replace(config.predict, history_path=Path("data/history.parquet")),
    )
    check = RetentionVerification(
        "suspended",
        START,
        ("metadata invalid. A . B",),
        ("data/served_forecasts/x.json",),
    )
    result = panel(config, check)
    assert result.table.rows == (
        (
            "dataset/served_forecasts/x.json: Protection could not be verified; see the stored diagnostic.",
        ),
    )
    config = replace(
        config, predict=replace(config.predict, history_path=Path("history.parquet"))
    )
    assert panel(config, check).table.rows == result.table.rows


def test_specific_roots_outside_paths_and_symlinks(tmp_path):
    config, _, _ = setup(tmp_path)
    nested = replace(config, artifacts_dir=config.dataset.dir / "artifacts")
    assert artifact_label(nested.artifacts_dir / "x.json", nested) == "artifacts/x.json"
    assert artifact_label(config.dataset.dir / "x.json", config) == "dataset/x.json"
    assert artifact_label(tmp_path.parent / "outside.json", config) == "outside.json"
    link = config.dataset.dir / "linked.json"
    link.symlink_to(tmp_path.parent / "outside.json")
    assert artifact_label(link, config) == "dataset/linked.json"


def test_legacy_prefix_normalization_is_idempotent(tmp_path):
    config, _, _ = setup(tmp_path)
    path = str(config.artifacts_dir / "releases/bad.json")
    check = RetentionVerification(
        "suspended", START, (f"{path}: invalid metadata",), (path,)
    )
    value = panel(config, check).table.rows[0][0]
    assert (
        value
        == "artifacts/releases/bad.json: Protection could not be verified; see the stored diagnostic."
    )


def test_clear_and_missing_empty_states_and_bounded_compatibility_notes(tmp_path):
    config, _, _ = setup(tmp_path)
    clear = panel(config, RetentionVerification("clear", START, check_id="clear"))
    assert clear.table is None and "verified clear" in clear.empty_reason
    missing = panel(config, RetentionVerification())
    assert missing.table is None and "No recorded" in missing.empty_reason
    noisy = panel(
        config, RetentionVerification(), notes=tuple(f"legacy {i}" for i in range(100))
    )
    assert noisy.table is None and noisy.empty_reason
    assert "20 legacy compatibility notes" in noisy.intro
    assert "Protection could not be verified" not in noisy.intro
    assert "80 additional compatibility notes omitted" in noisy.intro


def test_audit_unavailable_remains_visible_when_issue_table_is_full(tmp_path):
    config, _, _ = setup(tmp_path)
    check = RetentionVerification(
        "suspended",
        START,
        ("schema invalid",) * 20,
        (None,) * 20,
        check_id="check",
        issue_count=30,
        audit_error="audit unavailable: PermissionError",
    )
    result = panel(config, check)
    assert len(result.table.rows) == 20
    assert any(
        stat.label == "audit state" and stat.value == "unavailable"
        for stat in result.stats
    )
    assert "Full verification audit is unavailable" in result.intro
