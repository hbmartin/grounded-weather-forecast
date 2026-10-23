"""Daily reads preserve source bytes, independent availability and evidence identity."""

import json
import sqlite3
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest
from test_daily_read import (
    NOW,
    START,
    file_state,
    provider_db,
    quality,
    read,
    setup,
    write_json,
    write_quality,
)

from grounded_weather_forecast import cli, daily_read as reader
from grounded_weather_forecast.config import config_fingerprint
from grounded_weather_forecast.health_policy import truth_qc_freshness


def modern_release(config, identifier="release1", dataset="dataset1"):
    path = config.artifacts_dir / "releases" / f"{identifier}.json"
    initial = json.loads((config.artifacts_dir / "releases/release1.json").read_text())
    raw = deepcopy(initial)
    raw.update(
        release_id=identifier,
        dataset_fingerprint=dataset,
        promoted_at=(START - timedelta(days=2)).isoformat(),
    )
    for selection in raw["selections"].values():
        selection["dataset_fingerprint"] = dataset
    raw["evaluation_contexts"] = [
        {
            "evaluation_id": "eval1",
            "dataset_fingerprint": dataset,
            "source_kind": "live",
            "source_set_json": '["nws"]',
            "feature_set_json": '["lead_bucket"]',
            "semantics": {v.name: "inst" for v in reader.HOURLY_VARIABLES},
            "config_fingerprint": config_fingerprint(config),
            "code_version": "code1",
            "window": "expanding",
        }
    ]
    write_json(path, raw)
    return path, raw


def temperature(config, forecast, *, realized=False):
    output = read(config, forecast)["prediction_quality"][
        "realized_live" if realized else "backtest"
    ]
    return next(row for row in output["slices"] if row["variable"] == "temp_c")


@pytest.mark.parametrize("sidecars", [(), ("-wal",), ("-shm",), ("-wal", "-shm")])
def test_wal_never_creates_or_changes_any_source_file(tmp_path, sidecars):
    config, forecast, _ = setup(tmp_path)
    connection = sqlite3.connect(config.station.db_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE observations(ts TEXT)")
    connection.commit()
    connection.close()
    for suffix in sidecars:
        config.station.db_path.with_name(
            config.station.db_path.name + suffix
        ).write_bytes(b"existing sidecar")
    write_json(tmp_path / "published.json", {"metadata": {"observed_at": START}})
    before = file_state(tmp_path)
    result = read(config, forecast)
    assert (
        result["operational_health"]["station"]["source"]
        == "published_station_snapshot"
    )
    assert before == file_state(tmp_path)


def test_rollback_database_is_queried_only_in_memory(tmp_path, monkeypatch):
    config, forecast, _ = setup(tmp_path)
    connection = sqlite3.connect(config.station.db_path)
    connection.execute("CREATE TABLE observations(ts TEXT)")
    connection.execute("INSERT INTO observations VALUES (?)", (START.isoformat(),))
    connection.commit()
    connection.close()
    original = sqlite3.connect
    destinations = []

    def connect(path, *args, **kwargs):
        destinations.append(path)
        assert path == ":memory:"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(reader.sqlite3, "connect", connect)
    before = file_state(tmp_path)
    assert (
        read(config, forecast)["operational_health"]["station"]["source"]
        == "live_station_sqlite"
    )
    assert destinations and before == file_state(tmp_path)


@pytest.mark.parametrize(
    "stamp", ["0001-01-01T00:00:00+01:00", "9999-12-31T23:00:00-01:00"]
)
def test_overflowing_source_is_component_local_and_cli_time_is_structured(
    tmp_path, capsys, stamp
):
    config, forecast, raw = setup(tmp_path)
    raw["issued_at"] = stamp
    write_json(forecast, raw)
    result = read(config, forecast)
    assert result["forecast"]["status"] == "unknown"
    assert "station" in result["operational_health"]
    code = cli.main(
        [
            "--config",
            str(tmp_path / "config.toml"),
            "read-daily-report",
            "--start",
            stamp,
            "--as-of",
            NOW.isoformat(),
        ]
    )
    assert code == 2 and json.loads(capsys.readouterr().out)["schema_version"] == 1


def test_reader_defaults_resolve_home_at_call_time(tmp_path, monkeypatch):
    config, forecast, _ = setup(tmp_path)
    first, second = tmp_path / "home1", tmp_path / "home2"
    for home in (first, second):
        published = (
            home
            / "Library/Application Support/grounded-weather-forecast/staged/current.json"
        )
        write_json(published, {"metadata": {"observed_at": START}})
        monkeypatch.setenv("HOME", str(home))
        result = reader.read_daily_report(
            config, forecast_path=forecast, start=START, as_of=NOW
        )
        assert result["operational_health"]["station"]["published_station_path"] == str(
            published
        )
        assert (
            result["operational_health"]["station"]["source"]
            == "published_station_snapshot"
        )


def test_invalid_home_is_structured_and_explicit_override_bypasses_lookup(
    tmp_path, monkeypatch, capsys
):
    _, forecast, _ = setup(tmp_path)
    monkeypatch.setenv("HOME", "")
    args = [
        "--config",
        str(tmp_path / "config.toml"),
        "read-daily-report",
        "--start",
        START.isoformat(),
        "--as-of",
        NOW.isoformat(),
        "--forecast",
        str(forecast),
    ]
    assert cli.main(args) == 2
    assert json.loads(capsys.readouterr().out)["limitations"] == [
        "invalid_configuration"
    ]
    monkeypatch.setattr(
        Path,
        "home",
        classmethod(
            lambda _cls: (_ for _ in ()).throw(RuntimeError("home unavailable"))
        ),
    )
    assert (
        cli.main([*args, "--published-station", str(tmp_path / "explicit.json")]) == 0
    )
    assert json.loads(capsys.readouterr().out)["schema_version"] == 1


def test_strict_serialization_failure_is_json_error(tmp_path, monkeypatch, capsys):
    setup(tmp_path)
    monkeypatch.setattr(
        reader, "read_daily_report", lambda *_a, **_k: {"bad": float("nan")}
    )
    code = cli.main(
        [
            "--config",
            str(tmp_path / "config.toml"),
            "read-daily-report",
            "--start",
            START.isoformat(),
            "--as-of",
            NOW.isoformat(),
            "--published-station",
            str(tmp_path / "explicit.json"),
        ]
    )
    assert code == 2
    assert json.loads(capsys.readouterr().out)["limitations"] == [
        "invalid_report_serialization"
    ]


def test_retained_backtest_uses_selection_dataset_and_legacy_semantic_fallback(
    tmp_path,
):
    config, forecast, _ = setup(tmp_path)
    path, raw = modern_release(config)
    raw["dataset_fingerprint"] = "creation-dataset"
    for selection in raw["selections"].values():
        selection["dataset_fingerprint"] = "dataset1"
        selection.pop("truth_semantics")
    write_json(path, raw)
    write_quality(config, [quality(config)])
    assert temperature(config, forecast)["status"] == "available"
    assert temperature(config, forecast)["evaluation_id"] == "eval1"


def test_missing_hours_do_not_create_quality_cohorts(tmp_path):
    config, forecast, raw = setup(tmp_path)
    write_quality(config, [quality(config)])
    raw["hourly"].pop(0)
    write_json(forecast, raw)
    output = read(config, forecast)
    assert output["forecast"]["coverage"]["missing_hours"]
    slices = output["prediction_quality"]["backtest"]["slices"]
    assert len(slices) == 8 and all(s["method_id"] and s["release_id"] for s in slices)


def test_compatible_fallback_preserves_forecast_and_source_evidence_identity(tmp_path):
    config, forecast, _ = setup(tmp_path)
    modern_release(config)
    modern_release(config, "olderrelease", "older-evaluation-dataset")
    write_quality(
        config,
        [
            quality(
                config,
                realized=True,
                release_id="olderrelease",
                dataset_fingerprint="older-serving-dataset",
            )
        ],
        realized=True,
    )
    row = temperature(config, forecast, realized=True)
    assert row["status"] == "available" and row["mae"] == 2.0
    assert row["release_id"] == "release1"
    assert row["evidence_match"] == "compatible_release"
    assert row["evidence_release_id"] == "olderrelease"
    assert row["evidence_dataset_fingerprint"] == "older-serving-dataset"
    assert "older_compatible_release_evidence" in row["limitations"]


@pytest.mark.parametrize(
    "change",
    [
        "code",
        "config",
        "sources",
        "features",
        "semantics",
        "unknown",
        "legacy",
        "future",
    ],
)
def test_incomplete_or_incompatible_evidence_cannot_supply_fallback(tmp_path, change):
    config, forecast, _ = setup(tmp_path)
    modern_release(config)
    path, raw = modern_release(config, "olderrelease")
    context = raw["evaluation_contexts"][0]
    if change == "code":
        raw["selections"]["hourly.temp_c.1-6h"]["code_version"] = "other"
    elif change == "config":
        raw["config_fingerprint"] = "other"
    elif change == "sources":
        context["source_set_json"] = '["different"]'
    elif change == "features":
        context["feature_set_json"] = '["different"]'
    elif change == "semantics":
        context["semantics"]["temp_c"] = "different"
    elif change == "unknown":
        context["evaluation_id"] = "unknown"
        raw["selections"]["hourly.temp_c.1-6h"]["evaluation_id"] = "unknown"
    elif change == "legacy":
        raw["selections"]["hourly.temp_c.1-6h"]["legacy_context_missing"] = True
    else:
        raw["promoted_at"] = (NOW + timedelta(days=1)).isoformat()
    write_json(path, raw)
    write_quality(
        config,
        [quality(config, realized=True, release_id="olderrelease")],
        realized=True,
    )
    assert temperature(config, forecast, realized=True)["status"] == "unknown"


@pytest.mark.parametrize(
    "changes,status",
    [({"n": 0}, "insufficient"), ({"recorded_at": NOW - timedelta(days=3)}, "stale")],
)
def test_exact_evidence_precedes_healthy_compatible_evidence(tmp_path, changes, status):
    config, forecast, _ = setup(tmp_path)
    modern_release(config)
    modern_release(config, "olderrelease")
    write_quality(
        config,
        [
            quality(config, realized=True, **changes),
            quality(config, realized=True, release_id="olderrelease"),
        ],
        realized=True,
    )
    row = temperature(config, forecast, realized=True)
    assert row["status"] == status and row["evidence_match"] == "exact_release"


def test_provider_rates_are_scoped_to_configured_location(tmp_path):
    config, forecast, _ = setup(tmp_path)
    provider_db(config.forecasts.db_path)
    connection = sqlite3.connect(config.forecasts.db_path)
    connection.execute(
        "INSERT INTO forecast_runs VALUES (9, ?, 1, 2)", (START.isoformat(),)
    )
    connection.execute(
        "INSERT INTO provider_results VALUES (9,9,'foreign','error',NULL,NULL,NULL)"
    )
    connection.commit()
    connection.close()
    row = read(config, forecast)["operational_health"]["providers"]
    assert [p["provider"] for p in row["providers"]] == ["provider"]
    assert row["providers"][0]["attempts"] == 2


@pytest.mark.parametrize("verdict", [None, [], {"as_of_date": "bad"}])
def test_malformed_optional_qc_keeps_freshness_and_alerts(tmp_path, verdict):
    config, forecast, _ = setup(tmp_path)
    write_json(
        config.artifacts_dir / "truth_qc.json",
        {
            "schema_version": 3,
            "generated_at": START,
            "drift_alert": False,
            "correlation_alert": False,
            "shield_alert": None,
            "drift_verdict": verdict,
        },
    )
    row = read(config, forecast)["prediction_quality"]["truth_qc"]
    assert row["freshness"]["status"] == "fresh"
    assert row["status"] == "partial" and row["current_alerts"]["drift_alert"] is False
    assert row["evaluation_date"] is None


def test_date_only_qc_uses_shared_interval_without_inventing_instant(tmp_path):
    config, forecast, _ = setup(tmp_path)
    raw = {
        "schema_version": 3,
        "drift_verdict": {"as_of_date": str(START.date())},
        "drift_alert": False,
    }
    write_json(config.artifacts_dir / "truth_qc.json", raw)
    row = read(config, forecast)["prediction_quality"]["truth_qc"]
    shared = truth_qc_freshness(raw, pl.DataFrame(), NOW)
    assert not shared.age_unusable
    assert row["status"] == "ok" and row["freshness"]["status"] == "fresh"
    assert row["freshness"]["at"] is None
    assert type(row["freshness"]["max_age_minutes"]) is int


def test_unstable_quality_ledger_keeps_independent_components(tmp_path, monkeypatch):
    config, forecast, _ = setup(tmp_path)
    write_quality(config, [quality(config)])
    ledger = config.artifacts_dir / "history/quality.parquet"
    monkeypatch.setattr(reader.Snapshot, "changed_paths", lambda _self: {ledger})
    result = read(config, forecast)
    assert result["prediction_quality"]["backtest"]["limitations"] == [
        "concurrent_replacement_after_retry"
    ]
    assert result["forecast"]["status"] == "ok"
    assert result["operational_health"]["station"]["limitations"] != [
        "concurrent_replacement_after_retry"
    ]
    assert result["prediction_quality"]["drift"]["limitations"] != [
        "concurrent_replacement_after_retry"
    ]


def test_release_json_is_parsed_once_for_all_cohorts(tmp_path, monkeypatch):
    config, forecast, _ = setup(tmp_path)
    write_quality(config, [quality(config)])
    original = json.loads
    parses = []

    def loads(raw, *args, **kwargs):
        value = original(raw, *args, **kwargs)
        if isinstance(value, dict) and value.get("release_id") == "release1":
            parses.append(1)
        return value

    monkeypatch.setattr(reader.json, "loads", loads)
    assert temperature(config, forecast)["status"] == "available"
    assert parses == [1]


@pytest.mark.parametrize(
    "field", ["code_version", "config_fingerprint", "dataset_fingerprint"]
)
def test_conflicting_context_identity_cannot_supply_fallback(tmp_path, field):
    config, forecast, _ = setup(tmp_path)
    modern_release(config)
    path, raw = modern_release(config, "olderrelease")
    raw["evaluation_contexts"][0][field] = "conflicting"
    write_json(path, raw)
    write_quality(
        config,
        [quality(config, realized=True, release_id="olderrelease")],
        realized=True,
    )
    assert temperature(config, forecast, realized=True)["status"] == "unknown"


def test_broken_journal_link_is_unsafe_even_for_rollback_database(tmp_path):
    path = tmp_path / "source.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE observations(ts TEXT)")
    connection.close()
    path.with_name(path.name + "-journal").symlink_to(tmp_path / "missing-journal")
    with pytest.raises(reader.SourceError, match="journal_dependent"):
        reader._open_readonly(path)


def test_failed_latest_qc_refresh_uses_shared_warning_and_preserves_payload(tmp_path):
    config, forecast, _ = setup(tmp_path)
    raw = {
        "schema_version": 3,
        "generated_at": START,
        "drift_verdict": {"as_of_date": str(START.date())},
        "drift_alert": False,
        "correlation_alert": False,
    }
    write_json(config.artifacts_dir / "truth_qc.json", raw)
    pl.DataFrame(
        {
            "command": ["truth-qc"],
            "started_at": [START + timedelta(minutes=1)],
            "ended_at": [NOW],
            "exit_code": [1],
        }
    ).write_parquet(config.dataset.dir / "runs.parquet")
    row = read(config, forecast)["prediction_quality"]["truth_qc"]
    assert row["status"] == "partial" and row["freshness"]["status"] == "fresh"
    assert row["historical_alerts"]["drift_alert"] is False
    assert row["current_alerts"] is None
    assert "latest_truth_qc_refresh_failed" in row["limitations"]
