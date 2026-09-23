from dataclasses import replace
from datetime import timedelta

import pytest
from conftest import utc, write_config

import grounded_weather_forecast.serve.publish as publish_module
import grounded_weather_forecast.cli as cli_module
from grounded_weather_forecast.cli import main
from grounded_weather_forecast.serve.history import load_history
from grounded_weather_forecast.serve.publish import publish_document, schedule_recovery
from grounded_weather_forecast.serve.schema import Forecast, HourlyPoint

NOW = utc(2026, 8, 9, 12)


def forecast(*, status="ready", reason=None, issued_at=NOW):
    return Forecast(
        schema_version=5,
        issued_at=issued_at.isoformat(),
        latitude=34.0,
        longitude=-117.0,
        dataset_fingerprint="dataset-a",
        sources=["alpha"],
        observation_at=None,
        minutely=[],
        hourly=[
            HourlyPoint(
                valid_time=(issued_at + timedelta(hours=1)).isoformat(),
                lead_hours=1.0,
                lead_bucket="0-1h",
                values={"temp_c": 10.0},
                methods={"temp_c": "equal_weight"},
            )
        ],
        daily=[],
        status=status,
        status_reason=reason,
    )


def test_ready_document_replaces_output_and_appends_history(tmp_path):
    destination = tmp_path / "forecast.json"
    history = tmp_path / "history.parquet"
    destination.write_text("old", encoding="utf-8")

    outcome = publish_document(forecast(), destination, history)

    assert outcome.action == "published_ready"
    assert outcome.history_rows == 1
    assert Forecast.from_json(destination.read_text(encoding="utf-8")).status == "ready"
    assert load_history(history).height == 1


def test_degraded_candidate_holds_existing_ready_without_history(tmp_path):
    destination = tmp_path / "forecast.json"
    history = tmp_path / "history.parquet"
    original = forecast().to_json()
    destination.write_text(original, encoding="utf-8")

    outcome = publish_document(
        forecast(status="degraded", reason="no evidence"),
        destination,
        history,
    )

    assert outcome.action == "held_last_good"
    assert destination.read_text(encoding="utf-8") == original
    assert not history.exists()


def test_degraded_candidate_publishes_when_no_last_good_exists(tmp_path):
    destination = tmp_path / "forecast.json"
    history = tmp_path / "history.parquet"

    outcome = publish_document(
        forecast(status="degraded", reason="cold start"),
        destination,
        history,
    )

    assert outcome.action == "published_degraded"
    assert Forecast.from_json(destination.read_text(encoding="utf-8")).status == (
        "degraded"
    )
    assert load_history(history).height == 1


def test_recovery_is_deduped_for_six_hours_and_resets_on_signature_change(
    tmp_path, monkeypatch
):
    config = write_config(tmp_path)
    calls = []

    class Process:
        pid = 123

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    monkeypatch.setattr(publish_module.subprocess, "Popen", popen)
    degraded = forecast(status="degraded", reason="cold start")

    assert schedule_recovery(config, tmp_path / "config.toml", degraded, now=NOW)
    assert not schedule_recovery(
        config,
        tmp_path / "config.toml",
        degraded,
        now=NOW + timedelta(hours=5, minutes=59),
    )
    assert schedule_recovery(
        config,
        tmp_path / "config.toml",
        degraded,
        now=NOW + timedelta(hours=6),
    )
    changed = forecast(status="degraded", reason="promotion gates failed")
    assert schedule_recovery(
        config,
        tmp_path / "config.toml",
        changed,
        now=NOW + timedelta(hours=6, minutes=1),
    )
    assert len(calls) == 3
    assert calls[0][0][-1] == "recover"
    assert (config.artifacts_dir / "auto-restore.log").exists()


def test_changed_dataset_identity_is_immediately_eligible(tmp_path, monkeypatch):
    config = write_config(tmp_path)
    calls = []

    class Process:
        pid = 123

    monkeypatch.setattr(
        publish_module.subprocess,
        "Popen",
        lambda *args, **kwargs: calls.append((args, kwargs)) or Process(),
    )
    degraded = forecast(status="degraded", reason="no release")

    assert schedule_recovery(config, tmp_path / "config.toml", degraded, now=NOW)
    assert schedule_recovery(
        config,
        tmp_path / "config.toml",
        replace(degraded, dataset_fingerprint="dataset-b"),
        now=NOW + timedelta(minutes=1),
    )
    assert len(calls) == 2


def test_failed_spawn_does_not_consume_the_cooldown(tmp_path, monkeypatch):
    config = write_config(tmp_path)
    degraded = forecast(status="degraded", reason="no release")
    monkeypatch.setattr(
        publish_module.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("spawn failed")),
    )

    with pytest.raises(OSError, match="spawn failed"):
        schedule_recovery(config, tmp_path / "config.toml", degraded, now=NOW)

    assert not (config.artifacts_dir / "auto-restore.json").exists()


def test_ready_document_never_schedules_recovery(tmp_path, monkeypatch):
    config = write_config(tmp_path)
    monkeypatch.setattr(
        publish_module.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError),
    )
    assert not schedule_recovery(
        config,
        tmp_path / "config.toml",
        forecast(),
        now=NOW,
    )


def test_publish_command_returns_failure_when_recovery_cannot_spawn(
    tmp_path, monkeypatch
):
    write_config(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "_forecast_document",
        lambda *args, **kwargs: forecast(status="degraded", reason="no release"),
    )
    monkeypatch.setattr(
        publish_module.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("spawn failed")),
    )
    destination = tmp_path / "forecast.json"

    code = main(
        [
            "--config",
            str(tmp_path / "config.toml"),
            "publish",
            "--out",
            str(destination),
        ]
    )

    assert code == 1
    assert destination.exists()
