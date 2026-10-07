import json
from dataclasses import replace
from datetime import timedelta

import pytest
from conftest import utc, write_config

import grounded_weather_forecast.cli as cli_module
import grounded_weather_forecast.serve.publish as publish_module
from grounded_weather_forecast.cli import main
from grounded_weather_forecast.serve.history import load_history
from grounded_weather_forecast.serve.publish import (
    publish_document,
    record_publish_attempt,
    schedule_recovery,
)
from grounded_weather_forecast.serve.schema import Forecast, HourlyPoint
from grounded_weather_forecast.serve.dataset_snapshot import DatasetIntegrityError

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
        now=NOW,
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


@pytest.mark.parametrize(
    "previous",
    [
        forecast(issued_at=NOW - timedelta(hours=6, seconds=1)),
        replace(forecast(), latitude=35.0),
        replace(forecast(), timezone="America/Los_Angeles"),
        replace(forecast(), schema_version=4),
    ],
)
def test_stale_or_incompatible_ready_document_is_not_held(tmp_path, previous):
    destination = tmp_path / "forecast.json"
    destination.write_text(previous.to_json(), encoding="utf-8")
    candidate = forecast(status="degraded", reason="no evidence")
    outcome = publish_document(
        candidate, destination, tmp_path / "history.parquet", now=NOW
    )
    assert outcome.action == "published_degraded"
    assert (
        Forecast.from_json(destination.read_text(encoding="utf-8")).status == "degraded"
    )


def test_history_failure_after_visible_publication_returns_outcome(
    tmp_path, monkeypatch
):
    destination = tmp_path / "forecast.json"
    monkeypatch.setattr(
        publish_module,
        "append_history",
        lambda *_args: (_ for _ in ()).throw(OSError("history unavailable")),
    )
    outcome = publish_document(
        forecast(), destination, tmp_path / "history.parquet", now=NOW
    )
    assert outcome.action == "published_ready"
    assert "history unavailable" in (outcome.history_error or "")
    assert Forecast.from_json(destination.read_text(encoding="utf-8")).status == "ready"


def test_symlink_output_preserves_link_and_replaces_target(tmp_path):
    target = tmp_path / "real.json"
    target.write_text("old", encoding="utf-8")
    link = tmp_path / "forecast.json"
    link.symlink_to(target)
    outcome = publish_document(forecast(), link, tmp_path / "history.parquet", now=NOW)
    assert outcome.action == "published_ready"
    assert link.is_symlink()
    assert Forecast.from_json(target.read_text(encoding="utf-8")).status == "ready"


def test_publish_attempt_records_held_degradation(tmp_path):
    config = write_config(tmp_path)
    target = tmp_path / "forecast.json"
    target.write_text(forecast().to_json(), encoding="utf-8")
    candidate = forecast(status="degraded", reason="no evidence")
    outcome = publish_document(candidate, target, tmp_path / "history.parquet", now=NOW)
    record_publish_attempt(config, candidate, outcome, target, now=NOW)
    attempt = json.loads(
        (config.artifacts_dir / "latest_publish_attempt.json").read_text()
    )
    assert attempt["candidate_status"] == "degraded"
    assert attempt["action"] == "held_last_good"
    assert attempt["status_reason"] == "no evidence"


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
    assert "recover" in calls[0][0]
    assert (config.artifacts_dir / "auto-restore.log").exists()


def test_changed_dataset_identity_keeps_six_hour_cooldown(tmp_path, monkeypatch):
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
    assert not schedule_recovery(
        config,
        tmp_path / "config.toml",
        replace(degraded, dataset_fingerprint="dataset-b"),
        now=NOW + timedelta(minutes=1),
    )
    assert len(calls) == 1


def test_recovery_profile_is_forwarded_to_detached_command(tmp_path, monkeypatch):
    config = write_config(tmp_path)
    commands = []

    class Process:
        pid = 123

    monkeypatch.setattr(
        publish_module.subprocess,
        "Popen",
        lambda command, **_kwargs: commands.append(command) or Process(),
    )
    assert schedule_recovery(
        config,
        tmp_path / "config.toml",
        forecast(status="degraded", reason="no evidence"),
        now=NOW,
        semantics="mean",
        methods="equal_weight",
        window="rolling",
    )
    command = commands[0]
    assert command[command.index("--semantics") + 1] == "mean"
    assert command[command.index("--methods") + 1] == "equal_weight"
    assert command[command.index("--window") + 1] == "rolling"


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


def test_publish_history_failure_still_records_attempt_and_schedules_recovery(
    tmp_path, monkeypatch, capsys
):
    config = write_config(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "_forecast_document",
        lambda *_args, **_kwargs: forecast(status="degraded", reason="no release"),
    )
    monkeypatch.setattr(
        publish_module,
        "append_history",
        lambda *_args: (_ for _ in ()).throw(OSError("history unavailable")),
    )
    scheduled = []
    monkeypatch.setattr(
        publish_module,
        "schedule_recovery",
        lambda *_args, **kwargs: scheduled.append(kwargs) or True,
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
    assert scheduled
    assert "document published but history failed" in capsys.readouterr().err
    assert (config.artifacts_dir / "latest_publish_attempt.json").exists()


def test_publish_corrupt_dataset_schedules_repair(tmp_path, monkeypatch):
    write_config(tmp_path)

    def corrupt(*_args, **_kwargs):
        raise DatasetIntegrityError("hourly_matrix differs")

    monkeypatch.setattr(cli_module, "_forecast_document", corrupt)
    reasons = []
    monkeypatch.setattr(
        publish_module,
        "schedule_recovery",
        lambda *_args, **kwargs: reasons.append(kwargs["reason"]) or True,
    )
    code = main(
        [
            "--config",
            str(tmp_path / "config.toml"),
            "publish",
            "--out",
            str(tmp_path / "forecast.json"),
        ]
    )
    assert code == 1
    assert reasons == ["live dataset unreadable"]


def test_recover_recheck_treats_corrupt_dataset_as_repairable(tmp_path, monkeypatch):
    config = write_config(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "_forecast_document",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            DatasetIntegrityError("corrupt parquet")
        ),
    )
    assert cli_module._recovery_needed(config)


def test_predict_records_local_output_releases_through_symlink(tmp_path, monkeypatch):
    config = write_config(tmp_path)
    document = replace(forecast(), release_ids=["r1", "r2"])
    monkeypatch.setattr(
        cli_module, "_forecast_document", lambda *_args, **_kwargs: document
    )
    target = tmp_path / "real.json"
    target.write_text("old", encoding="utf-8")
    link = tmp_path / "forecast.json"
    link.symlink_to(target)
    assert (
        main(
            [
                "--config",
                str(tmp_path / "config.toml"),
                "predict",
                "--out",
                str(link),
                "--no-history",
            ]
        )
        == 0
    )
    assert link.is_symlink()
    active = json.loads((config.artifacts_dir / "active_release.json").read_text())
    assert active["release_ids"] == ["r1", "r2"]
    assert active["output_path"] == str(target)
