import importlib
import json
from dataclasses import replace
from datetime import timedelta, timezone

import polars as pl
import pytest
from conftest import utc
from test_predict import build_fixture

from grounded_weather_forecast.cli import main
from grounded_weather_forecast.contracts import TruthSemantics
from grounded_weather_forecast.serve.history import append_history
from grounded_weather_forecast.serve.shadow_capture import capture_issue


@pytest.fixture
def config(tmp_path):
    return build_fixture(tmp_path)


def test_prediction_exports_exact_inputs_without_future_labels(config, tmp_path):
    module = importlib.import_module("grounded_weather_forecast.serve.predict")
    now = utc(2026, 3, 22, 17)
    document = module.predict(
        config,
        {},
        now=now,
        semantics=TruthSemantics.INTERVAL_MEAN,
        shadow_capture=tmp_path / "capsules",
    )
    capsule = tmp_path / "capsules" / "20260322T170000Z"
    assert json.loads((capsule / "production.json").read_text()) == json.loads(
        document.to_json()
    )
    issue = pl.read_parquet(capsule / "issue.parquet")
    assert not any(c.startswith("t__") for c in issue.columns)
    assert issue["issue_time"].unique().to_list() == [now]
    assert [t.isoformat() for t in issue["valid_time"]] == [
        p.valid_time for p in document.hourly
    ]
    metadata = json.loads((capsule / "capsule.json").read_text())
    assert metadata["capture_kind"] == "historical_reconstruction"
    assert "not separately recorded" in metadata["provenance"]["station_receipt_clock"]
    assert "provider_completions.parquet" in metadata["input_sha256"]


def test_capture_failure_leaves_forecast_values_unchanged(
    config, tmp_path, monkeypatch, caplog
):
    module = importlib.import_module("grounded_weather_forecast.serve.predict")
    now = utc(2026, 3, 22, 17)
    original = module.predict(config, {}, now=now)

    def denied(*_args, **_kwargs):
        raise PermissionError("fixture capture denied")

    monkeypatch.setattr(module, "capture_issue", denied)
    result = module.predict(config, {}, now=now, shadow_capture=tmp_path / "capsules")
    assert result.to_json() == original.to_json()
    assert "Shadow input capture failed" in caplog.text


def test_capture_is_exclusive(config, tmp_path):
    module = importlib.import_module("grounded_weather_forecast.serve.predict")
    now = utc(2026, 3, 22, 17)
    snapshot = module.build_snapshot(config, now, capture=True)
    document = module.predict(
        config, {}, now=now, semantics=TruthSemantics.INTERVAL_MEAN
    )
    options = {
        "issue": snapshot.hourly,
        "training": snapshot.hourly,
        "truth": snapshot.truth_hourly,
        "completions": snapshot.completions,
        "historical": True,
    }
    capsule = capture_issue(tmp_path / "capsules", config, document, **options)
    before = {p.name: p.read_bytes() for p in capsule.iterdir()}
    with pytest.raises(FileExistsError, match="already captured"):
        capture_issue(tmp_path / "capsules", config, document, **options)
    assert {p.name: p.read_bytes() for p in capsule.iterdir()} == before


@pytest.mark.parametrize("kind", ["empty", "instantaneous"])
def test_capture_abstains_on_incompatible_inputs(config, tmp_path, kind):
    module = importlib.import_module("grounded_weather_forecast.serve.predict")
    now = utc(2026, 3, 22, 17)
    snapshot = module.build_snapshot(config, now, capture=True)
    document = module.predict(
        config, {}, now=now, semantics=TruthSemantics.INTERVAL_MEAN
    )
    issue = snapshot.hourly
    if kind == "empty":
        issue = issue.head(0)
    else:
        document.hourly[0].truth_semantics["temp_c"] = "inst"
    with pytest.raises(ValueError, match="temperature shadow requires"):
        capture_issue(
            tmp_path / "capsules",
            config,
            document,
            issue=issue,
            training=issue,
            truth=snapshot.truth_hourly,
            completions=snapshot.completions,
            historical=True,
        )
    assert not (tmp_path / "capsules").exists()


def test_capture_uses_utc_for_offset_aware_issuance(config, tmp_path):
    module = importlib.import_module("grounded_weather_forecast.serve.predict")
    now = utc(2026, 3, 22, 17)
    snapshot = module.build_snapshot(config, now, capture=True)
    document = module.predict(
        config, {}, now=now, semantics=TruthSemantics.INTERVAL_MEAN
    )
    document = replace(
        document, issued_at=now.astimezone(timezone(timedelta(hours=-7))).isoformat()
    )
    capsule = capture_issue(
        tmp_path / "capsules",
        config,
        document,
        issue=snapshot.hourly,
        training=snapshot.hourly,
        truth=snapshot.truth_hourly,
        completions=snapshot.completions,
        historical=True,
    )
    assert capsule.name == "20260322T170000Z"


def test_cached_capture_preserves_archived_forecast(config, tmp_path, capsys):
    module = importlib.import_module("grounded_weather_forecast.serve.predict")
    now = utc(2026, 3, 22, 17)
    document = module.predict(config, {}, now=now)
    append_history(document, config.predict.history_path)
    assert (
        main(
            [
                "--config",
                str(tmp_path / "config.toml"),
                "predict",
                "--now",
                now.isoformat(),
                "--shadow-capture",
                str(tmp_path / "capsules"),
                "--no-history",
            ]
        )
        == 0
    )
    assert not (tmp_path / "capsules").exists()
