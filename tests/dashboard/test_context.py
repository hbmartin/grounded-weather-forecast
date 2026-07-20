import json
from datetime import UTC, datetime

import polars as pl
from conftest import (
    make_forecast_db,
    make_station_db,
    synthetic_hourly_matrix,
    write_config,
)

from grounded_weather_forecast.dashboard.context import collect_context
from grounded_weather_forecast.dataset.matrix import matrix_path

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def test_bare_config_collects_all_absent_without_raising(tmp_path):
    ctx = collect_context(write_config(tmp_path), now=NOW)
    assert ctx.manifest is None
    assert ctx.truth_minute is None
    assert ctx.truth_hourly is None
    assert ctx.hourly_matrix is None
    assert ctx.score_frames == {}
    assert ctx.history is None
    assert ctx.latest_forecast is None
    assert ctx.releases == ()
    assert ctx.alignment is None
    assert ctx.drift is None
    assert ctx.observability_states == ()
    assert ctx.observability_history.is_empty()
    assert ctx.runs.is_empty()


def test_corrupted_manifest_loads_as_none(tmp_path):
    config = write_config(tmp_path)
    config.dataset.dir.mkdir(parents=True, exist_ok=True)
    (config.dataset.dir / "manifest.json").write_text("{not json", encoding="utf-8")
    assert collect_context(config, now=NOW).manifest is None


def test_populated_context_loads_matrix_and_manifest(tmp_path):
    config = write_config(tmp_path)
    config.dataset.dir.mkdir(parents=True, exist_ok=True)
    matrix = synthetic_hourly_matrix(days=5)
    matrix.write_parquet(matrix_path(config.dataset.dir, "hourly", "live"))
    (config.dataset.dir / "manifest.json").write_text(
        json.dumps({"fingerprint": "abc", "sources": ["alpha", "beta"]}),
        encoding="utf-8",
    )
    ctx = collect_context(config, now=NOW)
    assert ctx.hourly_matrix is not None
    assert ctx.hourly_matrix.height == matrix.height
    assert ctx.manifest is not None
    assert ctx.manifest["fingerprint"] == "abc"


def test_context_reads_actual_archive_location(tmp_path):
    config = write_config(tmp_path)
    make_forecast_db(
        config.forecasts.db_path,
        [
            {
                "completed_at": NOW.isoformat(),
                "latitude": 35.0,
                "longitude": -118.0,
                "results": [],
            }
        ],
    )

    assert collect_context(config, now=NOW).archive_location == (35.0, -118.0)


def test_qc_distinguishes_recovered_flatline_from_active_state(tmp_path):
    config = write_config(
        tmp_path,
        extra_toml="\n[qc.flatline_minutes]\ntemp = 2\n",
    )
    make_station_db(
        config.station.db_path,
        [
            ("2026-07-18 04:00:00", {"outTemp": 70.0}),
            ("2026-07-18 04:01:00", {"outTemp": 70.0}),
            ("2026-07-18 04:02:00", {"outTemp": 70.0}),
            ("2026-07-18 04:03:00", {"outTemp": 71.0}),
        ],
    )

    qc = collect_context(config, now=NOW).qc

    assert qc is not None
    temp = qc.filter(pl.col("channel") == "temp").row(0, named=True)
    assert temp["flatline"] > 0
    assert temp["active_flatline"] is False


def test_a_corrupt_artifact_is_distinguishable_from_a_missing_one(tmp_path):
    """Absence is a young archive; a corrupt file is a fault. Not the same."""
    from grounded_weather_forecast.dashboard.context import collect_context
    from grounded_weather_forecast.dataset.matrix import DatasetPaths

    config = write_config(tmp_path)
    paths = DatasetPaths.in_dir(config.dataset.dir)
    config.dataset.dir.mkdir(parents=True, exist_ok=True)

    missing = collect_context(config, now=NOW)
    assert missing.truth_hourly is None
    assert missing.unreadable_artifacts == ()

    paths.truth_hourly.write_bytes(b"not a parquet file")
    corrupt = collect_context(config, now=NOW)
    assert corrupt.truth_hourly is None, "still unusable"
    assert f"dataset/{paths.truth_hourly.name}" in corrupt.unreadable_artifacts


def test_invalid_utf8_json_is_reported_without_aborting(tmp_path):
    config = write_config(tmp_path)
    config.dataset.dir.mkdir(parents=True, exist_ok=True)
    (config.dataset.dir / "manifest.json").write_bytes(b"\xff")

    context = collect_context(config, now=NOW)

    assert context.manifest is None
    assert "dataset/manifest.json" in context.unreadable_artifacts


def test_every_swallowed_artifact_failure_is_reported(tmp_path):
    from grounded_weather_forecast.artifacts import ArtifactStore

    config = write_config(tmp_path)
    config.dataset.dir.mkdir(parents=True, exist_ok=True)

    served = config.predict.history_path.parent / "served_forecasts"
    served.mkdir(parents=True)
    (served / "2026-07-18T12-00-00+00-00.json").write_bytes(b"{broken")

    releases = config.artifacts_dir / "releases"
    releases.mkdir(parents=True)
    (releases / "broken.json").write_bytes(b"{broken")

    scores = config.dataset.dir / "scores"
    scores.mkdir(parents=True)
    (scores / "scores_hourly_live_broken.parquet").write_bytes(b"broken")
    (scores / "scores_hourly_synthetic_broken.parquet").write_bytes(b"broken")

    (config.dataset.dir / "runs.parquet").write_bytes(b"broken")
    observability = config.artifacts_dir / "observability"
    store = ArtifactStore(observability)
    slot = store.save(
        fingerprint="fp",
        method_id="gbm",
        product="hourly",
        variable="temp_c",
        state={},
    )
    (slot / "state.json").write_bytes(b"{broken")
    (observability / "history.parquet").write_bytes(b"broken")

    context = collect_context(config, now=NOW)

    failures = set(context.unreadable_artifacts)
    assert "predict/served_forecasts/2026-07-18T12-00-00+00-00.json" in failures
    assert "artifacts/releases/broken.json" in failures
    assert "dataset/scores/scores_hourly_live_broken.parquet" in failures
    assert "dataset/scores/scores_hourly_synthetic_broken.parquet" in failures
    assert "dataset/runs.parquet" in failures
    assert "artifacts/observability/history.parquet" in failures
    assert any(
        failure.startswith("artifacts/observability/fp/gbm/hourly.temp_c")
        for failure in failures
    )
