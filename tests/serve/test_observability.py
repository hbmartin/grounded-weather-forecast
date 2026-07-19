from datetime import UTC, datetime, timedelta

import polars as pl
from conftest import synthetic_hourly_matrix, write_config

from grounded_weather_forecast.artifacts import ArtifactStore
from grounded_weather_forecast.blenders import get_factory
from grounded_weather_forecast.contracts import hourly_variable
from grounded_weather_forecast.dataset.matrix import to_supervised_slice
from grounded_weather_forecast.serve.observability import (
    OBSERVABILITY_HISTORY_SCHEMA,
    load_observability_history,
    load_observability_states,
    observability_root,
    snapshot_observability,
)
from grounded_weather_forecast.storage import atomic_write_parquet

TEMP = hourly_variable("temp_c")
ISSUE = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def fitted(method_id="grounded_equal_weight", days=15):
    matrix = synthetic_hourly_matrix(days=days)
    return get_factory(method_id)().fit(to_supervised_slice(matrix, TEMP))


def snap(config, blender, method_id):
    snapshot_observability(
        blender,
        method_id=method_id,
        product="hourly",
        variable="temp_c",
        config=config,
        issue_time=ISSUE,
    )


def test_snapshot_round_trip(tmp_path):
    config = write_config(tmp_path)
    snap(config, fitted(), "grounded_equal_weight")
    snapshots = load_observability_states(config.artifacts_dir)
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.method_id == "grounded_equal_weight"
    assert snapshot.product == "hourly"
    assert snapshot.variable == "temp_c"
    assert snapshot.issue_time == ISSUE.isoformat()
    assert "grounding" in snapshot.state


def test_latest_snapshot_overwrites_in_place(tmp_path):
    config = write_config(tmp_path)
    blender = fitted()
    snap(config, blender, "grounded_equal_weight")
    snap(config, blender, "grounded_equal_weight")
    assert len(load_observability_states(config.artifacts_dir)) == 1


def test_stateless_blender_writes_nothing(tmp_path):
    config = write_config(tmp_path)
    snap(config, fitted("equal_weight"), "equal_weight")
    assert load_observability_states(config.artifacts_dir) == ()
    assert not observability_root(config).exists()


def test_history_appends_only_trajectory_methods(tmp_path):
    config = write_config(tmp_path)
    snap(config, fitted(), "grounded_equal_weight")
    snap(config, fitted("ewa", days=20), "ewa")
    history = load_observability_history(config.artifacts_dir)
    assert history["method_id"].to_list() == ["ewa"]


def test_history_prunes_beyond_rolling_window(tmp_path):
    config = write_config(tmp_path)
    stale_time = datetime.now(tz=UTC) - timedelta(
        days=config.backtest.rolling_window_days + 2
    )
    stale = pl.DataFrame(
        [
            {
                "captured_at": stale_time,
                "issue_time": stale_time,
                "method_id": "ewa",
                "product": "hourly",
                "variable": "temp_c",
                "dataset_fingerprint": "unknown",
                "state_json": "{}",
            }
        ],
        schema=OBSERVABILITY_HISTORY_SCHEMA,
    )
    atomic_write_parquet(stale, observability_root(config) / "history.parquet")
    snap(config, fitted("ewa", days=20), "ewa")
    history = load_observability_history(config.artifacts_dir)
    assert history.height == 1
    assert history["issue_time"].to_list() == [ISSUE]


def test_store_failure_is_swallowed(tmp_path, monkeypatch):
    config = write_config(tmp_path)

    def boom(self, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(ArtifactStore, "save", boom)
    snap(config, fitted(), "grounded_equal_weight")
    assert load_observability_states(config.artifacts_dir) == ()
