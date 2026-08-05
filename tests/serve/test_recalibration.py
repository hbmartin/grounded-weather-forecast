"""Serve-side quantile recalibration: provenance-guarded, never-fail."""

import json

import numpy as np
import polars as pl
import pytest
from conftest import write_config

import grounded_weather_forecast.serve.recalibration as recal_module
from grounded_weather_forecast.backtest.scores import SCORES_SCHEMA, write_scores
from grounded_weather_forecast.contracts import TruthSemantics, hourly_variable
from grounded_weather_forecast.serve.recalibration import (
    QuantileRecalibrator,
    recalibrate_variable,
)
from grounded_weather_forecast.serve.selection import Selection
from grounded_weather_forecast.timeutil import utc

TEMP = hourly_variable("temp_c")
LEVELS = (0.05, 0.1, 0.25, 0.75, 0.9, 0.95)
_Z = (-1.645, -1.282, -0.674, 0.674, 1.282, 1.645)


def make_scores(
    method_id="idr",
    n=60,
    bias=5.0,
    bucket="12-24h",
    evaluation_id="eval1",
    created=None,
    config_fingerprint="cfg1",
):
    """Native grids one unit wide around a point that runs ``bias`` cold."""
    created = created if created is not None else utc(2026, 8, 1)
    y_true = np.linspace(10.0, 30.0, n)
    y_pred = y_true - bias

    def constant(value):
        return [value] * n

    return pl.DataFrame(
        {
            "method_id": constant(method_id),
            "variable": constant("temp_c"),
            "product": constant("hourly"),
            "source_kind": constant("live"),
            "evaluation_id": constant(evaluation_id),
            "evaluation_created_at": constant(created),
            "dataset_fingerprint": constant("ds1"),
            "source_set_json": constant("[]"),
            "feature_set_json": constant("[]"),
            "semantics": constant("inst"),
            "code_version": constant("code1"),
            "config_fingerprint": constant(config_fingerprint),
            "window": constant("expanding"),
            "fold_origin": constant(created),
            "issue_time": constant(created),
            "valid_time": constant(created),
            "lead_hours": constant(18.0),
            "lead_bucket": constant(bucket),
            "y_pred": y_pred,
            "y_true": y_true,
            "quantile_levels_json": constant(json.dumps(list(LEVELS))),
            "quantiles_json": [
                json.dumps([float(point + z) for z in _Z]) for point in y_pred
            ],
        }
    ).cast(SCORES_SCHEMA)


@pytest.fixture
def env(tmp_path, monkeypatch):
    config = write_config(tmp_path)
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    monkeypatch.setattr(recal_module, "dataset_fingerprint", lambda _config: "ds1")
    monkeypatch.setattr(recal_module, "config_fingerprint", lambda _config: "cfg1")
    monkeypatch.setattr(recal_module, "code_identity", lambda: "code1")
    return config, scores_dir


def native_row(point=20.0):
    return {str(level): point + z for level, z in zip(LEVELS, _Z, strict=True)}


def recalibrate_one(
    config,
    scores_dir,
    mode="cqr",
    method_id="idr",
    bucket="12-24h",
    native=None,
    existing=None,
):
    recalibrator = QuantileRecalibrator(scores_dir, "hourly", config, mode)
    quantiles = [dict(native) if native is not None else native_row()]
    sources = recalibrate_variable(
        recalibrator,
        TEMP,
        TruthSemantics.INSTANTANEOUS,
        [Selection(method_id, reason="test")],
        [bucket],
        quantiles,
        existing,
    )
    return quantiles[0], sources[0]


class TestTransformApplication:
    def test_cqr_shifts_the_band_onto_the_bias(self, env):
        config, scores_dir = env
        write_scores(make_scores(), scores_dir / "scores_hourly_live_a.parquet")
        native = native_row()
        transformed, label = recalibrate_one(config, scores_dir, native=native)
        assert label == "recalibrated_cqr_bucket"
        # pool truth ran +5 above the band; margins must lift the upper tail
        assert transformed["0.95"] > native["0.95"] + 3.0
        values = [transformed[str(level)] for level in LEVELS]
        assert values == sorted(values)

    def test_pit_mode_labels_and_stays_within_the_grid(self, env):
        config, scores_dir = env
        write_scores(make_scores(), scores_dir / "scores_hourly_live_a.parquet")
        native = native_row()
        transformed, label = recalibrate_one(
            config, scores_dir, mode="pit", native=native
        )
        assert label == "recalibrated_pit_bucket"
        assert max(transformed.values()) <= native["0.95"] + 1e-9

    def test_thin_bucket_borrows_the_variable_pool(self, env):
        config, scores_dir = env
        write_scores(
            pl.concat([make_scores(n=30), make_scores(n=60, bucket="24-48h")]),
            scores_dir / "scores_hourly_live_a.parquet",
        )
        _, label = recalibrate_one(config, scores_dir)
        assert label == "recalibrated_cqr_variable"


class TestGuards:
    def test_dressed_rows_are_never_touched(self, env):
        config, scores_dir = env
        write_scores(make_scores(), scores_dir / "scores_hourly_live_a.parquet")
        native = native_row()
        transformed, label = recalibrate_one(
            config, scores_dir, native=native, existing=("dressed_bucket",)
        )
        assert label == "dressed_bucket"
        assert transformed == native

    def test_fingerprint_mismatch_is_a_noop(self, env):
        config, scores_dir = env
        write_scores(
            make_scores(config_fingerprint="other"),
            scores_dir / "scores_hourly_live_a.parquet",
        )
        native = native_row()
        transformed, label = recalibrate_one(config, scores_dir, native=native)
        assert label is None
        assert transformed == native

    def test_corrupt_scores_disable_without_raising(self, env):
        config, scores_dir = env
        (scores_dir / "scores_hourly_live_a.parquet").write_text("not parquet")
        transformed, label = recalibrate_one(config, scores_dir)
        assert label is None

    def test_level_mismatch_is_skipped(self, env):
        config, scores_dir = env
        write_scores(make_scores(), scores_dir / "scores_hourly_live_a.parquet")
        native = {"0.25": 19.0, "0.75": 21.0}
        transformed, label = recalibrate_one(config, scores_dir, native=native)
        assert label is None
        assert transformed == native

    def test_point_only_rows_are_skipped(self, env):
        config, scores_dir = env
        write_scores(make_scores(), scores_dir / "scores_hourly_live_a.parquet")
        transformed, label = recalibrate_one(config, scores_dir, native={})
        assert label is None
        assert transformed == {}
