"""Quantile completion: point-only winners get residual bands, guarded by
the same provenance rules the promotion path enforces."""

import numpy as np
import polars as pl
import pytest
from conftest import write_config

import grounded_weather_forecast.serve.dressing as dressing_module
from grounded_weather_forecast.backtest.scores import SCORES_SCHEMA, write_scores
from grounded_weather_forecast.contracts import TruthSemantics, hourly_variable
from grounded_weather_forecast.serve.dressing import (
    DRESSING_LEVELS,
    ResidualDresser,
    corrected_error_quantiles,
    dress_variable,
)
from grounded_weather_forecast.serve.schema import Forecast, HourlyPoint
from grounded_weather_forecast.serve.selection import Selection
from grounded_weather_forecast.timeutil import utc

HUMIDITY = hourly_variable("humidity_pct")
POP = hourly_variable("pop")


def make_scores(
    method_id="equal_weight",
    n=40,
    bias=-5.0,
    bucket="168-240h",
    evaluation_id="eval1",
    created=None,
    source_kind="live",
    variable="humidity_pct",
    config_fingerprint="cfg1",
):
    created = created if created is not None else utc(2026, 8, 1)
    y_true = np.linspace(40.0, 60.0, n)

    def constant(value):
        return [value] * n

    return pl.DataFrame(
        {
            "method_id": constant(method_id),
            "variable": constant(variable),
            "product": constant("hourly"),
            "source_kind": constant(source_kind),
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
            "lead_hours": constant(200.0),
            "lead_bucket": constant(bucket),
            "y_pred": (y_true - bias),
            "y_true": y_true,
            "quantile_levels_json": constant("[]"),
            "quantiles_json": constant(None),
        }
    ).cast(SCORES_SCHEMA)


@pytest.fixture
def env(tmp_path, monkeypatch):
    config = write_config(tmp_path)
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    monkeypatch.setattr(dressing_module, "dataset_fingerprint", lambda _config: "ds1")
    monkeypatch.setattr(dressing_module, "config_fingerprint", lambda _config: "cfg1")
    monkeypatch.setattr(dressing_module, "code_identity", lambda: "code1")
    return config, scores_dir


def dress_one(
    config,
    scores_dir,
    variable=HUMIDITY,
    method_id="equal_weight",
    bucket="168-240h",
    point_value=50.0,
    native=None,
):
    dresser = ResidualDresser(scores_dir, "hourly", config)
    point = np.array([point_value])
    quantiles = [dict(native) if native else {}]
    sources = dress_variable(
        dresser,
        variable,
        TruthSemantics.INSTANTANEOUS,
        [Selection(method_id, reason="test")],
        [bucket],
        point,
        quantiles,
    )
    return quantiles[0], sources[0]


class TestCorrectedErrorQuantiles:
    def test_conformal_rank_rule(self):
        errors = np.arange(1.0, 100.0)  # n = 99, ranks are transparent
        values = corrected_error_quantiles(errors)
        expected = {0.05: 5.0, 0.1: 10.0, 0.25: 25.0, 0.75: 75.0, 0.9: 90.0, 0.95: 95.0}
        for level, value in zip(DRESSING_LEVELS, values, strict=True):
            assert value == expected[level]

    def test_extreme_ranks_clip_to_sample(self):
        errors = np.arange(1.0, 25.0)  # n = 24, ceil(25 * 0.95) = 24
        values = corrected_error_quantiles(errors)
        assert values[-1] == 24.0
        assert values[0] == 1.0


class TestResidualDressing:
    def test_bias_lands_inside_the_band(self, env):
        config, scores_dir = env
        write_scores(
            make_scores(bias=-5.0), scores_dir / "scores_hourly_live_e1.parquet"
        )
        quantiles, source = dress_one(config, scores_dir)
        assert source == "dressed_bucket"
        # error = y_true - y_pred is exactly -5, so the whole band sits below
        # the point: the asymmetry a symmetric radius would hide.
        assert quantiles["0.9"] == pytest.approx(45.0)
        assert quantiles["0.1"] == pytest.approx(45.0)

    def test_bucket_pool_falls_back_to_variable_global(self, env):
        config, scores_dir = env
        write_scores(
            make_scores(bucket="24-48h"), scores_dir / "scores_hourly_live_e1.parquet"
        )
        quantiles, source = dress_one(config, scores_dir, bucket="168-240h")
        assert source == "dressed_variable"
        assert quantiles

    def test_pool_floor(self, env):
        config, scores_dir = env
        write_scores(make_scores(n=23), scores_dir / "scores_hourly_live_e1.parquet")
        quantiles, source = dress_one(config, scores_dir)
        assert quantiles == {}
        assert source is None
        write_scores(make_scores(n=24), scores_dir / "scores_hourly_live_e2.parquet")
        quantiles, source = dress_one(config, scores_dir)
        assert source == "dressed_bucket"

    def test_native_quantiles_untouched(self, env):
        config, scores_dir = env
        write_scores(make_scores(), scores_dir / "scores_hourly_live_e1.parquet")
        native = {"0.5": 51.0}
        quantiles, source = dress_one(config, scores_dir, native=native)
        assert quantiles == native
        assert source is None

    def test_synthetic_rows_never_pool(self, env):
        config, scores_dir = env
        write_scores(
            make_scores(source_kind="synthetic"),
            scores_dir / "scores_hourly_live_e1.parquet",
        )
        quantiles, source = dress_one(config, scores_dir)
        assert quantiles == {}
        assert source is None

    def test_stale_config_fingerprint_never_pools(self, env):
        config, scores_dir = env
        write_scores(
            make_scores(config_fingerprint="other"),
            scores_dir / "scores_hourly_live_e1.parquet",
        )
        quantiles, source = dress_one(config, scores_dir)
        assert quantiles == {}

    def test_only_newest_evaluation_pools(self, env):
        config, scores_dir = env
        stale = make_scores(bias=-5.0, evaluation_id="eval1", created=utc(2026, 7, 1))
        fresh = make_scores(bias=2.0, evaluation_id="eval2", created=utc(2026, 8, 1))
        write_scores(
            pl.concat([stale, fresh]), scores_dir / "scores_hourly_live_e2.parquet"
        )
        quantiles, _source = dress_one(config, scores_dir)
        assert quantiles["0.9"] == pytest.approx(52.0)

    def test_probability_band_clamped_to_unit_interval(self, env):
        config, scores_dir = env
        write_scores(
            make_scores(variable="pop", bias=0.3),
            scores_dir / "scores_hourly_live_e1.parquet",
        )
        quantiles, _source = dress_one(
            config, scores_dir, variable=POP, point_value=0.9
        )
        assert quantiles["0.95"] == 1.0

    def test_corrupt_file_disables_quietly(self, env):
        config, scores_dir = env
        (scores_dir / "scores_hourly_live_bad.parquet").write_bytes(b"not parquet")
        quantiles, source = dress_one(config, scores_dir)
        assert quantiles == {}
        assert source is None

    def test_missing_scores_dir_dresses_nothing(self, env):
        config, scores_dir = env
        quantiles, source = dress_one(config, scores_dir / "absent", point_value=50.0)
        assert quantiles == {}
        assert source is None


class TestSchemaRoundTrip:
    def test_quantiles_source_survives_json(self):
        point = HourlyPoint(
            valid_time="2026-08-04T00:00:00+00:00",
            lead_hours=3.0,
            quantiles={"temp_c": {"0.1": 20.0, "0.9": 24.0}},
            quantiles_source={"temp_c": "dressed_bucket"},
        )
        document = Forecast(
            schema_version=5,
            issued_at="2026-08-04T00:00:00+00:00",
            latitude=34.0,
            longitude=-117.0,
            dataset_fingerprint="ds1",
            sources=["nws"],
            observation_at=None,
            minutely=[],
            hourly=[point],
            daily=[],
        )
        restored = Forecast.from_json(document.to_json())
        assert restored.hourly[0].quantiles_source == {"temp_c": "dressed_bucket"}

    def test_schema_v4_documents_still_load(self):
        legacy = {"valid_time": "2026-08-04T00:00:00+00:00", "lead_hours": 3.0}
        assert HourlyPoint(**legacy).quantiles_source == {}
