import json

import numpy as np
import polars as pl
import pytest
from conftest import synthetic_hourly_matrix

from grounded_weather_forecast.blenders import get_factory
from grounded_weather_forecast.blenders.experts import OnlineExperts
from grounded_weather_forecast.blenders.grounding import AffineGrounding
from grounded_weather_forecast.contracts import hourly_variable
from grounded_weather_forecast.dataset.matrix import to_supervised_slice
from grounded_weather_forecast.metrics.deterministic import mae

TEMP = hourly_variable("temp_c")
SCHEMES = ("ewa", "boa")


def grounded_expert_mae(train, expert_index):
    corrected = AffineGrounding().fit(train).transform(train.x)
    column = corrected[:, expert_index]
    usable = ~np.isnan(column)
    return mae(column[usable], train.y[usable])


class TestOnlineExperts:
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_weights_concentrate_on_better_expert(self, scheme):
        matrix = synthetic_hourly_matrix(
            days=40, noise_sd={"alpha": 4.0, "beta": 0.3}, seed=11
        )
        train = to_supervised_slice(matrix, TEMP)
        blender = get_factory(scheme)().fit(train)
        assert isinstance(blender, OnlineExperts)
        weights = blender.bucket_weights("6-12h")
        assert weights is not None
        assert weights[1] > 0.75  # beta (index 1) is clearly better

    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_regret_close_to_best_expert(self, scheme):
        matrix = synthetic_hourly_matrix(
            days=40, noise_sd={"alpha": 3.0, "beta": 0.5}, seed=12
        )
        train = to_supervised_slice(matrix, TEMP)
        blender = get_factory(scheme)().fit(train)
        blended_mae = mae(blender.predict(train.x).point, train.y)
        best_expert_mae = grounded_expert_mae(train, 1)
        equal_weight_mae = mae(
            get_factory("grounded_equal_weight")().fit(train).predict(train.x).point,
            train.y,
        )
        # Tracks the best expert (the point of the regret bound) rather than
        # paying the equal-weight penalty for carrying a much noisier source.
        assert blended_mae <= best_expert_mae * 1.15
        assert blended_mae < equal_weight_mae * 0.6

    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_adapts_to_drift(self, scheme):
        # alpha is good in the first half, then degrades sharply; beta reverse.
        # Without fixed share, an early leader can never be overtaken.
        matrix = synthetic_hourly_matrix(days=60, noise_sd=0.3, seed=13)
        half = matrix.height // 2
        rng = np.random.default_rng(13)
        alpha = matrix["fx__alpha__temp_c"].to_numpy().copy()
        beta = matrix["fx__beta__temp_c"].to_numpy().copy()
        alpha[half:] += rng.normal(0.0, 6.0, matrix.height - half)
        beta[:half] += rng.normal(0.0, 6.0, half)
        drifted = matrix.with_columns(
            pl.Series("fx__alpha__temp_c", alpha),
            pl.Series("fx__beta__temp_c", beta),
        )
        train = to_supervised_slice(drifted, TEMP)
        blender = get_factory(scheme)().fit(train)
        weights = blender.bucket_weights("6-12h")
        assert weights is not None
        assert weights[1] > 0.6  # ended favoring beta (good in second half)

    def test_sleeping_source_raggedness(self):
        matrix = synthetic_hourly_matrix(
            days=30, noise_sd={"alpha": 2.0, "beta": 0.3}, beta_max_lead=24, seed=14
        )
        train = to_supervised_slice(matrix, TEMP)
        blender = get_factory("boa")().fit(train)
        point = blender.predict(train.x).point
        long_lead = train.x.lead_hours > 24.0
        # beta is asleep beyond 24h; predictions still exist from alpha alone
        assert np.isfinite(point[long_lead]).all()
        short_weights = blender.bucket_weights("6-12h")
        assert short_weights is not None
        assert short_weights[1] > 0.6  # beta trusted where it is awake

    def test_state_serializable(self):
        matrix = synthetic_hourly_matrix(days=10)
        train = to_supervised_slice(matrix, TEMP)
        blender = get_factory("ewa")().fit(train)
        assert isinstance(blender, OnlineExperts)
        encoded = json.dumps(blender.to_state())
        assert "buckets" in encoded
