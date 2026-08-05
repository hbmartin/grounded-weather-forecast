"""Daily Tmax heads: direct marginal EMOS vs grounded path-extreme ensemble."""

import numpy as np
import polars as pl

from grounded_weather_forecast.blenders import get_factory
from grounded_weather_forecast.blenders.registry import supports_product
from grounded_weather_forecast.contracts import (
    ForecastMatrix,
    Product,
    SourceKind,
    SupervisedSlice,
    daily_variable,
    hourly_variable,
)
from grounded_weather_forecast.metrics.deterministic import mae

TMAX = daily_variable("temp_max_c")
TEMP = hourly_variable("temp_c")


def daily_slice(n=400, member_bias=(-3.0, -2.0, -4.0), with_path=True, seed=17):
    """Provider daily values run cold; path extremes carry the real signal."""
    rng = np.random.default_rng(seed)
    truth = 30.0 + rng.normal(0.0, 3.0, n)
    values = np.column_stack([truth - 4.0 + rng.normal(0.0, 1.0, n) for _ in range(3)])
    features = {"valid_hour_local": [0] * n}
    if with_path:
        for index, bias in enumerate(member_bias):
            features[f"path__s{index}__max"] = truth + bias + rng.normal(0.0, 0.5, n)
    x = ForecastMatrix.build(
        sources=("a", "b", "c"),
        values=values,
        lead_hours=rng.uniform(0.0, 240.0, n),  # daily leads, in hours
        features=pl.DataFrame(features),
        product=Product.DAILY,
    )
    return SupervisedSlice(x=x, y=truth, variable=TMAX, source_kind=SourceKind.LIVE)


class TestDailyMarginalEmos:
    def test_beats_the_cold_base_blend(self):
        train = daily_slice()
        result = get_factory("daily_marginal_emos")().fit(train).predict(train.x)
        base = np.nanmean(train.x.values, axis=1)
        assert mae(result.point, train.y) < mae(base, train.y)
        assert result.quantiles is not None

    def test_survives_a_pathless_matrix(self):
        train = daily_slice(with_path=False)
        result = get_factory("daily_marginal_emos")().fit(train).predict(train.x)
        assert result.point.shape == (train.x.n_rows,)


class TestDailyPathExtreme:
    def test_grounding_removes_member_bias(self):
        train = daily_slice()
        result = get_factory("daily_path_extreme")().fit(train).predict(train.x)
        assert mae(result.point, train.y) < 1.0
        assert result.quantiles is not None
        assert np.isfinite(result.quantiles[np.isfinite(result.point)]).all()

    def test_abstains_without_path_features(self):
        train = daily_slice(with_path=False)
        result = get_factory("daily_path_extreme")().fit(train).predict(train.x)
        assert np.isnan(result.point).all()
        assert result.quantiles is None

    def test_state_reports_members_and_biases(self):
        train = daily_slice()
        state = get_factory("daily_path_extreme")().fit(train).to_state()
        assert state["members"] == [
            "path__s0__max",
            "path__s1__max",
            "path__s2__max",
        ]
        biases = state["biases"]["global"]
        np.testing.assert_allclose(biases, [3.0, 2.0, 4.0], atol=0.2)


class TestScope:
    def test_daily_heads_are_daily_scoped(self):
        for method_id in ("daily_marginal_emos", "daily_path_extreme"):
            assert supports_product(method_id, Product.DAILY, TMAX)
            assert not supports_product(method_id, Product.HOURLY, TEMP)
