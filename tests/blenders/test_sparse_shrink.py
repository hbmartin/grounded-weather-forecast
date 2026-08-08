"""Sparse-source shrink: trust is the source count, nothing is fitted."""

import numpy as np
from conftest import synthetic_hourly_matrix

from grounded_weather_forecast.blenders import get_factory
from grounded_weather_forecast.blenders.baselines import (
    EqualWeight,
    HarmonicClimatology,
)
from grounded_weather_forecast.blenders.registry import supports_product
from grounded_weather_forecast.contracts import (
    ForecastMatrix,
    Product,
    SourceKind,
    SupervisedSlice,
    daily_variable,
    hourly_variable,
)
from grounded_weather_forecast.dataset.matrix import to_supervised_slice

TEMP = hourly_variable("temp_c")


def fitted_on_weather(days=20):
    matrix = synthetic_hourly_matrix(days=days, noise_sd=0.3, seed=9)
    train = to_supervised_slice(matrix, TEMP)
    blender = get_factory("precip_sparse_shrink")().fit(train)
    return blender, train


def test_variable_scope_is_daily_precip_only():
    precip = daily_variable("precip_sum_mm")
    assert supports_product("precip_sparse_shrink", Product.DAILY, precip)
    assert not supports_product(
        "precip_sparse_shrink", Product.HOURLY, hourly_variable("temp_c")
    )
    # The daily-only scope rests entirely on the variable allowlist (there is
    # no product check): this fires if the scope is ever widened to precip_mm.
    assert not supports_product(
        "precip_sparse_shrink", Product.HOURLY, hourly_variable("precip_mm")
    )


def test_weight_is_the_source_count(tmp_path):
    blender, train = fitted_on_weather()
    base = EqualWeight().fit(train).predict(train.x).point
    anchor = HarmonicClimatology().fit(train).predict(train.x).point
    point = blender.predict(train.x).point
    counts = train.x.availability.sum(axis=1)
    # k = 2 pseudo-sources: two providers -> w = 0.5, exactly halfway
    weight = counts / (counts + 2.0)
    expected = weight * base + (1.0 - weight) * anchor
    usable = np.isfinite(base)
    assert np.allclose(point[usable], expected[usable], atol=1e-9)


def test_fewer_sources_pull_harder_toward_climatology():
    blender, train = fitted_on_weather()
    x = train.x
    thinned_values = x.values.copy()
    thinned_values[:, 1] = np.nan  # drop the second source everywhere
    thinned = ForecastMatrix.build(
        x.sources, thinned_values, x.lead_hours, x.features, x.product
    )
    anchor = HarmonicClimatology().fit(train).predict(x).point
    full_point = blender.predict(x).point
    thin_point = blender.predict(thinned).point
    # one source (w=1/3) must sit closer to climatology than two (w=1/2)
    finite = np.isfinite(full_point) & np.isfinite(thin_point)
    assert (
        np.abs(thin_point - anchor)[finite].mean()
        < np.abs(full_point - anchor)[finite].mean()
    )


def test_rows_without_any_source_serve_climatology():
    blender, train = fitted_on_weather()
    x = train.x
    empty_values = x.values.copy()
    empty_values[:5, :] = np.nan
    empty = ForecastMatrix.build(
        x.sources, empty_values, x.lead_hours, x.features, x.product
    )
    anchor = HarmonicClimatology().fit(train).predict(x).point
    point = blender.predict(empty).point
    assert np.allclose(point[:5], anchor[:5], atol=1e-9)


def test_state_is_transparent():
    blender, _train = fitted_on_weather()
    state = blender.to_state()
    assert state == {"method_id": "precip_sparse_shrink", "pseudo_sources": 2.0}
