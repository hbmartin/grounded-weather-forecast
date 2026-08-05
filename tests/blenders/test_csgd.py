"""CSGD-EMOS: censored shifted-gamma precipitation head."""

import numpy as np
import polars as pl

from grounded_weather_forecast.blenders import get_factory
from grounded_weather_forecast.blenders.registry import supports_product
from grounded_weather_forecast.contracts import (
    ForecastMatrix,
    Product,
    SourceKind,
    SupervisedSlice,
    hourly_variable,
)

PRECIP = hourly_variable("precip_mm")
TEMP = hourly_variable("temp_c")


def precip_slice(n=400, wet_fraction=0.3, seed=13):
    """Zero-inflated truth whose wet amounts scale with the provider signal."""
    rng = np.random.default_rng(seed)
    signal = rng.uniform(0.0, 4.0, n)
    wet = rng.uniform(0.0, 1.0, n) < wet_fraction * np.minimum(signal, 1.0)
    y = np.where(wet, rng.gamma(2.0, 0.8, n) * (0.5 + 0.5 * signal), 0.0)
    values = np.column_stack(
        [
            np.maximum(signal + rng.normal(0.0, 0.3, n), 0.0),
            np.maximum(signal + rng.normal(0.0, 0.3, n), 0.0),
        ]
    )
    x = ForecastMatrix.build(
        sources=("a", "b"),
        values=values,
        lead_hours=rng.uniform(0.0, 48.0, n),
        features=pl.DataFrame({"valid_hour_local": [0] * n}),
    )
    return SupervisedSlice(x=x, y=y, variable=PRECIP, source_kind=SourceKind.LIVE)


class TestCsgdEmos:
    def test_wet_archive_fits_and_emits_quantiles(self):
        train = precip_slice()
        blender = get_factory("csgd_emos")().fit(train)
        state = blender.to_state()
        assert state["fit_status"] == "converged"
        assert state["coefficients"]["shift"] < 0.0
        result = blender.predict(train.x)
        assert result.quantiles is not None
        assert (result.quantiles >= 0.0).all()
        for row in range(0, train.x.n_rows, 97):
            grid = result.quantiles[row]
            assert (np.diff(grid) >= -1e-9).all()

    def test_dry_mass_lands_on_zero(self):
        train = precip_slice()
        result = get_factory("csgd_emos")().fit(train).predict(train.x)
        assert result.quantiles is not None
        # somewhere in a zero-inflated season the low quantiles must be dry
        assert (result.quantiles[:, 0] == 0.0).any()

    def test_dry_archive_abstains_from_the_distribution(self):
        train = precip_slice(wet_fraction=0.0)
        blender = get_factory("csgd_emos")().fit(train)
        assert blender.to_state()["fit_status"] == "insufficient_wet_rows"
        result = blender.predict(train.x)
        assert result.quantiles is None

    def test_scoped_to_precip(self):
        assert supports_product("csgd_emos", Product.HOURLY, PRECIP)
        assert not supports_product("csgd_emos", Product.HOURLY, TEMP)
