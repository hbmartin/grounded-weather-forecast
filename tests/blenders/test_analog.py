"""Analog ensemble: distributions must come from the verifying observations
of similar past forecasts, pooled by lead bucket."""

import numpy as np
import polars as pl

from grounded_weather_forecast.blenders import available_methods, get_factory
from grounded_weather_forecast.contracts import (
    ForecastMatrix,
    SourceKind,
    SupervisedSlice,
    hourly_variable,
)

TEMP = hourly_variable("temp_c")


def make_slice(values, y, leads=None, variable=TEMP):
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    x = ForecastMatrix.build(
        sources=tuple(f"s{i}" for i in range(values.shape[1])),
        values=values,
        lead_hours=np.asarray(
            leads if leads is not None else np.ones(n), dtype=np.float64
        ),
        features=pl.DataFrame({"valid_hour_local": [0] * n}),
    )
    return SupervisedSlice(
        x=x,
        y=np.asarray(y, dtype=np.float64),
        variable=variable,
        source_kind=SourceKind.LIVE,
    )


def two_regime_training(seed=0):
    """Forecast 0 verifies near 0; forecast 10 verifies near 10."""
    rng = np.random.default_rng(seed)
    base = np.concatenate([np.zeros(100), np.full(100, 10.0)])
    values = (base + rng.normal(0.0, 0.01, 200))[:, np.newaxis]
    return make_slice(values, base)


class TestAnalogEnsemble:
    def test_registered(self):
        assert "analog_ensemble" in available_methods()

    def test_matches_conditional_outcomes(self):
        fitted = get_factory("analog_ensemble")().fit(two_regime_training())
        x = make_slice([[0.0], [10.0]], [0.0, 10.0]).x
        result = fitted.predict(x)
        assert abs(result.point[0]) < 0.5
        assert abs(result.point[1] - 10.0) < 0.5
        assert result.quantiles is not None
        assert result.quantiles.shape == (2, 19)
        assert (np.diff(result.quantiles, axis=1) >= -1e-9).all()

    def test_quantiles_stay_inside_observed_support(self):
        train = two_regime_training(seed=1)
        fitted = get_factory("analog_ensemble")().fit(train)
        result = fitted.predict(train.x)
        assert result.quantiles is not None
        finite = np.isfinite(result.quantiles)
        assert result.quantiles[finite].min() >= train.y.min() - 1e-9
        assert result.quantiles[finite].max() <= train.y.max() + 1e-9

    def test_thin_training_abstains(self):
        rng = np.random.default_rng(2)
        train = make_slice(rng.normal(0.0, 1.0, (50, 1)), rng.normal(0.0, 1.0, 50))
        result = get_factory("analog_ensemble")().fit(train).predict(train.x)
        assert np.isnan(result.point).all()
        assert result.quantiles is None

    def test_lead_bucket_pools_separate_regimes(self):
        rng = np.random.default_rng(3)
        values = (5.0 + rng.normal(0.0, 0.01, 200))[:, np.newaxis]
        leads = np.concatenate([np.full(100, 2.0), np.full(100, 30.0)])
        y = np.where(leads < 24.0, 0.0, 8.0)
        fitted = get_factory("analog_ensemble")().fit(make_slice(values, y, leads))
        x = make_slice([[5.0]], [8.0], leads=[30.0]).x
        result = fitted.predict(x)
        assert result.point[0] > 7.0
