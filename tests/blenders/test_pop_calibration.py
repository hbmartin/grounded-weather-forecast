"""PoP recalibration heads: Platt vs beta, scope, and the dry-season guard."""

import numpy as np
import polars as pl
import pytest

from grounded_weather_forecast.blenders import get_factory
from grounded_weather_forecast.blenders.registry import supports_product
from grounded_weather_forecast.contracts import (
    ForecastMatrix,
    Product,
    SourceKind,
    SupervisedSlice,
    hourly_variable,
)
from grounded_weather_forecast.metrics.probabilistic import brier

POP = hourly_variable("pop")
TEMP = hourly_variable("temp_c")


def pop_slice(n=600, distorted=True, dry=False, seed=11):
    """Providers emit a miscalibrated (over-sharp) version of the true PoP."""
    rng = np.random.default_rng(seed)
    truth_probability = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(0.0, 1.0, n) < truth_probability).astype(np.float64)
    if dry:
        y = np.zeros(n)
    logit = np.log(truth_probability / (1.0 - truth_probability))
    provider = (
        1.0 / (1.0 + np.exp(-(2.5 * logit + 0.8))) if distorted else truth_probability
    )
    values = np.column_stack(
        [
            np.clip(provider + rng.normal(0.0, 0.02, n), 0.0, 1.0),
            np.clip(provider + rng.normal(0.0, 0.02, n), 0.0, 1.0),
        ]
    )
    x = ForecastMatrix.build(
        sources=("a", "b"),
        values=values,
        lead_hours=rng.uniform(0.0, 48.0, n),
        features=pl.DataFrame({"valid_hour_local": [0] * n}),
    )
    return SupervisedSlice(x=x, y=y, variable=POP, source_kind=SourceKind.LIVE)


@pytest.mark.parametrize("method_id", ["pop_platt", "pop_beta"])
class TestPopCalibrators:
    def test_calibration_beats_the_raw_mean(self, method_id):
        train = pop_slice()
        result = get_factory(method_id)().fit(train).predict(train.x)
        base = np.nanmean(train.x.values, axis=1)
        assert brier(result.point, train.y) < brier(base, train.y)

    def test_dry_window_serves_the_identity(self, method_id):
        train = pop_slice(dry=True)
        result = get_factory(method_id)().fit(train).predict(train.x)
        base = np.nanmean(train.x.values, axis=1)
        # identity up to the logit-clip epsilon at exact 0/1 probabilities
        np.testing.assert_allclose(result.point, base, atol=2e-4)

    def test_probabilities_stay_in_range(self, method_id):
        train = pop_slice()
        point = get_factory(method_id)().fit(train).predict(train.x).point
        finite = point[np.isfinite(point)]
        assert ((finite >= 0.0) & (finite <= 1.0)).all()

    def test_compliance_on_the_pop_fixture(self, method_id):
        train = pop_slice(n=120)
        blender = get_factory(method_id)()
        assert blender.method_id == method_id
        fitted = blender.fit(train)
        assert fitted is blender
        result = fitted.predict(train.x)
        assert result.point.shape == (train.x.n_rows,)
        state = fitted.to_state()
        assert state["schema_version"] == 1

    def test_scoped_to_pop_only(self, method_id):
        assert supports_product(method_id, Product.HOURLY, POP)
        assert not supports_product(method_id, Product.HOURLY, TEMP)


class TestScopeMechanism:
    def test_unscoped_methods_are_universal(self):
        assert supports_product("equal_weight", Product.HOURLY, TEMP)
        assert supports_product("equal_weight", Product.HOURLY, POP)
