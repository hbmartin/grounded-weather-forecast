import numpy as np
import pytest

from grounded_weather_forecast.blenders import (
    UnknownMethodError,
    available_methods,
    get_factory,
    register,
)
from grounded_weather_forecast.blenders.registry import supports_product
from grounded_weather_forecast.contracts import BlendResult, Product, hourly_variable


class FakeBlender:
    method_id = "fake"

    def fit(self, train):  # noqa: ARG002 - protocol signature
        return self

    def predict(self, x):
        return BlendResult(point=np.zeros(x.n_rows))


class TestRegistry:
    def test_register_and_get(self):
        register("fake_for_test", FakeBlender)
        assert "fake_for_test" in available_methods()
        assert isinstance(get_factory("fake_for_test")(), FakeBlender)

    def test_duplicate_rejected(self):
        register("fake_dup", FakeBlender)
        with pytest.raises(ValueError, match="already registered"):
            register("fake_dup", FakeBlender)

    def test_unknown_method(self):
        with pytest.raises(UnknownMethodError, match="unknown method"):
            get_factory("does_not_exist")

    def test_daily_excludes_hourly_only_methods(self):
        assert not supports_product("persistence", Product.DAILY)
        assert not supports_product("anchored_inverse_mse", Product.DAILY)
        assert supports_product("equal_weight", Product.DAILY)
        assert not supports_product(
            "persistence", Product.HOURLY, hourly_variable("precip_mm")
        )
