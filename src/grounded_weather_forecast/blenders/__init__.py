"""Blenders: one module per method family, all implementing the Blender
protocol and self-registering with the registry on import."""

from grounded_weather_forecast.blenders import anchoring as _anchoring  # noqa: F401
from grounded_weather_forecast.blenders import baselines as _baselines  # noqa: F401
from grounded_weather_forecast.blenders import combine as _combine  # noqa: F401
from grounded_weather_forecast.blenders import experts as _experts  # noqa: F401
from grounded_weather_forecast.blenders import gbm as _gbm  # noqa: F401
from grounded_weather_forecast.blenders.registry import (
    BlenderFactory,
    UnknownMethodError,
    available_methods,
    get_factory,
    register,
)

__all__ = [
    "BlenderFactory",
    "UnknownMethodError",
    "available_methods",
    "get_factory",
    "register",
]
