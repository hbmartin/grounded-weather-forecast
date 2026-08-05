"""Blenders: one module per method family, all implementing the Blender
protocol and self-registering with the registry on import."""

from grounded_weather_forecast.blenders import analog as _analog  # noqa: F401
from grounded_weather_forecast.blenders import anchoring as _anchoring  # noqa: F401
from grounded_weather_forecast.blenders import baselines as _baselines  # noqa: F401
from grounded_weather_forecast.blenders import cluster as _cluster  # noqa: F401
from grounded_weather_forecast.blenders import combine as _combine  # noqa: F401
from grounded_weather_forecast.blenders import conformal as _conformal  # noqa: F401
from grounded_weather_forecast.blenders import csgd as _csgd  # noqa: F401
from grounded_weather_forecast.blenders import (  # noqa: F401
    daily_heads as _daily_heads,
)
from grounded_weather_forecast.blenders import damped as _damped  # noqa: F401
from grounded_weather_forecast.blenders import emos as _emos  # noqa: F401
from grounded_weather_forecast.blenders import (  # noqa: F401
    ewma_grounding as _ewma_grounding,
)
from grounded_weather_forecast.blenders import experts as _experts  # noqa: F401
from grounded_weather_forecast.blenders import gbm as _gbm  # noqa: F401
from grounded_weather_forecast.blenders import (  # noqa: F401
    harmonic_grounding as _harmonic_grounding,
)
from grounded_weather_forecast.blenders import idr as _idr  # noqa: F401
from grounded_weather_forecast.blenders import (  # noqa: F401
    pop_calibration as _pop_calibration,
)
from grounded_weather_forecast.blenders import trimmed as _trimmed  # noqa: F401
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
