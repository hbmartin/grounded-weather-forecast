"""Zone builders, ordered A through G."""

from collections.abc import Callable

from grounded_weather_forecast.dashboard.context import DashboardContext
from grounded_weather_forecast.dashboard.derive import Derived
from grounded_weather_forecast.dashboard.model import Zone
from grounded_weather_forecast.dashboard.zones import (
    data_trust,
    evaluation,
    explain,
    internals,
    liveness,
    readiness,
    serving,
)

type ZoneBuilder = Callable[[DashboardContext, Derived], Zone]

ALL_ZONES: tuple[ZoneBuilder, ...] = (
    liveness.build,
    data_trust.build,
    readiness.build,
    evaluation.build,
    internals.build,
    serving.build,
    explain.build,
)
