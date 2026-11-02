"""Establish shadow-only numerical limits before importing numerical libraries."""

import os

from grounded_weather_forecast.shadow_artifacts import THREAD_CONTROLS

for name in THREAD_CONTROLS:
    os.environ[name] = "2"
