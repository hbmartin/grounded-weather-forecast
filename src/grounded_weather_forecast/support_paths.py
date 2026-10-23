"""Lazy support/staging paths shared by readers and scheduled wrapper startup."""

import os
from pathlib import Path


def application_support_dir() -> Path:
    if "HOME" in os.environ and not os.environ["HOME"].strip():
        raise ValueError("HOME is empty; specify the published station path explicitly")
    home = Path.home()
    if not home.is_absolute():
        raise ValueError("home directory must be absolute")
    return home / "Library/Application Support/grounded-weather-forecast"


def published_station_path() -> Path:
    return application_support_dir() / "staged/current.json"
