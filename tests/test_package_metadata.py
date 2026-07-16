from importlib.metadata import version

import pytest

import grounded_weather_forecast
from grounded_weather_forecast.cli import main


def test_package_version_comes_from_distribution_metadata():
    assert grounded_weather_forecast.__version__ == version("grounded-weather-forecast")


def test_cli_reports_installed_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    assert (
        capsys.readouterr().out.strip()
        == f"grounded-weather-forecast {version('grounded-weather-forecast')}"
    )
