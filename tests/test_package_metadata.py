from importlib.metadata import metadata, version

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


def test_backfill_extra_is_published_in_distribution_metadata():
    package = metadata("grounded-weather-forecast")
    assert "backfill" in package.get_all("Provides-Extra", [])
    requirements = package.get_all("Requires-Dist", [])
    assert any(
        requirement.startswith("dynamical-catalog")
        and "extra == 'backfill'" in requirement
        for requirement in requirements
    )
    assert any(
        requirement.startswith("xarray") and "extra == 'backfill'" in requirement
        for requirement in requirements
    )
    assert any(
        requirement.startswith("zarr") and "extra == 'backfill'" in requirement
        for requirement in requirements
    )
