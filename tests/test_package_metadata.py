from importlib.metadata import version

import pytest

import omni_forecast
from omni_forecast.cli import main


def test_package_version_comes_from_distribution_metadata():
    assert omni_forecast.__version__ == version("omni-forecast")


def test_cli_reports_installed_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    assert (
        capsys.readouterr().out.strip() == f"omni-forecast {version('omni-forecast')}"
    )
