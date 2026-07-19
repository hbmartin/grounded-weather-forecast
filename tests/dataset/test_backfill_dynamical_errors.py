"""Lightweight dynamical-backfill boundary tests that run without xarray.

The numerical extraction tests live behind the optional ``backfill`` extra,
but argument and exception normalization are part of the default CLI contract
and must remain covered in the default CI environment.
"""

from datetime import date

import pytest
from conftest import write_config

import grounded_weather_forecast.dataset.backfill_dynamical as module
from grounded_weather_forecast.dataset.backfill_dynamical import (
    DynamicalBackfillError,
    backfill_dynamical_long,
)


@pytest.fixture
def config(tmp_path):
    return write_config(
        tmp_path,
        extra_toml=(
            "[backfill.dynamical]\n"
            'models = ["gefs"]\n'
            "publication_lag_hours = 6.0\n"
            "max_lead_hours = 48.0\n"
        ),
    )


def test_catalog_base_errors_are_normalized(config, monkeypatch):
    class CatalogError(Exception):
        pass

    monkeypatch.setattr(module, "_dynamical_error_types", lambda: (CatalogError,))

    def fail_open(_catalog_id):
        raise CatalogError("dataset cannot be opened")

    with pytest.raises(
        DynamicalBackfillError,
        match=r"'gefs'.*CatalogError: dataset cannot be opened",
    ):
        backfill_dynamical_long(
            config,
            date(2026, 6, 1),
            date(2026, 6, 2),
            opener=fail_open,
        )


def test_programming_errors_are_not_normalized(config):
    def fail_open(_catalog_id):
        raise RuntimeError("bug")

    with pytest.raises(RuntimeError, match="bug"):
        backfill_dynamical_long(
            config,
            date(2026, 6, 1),
            date(2026, 6, 2),
            opener=fail_open,
        )


def test_unknown_model_is_rejected_without_optional_dependencies(config):
    with pytest.raises(DynamicalBackfillError, match="unknown dynamical models"):
        backfill_dynamical_long(
            config,
            date(2026, 6, 1),
            date(2026, 6, 2),
            models=("hrrr",),
        )


def test_reversed_window_is_rejected_without_optional_dependencies(config):
    with pytest.raises(DynamicalBackfillError, match="precedes"):
        backfill_dynamical_long(
            config,
            date(2026, 6, 2),
            date(2026, 6, 1),
        )
