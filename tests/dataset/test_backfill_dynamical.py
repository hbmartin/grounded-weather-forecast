from datetime import date

import numpy as np
import polars as pl
import pytest
from conftest import write_config

xr = pytest.importorskip("xarray")

from grounded_weather_forecast.dataset.backfill_dynamical import (  # noqa: E402
    DynamicalBackfillError,
    backfill_dynamical_long,
)

LATITUDES = np.array([33.75, 34.25, 34.75])
LONGITUDES = np.array([242.25, 242.75, 243.25])  # 0..360 convention, like GEFS
SITE_CELL = (1, 1)  # 34.25 N, 242.75 E == -117.25 E: nearest to the site


def fake_dataset(n_init=2, with_humidity=True, with_members=True):
    init_times = np.array(
        ["2026-06-01T00:00", "2026-06-01T06:00"], dtype="datetime64[ns]"
    )[:n_init]
    lead_times = np.array([0, 6, 12], dtype="timedelta64[h]").astype("timedelta64[ns]")
    member = np.arange(3 if with_members else 1)
    shape = (n_init, lead_times.size, member.size, LATITUDES.size, LONGITUDES.size)
    temp = np.full(shape, 15.0)
    temp[:, :, :, *SITE_CELL] = 20.0  # the site cell is distinguishable
    data = {
        "temperature_2m": temp,
        "wind_u_10m": np.full(shape, 3.0),
        "wind_v_10m": np.full(shape, 4.0),
        "pressure_reduced_to_mean_sea_level": np.full(shape, 101_300.0),
    }
    if with_humidity:
        data["relative_humidity_2m"] = np.full(shape, 50.0)
    dims = ("init_time", "lead_time", "ensemble_member", "latitude", "longitude")
    dataset = xr.Dataset(
        {name: (dims, values) for name, values in data.items()},
        coords={
            "init_time": init_times,
            "lead_time": lead_times,
            "ensemble_member": member,
            "latitude": LATITUDES,
            "longitude": LONGITUDES,
        },
    )
    if not with_members:
        dataset = dataset.isel(ensemble_member=0, drop=True)
    return dataset


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


class TestBackfillDynamicalLong:
    def run(self, config, dataset=None):
        return backfill_dynamical_long(
            config,
            date(2026, 6, 1),
            date(2026, 6, 2),
            opener=lambda _id: dataset if dataset is not None else fake_dataset(),
        )

    def test_publication_lag_shapes_leads(self, config):
        frame = self.run(config)
        # step 0h -> lead -6h (dropped); step 6h -> lead 0h; step 12h -> lead 6h
        assert sorted(frame["lead_hours"].unique().to_list()) == [0.0, 6.0]
        zero_lead = frame.filter(pl.col("lead_hours") == 0.0)
        assert (zero_lead["fetched_at"] == zero_lead["valid_time"]).all()
        first = frame.row(0, named=True)
        assert first["source"] == "dynamical_gefs"
        assert first["source_kind"] == "synthetic"

    def test_nearest_cell_and_units(self, config):
        frame = self.run(config)
        row = frame.row(0, named=True)
        assert row["temp_c"] == pytest.approx(20.0)  # the site cell, not 15.0
        assert row["wind_speed_ms"] == pytest.approx(5.0)  # hypot(3, 4)
        assert row["pressure_sea_hpa"] == pytest.approx(1013.0)
        # Magnus dew point at 20 degC / 50% RH is about 9.3 degC
        assert row["dew_point_c"] == pytest.approx(9.26, abs=0.1)
        assert row["wind_gust_ms"] is None  # deliberately absent
        assert row["precip_mm"] is None

    def test_member_mean_matches_memberless(self, config):
        with_members = self.run(config, fake_dataset(with_members=True))
        without = self.run(config, fake_dataset(with_members=False))
        assert with_members["temp_c"].to_list() == without["temp_c"].to_list()

    def test_missing_humidity_leaves_nulls(self, config):
        frame = self.run(config, fake_dataset(with_humidity=False))
        assert frame["humidity_pct"].null_count() == frame.height
        assert frame["dew_point_c"].null_count() == frame.height

    def test_unknown_model_rejected(self, config):
        with pytest.raises(DynamicalBackfillError, match="unknown dynamical models"):
            backfill_dynamical_long(
                config,
                date(2026, 6, 1),
                date(2026, 6, 2),
                models=("hrrr",),
                opener=lambda _id: fake_dataset(),
            )

    def test_reversed_window_rejected(self, config):
        with pytest.raises(DynamicalBackfillError, match="precedes"):
            backfill_dynamical_long(
                config,
                date(2026, 6, 2),
                date(2026, 6, 1),
                opener=lambda _id: fake_dataset(),
            )

    def test_catalog_errors_are_normalized_with_model_context(self, config):
        def fail_open(_catalog_id):
            raise LookupError("catalog unavailable")

        with pytest.raises(
            DynamicalBackfillError,
            match=r"'gefs'.*LookupError: catalog unavailable",
        ):
            backfill_dynamical_long(
                config,
                date(2026, 6, 1),
                date(2026, 6, 2),
                opener=fail_open,
            )


class TestMergeSyntheticLong:
    def test_re_running_a_backfill_is_idempotent(self, config):
        from grounded_weather_forecast.dataset.matrix import merged_synthetic_long

        frame = backfill_dynamical_long(
            config,
            date(2026, 6, 1),
            date(2026, 6, 2),
            opener=lambda _id: fake_dataset(),
        )
        store = config.dataset.dir / "forecasts_long_synthetic.parquet"
        store.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(store)
        merged = merged_synthetic_long(config, frame)
        assert merged.height == frame.height
