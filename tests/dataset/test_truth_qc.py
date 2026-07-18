from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from conftest import write_config

from grounded_weather_forecast.dataset.neighbors import (
    cross_check,
    fetch_neighbor_checks,
    neighbor_consensus,
    parse_neighbors,
)
from grounded_weather_forecast.dataset.truth_qc import fit_shield_error, solar_load
from grounded_weather_forecast.timeutil import utc

SITE_ELEVATION = 1400.0
START = utc(2026, 6, 1)


def payload(n_hours=24 * 10, offsets=(0.0, 0.5, -0.5, 0.2)):
    times = [
        (START + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for h in range(n_hours)
    ]
    temps = [20.0 + 5.0 * np.sin(2 * np.pi * h / 24) for h in range(n_hours)]
    stations = []
    for index, offset in enumerate(offsets):
        stations.append(
            {
                "STID": f"N{index}",
                # 100 m lower per index step; N3 sits outside the band
                "ELEVATION": (SITE_ELEVATION - 100.0 * index) * 3.28084
                if index < 3
                else (SITE_ELEVATION - 900.0) * 3.28084,
                "OBSERVATIONS": {
                    "date_time": times,
                    "air_temp_set_1": [t + offset for t in temps],
                },
            }
        )
    return {"STATION": stations}


def station_truth(n_hours=24 * 10, bias=0.0, noise_seed=1):
    rng = np.random.default_rng(noise_seed)
    hours = [START + timedelta(hours=h) for h in range(n_hours)]
    temps = [
        20.0 + 5.0 * np.sin(2 * np.pi * h / 24) + bias + rng.normal(0, 0.1)
        for h in range(n_hours)
    ]
    return pl.DataFrame(
        {"valid_hour": hours, "t__temp_c__inst": temps},
        schema_overrides={"valid_hour": pl.Datetime("us", "UTC")},
    )


class TestNeighbors:
    def test_elevation_band_and_lapse_adjustment(self):
        neighbors = parse_neighbors(payload(), SITE_ELEVATION, 300.0, 6.5)
        assert sorted(neighbors["stid"].unique().to_list()) == ["N0", "N1", "N2"]
        # N1 sits 100 m lower: its reading is adjusted DOWN by 0.65 degC
        n0 = neighbors.filter(pl.col("stid") == "N0")["temp_c"][0]
        n1 = neighbors.filter(pl.col("stid") == "N1")["temp_c"][0]
        assert n1 - n0 == pytest.approx(0.5 - 0.65, abs=1e-6)

    def test_consensus_needs_three(self):
        thin = parse_neighbors(payload(offsets=(0.0, 0.5)), SITE_ELEVATION, 300.0, 6.5)
        assert neighbor_consensus(thin).is_empty()

    def test_healthy_station_raises_no_alerts(self):
        neighbors = parse_neighbors(payload(), SITE_ELEVATION, 300.0, 6.5)
        checks = cross_check(station_truth(), neighbor_consensus(neighbors))
        assert not checks.drift_alert
        assert not checks.correlation_alert

    def test_biased_station_trips_the_drift_alert(self):
        neighbors = parse_neighbors(payload(), SITE_ELEVATION, 300.0, 6.5)
        checks = cross_check(station_truth(bias=2.0), neighbor_consensus(neighbors))
        assert checks.drift_alert

    def test_decorrelated_station_trips_the_correlation_alert(self):
        rng = np.random.default_rng(2)
        broken = station_truth().with_columns(
            pl.Series("t__temp_c__inst", rng.normal(20.0, 5.0, 240))
        )
        neighbors = parse_neighbors(payload(), SITE_ELEVATION, 300.0, 6.5)
        checks = cross_check(broken, neighbor_consensus(neighbors))
        assert checks.correlation_alert

    def test_no_overlap_is_unknown_not_healthy(self):
        consensus = pl.DataFrame(
            {
                "valid_hour": [START - timedelta(days=1)],
                "consensus_c": [20.0],
            },
            schema_overrides={"valid_hour": pl.Datetime("us", "UTC")},
        )

        checks = cross_check(station_truth(), consensus)

        assert checks.drift_alert is None
        assert checks.correlation_alert is None
        assert checks.overlap_hours == 0
        assert "no overlapping" in checks.drift_reason

    def test_comparison_exposes_independent_residual_and_wind(self):
        neighbors = parse_neighbors(payload(), SITE_ELEVATION, 300.0, 6.5)
        truth = station_truth().with_columns(
            pl.lit(2.0).alias("t__wind_speed_ms__inst")
        )

        checks = cross_check(truth, neighbor_consensus(neighbors))

        assert {
            "difference",
            "consensus_c",
            "t__wind_speed_ms__inst",
        } <= set(checks.comparison.columns)
        row = checks.comparison.row(0, named=True)
        assert row["difference"] == pytest.approx(
            row["t__temp_c__inst"] - row["consensus_c"]
        )

    def test_fetch_defaults_to_thirty_days(self, tmp_path):
        config = write_config(
            tmp_path,
            extra_toml='[truth_qc]\nsynoptic_token = "test-token"\n',
        )
        urls: list[str] = []

        def fetcher(url):
            urls.append(url)
            return {"STATION": []}

        fetch_neighbor_checks(config, station_truth(), fetcher=fetcher)

        assert "recent=43200" in urls[0]


class TestShieldFit:
    def test_recovers_a_planted_slope(self):
        rng = np.random.default_rng(3)
        n = 2000
        toa = rng.uniform(0.0, 1000.0, n)
        wind = rng.uniform(0.0, 8.0, n)
        load = solar_load(toa, wind)
        residual = 2.5 * load + rng.normal(0.0, 0.3, n)
        fit = fit_shield_error(residual, toa, wind)
        assert fit is not None
        assert fit.slope_c_per_unit == pytest.approx(2.5, abs=0.2)
        assert fit.significant

    def test_no_solar_dependence_is_insignificant(self):
        rng = np.random.default_rng(4)
        n = 2000
        toa = rng.uniform(100.0, 1000.0, n)
        wind = rng.uniform(0.0, 8.0, n)
        residual = rng.normal(0.0, 0.3, n)
        fit = fit_shield_error(residual, toa, wind)
        assert fit is not None
        assert not fit.significant

    def test_thin_daytime_sample_returns_none(self):
        toa = np.full(50, 800.0)
        assert fit_shield_error(np.zeros(50), toa, np.ones(50)) is None


class TestConfigSection:
    def test_defaults_and_token_env(self, tmp_path, monkeypatch):
        config = write_config(
            tmp_path,
            extra_toml='[truth_qc]\nsynoptic_token = "$SYNTOKEN"\nradius_km = 10.0\n',
        )
        assert config.truth_qc.radius_km == 10.0
        from grounded_weather_forecast.dataset.neighbors import resolve_token

        monkeypatch.setenv("SYNTOKEN", "abc123")
        assert resolve_token(config.truth_qc.synoptic_token) == "abc123"
