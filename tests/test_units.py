import math

import polars as pl
import pytest

from grounded_weather_forecast.units import (
    convert_expr,
    dew_point_c,
    dew_point_expr,
    f_to_c,
    inch_to_mm,
    inhg_to_hpa,
    mph_to_ms,
    sea_level_pressure_expr,
    sea_level_pressure_hpa,
)


class TestScalarConversions:
    def test_temperature(self):
        assert f_to_c(32.0) == 0.0
        assert f_to_c(212.0) == 100.0
        assert math.isclose(f_to_c(-40.0), -40.0)

    def test_wind(self):
        assert math.isclose(mph_to_ms(1.0), 0.44704)

    def test_pressure(self):
        assert math.isclose(inhg_to_hpa(29.92), 1013.208, abs_tol=0.01)

    def test_rain(self):
        assert inch_to_mm(1.0) == 25.4


class TestConvertExpr:
    def test_known_units(self):
        frame = pl.DataFrame({"v": [32.0, 50.0]})
        cases = {
            "degF": [0.0, 10.0],
            "pct": [32.0, 50.0],
            "mph": [32.0 * 0.44704, 50.0 * 0.44704],
            "inch": [32.0 * 25.4, 50.0 * 25.4],
        }
        for unit, expected in cases.items():
            got = frame.select(convert_expr(pl.col("v"), unit).alias("out"))[
                "out"
            ].to_list()
            assert got == pytest.approx(expected)

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="unknown source unit"):
            convert_expr(pl.col("v"), "furlongs")


class TestDewPoint:
    def test_saturated_air(self):
        assert math.isclose(dew_point_c(20.0, 100.0), 20.0, abs_tol=1e-9)

    def test_half_humidity(self):
        assert math.isclose(dew_point_c(20.0, 50.0), 9.26, abs_tol=0.1)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="humidity out of range"):
            dew_point_c(20.0, 0.0)
        with pytest.raises(ValueError, match="humidity out of range"):
            dew_point_c(20.0, 101.0)

    def test_expr_matches_scalar_and_nulls_bad_rh(self):
        frame = pl.DataFrame({"t": [20.0, 10.0, 15.0], "rh": [100.0, 50.0, 0.0]})
        got = frame.select(dew_point_expr(pl.col("t"), pl.col("rh")).alias("dp"))[
            "dp"
        ].to_list()
        assert got[0] == pytest.approx(dew_point_c(20.0, 100.0), abs=1e-9)
        assert got[1] == pytest.approx(dew_point_c(10.0, 50.0), abs=1e-9)
        assert got[2] is None


class TestSeaLevelPressure:
    def test_sea_level_is_identity(self):
        assert sea_level_pressure_hpa(1000.0, 0.0, 15.0) == 1000.0

    def test_crestline_reduction_is_plausible(self):
        # AbsPress ~25 inHg (846.6 hPa) at ~1,400 m should reduce to ~1000 hPa.
        slp = sea_level_pressure_hpa(846.6, 1400.0, 15.0)
        assert 985.0 < slp < 1015.0

    def test_monotone_in_elevation(self):
        low = sea_level_pressure_hpa(900.0, 500.0, 15.0)
        high = sea_level_pressure_hpa(900.0, 1500.0, 15.0)
        assert high > low

    def test_expr_matches_scalar(self):
        frame = pl.DataFrame({"p": [846.6, 900.0], "t": [15.0, 0.0]})
        got = frame.select(
            sea_level_pressure_expr(pl.col("p"), 1400.0, pl.col("t")).alias("slp")
        )["slp"].to_list()
        assert got[0] == pytest.approx(sea_level_pressure_hpa(846.6, 1400.0, 15.0))
        assert got[1] == pytest.approx(sea_level_pressure_hpa(900.0, 1400.0, 0.0))
