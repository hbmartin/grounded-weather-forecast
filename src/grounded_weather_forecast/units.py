"""Unit conversions from station-native imperial units to canonical metric,
plus the derived meteorological quantities (dew point, sea-level pressure)."""

import math

import polars as pl

type SourceUnit = str

_MPH_TO_MS = 0.44704
_INHG_TO_HPA = 33.8639
_INCH_TO_MM = 25.4


def f_to_c(temp_f: float) -> float:
    return (temp_f - 32.0) * 5.0 / 9.0


def mph_to_ms(speed_mph: float) -> float:
    return speed_mph * _MPH_TO_MS


def inhg_to_hpa(pressure_inhg: float) -> float:
    return pressure_inhg * _INHG_TO_HPA


def inch_to_mm(depth_inch: float) -> float:
    return depth_inch * _INCH_TO_MM


def convert_expr(value: pl.Expr, unit: SourceUnit) -> pl.Expr:
    """Convert a station column expression from its source unit to metric."""
    match unit:
        case "degF":
            return (value - 32.0) * (5.0 / 9.0)
        case "degC" | "pct" | "deg" | "wm2" | "index" | "hpa" | "ms" | "mm":
            return value
        case "mph":
            return value * _MPH_TO_MS
        case "inHg":
            return value * _INHG_TO_HPA
        case "inch":
            return value * _INCH_TO_MM
        case _:
            msg = f"unknown source unit: {unit!r}"
            raise ValueError(msg)


# Magnus formula constants (Alduchov & Eskridge 1996).
_MAGNUS_A = 17.625
_MAGNUS_B = 243.04


def dew_point_c(temp_c: float, humidity_pct: float) -> float:
    """Dew point via the Magnus approximation; valid for ordinary conditions."""
    if not 0.0 < humidity_pct <= 100.0:
        msg = f"humidity out of range (0, 100]: {humidity_pct}"
        raise ValueError(msg)
    gamma = math.log(humidity_pct / 100.0) + _MAGNUS_A * temp_c / (_MAGNUS_B + temp_c)
    return _MAGNUS_B * gamma / (_MAGNUS_A - gamma)


def dew_point_expr(temp_c: pl.Expr, humidity_pct: pl.Expr) -> pl.Expr:
    """Vectorized :func:`dew_point_c`; null where humidity is out of range."""
    rh = (
        pl.when((humidity_pct > 0.0) & (humidity_pct <= 100.0))
        .then(humidity_pct)
        .otherwise(None)
    )
    gamma = (rh / 100.0).log() + _MAGNUS_A * temp_c / (_MAGNUS_B + temp_c)
    return _MAGNUS_B * gamma / (_MAGNUS_A - gamma)


_LAPSE_RATE_K_PER_M = 0.0065
_BAROMETRIC_EXPONENT = 5.257
_KELVIN_OFFSET = 273.15


def sea_level_pressure_hpa(
    station_pressure_hpa: float, elevation_m: float, temp_c: float
) -> float:
    """Reduce absolute station pressure to sea level (international formula).

    The station's ``RelPress`` cannot be trusted for this: at Crestline it is
    nearly identical to ``AbsPress`` (~25 inHg at ~1,400 m), i.e. not reduced.
    """
    lapse_term = _LAPSE_RATE_K_PER_M * elevation_m
    denominator = temp_c + lapse_term + _KELVIN_OFFSET
    return station_pressure_hpa * (1.0 - lapse_term / denominator) ** (
        -_BAROMETRIC_EXPONENT
    )


def sea_level_pressure_expr(
    station_pressure_hpa: pl.Expr, elevation_m: float, temp_c: pl.Expr
) -> pl.Expr:
    """Vectorized :func:`sea_level_pressure_hpa`."""
    lapse_term = _LAPSE_RATE_K_PER_M * elevation_m
    denominator = temp_c + (lapse_term + _KELVIN_OFFSET)
    return station_pressure_hpa * (1.0 - lapse_term / denominator).pow(
        -_BAROMETRIC_EXPONENT
    )
