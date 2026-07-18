"""Physical truth-QC: the radiation-shield error model.

Daytime error of a passively ventilated radiation shield is well modeled as
increasing with solar load and decreasing with ventilation — the classic
``S / (1 + u)`` regressor (Nakamura-Mahrt form): 1-3 °C of spurious warmth
at high sun and calm wind is typical for consumer shields. The station has a
co-located anemometer, and top-of-atmosphere irradiance from the solar module
is the load proxy, so the signature is directly fittable:

- a significant positive slope means sunny-calm readings run warm *relative
  to the residual baseline* — a shield conversation;
- a slope that grows across refits is the failing-shield trajectory;
- the fitted curve doubles as a correction *candidate* and an
  observation-error inflation signal for anchoring. Neither is auto-applied:
  truth is never silently adjusted (the project's own rule).

Gauge catch-efficiency adjustment (WMO-SPICE transfer functions) is
deliberately deferred: the published coefficients must be transcribed from
Kochendorfer et al. (2018, HESS 22) with care, and the sample archive has
essentially no precipitation to validate against yet.
"""

from dataclasses import dataclass

import numpy as np

from grounded_weather_forecast.contracts import FloatArray

_MIN_DAYTIME_ROWS = 100
_DAYTIME_TOA_WM2 = 50.0


@dataclass(frozen=True, slots=True)
class ShieldFit:
    """``residual ~ intercept + slope * S/(1+u)`` on daytime rows."""

    slope_c_per_unit: float
    intercept_c: float
    slope_se: float
    n_daytime: int

    @property
    def significant(self) -> bool:
        return self.n_daytime >= _MIN_DAYTIME_ROWS and (
            self.slope_se > 0.0 and self.slope_c_per_unit / self.slope_se > 2.0
        )

    def predicted_error_c(self, load: FloatArray) -> FloatArray:
        return self.intercept_c + self.slope_c_per_unit * load


def solar_load(toa_wm2: FloatArray, wind_ms: FloatArray) -> FloatArray:
    """The shield-error regressor, scaled so a slope of 1 means 1 °C at
    1000 W/m² and calm air."""
    return (toa_wm2 / 1000.0) / (1.0 + np.maximum(wind_ms, 0.0))


def fit_shield_error(
    residual_c: FloatArray, toa_wm2: FloatArray, wind_ms: FloatArray
) -> ShieldFit | None:
    """Fit the daytime shield-error curve; None below the sample floor.

    ``residual_c`` is station-minus-reference (neighbor consensus where
    available, the blend's now-forecast otherwise).
    """
    usable = (
        np.isfinite(residual_c)
        & np.isfinite(toa_wm2)
        & np.isfinite(wind_ms)
        & (toa_wm2 > _DAYTIME_TOA_WM2)
    )
    n = int(usable.sum())
    if n < _MIN_DAYTIME_ROWS:
        return None
    load = solar_load(toa_wm2[usable], wind_ms[usable])
    residual = residual_c[usable]
    design = np.column_stack([np.ones(n), load])
    coefficients, _, rank, _ = np.linalg.lstsq(design, residual, rcond=None)
    if rank < design.shape[1]:
        return None
    fitted_residual = residual - design @ coefficients
    dof = max(n - 2, 1)
    sigma2 = float(fitted_residual @ fitted_residual) / dof
    gram_inverse = np.linalg.pinv(design.T @ design)
    slope_variance = max(float(sigma2 * gram_inverse[1, 1]), 0.0)
    slope_se = float(np.sqrt(slope_variance))
    return ShieldFit(
        slope_c_per_unit=float(coefficients[1]),
        intercept_c=float(coefficients[0]),
        slope_se=slope_se,
        n_daytime=n,
    )
