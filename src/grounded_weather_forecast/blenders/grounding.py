"""Grounding: per-source correction toward the station, by variable x lead bucket.

Most providers repackage the same global models, so their *shared* bias cannot
be removed by any weighting scheme — only by correcting each source toward the
station. That correction is the single biggest win available here.

The correction is ``y ~ a + b*x``, but the slope is deliberately shrinkable,
and the default is a **bias-only** correction (``b = 1``). That default is not
timidity, it is a lesson from the data: an unconstrained least-squares slope
comes out well below 1 (regression dilution — a noisy predictor gets shrunk
toward the *training* mean). Inside the training distribution that lowers MSE,
but the moment the test period is seasonally different, "shrink toward the
training mean" injects a mean-dependent tilt and re-introduces exactly the bias
grounding exists to remove. A bias-only correction is equivariant to level
shifts and cannot do that. Set ``slope_shrinkage`` above zero to buy the slope
back once the training window is seasonally representative; the leaderboard
carries both variants so the data, not this docstring, decides.
"""

from dataclasses import dataclass, field

import numpy as np

from grounded_weather_forecast.blenders.protocol import FittedBuckets, PerBucketFitter
from grounded_weather_forecast.contracts import (
    FloatArray,
    ForecastMatrix,
    SupervisedSlice,
)
from grounded_weather_forecast.leads import (
    HOURLY_BUCKETS,
    LeadBucket,
    buckets_for_product,
)

IDENTITY: tuple[float, float] = (0.0, 1.0)
BIAS_ONLY = 0.0
FREE_SLOPE = 1.0
_MIN_FIT_ROWS = 24
_MIN_VARIANCE = 1e-9
_MAX_ABS_SLOPE = 5.0


def fit_affine(
    x: FloatArray, y: FloatArray, slope_shrinkage: float = BIAS_ONLY
) -> tuple[float, float]:
    """Fit ``y ~ a + b*x`` with the slope shrunk toward the identity.

    ``slope_shrinkage`` interpolates between a pure bias correction (0, the
    default) and the unconstrained least-squares slope (1). The intercept is
    always chosen so the correction passes through the training centroid, so
    the training bias is fully removed whatever the slope.
    """
    n = x.shape[0]
    if n < _MIN_FIT_ROWS:
        return IDENTITY
    x_mean, y_mean = float(x.mean()), float(y.mean())
    centered = x - x_mean
    sxx = float(centered @ centered)
    if sxx < _MIN_VARIANCE:
        return IDENTITY
    slope_ols = float(centered @ (y - y_mean)) / sxx
    if not np.isfinite(slope_ols) or abs(slope_ols) > _MAX_ABS_SLOPE:
        return IDENTITY
    slope = 1.0 + slope_shrinkage * (slope_ols - 1.0)
    return y_mean - slope * x_mean, slope


@dataclass
class AffineGrounding:
    """Fitted per-source, per-lead-bucket corrections toward the station."""

    slope_shrinkage: float = BIAS_ONLY
    buckets: tuple[LeadBucket, ...] = HOURLY_BUCKETS
    _by_source: dict[str, FittedBuckets[tuple[float, float]]] = field(
        default_factory=dict
    )
    _sources: tuple[str, ...] = ()

    def fit(self, train: SupervisedSlice) -> "AffineGrounding":
        self._sources = train.x.sources
        self.buckets = buckets_for_product(train.x.product)
        values, y = train.x.values, train.y
        for index, source in enumerate(self._sources):

            def fit_one(rows: np.ndarray, index: int = index) -> tuple[float, float]:
                available = rows[train.x.availability[rows, index]]
                return fit_affine(
                    values[available, index], y[available], self.slope_shrinkage
                )

            fitter = PerBucketFitter[tuple[float, float]](
                buckets=self.buckets, fit_one=fit_one, min_rows=_MIN_FIT_ROWS
            )
            self._by_source[source] = fitter.fit(train.x.lead_hours)
        return self

    def transform(self, x: ForecastMatrix) -> FloatArray:
        """Corrected forecast values; unknown sources pass through unchanged."""
        corrected = x.values.copy()
        for index, source in enumerate(x.sources):
            fitted = self._by_source.get(source)
            if fitted is None:
                continue

            def use(
                state: tuple[float, float], rows: np.ndarray, index: int = index
            ) -> None:
                intercept, slope = state
                corrected[rows, index] = intercept + slope * x.values[rows, index]

            fitted.apply(x.lead_hours, use)
        return corrected

    def to_state(self) -> dict[str, object]:
        """JSON-serializable coefficients for the artifact store."""
        return {
            source: {
                "global": list(fitted.global_state),
                "buckets": {
                    label: list(state) for label, state in fitted.states.items()
                },
            }
            for source, fitted in self._by_source.items()
        }
