"""Damped blend: alpha must follow where provider skill actually lives.

The fixture gives truth a per-valid-time weather anomaly the providers track
(climatology cannot), then destroys provider signal beyond 24 h — so the
fitted damping must stay near 1 at short leads and collapse toward
climatology in the far bucket.
"""

import numpy as np
import polars as pl
from conftest import synthetic_hourly_matrix

from grounded_weather_forecast.blenders import available_methods, get_factory
from grounded_weather_forecast.contracts import (
    ForecastMatrix,
    SourceKind,
    SupervisedSlice,
    hourly_variable,
)
from grounded_weather_forecast.dataset.matrix import to_supervised_slice
from grounded_weather_forecast.metrics.deterministic import mae

TEMP = hourly_variable("temp_c")
FX_COLUMNS = ("fx__alpha__temp_c", "fx__beta__temp_c")
TRUTH_COLUMNS = ("t__temp_c__inst", "t__temp_c__mean")


def weather_matrix(days=30, seed=3, far_noise_sd=8.0):
    """Anomaly-tracking providers that turn to noise beyond 24 h lead."""
    matrix = synthetic_hourly_matrix(days=days, noise_sd=0.3, seed=seed)
    rng = np.random.default_rng(seed + 1)
    times = matrix["valid_time"].unique().sort()
    anomaly = pl.DataFrame(
        {"valid_time": times, "anomaly": rng.normal(0.0, 3.0, times.len())}
    )
    matrix = (
        matrix.join(anomaly, on="valid_time")
        .with_columns(
            *(
                (pl.col(column) + pl.col("anomaly")).alias(column)
                for column in (*FX_COLUMNS, *TRUTH_COLUMNS)
            )
        )
        .drop("anomaly")
    )
    far = pl.col("lead_hours") > 24.0
    far_noise = rng.normal(0.0, far_noise_sd, matrix.height)
    return matrix.with_columns(
        *(
            pl.when(far)
            .then(pl.col(column) + pl.Series(far_noise))
            .otherwise(pl.col(column))
            .alias(column)
            for column in FX_COLUMNS
        )
    )


class TestDampedBlend:
    def test_registered(self):
        assert "damped_grounded_equal_weight" in available_methods()

    def test_alpha_collapses_where_providers_break(self):
        train = to_supervised_slice(weather_matrix(), TEMP)
        fitted = get_factory("damped_grounded_equal_weight")().fit(train)
        alpha = fitted.to_state()["alpha"]["buckets"]
        assert alpha["12-24h"] > 0.8
        assert alpha["24-48h"] < 0.5
        assert alpha["24-48h"] < alpha["12-24h"]

    def test_beats_base_when_far_leads_are_noise(self):
        train = to_supervised_slice(weather_matrix(seed=5), TEMP)
        damped = get_factory("damped_grounded_equal_weight")().fit(train)
        base = get_factory("grounded_equal_weight")().fit(train)
        far = train.x.lead_hours > 24.0
        damped_mae = mae(damped.predict(train.x).point[far], train.y[far])
        base_mae = mae(base.predict(train.x).point[far], train.y[far])
        assert damped_mae < base_mae

    def test_rows_without_sources_fall_back_to_climatology(self):
        n = 240
        rng = np.random.default_rng(0)
        leads = np.where(np.arange(n) < 120, 2.0, 300.0)
        y = 10.0 + rng.normal(0.0, 1.0, n)
        values = np.where(leads[:, np.newaxis] < 24.0, y[:, np.newaxis] + 0.1, np.nan)
        x = ForecastMatrix.build(
            sources=("s0",),
            values=values,
            lead_hours=leads,
            features=pl.DataFrame({"valid_hour_local": [0] * n}),
        )
        train = SupervisedSlice(x=x, y=y, variable=TEMP, source_kind=SourceKind.LIVE)
        fitted = get_factory("damped_grounded_equal_weight")().fit(train)
        point = fitted.predict(x).point
        assert np.isfinite(point).all()
        far = leads >= 240.0
        assert np.allclose(point[far], point[far][0])
        assert abs(point[far][0] - y.mean()) < 1.0
