from datetime import timedelta

import numpy as np
import polars as pl

from grounded_weather_forecast.blenders import available_methods, get_factory
from grounded_weather_forecast.blenders.protocol import finalize_point
from grounded_weather_forecast.blenders.registry import supports_product
from grounded_weather_forecast.contracts import (
    ForecastMatrix,
    Product,
    SourceKind,
    SupervisedSlice,
    TargetKind,
    hourly_variable,
    obs_col,
)
from grounded_weather_forecast.timeutil import utc

WIND = hourly_variable("wind_speed_ms")
HUMIDITY = hourly_variable("humidity_pct")
TEMP = hourly_variable("temp_c")


class TestFinalizePoint:
    def test_probability_clips_to_unit_interval(self):
        point = finalize_point(np.array([-0.2, 0.5, 1.4]), TargetKind.PROBABILITY)
        assert point.tolist() == [0.0, 0.5, 1.0]

    def test_variable_minimum_clamped(self):
        point = finalize_point(
            np.array([-1.0, 2.0, np.nan]), TargetKind.CONTINUOUS, WIND
        )
        assert point[0] == 0.0
        assert point[1] == 2.0
        assert np.isnan(point[2])

    def test_variable_bounds_clamped_both_sides(self):
        point = finalize_point(
            np.array([-5.0, 50.0, 130.0]), TargetKind.CONTINUOUS, HUMIDITY
        )
        assert point.tolist() == [0.0, 50.0, 100.0]

    def test_no_variable_passes_through(self):
        point = finalize_point(np.array([-1.0]), TargetKind.CONTINUOUS)
        assert point[0] == -1.0

    def test_unbounded_variable_passes_through(self):
        point = finalize_point(np.array([-40.0]), TargetKind.CONTINUOUS, TEMP)
        assert point[0] == -40.0


def make_wind_slice(n=600):
    """Wind truth near zero with negative-running sources, so any method that
    subtracts bias or extrapolates a harmonic can emit physically impossible
    negative speeds — exactly what the finalize_point clamp must stop."""
    rng = np.random.default_rng(0)
    start = utc(2026, 1, 1)
    issue_times, valid_times = [], []
    leads, hours, months, truth = [], [], [], []
    for i in range(n):
        issue = start + timedelta(hours=6 * (i // 48))
        lead = (i % 48) + 1
        valid = issue + timedelta(hours=lead)
        issue_times.append(issue)
        valid_times.append(valid)
        leads.append(float(lead))
        hours.append(valid.hour)
        months.append(valid.month)
        truth.append(max(0.0, 1.5 * float(np.sin(2 * np.pi * (valid.hour - 15) / 24))))
    y = np.asarray(truth)
    values = np.column_stack(
        [
            y - 2.0 + rng.normal(0.0, 0.3, n),
            y - 1.5 + rng.normal(0.0, 0.3, n),
        ]
    )
    features = pl.DataFrame(
        {
            "issue_time": issue_times,
            "valid_hour_local": hours,
            "valid_month": months,
            obs_col("wind_speed_ms"): np.maximum(0.0, y + rng.normal(0.0, 0.1, n)),
        },
        schema_overrides={"issue_time": pl.Datetime("us", "UTC")},
    )
    x = ForecastMatrix.build(
        sources=("a", "b"),
        values=values,
        lead_hours=np.asarray(leads),
        features=features,
    )
    return SupervisedSlice(x=x, y=y, variable=WIND, source_kind=SourceKind.LIVE)


class TestBoundsAcrossMethods:
    def test_every_registered_method_respects_the_wind_minimum(self):
        train = make_wind_slice()
        raw_mean = np.where(train.x.availability, train.x.values, 0.0).mean(axis=1)
        assert (raw_mean < 0).any()  # the fixture genuinely exercises the clamp
        for method_id in available_methods():
            if not supports_product(method_id, Product.HOURLY, WIND):
                continue  # variable-scoped away from wind
            point = get_factory(method_id)().fit(train).predict(train.x).point
            negatives = point[np.isfinite(point) & (point < 0.0)]
            assert negatives.size == 0, f"{method_id} emitted negative wind"
