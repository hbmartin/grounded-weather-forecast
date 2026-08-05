"""raft_grounded, seamless_regression, inverse_covariance."""

import numpy as np
import polars as pl
from conftest import synthetic_hourly_matrix

from grounded_weather_forecast.blenders import get_factory
from grounded_weather_forecast.blenders.invcov import gls_weights
from grounded_weather_forecast.blenders.seamless import _forward_filled
from grounded_weather_forecast.contracts import (
    ForecastMatrix,
    SourceKind,
    SupervisedSlice,
    hourly_variable,
)
from grounded_weather_forecast.dataset.matrix import to_supervised_slice
from grounded_weather_forecast.metrics.deterministic import mae
from grounded_weather_forecast.timeutil import utc

TEMP = hourly_variable("temp_c")


def temp_slice(days=30, seed=61, **kwargs):
    matrix = synthetic_hourly_matrix(days=days, seed=seed, **kwargs)
    return to_supervised_slice(matrix, TEMP)


class TestRaftGrounded:
    def test_fitted_response_beats_the_unanchored_base(self):
        train = temp_slice()
        raft = get_factory("raft_grounded")().fit(train)
        base = get_factory("grounded_equal_weight")().fit(train)
        raft_point = raft.predict(train.x).point
        base_point = base.predict(train.x).point
        short = train.x.lead_hours <= 6.0
        finite = np.isfinite(raft_point) & np.isfinite(base_point) & short
        assert mae(raft_point[finite], train.y[finite]) <= mae(
            base_point[finite], train.y[finite]
        )

    def test_slopes_shrink_with_lead(self):
        state = get_factory("raft_grounded")().fit(temp_slice()).to_state()
        buckets = state["slopes"]["buckets"]
        if "0-1h" in buckets and "24-48h" in buckets:
            assert buckets["0-1h"] >= buckets["24-48h"] - 0.15

    def test_missing_observation_column_degrades_to_base(self):
        train = temp_slice()
        stripped = SupervisedSlice(
            x=ForecastMatrix.build(
                sources=train.x.sources,
                values=train.x.values,
                lead_hours=train.x.lead_hours,
                features=train.x.features.select("issue_time", "valid_time"),
            ),
            y=train.y,
            variable=train.variable,
            source_kind=train.source_kind,
        )
        raft = get_factory("raft_grounded")().fit(stripped)
        result = raft.predict(stripped.x)
        assert result.point.shape == (stripped.x.n_rows,)


class TestSeamlessRegression:
    def test_learns_the_blend(self):
        train = temp_slice(days=40)
        fitted = get_factory("seamless_regression")().fit(train)
        point = fitted.predict(train.x).point
        finite = np.isfinite(point)
        assert finite.any()
        equal_weight = np.nanmean(train.x.values, axis=1)
        assert mae(point[finite], train.y[finite]) <= mae(
            equal_weight[finite], train.y[finite]
        )

    def test_forward_fill_carries_the_last_lead(self):
        values = np.array([[10.0, 20.0], [11.0, np.nan], [12.0, np.nan]])
        x = ForecastMatrix.build(
            sources=("a", "b"),
            values=values,
            lead_hours=np.array([1.0, 2.0, 3.0]),
            features=pl.DataFrame(
                {"issue_time": [utc(2026, 8, 1)] * 3},
                schema={"issue_time": pl.Datetime("us", "UTC")},
            ),
        )
        filled = _forward_filled(x)
        assert filled[:, 1].tolist() == [20.0, 20.0, 20.0]

    def test_thin_slice_abstains(self):
        train = temp_slice(days=1, max_lead=6)
        result = get_factory("seamless_regression")().fit(train).predict(train.x)
        assert np.isnan(result.point).all()


class TestInverseCovariance:
    def test_downweights_the_duplicated_source(self):
        rng = np.random.default_rng(71)
        n = 400
        y = rng.normal(20.0, 3.0, n)
        shared = rng.normal(0.0, 1.0, n)
        # a and b are near-duplicates; c is independent with equal variance
        errors = np.column_stack(
            [
                shared + rng.normal(0.0, 0.2, n),
                shared + rng.normal(0.0, 0.2, n),
                rng.normal(0.0, 1.0, n),
            ]
        )
        weights = gls_weights(errors)
        assert weights[2] > weights[0]
        assert weights[2] > weights[1]
        assert abs(float(weights.sum()) - 1.0) < 1e-9

    def test_caps_keep_weights_in_range(self):
        rng = np.random.default_rng(72)
        errors = np.column_stack([rng.normal(0.0, sd, 300) for sd in (0.1, 1.0, 5.0)])
        weights = gls_weights(errors)
        assert (weights >= 0.0).all()
        assert (weights <= 2.0 / 3.0 + 1e-9).all()

    def test_incomplete_archive_abstains(self):
        values = np.column_stack([np.full(50, 20.0), np.full(50, np.nan)])
        x = ForecastMatrix.build(
            sources=("a", "b"),
            values=values,
            lead_hours=np.ones(50),
            features=pl.DataFrame({"valid_hour_local": [0] * 50}),
        )
        train = SupervisedSlice(
            x=x, y=np.full(50, 20.0), variable=TEMP, source_kind=SourceKind.LIVE
        )
        result = get_factory("inverse_covariance")().fit(train).predict(x)
        assert np.isnan(result.point).all()
