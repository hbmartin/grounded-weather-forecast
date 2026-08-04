"""Cluster-pruned equal weight: near-duplicate sources get one vote."""

import numpy as np
import polars as pl

from grounded_weather_forecast.blenders import available_methods, get_factory
from grounded_weather_forecast.contracts import (
    ForecastMatrix,
    SourceKind,
    SupervisedSlice,
    hourly_variable,
)

TEMP = hourly_variable("temp_c")


def make_slice(values, y, variable=TEMP):
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    x = ForecastMatrix.build(
        sources=tuple(f"s{i}" for i in range(values.shape[1])),
        values=values,
        lead_hours=np.ones(n),
        features=pl.DataFrame({"valid_hour_local": [0] * n}),
    )
    return SupervisedSlice(
        x=x,
        y=np.asarray(y, dtype=np.float64),
        variable=variable,
        source_kind=SourceKind.LIVE,
    )


def duplicate_pair_training(n=200, seed=0):
    """s0 and s1 share one error signal (s1 worse); s2 is independent."""
    rng = np.random.default_rng(seed)
    y = 10.0 + rng.normal(0.0, 2.0, n)
    shared = rng.normal(0.0, 1.0, n)
    values = np.column_stack(
        [y + shared, y + shared + 0.5, y + rng.normal(0.0, 1.0, n)]
    )
    return make_slice(values, y)


class TestClusterEqualWeight:
    def test_registered(self):
        assert "cluster_equal_weight" in available_methods()

    def test_prunes_near_duplicates_keeps_lowest_mae(self):
        fitted = get_factory("cluster_equal_weight")().fit(duplicate_pair_training())
        state = fitted.to_state()
        assert state["pruned"] == ["s1"]
        assert set(state["kept"]) == {"s0", "s2"}

    def test_prediction_averages_survivors_only(self):
        train = duplicate_pair_training(seed=1)
        fitted = get_factory("cluster_equal_weight")().fit(train)
        point = fitted.predict(train.x).point
        survivors = train.x.values[:, [0, 2]].mean(axis=1)
        np.testing.assert_allclose(point, survivors)

    def test_row_with_only_pruned_source_falls_back(self):
        train = duplicate_pair_training(seed=2)
        fitted = get_factory("cluster_equal_weight")().fit(train)
        x = make_slice([[np.nan, 7.0, np.nan]], [7.0]).x
        assert fitted.predict(x).point[0] == 7.0

    def test_thin_overlap_never_prunes(self):
        train = duplicate_pair_training(seed=3)
        sparse = np.full(train.x.n_rows, np.nan)
        sparse[:10] = train.x.values[:10, 0]
        padded = ForecastMatrix.build(
            sources=(*train.x.sources, "s3"),
            values=np.column_stack([train.x.values, sparse]),
            lead_hours=train.x.lead_hours,
            features=train.x.features,
        )
        padded_slice = SupervisedSlice(
            x=padded,
            y=train.y,
            variable=train.variable,
            source_kind=train.source_kind,
        )
        state = get_factory("cluster_equal_weight")().fit(padded_slice).to_state()
        assert "s3" in state["kept"]
