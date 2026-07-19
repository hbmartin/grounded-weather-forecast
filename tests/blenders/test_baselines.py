import numpy as np
import polars as pl
import pytest
from conftest import synthetic_hourly_matrix

from grounded_weather_forecast.blenders import available_methods, get_factory
from grounded_weather_forecast.blenders.protocol import (
    finalize_point,
    masked_average,
    renormalize_weights,
)
from grounded_weather_forecast.contracts import (
    ForecastMatrix,
    SourceKind,
    SupervisedSlice,
    TargetKind,
    VariableSpec,
    hourly_variable,
)
from grounded_weather_forecast.dataset.matrix import to_supervised_slice

TEMP = hourly_variable("temp_c")


def make_slice(values, y, leads=None, features=None, variable=TEMP):
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    x = ForecastMatrix.build(
        sources=tuple(f"s{i}" for i in range(values.shape[1])),
        values=values,
        lead_hours=np.asarray(leads if leads is not None else np.ones(n)),
        features=features
        if features is not None
        else pl.DataFrame({"valid_hour_local": [0] * n}),
    )
    return SupervisedSlice(
        x=x,
        y=np.asarray(y, dtype=np.float64),
        variable=variable,
        source_kind=SourceKind.LIVE,
    )


class TestProtocolHelpers:
    def test_renormalize(self):
        weights = np.array([0.5, 0.5])
        availability = np.array([[True, True], [True, False], [False, False]])
        normalized = renormalize_weights(weights, availability)
        assert normalized[0].tolist() == [0.5, 0.5]
        assert normalized[1].tolist() == [1.0, 0.0]
        assert normalized[2].tolist() == [0.0, 0.0]

    def test_masked_average(self):
        values = np.array([[1.0, 3.0], [5.0, np.nan], [np.nan, np.nan]])
        availability = ~np.isnan(values)
        point = masked_average(values, availability)
        assert point[0] == 2.0
        assert point[1] == 5.0
        assert np.isnan(point[2])

    def test_finalize_clips_probability(self):
        point = np.array([-0.2, 0.5, 1.4])
        clipped = finalize_point(point, TargetKind.PROBABILITY)
        assert clipped.tolist() == [0.0, 0.5, 1.0]
        untouched = finalize_point(point, TargetKind.CONTINUOUS)
        assert untouched.tolist() == [-0.2, 0.5, 1.4]


class TestProtocolCompliance:
    """Every registered blender must obey the contract."""

    @pytest.fixture
    def train_slice(self):
        matrix = synthetic_hourly_matrix(days=10)
        return to_supervised_slice(matrix, TEMP)

    @pytest.mark.parametrize("method_id", available_methods())
    def test_fit_predict_shapes(self, method_id, train_slice):
        blender = get_factory(method_id)()
        assert blender.method_id == method_id
        fitted = blender.fit(train_slice)
        assert fitted is blender
        result = fitted.predict(train_slice.x)
        assert result.point.shape == (train_slice.x.n_rows,)

    @pytest.mark.parametrize("method_id", available_methods())
    def test_invariant_to_all_nan_source(self, method_id, train_slice):
        blender = get_factory(method_id)().fit(train_slice)
        baseline = blender.predict(train_slice.x).point

        x = train_slice.x
        padded = ForecastMatrix.build(
            sources=(*x.sources, "ghost"),
            values=np.column_stack([x.values, np.full(x.n_rows, np.nan)]),
            lead_hours=x.lead_hours,
            features=x.features,
        )
        padded_slice = SupervisedSlice(
            x=padded,
            y=train_slice.y,
            variable=train_slice.variable,
            source_kind=train_slice.source_kind,
        )
        refit = get_factory(method_id)().fit(padded_slice)
        padded_point = refit.predict(padded).point
        np.testing.assert_allclose(padded_point, baseline, equal_nan=True)


class TestEqualWeight:
    def test_mean_with_missing(self):
        s = make_slice([[1.0, 3.0], [5.0, np.nan]], [2.0, 5.0])
        result = get_factory("equal_weight")().fit(s).predict(s.x)
        assert result.point[0] == 2.0
        assert result.point[1] == 5.0

    def test_all_missing_row_is_nan(self):
        s = make_slice([[np.nan, np.nan]], [1.0])
        result = get_factory("equal_weight")().fit(s).predict(s.x)
        assert np.isnan(result.point[0])

    def test_pop_clipped(self):
        pop = hourly_variable("pop")
        s = make_slice([[1.4, 1.2]], [1.0], variable=pop)
        result = get_factory("equal_weight")().fit(s).predict(s.x)
        assert result.point[0] == 1.0


class TestBestProvider:
    def test_picks_lower_mae_source(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=200)
        good = y + rng.normal(0, 0.1, 200)
        bad = y + 5.0
        s = make_slice(np.column_stack([bad, good]), y)
        result = get_factory("best_provider")().fit(s).predict(s.x)
        np.testing.assert_allclose(result.point, good)

    def test_to_state_ranks_sources_by_name(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=200)
        good = y + rng.normal(0, 0.1, 200)
        bad = y + 5.0
        s = make_slice(np.column_stack([bad, good]), y)
        state = get_factory("best_provider")().fit(s).to_state()
        assert state["sources"] == ["s0", "s1"]
        assert state["global"] == ["s1", "s0"]
        for names in state["buckets"].values():
            assert sorted(names) == ["s0", "s1"]

    def test_falls_back_when_best_unavailable(self):
        y = np.zeros(50)
        good = y.copy()
        bad = y + 5.0
        train = make_slice(np.column_stack([bad, good]), y)
        blender = get_factory("best_provider")().fit(train)
        test_values = np.column_stack([bad[:3], [np.nan, 0.0, np.nan]])
        x = ForecastMatrix.build(
            sources=("s0", "s1"),
            values=test_values,
            lead_hours=np.ones(3),
            features=pl.DataFrame({"valid_hour_local": [0, 0, 0]}),
        )
        result = blender.predict(x)
        assert result.point[0] == 5.0  # fell back to bad source
        assert result.point[1] == 0.0  # best available


class TestClimatology:
    def test_learns_diurnal_cycle(self):
        matrix = synthetic_hourly_matrix(days=40, noise_sd=0.0)
        s = to_supervised_slice(matrix, TEMP)
        result = get_factory("climatology")().fit(s).predict(s.x)
        mae = float(np.mean(np.abs(result.point - s.y)))
        assert mae < 2.0  # sinusoid captured well

    def test_short_window_does_not_explode_out_of_season(self):
        # Train on a 60-day arc of the annual cycle, predict a different season.
        # Unpenalized, the near-collinear annual sin/cos blow up on extrapolation.
        train_matrix = synthetic_hourly_matrix(days=60, noise_sd=0.0)
        train = to_supervised_slice(train_matrix, TEMP)
        fitted = get_factory("climatology")().fit(train)

        winter = train_matrix.with_columns(
            pl.lit(12, dtype=pl.Int8).alias("valid_month")
        )
        out_of_season = to_supervised_slice(winter, TEMP)
        point = fitted.predict(out_of_season.x).point
        # the true range of this synthetic climate is roughly -3..23 C
        assert np.isfinite(point).all()
        assert point.min() > -30.0
        assert point.max() < 50.0


class TestPersistence:
    def test_returns_issue_time_obs(self):
        matrix = synthetic_hourly_matrix(days=5)
        s = to_supervised_slice(matrix, TEMP)
        result = get_factory("persistence")().fit(s).predict(s.x)
        obs = s.x.features["obs__temp_c"].to_numpy()
        np.testing.assert_allclose(result.point, obs)

    def test_nan_without_obs_feature(self):
        s = make_slice([[1.0, 2.0]], [1.5])
        variable = VariableSpec("precip_mm", TargetKind.CONTINUOUS, "mm")
        s = SupervisedSlice(
            x=s.x, y=s.y, variable=variable, source_kind=SourceKind.LIVE
        )
        result = get_factory("persistence")().fit(s).predict(s.x)
        assert np.isnan(result.point[0])
