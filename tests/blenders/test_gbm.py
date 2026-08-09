import numpy as np
import polars as pl
import pytest
from conftest import synthetic_hourly_matrix

from grounded_weather_forecast.blenders import available_methods, get_factory
from grounded_weather_forecast.contracts import ForecastMatrix, hourly_variable
from grounded_weather_forecast.dataset.matrix import to_supervised_slice
from grounded_weather_forecast.metrics.deterministic import mae

lightgbm = pytest.importorskip("lightgbm")

from grounded_weather_forecast.blenders.gbm import (  # noqa: E402
    QUANTILE_LEVELS,
    GbmQuantile,
    GbmStacker,
    build_features,
)

TEMP = hourly_variable("temp_c")


def hour_dependent_bias_matrix(days=40, seed=4):
    """Sources carry an hour-of-day-dependent bias no affine bucket can fix."""
    matrix = synthetic_hourly_matrix(days=days, noise_sd=0.3, seed=seed)
    hours = matrix["valid_hour_local"].to_numpy().astype(float)
    wobble = 4.0 * np.sin(2 * np.pi * hours / 24.0 + 1.0)
    return matrix.with_columns(
        (pl.col("fx__alpha__temp_c") + pl.Series(wobble)).alias("fx__alpha__temp_c"),
        (pl.col("fx__beta__temp_c") + pl.Series(wobble)).alias("fx__beta__temp_c"),
    )


class TestGbmStacker:
    def test_registered(self):
        assert "gbm" in available_methods()

    def test_beats_grounding_on_nonlinear_bias(self):
        matrix = hour_dependent_bias_matrix()
        train = to_supervised_slice(matrix, TEMP)
        grounded = get_factory("grounded_equal_weight")().fit(train)
        gbm = get_factory("gbm")().fit(train)
        grounded_mae = mae(grounded.predict(train.x).point, train.y)
        gbm_mae = mae(gbm.predict(train.x).point, train.y)
        assert gbm_mae < 0.75 * grounded_mae

    def test_state_round_trip(self):
        matrix = synthetic_hourly_matrix(days=15, seed=1)
        train = to_supervised_slice(matrix, TEMP)
        fitted = GbmStacker().fit(train)
        restored = GbmStacker.from_state(fitted.to_state())
        np.testing.assert_allclose(
            restored.predict(train.x).point, fitted.predict(train.x).point
        )

    def test_feature_alignment_on_missing_columns(self):
        matrix = synthetic_hourly_matrix(days=15, seed=1)
        train = to_supervised_slice(matrix, TEMP)
        fitted = GbmStacker().fit(train)
        stripped = ForecastMatrix.build(
            sources=train.x.sources,
            values=train.x.values,
            lead_hours=train.x.lead_hours,
            features=train.x.features.drop("obs__temp_c"),
        )
        result = fitted.predict(stripped)
        assert result.point.shape == (train.x.n_rows,)
        assert np.isfinite(result.point).all()

    def test_build_features_names(self):
        matrix = synthetic_hourly_matrix(days=2)
        train = to_supervised_slice(matrix, TEMP)
        features, names = build_features(train.x)
        assert features.shape == (train.x.n_rows, len(names))
        assert names[0] == "src__alpha"
        assert "lead_hours" in names
        assert "source_spread" in names
        assert "n_available" in names
        assert "blend_mean" in names

    def test_blend_mean_is_monotone_constrained(self):
        matrix = hour_dependent_bias_matrix()
        train = to_supervised_slice(matrix, TEMP)
        fitted = GbmStacker().fit(train)
        features, names = build_features(train.x)
        column = names.index("blend_mean")
        sweep = np.repeat(features[[0]], 50, axis=0)
        base = sweep[0, column]
        sweep[:, column] = np.linspace(base - 5.0, base + 5.0, 50)
        predictions = np.asarray(fitted._booster.predict(sweep))
        assert (np.diff(predictions) >= -1e-9).all()


class TestMinFitRowsFloor:
    def test_abstains_below_floor(self):
        matrix = synthetic_hourly_matrix(days=10, seed=2)
        train = to_supervised_slice(matrix, TEMP)
        stacker = GbmStacker(min_fit_rows=10_000).fit(train)
        result = stacker.predict(train.x)
        assert np.isnan(result.point).all()
        state = stacker.observability_state()
        assert state["fit_status"] == "insufficient_rows"
        assert state["training_rows"] == train.x.n_rows
        assert "model" not in stacker.to_state()

    def test_default_floor_fits_normal_folds(self):
        matrix = synthetic_hourly_matrix(days=10, seed=2)
        train = to_supervised_slice(matrix, TEMP)
        state = GbmStacker().fit(train).observability_state()
        assert state["fit_status"] == "fit"
        assert state["training_rows"] == train.x.n_rows


class TestGbmQuantile:
    def test_registered(self):
        assert "gbm_quantile" in available_methods()

    def test_quantiles_shape_monotone_and_median_point(self):
        matrix = synthetic_hourly_matrix(days=15, seed=1)
        train = to_supervised_slice(matrix, TEMP)
        result = get_factory("gbm_quantile")().fit(train).predict(train.x)
        assert result.quantiles is not None
        assert result.quantiles.shape == (train.x.n_rows, len(QUANTILE_LEVELS))
        assert result.quantile_levels == QUANTILE_LEVELS
        assert (np.diff(result.quantiles, axis=1) >= -1e-9).all()
        np.testing.assert_allclose(
            result.point, result.quantiles[:, QUANTILE_LEVELS.index(0.5)]
        )
        # in-sample sanity: the q05-q95 band should cover most of the truth
        inside = (train.y >= result.quantiles[:, 0]) & (
            train.y <= result.quantiles[:, -1]
        )
        assert inside.mean() > 0.6

    def test_state_round_trip(self):
        matrix = synthetic_hourly_matrix(days=15, seed=1)
        train = to_supervised_slice(matrix, TEMP)
        fitted = GbmQuantile().fit(train)
        restored = GbmQuantile.from_state(fitted.to_state())
        original = fitted.predict(train.x)
        rebuilt = restored.predict(train.x)
        np.testing.assert_allclose(rebuilt.point, original.point)
        np.testing.assert_allclose(rebuilt.quantiles, original.quantiles)

    def test_abstains_below_floor(self):
        matrix = synthetic_hourly_matrix(days=10, seed=2)
        train = to_supervised_slice(matrix, TEMP)
        stacker = GbmQuantile(min_fit_rows=10_000).fit(train)
        result = stacker.predict(train.x)
        assert np.isnan(result.point).all()
        assert result.quantiles is None
        assert "models" not in stacker.to_state()
        assert stacker.observability_state()["fit_status"] == "insufficient_rows"


def test_observability_state_is_compact():
    matrix = hour_dependent_bias_matrix()
    train = to_supervised_slice(matrix, TEMP)
    state = get_factory("gbm")().fit(train).observability_state()
    assert "model" not in state
    assert state["num_trees"] > 0
    assert set(state["importance_gain"]) == set(state["feature_names"])
    assert set(state["importance_split"]) == set(state["feature_names"])
    assert "src__alpha" in state["feature_names"]
