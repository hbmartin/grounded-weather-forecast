from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from conftest import synthetic_hourly_matrix

from grounded_weather_forecast.blenders import get_factory
from grounded_weather_forecast.blenders.ewma_grounding import EwmaBiasGrounding
from grounded_weather_forecast.blenders.harmonic_grounding import HarmonicGrounding
from grounded_weather_forecast.contracts import ForecastMatrix, hourly_variable
from grounded_weather_forecast.dataset.matrix import to_supervised_slice
from grounded_weather_forecast.metrics.deterministic import mae

TEMP = hourly_variable("temp_c")


def diurnal_bias_matrix(days=40, amplitude=4.0, seed=11):
    """Both sources carry a zero-mean diurnal bias the scalar fit cannot see."""
    matrix = synthetic_hourly_matrix(days=days, noise_sd=0.3, seed=seed)
    hour_angle = 2.0 * np.pi * (pl.col("valid_hour_local") - 6) / 24.0
    bias = amplitude * hour_angle.sin()
    return matrix.with_columns(
        (pl.col("fx__alpha__temp_c") + bias).alias("fx__alpha__temp_c"),
        (pl.col("fx__beta__temp_c") + bias).alias("fx__beta__temp_c"),
        hour_angle.sin().alias("hour_sin"),
        hour_angle.cos().alias("hour_cos"),
    )


def _residual_by_hour(corrected, train):
    frame = pl.DataFrame(
        {
            "hour": train.x.features["valid_hour_local"],
            "residual": np.nanmean(corrected, axis=1) - train.y,
        }
    )
    return frame.group_by("hour").agg(pl.col("residual").mean())


class TestEwmaDiurnalBias:
    def test_recovers_what_the_static_fit_cannot(self):
        train = to_supervised_slice(diurnal_bias_matrix(), TEMP)
        ewma = get_factory("ewma_grounded_equal_weight")().fit(train)
        static = get_factory("grounded_equal_weight")().fit(train)
        ewma_mae = mae(ewma.predict(train.x).point, train.y)
        static_mae = mae(static.predict(train.x).point, train.y)
        # the diurnal bias has zero mean, so the scalar intercept is helpless
        assert ewma_mae < 0.5 * static_mae

    def test_hourly_residual_bias_is_flattened(self):
        train = to_supervised_slice(diurnal_bias_matrix(), TEMP)
        grounding = EwmaBiasGrounding().fit(train)
        by_hour = _residual_by_hour(grounding.transform(train.x), train)
        worst = float(by_hour["residual"].abs().max())
        # 3-hour bins quantize a continuous sinusoid: within-bin variation of a
        # 4.0-amplitude sine reaches ~1.0 at the steep edges. The bound asserts
        # the structure is removed to the bin resolution (raw peak: 4.0); the
        # smooth harmonic variant gets below 1.0 in its own test.
        assert worst < 1.5

    def test_reconverges_after_a_backend_swap(self):
        """A provider bias step mid-archive is relearned in ~1/w updates."""
        matrix = synthetic_hourly_matrix(days=60, noise_sd=0.2, seed=13)
        midpoint = (
            matrix["issue_time"].max()
            - (matrix["issue_time"].max() - matrix["issue_time"].min()) / 2
        )
        swapped = matrix.with_columns(
            pl.when(pl.col("issue_time") > midpoint)
            .then(pl.col("fx__alpha__temp_c") + 4.0)
            .otherwise(pl.col("fx__alpha__temp_c"))
            .alias("fx__alpha__temp_c")
        )
        train = to_supervised_slice(swapped, TEMP)
        adaptive = EwmaBiasGrounding().fit(train).transform(train.x)
        recent = (
            train.x.features["issue_time"]
            > matrix["issue_time"].max() - timedelta(days=5)
        ).to_numpy()
        alpha_recent_bias = float(np.nanmean(adaptive[recent, 0] - train.y[recent]))
        assert abs(alpha_recent_bias) < 1.0  # static all-history fit sits near +2

    def test_constant_bias_matches_static_grounding(self):
        matrix = synthetic_hourly_matrix(days=40, biases={"alpha": 3.0}, noise_sd=0.2)
        train = to_supervised_slice(matrix, TEMP)
        adaptive = EwmaBiasGrounding().fit(train).transform(train.x)
        late = np.arange(train.x.n_rows) > train.x.n_rows // 2  # after warm-up
        assert float(np.nanmean(adaptive[late, 0] - train.y[late])) == pytest.approx(
            0.0, abs=0.4
        )

    def test_state_is_json_shaped(self):
        train = to_supervised_slice(synthetic_hourly_matrix(days=10), TEMP)
        state = EwmaBiasGrounding().fit(train).to_state()
        assert set(state["sources"]) == {"alpha", "beta"}
        assert state["learning_rate"] == pytest.approx(0.05)


class TestHarmonicGrounding:
    def test_recovers_diurnal_curve_via_hour_harmonics(self):
        train = to_supervised_slice(diurnal_bias_matrix(), TEMP)
        harmonic = HarmonicGrounding().fit(train)
        by_hour = _residual_by_hour(harmonic.transform(train.x), train)
        assert float(by_hour["residual"].abs().max()) < 1.0

    def test_registered_variant_beats_static(self):
        train = to_supervised_slice(diurnal_bias_matrix(), TEMP)
        harmonic = get_factory("harmonic_grounded_equal_weight")().fit(train)
        static = get_factory("grounded_equal_weight")().fit(train)
        assert mae(harmonic.predict(train.x).point, train.y) < 0.5 * mae(
            static.predict(train.x).point, train.y
        )

    def test_no_phase_features_degrades_to_scalar_bias(self):
        matrix = synthetic_hourly_matrix(days=30, biases={"alpha": 2.0}, noise_sd=0.2)
        train = to_supervised_slice(matrix, TEMP)
        harmonic = HarmonicGrounding().fit(train)
        corrected = harmonic.transform(train.x)
        assert float(np.nanmean(corrected[:, 0] - train.y)) == pytest.approx(
            0.0, abs=0.2
        )

    def test_same_width_different_phase_schema_is_not_applied(self):
        train = to_supervised_slice(diurnal_bias_matrix(), TEMP)
        harmonic = HarmonicGrounding().fit(train)
        alternate_features = train.x.features.drop("hour_sin", "hour_cos").with_columns(
            pl.lit(30.0).alias("solar_elevation_deg")
        )
        alternate = ForecastMatrix.build(
            sources=train.x.sources,
            values=train.x.values,
            lead_hours=train.x.lead_hours,
            features=alternate_features,
            product=train.x.product,
        )

        np.testing.assert_array_equal(harmonic.transform(alternate), alternate.values)
