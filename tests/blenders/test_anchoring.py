import numpy as np
import polars as pl
import pytest
from conftest import synthetic_hourly_matrix

from grounded_weather_forecast.blenders import get_factory
from grounded_weather_forecast.contracts import ForecastMatrix, hourly_variable
from grounded_weather_forecast.dataset.matrix import to_supervised_slice
from grounded_weather_forecast.metrics.deterministic import mae

TEMP = hourly_variable("temp_c")


def matrix_with_persistent_residual(days=40, residual_sd=3.0, decay_hours=8.0, seed=2):
    """Truth carries a per-snapshot offset that no provider forecasts.

    The offset is visible in the issue-time observation and decays with lead,
    which is exactly the regime anchoring exploits.
    """
    matrix = synthetic_hourly_matrix(days=days, noise_sd=0.3, seed=seed)
    rng = np.random.default_rng(seed)
    issues = matrix["issue_time"].unique(maintain_order=True)
    offsets = dict(
        zip(issues.to_list(), rng.normal(0.0, residual_sd, issues.len()), strict=True)
    )
    offset = pl.Series("offset", [offsets[t] for t in matrix["issue_time"].to_list()])
    decayed = offset.to_numpy() * np.exp(-matrix["lead_hours"].to_numpy() / decay_hours)
    return matrix.with_columns(
        (pl.col("t__temp_c__inst") + pl.Series(decayed)).alias("t__temp_c__inst"),
        (pl.col("t__temp_c__mean") + pl.Series(decayed)).alias("t__temp_c__mean"),
        (pl.col("obs__temp_c") + offset).alias("obs__temp_c"),
    )


@pytest.fixture(scope="class")
def slices():
    matrix = matrix_with_persistent_residual()
    return to_supervised_slice(matrix, TEMP)


class TestAnchoring:
    def test_wins_at_short_leads_converges_later(self, slices):
        train = slices
        base = get_factory("grounded_equal_weight")().fit(train)
        anchored = get_factory("anchored_grounded_equal_weight")().fit(train)
        base_point = base.predict(train.x).point
        anchored_point = anchored.predict(train.x).point

        short = train.x.lead_hours <= 3.0
        long = train.x.lead_hours >= 24.0
        base_short = mae(base_point[short], train.y[short])
        anchored_short = mae(anchored_point[short], train.y[short])
        assert anchored_short < 0.75 * base_short

        base_long = mae(base_point[long], train.y[long])
        anchored_long = mae(anchored_point[long], train.y[long])
        assert anchored_long == pytest.approx(base_long, rel=0.05)

    def test_no_residual_signal_degrades_to_base(self):
        # plain synthetic: obs residual is pure noise, so anchoring must not hurt
        matrix = synthetic_hourly_matrix(days=30, noise_sd=0.3, seed=9)
        train = to_supervised_slice(matrix, TEMP)
        base = get_factory("grounded_equal_weight")().fit(train)
        anchored = get_factory("anchored_grounded_equal_weight")().fit(train)
        base_mae = mae(base.predict(train.x).point, train.y)
        anchored_mae = mae(anchored.predict(train.x).point, train.y)
        assert anchored_mae <= base_mae * 1.02

    def test_missing_obs_column_degrades_to_base(self, slices):
        train = slices
        anchored = get_factory("anchored_grounded_equal_weight")().fit(train)
        stripped_features = train.x.features.drop("obs__temp_c")
        x = ForecastMatrix.build(
            sources=train.x.sources,
            values=train.x.values,
            lead_hours=train.x.lead_hours,
            features=stripped_features,
        )
        base = get_factory("grounded_equal_weight")().fit(train)
        np.testing.assert_allclose(anchored.predict(x).point, base.predict(x).point)


class TestAnchoredEmpirical:
    """LAMP-style fitted per-lead anchor weights."""

    def persistence_matrix(self, sigma=10.0, days=40, seed=21):
        """Every snapshot carries one shared offset: errors persist across
        leads, so the fitted residual weight should approach 1."""
        matrix = synthetic_hourly_matrix(days=days, noise_sd=0.0, seed=seed)
        issues = matrix.select("issue_time").unique().sort("issue_time")
        rng = np.random.default_rng(seed)
        offsets = issues.with_columns(
            pl.Series("offset", rng.normal(0.0, sigma, issues.height))
        )
        return (
            matrix.join(offsets, on="issue_time")
            .with_columns(
                (pl.col("fx__alpha__temp_c") + pl.col("offset")).alias(
                    "fx__alpha__temp_c"
                ),
                (pl.col("fx__beta__temp_c") + pl.col("offset")).alias(
                    "fx__beta__temp_c"
                ),
            )
            .drop("offset")
        )

    def test_persistent_errors_earn_high_weights(self):
        train = to_supervised_slice(self.persistence_matrix(), TEMP)
        anchored = get_factory("anchored_fitted_grounded")().fit(train)
        weights = anchored._residual_weights
        assert weights is not None
        assert float(weights[0]) > 0.8  # 0-1h bin
        assert float(weights[-1]) > 0.5  # even 12-24h persists here
        base = anchored._base.predict(train.x).point
        point = anchored.predict(train.x).point
        short = train.x.lead_hours <= 12.0
        base_mae = mae(base[short], train.y[short])
        anchored_mae = mae(point[short], train.y[short])
        assert anchored_mae < 0.4 * base_mae

    def test_independent_noise_earns_no_weight(self):
        matrix = synthetic_hourly_matrix(days=40, noise_sd=1.0, seed=22)
        train = to_supervised_slice(matrix, TEMP)
        anchored = get_factory("anchored_fitted_grounded")().fit(train)
        weights = anchored._residual_weights
        assert weights is not None
        assert float(weights.max()) < 0.3
        base = anchored._base.predict(train.x).point
        point = anchored.predict(train.x).point
        assert mae(point, train.y) <= 1.03 * mae(base, train.y)

    def test_weights_never_rise_with_lead(self):
        train = to_supervised_slice(self.persistence_matrix(sigma=3.0), TEMP)
        anchored = get_factory("anchored_fitted_grounded")().fit(train)
        weights = anchored._residual_weights
        assert weights is not None
        assert (np.diff(weights) <= 1e-12).all()

    def test_last_fitted_bin_tapers_continuously_to_24_hours(self):
        train = to_supervised_slice(self.persistence_matrix(sigma=3.0), TEMP)
        anchored = get_factory("anchored_fitted_grounded")().fit(train)
        weights = anchored._residual_weights
        assert weights is not None
        sampled = anchored._weights_at(np.array([18.0, 18.01, 21.0, 24.0]), weights)
        assert sampled[0] == pytest.approx(weights[-1])
        assert sampled[1] < sampled[0]
        assert sampled[1] > sampled[2] > sampled[3]
        assert sampled[3] == 0.0

    def test_trend_variant_registered_and_sane(self):
        train = to_supervised_slice(self.persistence_matrix(), TEMP)
        fitted = get_factory("anchored_fitted_grounded")().fit(train)
        trend = get_factory("anchored_trend_grounded")().fit(train)
        fitted_mae = mae(fitted.predict(train.x).point, train.y)
        trend_mae = mae(trend.predict(train.x).point, train.y)
        assert trend_mae <= 1.1 * fitted_mae

    def test_trend_fit_uses_finite_subset(self):
        anchored = get_factory("anchored_trend_grounded")()
        lead = np.full(30, 0.5)
        residuals = np.ones(30)
        trend = np.linspace(-1.0, 1.0, 30)
        errors = residuals + 2.0 * trend
        trend[:5] = np.nan

        anchored._fit_bins(lead, residuals, errors, trend)

        assert anchored._trend_weights is not None
        assert anchored._trend_weights[0] == pytest.approx(2.0)
