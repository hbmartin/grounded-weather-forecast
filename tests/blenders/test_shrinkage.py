"""Empirical-Bayes shrinkage of thin lead buckets toward the global fit.

The measured failure this guards against: affine grounding promoted for a
bucket that barely cleared ``min_rows`` served a constant biased correction
fitted on a sliver of data. With a ``blend`` supplied, ``PerBucketFitter``
interpolates every bucket's state toward the global fit with weight
``w = n / (n + prior_rows)`` instead of granting full local weight at a cliff.
"""

from itertools import pairwise

import numpy as np
import polars as pl
import pytest

from grounded_weather_forecast.blenders.combine import (
    InverseErrorWeights,
    _blend_source_weights,
)
from grounded_weather_forecast.blenders.grounding import AffineGrounding
from grounded_weather_forecast.blenders.harmonic_grounding import HarmonicGrounding
from grounded_weather_forecast.blenders.protocol import PerBucketFitter
from grounded_weather_forecast.contracts import (
    ForecastMatrix,
    SourceKind,
    SupervisedSlice,
    hourly_variable,
)
from grounded_weather_forecast.leads import LeadBucket

TEMP = hourly_variable("temp_c")
BUCKET = LeadBucket("near", 0.0, 1.0)
IN_BUCKET = 0.5
OUTSIDE_EVERY_BUCKET = 500.0


def mean_blend(local, global_state, weight):
    return weight * local + (1.0 - weight) * global_state


def weight_recorder(local, global_state, weight):  # noqa: ARG001
    return weight


def mean_fitter(values, blend, **kwargs):
    def fit_one(rows):
        return float(values[rows].mean())

    return PerBucketFitter[float](
        buckets=(BUCKET,), fit_one=fit_one, blend=blend, **kwargs
    )


def bucket_plus_background(n, local_value, background_rows=600):
    """``n`` rows inside BUCKET plus a zero-valued background outside it."""
    lead = np.concatenate(
        [np.full(n, IN_BUCKET), np.full(background_rows, OUTSIDE_EVERY_BUCKET)]
    )
    values = np.concatenate([np.full(n, local_value), np.zeros(background_rows)])
    return lead, values


class TestPerBucketFitterShrinkage:
    def test_thin_adversarial_bucket_stays_near_the_global_fit(self):
        lead, values = bucket_plus_background(3, local_value=30.0)
        fitted = mean_fitter(values, blend=mean_blend).fit(lead)
        blended, global_fit = fitted.state_for(IN_BUCKET), fitted.global_state
        # w = 3 / (3 + 24 / 4) = 1/3: two thirds of the adversarial offset gone
        assert abs(blended - global_fit) <= abs(30.0 - global_fit) / 3 + 1e-9
        assert abs(blended - global_fit) < abs(blended - 30.0)

    def test_rich_bucket_reproduces_the_local_fit_within_tolerance(self):
        lead, values = bucket_plus_background(600, local_value=10.0)
        fitted = mean_fitter(values, blend=mean_blend).fit(lead)
        blended = fitted.state_for(IN_BUCKET)
        # w = 600 / 606: the residual global pull is 1% of the gap
        assert abs(blended - 10.0) <= 0.01 * abs(10.0 - fitted.global_state) + 1e-9

    def test_transition_is_monotone_in_bucket_size(self):
        weights = []
        for n in (1, 2, 3, 6, 12, 24, 48, 96, 192):
            lead, values = bucket_plus_background(n, local_value=1.0)
            fitted = mean_fitter(values, blend=weight_recorder).fit(lead)
            weights.append(fitted.state_for(IN_BUCKET))
        assert all(0.0 < w < 1.0 for w in weights)
        assert all(a < b for a, b in pairwise(weights))

    def test_local_fit_dominates_at_min_rows(self):
        lead, values = bucket_plus_background(24, local_value=1.0)
        fitted = mean_fitter(values, blend=weight_recorder).fit(lead)
        # the derived prior is min_rows / 4, so w = 24 / (24 + 6)
        assert fitted.state_for(IN_BUCKET) == pytest.approx(0.8)

    def test_prior_rows_is_configurable(self):
        lead, values = bucket_plus_background(12, local_value=1.0)
        fitter = mean_fitter(values, blend=weight_recorder, prior_rows=12.0)
        assert fitter.fit(lead).state_for(IN_BUCKET) == pytest.approx(0.5)

    def test_empty_bucket_still_falls_back_to_the_global_fit(self):
        lead, values = bucket_plus_background(0, local_value=0.0)
        fitted = mean_fitter(values, blend=mean_blend).fit(lead)
        assert not fitted.states
        assert fitted.state_for(IN_BUCKET) == fitted.global_state


class TestNoBlendCliffRegression:
    def test_without_blend_the_cliff_is_byte_identical(self):
        """Users that supply no blend keep the exact pre-shrinkage behavior."""
        rng = np.random.default_rng(0)
        buckets = (LeadBucket("rich", 0.0, 1.0), LeadBucket("thin", 1.0, 2.0))
        lead = np.concatenate([np.full(30, 0.5), np.full(5, 1.5)])
        values = rng.normal(0.0, 5.0, lead.shape[0])

        def fit_one(rows):
            return float(values[rows].mean())

        fitted = PerBucketFitter[float](buckets=buckets, fit_one=fit_one).fit(lead)

        assert set(fitted.states) == {"rich"}
        assert fitted.states["rich"] == fit_one(np.arange(30))
        assert fitted.global_state == fit_one(np.arange(35))
        assert fitted.state_for(1.5) == fitted.global_state


def two_bucket_slice(sources_values, truth, thin_rows):
    """A slice with ``thin_rows`` rows at lead 4 (3-6h), the rest at lead 10."""
    n = truth.shape[0]
    lead = np.concatenate([np.full(thin_rows, 4.0), np.full(n - thin_rows, 10.0)])
    x = ForecastMatrix.build(
        sources=tuple(sources_values),
        values=np.column_stack(list(sources_values.values())),
        lead_hours=lead,
        features=pl.DataFrame({"lead_hours": lead}),
    )
    return SupervisedSlice(x=x, y=truth, variable=TEMP, source_kind=SourceKind.LIVE)


def affine_slice(thin_rows=3, thin_bias=-30.0, rich_rows=500, rich_bias=2.0):
    """One source: a +30 residual in the thin 3-6h bucket, -2 elsewhere."""
    rng = np.random.default_rng(1)
    truth = rng.normal(10.0, 3.0, thin_rows + rich_rows)
    forecast = truth + np.concatenate(
        [np.full(thin_rows, thin_bias), np.full(rich_rows, rich_bias)]
    )
    return two_bucket_slice({"src": forecast}, truth, thin_rows)


class TestAffineGroundingShrinkage:
    def test_thin_bucket_shrinks_toward_the_global_intercept(self):
        state = AffineGrounding().fit(affine_slice()).to_state()["src"]
        global_intercept, _ = state["global"]
        thin_intercept, thin_slope = state["buckets"]["3-6h"]
        local_intercept = 30.0  # mean residual of the 3 adversarial rows
        assert thin_slope == pytest.approx(1.0)  # bias-only on both sides
        # w = 3 / (3 + 6): only a third of the local offset survives
        assert abs(thin_intercept - global_intercept) <= (
            abs(local_intercept - global_intercept) / 3 + 1e-9
        )
        assert abs(thin_intercept - global_intercept) < abs(
            thin_intercept - local_intercept
        )

    def test_rich_bucket_reproduces_the_local_fit_within_tolerance(self):
        state = AffineGrounding().fit(affine_slice()).to_state()["src"]
        rich_intercept, rich_slope = state["buckets"]["6-12h"]
        # the exact local fit is the -2 residual; w = 500 / 506 keeps it
        assert rich_intercept == pytest.approx(-2.0, abs=0.02)
        assert rich_slope == pytest.approx(1.0)


def combine_slice(thin_rows=3, rich_rows=500):
    """Two unbiased sources: ``good`` (sd 0.2) and ``bad`` (sd 2.0)."""
    rng = np.random.default_rng(2)
    n = thin_rows + rich_rows
    truth = rng.normal(10.0, 3.0, n)
    values = {
        "good": truth + rng.normal(0.0, 0.2, n),
        "bad": truth + rng.normal(0.0, 2.0, n),
    }
    return two_bucket_slice(values, truth, thin_rows)


def shares(weights):
    scaled = np.asarray(weights)
    return scaled / scaled.sum()


class TestInverseErrorWeightsShrinkage:
    def test_blend_interpolates_relative_weights(self):
        local, global_state = np.array([4.0, 1.0]), np.array([1.0, 4.0])
        halfway = _blend_source_weights(local, global_state, 0.5)
        assert halfway[0] == pytest.approx(halfway[1])  # mirror inputs cancel
        local_only = _blend_source_weights(local, global_state, 1.0)
        np.testing.assert_allclose(shares(local_only), [0.8, 0.2])

    def test_blend_is_invariant_to_a_weakest_extra_source(self):
        # an all-NaN source receives the fallback (weakest) weight; appending
        # it must not perturb the real sources' blended weights
        base = _blend_source_weights(np.array([4.0, 1.0]), np.array([2.0, 3.0]), 0.4)
        padded = _blend_source_weights(
            np.array([4.0, 1.0, 1.0]), np.array([2.0, 3.0, 2.0]), 0.4
        )
        np.testing.assert_array_equal(padded[:2], base)

    def test_thin_bucket_weights_stay_ordered_like_the_global_fit(self):
        state = InverseErrorWeights().fit(combine_slice()).to_state()["weights"]
        global_share = shares(state["global"])
        thin_share = shares(state["buckets"]["3-6h"])
        assert global_share[0] > 0.7  # the fixture really makes "good" dominant
        # the 3-row bucket's local evidence is vacuous (uniform fallback);
        # shrinkage keeps "good" on top instead of serving equal weights
        assert 0.5 < thin_share[0] < global_share[0]

    def test_rich_bucket_still_favors_the_accurate_source(self):
        state = InverseErrorWeights().fit(combine_slice()).to_state()["weights"]
        assert shares(state["buckets"]["6-12h"])[0] > 0.9  # 100x loss ratio


class TestHarmonicGroundingKeepsTheCliff:
    def test_thin_bucket_serves_exactly_the_global_curve(self):
        state = HarmonicGrounding().fit(affine_slice()).to_state()["src"]
        assert "3-6h" not in state["buckets"]  # below min_rows: pure global
        assert "6-12h" in state["buckets"]
