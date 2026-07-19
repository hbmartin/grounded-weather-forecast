import json
from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from conftest import synthetic_hourly_matrix

from grounded_weather_forecast.blenders import get_factory
from grounded_weather_forecast.blenders.experts import OnlineExperts
from grounded_weather_forecast.blenders.grounding import AffineGrounding
from grounded_weather_forecast.contracts import (
    ForecastMatrix,
    SupervisedSlice,
    hourly_variable,
)
from grounded_weather_forecast.dataset.matrix import to_supervised_slice
from grounded_weather_forecast.metrics.deterministic import mae

TEMP = hourly_variable("temp_c")
SCHEMES = ("ewa", "boa")


def grounded_expert_mae(train, expert_index):
    corrected = AffineGrounding().fit(train).transform(train.x)
    column = corrected[:, expert_index]
    usable = ~np.isnan(column)
    return mae(column[usable], train.y[usable])


class TestOnlineExperts:
    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_weights_concentrate_on_better_expert(self, scheme):
        matrix = synthetic_hourly_matrix(
            days=40, noise_sd={"alpha": 4.0, "beta": 0.3}, seed=11
        )
        train = to_supervised_slice(matrix, TEMP)
        blender = get_factory(scheme)().fit(train)
        assert isinstance(blender, OnlineExperts)
        weights = blender.bucket_weights("6-12h")
        assert weights is not None
        assert weights[1] > 0.75  # beta (index 1) is clearly better

    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_regret_close_to_best_expert(self, scheme):
        matrix = synthetic_hourly_matrix(
            days=40, noise_sd={"alpha": 3.0, "beta": 0.5}, seed=12
        )
        train = to_supervised_slice(matrix, TEMP)
        blender = get_factory(scheme)().fit(train)
        blended_mae = mae(blender.predict(train.x).point, train.y)
        best_expert_mae = grounded_expert_mae(train, 1)
        equal_weight_mae = mae(
            get_factory("grounded_equal_weight")().fit(train).predict(train.x).point,
            train.y,
        )
        # Tracks the best expert (the point of the regret bound) rather than
        # paying the equal-weight penalty for carrying a much noisier source.
        assert blended_mae <= best_expert_mae * 1.15
        assert blended_mae < equal_weight_mae * 0.6

    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_adapts_to_drift(self, scheme):
        # alpha is good in the first half, then degrades sharply; beta reverse.
        # Without fixed share, an early leader can never be overtaken.
        matrix = synthetic_hourly_matrix(days=60, noise_sd=0.3, seed=13)
        half = matrix.height // 2
        rng = np.random.default_rng(13)
        alpha = matrix["fx__alpha__temp_c"].to_numpy().copy()
        beta = matrix["fx__beta__temp_c"].to_numpy().copy()
        alpha[half:] += rng.normal(0.0, 6.0, matrix.height - half)
        beta[:half] += rng.normal(0.0, 6.0, half)
        drifted = matrix.with_columns(
            pl.Series("fx__alpha__temp_c", alpha),
            pl.Series("fx__beta__temp_c", beta),
        )
        train = to_supervised_slice(drifted, TEMP)
        blender = get_factory(scheme)().fit(train)
        weights = blender.bucket_weights("6-12h")
        assert weights is not None
        assert weights[1] > 0.6  # ended favoring beta (good in second half)

    def test_sleeping_source_raggedness(self):
        matrix = synthetic_hourly_matrix(
            days=30, noise_sd={"alpha": 2.0, "beta": 0.3}, beta_max_lead=24, seed=14
        )
        train = to_supervised_slice(matrix, TEMP)
        blender = get_factory("boa")().fit(train)
        point = blender.predict(train.x).point
        long_lead = train.x.lead_hours > 24.0
        # beta is asleep beyond 24h; predictions still exist from alpha alone
        assert np.isfinite(point[long_lead]).all()
        short_weights = blender.bucket_weights("6-12h")
        assert short_weights is not None
        assert short_weights[1] > 0.6  # beta trusted where it is awake

    def test_state_serializable(self):
        matrix = synthetic_hourly_matrix(days=10)
        train = to_supervised_slice(matrix, TEMP)
        blender = get_factory("ewa")().fit(train)
        assert isinstance(blender, OnlineExperts)
        encoded = json.dumps(blender.to_state())
        assert "buckets" in encoded


class TestOnlineAdvance:
    """The training matrix is the pending-loss queue; advance() consumes it."""

    def test_incremental_equals_batch_replay_with_shared_grounding(self):
        """With one fixed grounding, split replay lands on the batch weights
        exactly — the matrix-as-queue design has no other state."""
        matrix = synthetic_hourly_matrix(days=30, noise_sd=0.4, seed=61)
        full = to_supervised_slice(matrix, TEMP)
        midpoint = (
            matrix["valid_time"].min()
            + (matrix["valid_time"].max() - matrix["valid_time"].min()) / 2
        )
        first_half = to_supervised_slice(
            matrix.filter(pl.col("valid_time") <= midpoint), TEMP
        )
        grounding = AffineGrounding().fit(full)
        batch = OnlineExperts(method_id="boa", scheme="boa")
        batch._grounding = grounding
        batch._replay(full)
        incremental = OnlineExperts(method_id="boa", scheme="boa")
        incremental._grounding = grounding
        incremental._replay(first_half)
        incremental._replay(full)
        for label, state in batch._states.items():
            np.testing.assert_allclose(
                incremental._states[label].weights, state.weights, atol=1e-12
            )

    @pytest.mark.parametrize("scheme", SCHEMES)
    def test_public_advance_equals_batch_replay(self, scheme):
        """A refitted grounder must replay every historical expert loss."""
        matrix = synthetic_hourly_matrix(days=30, noise_sd=0.4, seed=61)
        full = to_supervised_slice(matrix, TEMP)
        midpoint = (
            matrix["valid_time"].min()
            + (matrix["valid_time"].max() - matrix["valid_time"].min()) / 2
        )
        first_half = to_supervised_slice(
            matrix.filter(pl.col("valid_time") <= midpoint), TEMP
        )
        batch = OnlineExperts(method_id=scheme, scheme=scheme).fit(full)
        incremental = OnlineExperts(method_id=scheme, scheme=scheme).fit(first_half)
        restored = OnlineExperts.from_state(incremental.to_state(), scheme)
        restored.advance(full)
        for label, state in batch._states.items():
            weights = restored._states[label].weights
            np.testing.assert_allclose(weights, state.weights, atol=1e-12)

    def test_advance_is_idempotent(self):
        matrix = synthetic_hourly_matrix(days=20, noise_sd=0.4, seed=62)
        train = to_supervised_slice(matrix, TEMP)
        experts = OnlineExperts(method_id="ewa", scheme="ewa").fit(train)
        weights_before = {
            label: state.weights.copy() for label, state in experts._states.items()
        }
        experts.advance(train)  # same data: nothing past the watermark
        for label, weights in weights_before.items():
            np.testing.assert_array_equal(experts._states[label].weights, weights)

    def test_state_round_trip(self):
        matrix = synthetic_hourly_matrix(days=15, noise_sd=0.4, seed=63)
        train = to_supervised_slice(matrix, TEMP)
        experts = OnlineExperts(method_id="boa", scheme="boa").fit(train)
        state = experts.to_state()
        restored = OnlineExperts.from_state(state, "boa")
        assert restored._progress == experts._progress
        assert restored._sources == experts._sources
        for label, bucket_state in experts._states.items():
            np.testing.assert_array_equal(
                restored._states[label].weights, bucket_state.weights
            )

    def test_later_resolving_lead_for_same_issue_is_consumed(self):
        matrix = synthetic_hourly_matrix(days=4, max_lead=12, seed=64)
        first_resolution = matrix["valid_time"].min()
        short = to_supervised_slice(
            matrix.filter(
                pl.col("valid_time") <= first_resolution + timedelta(hours=3)
            ),
            TEMP,
        )
        full = to_supervised_slice(matrix, TEMP)
        experts = OnlineExperts(method_id="ewa", scheme="ewa").fit(short)

        assert "6-12h" not in experts._states
        experts.advance(full)

        assert experts._states["6-12h"].steps > 0
        assert experts._progress["6-12h"].rows > 0

    def test_historical_correction_invalidates_incremental_state(self):
        matrix = synthetic_hourly_matrix(days=4, max_lead=12, seed=65)
        train = to_supervised_slice(matrix, TEMP)
        experts = OnlineExperts(method_id="ewa", scheme="ewa").fit(train)
        corrected = matrix.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(pl.col("fx__alpha__temp_c") + 5.0)
            .otherwise(pl.col("fx__alpha__temp_c"))
            .alias("fx__alpha__temp_c")
        )

        with pytest.raises(ValueError, match="history changed"):
            experts.advance(to_supervised_slice(corrected, TEMP))

    def test_legacy_state_requires_full_replay(self):
        with pytest.raises(ValueError, match="legacy"):
            OnlineExperts.from_state(
                {"scheme": "ewa", "sources": ["alpha"], "buckets": {}}, "ewa"
            )

    def test_reordered_sources_require_full_replay(self):
        train = to_supervised_slice(synthetic_hourly_matrix(days=4), TEMP)
        experts = OnlineExperts(method_id="ewa", scheme="ewa").fit(train)
        reordered_x = ForecastMatrix.build(
            sources=tuple(reversed(train.x.sources)),
            values=train.x.values[:, ::-1],
            lead_hours=train.x.lead_hours,
            features=train.x.features,
            product=train.x.product,
        )
        reordered = SupervisedSlice(
            x=reordered_x,
            y=train.y,
            variable=train.variable,
            source_kind=train.source_kind,
        )

        with pytest.raises(ValueError, match="source order changed"):
            experts.advance(reordered)

    def test_state_later_than_training_history_requires_full_replay(self):
        matrix = synthetic_hourly_matrix(days=8)
        full = to_supervised_slice(matrix, TEMP)
        cutoff = matrix["valid_time"].min() + timedelta(days=3)
        historical = to_supervised_slice(
            matrix.filter(pl.col("valid_time") <= cutoff), TEMP
        )
        experts = OnlineExperts(method_id="ewa", scheme="ewa").fit(full)

        with pytest.raises(ValueError, match="extends beyond"):
            experts.advance(historical)


def test_observability_state_drops_replay_cursors():
    matrix = synthetic_hourly_matrix(days=20)
    train = to_supervised_slice(matrix, TEMP)
    state = get_factory("ewa")().fit(train).observability_state()
    assert "progress" not in state
    assert "buckets" in state
    assert "grounding" in state
    assert state["sources"] == list(train.x.sources)


class TestStateIntegrity:
    """`advance` raises as a documented signal, so it must not half-mutate."""

    def _fitted(self):
        matrix = synthetic_hourly_matrix(days=6, max_lead=48, seed=11)
        train = to_supervised_slice(matrix, TEMP)
        return matrix, OnlineExperts(method_id="ewa", scheme="ewa").fit(train)

    def test_a_failed_advance_leaves_state_untouched(self):
        """Regression: a late bucket's raise left earlier buckets advanced.

        `_replay` walked buckets in sorted order, mutating `_states`,
        `_progress`, `_sources` and the grounding as it went, so a digest
        mismatch on the last bucket left the object in a state that was
        neither the old one nor a valid new one.
        """
        matrix, experts = self._fitted()
        before = json.dumps(experts.to_state(), sort_keys=True)
        # Corrupt a row in the lexicographically LAST bucket, so every earlier
        # bucket has already been advanced by the time the raise happens.
        last_bucket = sorted(experts.to_state()["buckets"])[-1]
        corrupted = matrix.with_columns(
            pl.when(pl.col("lead_bucket") == last_bucket)
            .then(pl.col("fx__alpha__temp_c") + 5.0)
            .otherwise(pl.col("fx__alpha__temp_c"))
            .alias("fx__alpha__temp_c")
        )

        with pytest.raises(ValueError, match="history changed"):
            experts.advance(to_supervised_slice(corrupted, TEMP))

        assert json.dumps(experts.to_state(), sort_keys=True) == before

    def test_a_vanished_bucket_reports_retention_loss(self):
        """The message for this case used to claim the history had changed."""
        matrix, experts = self._fitted()
        # Drop the short-lead buckets while preserving max(resolution), so the
        # causal guard in `advance` does not pre-empt the bucket check.
        thinned = matrix.filter(pl.col("lead_hours") > 6)

        with pytest.raises(ValueError, match="vanished"):
            experts.advance(to_supervised_slice(thinned, TEMP))

    def test_from_state_rejects_a_mismatched_weight_vector(self):
        """Wrong arity used to surface as an IndexError deep inside `_step`."""
        _, experts = self._fitted()
        state = experts.to_state()
        label = next(iter(state["buckets"]))
        state["buckets"][label]["weights"] = [1.0]

        with pytest.raises(ValueError, match="weights"):
            OnlineExperts.from_state(state, "ewa")

    def test_from_state_rejects_state_without_sources(self):
        _, experts = self._fitted()
        state = experts.to_state()
        state["sources"] = []

        with pytest.raises(ValueError, match="no sources"):
            OnlineExperts.from_state(state, "ewa")

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("weights", [[100.0, -99.0], [100.0, -100.0]], "shapes"),
            ("weights", [2.0, -1.0], "invalid weights"),
            ("regret_variance", [float("nan"), 1.0], "regret variance"),
        ],
    )
    def test_from_state_rejects_invalid_bucket_arrays(self, field, value, message):
        _, experts = self._fitted()
        state = experts.to_state()
        label = next(iter(state["buckets"]))
        state["buckets"][label][field] = value

        with pytest.raises(ValueError, match=message):
            OnlineExperts.from_state(state, "ewa")

    @pytest.mark.parametrize("share", [-0.1, 1.1, float("nan"), True])
    def test_from_state_rejects_invalid_share(self, share):
        _, experts = self._fitted()
        state = experts.to_state()
        state["share"] = share

        with pytest.raises(ValueError, match="fixed-share"):
            OnlineExperts.from_state(state, "ewa")

    def test_from_state_rejects_scheme_mismatch(self):
        _, experts = self._fitted()

        with pytest.raises(ValueError, match="does not match"):
            OnlineExperts.from_state(experts.to_state(), "boa")


def test_non_finite_truth_is_rejected_and_filtered():
    from grounded_weather_forecast.contracts import (
        ContractViolationError,
        SupervisedSlice,
        TruthSemantics,
        truth_col,
    )

    matrix = synthetic_hourly_matrix(days=4)
    train = to_supervised_slice(matrix, TEMP)
    with pytest.raises(ContractViolationError, match="NaN or infinite"):
        SupervisedSlice(
            x=train.x,
            y=np.where(np.arange(train.y.size) == 0, np.inf, train.y),
            variable=train.variable,
            source_kind=train.source_kind,
        )

    truth = truth_col("temp_c", TruthSemantics.INSTANTANEOUS)
    poisoned = matrix.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(float("inf"))
        .otherwise(pl.col(truth))
        .alias(truth)
    )
    filtered = to_supervised_slice(poisoned, TEMP)
    assert filtered.x.n_rows == train.x.n_rows - 1
    assert np.isfinite(filtered.y).all()

    def test_observability_state_still_drops_only_the_cursors(self):
        """The dashboard view must survive the atomicity rework."""
        _, experts = self._fitted()
        state = experts.observability_state()
        assert "progress" not in state
        assert "buckets" in state
