"""The leakage gauntlet: no future data may reach a blender, ever.

1. Poisoning sentinel — for each fold, corrupt every truth value that was not
   yet knowable at that fold's origin and assert the fold's test predictions
   are bit-identical to the clean run.
2. Fresh instances — the engine must construct a new blender per fold.
3. Provenance wall — mixed live/synthetic matrices are rejected.
"""

import numpy as np
import polars as pl
import pytest
from conftest import synthetic_hourly_matrix, write_config

from grounded_weather_forecast.backtest.engine import BacktestRequest, run_backtest
from grounded_weather_forecast.backtest.splits import fold_plans, hourly_truth_known_at
from grounded_weather_forecast.blenders import get_factory
from grounded_weather_forecast.contracts import MixedProvenanceError, hourly_variable

REQUEST = BacktestRequest(
    variables=(hourly_variable("temp_c"),),
    methods=("climatology", "best_provider", "equal_weight"),
)


@pytest.fixture
def config(tmp_path):
    return write_config(
        tmp_path,
        extra_toml="\n[backtest]\ninitial_train_days = 8\nstep_days = 6\n",
    )


class TestPoisoningSentinel:
    def test_post_origin_truth_cannot_change_predictions(self, config):
        matrix = synthetic_hourly_matrix(days=25, biases={"alpha": 1.5}, seed=3)
        clean = run_backtest(matrix, REQUEST, config)
        assert not clean.is_empty()

        truth_known = hourly_truth_known_at(matrix)
        folds = fold_plans(
            matrix["issue_time"], truth_known, config.backtest, "expanding"
        )
        known = truth_known.cast(pl.Int64).to_numpy()
        for fold in folds:
            origin_us = int(fold.origin.timestamp() * 1_000_000)
            poison_mask = pl.Series(known > origin_us)
            poisoned = matrix.with_columns(
                pl.when(poison_mask)
                .then(pl.col("t__temp_c__inst") + 1e6)
                .otherwise(pl.col("t__temp_c__inst"))
                .alias("t__temp_c__inst")
            )
            poisoned_scores = run_backtest(poisoned, REQUEST, config)
            fold_clean = clean.filter(pl.col("fold_origin") == fold.origin).sort(
                "method_id", "issue_time", "valid_time"
            )
            fold_poisoned = poisoned_scores.filter(
                pl.col("fold_origin") == fold.origin
            ).sort("method_id", "issue_time", "valid_time")
            assert fold_clean.height == fold_poisoned.height
            np.testing.assert_array_equal(
                fold_clean["y_pred"].to_numpy(),
                fold_poisoned["y_pred"].to_numpy(),
            )


class TestFreshInstances:
    def test_new_blender_per_fold(self, config):
        matrix = synthetic_hourly_matrix(days=25)
        instances = []

        def recording_factory():
            blender = get_factory("equal_weight")()
            instances.append(id(blender))
            return blender

        request = BacktestRequest(
            variables=(hourly_variable("temp_c"),), methods=("equal_weight",)
        )
        run_backtest(
            matrix, request, config, factories={"equal_weight": recording_factory}
        )
        assert len(instances) >= 2  # one per fold
        assert len(set(instances)) == len(instances)


class TestProvenanceWall:
    def test_mixed_matrix_rejected(self, config):
        live = synthetic_hourly_matrix(days=12)
        synthetic = synthetic_hourly_matrix(days=12, source_kind="synthetic")
        mixed = pl.concat([live, synthetic])
        with pytest.raises(MixedProvenanceError):
            run_backtest(mixed, REQUEST, config)
