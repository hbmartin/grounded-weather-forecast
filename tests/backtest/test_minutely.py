"""The minutely path-construction backtest: math, methods, runner, direction."""

from datetime import timedelta

import numpy as np
import polars as pl
from conftest import synthetic_hourly_matrix, utc, write_config

from grounded_weather_forecast.backtest.minutely import (
    AnchorFull,
    AnchorTau,
    InterpPath,
    MinuteFrame,
    MinutelyRequest,
    ObsPersistence,
    _FittedResponse,
    anchor_weight,
    lead_zero_path,
    minutely_methods,
    now_forecast,
    run_minutely_backtest,
)
from grounded_weather_forecast.backtest.scores import SCORES_SCHEMA
from grounded_weather_forecast.leads import MINUTELY_BUCKET_LABELS


def minutely_case(days=25, snapshot_bias_sd=0.0, seed=0):
    """Hourly matrix + per-minute truth grid over the same sinusoid.

    ``snapshot_bias_sd`` adds one shared error to every lead of a snapshot —
    exactly the issue-time bias the anchor residual exists to remove, so a
    large value makes anchoring win and zero makes it useless beyond 0-5m.
    """
    matrix = synthetic_hourly_matrix(days=days, max_lead=6, noise_sd=0.3, seed=seed)
    if snapshot_bias_sd:
        rng = np.random.default_rng(seed + 1)
        issues = matrix["issue_time"].unique().sort()
        bias_by_issue = dict(
            zip(
                issues.to_list(),
                rng.normal(0.0, snapshot_bias_sd, issues.len()),
                strict=True,
            )
        )
        bias = pl.Series(
            "___bias",
            [bias_by_issue[issue] for issue in matrix["issue_time"].to_list()],
        )
        matrix = matrix.with_columns(
            (pl.col("fx__alpha__temp_c") + bias).alias("fx__alpha__temp_c"),
            (pl.col("fx__beta__temp_c") + bias).alias("fx__beta__temp_c"),
        )
    start = utc(2026, 1, 1)
    minutes = int(days * 24 * 60)
    times = [start + timedelta(minutes=m) for m in range(minutes)]
    truth = [
        10.0
        + 8.0 * float(np.sin(2 * np.pi * ((t.hour + t.minute / 60.0) - 15) / 24))
        + 5.0 * float(np.sin(2 * np.pi * t.timetuple().tm_yday / 365))
        for t in times
    ]
    grid = pl.DataFrame({"valid_time": times, "temp_c": truth}).with_columns(
        pl.col("valid_time").dt.replace_time_zone(None).dt.replace_time_zone("UTC")
    )
    return matrix, grid


def frame(leads, base, observation, residual, y_true):
    n = len(leads)
    issue = utc(2026, 1, 1)
    return MinuteFrame(
        issue_time=pl.Series("issue_time", [issue] * n, dtype=pl.Datetime("us", "UTC")),
        valid_time=pl.Series(
            "valid_time",
            [issue + timedelta(minutes=i + 1) for i in range(n)],
            dtype=pl.Datetime("us", "UTC"),
        ),
        lead_hours=np.asarray(leads, dtype=np.float64),
        base=np.asarray(base, dtype=np.float64),
        observation=np.asarray(observation, dtype=np.float64),
        residual=np.asarray(residual, dtype=np.float64),
        y_true=np.asarray(y_true, dtype=np.float64),
    )


class TestPathMath:
    def test_now_forecast_extrapolates_past_the_first_lead(self):
        leads = np.asarray([0.5, 1.5])
        path = np.asarray([10.0, 12.0])
        # slope 2/h back from lead 0.5: 10 - 2*0.5 = 9
        assert now_forecast(leads, path) == 9.0

    def test_lead_zero_path_prepends_the_extrapolation(self):
        leads, path = lead_zero_path(np.asarray([1.5, 0.5]), np.asarray([12.0, 10.0]))
        assert leads.tolist() == [0.0, 0.5, 1.5]
        assert path.tolist() == [9.0, 10.0, 12.0]

    def test_anchor_weight_floors_at_five_percent(self):
        # tau 0.25h at minute 45: exp(-3) = 0.0498 < 0.05 -> hard zero
        weights = anchor_weight(np.asarray([44.0 / 60.0, 45.0 / 60.0]), 0.25)
        assert weights[0] > 0.0
        assert weights[1] == 0.0


class TestMethods:
    def test_closed_form_constructions(self):
        test = frame(
            leads=[1 / 60, 30 / 60],
            base=[10.0, 11.0],
            observation=[12.0, 12.0],
            residual=[2.0, 2.0],
            y_true=[12.0, 12.5],
        )
        assert InterpPath().predict(test).tolist() == [10.0, 11.0]
        assert ObsPersistence().predict(test).tolist() == [12.0, 12.0]
        full = AnchorFull().predict(test)
        assert full.tolist() == [12.0, 13.0]
        tau = AnchorTau(0.5).predict(test)
        expected_first = 10.0 + np.exp(-(1 / 60) / 0.5) * 2.0
        assert abs(tau[0] - expected_first) < 1e-12

    def test_missing_observation_degrades_anchor_to_the_base(self):
        test = frame(
            leads=[0.5],
            base=[10.0],
            observation=[np.nan],
            residual=[np.nan],
            y_true=[10.0],
        )
        assert AnchorFull().predict(test).tolist() == [10.0]
        assert np.isnan(ObsPersistence().predict(test)[0])

    def test_fitted_slope_clips_and_falls_back_globally(self):
        rng = np.random.default_rng(4)
        n = 400
        residual = rng.normal(0.0, 1.0, n)
        leads = np.full(n, 20.0 / 60.0)  # all rows in one bucket
        # true response 3.0 -> must clip at the RAFT ceiling 1.5
        train = frame(
            leads=leads,
            base=np.zeros(n),
            observation=residual,
            residual=residual,
            y_true=3.0 * residual,
        )
        fitted = _FittedResponse(method_id="s", clip=(-0.5, 1.5)).fit(train)
        prediction = fitted.predict(frame([20.0 / 60.0], [0.0], [1.0], [1.0], [0.0]))
        assert abs(prediction[0] - 1.5) < 1e-9
        # a bucket with no rows serves the global fit, not zero
        other_bucket = fitted.predict(frame([2.0 / 60.0], [0.0], [1.0], [1.0], [0.0]))
        assert abs(other_bucket[0] - 1.5) < 1e-9

    def test_ramp_weights_are_monotone_non_increasing(self):
        rng = np.random.default_rng(5)
        n = 1200
        leads = rng.choice([2, 10, 20, 40, 55], n) / 60.0
        residual = rng.normal(0.0, 1.0, n)
        # response decays with lead: 1.0 near zero, ~0.2 at the horizon
        response = np.clip(1.2 - 1.2 * leads, 0.0, 1.0)
        train = frame(
            leads=leads,
            base=np.zeros(n),
            observation=residual,
            residual=residual,
            y_true=response * residual + rng.normal(0.0, 0.05, n),
        )
        fitted = _FittedResponse(
            method_id="r", clip=(0.0, 1.0), monotone_non_increasing=True
        ).fit(train)
        weights = fitted._weights
        assert (np.diff(weights) <= 1e-12).all()
        assert 0.0 <= weights[-1] <= weights[0] <= 1.0

    def test_method_set_is_stable(self):
        ids = [method.method_id for method in minutely_methods()]
        assert ids == [
            "minutely_interp",
            "minutely_persistence",
            "minutely_anchor_tau_0.25h",
            "minutely_anchor_tau_0.5h",
            "minutely_anchor_tau_1h",
            "minutely_anchor_tau_3h",
            "minutely_anchor_full",
            "minutely_ramp",
            "minutely_fitted_slope",
        ]


class TestRunner:
    def scores_for(self, tmp_path, **case_kwargs):
        config = write_config(
            tmp_path,
            extra_toml="\n[backtest]\ninitial_train_days = 10\nstep_days = 5\n",
        )
        matrix, grid = minutely_case(**case_kwargs)
        return run_minutely_backtest(matrix, grid, MinutelyRequest(), config)

    def test_schema_stamping_and_buckets(self, tmp_path):
        scores = self.scores_for(tmp_path)
        assert scores.schema == SCORES_SCHEMA
        assert set(scores["product"].unique()) == {"minutely"}
        assert set(scores["semantics"].unique()) == {"inst"}
        assert set(scores["lead_bucket"].unique()) <= set(MINUTELY_BUCKET_LABELS)
        # minute 60 (lead exactly 1.0) stays inside the product
        minute60 = scores.filter(pl.col("lead_hours") == 1.0)
        assert set(minute60["lead_bucket"].unique()) == {"45-60m"}
        assert set(scores["method_id"].unique()) == {
            method.method_id for method in minutely_methods()
        }

    def test_no_test_row_precedes_its_fold_origin(self, tmp_path):
        scores = self.scores_for(tmp_path)
        assert scores.filter(pl.col("issue_time") <= pl.col("fold_origin")).is_empty()

    def test_anchoring_wins_when_issue_bias_persists(self, tmp_path):
        scores = self.scores_for(tmp_path, snapshot_bias_sd=3.0)
        mae = (
            scores.drop_nulls(["y_pred", "y_true"])
            .with_columns((pl.col("y_pred") - pl.col("y_true")).abs().alias("ae"))
            .group_by("method_id")
            .agg(pl.col("ae").mean())
        )
        by_method = dict(mae.iter_rows())
        # the anchor residual removes the injected per-snapshot bias
        assert by_method["minutely_anchor_full"] < 0.5 * by_method["minutely_interp"]

    def test_unbiased_snapshots_favor_persistence_near_the_observation(self, tmp_path):
        scores = self.scores_for(tmp_path, snapshot_bias_sd=0.0)
        mae = dict(
            scores.drop_nulls(["y_pred", "y_true"])
            .with_columns((pl.col("y_pred") - pl.col("y_true")).abs().alias("ae"))
            .group_by("method_id")
            .agg(pl.col("ae").mean())
            .iter_rows()
        )
        assert mae["minutely_anchor_full"] < 1.5 * mae["minutely_interp"]
        # near the observation, provider noise (0.3) dwarfs the <=5-minute
        # sinusoid drift (~0.06): the nowcast floor crushes pure
        # interpolation there (with exact synthetic observations the anchored
        # constructions beat both — the real-data persistence win at 0-5m is
        # an empirical property of imperfect observations, not a contract)
        near = dict(
            scores.filter(pl.col("lead_bucket") == "0-5m")
            .drop_nulls(["y_pred", "y_true"])
            .with_columns((pl.col("y_pred") - pl.col("y_true")).abs().alias("ae"))
            .group_by("method_id")
            .agg(pl.col("ae").mean())
            .iter_rows()
        )
        assert near["minutely_persistence"] < 0.5 * near["minutely_interp"]

    def test_empty_inputs_yield_empty_scores(self, tmp_path):
        config = write_config(tmp_path)
        matrix, grid = minutely_case(days=2)
        empty = run_minutely_backtest(matrix.head(0), grid, MinutelyRequest(), config)
        assert empty.is_empty()
        assert empty.schema == SCORES_SCHEMA


def test_runner_never_emits_unscoreable_rows(tmp_path):
    """Null y_true would reach Selection.mae as NaN and disarm the live gate."""
    config = write_config(
        tmp_path,
        extra_toml="\n[backtest]\ninitial_train_days = 10\nstep_days = 5\n",
    )
    matrix, grid = minutely_case(days=25)
    # punch a truth hole covering some test minutes
    holed = grid.filter(
        ~pl.col("valid_time").is_between(
            utc(2026, 1, 18), utc(2026, 1, 19), closed="left"
        )
    )
    scores = run_minutely_backtest(matrix, holed, MinutelyRequest(), config)
    assert scores.height > 0
    assert scores["y_true"].null_count() == 0
    assert scores["y_true"].is_nan().sum() == 0
