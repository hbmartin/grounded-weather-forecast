from datetime import date, timedelta

import numpy as np
import polars as pl
from conftest import utc, write_config
from hypothesis import given, settings
from hypothesis import strategies as st

from grounded_weather_forecast.backtest.splits import (
    daily_truth_known_at,
    fold_plans,
    hourly_truth_known_at,
)


def backtest_config(tmp_path, **kw):
    extra = "\n[backtest]\n" + "\n".join(f"{k} = {v}" for k, v in kw.items())
    return write_config(tmp_path, extra_toml=extra).backtest


class TestTruthKnownAt:
    def test_hourly(self):
        frame = pl.DataFrame(
            {"valid_time": [utc(2026, 3, 22, 18)]},
            schema={"valid_time": pl.Datetime("us", "UTC")},
        )
        assert hourly_truth_known_at(frame)[0] == utc(2026, 3, 22, 20)

    def test_daily_ends_at_local_midnight(self):
        frame = pl.DataFrame({"forecast_date": [date(2026, 3, 22)]})
        known = daily_truth_known_at(frame, "America/Los_Angeles")
        # Mar 23 00:00 PDT == Mar 23 07:00 UTC, +1h ingest lag
        assert known[0] == utc(2026, 3, 23, 8)


class TestFoldPlans:
    def make_series(self, n_days, snapshots_per_day=2, max_lead_hours=48):
        issues, knowns = [], []
        start = utc(2026, 1, 1)
        for day in range(n_days):
            for snap in range(snapshots_per_day):
                issue = start + timedelta(days=day, hours=12 * snap)
                for lead in (1, max_lead_hours):
                    issues.append(issue)
                    knowns.append(issue + timedelta(hours=lead + 2))
        return (
            pl.Series("issue", issues, dtype=pl.Datetime("us", "UTC")),
            pl.Series("known", knowns, dtype=pl.Datetime("us", "UTC")),
        )

    def test_basic_folds(self, tmp_path):
        config = backtest_config(
            tmp_path, initial_train_days=10, step_days=5, rolling_window_days=20
        )
        issues, knowns = self.make_series(30)
        folds = fold_plans(issues, knowns, config, "expanding")
        assert len(folds) >= 3
        issue_np = issues.cast(pl.Int64).to_numpy()
        known_np = knowns.cast(pl.Int64).to_numpy()
        for fold in folds:
            origin_us = int(fold.origin.timestamp() * 1_000_000)
            assert known_np[fold.train_rows].max() <= origin_us
            assert issue_np[fold.test_rows].min() > origin_us
            assert not set(fold.train_rows) & set(fold.test_rows)

    def test_rolling_window_bounds_train_span(self, tmp_path):
        config = backtest_config(
            tmp_path, initial_train_days=25, step_days=5, rolling_window_days=10
        )
        issues, knowns = self.make_series(40)
        folds = fold_plans(issues, knowns, config, "rolling")
        issue_np = issues.cast(pl.Int64).to_numpy()
        for fold in folds:
            span_us = issue_np[fold.train_rows].max() - issue_np[fold.train_rows].min()
            assert span_us <= 10 * 86_400_000_000

    def test_empty_series(self, tmp_path):
        config = backtest_config(tmp_path)
        empty = pl.Series("x", [], dtype=pl.Datetime("us", "UTC"))
        assert fold_plans(empty, empty, config, "expanding") == []

    @settings(max_examples=25, deadline=None)
    @given(
        n_days=st.integers(min_value=5, max_value=60),
        initial=st.integers(min_value=1, max_value=30),
        step=st.integers(min_value=1, max_value=14),
        window_days=st.integers(min_value=2, max_value=30),
        mode=st.sampled_from(["expanding", "rolling"]),
        truth_delay_hours=st.integers(min_value=1, max_value=72),
    )
    def test_invariants_hold_for_random_configs(
        self,
        tmp_path_factory,
        n_days,
        initial,
        step,
        window_days,
        mode,
        truth_delay_hours,
    ):
        tmp_path = tmp_path_factory.mktemp("cfg")
        config = backtest_config(
            tmp_path,
            initial_train_days=initial,
            step_days=step,
            rolling_window_days=window_days,
        )
        start = utc(2026, 1, 1)
        issues, knowns = [], []
        for day in range(n_days):
            issue = start + timedelta(days=day)
            issues.append(issue)
            knowns.append(issue + timedelta(hours=truth_delay_hours))
        issue_series = pl.Series("i", issues, dtype=pl.Datetime("us", "UTC"))
        known_series = pl.Series("k", knowns, dtype=pl.Datetime("us", "UTC"))
        folds = fold_plans(issue_series, known_series, config, mode)
        issue_np = issue_series.cast(pl.Int64).to_numpy()
        known_np = known_series.cast(pl.Int64).to_numpy()
        for fold in folds:
            origin_us = int(fold.origin.timestamp() * 1_000_000)
            assert known_np[fold.train_rows].max() <= origin_us
            assert issue_np[fold.test_rows].min() > origin_us
            assert len(np.intersect1d(fold.train_rows, fold.test_rows)) == 0
