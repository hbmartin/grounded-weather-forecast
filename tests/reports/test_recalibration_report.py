"""The offline PIT-vs-CQR quantile recalibration A/B."""

import json
from datetime import timedelta

import numpy as np
import polars as pl

from grounded_weather_forecast.reports.recalibration import (
    _split_index,
    apply_cqr_margins,
    fit_cqr_margins,
    fit_pit_levels,
    quantile_cases,
    recalibration_report,
)
from grounded_weather_forecast.timeutil import utc

LEVELS = (0.05, 0.1, 0.25, 0.75, 0.9, 0.95)
_Z = {0.05: -1.645, 0.1: -1.282, 0.25: -0.674, 0.75: 0.674, 0.9: 1.282, 0.95: 1.645}


def make_scores(
    method_id="idr",
    lead_bucket="0-1h",
    n=200,
    assumed_sd=1.0,
    true_sd=2.0,
    bias=0.0,
    start_hour=0,
    seed=5,
):
    rng = np.random.default_rng(seed)
    y = bias + rng.normal(0.0, true_sd, n)
    start = utc(2026, 7, 1)
    rows = []
    for index in range(n):
        grid = [assumed_sd * _Z[level] for level in LEVELS]
        rows.append(
            {
                "product": "hourly",
                "variable": "temp_c",
                "semantics": "mean",
                "lead_bucket": lead_bucket,
                "method_id": method_id,
                "valid_time": start + timedelta(hours=start_hour + index),
                "y_true": float(y[index]),
                "quantiles_json": json.dumps(grid),
                "quantile_levels_json": json.dumps(list(LEVELS)),
                "evaluation_id": "eval1",
                "evaluation_created_at": start,
            }
        )
    return pl.DataFrame(rows)


def by_transform(report, lead_bucket="0-1h"):
    subset = report.filter(pl.col("lead_bucket") == lead_bucket)
    return {row["transform"]: row for row in subset.iter_rows(named=True)}


class TestTransforms:
    def test_cqr_restores_coverage_where_pit_is_grid_capped(self):
        report = recalibration_report(make_scores(n=400))
        rows = by_transform(report)
        assert rows["raw"]["coverage80"] < 0.6
        # margins widen past the native grid and restore the interval
        assert rows["cqr"]["coverage80"] >= 0.7
        assert rows["cqr"]["pinball"] < rows["raw"]["pinball"]
        # PIT remapping improves but cannot escape the outermost grid levels
        assert rows["pit"]["coverage80"] >= rows["raw"]["coverage80"]
        assert rows["pit"]["coverage80"] < rows["cqr"]["coverage80"]

    def test_constant_bias_margin_is_the_bias(self):
        y = np.full(99, 3.0)
        grids = np.zeros((99, len(LEVELS)))
        margins = fit_cqr_margins(y, grids, LEVELS)
        assert np.allclose(margins, 3.0)
        assert np.allclose(apply_cqr_margins(grids, margins), 3.0)

    def test_calibrated_input_stays_nearly_unchanged(self):
        scores = make_scores(assumed_sd=2.0, true_sd=2.0, n=400)
        rows = by_transform(recalibration_report(scores))
        for transform in ("raw", "pit", "cqr"):
            assert abs(rows[transform]["coverage80"] - 0.8) < 0.12

    def test_pit_levels_are_levels_when_calibrated(self):
        rng = np.random.default_rng(9)
        y = rng.normal(0.0, 1.0, 4000)
        grids = np.tile([_Z[level] for level in LEVELS], (4000, 1))
        adjusted = fit_pit_levels(y, grids, LEVELS)
        # inner levels recover their nominal values; the outer two saturate at
        # the clamped-PIT atoms (0 and 1), which interpolation maps back to
        # the native outer grid values — the documented grid-cap property
        assert np.allclose(adjusted[1:-1], LEVELS[1:-1], atol=0.03)
        assert adjusted[0] <= LEVELS[0]
        assert adjusted[-1] >= LEVELS[-1]


class TestSplitAndLadder:
    def test_split_is_strictly_chronological(self):
        times = np.arange(100, dtype=np.int64)
        split = _split_index(times, 0.7)
        assert times[:split].max() < times[split:].min()

    def test_thin_bucket_borrows_the_variable_pool(self):
        thin = make_scores(lead_bucket="12-24h", n=70, start_hour=300, seed=6)
        wide = make_scores(lead_bucket="0-1h", n=200, seed=7)
        report = recalibration_report(pl.concat([wide, thin]))
        rows = by_transform(report, "12-24h")
        assert rows["cqr"]["fit_scope"] == "variable"
        assert rows["cqr"]["n_fit"] > 70

    def test_thin_everything_is_skipped(self):
        report = recalibration_report(make_scores(n=70, start_hour=0))
        assert report.is_empty()

    def test_no_quantiles_yields_typed_empty_frame(self):
        frame = make_scores(n=30).drop("quantiles_json")
        report = recalibration_report(frame)
        assert report.is_empty()
        assert "transform" in report.columns

    def test_only_newest_evaluation_pools(self):
        newest = make_scores(n=200, seed=8)
        stale = make_scores(n=200, seed=9).with_columns(
            pl.lit("eval0").alias("evaluation_id"),
            pl.lit(utc(2026, 6, 1)).alias("evaluation_created_at"),
        )
        cases = list(quantile_cases(pl.concat([stale, newest])))
        assert len(cases) == 1
        assert cases[0].y.shape[0] == 200
