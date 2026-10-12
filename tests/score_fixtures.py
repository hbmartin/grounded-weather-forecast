"""Shared score archives for serving, reporting, and dashboard tests."""

from datetime import timedelta

import polars as pl

from grounded_weather_forecast.backtest.scores import SCORES_SCHEMA
from grounded_weather_forecast.timeutil import utc


def scores(
    evaluation: str = "eval",
    day: int = 1,
    methods: tuple[str, ...] = (
        "equal_weight",
        "best_provider",
        "damped_grounded_equal_weight",
        "challenger",
    ),
) -> pl.DataFrame:
    rows = []
    for method in methods:
        for step in range(8):
            rows.append(
                {
                    "method_id": method,
                    "variable": "temp_c",
                    "product": "hourly",
                    "source_kind": "live",
                    "evaluation_id": evaluation,
                    "evaluation_created_at": utc(2026, 8, day),
                    "dataset_fingerprint": "dataset",
                    "config_fingerprint": "config",
                    "code_version": "code",
                    "source_set_json": '["nws"]',
                    "feature_set_json": '["feature"]',
                    "semantics": "inst",
                    "window": "expanding",
                    "fold_origin": utc(2026, 7, 1),
                    "issue_time": utc(2026, 7, 1) + timedelta(hours=step),
                    "valid_time": utc(2026, 7, 1, 1) + timedelta(hours=step),
                    "lead_hours": 1.0,
                    "lead_bucket": "1-3h",
                    "y_pred": 1.0,
                    "y_true": 0.0,
                    "quantile_levels_json": "[]",
                    "quantiles_json": None,
                }
            )
    return pl.DataFrame(rows, schema=SCORES_SCHEMA)
