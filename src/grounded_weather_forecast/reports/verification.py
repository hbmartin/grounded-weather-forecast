"""Self-verification: score what we actually served against what happened.

Joins the emitted-forecast history to the truth that arrived afterwards, so the
leaderboard's backtest estimate can be compared with realized live skill. A gap
between the two is the signal that the serving path has drifted from the
backtested one — the one failure a backtest can never catch by itself.
"""

from pathlib import Path

import polars as pl

from grounded_weather_forecast.contracts import (
    TruthSemantics,
    hourly_variable,
    truth_col,
)
from grounded_weather_forecast.metrics.deterministic import bias, mae, rmse
from grounded_weather_forecast.serve.history import load_history

_MIN_SCORED = 5


def _truth_column(variable: str) -> str:
    try:
        spec = hourly_variable(variable)
    except KeyError:
        return truth_col(variable)
    if spec.has_dual_semantics:
        return truth_col(variable, TruthSemantics.INSTANTANEOUS)
    return truth_col(variable)


def verify_history(
    history_path: Path,
    truth_hourly: pl.DataFrame,
    truth_minute: pl.DataFrame | None = None,
    truth_daily: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Per (product, variable, method): realized MAE/RMSE/bias of served rows."""
    history = load_history(history_path)
    if history.is_empty() or truth_hourly.is_empty():
        return pl.DataFrame()
    rows: list[dict[str, object]] = []
    hourly = truth_hourly.rename({"valid_hour": "valid_time"})
    for key, group in history.partition_by(
        ["product", "variable"], as_dict=True
    ).items():
        product, variable = (str(part) for part in key)
        match product:
            case "minutely":
                column = variable
                truth = (
                    truth_minute.select(
                        pl.col("ts").dt.truncate("1m").alias("valid_time"),
                        column,
                    )
                    .group_by("valid_time")
                    .agg(pl.col(column).mean())
                    if truth_minute is not None and column in truth_minute.columns
                    else pl.DataFrame()
                )
                join_key = "valid_time"
            case "daily":
                truth = (
                    truth_daily.rename({"date_local": "valid_date"})
                    if truth_daily is not None
                    else pl.DataFrame()
                )
                column = truth_col(variable)
                join_key = "valid_date"
            case _:
                truth = hourly
                column = _truth_column(variable)
                join_key = "valid_time"
        if truth.is_empty() or column not in truth.columns:
            continue
        joined = group.join(
            truth.select(join_key, column), on=join_key, how="inner"
        ).drop_nulls([column, "y_pred"])
        if joined.height < _MIN_SCORED:
            continue
        for method_id, scored in joined.partition_by("method_id", as_dict=True).items():
            pred = scored["y_pred"].to_numpy()
            y = scored[column].to_numpy()
            rows.append(
                {
                    "product": product,
                    "variable": variable,
                    "method_id": str(method_id[0]),
                    "n": scored.height,
                    "live_mae": mae(pred, y),
                    "live_rmse": rmse(pred, y),
                    "live_bias": bias(pred, y),
                }
            )
    return pl.DataFrame(rows).sort("product", "variable", "live_mae")


def compare_to_backtest(live: pl.DataFrame, board: pl.DataFrame) -> pl.DataFrame:
    """Live MAE beside the backtest's expectation for the same method."""
    if live.is_empty() or board.is_empty():
        return live
    expected = board.group_by("product", "variable", "method_id").agg(
        ((pl.col("mae") * pl.col("n")).sum() / pl.col("n").sum()).alias("backtest_mae")
    )
    return (
        live.join(expected, on=["product", "variable", "method_id"], how="left")
        .with_columns((pl.col("live_mae") - pl.col("backtest_mae")).alias("mae_gap"))
        .sort("product", "variable", "live_mae")
    )
