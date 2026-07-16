"""Rolling-origin fold plans keyed by issue time.

Training rows are selected by ``truth_known_at <= origin`` — NOT merely
``issue_time <= origin`` — because a row issued yesterday about tomorrow has
unknown truth today and must not be trained on. Test rows are the snapshots
issued in ``(origin, origin + step]``. Expanding and rolling windows are both
supported and reported side by side.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl

from grounded_weather_forecast.config import BacktestConfig
from grounded_weather_forecast.timeutil import local_day_start_utc

type WindowMode = Literal["expanding", "rolling"]

_HOURLY_TRUTH_DELAY = timedelta(hours=2)
_DAILY_TRUTH_DELAY = timedelta(hours=1)


@dataclass(frozen=True, slots=True)
class Fold:
    origin: datetime
    train_rows: npt.NDArray[np.int64]
    test_rows: npt.NDArray[np.int64]


def hourly_truth_known_at(frame: pl.DataFrame) -> pl.Series:
    """Interval truth for hour ``[H, H+1)`` is realized by H+1 (+1h ingest lag)."""
    return frame.select(
        (pl.col("valid_time") + _HOURLY_TRUTH_DELAY).alias("truth_known_at")
    )["truth_known_at"]


def daily_truth_known_at(frame: pl.DataFrame, timezone: str) -> pl.Series:
    """Daily truth is realized when the local day ends (+1h ingest lag)."""
    dates: list[date] = frame["forecast_date"].to_list()
    known = [
        local_day_start_utc(day + timedelta(days=1), timezone) + _DAILY_TRUTH_DELAY
        for day in dates
    ]
    return pl.Series("truth_known_at", known, dtype=pl.Datetime("us", "UTC"))


_US_PER_DAY = 86_400_000_000


def fold_plans(
    issue_time: pl.Series,
    truth_known_at: pl.Series,
    backtest: BacktestConfig,
    window: WindowMode,
) -> list[Fold]:
    """Build inspectable fold plans; may be empty if data is too short."""
    if issue_time.is_empty():
        return []
    if backtest.step_days <= 0:
        msg = f"step_days must be positive, got {backtest.step_days}"
        raise ValueError(msg)
    if backtest.initial_train_days <= 0 or backtest.rolling_window_days <= 0:
        msg = "initial_train_days and rolling_window_days must be positive"
        raise ValueError(msg)
    issues = issue_time.cast(pl.Int64).to_numpy()  # epoch microseconds, UTC
    known = truth_known_at.cast(pl.Int64).to_numpy()
    step_us = backtest.step_days * _US_PER_DAY
    rolling_us = backtest.rolling_window_days * _US_PER_DAY
    start_us = int(issues.min())
    end_us = int(issues.max())
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    folds: list[Fold] = []
    origin_us = start_us + backtest.initial_train_days * _US_PER_DAY
    while origin_us < end_us:
        train_mask = known <= origin_us
        if window == "rolling":
            train_mask &= issues > origin_us - rolling_us
        test_mask = (issues > origin_us) & (issues <= origin_us + step_us)
        if train_mask.any() and test_mask.any():
            folds.append(
                Fold(
                    origin=epoch + timedelta(microseconds=origin_us),
                    train_rows=np.flatnonzero(train_mask).astype(np.int64),
                    test_rows=np.flatnonzero(test_mask).astype(np.int64),
                )
            )
        origin_us += step_us
    return folds
