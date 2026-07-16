"""Timestamp parsing and DST-safe local-day arithmetic.

Station timestamps are UTC-naive text with *mixed* sub-second precision
(``2026-07-13 19:21:03.297163`` and ``2025-06-14 18:39:53`` coexist).
Forecast-archive timestamps are ISO 8601 with an explicit offset. Everything
is normalized to timezone-aware UTC microsecond datetimes; local calendar
bucketing goes through ``zoneinfo`` so DST transitions stay correct.
"""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import polars as pl

MINUTES_PER_STANDARD_DAY = 1440


def parse_station_ts_expr(ts: pl.Expr) -> pl.Expr:
    """Parse aw2sqlite ``ts`` text (UTC-naive, optional microseconds) to UTC."""
    with_fraction = ts.str.to_datetime(
        "%Y-%m-%d %H:%M:%S%.f", time_unit="us", strict=False
    )
    without_fraction = ts.str.to_datetime(
        "%Y-%m-%d %H:%M:%S", time_unit="us", strict=False
    )
    return pl.coalesce(with_fraction, without_fraction).dt.replace_time_zone("UTC")


def parse_iso_utc_expr(ts: pl.Expr) -> pl.Expr:
    """Parse ISO 8601 text with offset (archive ``fetched_at``) to UTC."""
    return ts.str.to_datetime(time_unit="us", time_zone="UTC")


def local_date_expr(dt: pl.Expr, timezone: str) -> pl.Expr:
    """Local calendar date of a UTC datetime column."""
    return dt.dt.convert_time_zone(timezone).dt.date()


def local_day_start_utc(day: date, timezone: str) -> datetime:
    """UTC instant at which a local calendar day begins."""
    return datetime.combine(day, time(), tzinfo=ZoneInfo(timezone)).astimezone(
        ZoneInfo("UTC")
    )


def local_day_minutes(day: date, timezone: str) -> int:
    """Length of a local calendar day in minutes (1380/1440/1500 across DST)."""
    start = local_day_start_utc(day, timezone)
    end = local_day_start_utc(day + timedelta(days=1), timezone)
    return int((end - start).total_seconds() // 60)


def utc(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    microsecond: int = 0,
) -> datetime:
    """Terse aware-UTC datetime constructor."""
    return datetime(
        year, month, day, hour, minute, second, microsecond, tzinfo=ZoneInfo("UTC")
    )
