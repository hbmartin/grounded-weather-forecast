"""Materialize truth tables from QC'd minute samples.

- ``truth_minute``: canonical variables in metric units, null when QC-flagged.
- ``truth_hourly``: BOTH semantics for state variables (``_inst`` = ±5-min
  centered mean with ±10-min fallback; ``_mean`` = hour-interval mean with a
  coverage gate), plus unambiguous gust max, reset-aware precip sum, and PoP.
- ``truth_daily``: local-calendar extremes and sums with DST-aware coverage.

Insufficient coverage yields null truth — rows drop out of training/scoring.
"""

from dataclasses import dataclass

import polars as pl

from grounded_weather_forecast.config import Config
from grounded_weather_forecast.dataset.qc import masked, qc_col
from grounded_weather_forecast.timeutil import local_date_expr, local_day_minutes
from grounded_weather_forecast.units import dew_point_expr, sea_level_pressure_expr

STATE_VARIABLES: tuple[str, ...] = (
    "temp_c",
    "humidity_pct",
    "dew_point_c",
    "wind_speed_ms",
    "pressure_sea_hpa",
)

_INST_WINDOW_MINUTES = 5
_INST_FALLBACK_MINUTES = 10
_GAP_ATTRIBUTION_LIMIT_MINUTES = 10.0
_MINUTES_PER_HOUR = 60


@dataclass(frozen=True, slots=True)
class TruthTables:
    minute: pl.DataFrame
    hourly: pl.DataFrame
    daily: pl.DataFrame


def truth_minute(minute_qc: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Map QC'd channels to canonical variables; flagged samples become null."""
    elevation = config.station.elevation_m
    temp = masked("temp")
    humidity = masked("humidity")
    return minute_qc.select(
        "ts",
        temp.alias("temp_c"),
        humidity.alias("humidity_pct"),
        dew_point_expr(temp, humidity).alias("dew_point_c"),
        masked("wind_speed").alias("wind_speed_ms"),
        masked("wind_gust").alias("wind_gust_ms"),
        sea_level_pressure_expr(masked("pressure_station"), elevation, temp).alias(
            "pressure_sea_hpa"
        ),
        masked("rain_counter").alias("rain_counter_mm"),
        *(
            pl.col(qc_col(channel))
            for channel in sorted(set(config.station.columns.values()))
            if qc_col(channel) in minute_qc.columns
        ),
    )


def _precip_deltas(minute: pl.DataFrame, reset_fraction: float) -> pl.DataFrame:
    """Reset-aware, noise-tolerant per-sample precipitation increments.

    The event counter climbs monotonically within an event and resets toward zero
    when the event ends, so within a *reset epoch* the true accumulation is the
    counter's running maximum: rain is credited only when the counter exceeds the
    highest value seen since the last reset. A dip-and-rebound (10.0 → 9.8 → 10.0)
    therefore contributes nothing, and a genuine reset — a drop below
    ``reset_fraction`` of the prior value — opens a new epoch whose value is the
    accumulation since it. The previous rule credited *every* decrease as a full
    reset, turning a one-count jitter into phantom rain. Deltas spanning gaps beyond
    the attribution limit are dropped as unattributable.
    """
    counter = pl.col("rain_counter_mm")
    previous = counter.shift(1)
    ordered = (
        minute.filter(counter.is_not_null())
        .sort("ts")
        .with_columns(
            (counter < reset_fraction * previous)
            .fill_null(value=False)
            .alias("_reset"),
            previous.is_null().alias("_first"),
            ((pl.col("ts") - pl.col("ts").shift(1)).dt.total_seconds() / 60.0).alias(
                "gap_minutes"
            ),
        )
        .with_columns(pl.col("_reset").cast(pl.Int32).cum_sum().alias("_epoch"))
        .with_columns(counter.cum_max().over("_epoch").alias("_epoch_max"))
        .with_columns(pl.col("_epoch_max").shift(1).alias("_prev_epoch_max"))
    )
    delta = (
        pl.when(pl.col("_first"))
        .then(None)
        .when(pl.col("_reset"))
        .then(counter)
        .when(counter > pl.col("_prev_epoch_max"))
        .then(counter - pl.col("_prev_epoch_max"))
        .otherwise(0.0)
    )
    return (
        ordered.with_columns(delta.alias("precip_delta_mm"))
        .filter(
            pl.col("precip_delta_mm").is_not_null()
            & (pl.col("gap_minutes") <= _GAP_ATTRIBUTION_LIMIT_MINUTES)
        )
        .select("ts", "precip_delta_mm")
    )


def _clean_minutes(variable: str) -> pl.Expr:
    return (
        pl.col("ts").dt.truncate("1m").filter(pl.col(variable).is_not_null()).n_unique()
    )


def _instantaneous(
    minute: pl.DataFrame, window_minutes: int, suffix: str
) -> pl.DataFrame:
    near = minute.with_columns(
        pl.col("ts").dt.round("1h").alias("valid_hour"),
        (pl.col("ts") - pl.col("ts").dt.round("1h"))
        .dt.total_seconds()
        .abs()
        .alias("dist_seconds"),
    ).filter(pl.col("dist_seconds") <= window_minutes * 60)
    return near.group_by("valid_hour").agg(
        *(
            pl.col(variable).mean().alias(f"{variable}_{suffix}")
            for variable in STATE_VARIABLES
        )
    )


def truth_hourly(minute: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Hourly truth with dual semantics for state variables."""
    min_coverage = config.dataset.min_hour_coverage
    pop_threshold = config.dataset.pop_threshold_mm

    interval = (
        minute.group_by(pl.col("ts").dt.truncate("1h").alias("valid_hour"))
        .agg(
            *(
                pl.col(variable).mean().alias(f"{variable}_mean_raw")
                for variable in STATE_VARIABLES
            ),
            *(
                (_clean_minutes(variable) / _MINUTES_PER_HOUR).alias(f"{variable}_cov")
                for variable in STATE_VARIABLES
            ),
            pl.col("wind_gust_ms").max().alias("wind_gust_max_raw"),
            (_clean_minutes("wind_gust_ms") / _MINUTES_PER_HOUR).alias(
                "wind_gust_ms_cov"
            ),
        )
        .with_columns(
            *(
                pl.when(pl.col(f"{variable}_cov") >= min_coverage)
                .then(pl.col(f"{variable}_mean_raw"))
                .otherwise(None)
                .alias(f"t__{variable}__mean")
                for variable in STATE_VARIABLES
            ),
            pl.when(pl.col("wind_gust_ms_cov") >= min_coverage)
            .then(pl.col("wind_gust_max_raw"))
            .otherwise(None)
            .alias("t__wind_gust_ms"),
        )
        .drop(
            *(f"{variable}_mean_raw" for variable in STATE_VARIABLES),
            "wind_gust_max_raw",
        )
    )

    inst_primary = _instantaneous(minute, _INST_WINDOW_MINUTES, "i5")
    inst_fallback = _instantaneous(minute, _INST_FALLBACK_MINUTES, "i10")
    inst = inst_primary.join(inst_fallback, on="valid_hour", how="full", coalesce=True)
    inst = inst.with_columns(
        *(
            pl.coalesce(pl.col(f"{variable}_i5"), pl.col(f"{variable}_i10")).alias(
                f"t__{variable}__inst"
            )
            for variable in STATE_VARIABLES
        )
    ).select("valid_hour", *(f"t__{variable}__inst" for variable in STATE_VARIABLES))

    deltas = _precip_deltas(minute, config.dataset.precip_reset_fraction)
    rain_channel_cov = minute.group_by(
        pl.col("ts").dt.truncate("1h").alias("valid_hour")
    ).agg((_clean_minutes("rain_counter_mm") / _MINUTES_PER_HOUR).alias("precip_cov"))
    precip = (
        deltas.group_by(pl.col("ts").dt.truncate("1h").alias("valid_hour"))
        .agg(pl.col("precip_delta_mm").sum().alias("precip_sum_raw"))
        .join(rain_channel_cov, on="valid_hour", how="full", coalesce=True)
        .with_columns(
            pl.when(pl.col("precip_cov") >= min_coverage)
            .then(pl.col("precip_sum_raw").fill_null(0.0))
            .otherwise(None)
            .alias("t__precip_mm")
        )
        .with_columns(
            pl.when(pl.col("t__precip_mm").is_null())
            .then(None)
            .otherwise((pl.col("t__precip_mm") >= pop_threshold).cast(pl.Float64))
            .alias("t__pop")
        )
        .select("valid_hour", "t__precip_mm", "t__pop", "precip_cov")
    )

    return (
        interval.join(inst, on="valid_hour", how="full", coalesce=True)
        .join(precip, on="valid_hour", how="full", coalesce=True)
        .sort("valid_hour")
    )


def truth_daily(minute: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Local-calendar daily truth with DST-aware coverage denominators."""
    timezone = config.station.timezone
    min_coverage = config.dataset.min_day_coverage
    pop_threshold = config.dataset.pop_threshold_mm

    by_day = (
        minute.with_columns(local_date_expr(pl.col("ts"), timezone).alias("date_local"))
        .group_by("date_local")
        .agg(
            pl.col("temp_c").max().alias("temp_max_raw"),
            pl.col("temp_c").min().alias("temp_min_raw"),
            _clean_minutes("temp_c").alias("temp_minutes"),
            _clean_minutes("rain_counter_mm").alias("rain_minutes"),
        )
        .sort("date_local")
    )
    day_lengths = pl.DataFrame(
        {
            "date_local": by_day["date_local"],
            "day_minutes": [
                local_day_minutes(d, timezone) for d in by_day["date_local"]
            ],
        }
    )
    deltas = _precip_deltas(minute, config.dataset.precip_reset_fraction).with_columns(
        local_date_expr(pl.col("ts"), timezone).alias("date_local")
    )
    precip = deltas.group_by("date_local").agg(
        pl.col("precip_delta_mm").sum().alias("precip_sum_raw")
    )
    return (
        by_day.join(day_lengths, on="date_local", how="left")
        .join(precip, on="date_local", how="left", coalesce=True)
        .with_columns(
            (pl.col("temp_minutes") / pl.col("day_minutes")).alias("coverage_frac"),
            (pl.col("rain_minutes") / pl.col("day_minutes")).alias("rain_coverage"),
        )
        .with_columns(
            pl.when(pl.col("coverage_frac") >= min_coverage)
            .then(pl.col("temp_max_raw"))
            .otherwise(None)
            .alias("t__temp_max_c"),
            pl.when(pl.col("coverage_frac") >= min_coverage)
            .then(pl.col("temp_min_raw"))
            .otherwise(None)
            .alias("t__temp_min_c"),
            pl.when(pl.col("rain_coverage") >= min_coverage)
            .then(pl.col("precip_sum_raw").fill_null(0.0))
            .otherwise(None)
            .alias("t__precip_sum_mm"),
        )
        .with_columns(
            pl.when(pl.col("t__precip_sum_mm").is_null())
            .then(None)
            .otherwise((pl.col("t__precip_sum_mm") >= pop_threshold).cast(pl.Float64))
            .alias("t__pop")
        )
        .select(
            "date_local",
            "t__temp_max_c",
            "t__temp_min_c",
            "t__precip_sum_mm",
            "t__pop",
            "coverage_frac",
        )
    )
