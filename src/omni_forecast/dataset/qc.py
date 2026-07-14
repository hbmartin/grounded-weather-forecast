"""Truth quality control: plausibility bounds, spike filter, flatline detection.

Adds a ``{channel}_qc`` UInt8 bitmask column per channel. Flagged samples are
excluded from truth downstream — never corrected or imputed.
"""

from collections.abc import Sequence

import polars as pl

from omni_forecast.config import QcConfig

QC_OK = 0
QC_OUT_OF_BOUNDS = 1
QC_SPIKE = 2
QC_FLATLINE = 4
_MAX_CONTIGUOUS_GAP_MINUTES = 5.0


def qc_col(channel: str) -> str:
    return f"{channel}_qc"


def _bounds_flag(channel: str, low: float, high: float) -> pl.Expr:
    value = pl.col(channel)
    return (
        pl.when(value.is_null())
        .then(0)
        .when((value < low) | (value > high))
        .then(QC_OUT_OF_BOUNDS)
        .otherwise(0)
    )


def _spike_flag(channel: str, max_step_per_minute: float) -> pl.Expr:
    """Flag isolated spikes: a jump up and back (or down and back) that exceeds
    the per-minute rate limit against *both* neighbors, with opposite signs."""
    value = pl.col(channel)
    minutes_prev = (
        (pl.col("ts") - pl.col("ts").shift(1)).dt.total_seconds() / 60.0
    ).clip(lower_bound=1.0)
    minutes_next = (
        (pl.col("ts").shift(-1) - pl.col("ts")).dt.total_seconds() / 60.0
    ).clip(lower_bound=1.0)
    diff_prev = value - value.shift(1)
    diff_next = value.shift(-1) - value
    exceeds_prev = diff_prev.abs() > max_step_per_minute * minutes_prev
    exceeds_next = diff_next.abs() > max_step_per_minute * minutes_next
    opposite = diff_prev * diff_next < 0
    return (
        pl.when(exceeds_prev & exceeds_next & opposite)
        .then(QC_SPIKE)
        .otherwise(0)
        .fill_null(0)
    )


def _flatline_run_id(channel: str) -> pl.Expr:
    value = pl.col(channel)
    gap_minutes = (pl.col("ts") - pl.col("ts").shift(1)).dt.total_seconds() / 60.0
    changed = (
        (value != value.shift(1))
        | value.is_null()
        | value.shift(1).is_null()
        | (gap_minutes > _MAX_CONTIGUOUS_GAP_MINUTES)
    )
    return changed.fill_null(value=True).cum_sum()


def _flatline_flag(channel: str, min_minutes: int, *, causal: bool = False) -> pl.Expr:
    """Flag runs of identical consecutive values lasting at least ``min_minutes``."""
    value = pl.col(channel)
    run_id = _flatline_run_id(channel)
    if causal:
        run_minutes = (
            pl.col("ts") - pl.col("ts").first().over(run_id)
        ).dt.total_seconds() / 60.0
    else:
        run_minutes = (pl.col("ts").max() - pl.col("ts").min()).dt.total_seconds().over(
            run_id
        ) / 60.0
    return (
        pl.when(value.is_not_null() & (run_minutes >= float(min_minutes)))
        .then(QC_FLATLINE)
        .otherwise(0)
    )


def apply_qc(
    minute: pl.DataFrame, qc: QcConfig, channels: Sequence[str]
) -> pl.DataFrame:
    """Add ``{channel}_qc`` bitmask columns; input must be ts-sorted."""
    flags: list[pl.Expr] = []
    for channel in channels:
        if channel not in minute.columns:
            continue
        flag: pl.Expr = pl.lit(QC_OK, dtype=pl.UInt8)
        if (bounds := qc.bounds.get(channel)) is not None:
            flag = flag | _bounds_flag(channel, *bounds).cast(pl.UInt8)
        if (max_step := qc.max_step.get(channel)) is not None:
            flag = flag | _spike_flag(channel, max_step).cast(pl.UInt8)
        if (flatline := qc.flatline_minutes.get(channel)) is not None:
            flag = flag | _flatline_flag(channel, flatline).cast(pl.UInt8)
        flags.append(flag.alias(qc_col(channel)))
    return minute.with_columns(flags)


def apply_causal_qc(
    minute: pl.DataFrame, qc: QcConfig, channels: Sequence[str]
) -> pl.DataFrame:
    """QC suitable for issue-time features without consulting future samples.

    Bounds are immediate. Flatlines are flagged only from the instant their
    duration crosses the threshold. The two-sided isolated-spike rule is
    intentionally excluded because it requires a following observation.
    """
    flags: list[pl.Expr] = []
    for channel in channels:
        if channel not in minute.columns:
            continue
        flag: pl.Expr = pl.lit(QC_OK, dtype=pl.UInt8)
        if (bounds := qc.bounds.get(channel)) is not None:
            flag = flag | _bounds_flag(channel, *bounds).cast(pl.UInt8)
        if (flatline := qc.flatline_minutes.get(channel)) is not None:
            flag = flag | _flatline_flag(channel, flatline, causal=True).cast(pl.UInt8)
        flags.append(flag.alias(qc_col(channel)))
    return minute.with_columns(flags)


def masked(channel: str) -> pl.Expr:
    """Channel value with QC-flagged samples nulled."""
    return (
        pl.when(pl.col(qc_col(channel)) == QC_OK).then(pl.col(channel)).otherwise(None)
    )


def qc_summary(minute: pl.DataFrame, channels: Sequence[str]) -> pl.DataFrame:
    """Per-channel counts of samples, nulls, and each flag kind."""
    rows: list[dict[str, object]] = []
    for channel in channels:
        if qc_col(channel) not in minute.columns:
            continue
        flag = minute[qc_col(channel)]
        rows.append(
            {
                "channel": channel,
                "samples": minute.height,
                "missing": int(minute[channel].null_count()),
                "out_of_bounds": int(((flag & QC_OUT_OF_BOUNDS) > 0).sum()),
                "spike": int(((flag & QC_SPIKE) > 0).sum()),
                "flatline": int(((flag & QC_FLATLINE) > 0).sum()),
                "clean": int(((flag == QC_OK) & minute[channel].is_not_null()).sum()),
            }
        )
    return pl.DataFrame(rows)
