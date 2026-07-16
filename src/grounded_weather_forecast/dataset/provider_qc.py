"""Plausibility QC for provider (forecast) values, applied before grounding.

Provider rows are otherwise trusted verbatim, so a single bad value (a snow depth
written into a liquid field, a pressure in the wrong unit, one provider's daily
low that is wildly colder than every other source) flows straight into the affine
grounding fit and the blend. Two conservative filters null implausible provider
values in the canonical long frame; a nulled value becomes ``NaN`` in the matrix
and is dropped by every blender's availability mask, so no blender needs to change.

- **Absolute physical bounds** per variable catch gross unit/garbage errors.
- **A robust cross-source outlier pass** nulls a value that disagrees with the
  other providers at the same valid time by more than ``mad_k`` scaled MADs *and*
  an absolute floor. It is deliberately conservative — only gross outliers are
  removed — because genuine provider diversity is what the blend relies on.

Truth is never consulted here; this operates purely on provider forecasts, so it
introduces no leakage.
"""

import logging
from collections.abc import Sequence

import polars as pl

from grounded_weather_forecast.config import Config, ProviderQcConfig

_logger = logging.getLogger(__name__)
_MAD_TO_STD = 1.4826  # scales the MAD to a std-equivalent for a normal sample


def _bounded(column: str, low: float, high: float) -> pl.Expr:
    """Null the column where it falls outside ``[low, high]`` (nulls pass through)."""
    value = pl.col(column)
    return (
        pl.when(value.is_null())
        .then(value)
        .when((value < low) | (value > high))
        .then(None)
        .otherwise(value)
        .alias(column)
    )


def _mask_cross_source(
    frame: pl.DataFrame,
    column: str,
    group_key: str | Sequence[str],
    qc: ProviderQcConfig,
) -> pl.DataFrame:
    """Null values that are robust outliers among the sources sharing a snapshot.

    Intermediates (per-group median, absolute deviation, MAD, available count) are
    materialized as temporary columns so every window is a single, standard
    aggregation over a real column — no nested window expressions.
    """
    value = pl.col(column)
    med = f"__med_{column}"
    dev = f"__dev_{column}"
    mad = f"__mad_{column}"
    count = f"__cnt_{column}"
    frame = frame.with_columns(
        value.median().over(group_key).alias(med),
        value.is_not_null().sum().over(group_key).alias(count),
    )
    frame = frame.with_columns((value - pl.col(med)).abs().alias(dev))
    frame = frame.with_columns(pl.col(dev).median().over(group_key).alias(mad))
    floor = pl.lit(qc.min_deviation.get(column, 0.0))
    threshold = pl.max_horizontal(qc.mad_k * _MAD_TO_STD * pl.col(mad), floor)
    is_outlier = (
        value.is_not_null()
        & (pl.col(count) >= qc.min_sources)
        & (pl.col(dev) > threshold)
    )
    return frame.with_columns(
        pl.when(is_outlier).then(None).otherwise(value).alias(column)
    ).drop(med, dev, mad, count)


def _log_nulled(
    before: pl.DataFrame, after: pl.DataFrame, columns: Sequence[str]
) -> None:
    for column in columns:
        added = after[column].null_count() - before[column].null_count()
        if added > 0:
            _logger.info("provider_qc nulled %d implausible %s value(s)", added, column)


def apply_provider_qc(
    frame: pl.DataFrame,
    config: Config,
    *,
    value_columns: Sequence[str],
    group_key: str | Sequence[str],
) -> pl.DataFrame:
    """Return ``frame`` with implausible provider values nulled per configuration.

    ``value_columns`` are the canonical forecast columns to check; ``group_key``
    identifies one snapshot's set of sources — ``["issue_time", "valid_time"]``
    for hourly and ``["issue_time", "forecast_date"]`` for daily — so the
    cross-source comparison only ever weighs forecasts active at the same
    snapshot, never mixing historical vintages of the same valid time.
    """
    qc = config.provider_qc
    if not qc.enabled or frame.is_empty():
        return frame
    keys = [group_key] if isinstance(group_key, str) else list(group_key)
    present = [column for column in value_columns if column in frame.columns]
    result = frame
    if bound_exprs := [
        _bounded(column, *qc.bounds[column])
        for column in present
        if column in qc.bounds
    ]:
        result = result.with_columns(bound_exprs)
    if all(key in result.columns for key in keys):
        for column in present:
            if column in qc.cross_source_variables:
                result = _mask_cross_source(result, column, keys, qc)
    _log_nulled(frame, result, present)
    return result
