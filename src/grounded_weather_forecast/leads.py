"""Lead buckets: the single source of truth for fit/eval stratification.

Hourly edges sit on provider raggedness cliffs (24/48/168/240 h) with quasi-log
spacing matching forecast error growth; 0-6 h isolates the anchoring regime.
Buckets stratify fitting and evaluation only — ``lead_hours`` stays continuous
for methods that want it.
"""

import math
from dataclasses import dataclass

import polars as pl

from grounded_weather_forecast.contracts import Product


@dataclass(frozen=True, slots=True)
class LeadBucket:
    """Left-closed interval ``[lo, hi)`` of lead values."""

    label: str
    lo: float
    hi: float

    def contains(self, lead: float) -> bool:
        return self.lo <= lead < self.hi


HOURLY_BUCKETS: tuple[LeadBucket, ...] = (
    LeadBucket("0-1h", 0.0, 1.0),
    LeadBucket("1-3h", 1.0, 3.0),
    LeadBucket("3-6h", 3.0, 6.0),
    LeadBucket("6-12h", 6.0, 12.0),
    LeadBucket("12-24h", 12.0, 24.0),
    LeadBucket("24-48h", 24.0, 48.0),
    LeadBucket("48-96h", 48.0, 96.0),
    LeadBucket("96-168h", 96.0, 168.0),
    LeadBucket("168-240h", 168.0, 240.0),
    LeadBucket("240h+", 240.0, math.inf),
)

DAILY_BUCKETS: tuple[LeadBucket, ...] = (
    LeadBucket("D1", 0.0, 2.0),
    LeadBucket("D2", 2.0, 3.0),
    LeadBucket("D3-4", 3.0, 5.0),
    LeadBucket("D5-7", 5.0, 8.0),
    LeadBucket("D8-10", 8.0, 11.0),
)

DAILY_BUCKETS_HOURS: tuple[LeadBucket, ...] = tuple(
    LeadBucket(bucket.label, bucket.lo * 24.0, bucket.hi * 24.0)
    for bucket in DAILY_BUCKETS
)

HOURLY_BUCKET_LABELS: tuple[str, ...] = tuple(b.label for b in HOURLY_BUCKETS)
DAILY_BUCKET_LABELS: tuple[str, ...] = tuple(b.label for b in DAILY_BUCKETS)


def _bucket_label(buckets: tuple[LeadBucket, ...], lead: float) -> str | None:
    if lead < 0:
        return None
    for bucket in buckets:
        if bucket.contains(lead):
            return bucket.label
    return None


def hourly_bucket(lead_hours: float) -> str | None:
    """Bucket label for a lead in hours; ``None`` for negative leads."""
    return _bucket_label(HOURLY_BUCKETS, lead_hours)


def daily_bucket(lead_days: float) -> str | None:
    """Bucket label for a lead in local calendar days; ``None`` if out of range.

    ``lead_days`` counts local-date difference: 0 or 1 is D1 (today/tomorrow
    from the product's perspective), 10 is the last product day.
    """
    return _bucket_label(DAILY_BUCKETS, lead_days)


def buckets_for_product(product: Product) -> tuple[LeadBucket, ...]:
    """Fit/evaluation buckets expressed in the matrix's hour lead unit."""
    return DAILY_BUCKETS_HOURS if product is Product.DAILY else HOURLY_BUCKETS


def bucket_for_product(product: Product, lead_hours: float) -> str | None:
    return _bucket_label(buckets_for_product(product), lead_hours)


def _bucket_expr(buckets: tuple[LeadBucket, ...], lead: pl.Expr) -> pl.Expr:
    expr: pl.Expr = pl.lit(None, dtype=pl.String)
    for bucket in reversed(buckets):
        expr = (
            pl.when((lead >= bucket.lo) & (lead < bucket.hi))
            .then(pl.lit(bucket.label))
            .otherwise(expr)
        )
    return expr


def hourly_bucket_expr(lead_hours: pl.Expr) -> pl.Expr:
    """Vectorized :func:`hourly_bucket` as a polars expression."""
    return _bucket_expr(HOURLY_BUCKETS, lead_hours)


def daily_bucket_expr(lead_days: pl.Expr) -> pl.Expr:
    """Vectorized :func:`daily_bucket` as a polars expression."""
    return _bucket_expr(DAILY_BUCKETS, lead_days)
