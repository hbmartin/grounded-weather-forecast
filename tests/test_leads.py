import itertools
import math

import polars as pl
from hypothesis import given
from hypothesis import strategies as st

from grounded_weather_forecast.leads import (
    DAILY_BUCKET_LABELS,
    HOURLY_BUCKET_LABELS,
    HOURLY_BUCKETS,
    daily_bucket,
    daily_bucket_expr,
    hourly_bucket,
    hourly_bucket_expr,
)


class TestHourlyBuckets:
    def test_edges(self):
        assert hourly_bucket(0.0) == "0-1h"
        assert hourly_bucket(0.999) == "0-1h"
        assert hourly_bucket(1.0) == "1-3h"
        assert hourly_bucket(23.999) == "12-24h"
        assert hourly_bucket(24.0) == "24-48h"
        assert hourly_bucket(168.0) == "168-240h"
        assert hourly_bucket(239.999) == "168-240h"
        assert hourly_bucket(240.0) == "240h+"
        assert hourly_bucket(360.0) == "240h+"

    def test_negative_lead_is_none(self):
        assert hourly_bucket(-0.5) is None

    def test_buckets_are_contiguous(self):
        for earlier, later in itertools.pairwise(HOURLY_BUCKETS):
            assert earlier.hi == later.lo
        assert HOURLY_BUCKETS[0].lo == 0.0
        assert math.isinf(HOURLY_BUCKETS[-1].hi)

    @given(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False))
    def test_every_nonnegative_lead_has_a_bucket(self, lead):
        assert hourly_bucket(lead) in HOURLY_BUCKET_LABELS


class TestDailyBuckets:
    def test_edges(self):
        assert daily_bucket(0.0) == "D1"
        assert daily_bucket(1.0) == "D1"
        assert daily_bucket(2.0) == "D2"
        assert daily_bucket(3.0) == "D3-4"
        assert daily_bucket(4.0) == "D3-4"
        assert daily_bucket(5.0) == "D5-7"
        assert daily_bucket(7.0) == "D5-7"
        assert daily_bucket(8.0) == "D8-10"
        assert daily_bucket(10.0) == "D8-10"

    def test_out_of_range_is_none(self):
        assert daily_bucket(-1.0) is None
        assert daily_bucket(11.0) is None

    def test_labels(self):
        assert DAILY_BUCKET_LABELS == ("D1", "D2", "D3-4", "D5-7", "D8-10")


class TestExpressions:
    @given(
        st.lists(
            st.floats(min_value=-10.0, max_value=400.0, allow_nan=False),
            min_size=1,
            max_size=50,
        )
    )
    def test_hourly_expr_matches_scalar(self, leads):
        frame = pl.DataFrame({"lead": leads})
        got = frame.select(hourly_bucket_expr(pl.col("lead")).alias("b"))["b"]
        assert got.to_list() == [hourly_bucket(lead) for lead in leads]

    @given(
        st.lists(
            st.floats(min_value=-5.0, max_value=15.0, allow_nan=False),
            min_size=1,
            max_size=50,
        )
    )
    def test_daily_expr_matches_scalar(self, leads):
        frame = pl.DataFrame({"lead": leads})
        got = frame.select(daily_bucket_expr(pl.col("lead")).alias("b"))["b"]
        assert got.to_list() == [daily_bucket(lead) for lead in leads]
