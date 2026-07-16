"""As-of snapshot construction.

A snapshot is the view of all sources at one issue time: for each source, its
latest forecast fetched at or before that instant and not older than the
staleness cap. Snapshot times anchor to ``forecast_runs.completed_at`` deduped
onto a 10-minute grid — NOT a uniform grid — so nonuniform polling cadences
are respected and a stale 6-hourly forecast is never duplicated across
snapshots it did not refresh.
"""

from datetime import timedelta

import polars as pl

_SNAPSHOT_GRID = "10m"


def snapshot_times(completions: pl.DataFrame) -> pl.DataFrame:
    """Dedupe run-completion instants onto the snapshot grid (keep latest)."""
    if completions.is_empty():
        return pl.DataFrame(schema={"issue_time": pl.Datetime("us", "UTC")})
    return (
        completions.sort("completed_at")
        .group_by(pl.col("completed_at").dt.truncate(_SNAPSHOT_GRID).alias("bucket"))
        .agg(pl.col("completed_at").max().alias("issue_time"))
        .select("issue_time")
        .sort("issue_time")
    )


def as_of_selection(
    long_frame: pl.DataFrame,
    snapshots: pl.DataFrame,
    max_age_hours: float,
) -> pl.DataFrame:
    """Pick, per (issue_time, source), the fetched_at visible at that snapshot.

    Returns columns ``issue_time, source, fetched_at`` (rows only where a
    fresh-enough forecast exists).
    """
    if long_frame.is_empty() or snapshots.is_empty():
        return pl.DataFrame(
            schema={
                "issue_time": pl.Datetime("us", "UTC"),
                "source": pl.String(),
                "fetched_at": pl.Datetime("us", "UTC"),
            }
        )
    issues = long_frame.select("source", "fetched_at").unique().sort("fetched_at")
    grid = snapshots.join(issues.select("source").unique(), how="cross").sort(
        "issue_time"
    )
    return (
        grid.join_asof(
            issues,
            left_on="issue_time",
            right_on="fetched_at",
            by="source",
            strategy="backward",
            tolerance=timedelta(hours=max_age_hours),
            # both frames are pre-sorted; polars cannot verify with `by` groups
            check_sortedness=False,
        )
        .drop_nulls("fetched_at")
        .select("issue_time", "source", "fetched_at")
    )


def snapshot_long(
    long_frame: pl.DataFrame,
    snapshots: pl.DataFrame,
    max_age_hours: float,
) -> pl.DataFrame:
    """Long frame replicated per snapshot: as-of selection joined back to rows."""
    selection = as_of_selection(long_frame, snapshots, max_age_hours)
    return selection.join(long_frame, on=["source", "fetched_at"], how="inner")
