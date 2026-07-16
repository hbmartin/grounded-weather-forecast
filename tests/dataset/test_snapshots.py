import polars as pl
from conftest import utc

from grounded_weather_forecast.dataset.snapshots import (
    as_of_selection,
    snapshot_long,
    snapshot_times,
)


def completions(*times):
    return pl.DataFrame(
        {"completed_at": list(times)},
        schema={"completed_at": pl.Datetime("us", "UTC")},
    )


def long_frame(rows):
    return pl.DataFrame(
        rows,
        schema={
            "source": pl.String,
            "fetched_at": pl.Datetime("us", "UTC"),
            "valid_time": pl.Datetime("us", "UTC"),
            "temp_c": pl.Float64,
        },
        orient="row",
    )


class TestSnapshotTimes:
    def test_dedupes_to_grid_keeping_latest(self):
        snaps = snapshot_times(
            completions(
                utc(2026, 3, 22, 12, 0, 30),
                utc(2026, 3, 22, 12, 8, 0),  # same 10-min bucket
                utc(2026, 3, 22, 18, 1, 0),
            )
        )
        assert snaps["issue_time"].to_list() == [
            utc(2026, 3, 22, 12, 8),
            utc(2026, 3, 22, 18, 1),
        ]

    def test_empty(self):
        assert snapshot_times(completions()).is_empty()


class TestAsOfSelection:
    def test_latest_within_age_cap(self):
        long = long_frame(
            [
                ("nws", utc(2026, 3, 22, 6), utc(2026, 3, 22, 18), 1.0),
                ("nws", utc(2026, 3, 22, 11), utc(2026, 3, 22, 18), 2.0),
                ("slow", utc(2026, 3, 21, 22), utc(2026, 3, 22, 18), 3.0),
            ]
        )
        snaps = snapshot_times(completions(utc(2026, 3, 22, 12)))
        selection = as_of_selection(long, snaps, max_age_hours=12.0)
        chosen = {r["source"]: r["fetched_at"] for r in selection.to_dicts()}
        assert chosen["nws"] == utc(2026, 3, 22, 11)  # latest, not first
        assert "slow" not in chosen  # 14h old > 12h cap

    def test_future_fetches_invisible(self):
        long = long_frame([("nws", utc(2026, 3, 22, 13), utc(2026, 3, 22, 18), 1.0)])
        snaps = snapshot_times(completions(utc(2026, 3, 22, 12)))
        assert as_of_selection(long, snaps, 12.0).is_empty()

    def test_snapshot_long_joins_rows(self):
        long = long_frame(
            [
                ("nws", utc(2026, 3, 22, 11), utc(2026, 3, 22, 18), 2.0),
                ("nws", utc(2026, 3, 22, 11), utc(2026, 3, 22, 19), 2.5),
                ("nws", utc(2026, 3, 22, 6), utc(2026, 3, 22, 18), 1.0),
            ]
        )
        snaps = snapshot_times(completions(utc(2026, 3, 22, 12)))
        snap = snapshot_long(long, snaps, 12.0)
        assert snap.height == 2  # both valid times of the chosen fetch
        assert set(snap["temp_c"].to_list()) == {2.0, 2.5}
        assert snap["issue_time"].unique().to_list() == [utc(2026, 3, 22, 12)]

    def test_empty_inputs(self):
        empty_snapshot = snapshot_times(completions())
        assert as_of_selection(long_frame([]), empty_snapshot, 12.0).is_empty()
