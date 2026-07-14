import polars as pl
from conftest import minute_series, utc

from omni_forecast.config import load_config
from omni_forecast.dataset.qc import (
    QC_FLATLINE,
    QC_OUT_OF_BOUNDS,
    QC_SPIKE,
    apply_causal_qc,
    apply_qc,
    masked,
    qc_summary,
)


def qc_config(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(
        """
[station]
db_path = "x.db"
latitude = 0.0
longitude = 0.0
[forecasts]
db_path = "y.sqlite"
""",
        encoding="utf-8",
    )
    return load_config(path).qc


def frame(values, step_seconds=60, start=None):
    ts = minute_series(start or utc(2026, 7, 13, 12, 0), len(values), step_seconds)
    return pl.DataFrame(
        {"ts": ts, "temp": values},
        schema={"ts": pl.Datetime("us", "UTC"), "temp": pl.Float64},
    )


class TestBounds:
    def test_out_of_bounds_flagged(self, tmp_path):
        qc = qc_config(tmp_path)
        flagged = apply_qc(frame([20.0, 99.0, -50.0, 21.0]), qc, ["temp"])
        oob = [(f & QC_OUT_OF_BOUNDS) > 0 for f in flagged["temp_qc"].to_list()]
        assert oob == [False, True, True, False]

    def test_null_not_flagged(self, tmp_path):
        qc = qc_config(tmp_path)
        flagged = apply_qc(frame([20.0, None, 21.0]), qc, ["temp"])
        assert flagged["temp_qc"].to_list() == [0, 0, 0]


class TestSpike:
    def test_isolated_spike_flagged(self, tmp_path):
        qc = qc_config(tmp_path)  # temp max_step = 5.0 per minute
        flagged = apply_qc(frame([20.0, 20.1, 40.0, 20.2, 20.3]), qc, ["temp"])
        assert flagged["temp_qc"].to_list() == [0, 0, QC_SPIKE, 0, 0]

    def test_fast_monotone_ramp_not_flagged(self, tmp_path):
        qc = qc_config(tmp_path)
        flagged = apply_qc(frame([20.0, 30.0, 40.0, 50.0]), qc, ["temp"])
        assert flagged["temp_qc"].to_list() == [0, 0, 0, 0]

    def test_gap_scales_allowance(self, tmp_path):
        qc = qc_config(tmp_path)
        # 10-minute gaps: a 35-degree excursion is within 5 deg/min * 10 min,
        # while the same excursion at 1-minute cadence is a spike.
        gapped = apply_qc(frame([10.0, 45.0, 10.0], step_seconds=600), qc, ["temp"])
        assert gapped["temp_qc"].to_list() == [0, 0, 0]
        rapid = apply_qc(frame([10.0, 45.0, 10.0]), qc, ["temp"])
        assert rapid["temp_qc"].to_list() == [0, QC_SPIKE, 0]


class TestFlatline:
    def test_long_identical_run_flagged(self, tmp_path):
        qc = qc_config(tmp_path)  # temp flatline at 180 minutes
        values = [15.0] * 200 + [16.0]
        flagged = apply_qc(frame(values), qc, ["temp"])
        assert flagged["temp_qc"][0] == QC_FLATLINE
        assert flagged["temp_qc"][199] == QC_FLATLINE
        assert flagged["temp_qc"][200] == 0

    def test_short_run_not_flagged(self, tmp_path):
        qc = qc_config(tmp_path)
        flagged = apply_qc(frame([15.0] * 60 + [16.0]), qc, ["temp"])
        assert flagged["temp_qc"].sum() == 0

    def test_nulls_break_runs(self, tmp_path):
        qc = qc_config(tmp_path)
        values = [15.0] * 100 + [None] + [15.0] * 100
        flagged = apply_qc(frame(values), qc, ["temp"])
        assert flagged["temp_qc"].sum() == 0

    def test_gap_breaks_flatline_run(self, tmp_path):
        qc = qc_config(tmp_path)
        first = frame([15.0] * 100)
        second = frame(
            [15.0] * 100,
            start=utc(2026, 7, 13, 16, 0),
        )
        flagged = apply_qc(pl.concat([first, second]), qc, ["temp"])
        assert flagged["temp_qc"].sum() == 0

    def test_causal_flags_do_not_change_when_future_rows_arrive(self, tmp_path):
        qc = qc_config(tmp_path)
        prefix = frame([10.0, 10.0, 10.0])
        future = frame([50.0], start=utc(2026, 7, 13, 12, 3))
        before = apply_causal_qc(prefix, qc, ["temp"])["temp_qc"]
        after = apply_causal_qc(pl.concat([prefix, future]), qc, ["temp"])[
            "temp_qc"
        ].head(3)
        assert before.to_list() == after.to_list()


class TestMaskedAndSummary:
    def test_masked_nulls_flagged_values(self, tmp_path):
        qc = qc_config(tmp_path)
        flagged = apply_qc(frame([20.0, 99.0, 21.0]), qc, ["temp"])
        clean = flagged.select(masked("temp").alias("temp"))
        assert clean["temp"].to_list() == [20.0, None, 21.0]

    def test_summary_counts(self, tmp_path):
        qc = qc_config(tmp_path)
        flagged = apply_qc(frame([20.0, 99.0, None, 21.0]), qc, ["temp"])
        summary = qc_summary(flagged, ["temp"])
        row = summary.row(0, named=True)
        assert row["samples"] == 4
        assert row["missing"] == 1
        assert row["out_of_bounds"] == 1
        assert row["clean"] == 2

    def test_absent_channel_skipped(self, tmp_path):
        qc = qc_config(tmp_path)
        flagged = apply_qc(frame([20.0]), qc, ["temp", "nonexistent"])
        assert "nonexistent_qc" not in flagged.columns
        assert qc_summary(flagged, ["temp", "nonexistent"]).height == 1
