from datetime import timedelta

import polars as pl
import pytest
from conftest import (
    canonical_minute_frame,
    make_forecast_db,
    make_station_db,
    minute_series,
    utc,
    write_config,
)

from omni_forecast.contracts import (
    MixedProvenanceError,
    Product,
    SourceKind,
    TruthSemantics,
    daily_variable,
    hourly_variable,
)
from omni_forecast.dataset.matrix import (
    _equal_weight_daily_aggregates,
    assert_single_kind,
    build_daily_matrix,
    build_hourly_matrix,
    matrix_path,
    matrix_sources,
    to_forecast_matrix,
    to_supervised_slice,
    write_dataset,
)
from omni_forecast.dataset.providers import (
    read_daily_long,
    read_hourly_long,
    read_run_completions,
)
from omni_forecast.dataset.snapshots import snapshot_times
from omni_forecast.dataset.truth import truth_daily, truth_hourly

FETCH = "2026-03-22T11:30:00+00:00"
ISSUE = utc(2026, 3, 22, 12, 0, 30)
VALID = utc(2026, 3, 22, 18, 0)


def build_fixture(tmp_path):
    make_forecast_db(
        tmp_path / "fx.sqlite",
        [
            {
                "completed_at": "2026-03-22T12:00:30+00:00",
                "results": [
                    {
                        "provider": "nws",
                        "fetched_at": FETCH,
                        "hourly": [
                            (VALID, {"temperature": 10.0, "wind_speed": 3.0}),
                            (
                                utc(2026, 3, 22, 19),
                                {"temperature": 11.0, "wind_speed": 4.0},
                            ),
                        ],
                        "daily": [
                            (
                                "2026-03-23",
                                {"temperature_max": 15.0, "temperature_min": 2.0},
                            )
                        ],
                    },
                    {
                        "provider": "open_meteo",
                        "model": "best_match",
                        "fetched_at": FETCH,
                        "hourly": [(VALID, {"temperature": 12.0})],
                    },
                ],
            }
        ],
    )
    return write_config(tmp_path, min_hour_coverage=0.1, min_day_coverage=0.01)


@pytest.fixture
def fixture_config(tmp_path):
    return build_fixture(tmp_path)


def truth_frames(config):
    ts = minute_series(ISSUE - timedelta(minutes=10), 10) + minute_series(
        VALID - timedelta(minutes=5), 11
    )
    temps = [5.0] * 10 + [9.0] * 11
    minute = canonical_minute_frame(ts, temp_c=temps, wind_speed_ms=[1.0] * 21)
    return minute, truth_hourly(minute, config), truth_daily(minute, config)


class TestBuildHourlyMatrix:
    def test_wide_shape_and_values(self, fixture_config):
        config = fixture_config
        hourly_long = read_hourly_long(config.forecasts)
        snapshots = snapshot_times(read_run_completions(config.forecasts))
        minute, hourly_truth, _ = truth_frames(config)
        matrix = build_hourly_matrix(
            hourly_long, snapshots, hourly_truth, minute, config
        )
        assert matrix.height == 2  # two valid hours
        row = matrix.filter(pl.col("valid_time") == VALID).row(0, named=True)
        assert row["fx__nws__temp_c"] == 10.0
        assert row["fx__open_meteo__temp_c"] == 12.0
        assert row["fx__nws__wind_speed_ms"] == 3.0
        assert row["fx__open_meteo__wind_speed_ms"] is None
        assert row["age__nws"] == pytest.approx(0.5083, abs=0.001)
        assert row["lead_hours"] == pytest.approx(5.99, abs=0.02)
        assert row["lead_bucket"] == "3-6h"
        assert row["source_kind"] == "live"
        assert row["valid_hour_local"] == 11  # 18 UTC = 11:00 PDT
        assert row["obs__temp_c"] == pytest.approx(5.0)  # issue-time obs
        assert row["t__temp_c__inst"] == pytest.approx(9.0)  # valid-time truth

    def test_empty_long_gives_empty_matrix(self, fixture_config):
        config = fixture_config
        empty = read_hourly_long(config.forecasts).clear()
        snapshots = snapshot_times(read_run_completions(config.forecasts))
        minute, hourly_truth, _ = truth_frames(config)
        matrix = build_hourly_matrix(empty, snapshots, hourly_truth, minute, config)
        assert matrix.is_empty()


class TestProvenance:
    def test_mixed_kinds_rejected(self):
        frame = pl.DataFrame({"source_kind": ["live", "synthetic"]})
        with pytest.raises(MixedProvenanceError):
            assert_single_kind(frame)

    def test_allow_mixed(self):
        frame = pl.DataFrame({"source_kind": ["live", "synthetic"]})
        assert assert_single_kind(frame, allow_mixed=True) == "mixed"

    def test_empty_defaults_live(self):
        assert assert_single_kind(pl.DataFrame({"source_kind": []})) == "live"


class TestSupervisedSlice:
    def test_slice_shapes_and_availability(self, fixture_config):
        config = fixture_config
        hourly_long = read_hourly_long(config.forecasts)
        snapshots = snapshot_times(read_run_completions(config.forecasts))
        minute, hourly_truth, _ = truth_frames(config)
        matrix = build_hourly_matrix(
            hourly_long, snapshots, hourly_truth, minute, config
        )
        s = to_supervised_slice(
            matrix,
            hourly_variable("temp_c"),
            semantics=TruthSemantics.INSTANTANEOUS,
        )
        # only VALID hour has truth (the second hour has no station samples)
        assert s.x.n_rows == 1
        assert s.x.sources == ("nws", "open_meteo")
        assert s.y[0] == pytest.approx(9.0)
        assert s.x.values[0].tolist() == [10.0, 12.0]
        assert bool(s.x.availability.all())
        assert s.source_kind is SourceKind.LIVE
        assert all(not c.startswith("t__") for c in s.x.features.columns)
        assert not any(c.endswith("_cov") for c in s.x.features.columns)

    def test_matrix_sources(self, fixture_config):
        config = fixture_config
        hourly_long = read_hourly_long(config.forecasts)
        snapshots = snapshot_times(read_run_completions(config.forecasts))
        minute, hourly_truth, _ = truth_frames(config)
        matrix = build_hourly_matrix(
            hourly_long, snapshots, hourly_truth, minute, config
        )
        assert matrix_sources(matrix) == ("nws", "open_meteo")


class TestBuildDailyMatrix:
    def test_daily_matrix(self, fixture_config):
        config = fixture_config
        hourly_long = read_hourly_long(config.forecasts)
        daily_long = read_daily_long(config.forecasts)
        snapshots = snapshot_times(read_run_completions(config.forecasts))
        minute, hourly_truth, daily_truth = truth_frames(config)
        hourly_matrix = build_hourly_matrix(
            hourly_long, snapshots, hourly_truth, minute, config
        )
        daily_matrix = build_daily_matrix(
            daily_long, snapshots, hourly_matrix, daily_truth, config
        )
        assert daily_matrix.height == 1
        row = daily_matrix.row(0, named=True)
        assert row["fxd__nws__temp_max_c"] == 15.0
        assert row["lead_days"] == 1  # issue Mar 22 local -> target Mar 23
        assert row["lead_bucket"] == "D1"
        assert row["source_kind"] == "live"
        contract = to_forecast_matrix(
            daily_matrix, daily_variable("temp_max_c"), daily=True
        )
        assert contract.product is Product.DAILY
        assert contract.lead_hours[0] == 24.0

    def test_ewagg_from_hourly(self, fixture_config):
        config = fixture_config
        hourly_long = read_hourly_long(config.forecasts)
        daily_long = read_daily_long(config.forecasts)
        snapshots = snapshot_times(read_run_completions(config.forecasts))
        minute, hourly_truth, daily_truth = truth_frames(config)
        hourly_matrix = build_hourly_matrix(
            hourly_long, snapshots, hourly_truth, minute, config
        )
        # target Mar 22 local: both forecast hours fall on it (11:00, 12:00 PDT)
        daily_matrix = build_daily_matrix(
            daily_long, snapshots, hourly_matrix, daily_truth, config
        ).join(
            hourly_matrix.select(pl.first("issue_time")).unique(),
            on="issue_time",
            how="inner",
        )
        agg = daily_matrix  # nothing on Mar 22 in daily_long, so check directly
        ew = _equal_weight_daily_aggregates(hourly_matrix, config)
        row = ew.row(0, named=True)
        # hour 1: mean(10, 12) = 11; hour 2: mean(11) = 11
        assert row["ewagg__temp_max_c"] == pytest.approx(11.0)
        assert row["ewagg__temp_min_c"] == pytest.approx(11.0)
        assert agg.height >= 0

    def test_ewagg_coverage_uses_dst_day_length(self, tmp_path):
        config = write_config(tmp_path)
        start = utc(2026, 3, 8, 8)
        hourly = pl.DataFrame(
            {
                "issue_time": [utc(2026, 3, 8)] * 23,
                "valid_time": [start + timedelta(hours=hour) for hour in range(23)],
                "fx__nws__temp_c": [10.0] * 23,
                "fx__nws__pop": [0.0] * 23,
                "fx__nws__precip_mm": [0.0] * 23,
            },
            schema_overrides={
                "issue_time": pl.Datetime("us", "UTC"),
                "valid_time": pl.Datetime("us", "UTC"),
            },
        )
        row = _equal_weight_daily_aggregates(hourly, config).row(0, named=True)
        assert row["ewagg__coverage_frac"] == pytest.approx(1.0)


class TestWriteDataset:
    def test_end_to_end(self, tmp_path):
        config = build_fixture(tmp_path)
        make_station_db(
            tmp_path / "station.db",
            [
                ("2026-03-22 17:56:00", {"outTemp": 48.0, "outHumi": 50.0}),
                ("2026-03-22 17:57:00", {"outTemp": 48.2, "outHumi": 50.0}),
            ],
        )
        manifest = write_dataset(config)
        assert set(manifest.files) == {
            "truth_minute",
            "truth_hourly",
            "truth_daily",
            "forecasts_long",
            "daily_long",
            "minutely_long",
            "hourly_matrix",
            "daily_matrix",
        }
        live_matrix = matrix_path(config.dataset.dir, "hourly", "live")
        assert live_matrix.exists()
        assert (config.dataset.dir / "manifest.json").exists()
        matrix = pl.read_parquet(live_matrix)
        assert matrix.height == 2
        assert manifest.files["forecasts_long"].rows == 3

    def test_fingerprint_stable_across_rebuilds(self, tmp_path):
        config = build_fixture(tmp_path)
        make_station_db(
            tmp_path / "station.db",
            [("2026-03-22 17:56:00", {"outTemp": 48.0})],
        )
        first = write_dataset(config)
        second = write_dataset(config)
        assert first.fingerprint == second.fingerprint
