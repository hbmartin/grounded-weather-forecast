from datetime import date, timedelta

import polars as pl
import pytest
from conftest import make_station_db, minute_series, utc, write_config

from omni_forecast.contracts import MixedProvenanceError, SourceKind, hourly_variable
from omni_forecast.dataset.backfill import (
    BACKFILL_VARIABLES,
    MAX_PREVIOUS_DAYS,
    BackfillError,
    backfill_long,
    build_url,
    parse_previous_runs,
)
from omni_forecast.dataset.matrix import (
    assert_single_kind,
    matrix_path,
    to_supervised_slice,
    write_synthetic_matrix,
)

START = utc(2026, 3, 1)
HOURS = 96


def fake_payload(hours=HOURS, previous_days=MAX_PREVIOUS_DAYS):
    """Previous Runs-shaped payload: each day offset shifts the value slightly."""
    times = [(START + timedelta(hours=h)).replace(tzinfo=None) for h in range(hours)]
    hourly: dict[str, object] = {"time": [t.isoformat() for t in times]}
    for variable in BACKFILL_VARIABLES:
        for day in range(1, previous_days + 1):
            # older runs are progressively more wrong: value + day
            hourly[f"{variable}_previous_day{day}"] = [
                float(h % 24) + float(day) for h in range(hours)
            ]
    return {"latitude": 34.28, "longitude": -117.17, "hourly": hourly}


def fetcher_for(payload):
    calls: list[str] = []

    def fetch(url: str):
        calls.append(url)
        return payload

    return fetch, calls


class TestBuildUrl:
    def test_requests_only_previous_day_offsets(self, tmp_path):
        config = write_config(tmp_path)
        url = build_url(config, "gfs_seamless", date(2026, 1, 1), date(2026, 1, 31))
        assert "previous-runs-api.open-meteo.com" in url
        assert "temperature_2m_previous_day1" in url
        assert "temperature_2m_previous_day7" in url
        # the unsuffixed field is an analysis, not a forecast: never requested
        assert "hourly=temperature_2m%2C" not in url
        assert "temperature_2m%2Ctemperature_2m_previous" not in url
        assert "models=gfs_seamless" in url
        assert "wind_speed_unit=ms" in url
        assert "start_date=2026-01-01" in url


class TestParsePreviousRuns:
    def test_leads_are_exact_day_multiples(self):
        frame = parse_previous_runs(fake_payload(), "gfs_seamless")
        leads = sorted(frame["lead_hours"].unique().to_list())
        assert leads == [24.0 * d for d in range(1, MAX_PREVIOUS_DAYS + 1)]
        assert frame["source"].unique().to_list() == ["open_meteo_gfs_seamless"]
        assert frame["source_kind"].unique().to_list() == [SourceKind.SYNTHETIC.value]

    def test_issue_time_is_valid_minus_offset(self):
        frame = parse_previous_runs(fake_payload(), "gfs_seamless")
        row = frame.filter(
            (pl.col("valid_time") == START) & (pl.col("lead_hours") == 48.0)
        ).row(0, named=True)
        assert row["fetched_at"] == START - timedelta(days=2)
        assert row["temp_c"] == 2.0  # value + day offset from the fixture

    def test_canonical_columns_present(self):
        frame = parse_previous_runs(fake_payload(), "gfs_seamless")
        for canonical in ("temp_c", "wind_speed_ms", "precip_mm", "pop"):
            assert canonical in frame.columns
        # Previous Runs has no probability field: pop is present but all-null
        assert frame["pop"].null_count() == frame.height

    def test_missing_hourly_block_raises(self):
        with pytest.raises(BackfillError, match=r"no hourly\.time"):
            parse_previous_runs({}, "gfs_seamless")

    def test_no_offsets_raises(self):
        payload = {"hourly": {"time": ["2026-03-01T00:00"]}}
        with pytest.raises(BackfillError, match="no usable day offsets"):
            parse_previous_runs(payload, "gfs_seamless")


class TestBackfillLong:
    def test_fetches_each_model_and_chunk(self, tmp_path):
        config = write_config(
            tmp_path,
            extra_toml=(
                "\n[backfill.open_meteo]\n"
                'models = ["gfs_seamless", "ecmwf_ifs025"]\n'
                "start_date = 2026-01-01\n"
            ),
        )
        fetch, calls = fetcher_for(fake_payload())
        frame = backfill_long(config, date(2026, 3, 1), fetcher=fetch, chunk_days=30)
        assert len(calls) == 2 * 2  # 2 models x 2 chunks over 60 days
        assert set(frame["source"].unique()) == {
            "open_meteo_gfs_seamless",
            "open_meteo_ecmwf_ifs025",
        }
        assert assert_single_kind(frame) == SourceKind.SYNTHETIC.value

    def test_requires_start_date(self, tmp_path):
        config = write_config(
            tmp_path,
            extra_toml='\n[backfill.open_meteo]\nmodels = ["gfs_seamless"]\n',
        )
        fetch, _ = fetcher_for(fake_payload())
        with pytest.raises(BackfillError, match="start_date"):
            backfill_long(config, date(2026, 3, 1), fetcher=fetch)

    def test_requires_models(self, tmp_path):
        config = write_config(
            tmp_path,
            extra_toml="\n[backfill.open_meteo]\nmodels = []\nstart_date = 2026-01-01\n",
        )
        fetch, _ = fetcher_for(fake_payload())
        with pytest.raises(BackfillError, match="models"):
            backfill_long(config, date(2026, 3, 1), fetcher=fetch)

    def test_rejects_nonpositive_chunk_size(self, tmp_path):
        config = write_config(
            tmp_path,
            extra_toml=(
                "\n[backfill.open_meteo]\n"
                'models = ["gfs_seamless"]\n'
                "start_date = 2026-01-01\n"
            ),
        )
        fetch, _ = fetcher_for(fake_payload())
        with pytest.raises(BackfillError, match="positive integer"):
            backfill_long(config, date(2026, 3, 1), fetcher=fetch, chunk_days=0)


class TestSyntheticMatrix:
    def test_matrix_is_tagged_and_scored(self, tmp_path):
        config = write_config(tmp_path, min_hour_coverage=0.05)
        # station truth covering the fixture's valid hours
        rows = [
            (ts.strftime("%Y-%m-%d %H:%M:%S"), {"outTemp": 50.0 + (ts.hour % 24)})
            for ts in minute_series(START - timedelta(minutes=5), HOURS * 12, 300)
        ]
        make_station_db(tmp_path / "station.db", rows)

        long_frame = parse_previous_runs(fake_payload(), "gfs_seamless")
        path, count = write_synthetic_matrix(config, long_frame)
        assert path == matrix_path(config.dataset.dir, "hourly", "synthetic")
        assert path.exists()
        assert count > 0
        assert matrix_path(config.dataset.dir, "daily", "synthetic").exists()

        matrix = pl.read_parquet(path)
        assert matrix["source_kind"].unique().to_list() == ["synthetic"]
        # 24h-quantized leads only populate buckets at and beyond 24h
        buckets = set(matrix["lead_bucket"].unique().to_list())
        assert buckets <= {"24-48h", "48-96h", "96-168h", "168-240h"}
        assert "0-1h" not in buckets

        scored = to_supervised_slice(matrix, hourly_variable("temp_c"))
        assert scored.source_kind is SourceKind.SYNTHETIC
        assert scored.x.n_rows > 0

    def test_never_pools_with_live(self, tmp_path):
        config = write_config(tmp_path, min_hour_coverage=0.05)
        make_station_db(
            tmp_path / "station.db",
            [(START.strftime("%Y-%m-%d %H:%M:%S"), {"outTemp": 50.0})],
        )
        long_frame = parse_previous_runs(fake_payload(), "gfs_seamless")
        path, _ = write_synthetic_matrix(config, long_frame)
        synthetic = pl.read_parquet(path)
        live = synthetic.with_columns(pl.lit("live").alias("source_kind"))
        with pytest.raises(MixedProvenanceError):
            assert_single_kind(pl.concat([synthetic, live]))
