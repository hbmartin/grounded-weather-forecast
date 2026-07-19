from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from datetime import datetime, timedelta
from threading import Barrier

import numpy as np
import polars as pl
import pytest
from conftest import (
    canonical_minute_frame,
    make_forecast_db,
    minute_series,
    write_config,
)

from grounded_weather_forecast.contracts import hourly_variable
from grounded_weather_forecast.config import EnsemblesConfig
from grounded_weather_forecast.dataset.ensembles import (
    EnsembleError,
    append_ensembles,
    build_ensemble_url,
    ensemble_features,
    ingest_ensembles,
    parse_ensemble,
)
from grounded_weather_forecast.dataset.matrix import (
    active_ensembles,
    build_hourly_matrix,
    to_supervised_slice,
)
from grounded_weather_forecast.dataset.providers import read_hourly_long
from grounded_weather_forecast.dataset.truth import truth_hourly
from grounded_weather_forecast.timeutil import utc

FETCHED = utc(2026, 3, 22, 11, 30)
ISSUE = utc(2026, 3, 22, 12, 0, 30)
VALID = utc(2026, 3, 22, 18, 0)
VARIABLES = ("temp_c", "wind_speed_ms")


def payload(times=("2026-03-22T18:00", "2026-03-22T19:00")):
    n = len(times)
    return {
        "hourly": {
            "time": list(times),
            "temperature_2m": [10.0] * n,
            "temperature_2m_member01": [12.0] * n,
            "temperature_2m_member02": [14.0] * n,
            "wind_speed_10m": [3.0] * n,
            "wind_speed_10m_member01": [None] * n,
        }
    }


class TestParseEnsemble:
    def test_statistics_over_members(self):
        frame = parse_ensemble(payload(), "gefs", FETCHED, VARIABLES)
        temp = frame.filter(
            (pl.col("variable") == "temp_c") & (pl.col("valid_time") == VALID)
        ).row(0, named=True)
        members = np.array([10.0, 12.0, 14.0])
        assert temp["mean"] == pytest.approx(12.0)
        assert temp["sd"] == pytest.approx(float(members.std(ddof=1)))
        assert temp["p50"] == pytest.approx(12.0)
        assert temp["p10"] == pytest.approx(float(np.quantile(members, 0.1)))
        assert temp["n_members"] == 3

    def test_null_members_excluded(self):
        frame = parse_ensemble(payload(), "gefs", FETCHED, VARIABLES)
        wind = frame.filter(pl.col("variable") == "wind_speed_ms").row(0, named=True)
        assert wind["n_members"] == 1
        assert wind["mean"] == pytest.approx(3.0)
        assert wind["sd"] is None  # a single member has no spread

    def test_missing_hourly_block_raises(self):
        with pytest.raises(EnsembleError, match=r"hourly\.time"):
            parse_ensemble({}, "gefs", FETCHED, VARIABLES)

    def test_no_requested_variables_raises(self):
        with pytest.raises(EnsembleError, match="no requested variables"):
            parse_ensemble(payload(), "gefs", FETCHED, ("pressure_sea_hpa",))


class TestIngestAndAppend:
    def make_config(self, tmp_path):
        return write_config(
            tmp_path,
            extra_toml='[ensembles]\nmodels = ["gefs", "aifs"]\n',
        )

    def test_url_carries_model_and_variables(self, tmp_path):
        config = self.make_config(tmp_path)
        url = build_ensemble_url(config, "gefs")
        assert "models=gefs" in url
        assert "temperature_2m" in url
        assert "wind_speed_unit=ms" in url

    def test_ingest_fetches_every_model(self, tmp_path):
        config = self.make_config(tmp_path)
        seen: list[str] = []

        def fetcher(url):
            seen.append(url)
            return payload()

        frame = ingest_ensembles(config, fetcher=fetcher, now=FETCHED)
        assert len(seen) == 2
        assert sorted(frame["model"].unique().to_list()) == ["aifs", "gefs"]

    def test_each_model_is_stamped_after_its_fetch(self, tmp_path, monkeypatch):
        config = self.make_config(tmp_path)
        stamps = iter(
            (
                utc(2026, 3, 22, 11, 31),
                utc(2026, 3, 22, 11, 32),
            )
        )

        class TickingDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                del tz
                return next(stamps)

        monkeypatch.setattr(
            "grounded_weather_forecast.dataset.ensembles.datetime",
            TickingDateTime,
        )
        frame = ingest_ensembles(config, fetcher=lambda _url: payload())
        fetched = {
            model: group["fetched_at"][0]
            for (model,), group in frame.group_by("model", maintain_order=True)
        }
        assert fetched == {
            "gefs": utc(2026, 3, 22, 11, 31),
            "aifs": utc(2026, 3, 22, 11, 32),
        }

    def test_no_models_raises(self, tmp_path):
        config = write_config(tmp_path)
        with pytest.raises(EnsembleError, match="models"):
            ingest_ensembles(config, fetcher=lambda _url: payload())

    def test_append_is_idempotent(self, tmp_path):
        config = self.make_config(tmp_path)
        frame = ingest_ensembles(config, fetcher=lambda _url: payload(), now=FETCHED)
        store = tmp_path / "data" / "ensembles.parquet"
        first_new, first_total = append_ensembles(store, frame)
        second_new, second_total = append_ensembles(store, frame)
        assert first_new == frame.height
        assert second_new == 0
        assert first_total == second_total == frame.height

    def test_concurrent_appends_do_not_lose_models(self, tmp_path):
        store = tmp_path / "data" / "ensembles.parquet"
        models = tuple(f"model-{index}" for index in range(6))
        frames = tuple(
            parse_ensemble(payload(), model, FETCHED, VARIABLES) for model in models
        )
        barrier = Barrier(len(frames))

        def append(frame):
            barrier.wait()
            return append_ensembles(store, frame)

        with ThreadPoolExecutor(max_workers=len(frames)) as executor:
            tuple(executor.map(append, frames))

        saved = pl.read_parquet(store)
        assert set(saved["model"].unique()) == set(models)
        assert saved.height == sum(frame.height for frame in frames)

    def test_failed_atomic_append_preserves_existing_store(self, tmp_path, monkeypatch):
        store = tmp_path / "data" / "ensembles.parquet"
        first = parse_ensemble(payload(), "first", FETCHED, VARIABLES)
        append_ensembles(store, first)

        def fail_write(_frame, _path):
            raise OSError("disk full")

        monkeypatch.setattr(
            "grounded_weather_forecast.dataset.ensembles.atomic_write_parquet",
            fail_write,
        )
        second = parse_ensemble(payload(), "second", FETCHED, VARIABLES)
        with pytest.raises(OSError, match="disk full"):
            append_ensembles(store, second)

        assert pl.read_parquet(store).equals(first)


def _snapshots(*times):
    return pl.DataFrame(
        {"issue_time": list(times)},
        schema={"issue_time": pl.Datetime("us", "UTC")},
    )


class TestEnsembleFeatures:
    def make_long(self, fetched_at=FETCHED):
        return parse_ensemble(payload(), "gefs", fetched_at, VARIABLES)

    def test_wide_columns_and_values(self):
        wide = ensemble_features(self.make_long(), _snapshots(ISSUE), 12.0)
        assert "ens__gefs__temp_c__mean" in wide.columns
        assert "ens__gefs__temp_c__sd" in wide.columns
        row = wide.filter(pl.col("valid_time") == VALID).row(0, named=True)
        assert row["issue_time"] == ISSUE
        assert row["ens__gefs__temp_c__mean"] == pytest.approx(12.0)

    def test_future_fetch_is_invisible(self):
        late = self.make_long(fetched_at=ISSUE + timedelta(minutes=5))
        wide = ensemble_features(late, _snapshots(ISSUE), 12.0)
        assert wide.is_empty()

    def test_staleness_cap_ages_runs_out(self):
        stale = self.make_long(fetched_at=ISSUE - timedelta(hours=20))
        wide = ensemble_features(stale, _snapshots(ISSUE), 12.0)
        assert wide.is_empty()

    def test_as_of_prefers_latest_visible_run(self):
        early = self.make_long(fetched_at=ISSUE - timedelta(hours=6))
        late = self.make_long(fetched_at=FETCHED).with_columns(
            (pl.col("mean") + 100.0).alias("mean")
        )
        combined = pl.concat([early, late.select(early.columns)])
        wide = ensemble_features(combined, _snapshots(ISSUE), 12.0)
        row = wide.filter(pl.col("valid_time") == VALID).row(0, named=True)
        assert row["ens__gefs__temp_c__mean"] == pytest.approx(112.0)

    def test_dataset_uses_only_currently_configured_rows(self, tmp_path):
        config = write_config(tmp_path)
        store = tmp_path / "data" / "ensembles.parquet"
        active = parse_ensemble(payload(), "gefs", FETCHED, VARIABLES)
        stale = parse_ensemble(payload(), "retired", FETCHED, VARIABLES)
        append_ensembles(store, pl.concat([active, stale]))

        assert active_ensembles(config) is None

        configured = replace(
            config,
            ensembles=EnsemblesConfig(models=("gefs",), variables=("temp_c",)),
        )
        selected = active_ensembles(configured)
        assert selected is not None
        assert set(selected["model"]) == {"gefs"}
        assert set(selected["variable"]) == {"temp_c"}


class TestMatrixIntegration:
    def build(self, tmp_path):
        make_forecast_db(
            tmp_path / "fx.sqlite",
            [
                {
                    "completed_at": "2026-03-22T12:00:30+00:00",
                    "results": [
                        {
                            "provider": "nws",
                            "fetched_at": "2026-03-22T11:30:00+00:00",
                            "hourly": [
                                (VALID, {"temperature": 10.0, "wind_speed": 3.0})
                            ],
                        }
                    ],
                }
            ],
        )
        return write_config(tmp_path, min_hour_coverage=0.1)

    def test_matrix_carries_leakage_safe_ens_features(self, tmp_path):
        config = self.build(tmp_path)
        ts = minute_series(VALID - timedelta(minutes=5), 11)
        minute = canonical_minute_frame(ts, temp_c=[9.0] * 11)
        hourly_truth = truth_hourly(minute, config)
        matrix = build_hourly_matrix(
            read_hourly_long(config.forecasts),
            _snapshots(ISSUE),
            hourly_truth,
            minute,
            config,
            ensembles=parse_ensemble(payload(), "gefs", FETCHED, VARIABLES),
        )
        assert "ens__gefs__temp_c__mean" in matrix.columns
        slice_ = to_supervised_slice(matrix, hourly_variable("temp_c"))
        assert "ens__gefs__temp_c__mean" in slice_.x.features.columns


class TestTrainServeSymmetry:
    """Serving must resolve the same ensemble rows the dataset build does.

    EMOS resolves its spread predictor by scanning `ens__*__sd` columns at
    call time, so if training filters retired models and serving does not,
    the coefficient is fitted against one predictor and applied to another.
    """

    def test_build_snapshot_uses_the_configured_rows(self, tmp_path, monkeypatch):
        import grounded_weather_forecast.serve.predict as predict

        config = replace(
            write_config(tmp_path),
            ensembles=EnsemblesConfig(models=("gefs",), variables=("temp_c",)),
        )
        store = tmp_path / "data" / "ensembles.parquet"
        append_ensembles(
            store,
            pl.concat(
                [
                    parse_ensemble(payload(), "gefs", FETCHED, VARIABLES),
                    parse_ensemble(payload(), "retired", FETCHED, VARIABLES),
                ]
            ),
        )

        class Reached(Exception):
            """Short-circuit once the argument under test has been captured."""

        seen: list[pl.DataFrame | None] = []
        empty = pl.DataFrame()

        # Stub only what build_snapshot needs to reach the ensembles seam.
        monkeypatch.setattr(
            predict, "build_truth", lambda _config: (empty, empty, empty)
        )
        monkeypatch.setattr(predict, "build_observation_features", lambda _config: empty)
        monkeypatch.setattr(
            predict,
            "read_forecast_archive",
            lambda _forecasts: SimpleNamespace(
                hourly=empty, daily=empty, minutely=empty, completions=empty
            ),
        )

        def spy(*_args, ensembles=None, **_kwargs):
            seen.append(ensembles)
            raise Reached

        monkeypatch.setattr(predict, "build_hourly_matrix", spy)
        with pytest.raises(Reached):
            predict.build_snapshot(config, issue_time=ISSUE)

        assert seen, "build_hourly_matrix was never reached"
        used = seen[0]
        assert used is not None
        assert set(used["model"].unique()) == {"gefs"}, (
            "serving must apply the same [ensembles].models filter as the "
            "dataset build, or EMOS is fitted and served on different spreads"
        )
