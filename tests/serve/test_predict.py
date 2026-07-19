import json
import math
from datetime import timedelta

import pytest
from conftest import make_forecast_db, make_station_db, minute_series, utc, write_config

from grounded_weather_forecast.cli import main
from grounded_weather_forecast.dataset.matrix import write_dataset
from grounded_weather_forecast.serve.predict import (
    NoForecastDataError,
    UnsupportedMethodError,
    build_snapshot,
    predict,
)
from grounded_weather_forecast.serve.schema import SCHEMA_VERSION
from grounded_weather_forecast.serve.selection import Selection, select_methods

NOW = utc(2026, 3, 22, 17, 0)
FETCH = "2026-03-22T16:30:00+00:00"


def build_fixture(tmp_path, *, obs_temp_f=68.0, extra_toml=""):
    """A station with recent obs plus two providers with 48h of hourly data.

    The temperature wobbles: a dead-flat series would (correctly) be caught by
    the flatline QC filter and never become truth.
    """
    samples = minute_series(NOW - timedelta(hours=6), 6 * 12, 300)
    station_rows = [
        (
            ts.strftime("%Y-%m-%d %H:%M:%S"),
            {
                "outTemp": obs_temp_f + 0.1 * (index % 5),
                "outHumi": 40.0 + 0.1 * (index % 3),
            },
        )
        for index, ts in enumerate(samples)
    ]
    make_station_db(tmp_path / "station.db", station_rows)

    results = []
    for provider, offset in (("nws", 0.0), ("open_meteo", 2.0)):
        hourly = [
            (
                NOW + timedelta(hours=lead),
                {
                    "temperature": 15.0 + offset,
                    "humidity": 140.0,
                    "wind_speed": -2.0,
                    "precipitation": -5.0,
                    "precipitation_probability": 2.0,
                },
            )
            for lead in range(49)
        ]
        daily = [
            (
                (NOW + timedelta(days=d)).strftime("%Y-%m-%d"),
                {
                    "temperature_max": 22.0 + offset,
                    "temperature_min": 8.0 + offset,
                    "precipitation_sum": -3.0,
                    "precipitation_probability_max": 2.0,
                },
            )
            for d in range(10)
        ]
        results.append(
            {
                "provider": provider,
                "fetched_at": FETCH,
                "hourly": hourly,
                "daily": daily,
                "minutely": [
                    (NOW + timedelta(minutes=m), 0.5, 0.4) for m in range(1, 31)
                ],
            }
        )
    results.append(
        {
            "provider": "stale_minutely",
            "fetched_at": (NOW - timedelta(hours=13)).isoformat(),
            "minutely": [(NOW + timedelta(minutes=1), 99.0, 1.0)],
        }
    )
    make_forecast_db(
        tmp_path / "fx.sqlite",
        [{"completed_at": FETCH, "results": results}],
    )
    config = write_config(
        tmp_path,
        min_hour_coverage=0.05,
        min_day_coverage=0.02,
        extra_toml=extra_toml,
    )
    write_dataset(config)
    return config


@pytest.fixture
def config(tmp_path):
    return build_fixture(tmp_path)


@pytest.fixture
def config_no_provider_qc(tmp_path):
    # The provider values in the fixture are deliberately out of physical bounds
    # (humidity 140, wind -2, ...) to exercise the serve-layer clamp; disable the
    # provider QC that would otherwise (correctly) null them before serving.
    return build_fixture(tmp_path, extra_toml="[provider_qc]\nenabled = false")


class TestSnapshot:
    def test_as_of_view(self, config):
        snapshot = build_snapshot(config, NOW)
        assert snapshot.issue_time == NOW
        assert not snapshot.hourly.is_empty()
        assert not snapshot.daily.is_empty()
        assert snapshot.observation_at is not None
        assert snapshot.observation["temp_c"] == pytest.approx(20.0, abs=0.1)
        assert (snapshot.hourly["lead_hours"] >= 0).all()
        assert "stale_minutely" not in snapshot.minutely["source"].to_list()

    def test_stale_archive_refuses(self, config):
        with pytest.raises(NoForecastDataError, match="no provider forecast"):
            build_snapshot(config, NOW + timedelta(days=30))


class TestPredict:
    def test_three_products(self, config):
        document = predict(config, {}, now=NOW)
        assert document.schema_version == SCHEMA_VERSION
        assert len(document.minutely) == 60
        assert len(document.hourly) == 49
        assert len(document.daily) == 10
        assert set(document.sources) == {"nws", "open_meteo"}

    def test_json_round_trip(self, config):
        payload = json.loads(predict(config, {}, now=NOW).to_json())
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["hourly"][0]["lead_bucket"] == "0-1h"
        assert "temp_c" in payload["hourly"][0]["values"]
        assert "temp_c" in payload["hourly"][0]["methods"]

    def test_minutely_is_anchored_to_the_observation(self, config):
        # providers say 15-17C; the station says 20C. The nowcast must start
        # near the observation and relax toward the blend.
        document = predict(config, {}, now=NOW)
        first = document.minutely[0].temp_c
        last = document.minutely[-1].temp_c
        blend_now = document.hourly[0].values["temp_c"]
        assert first is not None
        assert last is not None
        assert blend_now is not None
        assert abs(first - 20.0) < abs(first - blend_now)  # anchored to obs
        assert abs(last - blend_now) < abs(first - blend_now)  # relaxing back

    def test_minutely_precip_from_native_points(self, config):
        document = predict(config, {}, now=NOW)
        assert document.minutely[0].precip_intensity_mmh == pytest.approx(0.5)
        assert document.minutely[0].pop == pytest.approx(0.4)
        # providers only published 30 minutes of it
        assert document.minutely[-1].precip_intensity_mmh is None

    def test_daily_hi_lo(self, config):
        document = predict(config, {}, now=NOW)
        values = document.daily[0].values
        assert values["temp_max_c"] is not None
        assert values["temp_min_c"] is not None
        assert values["temp_max_c"] > values["temp_min_c"]

    def test_emitted_values_obey_physical_bounds(self, config_no_provider_qc):
        document = predict(config_no_provider_qc, {}, now=NOW)
        for point in document.hourly:
            assert point.values["humidity_pct"] == 100.0
            assert point.values["wind_speed_ms"] == 0.0
            assert point.values["precip_mm"] == 0.0
            assert point.values["pop"] == 1.0
        for point in document.daily:
            assert point.values["precip_sum_mm"] == 0.0
            assert point.values["pop"] == 1.0

    def test_force_method(self, config):
        document = predict(config, {}, now=NOW, force_method="equal_weight")
        methods = {m for p in document.hourly for m in p.methods.values()}
        assert methods == {"equal_weight"}

    def test_unsupported_forced_method_fails_the_whole_prediction(self, config):
        with pytest.raises(UnsupportedMethodError, match="hourly.precip_mm"):
            predict(config, {}, now=NOW, force_method="persistence")

    def test_unsupported_stale_selection_degrades_without_provenance(self, config):
        selections = {
            ("hourly", "precip_mm", "0-1h"): Selection(
                "persistence",
                reason="old release",
                release_id="stale-release",
                evaluation_id="old-evaluation",
                dataset_fingerprint="old-data",
            )
        }
        document = predict(config, selections, now=NOW)
        first = document.hourly[0]
        assert first.methods["precip_mm"] == "equal_weight"
        assert "degraded stale selection" in first.selection_reasons["precip_mm"]
        assert "stale-release" not in document.release_ids

    def test_unknown_stale_selection_degrades_without_provenance(self, config):
        selections = {
            ("hourly", "temp_c", "0-1h"): Selection(
                "retired_method",
                reason="old release",
                release_id="retired-release",
                evaluation_id="old-evaluation",
                dataset_fingerprint="old-data",
            )
        }
        document = predict(config, selections, now=NOW)
        first = document.hourly[0]
        assert first.methods["temp_c"] == "equal_weight"
        assert "unknown method retired_method" in first.selection_reasons["temp_c"]
        assert "retired-release" not in document.release_ids

    def test_fallback_method_when_no_scores(self, config):
        document = predict(config, {}, now=NOW)
        methods = {m for p in document.hourly for m in p.methods.values()}
        assert methods == {"equal_weight"}
        assert document.status == "degraded"

    def test_incompatible_historical_release_degrades_without_provenance(self, config):
        from grounded_weather_forecast.evaluation import (
            config_fingerprint,
            dataset_fingerprint,
        )

        release = {
            "release_id": "old-code-release",
            "promoted_at": (NOW - timedelta(days=1)).isoformat(),
            "dataset_fingerprint": dataset_fingerprint(config),
            "config_fingerprint": config_fingerprint(config),
            "evaluation_ids": ["old-evaluation"],
            "evaluation_contexts": [],
            "training_cutoff": None,
            "selections": {
                "hourly.temp_c.0-1h": {
                    "method_id": "gbm",
                    "reason": "historical winner",
                    "evaluation_id": "old-evaluation",
                    "code_version": "0.4.0+retired-implementation",
                    "n": 100,
                    "mae": 1.0,
                }
            },
        }
        release_dir = config.artifacts_dir / "releases"
        release_dir.mkdir(parents=True)
        (release_dir / "old-code-release.json").write_text(
            json.dumps(release), encoding="utf-8"
        )

        selections = select_methods(config, config.dataset.dir / "scores", as_of=NOW)
        document = predict(config, selections, now=NOW)

        assert selections == {}
        assert document.status == "degraded"
        assert document.release_ids == []


class TestPredictCli:
    def test_writes_json_and_history(self, tmp_path, capsys):
        config = build_fixture(tmp_path)
        out = tmp_path / "forecast.json"
        code = main(
            [
                "--config",
                str(tmp_path / "config.toml"),
                "predict",
                "--now",
                NOW.isoformat(),
                "--out",
                str(out),
            ]
        )
        assert code == 0
        printed = capsys.readouterr().out
        assert "wrote" in printed
        assert "appended" in printed
        assert out.exists()
        assert config.predict.history_path.exists()

    def test_stale_archive_exits_nonzero(self, tmp_path, capsys):
        build_fixture(tmp_path)
        code = main(
            [
                "--config",
                str(tmp_path / "config.toml"),
                "predict",
                "--now",
                (NOW + timedelta(days=30)).isoformat(),
                "--no-history",
            ]
        )
        assert code == 1
        assert "cannot predict" in capsys.readouterr().out

    def test_stdout_is_json_only(self, tmp_path, capsys):
        build_fixture(tmp_path)
        code = main(
            [
                "--config",
                str(tmp_path / "config.toml"),
                "predict",
                "--now",
                NOW.isoformat(),
            ]
        )
        assert code == 0
        captured = capsys.readouterr()
        assert json.loads(captured.out)["issued_at"] == NOW.isoformat()
        assert "appended" in captured.err

    def test_unsupported_forced_method_is_an_actionable_cli_error(
        self, tmp_path, capsys
    ):
        build_fixture(tmp_path)
        code = main(
            [
                "--config",
                str(tmp_path / "config.toml"),
                "predict",
                "--now",
                NOW.isoformat(),
                "--method",
                "persistence",
                "--no-history",
            ]
        )
        assert code == 1
        output = capsys.readouterr().out
        assert "cannot predict" in output
        assert "does not support hourly.precip_mm" in output

    def test_unknown_forced_method_is_an_actionable_cli_error(self, tmp_path, capsys):
        build_fixture(tmp_path)
        code = main(
            [
                "--config",
                str(tmp_path / "config.toml"),
                "predict",
                "--now",
                NOW.isoformat(),
                "--method",
                "retired_method",
                "--no-history",
            ]
        )
        assert code == 1
        output = capsys.readouterr().out
        assert "cannot predict" in output
        assert "unknown method 'retired_method'" in output


class TestMinutelySinglePass:
    """Anchoring is applied exactly once, by whichever stage was promoted."""

    def make_snapshot(self):
        import numpy as np
        import polars as pl

        from grounded_weather_forecast.serve.predict import Snapshot
        from grounded_weather_forecast.timeutil import utc

        issue = utc(2026, 3, 22, 12, 0)
        hourly = pl.DataFrame(
            {
                "valid_time": [issue + timedelta(hours=1), issue + timedelta(hours=2)],
                "lead_hours": [1.0, 2.0],
            },
            schema_overrides={"valid_time": pl.Datetime("us", "UTC")},
        )
        return Snapshot(
            issue_time=issue,
            hourly=hourly,
            daily=pl.DataFrame(),
            minutely=pl.DataFrame(),
            observation={"temp_c": 10.0},
            observation_at=issue,
        ), np.array([20.0, 21.0])

    def blend(self, path, method_id):
        from grounded_weather_forecast.serve.predict import VariableBlend

        return VariableBlend(
            point=path,
            methods=[method_id, method_id],
            reasons=["", ""],
            release_ids=[None, None],
            quantiles=[{}, {}],
        )

    def test_anchored_selection_is_not_re_anchored(self, la_config):
        from grounded_weather_forecast.serve.predict import minutely_product

        snapshot, path = self.make_snapshot()
        points = minutely_product(
            snapshot,
            {"temp_c": self.blend(path, "anchored_fitted_grounded")},
            la_config,
        )
        # pure interpolation on the lead-zero-extended hourly path; the
        # 10-degree observation is deliberately ignored (already anchored)
        assert points[0].temp_c == pytest.approx(19.0 + 1.0 / 60.0, abs=0.001)

    def test_unanchored_selection_converges_to_the_observation(self, la_config):
        from grounded_weather_forecast.serve.predict import minutely_product

        snapshot, path = self.make_snapshot()
        points = minutely_product(
            snapshot,
            {"temp_c": self.blend(path, "grounded_equal_weight")},
            la_config,
        )
        # Both the base path and residual use the 19.0 lead-zero extrapolation,
        # so the first minute stays continuous with the observation.
        expected_first = 19.0 + 1.0 / 60.0 + math.exp(-(1 / 60) / 3.0) * -9.0
        assert points[0].temp_c == pytest.approx(expected_first, abs=0.02)
        # minute 60: w = exp(-1/tau) with the 3h config default
        expected = 20.0 + math.exp(-1.0 / 3.0) * -9.0
        assert points[-1].temp_c == pytest.approx(expected, abs=0.1)
        assert points[-1].temp_c > points[0].temp_c  # decaying toward the path

    def test_minutely_dew_point_is_not_above_temperature(self, la_config):
        from grounded_weather_forecast.serve.predict import minutely_product

        snapshot, path = self.make_snapshot()
        points = minutely_product(
            snapshot,
            {
                "temp_c": self.blend(path, "anchored_fitted_grounded"),
                "dew_point_c": self.blend(path + 5.0, "anchored_fitted_grounded"),
            },
            la_config,
        )
        assert all(
            point.dew_point_c <= point.temp_c
            for point in points
            if point.dew_point_c is not None and point.temp_c is not None
        )

    def test_mixed_anchor_regimes_follow_the_lead_zero_regime(self, la_config):
        """The row owning lead zero decides anchoring for the whole range.

        Switching regime partway would both step the path and flip the
        anchoring decision mid-range. Here lead zero is anchored, so the
        config decay must not be re-applied and the path stays continuous.
        """
        from grounded_weather_forecast.serve.predict import (
            VariableBlend,
            minutely_product,
        )

        snapshot, path = self.make_snapshot()
        blend = VariableBlend(
            point=path,
            methods=["anchored_fitted_grounded", "grounded_equal_weight"],
            reasons=["", ""],
            release_ids=[None, None],
            quantiles=[{}, {}],
        )

        points = minutely_product(snapshot, {"temp_c": blend}, la_config)
        values = [point.temp_c for point in points]

        # Already anchored: the 10.0 C observation is not pulled in again.
        assert min(values) > 15.0
        # Continuous, and arriving at the hourly value at lead 1 h.
        assert values[-1] == pytest.approx(path[0])
        assert max(abs(b - a) for a, b in zip(values, values[1:])) < 0.1

    def test_mixed_anchor_regimes_do_not_step_inside_the_horizon(self, la_config):
        """Regression: the bracketing midpoint falling inside the 60-min horizon.

        With sub-hourly first leads (the normal case for a non-o'clock issue
        time) the old nearest-neighbour fallback produced a flat line with one
        large jump, and flipped `already_anchored` halfway, so the live
        observation was ignored for the minutes on the anchored side.
        """
        import numpy as np
        import polars as pl

        from grounded_weather_forecast.serve.predict import (
            Snapshot,
            VariableBlend,
            minutely_product,
        )
        from grounded_weather_forecast.timeutil import utc

        issue = utc(2026, 3, 22, 12, 37)
        hourly = pl.DataFrame(
            {
                "valid_time": [utc(2026, 3, 22, 13), utc(2026, 3, 22, 14)],
                "lead_hours": [23.0 / 60.0, 83.0 / 60.0],
            },
            schema_overrides={"valid_time": pl.Datetime("us", "UTC")},
        )
        snapshot = Snapshot(
            issue_time=issue,
            hourly=hourly,
            daily=pl.DataFrame(),
            minutely=pl.DataFrame(),
            observation={"temp_c": 21.0},
            observation_at=issue,
        )
        blend = VariableBlend(
            point=np.array([20.0, 23.0]),
            methods=["grounded_equal_weight", "anchored_fitted_grounded"],
            reasons=["", ""],
            release_ids=[None, None],
            quantiles=[{}, {}],
        )

        values = [
            point.temp_c
            for point in minutely_product(snapshot, {"temp_c": blend}, la_config)
        ]

        jumps = [abs(b - a) for a, b in zip(values, values[1:])]
        assert max(jumps) < 0.2, "minutely path must not step between regimes"
        # Lead zero is un-anchored, so the live observation is honoured.
        assert values[0] == pytest.approx(21.0, abs=0.5)

    @pytest.mark.parametrize("observation", [21.0, 22.4])
    def test_mixed_anchor_regimes_stay_inside_the_hourly_bracket(
        self, la_config, observation
    ):
        """Regression: interpolating across a regime boundary anchored twice.

        The far hourly row is ``anchored_*``, so its point already carries a
        fitted correction. Drawing the minutely path through it while
        reporting the segment un-anchored added the live-observation residual
        on top, pushing the nowcast outside the bracketing hourly values --
        which a monotone interpolation can never legitimately do.
        """
        import numpy as np
        import polars as pl

        from grounded_weather_forecast.serve.predict import (
            Snapshot,
            VariableBlend,
            minutely_product,
        )
        from grounded_weather_forecast.timeutil import utc

        low, high = 20.0, 23.0
        issue = utc(2026, 3, 22, 12, 37)
        hourly = pl.DataFrame(
            {
                "valid_time": [utc(2026, 3, 22, 13), utc(2026, 3, 22, 14)],
                "lead_hours": [23.0 / 60.0, 83.0 / 60.0],
            },
            schema_overrides={"valid_time": pl.Datetime("us", "UTC")},
        )
        snapshot = Snapshot(
            issue_time=issue,
            hourly=hourly,
            daily=pl.DataFrame(),
            minutely=pl.DataFrame(),
            observation={"temp_c": observation},
            observation_at=issue,
        )
        blend = VariableBlend(
            point=np.array([low, high]),
            methods=["grounded_equal_weight", "anchored_fitted_grounded"],
            reasons=["", ""],
            release_ids=[None, None],
            quantiles=[{}, {}],
        )

        values = [
            point.temp_c
            for point in minutely_product(snapshot, {"temp_c": blend}, la_config)
            if point.temp_c is not None
        ]

        assert values, "the nowcast must emit temperatures"
        assert max(values) <= high + 1e-9, (
            f"emitted {max(values):.3f} above the hourly bracket max {high}"
        )
        assert min(values) >= min(low, observation) - 1e-9


class TestPhysicalCoherence:
    def test_hourly_points_and_differing_quantile_grids_are_coherent(self):
        import numpy as np

        from grounded_weather_forecast.serve.predict import _cohere_hourly
        from grounded_weather_forecast.serve.schema import HourlyPoint

        point = HourlyPoint(
            valid_time=NOW.isoformat(),
            lead_hours=1.0,
            values={
                "temp_c": 1.0,
                "dew_point_c": 5.0,
                "wind_speed_ms": 3.0,
                "wind_gust_ms": 1.0,
            },
            quantiles={
                "temp_c": {"0.1": 0.0, "0.9": 2.0},
                "dew_point_c": {"0.25": 3.0, "0.75": 4.0},
                "wind_speed_ms": {"0.2": 2.0, "0.8": 4.0},
                "wind_gust_ms": {"0.1": 0.0, "0.9": 1.0},
            },
        )
        _cohere_hourly([point])

        assert point.values["dew_point_c"] <= point.values["temp_c"]
        assert point.values["wind_gust_ms"] >= point.values["wind_speed_ms"]
        levels = np.array([0.1, 0.2, 0.25, 0.75, 0.8, 0.9])

        def curve(name):
            raw = point.quantiles[name]
            x = np.array([float(level) for level in raw])
            y = np.array(list(raw.values()))
            order = np.argsort(x)
            return np.interp(levels, x[order], y[order])

        assert (curve("dew_point_c") <= curve("temp_c") + 1e-12).all()
        assert (curve("wind_gust_ms") >= curve("wind_speed_ms") - 1e-12).all()
        for name, value in point.values.items():
            quantiles = point.quantiles[name].values()
            assert min(quantiles) <= value <= max(quantiles)

    def test_daily_low_high_points_and_quantiles_are_coherent(self):
        import numpy as np

        from grounded_weather_forecast.serve.predict import _cohere_daily
        from grounded_weather_forecast.serve.schema import DailyPoint

        point = DailyPoint(
            date_local="2026-03-22",
            lead_days=1,
            values={"temp_min_c": 12.0, "temp_max_c": 8.0},
            quantiles={
                "temp_min_c": {"0.25": 10.0, "0.75": 14.0},
                "temp_max_c": {"0.1": 5.0, "0.9": 9.0},
            },
        )
        _cohere_daily([point])
        assert point.values["temp_min_c"] <= point.values["temp_max_c"]
        levels = np.array([0.1, 0.25, 0.75, 0.9])

        def curve(name):
            raw = point.quantiles[name]
            x = np.array([float(level) for level in raw])
            y = np.array(list(raw.values()))
            order = np.argsort(x)
            return np.interp(levels, x[order], y[order])

        assert (curve("temp_min_c") <= curve("temp_max_c") + 1e-12).all()

    def test_point_clipping_cannot_reintroduce_dew_point_violation(self):
        from grounded_weather_forecast.serve.predict import _cohere_hourly
        from grounded_weather_forecast.serve.schema import HourlyPoint

        point = HourlyPoint(
            valid_time=NOW.isoformat(),
            lead_hours=1.0,
            values={"temp_c": 10.0, "dew_point_c": 12.0},
            quantiles={"temp_c": {"0.1": 0.0, "0.9": 5.0}},
        )

        _cohere_hourly([point])

        assert point.values["temp_c"] == 5.0
        assert point.values["dew_point_c"] == 5.0


class TestCoherenceLeavesValidDistributionsAlone:
    """Coherence enforcement must be a no-op on input that is already coherent."""

    LEVELS = (0.05, 0.1, 0.25, 0.75, 0.9, 0.95)

    def _grid(self, mu, sd=2.0):
        from scipy.stats import norm

        return {str(level): float(norm.ppf(level, mu, sd)) for level in self.LEVELS}

    @pytest.mark.parametrize("depression", [0.5, 1.0, 2.0, 3.0])
    def test_shared_grid_coherent_dew_point_is_untouched(self, depression):
        """Regression: the lower q75 was being bounded by the upper q25.

        Both curves live on the same conformal grid and are coherent at every
        level, so nothing needs adjusting. Enforcing knot-to-neighbour bounds
        anyway shifted dew-point q75 down by up to 2.2 K -- worst exactly when
        the dew-point depression is small, i.e. fog and humid nights.
        """
        from grounded_weather_forecast.serve.predict import _cohere_pair

        temperature = 21.0
        dew_point = temperature - depression
        dew_q, temp_q = self._grid(dew_point), self._grid(temperature)
        for level in self.LEVELS:
            assert dew_q[str(level)] <= temp_q[str(level)], "fixture is incoherent"
        before = dict(dew_q)

        _cohere_pair(
            {"dew_point_c": dew_point, "temp_c": temperature},
            {"dew_point_c": dew_q, "temp_c": temp_q},
            "dew_point_c",
            "temp_c",
            adjust="lower",
        )

        for level in self.LEVELS:
            assert dew_q[str(level)] == pytest.approx(before[str(level)], abs=1e-9), (
                f"level {level} moved {dew_q[str(level)] - before[str(level)]:+.3f} K "
                "on an already-coherent shared grid"
            )

    def test_differing_grids_are_still_made_coherent(self):
        """The unequal-grid path -- which needs the neighbour bound -- survives."""
        import numpy as np

        from grounded_weather_forecast.serve.predict import _cohere_pair

        dew_q = {"0.25": 12.0, "0.75": 18.0}
        temp_q = {"0.1": 9.0, "0.5": 11.0, "0.9": 13.0}
        _cohere_pair(
            {"dew_point_c": 15.0, "temp_c": 11.0},
            {"dew_point_c": dew_q, "temp_c": temp_q},
            "dew_point_c",
            "temp_c",
            adjust="lower",
        )
        grid = np.linspace(0.25, 0.75, 11)
        dew = np.interp(
            grid, [float(k) for k in sorted(dew_q, key=float)], sorted(dew_q.values())
        )
        temp = np.interp(
            grid, [float(k) for k in sorted(temp_q, key=float)], sorted(temp_q.values())
        )
        assert np.all(dew <= temp + 1e-9), "dew point must not exceed temperature"

    @pytest.mark.parametrize("depression", [0.5, 1.0])
    def test_coherent_conformal_and_emos_grids_are_untouched(self, depression):
        """Real serving grids must not enter the lossy neighbour projection."""
        from scipy.stats import norm

        from grounded_weather_forecast.serve.predict import _cohere_pair

        conformal = (0.05, 0.1, 0.25, 0.75, 0.9, 0.95)
        emos = tuple(index / 20 for index in range(1, 20))
        temperature = 21.0
        dew_point = temperature - depression
        dew_q = {
            str(level): float(norm.ppf(level, dew_point, 2.0)) for level in conformal
        }
        temp_q = {
            str(level): float(norm.ppf(level, temperature, 2.0)) for level in emos
        }
        before = dict(dew_q)

        _cohere_pair(
            {"dew_point_c": dew_point, "temp_c": temperature},
            {"dew_point_c": dew_q, "temp_c": temp_q},
            "dew_point_c",
            "temp_c",
            adjust="lower",
        )

        assert dew_q == pytest.approx(before)


class TestServingPathFitting:
    """`_fit_methods` past its early returns: the warm-start path serving uses.

    Every other test returns at the `no truth sources` / `n_rows == 0` guards,
    so the artifact round trip, the stateful advance, and the observability
    snapshot all shipped unexercised through this entry point.
    """

    def _slice_frames(self):
        import polars as pl
        from conftest import synthetic_hourly_matrix

        matrix = synthetic_hourly_matrix(days=12, max_lead=12, seed=77)
        return matrix, matrix.filter(pl.col("lead_hours") <= 6.0)

    def _fit(self, config, method_ids, issue_time=NOW):
        import numpy as np  # noqa: F401

        from grounded_weather_forecast.contracts import TruthSemantics, hourly_variable
        from grounded_weather_forecast.serve.predict import _fit_methods

        train, predict_frame = self._slice_frames()
        return _fit_methods(
            train,
            predict_frame,
            hourly_variable("temp_c"),
            method_ids,
            daily=False,
            semantics=TruthSemantics.INSTANTANEOUS,
            config=config,
            issue_time=issue_time,
        )

    def test_fits_and_predicts_each_requested_method(self, tmp_path):
        config = write_config(tmp_path)
        fitted = self._fit(config, {"equal_weight", "grounded_equal_weight"})

        assert fitted is not None
        results, x = fitted
        assert set(results) == {"equal_weight", "grounded_equal_weight"}
        for result in results.values():
            assert result.point.shape[0] == x.n_rows

    def test_a_stateful_method_persists_and_warm_starts(self, tmp_path):
        """Second serve must load the artifact and advance it, not refit."""
        config = write_config(tmp_path)

        import numpy as np

        first = self._fit(config, {"ewa"})
        assert first is not None
        state_dir = config.artifacts_dir / "state"
        assert list(state_dir.rglob("*.json")), "the serve path must persist state"

        second = self._fit(config, {"ewa"}, issue_time=NOW + timedelta(minutes=10))
        assert second is not None
        # Same evidence replayed: the warm start must land where the first did.
        np.testing.assert_allclose(
            second[0]["ewa"].point, first[0]["ewa"].point, atol=1e-9
        )

    def test_a_corrupt_artifact_falls_back_to_a_full_refit(self, tmp_path):
        """The digest guard's whole point: never silently combine histories."""
        config = write_config(tmp_path)
        assert self._fit(config, {"ewa"}) is not None
        for path in (config.artifacts_dir / "state").rglob("*.json"):
            path.write_text('{"schema_version": 1}', encoding="utf-8")

        import numpy as np

        refitted = self._fit(config, {"ewa"})

        assert refitted is not None
        assert np.isfinite(refitted[0]["ewa"].point).any()

    def test_observability_is_snapshotted_through_the_serving_path(self, tmp_path):
        config = write_config(tmp_path)
        assert self._fit(config, {"ewa"}) is not None
        assert list((config.artifacts_dir / "observability").rglob("*.json")), (
            "the dashboard's only serving-path hook must actually fire"
        )
