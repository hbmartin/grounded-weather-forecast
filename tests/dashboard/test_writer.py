from datetime import UTC, date, datetime, timedelta

import polars as pl
from conftest import synthetic_hourly_matrix, write_config

from grounded_weather_forecast.backtest.engine import BacktestRequest, run_backtest
from grounded_weather_forecast.backtest.scores import scores_path, write_scores
from grounded_weather_forecast.contracts import (
    age_col,
    fxd_col,
    fx_col,
    hourly_variable,
)
from grounded_weather_forecast.dashboard import write_dashboard
from grounded_weather_forecast.dashboard.context import DashboardContext
from grounded_weather_forecast.dashboard.writer import _served_inputs
from grounded_weather_forecast.dataset.matrix import matrix_path
from grounded_weather_forecast.serve.schema import Forecast

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def _forecast(issue_time=NOW):
    return Forecast(
        schema_version=2,
        issued_at=issue_time.isoformat(),
        latitude=34.2768,
        longitude=-117.1692,
        dataset_fingerprint="f",
        sources=["alpha"],
        observation_at=None,
        minutely=[],
        hourly=[],
        daily=[],
    )


def test_cold_config_writes_full_not_yet_dashboard(tmp_path):
    config = write_config(tmp_path)
    path = write_dashboard(config, now=NOW)
    assert path == config.reports_dir / "dashboard.html"
    text = path.read_text(encoding="utf-8")
    for zone_id in "ABCDEFG":
        assert f'id="zone-{zone_id}"' in text
    assert "alert-strip" in text
    assert "not evaluable yet" in text
    assert "dashboard-data" in text
    # idempotent overwrite
    assert write_dashboard(config, now=NOW) == path


def test_real_scores_populate_the_leaderboard_panels(tmp_path):
    config = write_config(
        tmp_path,
        extra_toml="\n[backtest]\ninitial_train_days = 10\nstep_days = 5\n",
    )
    config.dataset.dir.mkdir(parents=True, exist_ok=True)
    matrix = synthetic_hourly_matrix(days=25, biases={"alpha": 4.0}, noise_sd=0.3)
    matrix.write_parquet(matrix_path(config.dataset.dir, "hourly", "live"))
    request = BacktestRequest(
        variables=(hourly_variable("temp_c"),),
        methods=("equal_weight", "best_provider", "climatology"),
    )
    scores = run_backtest(matrix, request, config)
    assert not scores.is_empty()
    destination = scores_path(config.dataset.dir / "scores", "hourly", "live")
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_scores(scores, destination)

    text = write_dashboard(config, now=NOW).read_text(encoding="utf-8")
    assert "Leaderboard — hourly_live" in text
    assert "equal_weight" in text
    assert "chart-d3-scores_hourly_live" in text  # baseline floor canvas
    assert "Provider error correlation" in text


def test_served_inputs_are_keyed_by_exact_issue_product_and_point(tmp_path):
    config = write_config(tmp_path)
    first_valid = NOW + timedelta(hours=1)
    second_valid = NOW + timedelta(hours=2)
    old_issue = NOW - timedelta(hours=1)
    hourly = pl.DataFrame(
        {
            "issue_time": [old_issue, NOW, NOW],
            "valid_time": [first_valid, first_valid, second_valid],
            fx_col("alpha", "temp_c"): [999.0, 10.0, 20.0],
            age_col("alpha"): [2.0, 1.25, 1.25],
        },
        schema_overrides={
            "issue_time": pl.Datetime("us", "UTC"),
            "valid_time": pl.Datetime("us", "UTC"),
        },
    )
    daily = pl.DataFrame(
        {
            "issue_time": [NOW],
            "forecast_date": [date(2026, 7, 19)],
            fxd_col("alpha", "temp_max_c"): [30.0],
        },
        schema_overrides={"issue_time": pl.Datetime("us", "UTC")},
    )
    ctx = DashboardContext(
        config=config,
        now=NOW,
        latest_forecast=_forecast(),
        hourly_matrix=hourly,
        daily_matrix=daily,
    )

    inputs = _served_inputs(ctx)

    assert inputs["hourly"][first_valid.isoformat()]["temp_c"]["alpha"] == {
        "value": 10.0,
        "age_hours": 1.25,
    }
    assert (
        inputs["hourly"][second_valid.isoformat()]["temp_c"]["alpha"]["value"] == 20.0
    )
    assert inputs["daily"]["2026-07-19"]["temp_max_c"]["alpha"] == {
        "value": 30.0,
        "age_hours": 1.25,
    }


def test_served_inputs_do_not_substitute_a_different_issue(tmp_path):
    config = write_config(tmp_path)
    later_issue = NOW + timedelta(hours=1)
    matrix = pl.DataFrame(
        {
            "issue_time": [later_issue],
            "valid_time": [later_issue + timedelta(hours=1)],
            fx_col("alpha", "temp_c"): [99.0],
        },
        schema_overrides={
            "issue_time": pl.Datetime("us", "UTC"),
            "valid_time": pl.Datetime("us", "UTC"),
        },
    )
    ctx = DashboardContext(
        config=config,
        now=NOW,
        latest_forecast=_forecast(),
        hourly_matrix=matrix,
    )

    assert _served_inputs(ctx) == {"hourly": {}, "daily": {}}


def test_served_inputs_use_latest_snapshot_visible_at_issue(tmp_path):
    config = write_config(tmp_path)
    prior_issue = NOW - timedelta(minutes=5)
    valid = NOW + timedelta(hours=1)
    matrix = pl.DataFrame(
        {
            "issue_time": [prior_issue],
            "valid_time": [valid],
            fx_col("alpha", "temp_c"): [12.0],
        },
        schema_overrides={
            "issue_time": pl.Datetime("us", "UTC"),
            "valid_time": pl.Datetime("us", "UTC"),
        },
    )
    ctx = DashboardContext(
        config=config,
        now=NOW,
        latest_forecast=_forecast(),
        hourly_matrix=matrix,
    )

    inputs = _served_inputs(ctx)

    assert inputs["hourly"][valid.isoformat()]["temp_c"]["alpha"]["value"] == 12.0


def test_corrupt_optional_artifacts_still_write_dashboard(tmp_path):
    config = write_config(tmp_path)
    config.dataset.dir.mkdir(parents=True, exist_ok=True)
    (config.dataset.dir / "manifest.json").write_text("{broken", encoding="utf-8")
    (config.dataset.dir / "runs.parquet").write_bytes(b"not parquet")
    observability = config.artifacts_dir / "observability"
    observability.mkdir(parents=True, exist_ok=True)
    (observability / "history.parquet").write_bytes(b"not parquet")

    text = write_dashboard(config, now=NOW).read_text(encoding="utf-8")

    assert "operator console" in text
    assert "dataset <code>unknown</code>" in text
