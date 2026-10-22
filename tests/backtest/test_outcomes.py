"""Expected short history is distinct from invalid or unevaluable inputs."""

from dataclasses import replace

import polars as pl
import pytest
from conftest import synthetic_hourly_matrix, write_config

from grounded_weather_forecast import cli
from grounded_weather_forecast.backtest import minutely
from grounded_weather_forecast.backtest.outcomes import (
    BacktestDiagnostics,
    ProductBacktestOutcome,
)
from grounded_weather_forecast.backtest.scores import load_scores
from grounded_weather_forecast.config import load_config
from grounded_weather_forecast.evaluation import config_fingerprint
from grounded_weather_forecast.serve import selection
from grounded_weather_forecast.status import EXIT_NO_FOLDS


def arguments(*extra):
    return cli.build_parser().parse_args(
        [
            "backtest",
            "--products",
            "hourly",
            "--hourly-variables",
            "temp_c",
            "--methods",
            "equal_weight",
            *extra,
        ]
    )


@pytest.mark.parametrize(
    "case",
    [
        "short",
        "empty",
        "missing",
        "corrupt",
        "no-truth",
        "missing-truth",
        "no-forecast",
        "no-finite-clock",
        "invalid-lead",
        "invalid-forecast",
    ],
)
def test_only_valid_short_history_is_pending(tmp_path, case):
    config = write_config(tmp_path)
    config.dataset.dir.mkdir()
    path = config.dataset.dir / "hourly_matrix_live.parquet"
    matrix = synthetic_hourly_matrix(days=2, max_lead=3)
    if case == "empty":
        matrix = matrix.head(0)
    elif case == "no-truth":
        matrix = matrix.with_columns(pl.lit(float("nan")).alias("t__temp_c__inst"))
    elif case == "missing-truth":
        matrix = matrix.drop("t__temp_c__inst")
    elif case == "no-forecast":
        matrix = matrix.drop("fx__alpha__temp_c", "fx__beta__temp_c")
    elif case == "no-finite-clock":
        matrix = matrix.with_columns(
            pl.lit(None, dtype=matrix.schema["issue_time"]).alias("issue_time")
        )
    elif case == "invalid-lead":
        matrix = matrix.with_columns(pl.lit("bad").alias("lead_hours"))
    elif case == "invalid-forecast":
        matrix = matrix.with_columns(pl.lit("bad").alias("fx__alpha__temp_c"))
    if case == "corrupt":
        path.write_bytes(b"invalid parquet")
    elif case != "missing":
        matrix.write_parquet(path)
    args = arguments()
    result = cli._cmd_backtest(config, args)
    assert result.exit_code == (EXIT_NO_FOLDS if case == "short" else 1)
    assert result.failure_kind == (
        "no_folds" if case == "short" else "backtest_product_failed"
    )
    assert not hasattr(args, "_failure_kind")
    if case == "corrupt":
        assert "Error" in result.error_detail
    if case == "short":
        files = list((config.dataset.dir / "scores").glob("scores_*.parquet"))
        assert len(files) == 1
        assert load_scores(files[0]).is_empty()


def test_adequate_history_without_usable_folds_is_failure(tmp_path):
    config = write_config(tmp_path)
    config = replace(config, backtest=replace(config.backtest, initial_train_days=1))
    config.dataset.dir.mkdir()
    # Truth is never known at any training origin, despite enough issue history.
    matrix = synthetic_hourly_matrix(days=4, max_lead=1).with_columns(
        pl.col("valid_time") + pl.duration(days=100)
    )
    matrix.write_parquet(config.dataset.dir / "hourly_matrix_live.parquet")
    args = arguments()
    assert cli._cmd_backtest(config, args).exit_code == 1
    assert getattr(args, "_failure_kind", None) is None


@pytest.mark.parametrize(
    ("last_status", "expected"),
    [("pending", EXIT_NO_FOLDS), ("failed", 1), ("evaluated", 0)],
)
def test_product_aggregation_preserves_partial_pending_and_errors(
    tmp_path, monkeypatch, last_status, expected
):
    config = write_config(tmp_path)
    monkeypatch.setattr(
        selection, "promote_methods", lambda *_a, **_k: selection.PromotionResult({})
    )

    def run(_config, _args, product, _methods):
        if product == "hourly":
            return ProductBacktestOutcome(product, "evaluated", 10)
        return ProductBacktestOutcome(
            product, last_status, 10 if last_status == "evaluated" else 0
        )

    monkeypatch.setattr(cli, "_run_matrix_backtest", run)
    args = arguments("--products", "hourly,daily")
    result = cli._cmd_backtest(config, args)
    assert result.exit_code == expected
    assert result.failure_kind == (
        {EXIT_NO_FOLDS: "no_folds", 1: "backtest_product_failed"}.get(expected)
    )


@pytest.mark.parametrize(
    "extra",
    [
        ["--methods", "unknown"],
        ["--products", "unsupported"],
        ["--methods", ""],
        ["--hourly-variables", "pop", "--methods", "seamless_regression"],
    ],
)
def test_unsupported_selection_is_not_pending(tmp_path, extra):
    config = write_config(tmp_path)
    config.dataset.dir.mkdir()
    synthetic_hourly_matrix(days=2).write_parquet(
        config.dataset.dir / "hourly_matrix_live.parquet"
    )
    args = arguments(*extra)
    # The final selection is supported but its truth column is absent.
    assert cli._cmd_backtest(config, args).exit_code == (
        1 if extra[-1] == "seamless_regression" else 2
    )
    assert getattr(args, "_failure_kind", None) is None


def test_minutely_uses_only_its_eligible_snapshot_clock(tmp_path):
    config = write_config(tmp_path)
    config = replace(config, backtest=replace(config.backtest, initial_train_days=2))
    matrix = synthetic_hourly_matrix(days=4, max_lead=6)
    first_day = matrix["issue_time"].min()
    # Later snapshots have no short-lead path, although the hourly clock is long.
    matrix = matrix.filter(
        (pl.col("issue_time") == first_day) | (pl.col("lead_hours") > 3)
    )
    grid = matrix.select(
        pl.col("valid_time"), pl.col("t__temp_c__inst").alias("temp_c")
    )
    diagnostics = BacktestDiagnostics()
    assert (
        list(
            minutely.iter_minutely_scores(
                matrix,
                grid,
                minutely.MinutelyRequest(variables=("temp_c",)),
                config,
                diagnostics=diagnostics,
            )
        )
        == []
    )
    assert diagnostics.history_too_short


def test_synthetic_minutely_is_not_applicable(tmp_path):
    outcome = cli._run_minutely_backtest(
        write_config(tmp_path), arguments("--source", "synthetic")
    )
    assert outcome.status == "not_applicable"


def test_daily_short_history_is_pending(tmp_path):
    config = write_config(tmp_path)
    config.dataset.dir.mkdir()
    matrix = synthetic_hourly_matrix(days=2, max_lead=1).select(
        "issue_time",
        "source_kind",
        "lead_bucket",
        pl.col("valid_time").dt.date().alias("forecast_date"),
        pl.lit(1.0).alias("lead_days"),
        pl.col("fx__alpha__temp_c").alias("fxd__alpha__temp_max_c"),
        pl.col("t__temp_c__inst").alias("t__temp_max_c"),
    )
    matrix.write_parquet(config.dataset.dir / "daily_matrix_live.parquet")
    args = arguments("--products", "daily", "--daily-variables", "temp_max_c")
    result = cli._cmd_backtest(config, args)
    assert result.exit_code == EXIT_NO_FOLDS
    assert result.failure_kind == "no_folds"


def test_schedule_setting_does_not_change_scientific_identity(tmp_path):
    config = write_config(tmp_path)
    with (tmp_path / "config.toml").open("a") as file:
        file.write('\n[maintenance]\nsynthetic_backtest = "off"\n')
    assert config_fingerprint(
        load_config(tmp_path / "config.toml")
    ) == config_fingerprint(config)
