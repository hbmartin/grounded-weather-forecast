"""Pending truth is tied to the published snapshot, never the wall clock."""

import hashlib
import json
from dataclasses import replace
from datetime import timedelta
from itertools import permutations

import polars as pl
import pytest
from conftest import synthetic_hourly_matrix, write_config

from grounded_weather_forecast import cli
from grounded_weather_forecast.backtest import minutely
from grounded_weather_forecast.backtest.outcomes import (
    BacktestDiagnostics,
    ProductBacktestOutcome,
)
from grounded_weather_forecast.backtest.splits import (
    daily_truth_known_at,
    hourly_truth_known_at,
)
from grounded_weather_forecast.dataset.evaluation_snapshot import evaluation_snapshot
from grounded_weather_forecast.serve import selection


def publish_metadata(config, moment, product="hourly", source="live", *, minute=False):
    names = {
        f"{product}_matrix": config.dataset.dir / f"{product}_matrix_{source}.parquet"
    }
    if minute:
        names["truth_minute"] = config.dataset.dir / "truth_minute.parquet"
    raw = {
        "created_at": moment.isoformat(),
        "files": {
            name: {"sha256_16": hashlib.sha256(path.read_bytes()).hexdigest()[:16]}
            for name, path in names.items()
        },
    }
    name = "manifest.json" if source == "live" else "manifest_synthetic.json"
    (config.dataset.dir / name).write_text(json.dumps(raw))
    return config.dataset.dir / name


def request(product="hourly", method="equal_weight", source="live"):
    return cli.build_parser().parse_args(
        [
            "backtest",
            "--products",
            product,
            "--methods",
            method,
            "--hourly-variables",
            "temp_c",
            "--daily-variables",
            "temp_max_c",
            "--source",
            source,
        ]
    )


@pytest.mark.parametrize("product", ["hourly", "daily"])
@pytest.mark.parametrize("source", ["live", "synthetic"])
def test_future_truth_pending_then_overdue_is_failure(
    tmp_path, product, source, monkeypatch
):
    config = write_config(tmp_path)
    config = replace(config, backtest=replace(config.backtest, initial_train_days=3))
    config.dataset.dir.mkdir()
    matrix = synthetic_hourly_matrix(days=4, max_lead=24, source_kind=source).filter(
        pl.col("lead_hours") == 24
    )
    cutoff = matrix["issue_time"].max()
    if product == "daily":
        matrix = matrix.select(
            "issue_time",
            "source_kind",
            "lead_bucket",
            pl.col("valid_time").dt.date().alias("forecast_date"),
            pl.lit(1.0).alias("lead_days"),
            pl.col("fx__alpha__temp_c").alias("fxd__alpha__temp_max_c"),
            pl.col("t__temp_c__inst").alias("t__temp_max_c"),
        )
    truth = "t__temp_max_c" if product == "daily" else "t__temp_c__inst"
    known = (
        daily_truth_known_at(matrix, config.station.timezone)
        if product == "daily"
        else hourly_truth_known_at(matrix)
    )
    matrix = matrix.with_columns(
        pl.when(known <= cutoff)
        .then(pl.col(truth))
        .otherwise(float("nan"))
        .alias(truth)
    )
    matrix.write_parquet(config.dataset.dir / f"{product}_matrix_{source}.parquet")
    manifest = publish_metadata(config, cutoff, product, source)
    monkeypatch.setattr(selection, "select_methods", lambda *_a, **_k: {})
    first = cli._cmd_backtest(config, request(product, source=source))
    assert first.exit_code == 3
    assert first.failure_kind == "no_folds"
    assert cli._cmd_backtest(config, request(product, source=source)) == first
    original_metadata = manifest.read_text()
    manifest.unlink()
    assert cli._cmd_backtest(config, request(product, source=source)).exit_code == 1
    manifest.write_text(original_metadata)
    # Advancing the snapshot without filling expected observations makes this
    # an actual data gap; the operating-system file timestamp is irrelevant.
    raw = json.loads(manifest.read_text())
    raw["created_at"] = (cutoff + timedelta(days=5)).isoformat()
    manifest.write_text(json.dumps(raw))
    failed = cli._cmd_backtest(config, request(product, source=source))
    assert failed.exit_code == 1
    assert failed.failure_kind is None


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "corrupt",
        "mismatch",
        "missing-time",
        "naive-time",
        "early-time",
        "dangling",
    ],
)
def test_snapshot_metadata_never_invents_maturity(tmp_path, fault):
    config = write_config(tmp_path)
    config.dataset.dir.mkdir()
    matrix = synthetic_hourly_matrix(days=4, max_lead=1)
    matrix.write_parquet(config.dataset.dir / "hourly_matrix_live.parquet")
    path = publish_metadata(config, matrix["issue_time"].max())
    raw = json.loads(path.read_text())
    if fault == "missing":
        path.unlink()
        assert evaluation_snapshot(config.dataset.dir, "hourly", "live") is None
        return
    if fault == "dangling":
        path.unlink()
        path.symlink_to(config.dataset.dir / "absent-manifest")
        with pytest.raises(ValueError, match="dangling"):
            evaluation_snapshot(config.dataset.dir, "hourly", "live")
        return
    if fault == "corrupt":
        path.write_text("{")
    else:
        if fault == "mismatch":
            raw["files"]["hourly_matrix"]["sha256_16"] = "wrong"
        elif fault == "missing-time":
            del raw["created_at"]
        elif fault == "early-time":
            raw["created_at"] = (
                matrix["issue_time"].min() - timedelta(days=1)
            ).isoformat()
        else:
            raw["created_at"] = "2026-01-01T00:00:00"
        path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=r"snapshot|Expecting"):
        evaluation_snapshot(config.dataset.dir, "hourly", "live")


def test_minutely_maturity_uses_actual_minute_targets(tmp_path):
    config = write_config(tmp_path)
    config = replace(config, backtest=replace(config.backtest, initial_train_days=1))
    matrix = synthetic_hourly_matrix(days=2, max_lead=3)
    cutoff = matrix["issue_time"].max()
    ticks = pl.datetime_range(
        matrix["issue_time"].min(), cutoff, interval="1m", eager=True
    )
    grid = pl.DataFrame({"valid_time": ticks, "temp_c": [10.0] * len(ticks)})
    for moment, reason in [
        (cutoff, "awaiting_truth"),
        (cutoff + timedelta(hours=2), None),
    ]:
        diagnostics = BacktestDiagnostics(snapshot_at=moment)
        batches = list(
            minutely.iter_minutely_scores(
                matrix,
                grid,
                minutely.MinutelyRequest(variables=("temp_c",)),
                config,
                diagnostics=diagnostics,
            )
        )
        assert batches == []
        assert diagnostics.pending_reason == reason


@pytest.mark.parametrize("method", ["persistence", "anchored_grounded_equal_weight"])
def test_hourly_only_methods_skip_daily_without_requiring_matrix(
    tmp_path, method, monkeypatch
):
    config = write_config(tmp_path)
    config.dataset.dir.mkdir()
    synthetic_hourly_matrix(days=2).write_parquet(
        config.dataset.dir / "hourly_matrix_live.parquet"
    )
    monkeypatch.setattr(selection, "select_methods", lambda *_a, **_k: {})
    result = cli._cmd_backtest(config, request("hourly,daily", method))
    assert result.exit_code == 3
    assert not (config.dataset.dir / "daily_matrix_live.parquet").exists()


@pytest.mark.parametrize("order", list(permutations(["hourly", "daily", "minutely"])))
def test_all_products_attempted_and_only_completed_promoted(
    tmp_path, monkeypatch, order
):
    config = write_config(tmp_path)
    seen = []

    def run(_config, _args, product, *_methods):
        seen.append(product)
        return ProductBacktestOutcome(
            product,
            "failed" if product == "daily" else "evaluated",
            0 if product == "daily" else 5,
        )

    monkeypatch.setattr(cli, "_run_matrix_backtest", run)
    monkeypatch.setattr(
        cli, "_run_minutely_backtest", lambda cfg, args: run(cfg, args, "minutely")
    )
    promoted = []
    monkeypatch.setattr(
        selection,
        "select_methods",
        lambda *_a, **kw: promoted.append(kw["products"]) or {},
    )
    assert cli._cmd_backtest(config, request(",".join(order))).exit_code == 1
    assert seen == list(order)
    assert promoted == [frozenset({"hourly", "minutely"})]


def test_entirely_inapplicable_is_explicit_configuration_error(tmp_path, capsys):
    config = write_config(tmp_path)
    result = cli._cmd_backtest(config, request("minutely", source="synthetic"))
    assert result.exit_code == 2
    assert "no requested product is applicable" in capsys.readouterr().err


def test_some_scores_complete_product_with_other_future_targets(tmp_path, monkeypatch):
    config = write_config(tmp_path)
    config = replace(config, backtest=replace(config.backtest, initial_train_days=1))
    config.dataset.dir.mkdir()
    matrix = synthetic_hourly_matrix(days=4, max_lead=3)
    cutoff = matrix["issue_time"].max()
    matrix = matrix.with_columns(
        pl.when(hourly_truth_known_at(matrix) <= cutoff)
        .then(pl.col("t__temp_c__inst"))
        .otherwise(float("nan"))
        .alias("t__temp_c__inst")
    )
    matrix.write_parquet(config.dataset.dir / "hourly_matrix_live.parquet")
    publish_metadata(config, cutoff)
    monkeypatch.setattr(selection, "select_methods", lambda *_a, **_k: {})
    assert cli._cmd_backtest(config, request()).exit_code == 0
