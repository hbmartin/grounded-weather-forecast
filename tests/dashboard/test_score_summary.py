import weakref

import polars as pl
import pytest
from conftest import synthetic_hourly_matrix, write_config
from serve.test_score_scan import scores

from grounded_weather_forecast.backtest.scores import load_scores, write_scores
from grounded_weather_forecast.dashboard import context, score_summary, writer
from grounded_weather_forecast.dashboard.derive import derive
from grounded_weather_forecast.dashboard.zones import data_trust, evaluation, readiness
from grounded_weather_forecast.dataset.matrix import matrix_path


def score_archive(tmp_path, count=3):
    config = write_config(tmp_path)
    directory = config.dataset.dir / "scores"
    directory.mkdir(parents=True)
    paths = []
    for index in range(count):
        path = directory / f"scores_hourly_live_expanding_eval{index}.parquet"
        write_scores(scores(str(index), index + 1), path)
        paths.append(path)
    synthetic_hourly_matrix(days=5).write_parquet(
        matrix_path(config.dataset.dir, "hourly", "live")
    )
    return config, paths


def test_dashboard_decodes_once_and_releases_each_full_frame(tmp_path, monkeypatch):
    config, paths = score_archive(tmp_path)
    decoded = []
    calls = []

    def load(path, **kwargs):
        assert all(reference() is None for reference in decoded)
        frame = load_scores(path, **kwargs)
        calls.append(path)
        decoded.append(weakref.ref(frame))
        return frame

    monkeypatch.setattr(context, "load_scores", load)
    destination = writer.write_dashboard(config)
    assert destination.exists()
    assert calls == sorted(paths)
    assert all(reference() is None for reference in decoded)


def test_collected_dashboard_renders_after_score_files_disappear(tmp_path, monkeypatch):
    config, paths = score_archive(tmp_path, count=1)
    collected = context.collect_context(config)
    paths[0].unlink()
    monkeypatch.setattr(writer, "collect_context", lambda *_args, **_kwargs: collected)
    text = writer.write_dashboard(config).read_text()
    assert "Leaderboard — hourly_live_expanding_eval0" in text
    assert readiness._fold_count(collected) == 1
    assert data_trust._provenance(collected).status == "ok"


def test_file_disappearing_during_collection_is_reported(tmp_path, monkeypatch):
    config, paths = score_archive(tmp_path, count=1)

    def load(path, **kwargs):
        path.unlink()
        return load_scores(path, **kwargs)

    monkeypatch.setattr(context, "load_scores", load)
    collected = context.collect_context(config)
    assert collected.score_summaries == {}
    assert collected.unreadable_scores == (paths[0].stem,)
    assert f"dataset/scores/{paths[0].name}" in collected.unreadable_artifacts
    assert derive(collected).live_scores_unusable


@pytest.mark.parametrize(
    "stage", ["leaderboard", "slice_winners", "winner_curse_adjusted"]
)
def test_statistical_failure_retains_provenance_and_fold_metadata(
    tmp_path, monkeypatch, stage
):
    config, paths = score_archive(tmp_path, count=1)

    def fail(*_args, **_kwargs):
        raise ValueError("invalid statistical evidence")

    monkeypatch.setattr(score_summary, stage, fail)
    collected = context.collect_context(config)
    summary = collected.score_summaries[paths[0].stem]
    assert summary.unusable
    assert summary.board is None
    assert summary.source_kinds == {"live"}
    assert readiness._fold_count(collected) == 1
    assert data_trust._provenance(collected).status == "ok"
    derived = derive(collected)
    assert derived.boards == {}
    assert derived.live_scores_unusable


def test_probabilistic_summaries_preserve_histogram_and_reliability(tmp_path):
    config, paths = score_archive(tmp_path, count=1)
    frame = scores().with_columns(
        pl.lit("pop").alias("variable"),
        pl.lit(0.5).alias("y_pred"),
        pl.lit(1.0).alias("y_true"),
        pl.lit("[0.1, 0.5, 0.9]").alias("quantile_levels_json"),
        pl.lit("[0.0, 1.0, 2.0]").alias("quantiles_json"),
    )
    write_scores(frame, paths[0])
    collected = context.collect_context(config)
    summary = collected.score_summaries[paths[0].stem]
    assert not summary.unusable
    assert summary.pit_rows == 32
    assert summary.pit_counts == (0, 0, 0, 0, 0, 32, 0, 0, 0, 0)
    assert summary.reliability == ((0.5, 1.0, 32),)
    panel = evaluation._calibration_panel(paths[0].stem, summary, summary.board)
    assert panel.chart.config["data"]["datasets"][0]["data"] == list(summary.pit_counts)
    reliability = evaluation._reliability_panel(paths[0].stem, summary)
    assert reliability is not None
    assert reliability.status == "ok"


def test_ragged_calibration_keeps_valid_point_scores_and_reliability(tmp_path):
    config, paths = score_archive(tmp_path, count=1)
    frame = (
        scores()
        .group_by("method_id", maintain_order=True)
        .head(7)
        .with_columns(
            pl.lit("pop").alias("variable"),
            pl.lit("[0.1, 0.5, 0.9]").alias("quantile_levels_json"),
            pl.when(pl.col("method_id") == "challenger")
            .then(pl.lit("[0.0, 1.0]"))
            .otherwise(pl.lit("[0.0, 1.0, 2.0]"))
            .alias("quantiles_json"),
        )
    )
    write_scores(frame, paths[0])
    collected = context.collect_context(config)
    summary = collected.score_summaries[paths[0].stem]
    assert summary.board.height == 4
    assert summary.winners is not None
    assert not summary.unusable
    assert summary.pit_unusable
    assert summary.reliability == ((1.0, 0.0, 28),)
    assert not summary.reliability_unusable
    derived = derive(collected)
    assert not derived.live_scores_unusable
    assert paths[0].stem in derived.boards
    text = writer.write_dashboard(config).read_text()
    assert "PIT calibration could not be calculated from this score file." in text
    assert "Leaderboard — hourly_live_expanding_eval0" in text


@pytest.mark.parametrize(
    "failed",
    [("pit_values",), ("reliability_summary",), ("pit_values", "reliability_summary")],
)
def test_calibration_failures_are_independent_local_warnings(
    tmp_path, monkeypatch, failed
):
    config, paths = score_archive(tmp_path, count=1)
    frame = scores().with_columns(
        pl.lit("pop").alias("variable"),
        pl.lit(0.5).alias("y_pred"),
        pl.lit(1.0).alias("y_true"),
        pl.lit("[0.1, 0.5, 0.9]").alias("quantile_levels_json"),
        pl.lit("[0.0, 1.0, 2.0]").alias("quantiles_json"),
    )
    write_scores(frame, paths[0])

    def fail(*_args):
        raise ValueError("calibration failed")

    for stage in failed:
        monkeypatch.setattr(score_summary, stage, fail)
    collected = context.collect_context(config)
    summary = collected.score_summaries[paths[0].stem]
    assert summary.board is not None
    assert summary.winners is not None
    assert not summary.unusable
    assert not derive(collected).live_scores_unusable
    assert summary.pit_unusable == ("pit_values" in failed)
    assert summary.reliability_unusable == ("reliability_summary" in failed)
    pit = evaluation._calibration_panel(paths[0].stem, summary, summary.board)
    assert pit.status == ("amber" if summary.pit_unusable else "ok")
    assert "best CRPS" in {stat.label for stat in pit.stats}
    assert (pit.chart is None) == summary.pit_unusable
    reliability = evaluation._reliability_panel(paths[0].stem, summary)
    assert reliability.status == ("amber" if summary.reliability_unusable else "ok")
    assert (reliability.chart is None) == summary.reliability_unusable
    if summary.reliability_unusable:
        assert (
            reliability.empty_reason
            == "PoP reliability could not be calculated from this score file."
        )
    else:
        assert summary.reliability == ((0.5, 1.0, 32),)
