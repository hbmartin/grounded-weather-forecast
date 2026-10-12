import json
import re
import weakref
from dataclasses import fields, replace
from datetime import timedelta

import polars as pl
import pytest
from conftest import synthetic_hourly_matrix, utc, write_config
from score_fixtures import scores

from grounded_weather_forecast.backtest import scores as score_module
from grounded_weather_forecast.backtest.scores import load_scores, write_scores
from grounded_weather_forecast.cli import _cmd_report
from grounded_weather_forecast.dashboard import (
    context,
    score_summary,
    write_dashboard,
    writer,
)
from grounded_weather_forecast.dashboard.derive import derive
from grounded_weather_forecast.dashboard.zones import data_trust, evaluation, readiness
from grounded_weather_forecast.dataset.matrix import matrix_path
from grounded_weather_forecast.reports import render


def assert_same_summary(actual, expected):
    for field in fields(expected):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(right, pl.DataFrame):
            assert left.equals(right), field.name
        else:
            assert left == right, field.name


def dashboard_charts(path):
    match = re.search(
        r'<script[^>]*id="dashboard-data"[^>]*>(.*?)</script>',
        path.read_text(),
        re.DOTALL,
    )
    return json.loads(match.group(1))["charts"]


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


def test_file_disappearing_during_collection_is_absent(tmp_path, monkeypatch):
    config, paths = score_archive(tmp_path, count=1)

    def load(path, **kwargs):
        path.unlink()
        return load_scores(path, **kwargs)

    monkeypatch.setattr(context, "load_scores", load)
    collected = context.collect_context(config)
    assert collected.score_summaries == {}
    assert collected.unreadable_scores == ()
    assert f"dataset/scores/{paths[0].name}" not in collected.unreadable_artifacts
    assert not derive(collected).live_scores_unusable


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


@pytest.mark.parametrize("provided", [("board",), ("winners",), ("board", "winners")])
def test_precomputed_statistics_bypass_their_derivation(
    tmp_path, monkeypatch, provided
):
    config = write_config(tmp_path)
    frame = scores()
    expected = score_summary.summarize_scores(config, "scores_hourly_live", frame, {})

    def unexpected(*_args, **_kwargs):
        pytest.fail("supplied statistics were recomputed")

    if "board" in provided:
        monkeypatch.setattr(score_summary, "leaderboard", unexpected)
    if "winners" in provided:
        monkeypatch.setattr(score_summary, "slice_winners", unexpected)
        monkeypatch.setattr(score_summary, "winner_curse_adjusted", unexpected)
    actual = score_summary.summarize_scores(
        config,
        "scores_hourly_live",
        frame,
        {},
        **{name: getattr(expected, name) for name in provided},
    )
    assert_same_summary(actual, expected)


@pytest.mark.parametrize("empty", [False, True])
def test_supplied_snapshot_bypasses_score_reads_but_collects_other_artifacts(
    tmp_path, monkeypatch, empty
):
    config, paths = score_archive(tmp_path, count=1)
    snapshot = {} if empty else dict(context.collect_context(config).score_summaries)
    if not empty:
        paths[0].unlink()
    (config.dataset.dir / "manifest.json").write_text('{"fingerprint": "collected"}')

    def unexpected(*_args, **_kwargs):
        pytest.fail("a supplied score snapshot reopened score files")

    monkeypatch.setattr(context, "load_scores", unexpected)
    collected = context.collect_context(config, score_summaries=snapshot)
    assert collected.manifest == {"fingerprint": "collected"}
    assert collected.unreadable_scores == ()
    assert set(collected.score_summaries) == set(snapshot)
    snapshot.clear()
    assert bool(collected.score_summaries) == (not empty)
    text = writer.write_dashboard(
        config, score_summaries=collected.score_summaries
    ).read_text()
    assert ("Leaderboard — hourly_live_expanding_eval0" in text) == (not empty)


@pytest.mark.parametrize("rule", ["mcs", "legacy"])
@pytest.mark.parametrize("scenario", ["live", "synthetic", "pooled_daily", "empty"])
def test_report_reuses_equivalent_summaries_and_releases_each_frame(
    tmp_path, monkeypatch, rule, scenario
):
    config, paths = score_archive(tmp_path, count=2)
    config = replace(config, promotion=replace(config.promotion, rule=rule))
    matrix_path(config.dataset.dir, "hourly", "live").unlink()
    for path in paths:
        frame = load_scores(path)
        if scenario == "synthetic":
            frame = frame.with_columns(pl.lit("synthetic").alias("source_kind"))
            destination = path.with_name(path.name.replace("_live_", "_synthetic_"))
            path.unlink()
        elif scenario == "pooled_daily":
            frame = frame.with_columns(
                pl.lit("daily").alias("product"),
                pl.lit("temp_max_c").alias("variable"),
                pl.when(pl.col("lead_hours") == 1.0)
                .then(
                    pl.when(pl.col("valid_time").dt.hour() <= 4)
                    .then(pl.lit("D3-4"))
                    .otherwise(pl.lit("D5-7"))
                )
                .otherwise(pl.col("lead_bucket"))
                .alias("lead_bucket"),
            )
            destination = path.with_name(path.name.replace("_hourly_", "_daily_"))
            path.unlink()
        else:
            destination = path
            if scenario == "empty":
                frame = frame.clear()
        write_scores(frame, destination)
    paths = sorted((config.dataset.dir / "scores").glob("*.parquet"))
    decoded, calls, snapshots = [], [], []

    def load(path, **kwargs):
        assert all(reference() is None for reference in decoded)
        frame = load_scores(path, **kwargs)
        calls.append(path)
        decoded.append(weakref.ref(frame))
        return frame

    def dashboard(config, *, score_summaries, score_read_failures=()):
        assert score_read_failures == ()
        snapshots.append(score_summaries)
        return writer.write_dashboard(
            config, now=utc(2026, 8, 5), score_summaries=score_summaries
        )

    monkeypatch.setattr(score_module, "load_scores", load)
    monkeypatch.setattr(context, "load_scores", load)
    monkeypatch.setattr(
        "grounded_weather_forecast.dashboard.write_dashboard", dashboard
    )
    with monkeypatch.context() as guard:

        def unexpected(*_args, **_kwargs):
            pytest.fail("dashboard recomputed report statistics")

        for stage in ("leaderboard", "slice_winners", "winner_curse_adjusted"):
            guard.setattr(score_summary, stage, unexpected)
        assert _cmd_report(config) == 0
    assert calls == paths
    assert all(reference() is None for reference in decoded)
    assert set(snapshots[0]) == {path.stem for path in paths}
    cached_charts = dashboard_charts(config.reports_dir / "dashboard.html")
    independent = context.collect_context(config, now=utc(2026, 8, 5))
    for stem, summary in snapshots[0].items():
        assert_same_summary(summary, independent.score_summaries[stem])
        if scenario == "pooled_daily" and rule == "mcs":
            assert summary.winners.height == 2
            assert summary.winners["gate"].unique().to_list() == ["pooled_D3-10"]
        if summary.board is not None:
            assert not any(
                column.startswith(("bh_", "ebh_")) for column in summary.board.columns
            )
    assert cached_charts == dashboard_charts(
        write_dashboard(config, now=utc(2026, 8, 5))
    )
    for path in paths:
        path.unlink()
    assert (
        dashboard_charts(
            writer.write_dashboard(
                config, now=utc(2026, 8, 5), score_summaries=snapshots[0]
            )
        )
        == cached_charts
    )


def test_sequential_report_dashboard_uses_final_persisted_wealth(tmp_path, monkeypatch):
    config, paths = score_archive(tmp_path, count=2)
    config = replace(
        config, promotion=replace(config.promotion, rule="seq_mcs", alpha=0.01)
    )
    matrix_path(config.dataset.dir, "hourly", "live").unlink()
    old = scores("old").with_columns(
        pl.when(pl.col("method_id") == "challenger")
        .then(1.0)
        .otherwise(4.0)
        .alias("y_pred")
    )
    newer = pl.concat(
        [
            old.with_columns(
                pl.lit("newer").alias("evaluation_id"),
                pl.lit(utc(2026, 8, 2)).alias("evaluation_created_at"),
                pl.col("issue_time") + timedelta(hours=8 + index * 8),
                pl.col("valid_time") + timedelta(hours=8 + index * 8),
            )
            for index in range(8)
        ]
    )
    write_scores(old, paths[0])
    write_scores(newer, paths[1])
    report_winners, snapshots, calls = {}, [], []
    real_report = render.write_markdown_report

    def report(directory, name, title, sections):
        for label, frame in sections:
            if label == "Per-slice winners":
                report_winners[name] = frame
        return real_report(directory, name, title, sections)

    def dashboard(config, *, score_summaries, score_read_failures=()):
        assert score_read_failures == ()
        snapshots.append(score_summaries)
        return writer.write_dashboard(config, score_summaries=score_summaries)

    def load(path, **kwargs):
        calls.append(path)
        return load_scores(path, **kwargs)

    monkeypatch.setattr(render, "write_markdown_report", report)
    monkeypatch.setattr(
        "grounded_weather_forecast.dashboard.write_dashboard", dashboard
    )
    monkeypatch.setattr(score_module, "load_scores", load)
    monkeypatch.setattr(context, "load_scores", load)
    assert _cmd_report(config) == 0
    assert snapshots == [None]
    assert calls == paths + paths
    assert (
        report_winners[paths[0].stem.replace("scores_", "leaderboard_")]["method_id"][0]
        != "challenger"
    )
    wealth_path = config.artifacts_dir / "eprocess" / "hourly_live.json"
    wealth = json.loads(wealth_path.read_text())
    final = context.collect_context(config).score_summaries[paths[0].stem]
    assert final.winners["method_id"][0] == "challenger"
    assert json.loads(wealth_path.read_text()) == wealth
    history = {
        path: pl.read_parquet(path)
        for path in (config.artifacts_dir / "history").glob("*.parquet")
    }
    assert history
    assert _cmd_report(config) == 0
    assert json.loads(wealth_path.read_text()) == wealth
    assert set((config.artifacts_dir / "history").glob("*.parquet")) == set(history)
    for path, frame in history.items():
        current = pl.read_parquet(path)
        # Final wealth can add verdicts for the earlier evaluation's new winner.
        # Every existing evidence row is retained without being rewritten.
        assert frame.join(
            current, on=frame.columns, how="anti", nulls_equal=True
        ).is_empty(), path.name
        if path.name != "verdicts.parquet":
            assert current.equals(frame), path.name
        history[path] = current
    report_tables = {
        path: path.read_text() for path in config.reports_dir.glob("leaderboard_*.md")
    }
    assert _cmd_report(config) == 0
    assert json.loads(wealth_path.read_text()) == wealth
    for path, frame in history.items():
        assert pl.read_parquet(path).equals(frame), path.name
    for path, markdown in report_tables.items():
        assert path.read_text() == markdown, path.name


@pytest.mark.parametrize("rule", ["mcs", "legacy", "seq_mcs"])
@pytest.mark.parametrize("fault", ["corrupt", "mixed", "permission"])
def test_report_preserves_skipped_score_faults_in_dashboard(
    tmp_path, monkeypatch, rule, fault, capsys
):
    config, paths = score_archive(tmp_path, count=2)
    config = replace(config, promotion=replace(config.promotion, rule=rule))
    matrix_path(config.dataset.dir, "hourly", "live").unlink()
    calls, collected = [], []
    if fault == "corrupt":
        paths[0].write_bytes(b"not parquet")
    elif fault == "mixed":
        frame = load_scores(paths[0]).with_columns(
            pl.when(pl.col("method_id") == "challenger")
            .then(pl.lit("synthetic"))
            .otherwise(pl.lit("live"))
            .alias("source_kind")
        )
        frame.write_parquet(paths[0])

    def load(path, **kwargs):
        calls.append(path)
        if path == paths[0] and fault == "permission":
            raise PermissionError(f"permission denied: {path}")
        return load_scores(path, **kwargs)

    def dashboard(config, *, score_summaries, score_read_failures):
        assert score_read_failures == (paths[0].name,)
        collected.append(
            context.collect_context(
                config,
                score_summaries=score_summaries,
                score_read_failures=score_read_failures,
            )
        )
        return writer.write_dashboard(
            config,
            score_summaries=score_summaries,
            score_read_failures=score_read_failures,
        )

    monkeypatch.setattr(score_module, "load_scores", load)
    monkeypatch.setattr(context, "load_scores", load)
    monkeypatch.setattr(
        "grounded_weather_forecast.dashboard.write_dashboard", dashboard
    )
    independent = context.collect_context(config)
    calls.clear()
    assert _cmd_report(config) == 1
    expected_reads = 3 if rule == "seq_mcs" else 1
    assert calls.count(paths[1]) == expected_reads
    actual = collected[0]
    assert actual.unreadable_scores == independent.unreadable_scores == (paths[0].stem,)
    label = f"dataset/scores/{paths[0].name}"
    assert actual.unreadable_artifacts.count(label) == 1
    assert label in independent.unreadable_artifacts
    assert derive(actual).live_scores_unusable
    assert paths[0].name in (config.reports_dir / "dashboard.html").read_text()
    assert paths[0].name in capsys.readouterr().err
    assert (
        (config.reports_dir / paths[1].stem.replace("scores_", "leaderboard_"))
        .with_suffix(".md")
        .exists()
    )


def test_empty_supplied_snapshot_keeps_failures_without_reading_scores(
    tmp_path, monkeypatch
):
    config, paths = score_archive(tmp_path, count=1)

    def unexpected(*_args, **_kwargs):
        pytest.fail("supplied snapshot reopened score files")

    monkeypatch.setattr(context, "load_scores", unexpected)
    failures = (paths[0].name, paths[0].name)
    actual = context.collect_context(
        config, score_summaries={}, score_read_failures=failures
    )
    assert actual.score_summaries == {}
    assert actual.unreadable_scores == (paths[0].stem,)
    assert actual.unreadable_artifacts.count(f"dataset/scores/{paths[0].name}") == 1
    assert derive(actual).live_scores_unusable


@pytest.mark.parametrize("rule", ["mcs", "legacy", "seq_mcs"])
@pytest.mark.parametrize("all_missing", [False, True])
def test_report_vanished_scores_are_absent(tmp_path, monkeypatch, rule, all_missing):
    config, paths = score_archive(tmp_path, count=1 if all_missing else 2)
    config = replace(config, promotion=replace(config.promotion, rule=rule))
    matrix_path(config.dataset.dir, "hourly", "live").unlink()
    snapshots = []

    def load(path, **kwargs):
        if path == paths[0]:
            path.unlink()
        return load_scores(path, **kwargs)

    def dashboard(config, *, score_summaries, score_read_failures):
        assert score_read_failures == ()
        snapshots.append(
            context.collect_context(config, score_summaries=score_summaries)
        )
        return writer.write_dashboard(config, score_summaries=score_summaries)

    monkeypatch.setattr(score_module, "load_scores", load)
    monkeypatch.setattr(
        "grounded_weather_forecast.dashboard.write_dashboard", dashboard
    )
    assert _cmd_report(config) == (1 if all_missing else 0)
    assert snapshots[0].unreadable_scores == ()
    assert not derive(snapshots[0]).live_scores_unusable
    assert (config.reports_dir / "dashboard.html").exists()


@pytest.mark.parametrize("rule", ["mcs", "legacy", "seq_mcs"])
def test_report_all_faults_writes_empty_snapshot_and_returns_failure(tmp_path, rule):
    config, paths = score_archive(tmp_path, count=1)
    config = replace(config, promotion=replace(config.promotion, rule=rule))
    matrix_path(config.dataset.dir, "hourly", "live").unlink()
    paths[0].write_bytes(b"corrupt parquet")
    assert _cmd_report(config) == 1
    assert paths[0].name in (config.reports_dir / "dashboard.html").read_text()


def test_recorded_fault_survives_later_file_removal(tmp_path):
    config, paths = score_archive(tmp_path, count=1)
    paths[0].unlink()
    collected = context.collect_context(
        config,
        score_summaries={},
        score_read_failures=(paths[0].name,),
    )
    assert collected.unreadable_scores == (paths[0].stem,)
    assert derive(collected).live_scores_unusable


@pytest.mark.parametrize("rule", ["mcs", "legacy", "seq_mcs"])
def test_report_readable_empty_scores_count_as_success(tmp_path, rule):
    from grounded_weather_forecast.backtest.scores import empty_scores

    config, paths = score_archive(tmp_path, count=1)
    config = replace(config, promotion=replace(config.promotion, rule=rule))
    matrix_path(config.dataset.dir, "hourly", "live").unlink()
    empty_scores().write_parquet(paths[0])
    assert _cmd_report(config) == 0
    assert (config.reports_dir / "dashboard.html").exists()
