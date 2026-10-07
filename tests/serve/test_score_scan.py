from datetime import timedelta

import polars as pl
import pytest
from conftest import utc

from grounded_weather_forecast.backtest.scores import SCORES_SCHEMA, write_scores
from grounded_weather_forecast.config import PromotionConfig
from grounded_weather_forecast.contracts import MixedProvenanceError, TruthSemantics
from grounded_weather_forecast.serve import score_scan

IDENTITY = score_scan.ScoreIdentity("dataset", "config", "code")
KEY = ("hourly", "temp_c", "1-3h")


def scores(
    evaluation="eval",
    day=1,
    methods=(
        "equal_weight",
        "best_provider",
        "damped_grounded_equal_weight",
        "challenger",
    ),
):
    rows = []
    for method in methods:
        for step in range(8):
            rows.append(
                {
                    "method_id": method,
                    "variable": "temp_c",
                    "product": "hourly",
                    "source_kind": "live",
                    "evaluation_id": evaluation,
                    "evaluation_created_at": utc(2026, 8, day),
                    "dataset_fingerprint": "dataset",
                    "config_fingerprint": "config",
                    "code_version": "code",
                    "source_set_json": '["nws"]',
                    "feature_set_json": '["feature"]',
                    "semantics": "inst",
                    "window": "expanding",
                    "fold_origin": utc(2026, 7, 1),
                    "issue_time": utc(2026, 7, 1) + timedelta(hours=step),
                    "valid_time": utc(2026, 7, 1, 1) + timedelta(hours=step),
                    "lead_hours": 1.0,
                    "lead_bucket": "1-3h",
                    "y_pred": 1.0,
                    "y_true": 0.0,
                    "quantile_levels_json": "[]",
                    "quantiles_json": None,
                }
            )
    return pl.DataFrame(rows, schema=SCORES_SCHEMA)


def select(directory, identity=IDENTITY):
    return score_scan.compatible_slices(directory, identity, PromotionConfig())


def test_evaluation_split_across_files_is_atomic(tmp_path):
    frame = scores()
    write_scores(
        frame.filter(pl.col("method_id").is_in(["equal_weight", "challenger"])),
        tmp_path / "scores_a.parquet",
    )
    write_scores(
        frame.filter(~pl.col("method_id").is_in(["equal_weight", "challenger"])),
        tmp_path / "scores_b.parquet",
    )
    actual = select(tmp_path)[KEY]
    assert actual.height == frame.height
    assert "fold_origin" not in actual.columns
    assert "quantiles_json" not in actual.columns
    assert set(actual["method_id"]) == set(frame["method_id"])


def test_multiple_evaluations_per_file_and_equal_time_order(tmp_path):
    frame = pl.concat(
        [scores("a"), scores("b"), scores("new-incomplete", 2, ("challenger",))]
    )
    write_scores(frame, tmp_path / "scores_mixed.parquet")
    assert select(tmp_path)[KEY]["evaluation_id"].unique().to_list() == ["b"]


@pytest.mark.parametrize(
    "column,value",
    [
        ("semantics", "mean"),
        ("source_set_json", '["other"]'),
        ("window", "rolling"),
        ("code_version", "other-code"),
        ("dataset_fingerprint", "other-dataset"),
        ("evaluation_created_at", utc(2026, 8, 3)),
    ],
)
def test_conflicting_evaluation_falls_back_as_a_whole(tmp_path, column, value, caplog):
    write_scores(scores("old"), tmp_path / "scores_old.parquet")
    current = scores("new", 2)
    write_scores(current, tmp_path / "scores_current.parquet")
    write_scores(
        current.head(1).with_columns(pl.lit(value).alias(column)),
        tmp_path / "scores_conflict.parquet",
    )
    assert select(tmp_path)[KEY]["evaluation_id"].unique().to_list() == ["old"]
    assert "conflicting evaluation evidence: new" in caplog.text


def test_conflict_in_one_frame_without_older_evidence_fails_closed(tmp_path):
    frame = scores()
    write_scores(
        pl.concat(
            [frame, frame.head(1).with_columns(pl.lit("mean").alias("semantics"))]
        ),
        tmp_path / "scores_conflict.parquet",
    )
    assert select(tmp_path) == {}


def test_source_provenance_is_checked_before_identity_filter(tmp_path):
    frame = scores()
    other = frame.head(1).with_columns(
        pl.lit("synthetic").alias("source_kind"),
        pl.lit("other").alias("config_fingerprint"),
    )
    write_scores(pl.concat([frame, other]), tmp_path / "scores_mixed.parquet")
    with pytest.raises(MixedProvenanceError):
        select(tmp_path)


def test_targeted_evaluation_preserves_other_slice(tmp_path):
    original = scores("old")
    other = original.with_columns(pl.lit("3-6h").alias("lead_bucket"))
    write_scores(pl.concat([original, other]), tmp_path / "scores_old.parquet")
    write_scores(scores("new", 2), tmp_path / "scores_new.parquet")
    slices = select(tmp_path)
    assert slices[KEY]["evaluation_id"][0] == "new"
    assert slices[("hourly", "temp_c", "3-6h")]["evaluation_id"][0] == "old"


def test_requested_semantics_do_not_hide_a_conflict(tmp_path):
    frame = scores()
    write_scores(
        pl.concat(
            [frame, frame.head(1).with_columns(pl.lit("mean").alias("semantics"))]
        ),
        tmp_path / "scores_conflict.parquet",
    )
    identity = score_scan.ScoreIdentity(
        "dataset", "config", "code", semantics={"temp_c": TruthSemantics.INSTANTANEOUS}
    )
    assert select(tmp_path, identity) == {}


def test_vanished_selected_fragment_cannot_mint_partial_evaluation(
    tmp_path, monkeypatch
):
    write_scores(scores("old"), tmp_path / "scores_old.parquet")
    current = scores("new", 2)
    write_scores(current, tmp_path / "scores_new_a.parquet")
    disappearing = tmp_path / "scores_new_b.parquet"
    write_scores(current.head(1), disappearing)
    real = score_scan.scan_scores
    calls = 0

    def vanish(path):
        nonlocal calls
        if path == disappearing:
            calls += 1
            if calls == 2:
                path.unlink()
        return real(path)

    monkeypatch.setattr(score_scan, "scan_scores", vanish)
    assert select(tmp_path)[KEY]["evaluation_id"][0] == "old"


@pytest.mark.parametrize(
    "column,value",
    [("feature_set_json", '["changed"]'), ("method_id", "challenger")],
)
def test_replaced_fragment_retries_without_mixing_evidence(
    tmp_path, monkeypatch, column, value
):
    write_scores(scores("old"), tmp_path / "scores_old.parquet")
    path = tmp_path / "scores_new.parquet"
    write_scores(scores("new", 2), path)
    real = score_scan.scan_scores
    calls = 0

    def change(candidate):
        nonlocal calls
        if candidate == path:
            calls += 1
            if calls == 2:
                write_scores(
                    scores("new", 2).with_columns(pl.lit(value).alias(column)),
                    path,
                )
        return real(candidate)

    monkeypatch.setattr(score_scan, "scan_scores", change)
    assert select(tmp_path)[KEY]["evaluation_id"][0] == "old"


def test_second_pass_prune_excludes_evaluation_and_reconsiders_older(
    tmp_path, monkeypatch
):
    first = tmp_path / "scores_first.parquet"
    second = tmp_path / "scores_second.parquet"
    write_scores(scores("newest", 3), first)
    write_scores(scores("next", 2), second)
    write_scores(scores("old"), tmp_path / "scores_old.parquet")
    real = score_scan.scan_scores
    counts = {}

    def disappear(path):
        counts[path] = counts.get(path, 0) + 1
        if path == first and counts[path] == 2:
            raise FileNotFoundError(path)
        if path == second and counts[path] == 3:
            path.unlink()
        return real(path)

    monkeypatch.setattr(score_scan, "scan_scores", disappear)
    assert select(tmp_path)[KEY]["evaluation_id"][0] == "old"


def test_identity_diagnosis_accepts_legacy_columns(tmp_path):
    frame = scores().select("source_kind", "dataset_fingerprint")
    write_scores(frame, tmp_path / "scores_legacy.parquet")
    assert score_scan.live_identity_sets(tmp_path, skip_missing=False) == (
        True,
        {"dataset"},
        set(),
        set(),
    )
    assert select(tmp_path) == {}
