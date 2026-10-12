from datetime import timedelta

import polars as pl
import pytest
from conftest import utc

from grounded_weather_forecast.backtest.scores import (
    SCORES_SCHEMA,
    load_scores,
    write_scores,
)
from grounded_weather_forecast.config import PromotionConfig
from grounded_weather_forecast.contracts import MixedProvenanceError, TruthSemantics
from grounded_weather_forecast.serve import score_scan
from grounded_weather_forecast.serve.evidence import (
    EvidenceSummary,
    evaluation_contexts,
)

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
    return score_scan.compatible_slices(directory, identity, PromotionConfig()).slices


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
    ("column", "value"),
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
    ("column", "value"),
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
        if path == second and counts[path] == 2:
            path.unlink()
        return real(path)

    monkeypatch.setattr(score_scan, "scan_scores", disappear)
    assert select(tmp_path)[KEY]["evaluation_id"][0] == "old"


def test_three_materialization_prunes_keep_the_oldest_complete_evaluation(
    tmp_path, monkeypatch
):
    paths = [tmp_path / f"scores_{index}.parquet" for index in range(4)]
    for index, path in enumerate(paths):
        write_scores(scores(str(index), index + 1), path)
    real = score_scan.scan_scores
    counts = {}
    removed = []

    def disappear(path):
        counts[path] = counts.get(path, 0) + 1
        assert counts[path] <= 2, "fallback did not invalidate a failed evaluation"
        if path != paths[0] and counts[path] == 2:
            path.unlink()
            removed.append(path)
        return real(path)

    monkeypatch.setattr(score_scan, "scan_scores", disappear)
    result = score_scan.compatible_slices(tmp_path, IDENTITY, PromotionConfig())
    assert result.slices[KEY]["evaluation_id"].unique().to_list() == ["0"]
    assert set(result.contexts) == {"0"}
    assert removed == list(reversed(paths[1:]))
    assert paths[0].exists()


def test_materialization_without_invalidation_progress_restarts_boundedly(
    tmp_path, monkeypatch
):
    write_scores(scores(), tmp_path / "scores_eval.parquet")
    catalogs = []

    def fail(catalog, *_args):
        catalogs.append(catalog)
        assert len(catalogs) <= 3, "unidentified changes caused an unbounded retry"
        raise score_scan.EvidenceChangedError("unidentified change")

    monkeypatch.setattr(score_scan, "_materialize", fail)
    assert select(tmp_path) == {}
    assert len(catalogs) == score_scan.MAX_SCORE_SCAN_ATTEMPTS == 3
    assert len({id(catalog) for catalog in catalogs}) == 3


def test_known_materialization_exclusions_survive_a_fresh_catalog(
    tmp_path, monkeypatch
):
    write_scores(scores("old"), tmp_path / "scores_old.parquet")
    write_scores(scores("new", 2), tmp_path / "scores_new.parquet")
    real = score_scan._materialize
    seen = []

    def change(catalog, selected, identity):
        seen.append({evidence.evaluation_id for evidence in selected.values()})
        if len(seen) == 1:
            raise score_scan.EvidenceChangedError("changed", frozenset({"new"}))
        if len(seen) == 2:
            raise score_scan.EvidenceChangedError("unknown change")
        assert len(seen) == 3
        return real(catalog, selected, identity)

    monkeypatch.setattr(score_scan, "_materialize", change)
    assert select(tmp_path)[KEY]["evaluation_id"][0] == "old"
    assert seen == [{"new"}, {"old"}, {"old"}]


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


def test_large_repeated_context_is_compact_without_changing_score_values(tmp_path):
    description = '["' + "feature" * 1_200 + '"]'
    frame = pl.concat([scores()] * 1_024).with_columns(
        pl.lit(description).alias("feature_set_json")
    )
    write_scores(frame, tmp_path / "scores_large.parquet")

    actual = select(tmp_path)[KEY]

    assert actual.height == frame.height
    assert actual["feature_set_json"].unique().to_list() == [description]
    assert actual["feature_set_json"].estimated_size() < actual.height * 8
    assert actual.select("issue_time", "y_pred", "y_true").equals(
        frame.select("issue_time", "y_pred", "y_true")
    )


@pytest.mark.parametrize("compact", [False, True])
def test_loader_types_values_and_provenance(tmp_path, compact):
    frame = scores()
    path = tmp_path / "scores_eval.parquet"
    write_scores(frame, path)
    assert load_scores(path).schema == SCORES_SCHEMA
    loaded = load_scores(path, compact=compact)
    assert loaded.cast(SCORES_SCHEMA).equals(frame)
    if not compact:
        assert loaded.schema == SCORES_SCHEMA
        assert pl.concat([frame, loaded]).height == 2 * frame.height
        assert (
            loaded.join(
                pl.DataFrame({"method_id": ["equal_weight"]}), on="method_id"
            ).height
            == 8
        )
    else:
        assert loaded.schema["method_id"] == pl.Categorical
        assert loaded.schema["quantiles_json"] == pl.String
    write_scores(
        pl.concat(
            [frame, frame.with_columns(pl.lit("synthetic").alias("source_kind"))]
        ),
        path,
    )
    with pytest.raises(MixedProvenanceError):
        load_scores(path, compact=compact)
    assert (
        load_scores(path, compact=compact, allow_mixed=True).height == 2 * frame.height
    )


def test_null_semantics_are_instantaneous_in_both_scan_passes(tmp_path):
    frame = scores()
    frame = pl.concat(
        [
            frame,
            frame.head(1).with_columns(
                pl.lit(None, dtype=pl.String).alias("semantics")
            ),
        ]
    )
    write_scores(frame, tmp_path / "scores_eval.parquet")
    identity = score_scan.ScoreIdentity(
        "dataset", "config", "code", semantics={"temp_c": TruthSemantics.INSTANTANEOUS}
    )
    selected = select(tmp_path, identity)[KEY]
    assert selected.height == frame.height
    assert selected["semantics"].unique().to_list() == ["inst"]


def test_two_catalog_disappearances_recover_from_fresh_listing(tmp_path, monkeypatch):
    paths = [tmp_path / f"scores_{name}.parquet" for name in ("a", "b", "c")]
    for index, path in enumerate(paths):
        write_scores(scores(str(index), index + 1), path)
    real = score_scan.scan_scores
    vanished = []

    def scan(path):
        if path in paths[:2] and path not in vanished:
            vanished.append(path)
            path.unlink()
        return real(path)

    monkeypatch.setattr(score_scan, "scan_scores", scan)
    assert select(tmp_path)[KEY]["evaluation_id"][0] == "2"
    assert len(vanished) == 2


def test_persistent_catalog_failure_exhausts_exact_retry_budget(tmp_path, monkeypatch):
    path = tmp_path / "scores_arbitrary.parquet"
    write_scores(scores(), path)
    calls = []

    def missing(path):
        calls.append(path)
        raise FileNotFoundError(path)

    monkeypatch.setattr(score_scan, "scan_scores", missing)
    assert select(tmp_path) == {}
    assert len(calls) == score_scan.MAX_SCORE_SCAN_ATTEMPTS == 3


def test_reused_contexts_and_cutoffs_only_attribute_selected_variables(tmp_path):
    original = scores()
    unrelated = original.with_columns(
        pl.lit("humidity_pct").alias("variable"),
        pl.col("valid_time") + timedelta(days=100),
    )
    write_scores(pl.concat([original, unrelated]), tmp_path / "scores_eval.parquet")
    result = score_scan.compatible_slices(tmp_path, IDENTITY, PromotionConfig())
    evidence = EvidenceSummary.from_contexts(
        result.contexts,
        result.cutoffs,
        {KEY: "eval", ("hourly", "humidity_pct", "1-3h"): None},
    )
    assert evidence.contexts == evaluation_contexts([original])
    assert evidence.evaluation_ids == ("eval",)
    assert evidence.training_cutoff == original["valid_time"].max()


def test_missing_multi_evaluation_file_excludes_all_affected_slices(
    tmp_path, monkeypatch
):
    second_key = ("hourly", "temp_c", "3-6h")
    humidity_key = ("hourly", "humidity_pct", "1-3h")
    old = scores("old")
    write_scores(
        pl.concat(
            [
                old,
                old.with_columns(pl.lit("3-6h").alias("lead_bucket")),
                old.with_columns(pl.lit("humidity_pct").alias("variable")),
            ]
        ),
        tmp_path / "scores_old.parquet",
    )
    disappearing = tmp_path / "scores_unrelated_name.parquet"
    write_scores(
        pl.concat(
            [
                scores("first", 2),
                scores("second", 2).with_columns(pl.lit("3-6h").alias("lead_bucket")),
            ]
        ),
        disappearing,
    )
    write_scores(
        scores("first", 2).with_columns(pl.lit("humidity_pct").alias("variable")),
        tmp_path / "scores_surviving_fragment.parquet",
    )
    real = score_scan.scan_scores
    calls = 0

    def scan(path):
        nonlocal calls
        if path == disappearing:
            calls += 1
            if calls == 2:
                path.unlink()
        return real(path)

    monkeypatch.setattr(score_scan, "scan_scores", scan)
    result = select(tmp_path)
    assert set(result) == {KEY, second_key, humidity_key}
    assert all(frame["evaluation_id"][0] == "old" for frame in result.values())
