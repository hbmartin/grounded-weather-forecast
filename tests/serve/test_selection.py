import json
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
from conftest import synthetic_hourly_matrix, utc, write_config

import grounded_weather_forecast.serve.selection as selection_module
from grounded_weather_forecast.backtest.engine import BacktestRequest, run_backtest
from grounded_weather_forecast.backtest.scores import (
    SCORES_SCHEMA,
    load_scores,
    scores_path,
    write_scores,
)
from grounded_weather_forecast.contracts import TruthSemantics, hourly_variable
from grounded_weather_forecast.serve.selection import (
    FALLBACK_METHOD,
    Selection,
    _eligible_release_ids,
    method_for,
    select_methods,
    selection_report,
)


def scored_config(tmp_path, extra=""):
    config = write_config(
        tmp_path,
        extra_toml="\n[backtest]\ninitial_train_days = 10\nstep_days = 5\n" + extra,
    )
    matrix = synthetic_hourly_matrix(days=25, biases={"alpha": 3.0})
    scores = run_backtest(
        matrix,
        BacktestRequest(
            variables=(hourly_variable("temp_c"),),
            methods=(
                "equal_weight",
                "grounded_equal_weight",
                "best_provider",
                "damped_grounded_equal_weight",
            ),
        ),
        config,
    )
    write_scores(scores, scores_path(config.dataset.dir / "scores", "hourly", "live"))
    return config


class TestSelectMethods:
    def test_winner_per_slice(self, tmp_path):
        config = scored_config(tmp_path)
        selections = select_methods(config, config.dataset.dir / "scores")
        assert selections
        for (product, variable, _bucket), chosen in selections.items():
            assert product == "hourly"
            assert variable == "temp_c"
            assert chosen.n > 0
            assert chosen.mae is not None
            assert "lowest backtest MAE" in chosen.reason
        # grounding removes alpha's +3C bias, so it must win somewhere
        assert any(c.method_id == "grounded_equal_weight" for c in selections.values())
        assert all(c.evaluation_id for c in selections.values())
        assert all(c.release_id for c in selections.values())
        assert list((config.artifacts_dir / "releases").glob("*.json"))

    def test_config_pin_overrides(self, tmp_path):
        config = scored_config(
            tmp_path,
            extra='\n[predict.methods]\n"hourly.temp_c" = "best_provider"\n',
        )
        selections = select_methods(config, config.dataset.dir / "scores")
        assert {c.method_id for c in selections.values()} == {"best_provider"}
        assert all(c.reason == "pinned in config" for c in selections.values())

    def test_report_frame(self, tmp_path):
        config = scored_config(tmp_path)
        report = selection_report(select_methods(config, config.dataset.dir / "scores"))
        assert set(report.columns) >= {
            "product",
            "variable",
            "lead_bucket",
            "method_id",
            "reason",
        }

    def test_no_scores_is_empty(self, tmp_path):
        config = write_config(tmp_path)
        assert select_methods(config, tmp_path / "nothing") == {}

    def test_historical_issue_rejects_future_evaluation(self, tmp_path):
        config = scored_config(tmp_path)
        as_of = utc(2026, 1, 1) - timedelta(days=1)
        assert select_methods(config, config.dataset.dir / "scores", as_of=as_of) == {}

    def test_historical_issue_loads_release_that_already_existed(self, tmp_path):
        config = scored_config(tmp_path)
        promoted = select_methods(config, config.dataset.dir / "scores")
        restored = select_methods(
            config,
            config.dataset.dir / "scores",
            as_of=datetime.now(tz=UTC) + timedelta(minutes=1),
        )
        assert restored
        assert {choice.release_id for choice in restored.values()} == {
            choice.release_id for choice in promoted.values()
        }

    def test_new_targeted_evaluation_updates_only_its_slice(self, tmp_path):
        config = scored_config(tmp_path)
        scores_dir = config.dataset.dir / "scores"
        original = load_scores(next(scores_dir.glob("scores_*.parquet")))
        original_evaluation = str(original["evaluation_id"][0])
        target_bucket = str(original["lead_bucket"].unique().sort()[0])
        targeted = original.filter(pl.col("lead_bucket") == target_bucket).with_columns(
            pl.lit("targeted-evaluation").alias("evaluation_id"),
            (pl.col("evaluation_created_at") + pl.duration(hours=1)).alias(
                "evaluation_created_at"
            ),
        )
        write_scores(targeted, scores_dir / "scores_hourly_live_targeted.parquet")

        selections = select_methods(config, scores_dir)
        assert selections[("hourly", "temp_c", target_bucket)].evaluation_id == (
            "targeted-evaluation"
        )
        assert {
            selected.evaluation_id
            for key, selected in selections.items()
            if key[2] != target_bucket
        } == {original_evaluation}

    def test_challenger_only_evaluation_is_ignored_for_promotion(self, tmp_path):
        config = scored_config(tmp_path)
        scores_dir = config.dataset.dir / "scores"
        original = load_scores(next(scores_dir.glob("scores_*.parquet")))
        original_evaluation = str(original["evaluation_id"][0])
        challenger_only = original.filter(
            pl.col("method_id") == "grounded_equal_weight"
        ).with_columns(
            pl.lit("challenger-only").alias("evaluation_id"),
            (pl.col("evaluation_created_at") + pl.duration(hours=1)).alias(
                "evaluation_created_at"
            ),
        )
        write_scores(
            challenger_only, scores_dir / "scores_hourly_live_challenger.parquet"
        )

        selections = select_methods(config, scores_dir)
        assert {selected.evaluation_id for selected in selections.values()} == {
            original_evaluation
        }

    def test_no_complete_reference_evaluation_fails_closed(self, tmp_path):
        config = scored_config(tmp_path)
        scores_dir = config.dataset.dir / "scores"
        path = next(scores_dir.glob("scores_*.parquet"))
        challenger_only = load_scores(path).filter(
            pl.col("method_id") == "grounded_equal_weight"
        )
        path.unlink()
        write_scores(challenger_only, scores_dir / "scores_hourly_live_partial.parquet")
        assert select_methods(config, scores_dir) == {}

    def test_structurally_invalid_release_is_ignored(self, tmp_path):
        config = scored_config(tmp_path)
        releases = config.artifacts_dir / "releases"
        releases.mkdir(parents=True, exist_ok=True)
        (releases / "broken.json").write_text(
            '{"config_fingerprint": "present-but-incomplete"}',
            encoding="utf-8",
        )

        assert select_methods(config, config.dataset.dir / "scores")


class TestMethodFor:
    def test_falls_back_explicitly(self, tmp_path):
        config = write_config(tmp_path)
        chosen = method_for({}, "hourly", "temp_c", "0-1h", config)
        assert chosen.method_id == FALLBACK_METHOD
        assert "no backtest evidence" in chosen.reason

    def test_pin_wins_over_scores(self, tmp_path):
        config = write_config(
            tmp_path, extra_toml='\n[predict.methods]\n"hourly.temp_c" = "gbm"\n'
        )
        selections = {("hourly", "temp_c", "0-1h"): Selection("equal_weight", "x")}
        assert (
            method_for(selections, "hourly", "temp_c", "0-1h", config).method_id
            == "gbm"
        )

    def test_matching_pin_preserves_release_provenance(self, tmp_path):
        config = write_config(
            tmp_path, extra_toml='\n[predict.methods]\n"hourly.temp_c" = "gbm"\n'
        )
        promoted = Selection(
            "gbm",
            "promoted",
            n=100,
            mae=1.0,
            evaluation_id="eval-r",
            dataset_fingerprint="dataset-r",
            release_id="release-r",
            code_version="0.4.0+implementation",
        )

        chosen = method_for(
            {("hourly", "temp_c", "0-1h"): promoted},
            "hourly",
            "temp_c",
            "0-1h",
            config,
        )

        assert chosen.pinned
        assert chosen.release_id == "release-r"
        assert chosen.evaluation_id == "eval-r"
        assert chosen.code_version == "0.4.0+implementation"

    def test_unknown_bucket_falls_back(self, tmp_path):
        config = write_config(tmp_path)
        assert (
            method_for({}, "hourly", "temp_c", None, config).method_id
            == FALLBACK_METHOD
        )


class TestScoresProvenance:
    def test_reads_both_kinds_separately(self, tmp_path):
        config = scored_config(tmp_path)
        synthetic = synthetic_hourly_matrix(days=25, source_kind="synthetic")
        scores = run_backtest(
            synthetic,
            BacktestRequest(
                variables=(hourly_variable("temp_c"),), methods=("equal_weight",)
            ),
            config,
        )
        write_scores(
            scores, scores_path(config.dataset.dir / "scores", "hourly", "synthetic")
        )
        selections = select_methods(config, config.dataset.dir / "scores")
        # both files load without a MixedProvenanceError
        assert selections
        live_evaluation = pl.read_parquet(
            scores_path(config.dataset.dir / "scores", "hourly", "live")
        )["evaluation_id"][0]
        assert {choice.evaluation_id for choice in selections.values()} == {
            live_evaluation
        }
        assert isinstance(
            pl.read_parquet(
                scores_path(config.dataset.dir / "scores", "hourly", "synthetic")
            ),
            pl.DataFrame,
        )
        assert utc(2026, 1, 1)  # sanity: fixture epoch unchanged


def test_release_eligibility_uses_implementation_not_promotion_age(tmp_path):
    import json

    from grounded_weather_forecast.evaluation import config_fingerprint

    config = write_config(tmp_path)
    releases = config.artifacts_dir / "releases"
    releases.mkdir(parents=True)
    release_context = {
        "evaluation_id": "eval-v1",
        "source_kind": "live",
        "source_set_json": json.dumps(["nws", "ecmwf"]),
        "feature_set_json": json.dumps(["lead_bucket"]),
        "semantics": {"temp_c": "inst"},
        "window": "expanding",
        "code_version": "0.4.0+implementation-v1",
        "config_fingerprint": config_fingerprint(config),
    }
    release = {
        "release_id": "release-old-but-active",
        "promoted_at": "2020-01-01T00:00:00+00:00",
        "dataset_fingerprint": "old-dataset",
        "config_fingerprint": config_fingerprint(config),
        "evaluation_contexts": [release_context],
        "selections": {
            "hourly.temp_c.0-1h": {
                "method_id": "gbm",
                "evaluation_id": "eval-v1",
                "code_version": "0.4.0+implementation-v1",
            }
        },
    }
    (releases / "release-old-but-active.json").write_text(
        json.dumps(release), encoding="utf-8"
    )
    key = ("hourly", "temp_c", "0-1h")
    matching = {
        key: Selection(
            "gbm",
            "won",
            evaluation_id="eval-current",
            code_version="0.4.0+implementation-v1",
        )
    }
    changed = {
        key: Selection(
            "gbm",
            "won",
            evaluation_id="eval-current",
            code_version="0.4.0+implementation-v2",
        )
    }
    current_contexts = (
        release_context
        | {
            "evaluation_id": "eval-current",
            "source_set_json": json.dumps(["ecmwf", "nws"]),
        },
    )

    assert _eligible_release_ids(config, matching, current_contexts)[
        (*key, "gbm")
    ] == frozenset({"release-old-but-active"})
    assert not _eligible_release_ids(config, changed, current_contexts)[(*key, "gbm")]


def test_release_eligibility_rejects_incompatible_evaluation_context(tmp_path):
    import json

    from grounded_weather_forecast.evaluation import config_fingerprint

    config = write_config(tmp_path)
    releases = config.artifacts_dir / "releases"
    releases.mkdir(parents=True)
    release = {
        "release_id": "release-old-context",
        "promoted_at": "2020-01-01T00:00:00+00:00",
        "dataset_fingerprint": "old-dataset",
        "config_fingerprint": config_fingerprint(config),
        "evaluation_contexts": [
            {
                "evaluation_id": "eval-old",
                "source_kind": "live",
                "source_set_json": json.dumps(["nws", "ecmwf"]),
                "feature_set_json": json.dumps(["lead_bucket"]),
                "semantics": {"temp_c": "inst"},
                "code_version": "implementation",
            }
        ],
        "selections": {
            "hourly.temp_c.0-1h": {
                "method_id": "gbm",
                "evaluation_id": "eval-old",
                "code_version": "implementation",
            }
        },
    }
    (releases / "release-old-context.json").write_text(
        json.dumps(release), encoding="utf-8"
    )
    key = ("hourly", "temp_c", "0-1h")
    selected = {
        key: Selection(
            "gbm",
            "won",
            evaluation_id="eval-current",
            code_version="implementation",
        )
    }

    def eligible(
        *,
        sources=("nws", "ecmwf"),
        features=("lead_bucket",),
        semantic="inst",
        present=True,
    ):
        contexts = (
            {
                "evaluation_id": "eval-current",
                "source_kind": "live",
                "source_set_json": json.dumps(sources),
                "feature_set_json": json.dumps(features),
                "semantics": {"temp_c": semantic},
                "code_version": "implementation",
            },
        )
        return _eligible_release_ids(config, selected, contexts if present else ())[
            (*key, "gbm")
        ]

    assert eligible() == frozenset({"release-old-context"})
    assert not eligible(sources=("nws", "gfs"))
    assert not eligible(features=("lead_bucket", "ens__temp_c__spread"))
    assert not eligible(semantic="mean")
    assert not eligible(present=False)


def test_selection_and_historical_replay_bind_requested_truth_semantics(tmp_path):
    config = scored_config(tmp_path)
    scores_dir = config.dataset.dir / "scores"
    matrix = synthetic_hourly_matrix(days=25, biases={"alpha": 3.0})
    mean_scores = run_backtest(
        matrix,
        BacktestRequest(
            variables=(hourly_variable("temp_c"),),
            methods=(
                "equal_weight",
                "grounded_equal_weight",
                "best_provider",
                "damped_grounded_equal_weight",
            ),
            semantics=TruthSemantics.INTERVAL_MEAN,
        ),
        config,
    )
    write_scores(mean_scores, scores_dir / "scores_hourly_live_mean.parquet")

    instantaneous = select_methods(
        config,
        scores_dir,
        semantics={"temp_c": TruthSemantics.INSTANTANEOUS},
    )
    interval_mean = select_methods(
        config,
        scores_dir,
        semantics={"temp_c": TruthSemantics.INTERVAL_MEAN},
    )

    assert {choice.truth_semantics for choice in instantaneous.values()} == {"inst"}
    assert {choice.truth_semantics for choice in interval_mean.values()} == {"mean"}
    assert {choice.release_id for choice in instantaneous.values()} != {
        choice.release_id for choice in interval_mean.values()
    }
    restored = select_methods(
        config,
        scores_dir,
        as_of=datetime.now(tz=UTC) + timedelta(minutes=1),
        semantics={"temp_c": TruthSemantics.INSTANTANEOUS},
    )
    assert {choice.release_id for choice in restored.values()} == {
        choice.release_id for choice in instantaneous.values()
    }


def test_historical_release_requires_current_implementation(tmp_path):
    import json

    config = scored_config(tmp_path)
    promoted = select_methods(config, config.dataset.dir / "scores")
    release_id = next(iter({choice.release_id for choice in promoted.values()}))
    assert release_id is not None
    release_path = config.artifacts_dir / "releases" / f"{release_id}.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    for selection in release["selections"].values():
        selection["code_version"] = "0.4.0+retired-implementation"
    for context in release["evaluation_contexts"]:
        context["code_version"] = "0.4.0+retired-implementation"
    release_path.write_text(json.dumps(release), encoding="utf-8")

    restored = select_methods(
        config,
        config.dataset.dir / "scores",
        as_of=datetime.now(tz=UTC) + timedelta(minutes=1),
    )

    assert restored == {}


def test_selection_positional_fields_keep_their_original_meaning():
    chosen = Selection("gbm", "won", 100, 1.25, "eval", "dataset", "release")

    assert chosen.n == 100
    assert chosen.mae == 1.25
    assert chosen.evaluation_id == "eval"
    assert chosen.dataset_fingerprint == "dataset"
    assert chosen.release_id == "release"
    assert not chosen.pinned


class TestNoEvidenceReason:
    """Degradation must name its cause: cold start vs invalidated evidence."""

    def _live_scores_row(self, dataset_fp, config_fp):
        import json as _json

        from grounded_weather_forecast.backtest.scores import SCORES_SCHEMA
        from grounded_weather_forecast.timeutil import utc

        issue = utc(2026, 3, 22, 12)
        return pl.DataFrame(
            {
                "issue_time": [issue],
                "valid_time": [issue],
                "lead_hours": [24.0],
                "lead_bucket": ["24-48h"],
                "method_id": ["equal_weight"],
                "variable": ["temp_c"],
                "product": ["hourly"],
                "source_kind": ["live"],
                "evaluation_id": ["eval1"],
                "evaluation_created_at": [issue],
                "dataset_fingerprint": [dataset_fp],
                "source_set_json": [_json.dumps(["nws"])],
                "feature_set_json": [_json.dumps(["lead_bucket"])],
                "semantics": ["inst"],
                "code_version": ["test"],
                "config_fingerprint": [config_fp],
                "window": ["expanding"],
                "fold_origin": [issue],
                "y_pred": [1.0],
                "y_true": [1.0],
                "quantile_levels_json": ["[]"],
                "quantiles_json": [None],
            }
        ).cast(SCORES_SCHEMA)

    def test_cold_start(self, tmp_path):
        from grounded_weather_forecast.serve.selection import no_evidence_reason

        config = write_config(tmp_path)
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        assert "cold start" in no_evidence_reason(config, scores_dir)

    def test_fingerprint_changed_after_rebuild(self, tmp_path):
        from grounded_weather_forecast.serve.selection import no_evidence_reason

        config = write_config(tmp_path)
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        stale = self._live_scores_row("oldfingerprint00", "oldconfig0000000")
        stale.write_parquet(scores_dir / "scores_hourly_live_expanding.parquet")
        reason = no_evidence_reason(config, scores_dir)
        assert "fingerprint changed" in reason
        assert "re-run" in reason

    def test_config_changed(self, tmp_path):
        from grounded_weather_forecast.evaluation import dataset_fingerprint
        from grounded_weather_forecast.serve.selection import no_evidence_reason

        config = write_config(tmp_path)
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        stale = self._live_scores_row(dataset_fingerprint(config), "oldconfig0000000")
        stale.write_parquet(scores_dir / "scores_hourly_live_expanding.parquet")
        assert "config changed" in no_evidence_reason(config, scores_dir)

    def test_implementation_changed(self, tmp_path):
        from grounded_weather_forecast.evaluation import (
            config_fingerprint,
            dataset_fingerprint,
        )
        from grounded_weather_forecast.serve.selection import no_evidence_reason

        config = write_config(tmp_path)
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        stale = self._live_scores_row(
            dataset_fingerprint(config), config_fingerprint(config)
        )
        stale.write_parquet(scores_dir / "scores_hourly_live_expanding.parquet")
        assert "implementation changed" in no_evidence_reason(config, scores_dir)

    def test_synthetic_only_evidence(self, tmp_path):
        from grounded_weather_forecast.serve.selection import no_evidence_reason

        config = write_config(tmp_path)
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        synthetic = self._live_scores_row("any", "any").with_columns(
            pl.lit("synthetic").alias("source_kind")
        )
        synthetic.write_parquet(
            scores_dir / "scores_hourly_synthetic_expanding.parquet"
        )
        assert "no live backtest evidence" in no_evidence_reason(config, scores_dir)


class TestForecastStatusReason:
    def test_round_trips_and_tolerates_absence(self):
        from grounded_weather_forecast.serve.schema import Forecast

        forecast = Forecast(
            schema_version=2,
            issued_at="2026-03-22T12:00:00+00:00",
            latitude=34.0,
            longitude=-117.0,
            dataset_fingerprint="fp",
            sources=[],
            observation_at=None,
            minutely=[],
            hourly=[],
            daily=[],
            status="degraded",
            status_reason="cold start: no backtest scores exist yet",
        )
        loaded = Forecast.from_json(forecast.to_json())
        assert loaded.status_reason == forecast.status_reason
        legacy = forecast.to_json().replace(
            '"status_reason": "cold start: no backtest scores exist yet",', ""
        )
        assert Forecast.from_json(legacy).status_reason is None


def daily_pool_scores(created):
    """Far daily buckets with 5 valid times each: only the pool can promote."""
    rng = np.random.default_rng(31)
    methods = (
        "inverse_mse",
        "equal_weight",
        "best_provider",
        "damped_grounded_equal_weight",
    )
    rows = []
    for bucket_index, bucket in enumerate(("D3-4", "D5-7", "D8-10")):
        for step in range(5):
            valid = utc(2026, 7, 3) + timedelta(days=bucket_index * 5 + step)
            truth = 30.0 + rng.normal(0.0, 0.1)
            for method in methods:
                offset = 0.05 if method == "inverse_mse" else 5.0
                rows.append(
                    {
                        "method_id": method,
                        "variable": "temp_max_c",
                        "product": "daily",
                        "source_kind": "live",
                        "evaluation_id": "evalpool",
                        "evaluation_created_at": created,
                        "dataset_fingerprint": "ds1",
                        "source_set_json": "[]",
                        "feature_set_json": "[]",
                        "semantics": "inst",
                        "code_version": "code1",
                        "config_fingerprint": "cfg1",
                        "window": "expanding",
                        "fold_origin": created,
                        "issue_time": utc(2026, 7, 1),
                        "valid_time": valid,
                        "lead_hours": 24.0 * (4 + bucket_index * 3),
                        "lead_bucket": bucket,
                        "y_pred": truth + offset + rng.normal(0.0, 0.05),
                        "y_true": truth,
                        "quantile_levels_json": "[]",
                        "quantiles_json": None,
                    }
                )
    return pl.DataFrame(rows).cast(SCORES_SCHEMA)


class TestPooledDailySelection:
    def test_far_buckets_serve_through_the_pool(self, tmp_path, monkeypatch):
        config = write_config(tmp_path)
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        monkeypatch.setattr(
            selection_module, "dataset_fingerprint", lambda _config: "ds1"
        )
        monkeypatch.setattr(
            selection_module, "config_fingerprint", lambda _config: "cfg1"
        )
        monkeypatch.setattr(selection_module, "code_identity", lambda: "code1")
        write_scores(
            daily_pool_scores(utc(2026, 8, 1)),
            scores_path(scores_dir, "daily", "live"),
        )
        selections = select_methods(config, scores_dir)
        for bucket in ("D3-4", "D5-7", "D8-10"):
            chosen = selections[("daily", "temp_max_c", bucket)]
            assert chosen.method_id == "inverse_mse"
            assert "pooled D3-10" in chosen.reason
            assert chosen.n == 15  # pooled evidence, not the fine bucket's 5


def near_tie_scores(created, evaluation_id, best_method, *, rival_offset=1.05):
    """One hourly 24-48h slice: `best_method` edges `inverse_mse`-vs-`cluster`.

    A shared per-time draw plays the weather (it cancels in the paired
    difference the retention SE is built on) and a small per-method draw
    supplies the idiosyncratic noise that gives the difference a real
    sampling variance. All three default references are present and clearly
    worse, keeping the MCS gate decisive so the winner is a true argmin pick.
    """
    rng = np.random.default_rng(11)
    pair = ("inverse_mse", "cluster_equal_weight")
    references = ("equal_weight", "best_provider", "damped_grounded_equal_weight")
    rows = []
    for step in range(30):
        valid = utc(2026, 7, 1) + timedelta(hours=step)
        truth = 25.0 + rng.normal(0.0, 0.1)
        shared_noise = rng.normal(0.0, 0.3)
        for method in pair + references:
            if method in references:
                offset = 5.0
            elif method == best_method:
                offset = 1.0
            else:
                offset = rival_offset
            offset += rng.normal(0.0, 0.15)
            rows.append(
                {
                    "method_id": method,
                    "variable": "temp_c",
                    "product": "hourly",
                    "source_kind": "live",
                    "evaluation_id": evaluation_id,
                    "evaluation_created_at": created,
                    "dataset_fingerprint": "ds1",
                    "source_set_json": "[]",
                    "feature_set_json": "[]",
                    "semantics": "inst",
                    "code_version": "code1",
                    "config_fingerprint": "cfg1",
                    "window": "expanding",
                    "fold_origin": created,
                    "issue_time": utc(2026, 7, 1),
                    "valid_time": valid,
                    "lead_hours": 30.0,
                    "lead_bucket": "24-48h",
                    "y_pred": truth + offset + shared_noise,
                    "y_true": truth,
                    "quantile_levels_json": "[]",
                    "quantiles_json": None,
                }
            )
    return pl.DataFrame(rows).cast(SCORES_SCHEMA)


class TestIncumbentRetention:
    KEY = ("hourly", "temp_c", "24-48h")

    def _pin_fingerprints(self, monkeypatch):
        monkeypatch.setattr(
            selection_module, "dataset_fingerprint", lambda _config: "ds1"
        )
        monkeypatch.setattr(
            selection_module, "config_fingerprint", lambda _config: "cfg1"
        )
        monkeypatch.setattr(selection_module, "code_identity", lambda: "code1")

    def test_near_tie_keeps_the_incumbent(self, tmp_path, monkeypatch):
        config = write_config(tmp_path)
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        self._pin_fingerprints(monkeypatch)
        write_scores(
            near_tie_scores(utc(2026, 8, 1), "evalone", "inverse_mse"),
            scores_path(scores_dir, "hourly", "live", "expanding", "evalone"),
        )
        first = select_methods(config, scores_dir)
        assert first[self.KEY].method_id == "inverse_mse"
        assert not first[self.KEY].retained
        # A fresh evaluation where the rival edges ahead on the board but the
        # paired common-case difference stays inside one bootstrap SE — it
        # must not evict the serving incumbent.
        write_scores(
            near_tie_scores(utc(2026, 8, 2), "evaltwo", "cluster_equal_weight"),
            scores_path(scores_dir, "hourly", "live", "expanding", "evaltwo"),
        )
        second = select_methods(config, scores_dir)
        chosen = second[self.KEY]
        assert chosen.method_id == "inverse_mse"
        assert chosen.retained
        assert "retained incumbent" in chosen.reason
        assert chosen.evaluation_id == "evaltwo"  # current evidence, not stale
        assert chosen.mae is not None

    def test_clear_winner_evicts_the_incumbent(self, tmp_path, monkeypatch):
        config = write_config(tmp_path)
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        self._pin_fingerprints(monkeypatch)
        write_scores(
            near_tie_scores(utc(2026, 8, 1), "evalone", "inverse_mse"),
            scores_path(scores_dir, "hourly", "live", "expanding", "evalone"),
        )
        select_methods(config, scores_dir)
        # The rival now wins by ~0.8 C — many SEs — so the argmin stands.
        write_scores(
            near_tie_scores(
                utc(2026, 8, 2),
                "evaltwo",
                "cluster_equal_weight",
                rival_offset=1.8,
            ),
            scores_path(scores_dir, "hourly", "live", "expanding", "evaltwo"),
        )
        second = select_methods(config, scores_dir)
        chosen = second[self.KEY]
        assert chosen.method_id == "cluster_equal_weight"
        assert not chosen.retained

    def test_retention_round_trips_through_the_release(self, tmp_path, monkeypatch):
        config = write_config(tmp_path)
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        self._pin_fingerprints(monkeypatch)
        write_scores(
            near_tie_scores(utc(2026, 8, 1), "evalone", "inverse_mse"),
            scores_path(scores_dir, "hourly", "live", "expanding", "evalone"),
        )
        select_methods(config, scores_dir)
        write_scores(
            near_tie_scores(utc(2026, 8, 2), "evaltwo", "cluster_equal_weight"),
            scores_path(scores_dir, "hourly", "live", "expanding", "evaltwo"),
        )
        second = select_methods(config, scores_dir)
        release_id = second[self.KEY].release_id
        raw = json.loads(
            (config.artifacts_dir / "releases" / f"{release_id}.json").read_text()
        )
        payload = raw["selections"]["hourly.temp_c.24-48h"]
        assert payload["retained"] is True
        assert payload["method_id"] == "inverse_mse"
        rehydrated = selection_module._selections_from_release(raw)
        assert rehydrated is not None
        assert rehydrated[self.KEY].retained is True
        # And a payload without the key defaults to False (older releases).
        del payload["retained"]
        older = selection_module._selections_from_release(raw)
        assert older is not None
        assert older[self.KEY].retained is False

    def test_fallback_incumbent_is_never_retained(self, tmp_path, monkeypatch):
        config = write_config(tmp_path)
        self._pin_fingerprints(monkeypatch)
        frame = near_tie_scores(utc(2026, 8, 2), "evaltwo", "cluster_equal_weight")
        board = selection_module.leaderboard(frame)
        row = {
            "product": "hourly",
            "variable": "temp_c",
            "lead_bucket": "24-48h",
            "truth_semantics": "inst",
            "method_id": "cluster_equal_weight",
            "mae": 1.0,
            "gate": None,
        }
        kept = selection_module._retained_incumbent(
            config, row, board, frame, (FALLBACK_METHOD, "inst"), "evaltwo"
        )
        assert kept is None

    def test_semantics_change_is_never_retained(self, tmp_path, monkeypatch):
        config = write_config(tmp_path)
        self._pin_fingerprints(monkeypatch)
        frame = near_tie_scores(utc(2026, 8, 2), "evaltwo", "cluster_equal_weight")
        board = selection_module.leaderboard(frame)
        row = {
            "product": "hourly",
            "variable": "temp_c",
            "lead_bucket": "24-48h",
            "truth_semantics": "inst",
            "method_id": "cluster_equal_weight",
            "mae": 1.0,
            "gate": None,
        }
        kept = selection_module._retained_incumbent(
            config, row, board, frame, ("inverse_mse", "mean"), "evaltwo"
        )
        assert kept is None

    def test_gate_forced_winner_is_never_overridden(self, tmp_path, monkeypatch):
        config = write_config(tmp_path)
        self._pin_fingerprints(monkeypatch)
        frame = near_tie_scores(utc(2026, 8, 2), "evaltwo", "cluster_equal_weight")
        board = selection_module.leaderboard(frame)
        row = {
            "product": "hourly",
            "variable": "temp_c",
            "lead_bucket": "24-48h",
            "truth_semantics": "inst",
            "method_id": "equal_weight",
            "mae": 1.0,
            "gate": "dm_not_significant",
        }
        kept = selection_module._retained_incumbent(
            config, row, board, frame, ("inverse_mse", "inst"), "evaltwo"
        )
        assert kept is None


class TestPruneRaceResilience:
    """A lock-free predict must survive prune deleting files mid-scan."""

    def test_scan_retries_once_when_a_file_vanishes(self, tmp_path, monkeypatch):
        config = scored_config(tmp_path)
        real = selection_module.load_scores
        calls = {"count": 0}

        def flaky(path, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise FileNotFoundError(path)
            return real(path, **kwargs)

        monkeypatch.setattr(selection_module, "load_scores", flaky)
        frames = selection_module._compatible_scores(
            config, config.dataset.dir / "scores", None, None
        )
        assert frames  # the clean second pass recovered the evidence
        assert calls["count"] >= 2

    def test_second_pass_tolerates_persistently_missing_files(
        self, tmp_path, monkeypatch
    ):
        config = scored_config(tmp_path)

        def always_missing(path, **kwargs):
            raise FileNotFoundError(path)

        monkeypatch.setattr(selection_module, "load_scores", always_missing)
        frames = selection_module._compatible_scores(
            config, config.dataset.dir / "scores", None, None
        )
        assert frames == []

    def test_no_evidence_reason_survives_vanishing_files(self, tmp_path, monkeypatch):
        config = scored_config(tmp_path)

        def always_missing(path, **kwargs):
            raise FileNotFoundError(path)

        monkeypatch.setattr(selection_module, "load_scores", always_missing)
        reason = selection_module.no_evidence_reason(
            config, config.dataset.dir / "scores"
        )
        assert "no live backtest evidence" in reason
