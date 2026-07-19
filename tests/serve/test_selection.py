from datetime import UTC, datetime, timedelta

import polars as pl
from conftest import synthetic_hourly_matrix, utc, write_config

from grounded_weather_forecast.backtest.engine import BacktestRequest, run_backtest
from grounded_weather_forecast.backtest.scores import (
    load_scores,
    scores_path,
    write_scores,
)
from grounded_weather_forecast.contracts import hourly_variable
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
            methods=("equal_weight", "grounded_equal_weight", "best_provider"),
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
    release = {
        "release_id": "release-old-but-active",
        "promoted_at": "2020-01-01T00:00:00+00:00",
        "dataset_fingerprint": "old-dataset",
        "config_fingerprint": config_fingerprint(config),
        "evaluation_contexts": [],
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
    matching = {key: Selection("gbm", "won", code_version="0.4.0+implementation-v1")}
    changed = {key: Selection("gbm", "won", code_version="0.4.0+implementation-v2")}

    assert _eligible_release_ids(config, matching)[(*key, "gbm")] == frozenset(
        {"release-old-but-active"}
    )
    assert not _eligible_release_ids(config, changed)[(*key, "gbm")]


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
