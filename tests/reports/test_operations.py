"""Operational evidence: freshness alarms, provider health, funnel, changes."""

import json
import os
import sqlite3
from datetime import timedelta

import polars as pl
from conftest import make_forecast_db, utc, write_config

from grounded_weather_forecast.reports import operations
from grounded_weather_forecast.reports.evidence import (
    EVALUATIONS_LEDGER,
    PIPELINE_LEDGER,
    PROVIDER_HEALTH_LEDGER,
    PROVIDER_HEALTH_SCHEMA,
    append_ledger,
    ledger_path,
    load_ledger,
)
from grounded_weather_forecast.runs import RUNS_SCHEMA

NOW = utc(2026, 8, 7, 12)


def station_db(config, timestamps):
    connection = sqlite3.connect(config.station.db_path)
    try:
        connection.execute("CREATE TABLE observations (ts TEXT)")
        connection.executemany(
            "INSERT INTO observations VALUES (?)",
            [(ts.strftime("%Y-%m-%d %H:%M:%S"),) for ts in timestamps],
        )
        connection.commit()
    finally:
        connection.close()


def runs_frame(rows):
    filled = [
        {column: row.get(column) for column in RUNS_SCHEMA.names()} for row in rows
    ]
    return pl.DataFrame(filled, schema=dict(RUNS_SCHEMA))


def healthy_run(command="predict", minutes_ago=30, exit_code=0):
    return {
        "run_id": f"{command}{minutes_ago}",
        "command": command,
        "started_at": NOW - timedelta(minutes=minutes_ago),
        "exit_code": exit_code,
    }


class TestFreshness:
    def healthy_config(self, tmp_path):
        config = write_config(tmp_path, station_db="live_station.db")
        # station: one sample per minute for the last 24 hours
        station_db(config, [NOW - timedelta(minutes=m) for m in range(1440)])
        make_forecast_db(
            config.forecasts.db_path,
            [
                {
                    "completed_at": (NOW - timedelta(minutes=20)).isoformat(),
                    "results": [
                        {
                            "provider": "alpha",
                            "fetched_at": (NOW - timedelta(minutes=20)).isoformat(),
                            "fetched_at_unix": int(
                                (NOW - timedelta(minutes=20)).timestamp()
                            ),
                        }
                    ],
                }
            ],
        )
        config.predict.history_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {"issued_at": [NOW - timedelta(minutes=40)]},
            schema={"issued_at": pl.Datetime("us", "UTC")},
        ).write_parquet(config.predict.history_path)
        documents = config.predict.history_path.parent / "served_forecasts"
        documents.mkdir()
        document = documents / "forecast.json"
        document.write_text("{}", encoding="utf-8")
        moment = (NOW - timedelta(minutes=40)).timestamp()
        os.utime(document, times=(moment, moment))
        return config

    def test_healthy_pipeline_raises_no_alarms(self, tmp_path):
        config = self.healthy_config(tmp_path)
        frame = runs_frame([healthy_run(minutes_ago=30 + h * 60) for h in range(24)])
        row, alarms = operations.freshness_row(config, frame, now=NOW)
        assert alarms == ()
        assert row["truth_samples_24h"] == 1440
        assert row["collector_runs_24h"] == 1
        assert row["provider_success_rate_24h"] == 1.0
        # ages measured against the injected clock, minute-exact
        assert abs(row["truth_age_minutes"] - 0.0) < 1.0
        assert abs(row["collector_age_minutes"] - 20.0) < 1.0
        assert abs(row["served_history_age_minutes"] - 40.0) < 1.0

    def test_stale_and_thin_truth_alarm(self, tmp_path):
        config = write_config(tmp_path, station_db="live_station.db")
        station_db(config, [NOW - timedelta(minutes=300)])
        _, alarms = operations.freshness_row(config, runs_frame([]), now=NOW)
        assert any(a.startswith("truth stale (300m") for a in alarms)
        assert any("thin truth (1 samples/24h" in a for a in alarms)

    def test_missing_edges_report_unavailable(self, tmp_path):
        config = write_config(tmp_path)
        row, alarms = operations.freshness_row(config, runs_frame([]), now=NOW)
        for name in ("truth", "collector", "served history", "forecast document"):
            assert f"{name} unavailable" in alarms
        # an empty run ledger is a young deployment, not a scheduler fault
        assert row["predict_runs_24h"] is None
        assert not any("predict runs" in a for a in alarms)

    def test_run_ledger_alarms(self, tmp_path):
        config = self.healthy_config(tmp_path)
        frame = runs_frame(
            [
                *(healthy_run(minutes_ago=30 + h * 60) for h in range(3)),
                healthy_run(command="report", minutes_ago=100, exit_code=70),
                # tolerated: live backtest exit 1 means "no folds yet"
                healthy_run(command="backtest", minutes_ago=90, exit_code=1),
            ]
        )
        row, alarms = operations.freshness_row(config, frame, now=NOW)
        assert row["predict_runs_24h"] == 3
        assert row["failed_runs_24h"] == 1
        assert any("few predict runs (3/24h" in a for a in alarms)
        assert any("failed cli runs in 24h: 1" in a for a in alarms)


def collector_with_leads(config):
    fetched = NOW - timedelta(hours=1)
    unix = int(fetched.timestamp())
    results = [
        {
            "provider": "alpha",
            "fetched_at": fetched.isoformat(),
            "fetched_at_unix": unix,
            "latency_ms": latency,
            "hourly": [(fetched + timedelta(hours=48), {"temperature": 10.0})],
            "daily": [
                ((fetched + timedelta(days=5)).strftime("%Y-%m-%d"), {}),
            ],
        }
        for latency in (10.0, 20.0, 30.0, 40.0)
    ]
    results += [
        {
            "provider": "beta",
            "status": "success" if ok else "error",
            "fetched_at": fetched.isoformat(),
            "fetched_at_unix": unix,
        }
        for ok in (True, True, False, False)
    ]
    make_forecast_db(
        config.forecasts.db_path,
        [{"completed_at": fetched.isoformat(), "results": results}],
    )


class TestProviderHealth:
    def test_rows_carry_rates_latency_and_leads(self, tmp_path):
        config = write_config(tmp_path)
        collector_with_leads(config)
        frame = operations.provider_health_rows(config, now=NOW)
        alpha = frame.filter(pl.col("provider") == "alpha").row(0, named=True)
        assert alpha["runs_24h"] == 4
        assert alpha["success_rate"] == 1.0
        # median of 10/20/30/40
        assert alpha["median_latency_ms"] == 25.0
        assert abs(alpha["max_hourly_lead_hours"] - 48.0) < 1e-9
        assert alpha["max_daily_lead_days"] == 5.0
        beta = frame.filter(pl.col("provider") == "beta").row(0, named=True)
        assert beta["success_rate"] == 0.5
        assert beta["max_hourly_lead_hours"] is None

    def test_missing_archive_yields_empty_typed_frame(self, tmp_path):
        config = write_config(tmp_path)
        frame = operations.provider_health_rows(config, now=NOW)
        assert frame.is_empty()
        assert frame.schema == PROVIDER_HEALTH_SCHEMA


def health_history(providers_days):
    rows = []
    for provider, days, daily_lead in providers_days:
        rows += [
            {
                "recorded_at": NOW - timedelta(days=day),
                "as_of_date": (NOW - timedelta(days=day)).date(),
                "provider": provider,
                "max_daily_lead_days": daily_lead,
                "max_hourly_lead_hours": 100.0,
            }
            for day in range(1, days + 1)
        ]
    return pl.DataFrame(
        [{c: row.get(c) for c in PROVIDER_HEALTH_SCHEMA.names()} for row in rows],
        schema=dict(PROVIDER_HEALTH_SCHEMA),
    )


def health_today(provider, *, daily=None, hourly=None, rate=1.0):
    row = {
        "recorded_at": NOW,
        "as_of_date": NOW.date(),
        "provider": provider,
        "success_rate": rate,
        "max_daily_lead_days": daily,
        "max_hourly_lead_hours": hourly,
    }
    return pl.DataFrame(
        [{c: row.get(c) for c in PROVIDER_HEALTH_SCHEMA.names()}],
        schema=dict(PROVIDER_HEALTH_SCHEMA),
    )


class TestProviderContractions:
    def test_daily_lead_contraction_against_own_median(self):
        history = health_history([("alpha", 5, 7.0)])
        fresh = health_today("alpha", daily=5.0, hourly=100.0)
        notes = operations.provider_contractions(fresh, history, now=NOW)
        assert notes == ("alpha: daily lead 5d < median 7d",)

    def test_thin_baseline_and_small_dips_stay_quiet(self):
        # 2 baseline days < the 3-day floor; and 7 -> 6.5 is within tolerance
        history = health_history([("alpha", 2, 7.0), ("gamma", 5, 7.0)])
        fresh = pl.concat(
            [
                health_today("alpha", daily=2.0, hourly=100.0),
                health_today("gamma", daily=6.5, hourly=100.0),
            ]
        )
        assert operations.provider_contractions(fresh, history, now=NOW) == ()

    def test_success_rate_note_needs_no_baseline(self):
        fresh = health_today("beta", rate=0.5)
        notes = operations.provider_contractions(fresh, pl.DataFrame(), now=NOW)
        assert notes == ("beta: success rate 50%",)


class TestBuildFunnel:
    def seeded_config(self, tmp_path):
        config = write_config(tmp_path)
        collector_with_leads(config)
        config.dataset.dir.mkdir(parents=True, exist_ok=True)
        fetched = NOW - timedelta(hours=2)
        pl.DataFrame(
            {
                "source": ["alpha", "alpha"],
                "fetched_at": [fetched, fetched],
                "lead_hours": [1.0, 47.0],
            }
        ).write_parquet(config.dataset.dir / "forecasts_long.parquet")
        pl.DataFrame(
            {
                "source": ["alpha"],
                "fetched_at": [fetched],
                "forecast_date": [(fetched + timedelta(days=4)).date()],
            }
        ).write_parquet(config.dataset.dir / "daily_long.parquet")
        pl.DataFrame(
            {
                "issue_time": [NOW - timedelta(hours=3)] * 2,
                "lead_hours": [1.0, 46.0],
                "fx__alpha__temp_c": [10.0, 11.0],
            }
        ).write_parquet(config.dataset.dir / "hourly_matrix_live.parquet")
        pl.DataFrame(
            {
                "issue_time": [NOW - timedelta(hours=3)] * 3,
                "lead_days": [0, 3, 1],
                "fxd__alpha__temp_max_c": [20.0, 21.0, None],
                "path__alpha__max": [20.5, None, 21.5],
            }
        ).write_parquet(config.dataset.dir / "daily_matrix_live.parquet")
        return config

    def test_layers_line_up_per_source(self, tmp_path):
        config = self.seeded_config(tmp_path)
        funnel = operations.build_funnel_rows(config, now=NOW)
        hourly = funnel.filter(
            (pl.col("granularity") == "hourly") & (pl.col("source") == "alpha")
        ).row(0, named=True)
        assert hourly["collector_rows"] == 4
        assert abs(hourly["collector_max_lead"] - 48.0) < 1e-9
        assert hourly["long_rows"] == 2
        assert hourly["long_max_lead"] == 47.0
        assert hourly["matrix_rows"] == 2
        assert hourly["matrix_max_lead"] == 46.0
        daily = funnel.filter(
            (pl.col("granularity") == "daily") & (pl.col("source") == "alpha")
        ).row(0, named=True)
        assert daily["collector_max_lead"] == 5.0
        assert daily["long_max_lead"] == 4.0
        # the native/path split: fxd reaches D3, path only D1
        assert daily["matrix_native_max_lead"] == 3.0
        assert daily["matrix_path_max_lead"] == 1.0
        assert daily["matrix_max_lead"] == 3.0
        assert daily["matrix_rows"] == 3

    def test_provider_model_pairs_use_the_grounded_slug(self, tmp_path):
        config = write_config(tmp_path)
        fetched = NOW - timedelta(hours=1)
        make_forecast_db(
            config.forecasts.db_path,
            [
                {
                    "completed_at": fetched.isoformat(),
                    "results": [
                        {
                            "provider": "storm",
                            "model": "sg",
                            "fetched_at": fetched.isoformat(),
                            "fetched_at_unix": int(fetched.timestamp()),
                            "daily": [
                                ((fetched + timedelta(days=3)).strftime("%Y-%m-%d"), {})
                            ],
                        }
                    ],
                }
            ],
        )
        funnel = operations.build_funnel_rows(config, now=NOW)
        assert funnel["source"].to_list() == ["storm_sg"]

    def test_nothing_anywhere_is_an_empty_typed_frame(self, tmp_path):
        config = write_config(tmp_path)
        funnel = operations.build_funnel_rows(config, now=NOW)
        assert funnel.is_empty()


class TestIdentityChanges:
    def test_first_report_is_the_baseline(self, tmp_path):
        config = write_config(tmp_path)
        assert operations.identity_changes(config, now=NOW).is_empty()
        snapshot = config.artifacts_dir / "observability" / "identity_snapshot.json"
        assert snapshot.exists()
        # unchanged identity on the next report records nothing either
        assert operations.identity_changes(config, now=NOW).is_empty()

    def test_config_change_names_the_keys(self, tmp_path):
        before = write_config(tmp_path)
        operations.identity_changes(before, now=NOW)
        after = write_config(tmp_path, extra_toml="\n[promotion]\nalpha = 0.2\n")
        changes = operations.identity_changes(after, now=NOW)
        assert changes.height == 1
        row = changes.row(0, named=True)
        assert row["kind"] == "config"
        assert "promotion.alpha" in row["detail"]

    def test_code_change_is_recorded(self, tmp_path, monkeypatch):
        config = write_config(tmp_path)
        operations.identity_changes(config, now=NOW)
        monkeypatch.setattr(operations, "code_identity", lambda: "0.0.0+next")
        changes = operations.identity_changes(config, now=NOW)
        assert changes["kind"].to_list() == ["code"]
        assert changes["to_value"].to_list() == ["0.0.0+next"]

    def test_secrets_never_reach_the_snapshot(self, tmp_path):
        config = write_config(
            tmp_path, extra_toml='\n[truth_qc]\nsynoptic_token = "SEKRIT"\n'
        )
        operations.identity_changes(config, now=NOW)
        snapshot = config.artifacts_dir / "observability" / "identity_snapshot.json"
        text = snapshot.read_text(encoding="utf-8")
        assert "SEKRIT" not in text
        assert "<redacted>" in text


class TestEvaluationCatalog:
    def test_row_summarizes_the_scores_file(self, tmp_path):
        path = tmp_path / "scores_hourly_live_expanding_abc123.parquet"
        path.write_bytes(b"x" * 2_000_000)
        fold1, fold2 = utc(2026, 7, 1), utc(2026, 7, 6)
        scores = pl.DataFrame(
            {
                "fold_origin": [fold1] * 3 + [fold2] * 5,
                "method_id": ["m1", "m2"] * 4,
                "issue_time": [utc(2026, 7, 2)] * 3 + [utc(2026, 7, 7)] * 5,
            }
        )
        row = operations.evaluation_catalog_row(path, scores)
        assert row["evaluation_id"] == "abc123"
        assert row["product"] == "hourly"
        assert row["source_kind"] == "live"
        assert row["window"] == "expanding"
        assert row["rows"] == 8
        assert row["n_methods"] == 2
        assert row["n_folds"] == 2
        assert row["fold_rows_min"] == 3
        assert row["fold_rows_median"] == 4.0
        assert row["issue_max"] == utc(2026, 7, 7)
        assert abs(row["file_size_mb"] - 2.0) < 1e-9

    def test_empty_scores_still_catalog(self, tmp_path):
        path = tmp_path / "scores_daily_live_expanding_e0.parquet"
        path.write_bytes(b"x")
        row = operations.evaluation_catalog_row(
            path, pl.DataFrame(schema={"fold_origin": pl.Datetime("us", "UTC")})
        )
        assert row["rows"] == 0
        assert row["n_folds"] == 0
        assert row["fold_rows_min"] is None


class TestRecordOperations:
    def test_ledgers_append_once_and_stay_idempotent(self, tmp_path):
        config = write_config(tmp_path, station_db="live_station.db")
        station_db(config, [NOW - timedelta(minutes=m) for m in range(30)])
        collector_with_leads(config)
        catalog_path = tmp_path / "scores_hourly_live_expanding_abc123.parquet"
        catalog_path.write_bytes(b"x")
        catalog_row = operations.evaluation_catalog_row(
            catalog_path,
            pl.DataFrame(schema={"fold_origin": pl.Datetime("us", "UTC")}),
        )
        first = operations.record_operations(
            config, runs_frame=runs_frame([]), catalog_rows=[catalog_row], now=NOW
        )
        report = operations.record_operations(
            config, runs_frame=runs_frame([]), catalog_rows=[catalog_row], now=NOW
        )
        assert first.freshness.height == 1
        assert report.freshness.height == 1
        pipeline = load_ledger(
            ledger_path(config, PIPELINE_LEDGER), PIPELINE_LEDGER.schema
        )
        assert pipeline.height == 1
        health = load_ledger(
            ledger_path(config, PROVIDER_HEALTH_LEDGER), PROVIDER_HEALTH_SCHEMA
        )
        assert health.height == 2
        catalog = load_ledger(
            ledger_path(config, EVALUATIONS_LEDGER), EVALUATIONS_LEDGER.schema
        )
        assert catalog["evaluation_id"].to_list() == ["abc123"]


def seeded_scores_dir(config, names_mtimes):
    scores_dir = config.dataset.dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    for name, age_days in names_mtimes:
        path = scores_dir / name
        path.write_bytes(b"x" * 1000)
        moment = (NOW - timedelta(days=age_days)).timestamp()
        os.utime(path, times=(moment, moment))
    return scores_dir


class TestPruneScores:
    def seeded(self, tmp_path):
        config = write_config(tmp_path)
        scores_dir = seeded_scores_dir(
            config,
            [
                (f"scores_hourly_live_expanding_e{i}.parquet", 10 - i)
                for i in range(6)  # e0 oldest ... e5 newest
            ]
            + [("scores_daily_live_expanding_d1.parquet", 1)],
        )
        # e1..e5 and d1 are cataloged; e0 predates the catalog
        rows = [
            {
                "recorded_at": NOW,
                "evaluation_id": evaluation_id,
                "file_name": name,
            }
            for evaluation_id, name in [
                (f"e{i}", f"scores_hourly_live_expanding_e{i}.parquet")
                for i in range(1, 6)
            ]
            + [("d1", "scores_daily_live_expanding_d1.parquet")]
        ]
        append_ledger(
            pl.DataFrame(
                [
                    {c: row.get(c) for c in EVALUATIONS_LEDGER.schema.names()}
                    for row in rows
                ],
                schema=dict(EVALUATIONS_LEDGER.schema),
            ),
            ledger_path(config, EVALUATIONS_LEDGER),
            EVALUATIONS_LEDGER,
            now=NOW,
        )
        releases = config.artifacts_dir / "releases"
        releases.mkdir(parents=True, exist_ok=True)
        (releases / "r1.json").write_text(
            json.dumps(
                {
                    "promoted_at": (NOW - timedelta(days=2)).isoformat(),
                    "evaluation_ids": ["e1"],
                }
            ),
            encoding="utf-8",
        )
        return config, scores_dir

    def test_retention_protection_and_catalog_guard(self, tmp_path):
        config, scores_dir = self.seeded(tmp_path)
        preview = operations.prune_scores_files(config, dry_run=True, now=NOW)
        # dry run deletes nothing
        assert (scores_dir / "scores_hourly_live_expanding_e2.parquet").exists()
        result = operations.prune_scores_files(config, dry_run=False, now=NOW)
        assert [p.name for p in preview.deleted] == [p.name for p in result.deleted]
        # newest three of the hourly group survive
        for kept in ("e3", "e4", "e5"):
            assert (
                scores_dir / f"scores_hourly_live_expanding_{kept}.parquet"
            ).exists()
        # e1 is referenced by a recent release; e0 predates the catalog
        assert (scores_dir / "scores_hourly_live_expanding_e1.parquet").exists()
        assert (scores_dir / "scores_hourly_live_expanding_e0.parquet").exists()
        assert any("e0" in note for note in result.skipped)
        # d1 is the newest of its own group
        assert (scores_dir / "scores_daily_live_expanding_d1.parquet").exists()
        # only e2 goes
        assert [p.name for p in result.deleted] == [
            "scores_hourly_live_expanding_e2.parquet"
        ]
        assert not (scores_dir / "scores_hourly_live_expanding_e2.parquet").exists()
        assert result.freed_mb > 0


def pipeline_history(samples, days):
    rows = [
        {
            "recorded_at": NOW - timedelta(days=day),
            "as_of_date": (NOW - timedelta(days=day)).date(),
            "truth_samples_24h": samples,
        }
        for day in range(1, days + 1)
    ]
    return pl.DataFrame(
        [
            {c: row.get(c) for c in operations.evidence.PIPELINE_SCHEMA.names()}
            for row in rows
        ],
        schema=dict(operations.evidence.PIPELINE_SCHEMA),
    )


class TestBaselineTruthAlarm:
    def test_half_rate_flags_against_own_median(self):
        # the Aug 4-6 half-rate episode: ~810/day against a ~1900 norm
        notes = operations._baseline_truth_alarm(810, pipeline_history(1900, 5), NOW)
        assert notes == ["thin truth vs baseline (810 < 70% of 14d median 1900)"]

    def test_mild_dip_stays_quiet(self):
        # 1500 > 0.7 * 1900 = 1330
        assert (
            operations._baseline_truth_alarm(1500, pipeline_history(1900, 5), NOW) == []
        )

    def test_thin_baseline_stays_quiet(self):
        assert (
            operations._baseline_truth_alarm(810, pipeline_history(1900, 2), NOW) == []
        )

    def test_no_history_stays_quiet(self):
        assert operations._baseline_truth_alarm(810, None, NOW) == []
        assert operations._baseline_truth_alarm(810, pl.DataFrame(), NOW) == []
