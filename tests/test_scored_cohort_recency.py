"""Score clocks describe scored forecasts; tied batches never choose by row order."""

import json
import random
from datetime import timedelta

import polars as pl
import pytest
from test_daily_read import NOW, START, quality, read, setup, write_json, write_quality
from test_daily_reader_regressions import modern_release, temperature

from grounded_weather_forecast import daily_read as reader
from grounded_weather_forecast.reports import evidence, verification
from grounded_weather_forecast.serve.history import HISTORY_SCHEMA


def test_recency_is_computed_after_truth_matching_and_sample_qualification(tmp_path):
    rows = []
    for index in range(8):
        rows.append(
            {
                "issued_at": START + timedelta(minutes=index),
                "valid_time": START,
                "lead_hours": 2.0,
                "product": "hourly",
                "variable": "temp_c",
                "truth_semantics": "inst",
                "method_id": "equal_weight",
                "dataset_fingerprint": "old",
                "release_id": "release",
                "y_pred": 1.0,
            }
        )
    # A more recent issuance without truth must not advance scored-cohort recency.
    rows.append(rows[-1] | {"issued_at": NOW, "valid_time": NOW})
    path = tmp_path / "history.parquet"
    pl.DataFrame(rows, schema=HISTORY_SCHEMA).write_parquet(path)
    truth = pl.DataFrame({"valid_hour": [START], "t__temp_c__inst": [0.0]})
    scored = verification.verify_history(path, truth)
    assert scored["latest_scored_issue_at"].to_list() == [START + timedelta(minutes=7)]
    assert scored["n"].to_list() == [8]
    written = evidence.served_quality_rows(
        scored, config_fingerprint_value="config", code_version="recorder", now=NOW
    )
    assert (
        written["latest_scored_issue_at"].to_list()
        == scored["latest_scored_issue_at"].to_list()
    )


def test_recency_precedes_forecast_dataset_after_batch_clocks_tie(tmp_path):
    config, forecast, _ = setup(tmp_path)
    modern_release(config)
    rows = [
        quality(
            config,
            realized=True,
            dataset_fingerprint=dataset,
            latest_scored_issue_at=clock,
            live_mae=score,
        )
        for dataset, clock, score in [
            ("dataset1", START - timedelta(days=2), 1.0),
            ("yesterday", START - timedelta(days=1), 3.0),
        ]
    ]
    for seed in range(5):
        random.Random(seed).shuffle(rows)
        write_quality(config, rows, realized=True)
        result = temperature(config, forecast, realized=True)
        assert result["evidence_dataset_fingerprint"] == "yesterday"
        assert result["mae"] == 3.0
        assert result["latest_scored_issue_at"] == START - timedelta(days=1)
        assert result["release_id"] == "release1"


@pytest.mark.parametrize(
    "missing", [None, "bad", "2026-10-20T00:00:00", NOW + timedelta(days=1)]
)
def test_missing_or_invalid_clock_does_not_order_a_legacy_tie(tmp_path, missing):
    config, forecast, _ = setup(tmp_path)
    rows = [
        quality(config, realized=True, dataset_fingerprint=d, latest_scored_issue_at=c)
        for d, c in [("old", START - timedelta(days=1)), ("other", missing)]
    ]
    # Write raw text clocks, preserving malformed values rather than fixture casting them away.
    frame = pl.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key != "latest_scored_issue_at"
            }
            for row in rows
        ]
    ).with_columns(
        pl.Series(
            "latest_scored_issue_at",
            [
                (START - timedelta(days=1)).isoformat(),
                missing.isoformat() if hasattr(missing, "isoformat") else missing,
            ],
            dtype=pl.String,
        )
    )
    path = config.artifacts_dir / "history/served_quality.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    result = temperature(config, forecast, realized=True)
    assert result["limitations"] == ["ambiguous_quality_evidence"]


def test_equal_recency_uses_dataset_preference_and_legacy_columns_are_optional(
    tmp_path,
):
    config, forecast, _ = setup(tmp_path)
    rows = [
        quality(
            config,
            realized=True,
            dataset_fingerprint=d,
            latest_scored_issue_at=START - timedelta(days=1),
            live_mae=m,
        )
        for d, m in [("dataset1", 2.0), ("old", 8.0)]
    ]
    write_quality(config, rows, realized=True)
    assert temperature(config, forecast, realized=True)["mae"] == 2.0
    path = config.artifacts_dir / "history/served_quality.parquet"
    pl.read_parquet(path).drop("latest_scored_issue_at").write_parquet(path)
    row = temperature(config, forecast, realized=True)
    assert row["mae"] == 2.0 and row["latest_scored_issue_at"] is None


def test_distinct_dataset_cohorts_append_independently_but_first_write_is_preserved(
    tmp_path,
):
    config, _, _ = setup(tmp_path)
    path = evidence.ledger_path(config, evidence.SERVED_QUALITY_LEDGER)
    for dataset, mae in [("a", 1.0), ("b", 2.0), ("a", 9.0)]:
        frame = pl.DataFrame(
            [quality(config, realized=True, dataset_fingerprint=dataset, live_mae=mae)],
            schema=evidence.SERVED_QUALITY_SCHEMA,
        )
        evidence.append_ledger(frame, path, evidence.SERVED_QUALITY_LEDGER, now=NOW)
    loaded = evidence.read_ledger(path, evidence.SERVED_QUALITY_SCHEMA)
    assert loaded.sort("dataset_fingerprint")["live_mae"].to_list() == [1.0, 2.0]


def test_fallback_checks_all_ties_but_never_reads_older_irrelevant_releases(
    tmp_path, monkeypatch
):
    config, forecast, _ = setup(tmp_path)
    modern_release(config)
    for identifier in ("first", "second", "irrelevant"):
        path, raw = modern_release(config, identifier)
        raw["promoted_at"] = (START - timedelta(days=5)).isoformat()
        write_json(path, raw)
    rows = [
        quality(
            config,
            realized=True,
            release_id=identifier,
            dataset_fingerprint=identifier,
            recorded_at=START
            if identifier != "irrelevant"
            else START - timedelta(days=1),
            latest_scored_issue_at=START - timedelta(hours=hours),
        )
        for identifier, hours in [("first", 3), ("second", 1), ("irrelevant", 0)]
    ]
    write_quality(config, rows, realized=True)
    original = reader.Snapshot.document
    reads = []

    def document(self, path, *args, **kwargs):
        if path.parent.name == "releases":
            reads.append(path.name)
        return original(self, path, *args, **kwargs)

    monkeypatch.setattr(reader.Snapshot, "document", document)
    row = temperature(config, forecast, realized=True)
    assert row["evidence_release_id"] == "second"
    assert set(reads) == {"release1.json", "first.json", "second.json"}


def test_reductions_coalesce_duplicates_and_keep_distinct_ties(tmp_path):
    config, _, _ = setup(tmp_path)
    a = quality(config, realized=True, dataset_fingerprint="a")
    b = a | {"live_mae": 9.0}
    old = a | {
        "recorded_at": START - timedelta(days=1),
        "as_of_date": (START - timedelta(days=1)).date(),
    }
    write_quality(config, [a, a, b, old], realized=True)
    frame = reader._quality_rows(reader.Snapshot(), config, NOW, realized=True)
    assert frame.height == 2 and sorted(frame["live_mae"]) == [2.0, 9.0]
