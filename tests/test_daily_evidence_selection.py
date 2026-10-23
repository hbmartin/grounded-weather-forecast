"""Daily evidence must be deterministic, attributed and independently retryable."""

import json
import random
import sqlite3
import subprocess
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest
from test_daily_read import NOW, START, quality, read, setup, write_json, write_quality
from test_daily_reader_regressions import modern_release, temperature

from grounded_weather_forecast import daily_read as reader
from grounded_weather_forecast.serve.provenance import (
    selection_code_version,
    selection_truth_semantics,
)


def older_release(config, identifier="olderrelease"):
    path, raw = modern_release(config, identifier)
    raw["promoted_at"] = (START - timedelta(days=3)).isoformat()
    write_json(path, raw)
    return path, raw


def test_tied_realized_cohorts_prefer_forecast_dataset_across_processes(tmp_path):
    config, forecast, _ = setup(tmp_path)
    modern_release(config)
    rows = [
        quality(config, realized=True, dataset_fingerprint=dataset, live_mae=mae)
        for dataset, mae in (("dataset1", 1.0), ("other", 9.0))
    ]
    for seed in range(4):
        random.Random(seed).shuffle(rows)
        write_quality(config, rows, realized=True)
        row = temperature(config, forecast, realized=True)
        assert (row["mae"], row["evidence_dataset_fingerprint"]) == (1.0, "dataset1")
    script = """
import json,sys
from pathlib import Path
from grounded_weather_forecast.config import load_config
from grounded_weather_forecast.daily_read import read_daily_report,timestamp,json_default
c=load_config(Path(sys.argv[1]))
r=read_daily_report(c,forecast_path=Path(sys.argv[2]),start=timestamp(sys.argv[3]),as_of=timestamp(sys.argv[4]),published_station=Path(sys.argv[2]).parent/'published.json')
print(json.dumps(r['prediction_quality']['realized_live'],default=json_default,sort_keys=True))
"""
    outputs = [
        subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(tmp_path / "config.toml"),
                str(forecast),
                START.isoformat(),
                NOW.isoformat(),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for _ in range(3)
    ]
    assert len(set(outputs)) == 1


@pytest.mark.parametrize("conflict", ["dataset", "score", "recorder"])
def test_ambiguous_exact_evidence_abstains_without_older_fallback(tmp_path, conflict):
    config, forecast, _ = setup(tmp_path)
    modern_release(config)
    older_release(config)
    first = quality(config, realized=True, dataset_fingerprint="other")
    second = (
        first | {"dataset_fingerprint": "another"}
        if conflict == "dataset"
        else first | {"live_mae": 8.0}
        if conflict == "score"
        else first | {"code_version": "recorder2"}
    )
    write_quality(
        config,
        [first, second, quality(config, realized=True, release_id="olderrelease")],
        realized=True,
    )
    row = temperature(config, forecast, realized=True)
    assert row["status"] == "unknown"
    assert row["limitations"] == ["ambiguous_quality_evidence"]


def test_identical_duplicates_coalesce_and_newer_clock_precedes_dataset(tmp_path):
    config, forecast, _ = setup(tmp_path)
    row = quality(config, realized=True)
    newer = row | {
        "recorded_at": NOW,
        "dataset_fingerprint": "older-serving-dataset",
        "live_mae": 3.0,
    }
    write_quality(config, [row, row, newer, newer], realized=True)
    result = temperature(config, forecast, realized=True)
    assert result["mae"] == 3.0
    assert result["evidence_dataset_fingerprint"] == "older-serving-dataset"


def test_realized_recorder_code_does_not_replace_evaluation_identity(tmp_path):
    config, forecast, _ = setup(tmp_path)
    modern_release(config)
    write_quality(
        config,
        [quality(config, realized=True, code_version="new-recorder")],
        realized=True,
    )
    assert temperature(config, forecast, realized=True)["status"] == "available"
    write_quality(config, [quality(config, code_version="new-recorder")])
    assert temperature(config, forecast)["status"] == "unknown"


@pytest.mark.parametrize(
    "clock",
    [
        None,
        "invalid",
        "2026-10-19T00:00:00",
        "0001-01-01T00:00:00+01:00",
        (START - timedelta(days=2)).isoformat(),
        (START - timedelta(days=1)).isoformat(),
    ],
)
@pytest.mark.parametrize("target", [False, True])
def test_fallback_requires_strictly_older_valid_content_clocks(tmp_path, clock, target):
    config, forecast, _ = setup(tmp_path)
    target_path, target_raw = modern_release(config)
    old_path, old_raw = older_release(config)
    path, raw = (target_path, target_raw) if target else (old_path, old_raw)
    # For target clocks, publication must precede the older source to reject it.
    raw["promoted_at"] = (
        clock
        if not target or clock is None or not clock.startswith("2026-10-2")
        else (START - timedelta(days=4)).isoformat()
    )
    write_json(path, raw)
    write_quality(
        config,
        [quality(config, realized=True, release_id="olderrelease")],
        realized=True,
    )
    assert temperature(config, forecast, realized=True)["status"] == "unknown"


def test_older_fallback_allows_new_recorder_and_preserves_attribution(tmp_path):
    config, forecast, _ = setup(tmp_path)
    modern_release(config)
    older_release(config)
    write_quality(
        config,
        [
            quality(
                config,
                realized=True,
                release_id="olderrelease",
                code_version="recorder2",
                dataset_fingerprint="served-old",
            )
        ],
        realized=True,
    )
    row = temperature(config, forecast, realized=True)
    assert (
        row["release_id"],
        row["evidence_release_id"],
        row["evidence_dataset_fingerprint"],
    ) == ("release1", "olderrelease", "served-old")


def test_method_bearing_missing_release_stays_visible(tmp_path):
    config, forecast, raw = setup(tmp_path)
    for point in raw["hourly"]:
        point["release_ids"].pop("temp_c")
    write_json(forecast, raw)
    write_quality(
        config, [quality(config, variable=v.name) for v in reader.HOURLY_VARIABLES]
    )
    result = read(config, forecast)["prediction_quality"]["backtest"]
    assert result["status"] == "partial" and len(result["slices"]) == 8
    temp = next(row for row in result["slices"] if row["variable"] == "temp_c")
    assert temp["release_id"] is None and temp["status"] == "unknown"


@pytest.mark.parametrize("identifier", ["", "unknown", 1, [], {}])
def test_invalid_evaluation_ids_never_borrow_context(identifier):
    release = {
        "evaluation_contexts": [
            {
                "evaluation_id": "other",
                "code_version": "code",
                "semantics": {"temp_c": "inst"},
            }
        ]
    }
    selection = {"evaluation_id": identifier}
    assert selection_code_version(release, selection) is None
    assert selection_truth_semantics(release, selection, "temp_c") is None


def test_absent_legacy_ids_recover_only_unambiguous_recorded_values():
    context = {"code_version": "old-code", "semantics": {"temp_c": "inst"}}
    release = {"evaluation_contexts": [context]}
    assert selection_code_version(release, {}) == "old-code"
    assert selection_truth_semantics(release, {}, "temp_c") == "inst"
    release["evaluation_contexts"].append(context | {"code_version": "different"})
    assert selection_code_version(release, {}) is None
    assert selection_code_version(release, {"legacy_context_missing": True}) is None
    assert (
        selection_code_version(
            release, {"evaluation_id": "", "code_version": "recorded"}
        )
        == "recorded"
    )


def test_component_first_changed_during_retry_gets_its_own_retry(tmp_path, monkeypatch):
    config, forecast, _ = setup(tmp_path)
    write_json(tmp_path / "published.json", {"metadata": {"observed_at": START}})
    changes = iter([{forecast}, {config.station.db_path}, set()])
    monkeypatch.setattr(reader.Snapshot, "changed_paths", lambda _s: next(changes))
    calls = Counter()
    original = reader.read_station

    def station(*args, **kwargs):
        calls["station"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(reader, "read_station", station)
    result = read(config, forecast)
    assert calls["station"] == 2
    assert (
        result["operational_health"]["station"]["source"]
        == "published_station_snapshot"
    )
    assert "concurrent_replacement_after_retry" not in result["limitations"]


def test_shared_unstable_dependencies_exhaust_only_two_reads(tmp_path, monkeypatch):
    snapshot = reader.Snapshot()
    path = tmp_path / "shared"
    calls = Counter()

    def reading(name):
        calls[name] += 1
        snapshot.track(path)
        return {"status": "ok"}

    monkeypatch.setattr(snapshot, "changed_paths", lambda: {path})
    result = {}
    assert reader._read_components(
        snapshot,
        {name: lambda name=name: reading(name) for name in ("station", "providers")},
        result,
    )
    assert calls == {"station": 2, "providers": 2}
    assert all(row["status"] == "unknown" for row in result.values())


def test_wal_header_rejection_reads_only_header(tmp_path, monkeypatch):
    path = tmp_path / "wal.db"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE sample(value BLOB)")
    connection.execute("INSERT INTO sample VALUES (zeroblob(2000000))")
    connection.commit()
    connection.close()
    original = Path.open
    reads = []

    class Source:
        def __enter__(self):
            self.source = original(path, "rb")
            return self

        def __exit__(self, *args):
            self.source.close()

        def read(self, size=-1):
            reads.append(size)
            return self.source.read(size)

    monkeypatch.setattr(
        Path,
        "open",
        lambda self, *a, **k: Source() if self == path else original(self, *a, **k),
    )
    with pytest.raises(reader.SourceError, match="sqlite_wal_source_unsafe"):
        reader._open_readonly(path)
    assert reads == [100]


def test_rollback_database_body_is_not_retained_in_report_cache(tmp_path):
    path = tmp_path / "db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE sample(value INTEGER)")
    connection.close()
    snapshot = reader.Snapshot()
    connection = reader._open_readonly(path, snapshot)
    assert connection.execute("SELECT * FROM sample").fetchall() == []
    connection.close()
    assert path in snapshot.signatures and path not in snapshot._bytes


@pytest.mark.parametrize("bad_clock", [None, "2020-01-01T00:00:00", "invalid"])
def test_bad_run_row_is_command_local_and_cannot_hide_newer_uncertainty(
    tmp_path, bad_clock
):
    config, _, _ = setup(tmp_path)
    good = {
        "command": "predict",
        "started_at": START.isoformat(),
        "ended_at": NOW.isoformat(),
        "exit_code": 0,
    }
    broken = {
        "command": "report",
        "started_at": bad_clock,
        "ended_at": (START - timedelta(days=90)).isoformat(),
        "exit_code": 0,
    }
    pl.DataFrame([good, broken]).write_parquet(config.dataset.dir / "runs.parquet")
    result = reader.read_runs(
        reader.Snapshot(), config.dataset.dir / "runs.parquet", NOW
    )
    assert result["status"] == "partial" and result["limitations"] == [
        "invalid_run_rows"
    ]
    assert {row["command"]: row["status"] for row in result["latest_by_command"]} == {
        "predict": "ok",
        "report": "unknown",
    }


def test_older_invalid_run_can_be_superseded_but_unorderable_run_cannot(tmp_path):
    config, _, _ = setup(tmp_path)
    path = config.dataset.dir / "runs.parquet"
    good = {
        "command": "predict",
        "started_at": START.isoformat(),
        "ended_at": NOW.isoformat(),
        "exit_code": 0,
    }
    broken = good | {
        "started_at": None,
        "ended_at": (START - timedelta(days=90)).isoformat(),
    }
    for ended, expected in ((broken["ended_at"], "ok"), ("bad-clock", "unknown")):
        pl.DataFrame([good, broken | {"ended_at": ended}]).write_parquet(path)
        result = reader.read_runs(reader.Snapshot(), path, NOW)
        assert result["latest_by_command"][0]["status"] == expected


def test_irrelevant_bad_run_and_empty_history(tmp_path):
    config, _, _ = setup(tmp_path)
    path = config.dataset.dir / "runs.parquet"
    pl.DataFrame(
        {
            "command": ["other"],
            "started_at": [None],
            "ended_at": ["bad"],
            "exit_code": [None],
        }
    ).write_parquet(path)
    result = reader.read_runs(reader.Snapshot(), path, NOW)
    assert result["limitations"] == ["no_completed_runs_as_of"]
