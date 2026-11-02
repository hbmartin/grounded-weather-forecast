"""Offline-only checks for the inactive prospective shadow recorder."""

import base64
import importlib
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def shadow(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("temperature_shadow")


@pytest.fixture
def frames():
    origin = datetime(2026, 10, 1, 12, tzinfo=UTC)
    valid = [origin - timedelta(hours=240 - i) for i in range(200)]
    target = np.sin(np.arange(200) / 8) + 15
    train = pl.DataFrame(
        {
            "issue_time": [v - timedelta(hours=12) for v in valid],
            "valid_time": valid,
            "lead_hours": [12.0] * 200,
            "lead_bucket": ["12-24h"] * 200,
            "valid_hour_local": [v.hour for v in valid],
            "fx__a__temp_c": target + 2,
            "fx__b__temp_c": target - 1,
            "t__temp_c__mean": target,
            "source_kind": ["live"] * 200,
        }
    )
    future = [origin + timedelta(hours=i) for i in (1, 13)]
    issue = pl.DataFrame(
        {
            "issue_time": [origin] * 2,
            "valid_time": future,
            "lead_hours": [1.0, 13.0],
            "lead_bucket": ["1-3h", "12-24h"],
            "valid_hour_local": [v.hour for v in future],
            "fx__a__temp_c": [17.0, 18.0],
            "fx__b__temp_c": [14.0, 15.0],
            "obs__temp_c": [15.0, 15.0],
            "source_kind": ["live"] * 2,
        }
    )
    truth = train.select(pl.col("valid_time").alias("valid_hour"), "t__temp_c__mean")
    return origin, train, issue, truth


def test_future_realizations_cannot_change_shadow_fit(shadow, frames):
    origin, train, issue, truth = frames
    values, state = shadow.predictions(train, issue, truth, origin)
    future = train.tail(1).with_columns(
        pl.lit(origin + timedelta(hours=1)).alias("valid_time"),
        pl.lit(origin - timedelta(hours=11)).alias("issue_time"),
        pl.lit(9999.0).alias("t__temp_c__mean"),
    )
    later, later_state = shadow.predictions(
        pl.concat([train, future]), issue, truth, origin
    )
    np.testing.assert_array_equal(values["candidate"], later["candidate"])
    assert state == later_state


def test_new_known_realizations_update_correction(shadow, frames):
    origin, train, issue, truth = frames
    _, before = shadow.predictions(train, issue, truth, origin)
    new = train.tail(1).with_columns(
        pl.lit(origin - timedelta(hours=3)).alias("valid_time"),
        pl.lit(origin - timedelta(hours=15)).alias("issue_time"),
        pl.lit(25.0).alias("t__temp_c__mean"),
    )
    _, after = shadow.predictions(pl.concat([train, new]), issue, truth, origin)
    assert before["corrections"] != after["corrections"]
    assert after["unique_train_hours"] == before["unique_train_hours"] + 1


def test_missing_provider_weights_are_renormalized(shadow):
    np.testing.assert_allclose(
        shadow.safe_blend(
            np.array([[1.0, np.nan], [np.nan, 3.0]]), np.array([0.8, 0.2])
        ),
        [1, 3],
    )


def publisher_receipt(shadow, capsule, public):
    public.update(schema_version=2, publisher_version="2.3.0")
    payload = json.dumps(public).encode()
    return {
        "schema_version": 1,
        "commit_binding": shadow.BINDING,
        "commit": "a" * 40,
        "forecast_base64": base64.b64encode(payload).decode(),
        "forecast_sha256": __import__("hashlib").sha256(payload).hexdigest(),
        "source_sha256": shadow.digest(capsule / "production.json"),
        "issued_at": public["issued_at"],
        "published_at": public["published_at"],
        "remote": "fixture",
        "branch": "data",
        "publisher_version": "2.3.0",
        "public_schema_version": 2,
    }


def bind_public(shadow, capsule, public, receipt):
    receipt.update(
        remote="fixture",
        branch="data",
        artifact_version=3,
        commit_binding=shadow.BINDING,
        publisher_receipt=publisher_receipt(shadow, capsule, public),
    )
    (capsule / "published.json").write_text(json.dumps(public))
    receipt["published_sha256"] = shadow.digest(capsule / "published.json")
    return receipt


def make_capsule(shadow, frames, tmp_path):
    origin, train, issue, truth = frames
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    train.write_parquet(capsule / "training.parquet")
    issue.write_parquet(capsule / "issue.parquet")
    truth.write_parquet(capsule / "truth_hourly.parquet")
    pl.DataFrame({"completed_at": [origin]}).write_parquet(
        capsule / "provider_completions.parquet"
    )
    document = {
        "issued_at": origin.isoformat(),
        "observation_at": (origin - timedelta(minutes=31)).isoformat(),
        "dataset_fingerprint": "fixture",
        "release_ids": ["fixture"],
        "hourly": [
            {
                "valid_time": v.isoformat(),
                "values": {"temp_c": 15.0},
                "methods": {"temp_c": "fixture"},
                "truth_semantics": {"temp_c": "mean"},
            }
            for v in issue["valid_time"]
        ],
    }
    (capsule / "production.json").write_text(json.dumps(document))
    (capsule / "capsule.json").write_text(
        json.dumps(
            {
                "artifact_version": 3,
                "input_as_of": origin.isoformat(),
                "capture_kind": "historical_reconstruction",
            }
        )
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        '{"window_start":"2026-11-02T08:00:00+00:00","window_end":"2026-12-14T08:00:00+00:00"}'
    )
    protocol.with_suffix(".lock.json").write_text(
        json.dumps(shadow.implementation_lock(protocol))
    )
    return capsule, protocol


def test_record_is_immutable_and_stale_observation_abstains(shadow, frames, tmp_path):
    capsule, protocol = make_capsule(shadow, frames, tmp_path)
    store = tmp_path / "store"
    target = shadow.record(capsule, store, protocol, mode="replay")
    record = json.loads((target / "record.json").read_text())
    assert record["mode"] == "replay"
    assert all(row["persistence"] is None for row in record["predictions"])
    with pytest.raises(FileExistsError):
        shadow.record(capsule, store, protocol, mode="replay")


def test_protocol_edit_requires_new_enrollment(shadow, frames, tmp_path):
    capsule, protocol = make_capsule(shadow, frames, tmp_path)
    protocol.write_text(protocol.read_text() + "\n")
    with pytest.raises(ValueError, match="frozen protocol"):
        shadow.record(capsule, tmp_path / "store", protocol, mode="replay")


def test_history_candidate_is_not_publication_evidence(shadow, frames, tmp_path):
    capsule, _ = make_capsule(shadow, frames, tmp_path)
    with pytest.raises(ValueError, match="committed payload"):
        shadow.production_rows(capsule, {}, prospective=True)


@pytest.mark.parametrize("temperature", [15.0, 15.123, 15.00277])
def test_committed_payload_validation_uses_public_rounded_temperature(
    shadow, frames, tmp_path, temperature
):
    capsule, _ = make_capsule(shadow, frames, tmp_path)
    document = json.loads((capsule / "production.json").read_text())
    for row in document["hourly"]:
        row["values"]["temp_c"] = temperature
    (capsule / "production.json").write_text(json.dumps(document))
    published = {
        **document,
        "published_at": document["issued_at"],
        "units": {"temp_f": "F"},
        "truth_semantics": {"temp_f": "mean"},
        "hourly": [
            {
                "valid_time": row["valid_time"],
                "values": {"temp_f": round(temperature * 9.0 / 5.0 + 32.0, 2)},
                "methods": {"temp_f": "fixture"},
            }
            for row in document["hourly"]
        ],
    }
    (capsule / "published.json").write_text(json.dumps(published))
    provenance = dict.fromkeys(
        (
            "production_code_identity",
            "config_fingerprint",
            "availability_assumptions",
            "provider_completion_clock",
            "station_receipt_clock",
        ),
        "explicit fixture assumption",
    )
    provenance["dataset_fingerprint"] = "fixture"
    receipt = {
        "commit": "a" * 40,
        "issued_at": document["issued_at"],
        "published_at": document["issued_at"],
        "source_sha256": shadow.digest(capsule / "production.json"),
        "published_sha256": shadow.digest(capsule / "published.json"),
    }
    bind_public(shadow, capsule, published, receipt)
    meta = {"publication": receipt, "provenance": provenance}
    _, rows = shadow.production_rows(capsule, meta, prospective=True)
    assert all(
        row["values"]["temp_c"]
        == pytest.approx((round(temperature * 9.0 / 5.0 + 32.0, 2) - 32) / 1.8)
        for row in rows.values()
    )
    published["hourly"][0]["values"]["temp_f"] = 100.0
    (capsule / "published.json").write_text(json.dumps(published))
    bind_public(shadow, capsule, published, receipt)
    with pytest.raises(ValueError, match="temperature differs"):
        shadow.production_rows(capsule, meta, prospective=True)


def test_replay_never_enters_prospective_analysis(shadow, frames, tmp_path):
    capsule, protocol = make_capsule(shadow, frames, tmp_path)
    store = tmp_path / "store"
    shadow.record(capsule, store, protocol, mode="replay")
    with pytest.raises(ValueError, match="replay is excluded"):
        shadow.evaluate(
            store,
            capsule / "truth_hourly.parquet",
            protocol,
            tmp_path / "scores",
            mode="prospective",
        )


def test_calendar_block_interval_preserves_constant_difference_with_gaps(shadow):
    daily = pl.DataFrame(
        {
            "day": [datetime(2026, 10, i, tzinfo=UTC).date() for i in (1, 2, 8, 9)],
            "difference": [0.2] * 4,
        }
    )
    protocol = {
        "uncertainty": {
            "seed": 1,
            "bootstrap_samples": 200,
            "family_comparisons": 12,
            "two_sided_family_alpha": 0.05,
        }
    }
    np.testing.assert_allclose(
        shadow.interval(daily, "difference", block=3, protocol=protocol), [0.2, 0.2]
    )


def test_secondary_hour_metrics_do_not_count_repeated_issuances_as_new_hours(shadow):
    first = datetime(2026, 10, 1, 12, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "valid_time": [first] * 10 + [first + timedelta(hours=1)],
            "horizon": ["0-6h"] * 11,
            "truth": [0.0] * 11,
            **{c: [0.0] * 10 + [2.0] for c in shadow.METHODS},
        }
    )
    protocol = {
        "window_end": "2026-12-14T08:00:00+00:00",
        "quality": {
            "minimum_common_coverage": 0.8,
            "minimum_matched_utc_dates_per_horizon": 28,
            "minimum_matched_valid_hours_per_horizon": 300,
            "minimum_weeks_with_data": 4,
        },
        "uncertainty": {
            "seed": 1,
            "bootstrap_samples": 100,
            "primary_block_days": 3,
            "sensitivity_block_days": 7,
            "family_comparisons": 12,
            "two_sided_family_alpha": 0.05,
        },
    }
    result = shadow.score_horizon(frame, "0-6h", "replay", 1.0, protocol)
    assert result["hourly_mae"]["candidate"] == 1.0
    assert result["bias"]["candidate"] == 1.0
    assert result["common_rows"] == 11
    assert result["unique_hours"] == 2


def template_protocol(shadow):
    return json.loads(
        (
            Path(shadow.__file__).parents[1]
            / "examples/temperature-shadow/protocol.json"
        ).read_text()
    )


def test_freeze_requires_future_window_and_immutable_enrollment(shadow, tmp_path):
    protocol = template_protocol(shadow)
    start = datetime.now(UTC) + timedelta(days=1)
    protocol.update(
        window_start=start.isoformat(),
        window_end=(start + timedelta(days=42)).isoformat(),
    )
    source = tmp_path / "proposal.json"
    source.write_text(json.dumps(protocol))
    target = shadow.freeze(source, tmp_path / "enrollment")
    assert json.loads(
        target.with_suffix(".lock.json").read_text()
    ) == shadow.implementation_lock(target)
    with pytest.raises(FileExistsError):
        shadow.freeze(source, tmp_path / "enrollment")
    protocol.update(
        window_start="2020-01-01T00:00:00+00:00", window_end="2020-02-12T00:00:00+00:00"
    )
    source.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="has started"):
        shadow.freeze(source, tmp_path / "late")


def test_freeze_rejects_changed_candidate_rules(shadow, tmp_path):
    protocol = template_protocol(shadow)
    protocol["quality"]["minimum_common_coverage"] = 0
    source = tmp_path / "proposal.json"
    source.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="rules are fixed"):
        shadow.freeze(source, tmp_path / "enrollment")


def test_input_hashes_reject_modified_packets(shadow, frames, tmp_path):
    capsule, _ = make_capsule(shadow, frames, tmp_path)
    metadata = json.loads((capsule / "capsule.json").read_text())
    metadata["input_sha256"] = {
        "issue.parquet": shadow.digest(capsule / "issue.parquet")
    }
    (capsule / "capsule.json").write_text(json.dumps(metadata))
    assert shadow.capsule_metadata(capsule)["input_sha256"]
    (capsule / "issue.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="input changed"):
        shadow.capsule_metadata(capsule)


@pytest.mark.parametrize(
    "name",
    [
        "../outside",
        "nested/../outside",
        "/outside",
        r"C:\outside",
        "D:outside",
        "issue.parquet:stream",
        "missing.parquet",
        r"..\outside",
        ".",
        "..",
        "",
    ],
)
def test_capsule_manifest_rejects_nonlocal_names_before_hashing(
    shadow, frames, tmp_path, monkeypatch, name
):
    capsule, _ = make_capsule(shadow, frames, tmp_path)
    metadata = json.loads((capsule / "capsule.json").read_text())
    metadata["input_sha256"] = {name: "fixture"}
    (capsule / "capsule.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(shadow, "digest", lambda *_a: pytest.fail("unsafe hash read"))
    with pytest.raises(ValueError, match="capsule filename"):
        shadow.capsule_metadata(capsule)


@pytest.mark.parametrize(
    "name", ["capsule.json", "publication.json", "issue.parquet", "unlisted.parquet"]
)
def test_capsule_rejects_declared_and_undeclared_symlinks(
    shadow, frames, tmp_path, name
):
    capsule, _ = make_capsule(shadow, frames, tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    target = capsule / name
    target.unlink(missing_ok=True)
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="capsule symlink"):
        shadow.capsule_metadata(capsule)


def test_operator_can_choose_a_capsule_root_alias(shadow, frames, tmp_path):
    capsule, _ = make_capsule(shadow, frames, tmp_path)
    alias = tmp_path / "operator-selected-alias"
    alias.symlink_to(capsule, target_is_directory=True)
    assert shadow.capsule_metadata(alias) == shadow.capsule_metadata(capsule)


@pytest.mark.parametrize("name", ["issue.parquet", "unlisted"])
def test_capsule_rejects_directories_before_parquet_reads(
    shadow, frames, tmp_path, name
):
    capsule, _ = make_capsule(shadow, frames, tmp_path)
    target = capsule / name
    target.unlink(missing_ok=True)
    target.mkdir()
    (target / "nested.parquet").symlink_to(tmp_path / "outside.parquet")
    with pytest.raises(ValueError, match="capsule child must be a regular file"):
        shadow.capsule_metadata(capsule)


def test_finalize_cannot_reconstruct_a_missing_push_receipt(shadow, frames, tmp_path):
    capsule, protocol = make_capsule(shadow, frames, tmp_path)
    with pytest.raises(ValueError, match="missing original successful-push receipt"):
        shadow.finalize(
            capsule, tmp_path / "records", protocol, source=None, mode="replay"
        )


def test_finalize_refuses_relabelled_held_source(shadow, frames, tmp_path):
    capsule, protocol = make_capsule(shadow, frames, tmp_path)
    other = tmp_path / "held.json"
    other.write_text("{}")
    with pytest.raises(ValueError, match="different source"):
        shadow.finalize(
            capsule, tmp_path / "records", protocol, source=other, mode="replay"
        )


def make_prospective_record(shadow, frames, tmp_path, monkeypatch):
    capsule, _ = make_capsule(shadow, frames, tmp_path)
    origin, *_ = frames

    class Clock(datetime):
        instant = origin - timedelta(days=2)

        @classmethod
        def now(cls, _tz=None):
            return cls.instant

    monkeypatch.setattr(shadow, "datetime", Clock)
    protocol = template_protocol(shadow)
    start = origin - timedelta(days=1)
    protocol.update(
        window_start=start.isoformat(),
        window_end=(start + timedelta(days=42)).isoformat(),
    )
    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps(protocol))
    enrolled = shadow.freeze(proposal, tmp_path / "enrollment")
    Clock.instant = origin + timedelta(minutes=5)
    document = json.loads((capsule / "production.json").read_text())
    document.update(
        timezone="America/Los_Angeles",
        observation_at=(origin - timedelta(minutes=1)).isoformat(),
    )
    (capsule / "production.json").write_text(json.dumps(document))
    metadata = json.loads((capsule / "capsule.json").read_text())
    metadata["capture_kind"] = "serving_snapshot"
    metadata["capsule_exported_at"] = (origin + timedelta(minutes=1)).isoformat()
    metadata["provenance"] = dict.fromkeys(
        (
            "production_code_identity",
            "config_fingerprint",
            "availability_assumptions",
            "provider_completion_clock",
            "station_receipt_clock",
        ),
        "explicit fixture assumption",
    )
    metadata["provenance"]["dataset_fingerprint"] = "fixture"
    metadata["provenance"]["production_code_identity"] = shadow.package_identity()
    metadata["input_sha256"] = {
        p.name: shadow.digest(p) for p in capsule.iterdir() if p.name != "capsule.json"
    }
    (capsule / "capsule.json").write_text(json.dumps(metadata))
    public = {
        **document,
        "published_at": (origin + timedelta(minutes=4)).isoformat(),
        "units": {"temp_f": "F"},
        "truth_semantics": {"temp_f": "mean"},
        "hourly": [
            {
                "valid_time": r["valid_time"],
                "values": {"temp_f": 59.0},
                "methods": {"temp_f": "fixture"},
            }
            for r in document["hourly"]
        ],
    }
    target = shadow.finalize(
        capsule,
        tmp_path / "records",
        enrolled,
        publication_receipt=publisher_receipt(shadow, capsule, public),
        source=capsule / "production.json",
        mode="prospective",
    )
    record = json.loads((target / "record.json").read_text())
    assert record["mode"] == "prospective"
    assert record["capsule"]["publication"]["commit"] == "a" * 40
    assert len(record["predictions"]) == 2
    assert all(r["production"] == 15.0 for r in record["predictions"])
    return capsule, enrolled, target, Clock


def test_prospective_publication_receipt_seals_original_inputs(
    shadow, frames, tmp_path, monkeypatch
):
    make_prospective_record(shadow, frames, tmp_path, monkeypatch)


@pytest.mark.parametrize(
    "missing", [None, "training.parquet", "provider_completions.parquet"]
)
def test_prospective_requires_a_complete_input_manifest(
    shadow, frames, tmp_path, missing
):
    capsule, _ = make_capsule(shadow, frames, tmp_path)
    metadata = json.loads((capsule / "capsule.json").read_text())
    if missing is not None:
        metadata["input_sha256"] = {
            name: shadow.digest(capsule / name)
            for name in shadow.INPUT_FILES
            if name != missing
        }
    (capsule / "capsule.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="complete captured input manifest"):
        shadow.capsule_metadata(capsule, prospective=True)


@pytest.mark.parametrize(
    "stage", ["published_only", "receipt_complete", "empty_record_dir"]
)
def test_finalize_recovers_partial_state(shadow, frames, tmp_path, monkeypatch, stage):
    capsule, protocol, target, _clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    (target / "record.json").unlink()
    (target / "record.lock.json").unlink()
    if stage != "empty_record_dir":
        target.rmdir()
    receipt = json.loads((capsule / "publication.json").read_text())[
        "publisher_receipt"
    ]
    if stage == "published_only":
        (capsule / "evidence.lock.json").unlink()
        (capsule / "publication.json").unlink()
    assert (
        shadow.finalize(
            capsule,
            target.parent,
            protocol,
            publication_receipt=receipt if stage == "published_only" else None,
            source=capsule / "production.json",
            mode="prospective",
        )
        == target
    )
    assert json.loads((target / "record.json").read_text())["mode"] == "prospective"


def test_held_record_is_validated_without_retiming_or_refitting(
    shadow, frames, tmp_path, monkeypatch
):
    capsule, protocol, target, clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    before = (target / "record.json").read_bytes()
    clock.instant += timedelta(hours=2)
    monkeypatch.setattr(
        shadow, "record", lambda *_a, **_k: pytest.fail("do not refit held issues")
    )
    assert (
        shadow.finalize(
            capsule,
            target.parent,
            protocol,
            source=capsule / "production.json",
            mode="prospective",
        )
        == target
    )
    assert (target / "record.json").read_bytes() == before
    invalid = json.loads(before)
    invalid["mode"] = "replay"
    (target / "record.json").write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="conflicts with this enrollment"):
        shadow.finalize(
            capsule,
            target.parent,
            protocol,
            source=capsule / "production.json",
            mode="prospective",
        )


@pytest.mark.parametrize(
    "change", ["duplicate", "method", "release", "rounding", "semantics"]
)
def test_publication_projection_rejects_grid_and_attribution_changes(
    shadow, frames, tmp_path, monkeypatch, change
):
    capsule, _protocol, _target, _clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    published = json.loads((capsule / "published.json").read_text())
    if change == "duplicate":
        published["hourly"].append(published["hourly"][0])
    elif change == "method":
        published["hourly"][0]["methods"]["temp_f"] = "wrong"
    elif change == "release":
        published["hourly"][0]["release_ids"] = {"temp_f": "wrong"}
    elif change == "semantics":
        published["truth_semantics"]["temp_f"] = "inst"
    else:
        published["hourly"][0]["values"]["temp_f"] += 0.01
    (capsule / "published.json").write_text(json.dumps(published))
    receipt = json.loads((capsule / "publication.json").read_text())
    bind_public(shadow, capsule, published, receipt)
    (capsule / "publication.json").write_text(json.dumps(receipt))
    with pytest.raises(
        ValueError,
        match=r"grid differs|duplicated published forecast grid|interval mean|attribution differs|temperature differs|payload mismatch",
    ):
        shadow.production_rows(
            capsule,
            shadow.capsule_metadata(capsule, prospective=True),
            prospective=True,
        )


def test_freeze_rejects_changed_candidate_bytes(shadow, tmp_path, monkeypatch):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(template_protocol(shadow)))
    digest = shadow.digest
    monkeypatch.setattr(
        shadow,
        "digest",
        lambda path: (
            "0" * 64 if path.name == "shadow_candidate_v1.py" else digest(path)
        ),
    )
    with pytest.raises(ValueError, match="reviewed protocol digest"):
        shadow.freeze(protocol, tmp_path / "enrollment")


def test_record_store_rejects_an_issue_directory_alias(
    shadow, frames, tmp_path, monkeypatch
):
    capsule, protocol, target, _clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    shutil.rmtree(target)
    outside = tmp_path / "outside"
    outside.mkdir()
    target.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="remain within the record store"):
        shadow.record(capsule, target.parent, protocol, mode="prospective")
    assert not (outside / "record.json").exists()


def test_evaluation_audit_hashes_only_the_scored_records(
    shadow, frames, tmp_path, monkeypatch
):
    capsule, protocol, target, clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    clock.instant = shadow.timestamp(
        json.loads(protocol.read_text())["window_end"]
    ) + timedelta(hours=98)
    truth_lock = shadow.freeze_truth(
        capsule / "truth_hourly.parquet",
        protocol,
        tmp_path / "truth-lock",
        clock.instant,
    )
    root = tmp_path / "capsules"
    root.mkdir()
    capsule.rename(root / target.name)
    original = json.loads((target / "record.json").read_text())
    for name, change in (
        ("replay", {"mode": "replay"}),
        ("other-protocol", {"protocol_sha256": "other"}),
    ):
        directory = target.parent / name
        directory.mkdir()
        (directory / "record.json").write_text(json.dumps(original | change))
    output = tmp_path / "evaluation"
    shadow.evaluate(
        target.parent,
        None,
        protocol,
        output,
        mode="prospective",
        capsules=root,
        truth_lock=truth_lock,
    )
    receipt = json.loads((output / "evaluation.json").read_text())
    assert receipt["record_sha256"] == {
        target.name: shadow.digest(target / "record.json")
    }
