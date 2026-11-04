"""Prospective scoring reconstructs retained evidence before admitting any row."""

import base64
import hashlib
import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest
from filelock import FileLock
from test_temperature_shadow import make_capsule, make_prospective_record


@pytest.fixture
def publication_fixture():
    # Function scope gives every mutation test its own complete parsed copy.
    return json.loads(
        (
            Path(__file__).with_name("fixtures") / "publication-receipt-v1.json"
        ).read_text()
    )


def test_copied_publication_fixture_git_object_chain(publication_fixture):
    def git_hash(kind, payload):
        return hashlib.sha1(
            f"{kind} {len(payload)}\0".encode() + payload, usedforsecurity=False
        ).hexdigest()

    for case in publication_fixture["cases"]:
        receipt = case["receipt"]
        commit = base64.b64decode(case["commit_base64"])
        tree = base64.b64decode(case["tree_base64"])
        payload = base64.b64decode(receipt["forecast_base64"])
        assert git_hash("commit", commit) == receipt["commit"]
        assert commit.splitlines()[0] == f"tree {git_hash('tree', tree)}".encode()
        assert tree == b"100644 forecast.json\0" + bytes.fromhex(
            git_hash("blob", payload)
        )


@pytest.mark.parametrize("value", [None, [], 42, "receipt"])
def test_receipt_root_requires_an_object(shadow, value):
    with pytest.raises(ValueError, match="expected an object"):
        shadow.publication(value)


@pytest.mark.parametrize(
    "field",
    [
        "commit",
        "forecast_base64",
        "issued_at",
        "published_at",
        "remote",
        "branch",
        "publisher_version",
        "source_sha256",
        "forecast_sha256",
    ],
)
@pytest.mark.parametrize("value", [None, 42, []])
def test_malformed_receipt_fields_raise_domain_errors(
    shadow, field, value, publication_fixture
):
    fixture = publication_fixture
    receipt = fixture["cases"][0]["receipt"]
    receipt[field] = value
    with pytest.raises(ValueError, match=r"commit identity|string fields"):
        shadow.publication(receipt)


@pytest.mark.parametrize("value", [None, [], 42, "forecast", True])
def test_receipt_payload_requires_an_object(shadow, value, publication_fixture):
    fixture = publication_fixture
    receipt = fixture["cases"][0]["receipt"]
    payload = json.dumps(value).encode()
    receipt["forecast_base64"] = base64.b64encode(payload).decode()
    receipt["forecast_sha256"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="expected an object"):
        shadow.publication(receipt)


def test_replay_rejects_prospective_only_truth_lock(shadow, frames, tmp_path):
    capsule, protocol = make_capsule(shadow, frames, tmp_path)
    target = shadow.record(capsule, tmp_path / "replay", protocol, mode="replay")
    with pytest.raises(ValueError, match="replay evaluation does not accept"):
        shadow.evaluate(
            target.parent,
            capsule / "truth_hourly.parquet",
            protocol,
            tmp_path / "result",
            mode="replay",
            truth_lock=tmp_path / "unused",
        )
    shadow.evaluate(
        target.parent,
        capsule / "truth_hourly.parquet",
        protocol,
        tmp_path / "valid",
        mode="replay",
    )
    result = json.loads((tmp_path / "valid/evaluation.json").read_text())
    assert result["truth_lock_sha256"] is None
    assert result["truth_sha256"] == shadow.digest(capsule / "truth_hourly.parquet")


def test_malformed_receipt_cli_has_clear_refusal(shadow, mature_record, tmp_path):
    receipt = tmp_path / "bad-receipt.json"
    receipt.write_text("[]")
    capsule, protocol, target, *_ = mature_record
    result = subprocess.run(  # noqa: S603 -- only the local standalone test CLI
        [
            sys.executable,
            shadow.__file__,
            "finalize",
            "--input",
            str(capsule),
            "--protocol",
            str(protocol),
            "--out",
            str(target.parent),
            "--publication-receipt",
            str(receipt),
            "--mode",
            "prospective",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "expected an object" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.fixture
def mature_record(shadow, frames, tmp_path, monkeypatch):
    capsule, protocol, target, clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    root = tmp_path / "capsules"
    root.mkdir(mode=0o700)
    retained = root / target.name
    capsule.rename(retained)
    clock.instant = shadow.timestamp(
        json.loads(protocol.read_text())["window_end"]
    ) + timedelta(hours=98)
    truth = tmp_path / "qc.parquet"
    frames[2].select(
        pl.col("valid_time").alias("valid_hour"), pl.lit(15.0).alias(shadow.TARGET)
    ).write_parquet(truth)
    truth_lock = shadow.freeze_truth(
        truth, protocol, tmp_path / "truth-lock", clock.instant
    )
    return retained, protocol, target, clock, root, truth_lock


def evaluate(shadow, context, output, replacement=None):
    _capsule, protocol, target, _clock, root, truth_lock = context
    shadow.evaluate(
        target.parent,
        replacement,
        protocol,
        output,
        mode="prospective",
        capsules=root,
        truth_lock=truth_lock,
    )


def test_mature_evaluation_reconstructs_and_scores_exact_record(
    shadow, mature_record, tmp_path
):
    evaluate(shadow, mature_record, tmp_path / "result")
    result = json.loads((tmp_path / "result/evaluation.json").read_text())
    assert result["record_sha256"] == {
        mature_record[2].name: shadow.digest(mature_record[2] / "record.json")
    }
    assert result["truth_sha256"] == shadow.digest(mature_record[5] / "truth.parquet")
    assert result["truth_lock_sha256"] == shadow.digest(
        mature_record[5] / "truth.lock.json"
    )
    matched = pl.read_parquet(tmp_path / "result/matched_rows.parquet")
    assert matched.height == 2
    assert matched["truth"].to_list() == [15.0, 15.0]


@pytest.mark.parametrize(
    "change",
    [
        "prediction",
        "attribution",
        "lead",
        "horizon",
        "record-clock",
        "implementation",
        "fitted",
        "input",
        "manifest",
        "receipt",
        "lock",
        "legacy",
    ],
)
def test_inconsistent_evidence_fails_before_output(
    shadow, mature_record, tmp_path, change
):
    capsule, protocol, target, *_ = mature_record
    path = target / "record.json"
    saved = json.loads(path.read_text())
    if change == "prediction":
        saved["predictions"][0]["candidate"] += 1
    elif change == "attribution":
        saved["predictions"][0]["production_method"] = "changed"
    elif change == "lead":
        saved["predictions"][0]["lead_hours"] += 1
    elif change == "horizon":
        saved["predictions"][0]["horizon"] = "72-96h"
    elif change == "record-clock":
        saved["recorded_at"] = saved["issued_at"]
    elif change == "implementation":
        saved["code_identity"] = "changed"
    elif change == "fitted":
        saved["fitted"]["sources"] = ["different"]
    elif change == "legacy":
        saved["artifact_version"] = 1
    elif change == "input":
        with (capsule / "issue.parquet").open("ab") as stream:
            stream.write(b"changed")
    elif change == "manifest":
        p = capsule / "capsule.json"
        meta = json.loads(p.read_text())
        meta["provenance"]["config_fingerprint"] = "changed"
        p.write_text(json.dumps(meta))
    elif change == "receipt":
        p = capsule / "publication.json"
        receipt = json.loads(p.read_text())
        receipt["commit"] = "b" * 40
        p.write_text(json.dumps(receipt))
    else:
        p = protocol.with_suffix(".lock.json")
        lock = json.loads(p.read_text())
        lock["runner_sha256"] = "0" * 64
        p.write_text(json.dumps(lock))
    path.write_text(json.dumps(saved))
    with pytest.raises(
        ValueError,
        match=r"differs|changed|fresh|clock|contemporaneous|requires|must contain|include timezone|mismatch|conflicts|incompatible",
    ):
        evaluate(shadow, mature_record, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_duplicate_issuance_cannot_inflate_coverage(shadow, mature_record, tmp_path):
    target = mature_record[2]
    duplicate = target.parent / "duplicate"
    duplicate.mkdir()
    (duplicate / "record.json").write_bytes((target / "record.json").read_bytes())
    with pytest.raises(ValueError, match="duplicated recorded issuance"):
        evaluate(shadow, mature_record, tmp_path / "result")


@pytest.mark.parametrize("kind", ["record", "capsule"])
def test_issuance_directory_alias_is_rejected(shadow, mature_record, tmp_path, kind):
    path = mature_record[2] if kind == "record" else mature_record[0]
    outside = tmp_path / "outside"
    path.rename(outside)
    path.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="within"):
        evaluate(shadow, mature_record, tmp_path / "result")


def test_truth_sealing_and_evaluation_require_maturity(shadow, mature_record, tmp_path):
    _capsule, protocol, _target, clock, *_ = mature_record
    clock.instant -= timedelta(microseconds=1)
    with pytest.raises(ValueError, match=r"\+98 hours"):
        shadow.freeze_truth(
            mature_record[5] / "truth.parquet",
            protocol,
            tmp_path / "early-truth",
            clock.instant,
        )
    with pytest.raises(ValueError, match=r"\+98 hours"):
        evaluate(shadow, mature_record, tmp_path / "early-result")
    assert not (tmp_path / "early-truth").exists()
    assert not (tmp_path / "early-result").exists()


@pytest.mark.parametrize(
    "change", ["duplicate", "naive", "off-hour", "target", "infinite"]
)
def test_freeze_truth_rejects_invalid_qc_shapes(
    shadow, mature_record, tmp_path, change
):
    truth = pl.read_parquet(mature_record[5] / "truth.parquet")
    if change == "duplicate":
        truth = pl.concat([truth, truth.head(1)])
    elif change == "naive":
        truth = truth.with_columns(pl.col("valid_hour").dt.replace_time_zone(None))
    elif change == "off-hour":
        truth = truth.with_columns(pl.col("valid_hour") + timedelta(minutes=1))
    elif change == "target":
        truth = truth.rename({shadow.TARGET: "t__temp_c__inst"})
    else:
        truth = truth.with_columns(pl.lit(float("inf")).alias(shadow.TARGET))
    source = tmp_path / "bad.parquet"
    truth.write_parquet(source)
    with pytest.raises(
        ValueError,
        match=r"differs|changed|fresh|clock|contemporaneous|requires|must contain|include timezone|mismatch|conflicts|incompatible",
    ):
        shadow.freeze_truth(
            source, mature_record[1], tmp_path / "bad-lock", mature_record[3].instant
        )
    assert not (tmp_path / "bad-lock").exists()


def test_truth_swap_is_rejected_and_same_bytes_are_accepted(
    shadow, mature_record, tmp_path
):
    truth = tmp_path / "truth.parquet"
    truth.write_bytes((mature_record[5] / "truth.parquet").read_bytes())
    evaluate(shadow, mature_record, tmp_path / "same", truth)
    pl.read_parquet(truth).with_columns(
        pl.lit(99.0).alias(shadow.TARGET)
    ).write_parquet(truth)
    with pytest.raises(ValueError, match="supplied truth differs"):
        evaluate(shadow, mature_record, tmp_path / "swapped", truth)
    assert not (tmp_path / "swapped").exists()


def test_timezone_aware_truth_is_matched_by_utc_instant(
    shadow, mature_record, tmp_path
):
    local = tmp_path / "local.parquet"
    pl.read_parquet(mature_record[5] / "truth.parquet").with_columns(
        pl.col("valid_hour").dt.convert_time_zone("America/Los_Angeles")
    ).write_parquet(local)
    lock = shadow.freeze_truth(
        local, mature_record[1], tmp_path / "local-lock", mature_record[3].instant
    )
    context = (*mature_record[:5], lock)
    evaluate(shadow, context, tmp_path / "local-result")
    assert pl.read_parquet(tmp_path / "local-result/matched_rows.parquet")[
        "truth"
    ].to_list() == [15.0, 15.0]


def test_scoring_failure_does_not_publish_partial_results(
    shadow, mature_record, tmp_path, monkeypatch
):
    def failed(*_args, **_kwargs):
        raise ValueError("injected scoring failure")

    monkeypatch.setattr(shadow, "score_horizon", failed)
    with pytest.raises(ValueError, match="injected"):
        evaluate(shadow, mature_record, tmp_path / "result")
    assert not (tmp_path / "result").exists()
    with FileLock(tmp_path / ".result.publishing", timeout=0):
        assert not list(tmp_path.glob("shadow-stage-*"))


@pytest.mark.parametrize("index", [0, 1])
def test_actual_publisher_schema_5_and_6_receipts_are_accepted(
    shadow, frames, tmp_path, monkeypatch, index, publication_fixture
):
    fixture = publication_fixture
    case = fixture["cases"][index]
    capsule, protocol, target, _clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    (target / "record.json").unlink()
    (target / "record.lock.json").unlink()
    for name in ("published.json", "publication.json", "evidence.lock.json"):
        (capsule / name).unlink()
    (capsule / "production.json").write_bytes(base64.b64decode(case["source_base64"]))
    meta = json.loads((capsule / "capsule.json").read_text())
    meta["input_sha256"]["production.json"] = shadow.digest(capsule / "production.json")
    (capsule / "capsule.json").write_text(json.dumps(meta))
    shadow.finalize(
        capsule,
        target.parent,
        protocol,
        publication_receipt=case["receipt"],
        source=capsule / "production.json",
        mode="prospective",
    )
    saved = json.loads((target / "record.json").read_text())
    assert saved["capsule"]["publication"]["commit"] == case["receipt"]["commit"]
    assert all(
        row["production"] == pytest.approx((59.22 - 32) / 1.8)
        for row in saved["predictions"]
    )
    public = json.loads((capsule / "published.json").read_bytes())
    assert public["truth_semantics"] == {"temp_f": "mean"}
    assert all("truth_semantics" not in row for row in public["hourly"])
