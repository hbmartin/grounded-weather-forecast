"""Durable seals, safe retries, and snapshot-bound evaluation regressions."""

import base64
import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from filelock import FileLock
from test_launchd import SCRIPT, load_script
from test_shadow_evidence import (
    evaluate,
)
from test_shadow_evidence import (
    mature_record as mature_record_fixture,
)
from test_temperature_shadow import make_prospective_record, publisher_receipt

from grounded_weather_forecast.shadow_artifacts import json_value


@pytest.fixture
def mature_record(shadow, frames, tmp_path, monkeypatch):
    return mature_record_fixture.__wrapped__(shadow, frames, tmp_path, monkeypatch)


def original_receipt(capsule):
    return json.loads((capsule / "publication.json").read_text())["publisher_receipt"]


def remove_record(target):
    (target / "record.json").unlink()
    (target / "record.lock.json").unlink()
    target.rmdir()


@pytest.mark.parametrize(
    ("field", "value"), [("hourly", [1, 2]), ("units", []), ("release_ids", 5)]
)
def test_nested_payload_refused_before_writes_and_corrected_retry_succeeds(
    shadow, frames, tmp_path, monkeypatch, field, value
):
    capsule, protocol, target, _clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    good = original_receipt(capsule)
    public = json.loads(base64.b64decode(good["forecast_base64"]))
    remove_record(target)
    for name in ("publication.json", "published.json", "evidence.lock.json"):
        (capsule / name).unlink()
    before = {p.name: p.read_bytes() for p in capsule.iterdir()}
    public[field] = value
    bad = publisher_receipt(shadow, capsule, public)
    with pytest.raises(ValueError, match=field):
        shadow.finalize(
            capsule,
            target.parent,
            protocol,
            publication_receipt=bad,
            source=capsule / "production.json",
            mode="prospective",
        )
    assert {p.name: p.read_bytes() for p in capsule.iterdir()} == before
    assert (
        shadow.finalize(
            capsule,
            target.parent,
            protocol,
            publication_receipt=good,
            source=capsule / "production.json",
            mode="prospective",
        )
        == target
    )


def test_held_push_keeps_original_seal_and_never_refits(
    shadow, frames, tmp_path, monkeypatch
):
    capsule, protocol, target, clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    before = {
        name: (capsule / name).read_bytes()
        for name in ("publication.json", "published.json", "evidence.lock.json")
    }
    record = (target / "record.json").read_bytes()
    public = json.loads(base64.b64decode(original_receipt(capsule)["forecast_base64"]))
    clock.instant += timedelta(hours=2)
    public["published_at"] = clock.instant.isoformat()
    later = publisher_receipt(shadow, capsule, public)
    later["commit"] = "b" * 40
    monkeypatch.setattr(
        shadow,
        "prepared_predictions",
        lambda *_a: pytest.fail("held record must not refit"),
    )
    assert (
        shadow.finalize(
            capsule,
            target.parent,
            protocol,
            publication_receipt=later,
            source=capsule / "production.json",
            mode="prospective",
        )
        == target
    )
    assert (target / "record.json").read_bytes() == record
    assert {name: (capsule / name).read_bytes() for name in before} == before


def test_late_recovery_requires_complete_timely_seal(
    shadow, frames, tmp_path, monkeypatch
):
    capsule, protocol, target, clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    sealed_at = json.loads((capsule / "evidence.lock.json").read_text())[
        "evidence_sealed_at"
    ]
    remove_record(target)
    clock.instant += timedelta(hours=2)
    assert (
        shadow.finalize(
            capsule, target.parent, protocol, source=None, mode="prospective"
        )
        == target
    )
    saved = json.loads((target / "record.json").read_text())
    assert saved["evidence_sealed_at"] == sealed_at
    assert saved["recorded_at"] == clock.instant.isoformat()
    remove_record(target)
    (capsule / "evidence.lock.json").unlink()
    with pytest.raises(ValueError, match=r"evidence.lock.json|complete.*seal"):
        shadow.finalize(
            capsule, target.parent, protocol, source=None, mode="prospective"
        )
    assert not target.exists()


def test_late_created_seal_cannot_authorize_recovery(
    shadow, frames, tmp_path, monkeypatch
):
    capsule, protocol, target, clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    remove_record(target)
    clock.instant += timedelta(hours=2)
    path = capsule / "evidence.lock.json"
    seal = json.loads(path.read_text())
    seal["evidence_sealed_at"] = clock.instant.isoformat()
    path.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="contemporaneous serving snapshot"):
        shadow.finalize(
            capsule, target.parent, protocol, source=None, mode="prospective"
        )
    assert not target.exists()


@pytest.mark.parametrize("field", ["remote", "branch"])
def test_held_push_destination_must_match_original(
    shadow, frames, tmp_path, monkeypatch, field
):
    capsule, protocol, target, _clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    receipt = original_receipt(capsule)
    receipt[field] = "different"
    before = (capsule / "publication.json").read_bytes()
    with pytest.raises(ValueError, match="held publication destination differs"):
        shadow.finalize(
            capsule,
            target.parent,
            protocol,
            publication_receipt=receipt,
            source=None,
            mode="prospective",
        )
    assert (capsule / "publication.json").read_bytes() == before


def test_incidental_regular_file_does_not_change_evidence(
    shadow, mature_record, tmp_path
):
    (mature_record[0] / ".DS_Store").write_bytes(b"Finder")
    evaluate(shadow, mature_record, tmp_path / "result")
    assert (tmp_path / "result/evaluation.json").exists()


def test_nanosecond_truth_normalizes_and_subhour_nanosecond_is_refused(
    shadow, mature_record, tmp_path
):
    _capsule, protocol, _target, clock, _root, old_lock = mature_record
    truth = pl.read_parquet(old_lock / "truth.parquet").with_columns(
        pl.col("valid_hour").cast(pl.Datetime("ns", "UTC"))
    )
    source = tmp_path / "ns.parquet"
    truth.write_parquet(source)
    locked = shadow.freeze_truth(source, protocol, tmp_path / "ns-lock", clock.instant)
    evaluate(shadow, (*mature_record[:5], locked), tmp_path / "result")
    truth.with_columns(
        (pl.col("valid_hour").cast(pl.Int64) + 1).cast(pl.Datetime("ns", "UTC"))
    ).write_parquet(source)
    with pytest.raises(ValueError, match="exact hours"):
        shadow.freeze_truth(source, protocol, tmp_path / "bad-lock", clock.instant)


def test_audit_uses_the_bytes_actually_scored(
    shadow, mature_record, tmp_path, monkeypatch
):
    _capsule, protocol, target, _clock, _root, lock = mature_record
    paths = [
        protocol,
        lock / "truth.parquet",
        lock / "truth.lock.json",
        target / "record.json",
    ]
    before = {p: shadow.digest(p) for p in paths}
    score = shadow.score_horizon
    changed = False

    def replace_after_join(*args, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            pl.read_parquet(paths[1]).with_columns(
                pl.lit(99.0).alias(shadow.TARGET)
            ).write_parquet(paths[1])
            for path in (paths[0], paths[2], paths[3]):
                path.write_text("{}")
        return score(*args, **kwargs)

    monkeypatch.setattr(shadow, "score_horizon", replace_after_join)
    evaluate(shadow, mature_record, tmp_path / "result")
    result = json.loads((tmp_path / "result/evaluation.json").read_text())
    assert result["protocol_sha256"] == before[protocol]
    assert result["truth_sha256"] == before[paths[1]]
    assert result["truth_lock_sha256"] == before[paths[2]]
    assert result["record_sha256"] == {target.name: before[paths[3]]}
    assert pl.read_parquet(tmp_path / "result/matched_rows.parquet")[
        "truth"
    ].to_list() == [15.0, 15.0]


def test_runtime_drift_is_refused_before_evaluation(
    shadow, mature_record, tmp_path, monkeypatch
):
    identity = shadow.runtime_identity

    def changed(path):
        result = identity(path)
        result["python"] = "different"
        return result

    monkeypatch.setattr(shadow, "runtime_identity", changed)
    with pytest.raises(ValueError, match="numerical runtime"):
        evaluate(shadow, mature_record, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_kernel_release_alone_does_not_invalidate_numerical_enrollment(
    shadow, frames, tmp_path, monkeypatch
):
    _capsule, protocol, _target, _clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    original = shadow.enrollment(protocol)
    monkeypatch.setattr(
        "grounded_weather_forecast.shadow_artifacts.platform.release",
        lambda: "unrelated-kernel-security-update",
    )
    assert shadow.enrollment(protocol) == original


def test_evaluation_reads_each_selected_lock_and_record_once(
    shadow, mature_record, tmp_path, monkeypatch
):
    capsule, protocol, target, _clock, _root, truth_lock = mature_record
    selected = {
        protocol,
        protocol.with_suffix(".lock.json"),
        capsule / "evidence.lock.json",
        target / "record.json",
        target / "record.lock.json",
        truth_lock / "truth.lock.json",
        truth_lock / "truth.parquet",
    }
    reads = dict.fromkeys(selected, 0)
    read = Path.read_bytes

    def counted(path):
        if path in reads:
            reads[path] += 1
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", counted)
    evaluate(shadow, mature_record, tmp_path / "result")
    assert reads == dict.fromkeys(selected, 1)


def test_interrupted_seal_recovers_only_matching_partial_evidence(
    shadow, frames, tmp_path, monkeypatch
):
    capsule, protocol, target, _clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    receipt = original_receipt(capsule)
    remove_record(target)
    for name in ("publication.json", "published.json", "evidence.lock.json"):
        (capsule / name).unlink()
    original = shadow.write_new

    def interrupted(path, value):
        if path.name == "evidence.lock.json":
            raise OSError("seal synchronization unavailable")
        return original(path, value)

    monkeypatch.setattr(shadow, "write_new", interrupted)
    with pytest.raises(OSError, match="seal synchronization"):
        shadow.finalize(
            capsule,
            target.parent,
            protocol,
            publication_receipt=receipt,
            source=None,
            mode="prospective",
        )
    assert (capsule / "publication.json").exists()
    assert not (capsule / "evidence.lock.json").exists()
    assert not target.exists()
    monkeypatch.setattr(shadow, "write_new", original)
    assert (
        shadow.finalize(
            capsule,
            target.parent,
            protocol,
            publication_receipt=receipt,
            source=None,
            mode="prospective",
        )
        == target
    )


def test_wrong_receipt_issuance_has_specific_refusal(
    shadow, frames, tmp_path, monkeypatch
):
    capsule, protocol, target, _clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    receipt = original_receipt(capsule)
    public = json.loads(base64.b64decode(receipt["forecast_base64"]))
    public["issued_at"] = (frames[0] + timedelta(hours=1)).isoformat()
    wrong = publisher_receipt(shadow, capsule, public)
    with pytest.raises(
        ValueError, match="public commit belongs to a different issuance"
    ):
        shadow.finalize(
            capsule,
            target.parent,
            protocol,
            publication_receipt=wrong,
            source=None,
            mode="prospective",
        )


def test_excessive_receipt_nesting_is_a_named_refusal():
    with pytest.raises(ValueError, match=r"excessive JSON nesting|invalid JSON"):
        json_value("[" * 2000 + "0" + "]" * 2000, "publisher receipt")


def test_sealing_lock_survives_concurrency_and_recovers_after_process_exit(
    shadow, frames, tmp_path, monkeypatch
):
    capsule, protocol, target, _clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    remove_record(target)
    lock = capsule / ".sealing"
    with (
        FileLock(lock),
        pytest.raises(OSError, match="finalization is already running"),
    ):
        shadow.finalize(
            capsule, target.parent, protocol, source=None, mode="prospective"
        )
    result = subprocess.run(  # noqa: S603 -- fixed local child holds a synthetic capsule lock
        [
            sys.executable,
            "-c",
            "import os,sys; from filelock import FileLock; lock=FileLock(sys.argv[1]); lock.acquire(); os._exit(7)",
            str(lock),
        ],
        check=False,
        timeout=10,
    )
    assert result.returncode == 7
    assert (
        shadow.finalize(
            capsule, target.parent, protocol, source=None, mode="prospective"
        )
        == target
    )


def test_preflight_validates_the_selected_runtime_enrollment(
    shadow, frames, tmp_path, monkeypatch
):
    _capsule, protocol, _target, clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    wrapper = load_script(SCRIPT.with_name("hourly-predict-publish.py"))
    monkeypatch.setattr(wrapper, "datetime", clock)
    paths = SimpleNamespace(
        project=Path(shadow.__file__).parents[1], command=lambda _name: [sys.executable]
    )
    assert wrapper.shadow_preflight(paths, protocol) == (True, None)
    locked = protocol.with_suffix(".lock.json")
    lock = json.loads(locked.read_text())
    lock["runner_sha256"] = "changed"
    locked.write_text(json.dumps(lock))
    enabled, error = wrapper.shadow_preflight(paths, protocol)
    assert not enabled
    assert "frozen protocol or implementation changed" in error


@pytest.mark.parametrize(
    "value", [{}, {"window_start": None}, {"window_start": "bad", "window_end": "bad"}]
)
def test_malformed_protocol_cli_is_a_named_refusal(shadow, tmp_path, value):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(value))
    result = subprocess.run(  # noqa: S603 -- local runner and synthetic evidence only
        [
            sys.executable,
            shadow.__file__,
            "freeze",
            "--protocol",
            str(protocol),
            "--out",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 2
    assert "shadow refused:" in result.stderr
    assert (
        "protocol.window_start" in result.stderr
        or "timestamp must include timezone" in result.stderr
    )
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("value", [None, 42, []])
def test_sealed_commit_type_is_refused(shadow, mature_record, value):
    capsule = mature_record[0]
    receipt = json.loads((capsule / "publication.json").read_text())
    receipt["commit"] = value
    (capsule / "publication.json").write_text(json.dumps(receipt))
    snapshot = shadow.CapsuleSnapshot.load(capsule, prospective=True)
    with pytest.raises(ValueError, match="commit identity"):
        shadow.production_rows(snapshot)
