"""Evidence transitions, system-prefix paths and one-owner validation regressions."""

import base64
import importlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pytest
from filelock import FileLock
from test_launchd import SCRIPT, load_script
from test_shadow_v3 import original_receipt, remove_record
from test_temperature_shadow import (
    make_capsule,
    make_prospective_record,
    publisher_receipt,
)

from grounded_weather_forecast.shadow_artifacts import EvidenceError


def unsealed(shadow, frames, tmp_path, monkeypatch):
    capsule, protocol, target, clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    expected = json.loads((target / "record.json").read_bytes())
    receipt = original_receipt(capsule)
    remove_record(target)
    for name in ("published.json", "publication.json", "evidence.lock.json"):
        (capsule / name).unlink()
    return capsule, protocol, target, clock, receipt, expected


@pytest.mark.parametrize("damage", ["source", "temperature"])
def test_replay_receipt_refused_before_guard_and_corrected_retry_succeeds(
    shadow, frames, tmp_path, monkeypatch, damage
):
    capsule, protocol, target, _clock, receipt, _expected = unsealed(
        shadow, frames, tmp_path, monkeypatch
    )
    (capsule / ".sealing").unlink()  # No process owns this synthetic fixture sidecar.
    before = {p.name: p.read_bytes() for p in capsule.iterdir()}
    bad = dict(receipt)
    if damage == "source":
        bad["source_sha256"] = "0" * 64
    else:
        public = json.loads(base64.b64decode(receipt["forecast_base64"]))
        public["hourly"][0]["values"]["temp_f"] += 1
        bad = publisher_receipt(shadow, capsule, public)
    with pytest.raises(
        EvidenceError, match="receipts require prospective finalization"
    ):
        shadow.finalize(
            capsule,
            target.parent,
            protocol,
            source=None,
            mode="replay",
            publication_receipt=bad,
        )
    assert {p.name: p.read_bytes() for p in capsule.iterdir()} == before
    snapshot = shadow.CapsuleSnapshot.load(capsule, prospective=False)
    enrolled = shadow.enrollment(protocol)
    with pytest.raises(
        EvidenceError, match="receipts require prospective finalization"
    ):
        shadow.seal_publication(
            snapshot,
            bad,
            enrolled,
        )
    assert not (capsule / "evidence.lock.json").exists()
    assert (
        shadow.finalize(
            capsule,
            target.parent,
            protocol,
            source=None,
            mode="prospective",
            publication_receipt=receipt,
        )
        == target
    )


@pytest.mark.parametrize("where", ["root", "issuance"])
@pytest.mark.parametrize("kind", ["alias", "file"])
def test_output_refused_before_sealing_or_fitting(
    shadow, frames, tmp_path, monkeypatch, where, kind
):
    capsule, protocol, target, _clock, receipt, _expected = unsealed(
        shadow, frames, tmp_path, monkeypatch
    )
    root = tmp_path / "invalid-records"
    selected = root if where == "root" else root / target.name
    if where == "issuance":
        root.mkdir(mode=0o700)
    if kind == "alias":
        destination = tmp_path / "other-records"
        destination.mkdir(mode=0o700)
        selected.symlink_to(destination, target_is_directory=True)
    else:
        selected.write_bytes(b"conflict")
    before = {p.name: p.read_bytes() for p in capsule.iterdir()}
    monkeypatch.setattr(
        shadow,
        "prepare_matrices",
        lambda *_a: pytest.fail("prepared before output refusal"),
    )
    expected = (
        "record store"
        if where == "issuance"
        else (
            "nonaliased local directory"
            if kind == "alias"
            else "expected a local directory"
        )
    )
    with pytest.raises(EvidenceError, match=expected):
        shadow.finalize(
            capsule,
            root,
            protocol,
            source=None,
            mode="prospective",
            publication_receipt=receipt,
        )
    assert {p.name: p.read_bytes() for p in capsule.iterdir()} == before


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory modes")
def test_new_record_parents_are_private(shadow, frames, tmp_path):
    capsule, protocol = make_capsule(shadow, frames, tmp_path)
    root = tmp_path / "new" / "middle" / "records"
    previous = os.umask(0o022)
    try:
        target = shadow.record(capsule, root, protocol, mode="replay")
    finally:
        os.umask(previous)
    for path in (root.parent.parent, root.parent, root, target):
        assert path.stat().st_mode & 0o777 == 0o700


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS system aliases")
def test_direct_system_prefix_outputs_and_replay_capsule(shadow, frames, tmp_path):
    capsule, _protocol = make_capsule(shadow, frames, tmp_path)
    with tempfile.TemporaryDirectory(prefix="shadow-prefix-", dir="/tmp") as temporary:
        alias = Path(temporary)
        alias.rmdir()
        capsule.rename(alias)
        snapshot = shadow.CapsuleSnapshot.load(alias, prospective=False)
        assert snapshot.capsule == alias.resolve()
        with pytest.raises(EvidenceError, match="private capsule directory"):
            shadow.CapsuleSnapshot.load(alias, prospective=True)
        output = alias.with_name(alias.name + "-evaluation")
        try:
            with shadow.staged_directory(output) as folder:
                (folder / "complete").write_bytes(b"result")
            assert (output / "complete").read_bytes() == b"result"
        finally:
            (output / "complete").unlink(missing_ok=True)
            output.rmdir()
            output.with_name("." + output.name + ".publishing").unlink(missing_ok=True)


@pytest.mark.parametrize("missing", ["protocol", "lock"])
def test_prospective_evaluation_missing_artifact_is_refusal(
    shadow, frames, tmp_path, missing
):
    _capsule, protocol = make_capsule(shadow, frames, tmp_path)
    path = protocol if missing == "protocol" else protocol.with_suffix(".lock.json")
    path.unlink()
    with pytest.raises(
        EvidenceError, match=f"{path}: missing required enrollment artifact"
    ):
        shadow.evaluation_inputs(
            tmp_path / "records", None, protocol, mode="prospective"
        )


def test_evaluation_permission_error_stays_operational(
    shadow, frames, tmp_path, monkeypatch
):
    _capsule, protocol = make_capsule(shadow, frames, tmp_path)
    read = Path.read_bytes

    def denied(path):
        if path == protocol:
            raise PermissionError("protocol access denied")
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", denied)
    with pytest.raises(PermissionError, match="protocol access denied"):
        shadow.evaluation_inputs(
            tmp_path / "records", None, protocol, mode="prospective"
        )


def test_first_seal_validates_once_and_keeps_results(
    shadow, frames, tmp_path, monkeypatch
):
    capsule, protocol, target, _clock, receipt, expected = unsealed(
        shadow, frames, tmp_path, monkeypatch
    )
    calls = Counter()
    evidence = importlib.import_module("shadow_evidence")
    for module, name in (
        (evidence.base64, "b64decode"),
        (shadow, "production_rows"),
        (shadow, "checked_issue"),
        (shadow, "prepare_matrices"),
    ):
        original = getattr(module, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, counted)
    shadow.finalize(
        capsule,
        target.parent,
        protocol,
        source=None,
        mode="prospective",
        publication_receipt=receipt,
    )
    assert calls == dict.fromkeys(
        ("b64decode", "production_rows", "checked_issue", "prepare_matrices"), 1
    )
    saved = json.loads((target / "record.json").read_bytes())
    assert saved["predictions"] == expected["predictions"]
    assert saved["fitted"] == expected["fitted"]


def test_sealing_release_reports_completed_issuance(
    shadow, frames, tmp_path, monkeypatch
):
    capsule, protocol, target, _clock, receipt, _expected = unsealed(
        shadow, frames, tmp_path, monkeypatch
    )
    release = FileLock.release

    def fail_sealing(self, **kwargs):
        held = self.is_locked
        release(self, **kwargs)
        if held and self.lock_file.endswith(".sealing"):
            raise OSError("injected sealing unlock failure")

    monkeypatch.setattr(FileLock, "release", fail_sealing)
    with pytest.raises(OSError, match=f"complete output preserved at {target}:"):
        shadow.finalize(
            capsule,
            target.parent,
            protocol,
            source=None,
            mode="prospective",
            publication_receipt=receipt,
        )
    assert (target / "record.lock.json").is_file()


def test_cli_signals_restore_handler_after_unwinding(shadow):
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as caught, shadow.cli_signals():
        shadow.terminate_shadow(signal.SIGTERM, None)
    assert caught.value.code == 128 + signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) is previous


def test_tagged_refusal_has_one_detail_and_separate_notes(capsys):
    wrapper = load_script(SCRIPT.with_name("hourly-predict-publish.py"))
    result = {
        "kind": "temperature_shadow_preflight",
        "schema_version": 1,
        "status": "refused",
        "detail": "missing lock",
    }
    enabled, error = wrapper.preflight_result(
        subprocess.CompletedProcess(
            [],
            2,
            json.dumps(result),
            "shadow refused: missing lock\nshadow cleanup detail: second note\n",
        )
    )
    assert not enabled
    assert error == "shadow unavailable: missing lock"
    assert (
        capsys.readouterr().out
        == "shadow preflight stderr: shadow cleanup detail: second note\n"
    )


@pytest.mark.parametrize("depth", [1, 3])
def test_file_output_ancestor_refused_before_sealing(
    shadow, frames, tmp_path, monkeypatch, depth
):
    capsule, protocol, _target, _clock, receipt, _expected = unsealed(
        shadow, frames, tmp_path, monkeypatch
    )
    blocker = tmp_path / "regular-file"
    blocker.write_bytes(b"preserve")
    output = blocker.joinpath(*(["missing"] * depth))
    before = {path.name: path.read_bytes() for path in capsule.iterdir()}
    monkeypatch.setattr(
        shadow,
        "prepare_matrices",
        lambda *_args: pytest.fail("prepared below a file ancestor"),
    )
    with pytest.raises(EvidenceError, match="ancestor must be a directory"):
        shadow.finalize(
            capsule,
            output,
            protocol,
            source=None,
            mode="prospective",
            publication_receipt=receipt,
        )
    assert {path.name: path.read_bytes() for path in capsule.iterdir()} == before
    assert blocker.read_bytes() == b"preserve"
