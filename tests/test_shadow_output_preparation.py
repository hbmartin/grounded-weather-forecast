"""Output staging must succeed before a capsule can be irreversibly sealed."""

import json
import signal
import subprocess
import sys
from datetime import UTC
from functools import partial
from pathlib import Path

import pytest
from test_launchd import SCRIPT, load_script
from test_shadow_lifecycle import unsealed
from test_temperature_shadow import make_capsule, make_prospective_record

from grounded_weather_forecast.shadow_artifacts import EvidenceError, required_artifact


@pytest.mark.parametrize("failure", ["dangling", "loop", "unwritable", "file"])
def test_output_failure_precedes_seal_and_fit(
    shadow, frames, tmp_path, monkeypatch, failure
):
    capsule, protocol, _target, _clock, receipt, _expected = unsealed(
        shadow, frames, tmp_path, monkeypatch
    )
    blocker = tmp_path / "blocker"
    if failure in {"dangling", "loop"}:
        blocker.symlink_to(
            tmp_path / ("absent" if failure == "dangling" else "blocker")
        )
        output = blocker / "records"
    elif failure == "unwritable":
        blocker.mkdir(mode=0o500)
        output = blocker
    else:
        blocker.write_bytes(b"preserve")
        output = blocker / "records"
    before = {p.name: p.read_bytes() for p in capsule.iterdir()}
    monkeypatch.setattr(
        shadow,
        "prepare_matrices",
        lambda *_: pytest.fail("fit before output preparation"),
    )
    expected_error = PermissionError if failure == "unwritable" else EvidenceError
    try:
        with pytest.raises(expected_error):
            shadow.finalize(
                capsule,
                output,
                protocol,
                source=None,
                mode="prospective",
                publication_receipt=receipt,
            )
        assert {p.name: p.read_bytes() for p in capsule.iterdir()} == before
        assert not (capsule / "evidence.lock.json").exists()
    finally:
        if failure == "unwritable":
            blocker.chmod(0o700)


def test_staging_lease_exists_before_sealing_and_is_reused(
    shadow, frames, tmp_path, monkeypatch
):
    capsule, protocol, target, _clock, receipt, expected = unsealed(
        shadow, frames, tmp_path, monkeypatch
    )
    seal = shadow.seal_publication
    stages = []

    def checked(*args):
        stages.extend(target.parent.glob("shadow-stage-*"))
        assert len(stages) == 1
        assert (target.parent / f".{target.name}.publishing").is_file()
        return seal(*args)

    monkeypatch.setattr(shadow, "seal_publication", checked)
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
    assert not stages[0].exists()
    actual = json.loads((target / "record.json").read_bytes())
    assert actual["predictions"] == expected["predictions"]
    assert actual["fitted"] == expected["fitted"]


@pytest.mark.parametrize("action", ["build", "completed"])
def test_internal_evidence_snapshot_mismatch_is_contract_failure(
    shadow, frames, tmp_path, monkeypatch, action
):
    capsule, protocol, target, clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    a = shadow.CapsuleSnapshot.load(capsule, prospective=True)
    b = shadow.CapsuleSnapshot.load(capsule, prospective=True)
    enrolled = shadow.enrollment(protocol)
    evidence = shadow.verified_seal(a, enrolled)
    if action == "build":
        call = partial(
            shadow.build_record, b, enrolled, now=clock.now(UTC), evidence=evidence
        )
    else:
        call = partial(
            shadow.completed_record,
            target / "record.json",
            json.loads((target / "record.json").read_bytes()),
            b,
            enrolled,
            evidence=evidence,
        )
    with pytest.raises(RuntimeError, match="must be the same object"):
        call()


def test_record_clock_uses_validated_sealing_time(
    shadow, frames, tmp_path, monkeypatch
):
    capsule, protocol, _target, clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    snapshot = shadow.CapsuleSnapshot.load(capsule, prospective=True)
    evidence = shadow.verified_seal(snapshot, shadow.enrollment(protocol))

    def timestamp_bug(*_):
        raise AssertionError("reparsed validated sealing clock")

    monkeypatch.setattr(shadow, "timestamp", timestamp_bug)
    assert shadow.record_clock(evidence, clock.now(UTC)) == (
        evidence.sealed_at,
        snapshot.hashes["evidence.lock.json"],
    )


def test_replay_evaluation_accepts_operator_protocol_alias(shadow, frames, tmp_path):
    capsule, protocol = make_capsule(shadow, frames, tmp_path)
    store = tmp_path / "records"
    shadow.record(capsule, store, protocol, mode="replay")
    truth = tmp_path / "truth.parquet"
    frames[-1].write_parquet(truth)
    alias = tmp_path / "protocol-alias.json"
    alias.symlink_to(protocol)
    result = shadow.evaluation_inputs(store, truth, alias, mode="replay")
    assert result.protocol_bytes == protocol.read_bytes()
    assert result.selected
    with pytest.raises(EvidenceError, match="regular local file"):
        required_artifact(alias)


@pytest.mark.parametrize("action", ["freeze", "freeze-truth"])
def test_missing_protocol_cli_is_named_refusal(shadow, tmp_path, action):
    command = [
        sys.executable,
        shadow.__file__,
        action,
        "--protocol",
        str(tmp_path / "missing.json"),
        "--out",
        str(tmp_path / "out"),
    ]
    if action == "freeze-truth":
        command.extend(["--truth", str(tmp_path / "truth.parquet")])
    result = subprocess.run(  # noqa: S603 -- fixed runner with synthetic missing paths
        command, capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 2
    assert "missing.json: missing required enrollment artifact" in result.stderr
    assert not (tmp_path / "out").exists()


def test_direct_truth_helper_missing_protocol_and_permission_classification(
    shadow, tmp_path, monkeypatch
):
    missing = tmp_path / "missing.json"
    with pytest.raises(EvidenceError, match="missing required enrollment artifact"):
        shadow.seal_truth(
            tmp_path / "truth.parquet",
            missing,
            tmp_path / "out",
            shadow.datetime.now(UTC),
        )

    def denied(*_):
        raise PermissionError("protocol permission")

    monkeypatch.setattr(Path, "read_bytes", denied)
    with pytest.raises(PermissionError, match="protocol permission"):
        required_artifact(Path(shadow.__file__), policy="operator")


def test_first_sigterm_ignores_repeat_until_cli_handlers_restore(shadow):
    original = signal.getsignal(signal.SIGTERM)
    with shadow.cli_signals():
        with pytest.raises(SystemExit) as caught:
            shadow.terminate_shadow(signal.SIGTERM, None)
        assert caught.value.code == 143
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
    assert signal.getsignal(signal.SIGTERM) is original


def test_multiline_refusal_printed_once_and_supplementary_stderr_bounded(capsys):
    wrapper = load_script(SCRIPT.with_name("hourly-predict-publish.py"))
    detail = "missing lock\nprotocol.json"
    value = {
        "kind": "temperature_shadow_preflight",
        "schema_version": 1,
        "status": "refused",
        "detail": detail,
    }
    check = subprocess.CompletedProcess(
        [],
        2,
        json.dumps(value),
        f"shadow refused: {detail}\n" + "secondary: " + "𝄞" * 10000,
    )
    ready, error = wrapper.preflight_result(check)
    output = capsys.readouterr().out
    assert not ready
    assert error == "shadow unavailable: " + detail
    assert "missing lock" not in output
    assert "protocol.json" not in output
    assert "secondary: " in output
    assert "[stderr truncated]" in output
    assert len(output.encode()) < 8300


def test_supplementary_stderr_bound_includes_per_line_labels(capsys):
    wrapper = load_script(SCRIPT.with_name("hourly-predict-publish.py"))
    wrapper.preflight_secondary(["x"] * 10000, "primary")
    output = capsys.readouterr().out
    assert len(output.encode()) <= 8192
    assert "[stderr truncated]" in output
