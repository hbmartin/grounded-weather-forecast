"""PR #102 recovery, input-boundary and snapshot regressions."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from filelock import FileLock
from test_launchd import SCRIPT, load_script
from test_shadow_evidence import evaluate
from test_shadow_evidence import mature_record as mature_record_fixture
from test_shadow_v3 import original_receipt, remove_record
from test_temperature_shadow import (
    make_capsule,
    make_prospective_record,
    template_protocol,
)

from grounded_weather_forecast import cli, status
from grounded_weather_forecast.shadow_artifacts import THREAD_CONTROLS, EvidenceError

pytest_plugins = ["test_temperature_shadow"]


@pytest.fixture
def mature_record(shadow, frames, tmp_path, monkeypatch):
    return mature_record_fixture.__wrapped__(shadow, frames, tmp_path, monkeypatch)


def run_shadow(shadow, *arguments):
    return subprocess.run(  # noqa: S603 -- synthetic local CLI inputs
        [sys.executable, shadow.__file__, *map(str, arguments)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize("after_rename", [False, True])
def test_hard_exit_releases_publication_guard(shadow, tmp_path, after_rename):
    target = tmp_path / "result"
    program = """
import os, sys
from pathlib import Path
from shadow_evidence import staged_directory
output = Path(sys.argv[1])
if sys.argv[2] == 'True':
    rename = Path.rename
    def exit_after_rename(self, target):
        rename(self, target)
        os._exit(7)
    Path.rename = exit_after_rename
with staged_directory(output) as folder:
    (folder / 'record.json').write_bytes(b'complete')
    if sys.argv[2] == 'False':
        os._exit(7)
"""
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(Path(shadow.__file__).parent), os.environ.get("PYTHONPATH", "")]
        ),
    }
    result = subprocess.run(  # noqa: S603 -- fixed child exits inside synthetic staging
        [sys.executable, "-c", program, str(target), str(after_rename)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 7, result.stderr
    with FileLock(tmp_path / ".result.publishing", timeout=0):
        assert target.exists() == after_rename
    if after_rename:
        assert (target / "record.json").read_bytes() == b"complete"
        with pytest.raises(FileExistsError), shadow.staged_directory(target):
            pytest.fail("published output must stay exclusive")
    else:
        with shadow.staged_directory(target) as folder:
            (folder / "record.json").write_bytes(b"retry")
        assert (target / "record.json").read_bytes() == b"retry"


def test_live_publication_owner_and_unsafe_guard_are_refused(shadow, tmp_path):
    target = tmp_path / "result"
    guard = tmp_path / ".result.publishing"
    with (
        FileLock(guard),
        pytest.raises(OSError, match="publication is already running"),
        shadow.staged_directory(target),
    ):
        pytest.fail("another writer owns publication")
    guard.unlink()
    guard.symlink_to(tmp_path / "elsewhere")
    with (
        pytest.raises(EvidenceError, match="regular local file"),
        shadow.staged_directory(target),
    ):
        pytest.fail("a guard alias must not be followed")


@pytest.mark.parametrize("after_rename", [False, True])
def test_finalize_recovers_a_hard_exit_at_record_publication(
    shadow, frames, tmp_path, monkeypatch, after_rename
):
    capsule, protocol, target, clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    remove_record(target)
    program = """
import os, runpy, sys
from pathlib import Path
script, capsule, output, protocol, after = sys.argv[1:]
sys.path.insert(0, str(Path(script).parent))
runner = runpy.run_path(script)
rename = Path.rename
def interrupted(self, target):
    if Path(target).parent == Path(output):
        if after == 'True':
            rename(self, target)
        os._exit(7)
    return rename(self, target)
Path.rename = interrupted
runner['finalize'](Path(capsule), Path(output), Path(protocol), source=None, mode='prospective')
"""
    result = subprocess.run(  # noqa: S603 -- real synthetic finalization interrupted at publication
        [
            sys.executable,
            "-c",
            program,
            shadow.__file__,
            str(capsule),
            str(target.parent),
            str(protocol),
            str(after_rename),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 7, result.stderr
    clock.instant = shadow.timestamp(
        json.loads(protocol.read_bytes())["window_end"]
    ) + timedelta(hours=98)
    if after_rename:
        monkeypatch.setattr(
            shadow,
            "predictions",
            lambda *_a: pytest.fail("completed recovery must not refit"),
        )
    assert (
        shadow.finalize(
            capsule, target.parent, protocol, source=None, mode="prospective"
        )
        == target
    )
    saved = json.loads((target / "record.json").read_bytes())
    shadow.completed_record(
        target / "record.json", saved, capsule, shadow.enrollment(protocol)
    )


def test_finalize_reads_state_only_after_sealing_acquisition(
    shadow, frames, tmp_path, monkeypatch
):
    capsule, protocol, target, _clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    publication = capsule / "publication.json"
    raw = publication.read_bytes()
    publication.unlink()
    acquire = FileLock.acquire

    def peer_finishes(self, *args, **kwargs):
        result = acquire(self, *args, **kwargs)
        publication.write_bytes(raw)
        return result

    monkeypatch.setattr(FileLock, "acquire", peer_finishes)
    monkeypatch.setattr(
        shadow, "predictions", lambda *_a: pytest.fail("held records must not refit")
    )
    assert (
        shadow.finalize(
            capsule, target.parent, protocol, source=None, mode="prospective"
        )
        == target
    )


@pytest.mark.parametrize("fault", ["identity", "protocol", "lock", "sync", "rename"])
def test_failed_freeze_never_accepts_partial_enrollment(
    shadow, tmp_path, monkeypatch, fault
):
    import shadow_evidence  # noqa: PLC0415 -- shadow fixture establishes script imports

    protocol = template_protocol(shadow)
    start = datetime.now(UTC) + timedelta(days=1)
    protocol.update(
        window_start=start.isoformat(),
        window_end=(start + timedelta(days=42)).isoformat(),
    )
    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps(protocol))
    output = tmp_path / "enrollment"
    with monkeypatch.context() as patch:

        def fail(*_args, **_kwargs):
            raise OSError("injected enrollment failure")

        if fault == "identity":
            patch.setattr(shadow, "implementation_lock", fail)
        elif fault in {"protocol", "lock"}:
            write = shadow.write_bytes_new

            def fail_write(path, raw):
                if path.name == (
                    "protocol.json" if fault == "protocol" else "protocol.lock.json"
                ):
                    fail()
                return write(path, raw)

            patch.setattr(shadow, "write_bytes_new", fail_write)
        elif fault == "sync":
            patch.setattr(shadow_evidence, "sync_files", fail)
        else:
            patch.setattr(Path, "rename", fail)
        with pytest.raises(OSError, match="injected enrollment failure"):
            shadow.freeze(proposal, output)
    assert not output.exists()
    assert not list(tmp_path.glob("shadow-stage-*"))
    frozen = shadow.freeze(proposal, output)
    assert shadow.enrollment(frozen).protocol == protocol


def test_cleanup_cannot_reverse_success_and_required_sync_failure_preserves_output(
    shadow, tmp_path, monkeypatch
):
    import shadow_evidence  # noqa: PLC0415 -- shadow fixture establishes script imports

    def fail(*_args, **_kwargs):
        raise OSError("injected cleanup")

    with monkeypatch.context() as patch:
        patch.setattr(shadow_evidence.shutil, "rmtree", fail)
        with shadow.staged_directory(tmp_path / "complete") as folder:
            (folder / "result").write_text("kept")
    assert (tmp_path / "complete/result").read_text() == "kept"
    with monkeypatch.context() as patch:
        patch.setattr(shadow_evidence, "sync_directory", fail)
        with (
            pytest.raises(OSError, match="complete output preserved at"),
            shadow.staged_directory(tmp_path / "uncertain") as folder,
        ):
            (folder / "result").write_text("preserved")
    assert (tmp_path / "uncertain/result").read_text() == "preserved"


@pytest.mark.parametrize(
    "fault", ["forecast-columns", "training-type", "truth-bytes", "completion-bytes"]
)
def test_all_input_schemas_are_refused_before_sealing(
    shadow, frames, tmp_path, monkeypatch, fault
):
    capsule, protocol, target, _clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    receipt = original_receipt(capsule)
    remove_record(target)
    for name in ("publication.json", "published.json", "evidence.lock.json"):
        (capsule / name).unlink()
    path = capsule / "training.parquet"
    expected = "matrix has no forecast columns"
    if fault == "forecast-columns":
        pl.read_parquet(path).select(pl.exclude("^fx__.*$")).write_parquet(path)
    elif fault == "training-type":
        pl.read_parquet(path).with_columns(
            pl.lit("bad").alias("fx__a__temp_c")
        ).write_parquet(path)
        expected = "training.parquet.fx__a__temp_c: expected numeric"
    else:
        path = capsule / (
            "truth_hourly.parquet"
            if fault == "truth-bytes"
            else "provider_completions.parquet"
        )
        path.write_bytes(b"bad parquet")
        expected = path.name + ": invalid Parquet"
    manifest = capsule / "capsule.json"
    meta = json.loads(manifest.read_bytes())
    meta["input_sha256"][path.name] = shadow.digest(path)
    manifest.write_text(json.dumps(meta))
    before = {p.name: p.read_bytes() for p in capsule.iterdir()}
    with pytest.raises(EvidenceError, match=expected):
        shadow.finalize(
            capsule,
            target.parent,
            protocol,
            publication_receipt=receipt,
            source=None,
            mode="prospective",
        )
    assert {p.name: p.read_bytes() for p in capsule.iterdir()} == before
    assert not target.exists()


@pytest.mark.parametrize("fault", ["identity", "rows", "value", "truth"])
def test_replay_input_errors_have_specific_cli_refusals(
    shadow, frames, tmp_path, fault
):
    capsule, protocol = make_capsule(shadow, frames, tmp_path)
    target = shadow.record(capsule, tmp_path / "records", protocol, mode="replay")
    path = target / "record.json"
    saved = json.loads(path.read_bytes())
    truth = capsule / "truth_hourly.parquet"
    if fault == "identity":
        del saved["code_identity"]
        expected = "code_identity: requires nonempty string"
    elif fault == "rows":
        saved["predictions"] = [1, 2]
        expected = "predictions[0]: expected an object"
    elif fault == "value":
        saved["predictions"][0]["candidate"] = True
        expected = "candidate: expected finite number or null"
    else:
        truth.write_bytes(b"not parquet")
        expected = "truth: invalid Parquet input"
    path.write_text(json.dumps(saved))
    output = tmp_path / "evaluation"
    result = run_shadow(
        shadow,
        "evaluate",
        "--input",
        target.parent,
        "--protocol",
        protocol,
        "--out",
        output,
        "--truth",
        truth,
    )
    assert result.returncode == 2
    assert expected in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_capsule_bytes_are_read_once_and_cached_source_is_unmodified(
    shadow, mature_record, tmp_path, monkeypatch
):
    capsule, *_ = mature_record
    selected = {p for p in capsule.iterdir() if p.name != ".sealing"}
    counts = Counter()
    file_bytes = {p.read_bytes() for p in selected}
    hashes = Counter()
    read = Path.read_bytes
    sha256 = hashlib.sha256

    def counted(path):
        if path in selected:
            counts[path.name] += 1
        return read(path)

    def hashed(payload=b"", *args, **kwargs):
        if isinstance(payload, bytes) and payload in file_bytes:
            hashes[payload] += 1
        return sha256(payload, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", counted)
        patch.setattr(hashlib, "sha256", hashed)
        evaluate(shadow, mature_record, tmp_path / "evaluation")
    assert counts == dict.fromkeys((p.name for p in selected), 1)
    assert hashes == dict.fromkeys(file_bytes, 1)
    snapshot = shadow.CapsuleSnapshot.load(capsule, prospective=True)
    original = snapshot.document("production.json", "source")
    shadow.production_rows(
        capsule, snapshot.metadata(), prospective=True, snapshot=snapshot
    )
    assert snapshot.document("production.json", "source") == original


def test_finalization_reuses_input_bytes_and_freezes_one_implementation_identity(
    shadow, frames, tmp_path, monkeypatch
):
    capsule, protocol, target, _clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    receipt = original_receipt(capsule)
    remove_record(target)
    for name in ("publication.json", "published.json", "evidence.lock.json"):
        (capsule / name).unlink()
    selected = {p for p in capsule.iterdir() if p.name != ".sealing"}
    counts = Counter()
    read, identity = Path.read_bytes, shadow.implementation_lock
    identities = []

    def counted(path):
        if path in selected:
            counts[path.name] += 1
        return read(path)

    def counted_identity(*args, **kwargs):
        identities.append(1)
        return identity(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", counted)
        patch.setattr(shadow, "implementation_lock", counted_identity)
        shadow.finalize(
            capsule,
            target.parent,
            protocol,
            publication_receipt=receipt,
            source=capsule / "production.json",
            mode="prospective",
        )
    assert counts == dict.fromkeys((p.name for p in selected), 1)
    assert len(identities) == 1
    shadow.validate_record(
        target / "record.json",
        json.loads((target / "record.json").read_bytes()),
        capsule,
        protocol,
    )


@pytest.mark.parametrize("code", [-1, -int(signal.SIGTERM)])
def test_internal_negatives_cannot_gain_pipeline_exemptions(monkeypatch, code):
    calls = []
    monkeypatch.setattr(cli, "_record_run", lambda *_a, **_k: None)
    result = cli._run_pipeline_steps(
        None,
        argparse.Namespace(command="maintain"),
        (
            (
                "backtest",
                lambda: status.CommandOutcome(code, "promotion_evidence_invalid"),
            ),
            ("report", lambda: calls.append("report") or 0),
        ),
        continue_on_failure_kinds=frozenset({"promotion_evidence_invalid"}),
    )
    assert result.exit_code == 1
    assert result.invalid
    assert calls == []
    assert "promotion_evidence_invalid" in status.failure_details_json(
        result.diagnostics
    )
    assert str(code) in status.failure_details_json(result.diagnostics)


@pytest.mark.parametrize("code", [True, -256, 256, 2**32])
def test_bad_subprocess_codes_still_fail_closed(code):
    result = status.CommandOutcome.from_subprocess(code)
    assert result.exit_code == 1
    assert result.invalid


@pytest.mark.parametrize("wrong_pool", [False, True])
def test_thread_contract_refusal_in_an_isolated_runtime(tmp_path, wrong_pool):
    program = """
import os, sys
from pathlib import Path
from grounded_weather_forecast.shadow_artifacts import THREAD_CONTROLS, EvidenceError, runtime_identity
os.environ.update({key: '2' for key in THREAD_CONTROLS})
if sys.argv[1] == 'True':
    os.environ['POLARS_MAX_THREADS'] = '1'
    import polars as pl
    assert pl.thread_pool_size() == 1
    os.environ['POLARS_MAX_THREADS'] = '2'
else:
    os.environ['OMP_NUM_THREADS'] = '1'
try:
    runtime_identity(Path(sys.argv[2]))
except (EvidenceError,) as error:
    print(error)
    raise SystemExit(2)
raise SystemExit(3)
"""
    lock = tmp_path / "uv.lock"
    lock.write_text("synthetic")
    result = subprocess.run(  # noqa: S603 -- isolated numerical initialization regression
        [sys.executable, "-c", program, str(wrong_pool), str(lock)],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 2, result.stderr
    assert "requires two-thread controls before numerical imports" in result.stdout
    assert all(os.environ[name] == "2" for name in THREAD_CONTROLS)


@pytest.mark.parametrize(
    "response",
    ["ready", "outside_window", "invalid", "timeout", "refusal", "runtime", "spawn"],
)
def test_wrapper_named_preflight_status_and_failure_isolation(
    tmp_path, monkeypatch, capsys, response
):
    wrapper = load_script(SCRIPT.with_name("hourly-predict-publish.py"))
    now = datetime.now(UTC)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "window_start": (now - timedelta(days=1)).isoformat(),
                "window_end": (now + timedelta(days=1)).isoformat(),
            }
        )
    )
    paths = SimpleNamespace(project=tmp_path, command=lambda _name: [sys.executable])
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["timeout"] == 30
        if response == "timeout":
            raise subprocess.TimeoutExpired(command, 30)
        if response == "spawn":
            raise OSError("selected runtime could not start")
        if response in {"refusal", "runtime"}:
            return subprocess.CompletedProcess(
                command,
                2 if response == "refusal" else 1,
                "",
                "specific child diagnostic",
            )
        return subprocess.CompletedProcess(
            command, 0, json.dumps({"status": response}), ""
        )

    monkeypatch.setattr(wrapper.subprocess, "run", run)
    clocks = iter((10.0, 40.0))
    monkeypatch.setattr(
        wrapper, "time", SimpleNamespace(monotonic=lambda: next(clocks))
    )
    enabled, error = wrapper.shadow_preflight(paths, protocol)
    assert commands[0][1:] == [
        str(tmp_path / "scripts/temperature_shadow.py"),
        "preflight",
        "--protocol",
        str(protocol),
    ]
    assert enabled == (response == "ready")
    assert (error is None) == (response in {"ready", "outside_window"})
    assert "shadow preflight duration_s=30.000" in capsys.readouterr().out
    if response == "refusal":
        assert error.startswith("shadow unavailable:")
        assert error.endswith("specific child diagnostic")
    elif response == "runtime":
        assert error.startswith("shadow preflight runtime failure:")
        assert error.endswith("specific child diagnostic")


def test_preflight_cli_has_no_output_requirement_or_writes(shadow, tmp_path):
    protocol = tmp_path / "protocol.json"
    start = datetime.now(UTC) + timedelta(days=1)
    protocol.write_text(
        json.dumps(
            {
                "window_start": start.isoformat(),
                "window_end": (start + timedelta(days=42)).isoformat(),
            }
        )
    )
    result = run_shadow(shadow, "preflight", "--protocol", protocol)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"status": "outside_window"}
    assert list(tmp_path.iterdir()) == [protocol]


def test_legacy_replay_needs_no_v3_manifest_or_reviewed_truth_seal(
    shadow, frames, tmp_path
):
    capsule, protocol = make_capsule(shadow, frames, tmp_path)
    target = shadow.record(capsule, tmp_path / "records", protocol, mode="replay")
    record = target / "record.json"
    saved = json.loads(record.read_bytes())
    for name in ("artifact_version", "evidence_lock_sha256", "evidence_sealed_at"):
        saved.pop(name)
    record.write_text(json.dumps(saved))
    (target / "record.lock.json").unlink()
    truth = tmp_path / "legacy.parquet"
    pl.DataFrame(
        {
            "valid_hour": [
                frames[0] + timedelta(hours=1),
                frames[0] + timedelta(minutes=90),
            ],
            shadow.TARGET: [float("nan"), 15.0],
        }
    ).write_parquet(truth)
    output = tmp_path / "evaluation"
    shadow.evaluate(target.parent, truth, protocol, output, mode="replay")
    audit = json.loads((output / "evaluation.json").read_bytes())
    assert audit["truth_lock_sha256"] is None
    assert audit["truth_sha256"] == shadow.digest(truth)
    assert audit["record_sha256"] == {target.name: shadow.digest(record)}
