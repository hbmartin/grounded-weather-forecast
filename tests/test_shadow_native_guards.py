"""Native ownership, explicit preflight outcomes and shared preparation regressions."""

import errno
import importlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from filelock import FileLock
from test_launchd import SCRIPT, load_script
from test_shadow_evidence import evaluate
from test_shadow_evidence import mature_record as mature_record_fixture
from test_shadow_v3 import original_receipt, remove_record
from test_temperature_shadow import make_capsule, make_prospective_record

from grounded_weather_forecast.shadow_artifacts import EvidenceError

pytest_plugins = ["test_temperature_shadow"]


def window_protocol(path):
    now = datetime.now(UTC)
    path.write_text(
        json.dumps(
            {
                "window_start": (now - timedelta(days=1)).isoformat(),
                "window_end": (now + timedelta(days=1)).isoformat(),
            }
        )
    )
    return path


@pytest.mark.parametrize("missing_root", [False, True])
def test_missing_capsule_does_not_create_inputs(shadow, frames, tmp_path, missing_root):
    _capsule, protocol = make_capsule(shadow, frames, tmp_path)
    root = tmp_path / "absent-capsules"
    if not missing_root:
        root.mkdir(mode=0o700)
    missing = root / "20261001T120000Z"
    with pytest.raises(EvidenceError, match="missing capsule directory"):
        shadow.finalize(
            missing, tmp_path / "records", protocol, source=None, mode="prospective"
        )
    assert not missing.exists()
    assert root.exists() == (not missing_root)
    assert not (tmp_path / "records").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission contract")
@pytest.mark.parametrize("unsafe", ["root", "issuance"])
def test_prospective_permissions_refused_before_guard_creation(
    shadow, frames, tmp_path, unsafe
):
    capsule, protocol = make_capsule(shadow, frames, tmp_path)
    selected = capsule.parent if unsafe == "root" else capsule
    selected.chmod(0o755)
    with pytest.raises(EvidenceError, match=f"{selected}: private capsule directory"):
        shadow.finalize(
            capsule, tmp_path / "records", protocol, source=None, mode="prospective"
        )
    assert not (capsule / ".sealing").exists()
    assert selected.stat().st_mode & 0o777 == 0o755
    # Legacy replay keeps its permissions contract and still reads these inputs.
    assert shadow.CapsuleSnapshot.load(capsule, prospective=False).capsule == capsule


@pytest.mark.parametrize("alias_root", [False, True])
def test_finalization_refuses_aliases_before_guard_creation(
    shadow, frames, tmp_path, alias_root
):
    capsule, protocol = make_capsule(shadow, frames, tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(
        capsule.parent if alias_root else capsule, target_is_directory=True
    )
    selected = alias / capsule.name if alias_root else alias
    with pytest.raises(EvidenceError, match="nonaliased local directory"):
        shadow.finalize(
            selected, tmp_path / "records", protocol, source=None, mode="replay"
        )
    assert not (capsule / ".sealing").exists()


@pytest.mark.skipif(os.name != "posix", reason="inject Unix native locking failure")
def test_unsupported_native_locking_never_acquires_soft_ownership(
    shadow, tmp_path, monkeypatch
):
    unix = importlib.import_module("filelock._unix")

    def unsupported(_descriptor):
        raise OSError(errno.ENOSYS, "native locking unsupported")

    monkeypatch.setattr(unix, "_lock_fd_nonblocking", unsupported)
    for _ in range(2):
        with (
            pytest.raises(OSError, match="native locking unsupported") as caught,
            shadow.native_guard(tmp_path / ".guard", "test publication"),
        ):
            pytest.fail("must never use existence as ownership")
        assert caught.value.errno == errno.ENOSYS
    assert not (tmp_path / "output").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_native_guard_mode_and_sidecar_survive_release(shadow, tmp_path):
    guard = tmp_path / ".guard"
    with shadow.native_guard(guard, "test publication"):
        assert guard.stat().st_mode & 0o777 == 0o600
        with (
            pytest.raises(OSError, match="already running"),
            shadow.native_guard(guard, "test publication"),
        ):
            pytest.fail("two simultaneous owners")
    assert guard.is_file()
    with shadow.native_guard(guard, "test publication"):
        pass


def fail_release(monkeypatch):
    release = FileLock.release

    def broken(self, **kwargs):
        held = self.is_locked
        release(self, **kwargs)
        if held:
            raise OSError("injected release failure")

    monkeypatch.setattr(FileLock, "release", broken)


@pytest.mark.parametrize(
    "primary",
    [
        EvidenceError("bad evidence"),
        ValueError("computation defect"),
        KeyboardInterrupt("cancelled"),
    ],
)
def test_release_failure_preserves_primary_and_displays_secondary(
    shadow, tmp_path, monkeypatch, capsys, primary
):
    fail_release(monkeypatch)
    with (
        pytest.raises(type(primary)) as caught,
        shadow.native_guard(tmp_path / ".guard", "test publication"),
    ):
        raise primary
    assert caught.value is primary
    shadow.report_error(caught.value, "primary diagnostic")
    diagnostic = capsys.readouterr().err
    assert str(primary) in diagnostic
    assert "injected release failure" in diagnostic


def test_release_after_publication_reports_preserved_output(
    shadow, tmp_path, monkeypatch
):
    fail_release(monkeypatch)
    output = tmp_path / "complete"
    with (
        pytest.raises(OSError, match=f"complete output preserved at {output}"),
        shadow.staged_directory(output) as folder,
    ):
        (folder / "result").write_bytes(b"complete")
    assert (output / "result").read_bytes() == b"complete"


def test_release_cannot_hide_postrename_sync_diagnostic(shadow, tmp_path, monkeypatch):
    evidence = importlib.import_module("shadow_evidence")
    fail_release(monkeypatch)

    def sync_failed(_path):
        raise OSError("injected synchronization failure")

    monkeypatch.setattr(evidence, "sync_directory", sync_failed)
    output = tmp_path / "complete"
    with (
        pytest.raises(OSError, match="complete output preserved at") as caught,
        shadow.staged_directory(output) as folder,
    ):
        (folder / "result").write_bytes(b"complete")
    assert "synchronization failure" in str(caught.value)
    assert "injected release failure" in " ".join(caught.value.__notes__)
    assert (output / "result").read_bytes() == b"complete"


@pytest.mark.parametrize("missing", ["protocol", "lock"])
def test_missing_enrollment_has_tagged_preflight_refusal(
    shadow, tmp_path, capsys, missing
):
    protocol = tmp_path / "protocol.json"
    if missing == "lock":
        window_protocol(protocol)
    assert shadow.preflight_cli(protocol) == 2
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["kind"] == "temperature_shadow_preflight"
    assert result["schema_version"] == 1
    assert result["status"] == "refused"
    assert "missing required enrollment artifact" in result["detail"]
    assert "shadow refused" in captured.err


def test_preflight_permission_failure_remains_operational(
    shadow, tmp_path, monkeypatch, capsys
):
    protocol = window_protocol(tmp_path / "protocol.json")

    def denied(*_args, **_kwargs):
        raise PermissionError("enrollment lock access denied")

    monkeypatch.setattr(shadow, "enrollment", denied)
    assert shadow.preflight_cli(protocol) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "runtime_failure"
    assert "shadow runtime failure: enrollment lock access denied" in captured.err


@pytest.mark.parametrize(
    ("result", "code"),
    [
        (None, 2),
        ({"status": "refused", "detail": "untagged refusal"}, 2),
        (
            {
                "kind": "temperature_shadow_preflight",
                "schema_version": True,
                "status": "refused",
                "detail": "bad version",
            },
            2,
        ),
        (
            {
                "kind": "temperature_shadow_preflight",
                "schema_version": 1,
                "status": "ready",
                "detail": "wrong exit",
            },
            2,
        ),
        (
            {
                "kind": "temperature_shadow_preflight",
                "schema_version": 1,
                "status": "refused",
                "detail": "wrong exit",
            },
            1,
        ),
    ],
)
def test_unknown_preflight_failures_are_not_capability_claims(
    tmp_path, monkeypatch, result, code
):
    wrapper = load_script(SCRIPT.with_name("hourly-predict-publish.py"))
    protocol = window_protocol(tmp_path / "protocol.json")
    monkeypatch.setattr(
        wrapper,
        "run_shadow_child",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, code, json.dumps(result), "launcher or usage failure"
        ),
    )
    paths = SimpleNamespace(project=tmp_path, command=lambda _name: [sys.executable])
    enabled, error = wrapper.shadow_preflight(paths, protocol)
    assert not enabled
    assert (
        error
        == f"shadow preflight runtime failure: selected runtime exited {code}: launcher or usage failure"
    )


def test_wrapper_unknown_action_preserves_ordinary_publishing(
    tmp_path, monkeypatch, capsys
):
    wrapper = load_script(SCRIPT.with_name("hourly-predict-publish.py"))
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "temperature_shadow.py").write_text(
        "import argparse\np=argparse.ArgumentParser()\np.add_argument('action', choices=['freeze','freeze-truth','record','finalize','evaluate'])\np.parse_args()\n"
    )
    protocol = window_protocol(tmp_path / "protocol.json")
    paths = SimpleNamespace(
        project=tmp_path,
        forecast=tmp_path / "forecast.json",
        command=lambda _name: [sys.executable],
    )
    monkeypatch.setattr(wrapper, "resolve_runtime", lambda: paths)
    monkeypatch.setattr(wrapper, "maybe_spawn_restore", lambda _paths: None)
    calls = []
    monkeypatch.setattr(
        wrapper, "run_predict", lambda _paths: calls.append("predict") or 0
    )
    monkeypatch.setattr(
        wrapper, "publish", lambda _paths, _station: calls.append("publish") or 0
    )
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "shadow"
    assert (
        wrapper.main(["--shadow-root", str(root), "--shadow-protocol", str(protocol)])
        == 1
    )
    assert calls == ["predict", "publish"]
    assert not root.exists()
    diagnostic = capsys.readouterr().out
    assert "ordinary publication exited 0" in diagnostic
    assert "invalid choice: 'preflight'" in diagnostic


@pytest.mark.parametrize(
    "invalid", ["fx__bad", "fx____temp_c", "fx__a__temp_c__extra", "fxd__a__temp_c"]
)
def test_known_invalid_forecast_names_are_explicit_refusals(
    shadow, frames, tmp_path, invalid
):
    capsule, protocol = make_capsule(shadow, frames, tmp_path)
    training = capsule / "training.parquet"
    pl.read_parquet(training).rename({"fx__a__temp_c": invalid}).write_parquet(training)
    with pytest.raises(EvidenceError, match="invalid temperature forecast column"):
        shadow.record(capsule, tmp_path / "records", protocol, mode="replay")
    assert not (tmp_path / "records").exists()


def test_unexpected_transform_defect_is_cli_failure(shadow, frames, tmp_path):
    capsule, protocol = make_capsule(shadow, frames, tmp_path)
    program = """
import runpy, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).parent))
import grounded_weather_forecast.dataset.matrix as matrix
def broken(*args, **kwargs):
    raise TypeError('injected transformation bug')
matrix.to_supervised_slice = broken
sys.argv = [sys.argv[1], 'record', '--input', sys.argv[2], '--protocol', sys.argv[3], '--out', sys.argv[4]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(  # noqa: S603 -- synthetic input and injected local fault
        [
            sys.executable,
            "-c",
            program,
            shadow.__file__,
            str(capsule),
            str(protocol),
            str(tmp_path / "records"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 1
    assert "TypeError: injected transformation bug" in result.stderr
    assert "shadow refused" not in result.stderr
    assert not (tmp_path / "records").exists()


def test_first_seal_prepares_once_and_keeps_predictions(
    shadow, frames, tmp_path, monkeypatch
):
    capsule, protocol, target, _clock = make_prospective_record(
        shadow, frames, tmp_path, monkeypatch
    )
    expected = json.loads((target / "record.json").read_bytes())
    receipt = original_receipt(capsule)
    remove_record(target)
    for name in ("published.json", "publication.json", "evidence.lock.json"):
        (capsule / name).unlink()
    calls = Counter()
    for name in ("known_training", "to_supervised_slice", "to_forecast_matrix"):
        original = getattr(shadow, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(shadow, name, counted)
    assert (
        shadow.finalize(
            capsule,
            target.parent,
            protocol,
            source=None,
            publication_receipt=receipt,
            mode="prospective",
        )
        == target
    )
    assert calls == dict.fromkeys(
        ("known_training", "to_supervised_slice", "to_forecast_matrix"), 1
    )
    saved = json.loads((target / "record.json").read_bytes())
    assert saved["predictions"] == expected["predictions"]
    assert saved["fitted"] == expected["fitted"]


def test_evaluation_prepares_once_per_record(shadow, frames, tmp_path, monkeypatch):
    retained = mature_record_fixture.__wrapped__(shadow, frames, tmp_path, monkeypatch)
    prepare = shadow.prepare_matrices
    calls = []

    def counted(*args):
        calls.append(True)
        return prepare(*args)

    monkeypatch.setattr(shadow, "prepare_matrices", counted)
    evaluate(shadow, retained, tmp_path / "evaluation")
    assert calls == [True]


def test_snapshot_equality_is_identity_and_mode_is_explicit(shadow, frames, tmp_path):
    capsule, _protocol = make_capsule(shadow, frames, tmp_path)
    left = shadow.CapsuleSnapshot.load(capsule, prospective=False)
    right = shadow.CapsuleSnapshot.load(capsule, prospective=False)
    left.parquet("training.parquet")
    right.parquet("training.parquet")
    assert left == left
    assert left != right
    with pytest.raises(TypeError, match="prospective"):
        shadow.CapsuleSnapshot.load(capsule)


def test_prediction_compatibility_wrapper_uses_same_prepared_math(shadow, frames):
    origin, train, issue, truth = frames
    original, fitted = shadow.predictions(train, issue, truth, origin)
    current, current_fitted = shadow.prepared_predictions(
        shadow.prepare_matrices(train, issue, truth, origin)
    )
    assert current_fitted == fitted
    for method in original:
        np.testing.assert_array_equal(current[method], original[method])


@pytest.mark.skipif(os.name != "posix", reason="launchd process-group ownership")
def test_timeout_terminates_descendant_and_reaps_launcher(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    support = importlib.import_module("launchd_paths")
    guard = tmp_path / ".descendant"
    marker = tmp_path / "started"
    descendant = """
import sys,time
from pathlib import Path
from filelock import FileLock
with FileLock(sys.argv[1]):
    Path(sys.argv[2]).write_text('holding native guard')
    print('descendant ready',flush=True)
    time.sleep(30)
"""
    launcher = """
import subprocess,sys
child = subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],sys.argv[3]])
child.wait()
"""
    popen = subprocess.Popen
    owners = []

    def capture_owner(*args, **kwargs):
        owner = popen(*args, **kwargs)
        owners.append(owner)
        return owner

    monkeypatch.setattr(support.subprocess, "Popen", capture_owner)
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        support.run_shadow_child(
            [sys.executable, "-c", launcher, descendant, str(guard), str(marker)],
            timeout=2,
            capture_output=True,
        )
    assert time.monotonic() - started < 10
    assert marker.read_text() == "holding native guard"
    assert b"descendant ready" in caught.value.output
    assert owners[0].returncode is not None
    with pytest.raises(ChildProcessError):
        os.waitpid(owners[0].pid, os.WNOHANG)
    with FileLock(guard, timeout=0):
        pass  # The descendant's native ownership ended with its process group.


@pytest.mark.skipif(os.name != "posix", reason="launchd process-group cancellation")
def test_cancellation_keeps_primary_when_process_group_cleanup_fails(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    support = importlib.import_module("launchd_paths")
    cancelled = KeyboardInterrupt("operator cancelled")
    closed = []
    process = SimpleNamespace(
        pid=123456,
        stdout=SimpleNamespace(close=lambda: closed.append(True)),
        stderr=None,
    )

    def interrupted(**_kwargs):
        raise cancelled

    process.communicate = interrupted
    monkeypatch.setattr(support.subprocess, "Popen", lambda *_args, **_kwargs: process)

    def denied(*_args):
        raise PermissionError("group cleanup denied")

    monkeypatch.setattr(support.os, "killpg", denied)
    with pytest.raises(KeyboardInterrupt) as caught:
        support.run_shadow_child(["synthetic"], timeout=30)
    assert caught.value is cancelled
    assert "group cleanup denied" in " ".join(cancelled.__notes__)
    assert closed == [True]
