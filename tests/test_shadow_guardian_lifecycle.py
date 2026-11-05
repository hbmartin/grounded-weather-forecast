"""Group liveness independent of reservation, plus resource and diagnostic faults."""

import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from shadow_supervision_helpers import raw_owner, statuses
from test_launchd import SCRIPT, load_script
from test_shadow_supervision import supervisor as supervisor
from test_shadow_supervision import unlocked, wait_for

from grounded_weather_forecast.shadow_artifacts import EvidenceError, required_artifact

pytestmark = pytest.mark.skipif(os.name != "posix", reason="native POSIX ownership")


def test_guardian_reserves_group_from_outside_and_clean_exit_has_no_grace(
    supervisor, tmp_path
):
    marker = tmp_path / "group"
    command = [
        sys.executable,
        "-c",
        "import os,sys; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpgrp()))",
        str(marker),
    ]
    with raw_owner(supervisor, command, timeout=10) as (process, control, channel):
        control.send(b"S")
        wait_for(lambda: statuses(channel) == ["worker_ready"])
        owner = channel.owned_pid
        assert owner is not None
        assert os.getpgid(owner) == os.getpgid(process.pid)
        assert os.getpgid(owner) != owner
        started = time.monotonic()
        control.send(b"A")
        wait_for(lambda: len(statuses(channel)) == 2)
        assert time.monotonic() - started < supervisor.TERM_GRACE
        assert marker.read_text() == str(owner)
        assert channel.messages[-1]["returncode"] == 0
        assert not supervisor.group_exists(owner)
        control.send(b"R")
        assert process.wait(timeout=3) == 0


def test_empty_group_cleanup_never_sends_a_signal(supervisor, monkeypatch):
    calls = []
    monkeypatch.setattr(supervisor, "group_exists", lambda _pid: False)
    monkeypatch.setattr(supervisor, "observe_exit", lambda _process: 0)
    monkeypatch.setattr(supervisor, "signal_group", lambda *_: calls.append(True))
    clock = supervisor.Clock(lambda: 0, lambda _: pytest.fail("unnecessary grace"))
    assert supervisor.stop_worker(SimpleNamespace(pid=123456), clock=clock) == 0
    assert calls == []


def test_cancel_after_spawn_before_guardian_ready_never_admits(
    supervisor, tmp_path, monkeypatch
):
    read, write = os.pipe()
    status_read, status_write = os.pipe()
    control = supervisor.ControlChannel(write)
    channel = supervisor.StatusChannel(status_read)
    original = supervisor.start_worker
    marker = tmp_path / "never-executed"

    def cancel_after_spawn(*args):
        worker = original(*args)
        control.cancel()
        return worker

    monkeypatch.setattr(supervisor, "start_worker", cancel_after_spawn)
    channel.start(control)
    control.send(b"S")
    try:
        assert (
            supervisor.supervise(
                [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                ],
                read,
                status_write,
                time.monotonic() + 5,
            )
            == 0
        )
        channel.finish()
        assert channel.messages[-1]["status"] == "cancelled"
        assert channel.messages[-1]["cleanup_complete"]
        assert not marker.exists()
    finally:
        channel.stop()
        control.close()
        os.close(status_read)


@pytest.mark.parametrize("name", ["PIPE", "XFSZ"])
def test_native_exec_restores_standard_signal_defaults(supervisor, name):
    command = ["/bin/sh", "-c", f"kill -s {name} $$; printf survived"]
    expected = subprocess.run(command, text=True, capture_output=True)
    actual = supervisor.run_supervised(command, timeout=5, capture_output=True)
    assert actual.returncode == expected.returncode
    assert actual.returncode < 0
    assert actual.stdout == ""


def test_reader_start_failure_preserves_error_closes_pipes_and_reaps(
    supervisor, monkeypatch
):
    allocated, owners = [], []
    pipe, popen = os.pipe, subprocess.Popen
    failure = RuntimeError("cannot start new thread")

    def tracked_pipe():
        pair = pipe()
        allocated.extend(pair)
        return pair

    def tracked_process(*args, **kwargs):
        process = popen(*args, **kwargs)
        owners.append(process)
        return process

    def unavailable(_self):
        raise failure

    monkeypatch.setattr(supervisor.os, "pipe", tracked_pipe)
    monkeypatch.setattr(supervisor.subprocess, "Popen", tracked_process)
    monkeypatch.setattr(supervisor.threading.Thread, "start", unavailable)
    with pytest.raises(
        OSError, match="cannot start shadow status reader: cannot start new thread"
    ) as caught:
        supervisor.run_supervised([sys.executable, "-c", "pass"], timeout=5)
    assert caught.value.__cause__ is failure
    assert not getattr(caught.value, "__notes__", ())
    assert owners[0].returncode is not None
    with pytest.raises(ChildProcessError):
        os.waitpid(owners[0].pid, os.WNOHANG)
    for descriptor in allocated:
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(descriptor)


def test_reader_resource_fault_allows_ordinary_publication(
    supervisor, tmp_path, monkeypatch
):
    wrapper = load_script(SCRIPT.with_name("hourly-predict-publish.py"))
    paths = wrapper.RuntimePaths(tmp_path, "unused")
    protocol = tmp_path / "protocol.json"
    now = datetime.now(UTC)
    protocol.write_text(
        json.dumps(
            {
                "window_start": (now - timedelta(days=1)).isoformat(),
                "window_end": (now + timedelta(days=1)).isoformat(),
            }
        )
    )
    calls = []

    def child(*_args, **_kwargs):
        return supervisor.run_supervised([sys.executable, "-c", "pass"], timeout=5)

    def unavailable(_self):
        raise RuntimeError("cannot start new thread")

    monkeypatch.setattr(supervisor.threading.Thread, "start", unavailable)
    monkeypatch.setattr(wrapper, "run_shadow_child", child)
    monkeypatch.setattr(wrapper, "resolve_runtime", lambda: paths)
    monkeypatch.setattr(wrapper, "support_dir", lambda: tmp_path)
    monkeypatch.setattr(wrapper, "run_predict", lambda *_: calls.append("predict") or 0)
    monkeypatch.setattr(wrapper, "maybe_spawn_restore", lambda *_: None)
    monkeypatch.setattr(wrapper, "publish", lambda *_: calls.append("publish") or 0)
    assert (
        wrapper.main(
            [
                "--shadow-root",
                str(tmp_path / "shadow"),
                "--shadow-protocol",
                str(protocol),
            ]
        )
        == 1
    )
    assert calls == ["predict", "publish"]


def test_latched_protocol_error_allows_graceful_cleanup(
    supervisor, tmp_path, monkeypatch
):
    ready, cleaned = tmp_path / "ready", tmp_path / "cleaned"
    program = """
import signal,sys,time
from pathlib import Path
def stop(*_):
 signal.signal(signal.SIGTERM, signal.SIG_IGN)
 time.sleep(.2)
 Path(sys.argv[2]).touch()
 raise SystemExit(143)
signal.signal(signal.SIGTERM,stop)
print('worker diagnostic',file=sys.stderr,flush=True)
Path(sys.argv[1]).touch()
time.sleep(30)
"""
    original = supervisor.StatusChannel.collect
    primary = OSError("injected malformed status")

    def collect(self, *, check_errors=True):
        if check_errors and ready.exists():
            self.error = self.error or primary
        return original(self, check_errors=check_errors)

    monkeypatch.setattr(supervisor.StatusChannel, "collect", collect)
    monkeypatch.setattr(
        supervisor,
        "emergency_stop",
        lambda *_: pytest.fail("bypassed graceful cleanup"),
    )
    with pytest.raises(OSError, match="injected malformed status") as caught:
        supervisor.run_supervised(
            [sys.executable, "-c", program, str(ready), str(cleaned)],
            timeout=10,
            capture_output=True,
        )
    assert caught.value is primary
    assert cleaned.exists()
    assert "injected malformed status" not in " ".join(primary.__notes__)
    assert "worker diagnostic" in " ".join(primary.__notes__)


def test_unconfirmed_stop_still_resumes_and_reaps(supervisor, tmp_path, monkeypatch):
    marker, guard = tmp_path / "ready", tmp_path / ".guard"
    program = "import sys,time; from pathlib import Path; from filelock import FileLock; lock=FileLock(sys.argv[1]); lock.acquire(); Path(sys.argv[2]).touch(); time.sleep(30)"
    with raw_owner(
        supervisor, [sys.executable, "-c", program, str(guard), str(marker)], timeout=15
    ) as (process, control, channel):
        channel.start(control)
        control.send(b"S")
        wait_for(marker.exists)

        def unconfirmed(owned, _deadline):
            os.kill(owned.pid, signal.SIGSTOP)
            raise OSError("could not verify stopped shadow supervisor")

        monkeypatch.setattr(supervisor, "stopped_supervisor", unconfirmed)
        try:
            with pytest.raises(
                OSError, match="could not verify stopped shadow supervisor"
            ):
                supervisor.emergency_stop(process, channel)
            assert process.returncode is not None
            assert unlocked(guard)
        finally:
            channel.stop()


def test_broken_control_cannot_skip_resume(supervisor, monkeypatch):
    calls = []

    def broken():
        raise OSError("control write failed")

    channel = supervisor.StatusChannel(-1, control=SimpleNamespace(cancel=broken))
    monkeypatch.setattr(
        supervisor.os, "kill", lambda pid, sig: calls.append((pid, sig))
    )
    with pytest.raises(OSError, match="control write failed"):
        supervisor.resume_supervisor(SimpleNamespace(pid=123456), channel)
    assert calls == [(123456, signal.SIGCONT)]


@pytest.mark.parametrize("status", ["missing", "cancelled", "runtime_failure"])
def test_every_result_error_retains_captured_stderr(supervisor, tmp_path, status):
    path = tmp_path / "status"
    path.write_bytes(
        b""
        if status == "missing"
        else supervisor.encode_status(
            status,
            {
                "detail": "primary failure",
                "notes": ["nested cleanup note"],
                "cleanup_complete": True,
            },
        )
    )
    descriptor = os.open(path, os.O_RDONLY)
    channel = supervisor.StatusChannel(descriptor)
    try:
        with pytest.raises(OSError) as caught:
            supervisor.result([], 1, channel, "", "helper traceback", supervisor_code=1)
        assert "helper traceback" in " ".join(caught.value.__notes__)
        if status != "missing":
            assert "nested cleanup note" in caught.value.__notes__
    finally:
        os.close(descriptor)


def test_runtime_failure_notes_are_not_duplicated(supervisor):
    with pytest.raises(OSError, match="No such file") as caught:
        supervisor.run_supervised(
            ["/missing-pr107-launcher"], timeout=5, capture_output=True
        )
    notes = " ".join(getattr(caught.value, "__notes__", ()))
    assert "No such file" not in notes
    assert "shadow supervisor failure" not in notes


@pytest.mark.parametrize("bootstrap", [False, True])
def test_missing_mode_descriptors_are_argument_errors(supervisor, bootstrap):
    args = ["--bootstrap"] if bootstrap else []
    result = subprocess.run(
        [
            sys.executable,
            supervisor.__file__,
            *args,
            "--deadline",
            str(time.monotonic() + 1),
            "--",
            "/usr/bin/true",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "requires --" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("policy", ["operator", "sealed"])
def test_required_file_under_regular_ancestor_is_a_refusal(tmp_path, policy):
    path = tmp_path / "notes.txt"
    path.write_text("preserve")
    with pytest.raises(EvidenceError, match="missing required enrollment artifact"):
        required_artifact(path / "protocol.json", policy=policy)


@pytest.mark.parametrize("primary", ["first\nsecond", "first\r\nsecond"])
def test_duplicate_refusal_requires_complete_normalized_lines(primary):
    wrapper = load_script(SCRIPT.with_name("hourly-predict-publish.py"))
    output = io.StringIO()
    lines = [
        "unrelated shadow refused: first",
        "shadow refused: first",
        "second",
        "extra context",
    ]
    with contextlib.redirect_stdout(output):
        wrapper.preflight_secondary(lines, primary)
    assert (
        output.getvalue()
        == "shadow preflight stderr: unrelated shadow refused: first\nshadow preflight stderr: extra context\n"
    )


def test_force_denial_remains_failure_with_injected_clock(supervisor, monkeypatch):
    elapsed = [0.0]
    attempts = []

    def denied(*_):
        attempts.append(True)
        raise PermissionError("group termination denied")

    def sleep(seconds):
        elapsed[0] += seconds

    monkeypatch.setattr(supervisor, "signal_group", denied)
    clock = supervisor.Clock(lambda: elapsed[0], sleep)
    with pytest.raises(PermissionError, match="group termination denied"):
        supervisor.force_group(123456, 0.1, clock=clock)
    assert len(attempts) == 3


def test_control_close_waits_for_sender_and_never_reuses_descriptor(supervisor):
    read, write = os.pipe()
    channel = supervisor.ControlChannel(write)
    entered, release = threading.Event(), threading.Event()

    def sender():
        with channel.lock:
            entered.set()
            assert release.wait(3)
            channel.send(b"C")

    thread = threading.Thread(target=sender)
    thread.start()
    try:
        assert entered.wait(1)
        with pytest.raises(OSError, match="descriptor preserved"):
            channel.close()
        os.fstat(write)
        release.set()
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert os.read(read, 1) == b"C"
        channel.close()
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(write)
    finally:
        release.set()
        thread.join(timeout=3)
        if not channel.closed:
            channel.close()
        os.close(read)
