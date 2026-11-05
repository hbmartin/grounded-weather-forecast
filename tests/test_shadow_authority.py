"""One shutdown authority, framed launcher evidence and bounded ownership."""

import errno
import os
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from shadow_supervision_helpers import raw_owner, statuses
from test_shadow_supervision import (
    supervisor as supervisor,  # noqa: PLC0414 -- pytest fixture
)
from test_shadow_supervision import unlocked, wait_for

pytestmark = pytest.mark.skipif(os.name != "posix", reason="native ownership")


@pytest.mark.parametrize("cause", ["cancelled", "timed_out", "group_term"])
def test_cleanup_exit_zero_cannot_replace_stop_reason(supervisor, tmp_path, cause):
    ready = tmp_path / "ready"
    program = """
import signal,sys,time
from pathlib import Path
def stop(*_):
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 time.sleep(.1)
 raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
Path(sys.argv[1]).touch()
time.sleep(30)
"""
    timeout = 1 if cause == "timed_out" else 10
    with raw_owner(
        supervisor, [sys.executable, "-c", program, str(ready)], timeout=timeout
    ) as (process, control, channel):
        control.send(b"S")
        wait_for(lambda: statuses(channel) == ["worker_ready"])
        control.send(b"A")
        wait_for(ready.exists)
        if cause == "cancelled":
            control.cancel()
        elif cause == "group_term":
            os.killpg(process.pid, signal.SIGTERM)
        wait_for(lambda: len(statuses(channel)) == 2, timeout=7)
        terminal = channel.messages[-1]
        assert terminal["status"] == ("cancelled" if cause == "group_term" else cause)
        assert terminal["returncode"] == 0
        assert terminal["cleanup_complete"]
        control.send(b"R")
        assert process.wait(timeout=3) == 0


def test_group_term_retains_guardian_and_cancellation_exception(supervisor, tmp_path):
    ready = tmp_path / "ready"
    with raw_owner(
        supervisor,
        [
            sys.executable,
            "-c",
            "import sys,time; from pathlib import Path; Path(sys.argv[1]).touch(); time.sleep(30)",
            str(ready),
        ],
        timeout=10,
    ) as (process, control, channel):
        control.send(b"S")
        wait_for(lambda: statuses(channel) == ["worker_ready"])
        guardian = channel.owned_pid
        control.send(b"A")
        wait_for(ready.exists)
        os.killpg(process.pid, signal.SIGTERM)
        wait_for(lambda: len(statuses(channel)) == 2)
        assert channel.messages[-1]["status"] == "cancelled"
        assert channel.messages[-1]["returncode"] == -signal.SIGTERM
        with pytest.raises(
            OSError, match=r"^shadow supervisor cancelled: worker cancelled$"
        ):
            supervisor.result([], 10, channel, None, None)
        assert guardian is not None
        assert (
            os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            is None
        )
        control.send(b"R")
        assert process.wait(timeout=3) == 0


def test_guardian_death_is_unknown_outcome_and_collects_reserved_identity(
    supervisor, tmp_path, monkeypatch
):
    ready, guard = tmp_path / "ready", tmp_path / ".guard"
    program = "import sys,time; from pathlib import Path; from filelock import FileLock; lock=FileLock(sys.argv[1]); lock.acquire(); Path(sys.argv[2]).touch(); time.sleep(30)"
    original = supervisor.StatusChannel.admit_identity
    threads = []

    def admit(channel, value):
        original(channel, value)

        def kill_owned():
            wait_for(ready.exists)
            os.kill(value["worker_pid"], signal.SIGKILL)

        thread = threading.Thread(target=kill_owned)
        thread.start()
        threads.append(thread)

    monkeypatch.setattr(supervisor.StatusChannel, "admit_identity", admit)
    with pytest.raises(OSError, match="worker outcome unknown"):
        supervisor.run_supervised(
            [sys.executable, "-c", program, str(guard), str(ready)],
            timeout=10,
            capture_output=True,
        )
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()
    wait_for(lambda: unlocked(guard))


def test_launcher_error_always_exits_without_guardian_signal(supervisor, monkeypatch):
    errors, exits = [], []

    def close_failed(_fd):
        raise OSError(errno.EBADF, "launcher close failed")

    def exit_child(code):
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(supervisor.os, "close", close_failed)
    monkeypatch.setattr(
        supervisor, "guardian_error", lambda _fd, error: errors.append(error)
    )
    monkeypatch.setattr(supervisor.os, "_exit", exit_child)
    monkeypatch.setattr(
        supervisor.os,
        "kill",
        lambda *_: pytest.fail("launcher entered guardian signals"),
    )
    with pytest.raises(SystemExit) as caught:
        supervisor.launcher_main(["unused"], 10, 11, 12, 13, 100)
    assert caught.value.code == 127
    assert exits == [127]
    assert errors[0].errno == errno.EBADF


def test_reaped_launcher_cannot_be_signalled_after_reporting_failure(
    supervisor, monkeypatch
):
    child = supervisor.OwnedLauncher(123456, -1)
    monkeypatch.setattr(supervisor.os, "waitpid", lambda *_: (123456, 0))

    def report_failed(*_args, **_kwargs):
        raise OSError(errno.EIO, "event report failed after reap")

    monkeypatch.setattr(supervisor, "guardian_event", report_failed)
    with pytest.raises(OSError, match="event report failed after reap"):
        child.collect()
    assert child.pid is None
    assert child.returncode == 0
    monkeypatch.setattr(
        supervisor.os, "kill", lambda *_: pytest.fail("historical PID signal")
    )
    child.stop_blocked(100)


@pytest.mark.parametrize("pid", [None, 0, 1, -1])
def test_owned_launcher_rejects_unsafe_identity(supervisor, pid):
    with pytest.raises(OSError, match="invalid owned launcher identity"):
        supervisor.OwnedLauncher(pid, -1)


def test_guardian_frames_survive_split_readiness_and_exit(supervisor):
    events = supervisor.GuardianEvents()
    ready = supervisor.encode_status("ready", {}, kind="shadow_guardian")
    exited = supervisor.encode_status(
        "launcher_exit", {"returncode": 143}, kind="shadow_guardian"
    )
    read, write = os.pipe()
    try:
        os.write(write, ready[:7])
        events.collect(read)
        assert not events.ready
        os.write(write, ready[7:] + exited[:11])
        events.collect(read)
        assert events.ready
        assert events.returncode is None
        os.write(write, exited[11:])
        os.close(write)
        write = -1
        events.collect(read)
        assert events.returncode == 143
        assert events.error is None
    finally:
        os.close(read)
        if write >= 0:
            os.close(write)


def test_malformed_guardian_frame_retains_valid_cleanup_evidence(supervisor):
    stream = supervisor.FrameStream("shadow_guardian")
    messages = stream.feed(
        b"garbage\n" + supervisor.encode_status("ready", {}, kind="shadow_guardian")
    )
    assert str(stream.error).startswith("malformed shadow_guardian frame")
    assert [message["status"] for message in messages] == ["ready"]


def test_startup_errno_reaches_supervisor_before_ready(supervisor, monkeypatch):
    read, write = os.pipe()

    def exhausted():
        raise OSError(errno.EMFILE, "Too many open files: guardian admission")

    try:
        monkeypatch.setattr(supervisor.os, "pipe", exhausted)
        assert supervisor.bootstrap(["unused"], -1, write, 100) == 127
        events = supervisor.GuardianEvents()
        events.collect(read)
        assert events.error.errno == errno.EMFILE
        assert "guardian admission" in str(events.error)
        assert not events.ready
    finally:
        os.close(read)
        os.close(write)


def test_shutdown_denial_preserves_collection_reserve(supervisor, monkeypatch):
    elapsed, calls = [0.0], []
    clock = supervisor.Clock(
        lambda: elapsed[0], lambda delay: elapsed.__setitem__(0, elapsed[0] + delay)
    )

    def denied(_pid, signum):
        calls.append(signum)
        raise PermissionError("persistent group denial")

    monkeypatch.setattr(supervisor, "group_exists", lambda _: True)
    monkeypatch.setattr(supervisor, "signal_group", denied)
    budget = supervisor.ShutdownBudget(0)
    shutdown = supervisor.GroupShutdown(123456, budget, clock)
    with pytest.raises(PermissionError, match="persistent group denial"):
        shutdown.run(lambda: None)
    assert 4 <= elapsed[0] < 4.1
    assert budget.deadline == 5
    assert signal.SIGKILL in calls


def test_failed_stop_reuses_primary_without_second_attempt(supervisor, monkeypatch):
    worker = supervisor.Worker(SimpleNamespace(), -1, -1)
    failure = OSError("first shutdown budget exhausted")
    calls = []

    def failed():
        calls.append(True)
        raise failure

    monkeypatch.setattr(worker, "stop_owned", failed)
    for _ in range(2):
        with pytest.raises(OSError, match="first shutdown budget exhausted") as caught:
            worker.stop()
        assert caught.value is failure
    assert calls == [True]


def test_terminal_runtime_detail_is_distinct_from_primary_protocol_error(
    supervisor, monkeypatch
):
    primary = OSError("malformed status frame")
    channel = supervisor.StatusChannel(
        -1,
        error=primary,
        released=True,
        messages=[
            {
                "status": "runtime_failure",
                "detail": "guardian collection EIO",
                "notes": ["path retained"],
                "cleanup_complete": True,
            }
        ],
    )
    monkeypatch.setattr(
        supervisor, "communicate_until", lambda *_args, **_kwargs: (None, None)
    )
    monkeypatch.setattr(channel, "finish", lambda **_: None)
    supervisor.finish_cancellation(SimpleNamespace(), channel, primary)
    supervisor.finish_cancellation(SimpleNamespace(), channel, primary)
    notes = primary.__notes__
    assert notes == [
        "shadow supervisor failure: shadow supervisor runtime_failure: guardian collection EIO",
        "shadow supervisor failure: path retained",
    ]


def test_partial_output_is_not_decoded_on_each_poll(supervisor, monkeypatch):
    payload = b"x" * 100000
    calls, decodes = [], []

    def communicate(**_kwargs):
        calls.append(True)
        if len(calls) < 4:
            raise subprocess.TimeoutExpired([], 0.05, output=payload, stderr=payload)
        return payload.decode(), payload.decode()

    monkeypatch.setattr(
        supervisor,
        "decoded_output",
        lambda value: decodes.append(value) or value.decode(),
    )
    channel = supervisor.StatusChannel(-1)
    monkeypatch.setattr(channel, "collect", lambda **_: None)
    result = supervisor.communicate_until(
        SimpleNamespace(communicate=communicate), channel, time.monotonic() + 1
    )
    assert result == (payload.decode(), payload.decode())
    assert len(calls) == 4
    assert decodes == []
    assert not hasattr(channel, "stdout")


def test_readiness_reuses_one_selector(supervisor, monkeypatch):
    read, write = os.pipe()
    errors, event_write = os.pipe()
    worker = supervisor.Worker(SimpleNamespace(), -1, errors)
    selectors, polls = [], []
    original = supervisor.selectors.DefaultSelector

    def selector():
        selectors.append(True)
        return original()

    def collect():
        polls.append(True)
        worker.events.ready = len(polls) == 4

    monkeypatch.setattr(supervisor.selectors, "DefaultSelector", selector)
    monkeypatch.setattr(worker, "collect", collect)
    monkeypatch.setattr(worker, "check_error", lambda: None)
    try:
        assert (
            worker.ready(read, time.monotonic() + 2, supervisor.Cancellation())
            == "ready"
        )
        assert len(polls) == 4
        assert selectors == [True]
    finally:
        for descriptor in (read, write, errors, event_write):
            os.close(descriptor)


def test_supervisor_death_activates_guardian_only_fallback(supervisor, tmp_path):
    ready, guard = tmp_path / "ready", tmp_path / ".guard"
    command = [
        sys.executable,
        "-c",
        "import sys,time; from pathlib import Path; from filelock import FileLock; lock=FileLock(sys.argv[1]); lock.acquire(); Path(sys.argv[2]).touch(); time.sleep(30)",
        str(guard),
        str(ready),
    ]
    with raw_owner(supervisor, command, timeout=10) as (process, control, channel):
        channel.start(control)
        control.send(b"S")
        wait_for(ready.exists)
        os.kill(process.pid, signal.SIGKILL)
        assert process.wait(timeout=3) == -signal.SIGKILL
        wait_for(lambda: unlocked(guard), timeout=7)
        channel.stop()


def test_shutdown_announcement_does_not_invoke_guardian_fallback(
    supervisor, monkeypatch
):
    launcher = supervisor.OwnedLauncher(123456, -1)
    launcher.pid, launcher.returncode = None, 0
    read, write = os.pipe()
    guardian = supervisor.Guardian(launcher, read, -1, reserved=True)
    monkeypatch.setattr(
        guardian, "fallback", lambda: pytest.fail("ordinary shutdown caused fallback")
    )
    try:
        supervisor.write_frame(
            write, "shadow_guardian_control", "shutdown", started=time.monotonic()
        )
        supervisor.write_frame(
            write, "shadow_guardian_control", "release", group_cleaned=True
        )
        assert guardian.run() is None
        assert guardian.budget is not None
    finally:
        os.close(read)
        os.close(write)


def test_guardian_fallback_cannot_renew_announced_budget(supervisor, monkeypatch):
    launcher = supervisor.OwnedLauncher(123456, -1)
    launcher.pid, launcher.returncode = None, 0
    budget = supervisor.ShutdownBudget(0)
    clock = supervisor.Clock(lambda: 10, lambda _: pytest.fail("expired budget slept"))
    shutdown = supervisor.GroupShutdown(123456, budget, clock)
    guardian = supervisor.Guardian(
        launcher, -1, -1, reserved=True, budget=budget, shutdown=shutdown
    )
    monkeypatch.setattr(supervisor, "group_exists", lambda _: True)
    monkeypatch.setattr(
        supervisor, "signal_group", lambda *_: pytest.fail("expired budget signalled")
    )
    with pytest.raises(OSError, match="collection reserve"):
        guardian.fallback()
    assert guardian.budget is budget
    assert guardian.shutdown is shutdown


def test_emergency_denial_reserves_time_for_resume_and_reap(supervisor, monkeypatch):
    elapsed, calls = [0.0], []
    clock = supervisor.Clock(
        lambda: elapsed[0], lambda delay: elapsed.__setitem__(0, elapsed[0] + delay)
    )
    original = supervisor.force_group

    def denied(*_):
        raise PermissionError("rescue signal denied")

    monkeypatch.setattr(supervisor, "CLOCK", clock)
    monkeypatch.setattr(supervisor, "stopped_supervisor", lambda *_: True)
    monkeypatch.setattr(supervisor, "signal_group", denied)
    monkeypatch.setattr(
        supervisor,
        "force_group",
        lambda pid, deadline: original(pid, deadline, clock=clock),
    )
    monkeypatch.setattr(
        supervisor, "resume_supervisor", lambda *_: calls.append("resumed")
    )
    monkeypatch.setattr(
        supervisor,
        "reap_supervisor",
        lambda _process, deadline: calls.append(deadline - elapsed[0]),
    )
    process = SimpleNamespace(returncode=None)
    channel = supervisor.StatusChannel(-1, owned_pid=123456)
    with pytest.raises(PermissionError, match="rescue signal denied"):
        supervisor.emergency_stop(process, channel)
    assert calls[0] == "resumed"
    assert 0.9 <= calls[1] <= 1


def test_secondary_ownership_probe_cannot_replace_stop_failure(supervisor, monkeypatch):
    primary = OSError("original shutdown failure")
    worker = supervisor.Worker(
        SimpleNamespace(),
        -1,
        -1,
        shutdown=supervisor.GroupShutdown(
            123456, supervisor.ShutdownBudget(0), complete=True
        ),
    )

    def failed_stop():
        raise primary

    def failed_probe(_):
        raise OSError("secondary ownership probe failed")

    monkeypatch.setattr(worker, "stop_owned", failed_stop)
    monkeypatch.setattr(supervisor, "observe_exit", failed_probe)
    with pytest.raises(OSError, match="original shutdown failure") as caught:
        worker.stop()
    assert caught.value is primary
    assert primary.__notes__ == [
        "guardian collection check failed: secondary ownership probe failed"
    ]


def test_event_read_failure_still_delivers_graceful_signal(supervisor, monkeypatch):
    worker = supervisor.Worker(SimpleNamespace(pid=123456), -1, -1)
    worker.events.ready, worker.events.returncode = True, 0
    failure, calls = OSError("guardian event read failed"), []
    probes = iter([True, False])

    def failed_read():
        raise failure

    monkeypatch.setattr(worker, "collect", failed_read)
    monkeypatch.setattr(worker, "command", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(supervisor, "group_exists", lambda _: next(probes))
    monkeypatch.setattr(
        supervisor, "signal_group", lambda pid, sig: calls.append((pid, sig))
    )
    monkeypatch.setattr(supervisor, "wait_exit", lambda *_: 0)
    assert worker.stop() == 0
    assert calls == [(123456, signal.SIGTERM)]
    assert worker.cleaned
    assert worker.events.error is failure
    with pytest.raises(OSError, match="guardian event read failed"):
        worker.exec_error()
