"""Startup provenance and bounded recovery through real helper and mocked group faults."""

import errno
import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_shadow_supervision import supervisor as supervisor
from test_shadow_supervision import wait_for

pytestmark = pytest.mark.skipif(os.name != "posix", reason="native POSIX ownership")


def test_bootstrap_waits_for_fork_authorization(supervisor, tmp_path, monkeypatch):
    marker = tmp_path / "not-admitted"
    worker = supervisor.start_worker(
        [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
        time.monotonic() + 5,
    )
    try:

        def ready():
            worker.collect()
            return worker.events.bootstrap_ready

        wait_for(ready)
        assert not worker.events.ready
        assert not worker.fork_authorized
        assert not marker.exists()
        monkeypatch.setattr(
            supervisor.GroupShutdown,
            "run",
            lambda *_: pytest.fail("no worker group was authorized"),
        )
        assert worker.stop() is None
        assert worker.cleaned
        assert worker.collected
        with pytest.raises(ChildProcessError):
            os.waitpid(worker.process.pid, os.WNOHANG)
    finally:
        worker.close_admission()
        os.close(worker.errors)
        if worker.process.returncode is None:
            worker.process.kill()
            worker.process.wait(timeout=3)


def test_dead_unstarted_helper_is_collected_without_group_cleanup(
    supervisor, monkeypatch
):
    process = subprocess.Popen([sys.executable, "-c", "pass"], process_group=0)
    read, write = os.pipe()
    worker = supervisor.Worker(process, write, read)
    try:
        wait_for(lambda: supervisor.observe_exit(process) is not None)
        with pytest.raises(
            OSError,
            match="guardian failed before fork authorization; native command never started",
        ):
            worker.check_error()
        monkeypatch.setattr(
            supervisor.GroupShutdown,
            "run",
            lambda *_: pytest.fail("zombie-only group cleanup"),
        )
        assert worker.stop() is None
        assert worker.cleaned
        assert worker.collected
        with pytest.raises(ChildProcessError):
            os.waitpid(process.pid, os.WNOHANG)
    finally:
        worker.close_admission()
        os.close(read)
        if process.returncode is None:
            process.kill()
            process.wait(timeout=3)


def test_timeout_before_bootstrap_ready_preserves_public_timeout(
    supervisor, monkeypatch
):
    control_read, control_write = os.pipe()
    status_read, status_write = os.pipe()
    admission_read, admission_write = os.pipe()
    error_read, error_write = os.pipe()
    os.close(admission_read)
    os.close(error_write)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], process_group=0
    )
    worker = supervisor.Worker(process, admission_write, error_read)
    monkeypatch.setattr(supervisor, "start_worker", lambda *_, **__: worker)
    control = supervisor.ControlChannel(control_write)
    channel = supervisor.StatusChannel(status_read)
    channel.start(control)
    control.send(b"S")
    try:
        assert (
            supervisor.supervise(
                ["never-executed"], control_read, status_write, time.monotonic() + 0.1
            )
            == 0
        )
        channel.finish()
        assert channel.messages[-1]["status"] == "timed_out"
        assert channel.messages[-1]["cleanup_complete"]
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            supervisor.result(["never-executed"], 0.1, channel, None, None)
        assert caught.value.timeout == 0.1
        assert process.returncode is not None
    finally:
        channel.stop()
        control.close()
        os.close(status_read)
        if process.returncode is None:
            process.kill()
            process.wait(timeout=3)


def test_failed_stop_reports_updated_cleanup_and_collects(supervisor, monkeypatch):
    process = SimpleNamespace(wait=Mock())
    worker = SimpleNamespace(
        cleaned=False, stop_error=None, process=process, collected=False
    )

    def stop():
        worker.cleaned = True
        raise OSError("native outcome unavailable after cleanup")

    worker.stop = stop
    sent = []
    monkeypatch.setattr(
        supervisor,
        "send_status",
        lambda _fd, status, **fields: sent.append(dict(status=status, **fields)),
    )
    monkeypatch.setattr(supervisor, "await_release", lambda _: None)
    primary = OSError("original failure")
    supervisor.failed_worker(worker, -1, -1, primary)
    assert sent[0]["cleanup_complete"] is True
    assert sent[0]["detail"] == "original failure"
    assert sent[0]["notes"] == [
        "worker cleanup failed: native outcome unavailable after cleanup"
    ]
    process.wait.assert_called_once()


def test_control_pipe_failure_is_secondary_to_owner_loss(supervisor, monkeypatch):
    read, write = os.pipe()
    os.close(write)
    worker = supervisor.Worker(
        SimpleNamespace(pid=123456), -1, read, fork_authorized=True
    )
    worker.events.ready = True
    worker.collect = lambda: None
    worker.command = Mock(side_effect=BrokenPipeError(errno.EPIPE, "Broken pipe"))
    monkeypatch.setattr(supervisor, "observe_exit", lambda _: 127)
    monkeypatch.setattr(supervisor, "group_exists", lambda _: False)
    try:
        with pytest.raises(
            OSError,
            match="guardian exited without a valid cleanup release; worker outcome unknown",
        ) as caught:
            worker.stop()
        assert not isinstance(caught.value, BrokenPipeError)
        assert any(
            "shutdown transport failed" in note and "Broken pipe" in note
            for note in caught.value.__notes__
        )
        assert worker.cleaned
    finally:
        os.close(read)


def test_signal_delivery_is_not_cleanup_confirmation(supervisor, monkeypatch):
    elapsed, signals = [0.0], []
    clock = supervisor.Clock(
        lambda: elapsed[0], lambda n: elapsed.__setitem__(0, elapsed[0] + n)
    )
    monkeypatch.setattr(supervisor, "group_exists", lambda _: True)
    monkeypatch.setattr(
        supervisor, "signal_group", lambda _pid, signum: signals.append(signum)
    )
    shutdown = supervisor.GroupShutdown(123456, supervisor.ShutdownBudget(0), clock)
    with pytest.raises(OSError, match="termination was not certified"):
        shutdown.run(lambda: None)
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert not shutdown.complete
    assert 4 <= elapsed[0] < 4.1


def test_expired_normal_budget_has_bounded_force_rescue(supervisor, monkeypatch):
    elapsed, signals = [20.0], []
    clock = supervisor.Clock(
        lambda: elapsed[0], lambda n: elapsed.__setitem__(0, elapsed[0] + n)
    )
    guardian = supervisor.Guardian(
        SimpleNamespace(collect=lambda: None),
        -1,
        -1,
        reserved=True,
        budget=supervisor.ShutdownBudget(0),
        clock=clock,
    )
    monkeypatch.setattr(supervisor, "group_exists", lambda _: True)

    def denied(_pid, signum):
        signals.append(signum)
        raise PermissionError("fallback kernel denial")

    monkeypatch.setattr(supervisor, "signal_group", denied)
    with pytest.raises(PermissionError, match="fallback kernel denial"):
        guardian.fallback()
    assert signals
    assert set(signals) == {signal.SIGKILL}
    assert 22 <= elapsed[0] < 22.1
    assert guardian.rescue.budget.deadline == 23
    first = guardian.rescue
    with pytest.raises(PermissionError, match="fallback kernel denial"):
        guardian.fallback()
    assert guardian.rescue is first


def test_clock_default_resolves_at_construction(supervisor, monkeypatch):
    clock = supervisor.Clock(lambda: 123, lambda _: None)
    monkeypatch.setattr(supervisor, "CLOCK", clock)
    shutdown = supervisor.GroupShutdown(123456, supervisor.ShutdownBudget(123))
    assert shutdown.clock is clock
    assert supervisor.Worker(SimpleNamespace(), -1, -1).clock is clock


def test_force_rescue_never_sends_term_after_kill(supervisor, monkeypatch):
    elapsed, signals = [0.0], []
    clock = supervisor.Clock(
        lambda: elapsed[0], lambda n: elapsed.__setitem__(0, elapsed[0] + n)
    )
    probes = iter([True, True, False])
    monkeypatch.setattr(supervisor, "group_exists", lambda _: next(probes))
    monkeypatch.setattr(
        supervisor, "signal_group", lambda _pid, sig: signals.append(sig)
    )
    supervisor.force_group(123456, 2, clock=clock)
    assert signals == [signal.SIGKILL]


def test_authorization_keeps_coalesced_shutdown_frame(supervisor):
    read, write = os.pipe()
    events_read, events_write = os.pipe()
    stream = supervisor.FrameStream("shadow_guardian_control")
    try:
        supervisor.write_frame(write, stream.kind, "reserve")
        supervisor.write_frame(write, stream.kind, "shutdown", started=123)
        assert supervisor.authorize_fork(
            read, events_write, time.monotonic() + 3, supervisor.CLOCK, stream
        )
        assert [value["status"] for value in stream.pending] == ["shutdown"]
        assert stream.pending[0]["started"] == 123
    finally:
        for fd in (read, write, events_read, events_write):
            os.close(fd)


def test_malformed_frame_cannot_authorize_fork(supervisor):
    read, write = os.pipe()
    events_read, events_write = os.pipe()
    stream = supervisor.FrameStream("shadow_guardian_control")
    try:
        os.write(write, b"malformed\n")
        supervisor.write_frame(write, stream.kind, "reserve")
        with pytest.raises(OSError, match="malformed shadow_guardian_control frame"):
            supervisor.authorize_fork(
                read, events_write, time.monotonic() + 3, supervisor.CLOCK, stream
            )
    finally:
        for fd in (read, write, events_read, events_write):
            os.close(fd)


def test_timeout_classification_retains_cleanup_uncertainty(supervisor):
    channel = supervisor.StatusChannel(
        -1,
        messages=[
            dict(
                status="timed_out",
                cleanup_complete=False,
                notes=["owner collection uncertain"],
            )
        ],
    )
    failure = OSError("missing cleanup acknowledgment")
    selected = supervisor.selected_stop_error(["native"], 0.2, channel, failure)
    assert isinstance(selected, subprocess.TimeoutExpired)
    assert selected.timeout == 0.2
    assert "owner collection uncertain" in selected.__notes__
    assert any("missing cleanup acknowledgment" in note for note in selected.__notes__)


def test_excess_diagnostics_do_not_discard_launcher_exit(supervisor):
    def frame(status, **fields):
        return (
            json.dumps(
                dict(
                    kind="shadow_guardian",
                    schema_version=supervisor.PROTOCOL_VERSION,
                    status=status,
                    **fields,
                )
            ).encode()
            + b"\n"
        )

    stream = supervisor.FrameStream("shadow_guardian")
    frames = [
        frame("ready"),
        frame("error", detail="x" * 47000, errno=None),
        frame("error", detail="y" * 47000, errno=None),
        frame("launcher_exit", returncode=143),
    ]
    accepted = []
    for raw in frames:
        accepted.extend(stream.feed(raw))
    assert stream.error is not None
    assert accepted[-1]["status"] == "launcher_exit"
    assert accepted[-1]["returncode"] == 143
    assert stream.payload == b""


def test_oversized_partial_frame_recovers_at_next_frame(supervisor):
    stream = supervisor.FrameStream("shadow_guardian")
    stream.feed(b"x" * (supervisor.STATUS_LIMIT + 1))
    assert stream.payload == b""
    assert stream.discarding
    values = stream.feed(
        b"discard\n"
        + supervisor.encode_status(
            "launcher_exit", dict(returncode=0), kind="shadow_guardian"
        )
    )
    assert [value["status"] for value in values] == ["launcher_exit"]


def test_guardian_errno_and_paths_are_reconstructed_once(supervisor):
    read, write = os.pipe()
    try:
        supervisor.guardian_error(
            write,
            FileNotFoundError(
                errno.ENOENT, "No such file or directory", "/missing/worker"
            ),
        )
        events = supervisor.GuardianEvents()
        events.collect(read)
        assert events.error.errno == errno.ENOENT
        assert events.error.filename == "/missing/worker"
        assert (
            str(events.error)
            == "[Errno 2] No such file or directory: '/missing/worker'"
        )
    finally:
        os.close(read)
        os.close(write)


def test_idle_frame_reads_configure_descriptor_once(supervisor, monkeypatch):
    read, write = os.pipe()
    original = os.set_blocking
    calls = []

    def configured(fd, blocking):
        calls.append((fd, blocking))
        original(fd, blocking)

    monkeypatch.setattr(supervisor.os, "set_blocking", configured)
    try:
        stream = supervisor.FrameStream("shadow_guardian")
        for _ in range(4):
            assert stream.read(read) == []
        assert calls == [(read, False)]
    finally:
        os.close(read)
        os.close(write)


def test_admission_reuses_selector_for_final_cancel_check(supervisor, monkeypatch):
    read, write = os.pipe()
    status_read, status_write = os.pipe()
    factory, selectors, admitted = supervisor.selectors.DefaultSelector, [], []

    def counted():
        selectors.append(True)
        return factory()

    worker = SimpleNamespace(
        ready=lambda *_: "ready",
        process=SimpleNamespace(pid=123456),
        clock=supervisor.CLOCK,
        admit=lambda: admitted.append(True),
    )
    monkeypatch.setattr(supervisor.selectors, "DefaultSelector", counted)
    monkeypatch.setattr(supervisor, "monitor", lambda *_, **__: "completed")
    try:
        os.write(write, b"A")
        assert (
            supervisor.worker_outcome(
                worker,
                read,
                status_write,
                time.monotonic() + 3,
                supervisor.Cancellation(),
            )
            == "completed"
        )
        assert admitted == [True]
        assert len(selectors) == 1
    finally:
        for fd in (read, write, status_read, status_write):
            os.close(fd)


def test_failed_exec_closed_error_reader_still_exits_127(supervisor):
    program = """
import importlib.util, os, sys, time
spec=importlib.util.spec_from_file_location('native_probe',sys.argv[1])
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
r,w=os.pipe();er,ew=os.pipe();os.write(w,b'G');os.close(w);os.close(er)
os._exit(m.execute_admitted(['/definitely-missing-shadow-executable'],r,ew,time.monotonic()+2))
"""
    result = subprocess.run(
        [sys.executable, "-c", program, supervisor.__file__],
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 127
    assert result.stderr == b""
