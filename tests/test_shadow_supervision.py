"""Synthetic process ownership through timeout, cancellation and wrapper death."""

import importlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from filelock import FileLock, Timeout
from test_launchd import SCRIPT, load_script
from test_shadow_native_guards import window_protocol

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX supervision")


@pytest.fixture
def supervisor(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    return importlib.import_module("shadow_supervisor")


def wait_for(predicate, *, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail("synthetic process did not reach expected state")


def unlocked(path):
    try:
        with FileLock(path, timeout=0):
            return True
    except (Timeout,):  # noqa: B013 -- project exception convention
        return False


@pytest.mark.parametrize("code", [0, 2, -signal.SIGTERM])
def test_worker_exit_and_streams_are_preserved(supervisor, code):
    program = "import os,signal,sys; print('worker stdout',flush=True); print('worker stderr',file=sys.stderr,flush=True); "
    program += f"os.kill(os.getpid(), {-code})" if code < 0 else f"sys.exit({code})"
    command = [sys.executable, "-c", program]
    result = supervisor.run_supervised(command, timeout=5, capture_output=True)
    assert result.args == command
    assert result.returncode == code
    assert result.stdout == "worker stdout\n"
    assert result.stderr == "worker stderr\n"


@pytest.mark.parametrize("stop", [signal.SIGINT, signal.SIGTERM, signal.SIGKILL])
def test_wrapper_death_releases_worker_guard(supervisor, tmp_path, stop):
    marker = tmp_path / "worker.pid"
    guard = tmp_path / ".worker"
    worker = """
import os,sys,time
from pathlib import Path
from filelock import FileLock
with FileLock(sys.argv[1]):
    Path(sys.argv[2]).write_text(str(os.getpid()))
    time.sleep(30)
"""
    wrapper = """
import sys
sys.path.insert(0,sys.argv[1])
from launchd_paths import run_shadow_child
run_shadow_child([sys.executable,'-c',sys.argv[2],sys.argv[3],sys.argv[4]],timeout=30)
"""
    owner = subprocess.Popen(  # noqa: S603 -- synthetic worker and paths
        [
            sys.executable,
            "-c",
            wrapper,
            str(SCRIPT.parent),
            worker,
            str(guard),
            str(marker),
        ],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker_pid = None
    cleaned = False
    try:
        wait_for(marker.exists)
        worker_pid = int(marker.read_text())
        assert not unlocked(guard)
        os.killpg(owner.pid, stop)
        owner.wait(timeout=8)
        wait_for(lambda: unlocked(guard), timeout=7)
        assert owner.returncode != 0
        cleaned = True
    finally:
        if owner.poll() is None:
            os.killpg(owner.pid, signal.SIGKILL)
            owner.wait(timeout=5)
        if worker_pid is not None and not cleaned:
            supervisor.signal_group(worker_pid, signal.SIGKILL)


def test_parent_death_before_start_handshake_launches_no_worker(supervisor, tmp_path):
    control_read, control_write = os.pipe()
    status_read, status_write = os.pipe()
    marker = tmp_path / "never-started"
    process = subprocess.Popen(  # noqa: S603 -- this supervisor and synthetic marker
        [
            sys.executable,
            supervisor.__file__,
            "--control-fd",
            str(control_read),
            "--status-fd",
            str(status_write),
            "--deadline",
            str(time.monotonic() + 5),
            "--",
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
            str(marker),
        ],
        pass_fds=(control_read, status_write),
        start_new_session=True,
    )
    os.close(control_read)
    os.close(status_write)
    os.close(control_write)
    try:
        assert process.wait(timeout=5) == 0
        channel = supervisor.StatusChannel(status_read)
        channel.collect()
        assert [value["status"] for value in channel.messages] == ["cancelled"]
        assert not marker.exists()
    finally:
        os.close(status_read)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


@pytest.mark.parametrize("graceful", [True, False])
def test_timeout_unwinds_or_escalates_and_reaps(supervisor, tmp_path, graceful):
    marker = tmp_path / "started"
    guard = tmp_path / ".held"
    cleanup = tmp_path / "cleaned"
    program = """
import signal,sys,time
from pathlib import Path
from filelock import FileLock
def stop(signum,frame):
    raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,stop if sys.argv[4]=='True' else signal.SIG_IGN)
try:
    with FileLock(sys.argv[1]):
        Path(sys.argv[2]).write_text('ready')
        print('worker ready',flush=True)
        time.sleep(30)
finally:
    Path(sys.argv[3]).write_text('unwound')
"""
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        supervisor.run_supervised(
            [
                sys.executable,
                "-c",
                program,
                str(guard),
                str(marker),
                str(cleanup),
                str(graceful),
            ],
            timeout=2,
            capture_output=True,
        )
    elapsed = time.monotonic() - started
    assert marker.read_text() == "ready"
    assert b"worker ready" in caught.value.output
    assert elapsed < 8
    assert cleanup.exists() == graceful
    assert unlocked(guard)
    if not graceful:
        assert elapsed >= 2 + supervisor.TERM_GRACE


def test_runner_sigterm_cleans_staging(supervisor, tmp_path):
    output = tmp_path / "record"
    program = """
import sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from temperature_shadow import cli_signals, staged_directory
with cli_signals(), staged_directory(Path(sys.argv[2])) as folder:
    (folder/'partial').write_bytes(b'partial')
    print('staging ready',flush=True)
    time.sleep(30)
"""
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        supervisor.run_supervised(
            [
                sys.executable,
                "-c",
                program,
                str(SCRIPT.parents[2] / "scripts"),
                str(output),
            ],
            timeout=2,
            capture_output=True,
        )
    assert b"staging ready" in caught.value.output
    assert not output.exists()
    assert not list(tmp_path.glob("shadow-stage-*"))
    with FileLock(tmp_path / ".record.publishing", timeout=0):
        pass


def test_worker_start_failure_is_operational(supervisor):
    with pytest.raises(OSError, match=r"supervisor runtime_failure.*No such file"):
        supervisor.run_supervised(
            ["/missing-shadow-runtime"], timeout=2, capture_output=True
        )


def test_pipe_descriptors_close_when_supervisor_cannot_start(supervisor, monkeypatch):
    allocated = []
    pipe = os.pipe

    def created():
        pair = pipe()
        allocated.extend(pair)
        return pair

    def unavailable(*_args, **_kwargs):
        raise OSError("supervisor process start denied")

    monkeypatch.setattr(supervisor.os, "pipe", created)
    monkeypatch.setattr(supervisor.subprocess, "Popen", unavailable)
    with pytest.raises(OSError, match="supervisor process start denied"):
        supervisor.run_supervised(["synthetic"], timeout=2)
    for descriptor in allocated:
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(descriptor)


def test_supervisor_fault_stops_and_reaps_worker(supervisor, monkeypatch):
    control_read, control_write = os.pipe()
    status_read, status_write = os.pipe()
    owners = []
    popen = subprocess.Popen

    def created(*args, **kwargs):
        process = popen(*args, **kwargs)
        owners.append(process)
        return process

    def broken(*_args):
        raise OSError("injected monitor fault")

    monkeypatch.setattr(supervisor.subprocess, "Popen", created)
    monkeypatch.setattr(supervisor, "monitor", broken)
    channel = supervisor.StatusChannel(status_read)
    control = supervisor.ControlChannel(control_write)
    channel.start(control)
    control.send(b"S")
    try:
        assert (
            supervisor.supervise(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                control_read,
                status_write,
                time.monotonic() + 5,
            )
            == 1
        )
        channel.finish()
        with pytest.raises(OSError, match="runtime_failure: injected monitor fault"):
            supervisor.result([], 5, channel, "", "")
        assert owners[0].returncode is not None
        with pytest.raises(ChildProcessError):
            os.waitpid(owners[0].pid, os.WNOHANG)
    finally:
        channel.stop()
        control.close()
        os.close(status_read)


@pytest.mark.parametrize("persistent", [False, True])
def test_permission_probes_retry_but_persistent_denial_fails(
    supervisor, monkeypatch, persistent
):
    clock = [0.0]
    probes = []

    def now():
        clock[0] += 0.5
        return clock[0]

    def exists(_pid):
        probes.append(True)
        if len(probes) == 1 or persistent:
            raise PermissionError("group probe denied")
        return False

    def sent(_pid, signum):
        if persistent and signum == signal.SIGKILL:
            raise PermissionError("group termination denied")

    timing = supervisor.Clock(now, lambda _seconds: None)
    monkeypatch.setattr(
        supervisor,
        "observe_exit",
        lambda _process: None if persistent or len(probes) == 1 else 0,
    )
    monkeypatch.setattr(supervisor, "group_exists", exists)
    monkeypatch.setattr(supervisor, "signal_group", sent)
    process = SimpleNamespace(pid=987654, poll=lambda: 0, wait=lambda **_kwargs: 0)
    if persistent:
        with pytest.raises(PermissionError, match="group termination denied"):
            supervisor.stop_worker(process, clock=timing)
    else:
        supervisor.stop_worker(process, clock=timing)
        assert len(probes) == 2


@pytest.mark.parametrize("payload", [b"garbage\n", b"{}\n", b"x" * 9000, b""])
def test_status_channel_refuses_malformed_or_missing_results(
    supervisor, tmp_path, payload
):
    path = tmp_path / "status"
    path.write_bytes(payload)
    read = os.open(path, os.O_RDONLY)
    channel = supervisor.StatusChannel(read)
    try:
        with pytest.raises(OSError, match="status"):
            supervisor.result([], 1, channel, "", "")
    finally:
        os.close(read)


def test_missing_supervisor_does_not_block_ordinary_publication(monkeypatch, tmp_path):
    wrapper = load_script(SCRIPT.with_name("hourly-predict-publish.py"))
    support = importlib.import_module("launchd_paths")

    def unavailable(*_args, **_kwargs):
        raise OSError("shadow supervisor helper unavailable beside installed wrappers")

    monkeypatch.setattr(support, "run_supervised", unavailable)
    protocol = tmp_path / "protocol.json"
    window_protocol(protocol)
    paths = type(
        "Paths",
        (),
        {"project": tmp_path, "command": lambda _self, _name: [sys.executable]},
    )()
    enabled, diagnostic = wrapper.shadow_preflight(paths, protocol)
    assert not enabled
    assert "supervisor helper unavailable" in diagnostic


def test_success_status_cannot_hide_failed_supervisor(supervisor, tmp_path):
    path = tmp_path / "status"
    path.write_text(
        json.dumps(
            {
                "kind": "shadow_supervision",
                "schema_version": 2,
                "status": "completed",
                "returncode": 0,
                "cleanup_complete": True,
            }
        )
        + "\n"
    )
    descriptor = os.open(path, os.O_RDONLY)
    channel = supervisor.StatusChannel(descriptor)
    try:
        with pytest.raises(
            OSError, match="supervisor exited 1 after completed"
        ) as caught:
            supervisor.result(
                [],
                5,
                channel,
                "",
                "cleanup failed",
                supervisor_code=1,
            )
        assert "cleanup failed" in " ".join(caught.value.__notes__)
    finally:
        os.close(descriptor)


def test_long_failure_is_drained_and_preserves_error_and_path(supervisor):
    read, write = os.pipe()
    detail = (
        "original primary failure: "
        + "𝄞" * 30000
        + "/Application Support/records/record.json"
    )
    note = "secondary cleanup failure: " + "𝄞" * 10000 + "supervisor left unreaped"
    channel = supervisor.StatusChannel(read)
    reader = threading.Thread(target=channel.read_loop)
    reader.start()
    try:
        supervisor.send_status(write, "worker_ready", worker_pid=123456)
        supervisor.send_status(
            write, "runtime_failure", detail=detail, notes=[note], cleanup_complete=True
        )
        os.close(write)
        write = None
        reader.join(timeout=2)
        assert not reader.is_alive()
        channel.collect()
        terminal = channel.messages[-1]
        assert terminal["detail"]
        assert terminal["detail"].startswith("original primary failure: ")
        assert terminal["detail"].endswith("/Application Support/records/record.json")
        assert "[truncated]" in terminal["detail"]
        assert terminal["notes"]
        assert terminal["notes"][0].startswith("secondary cleanup failure: ")
        assert terminal["notes"][0].endswith("supervisor left unreaped")
        with pytest.raises(OSError, match="runtime_failure: original"):
            supervisor.result([], 1, channel, "", "", supervisor_code=1)
    finally:
        os.close(read)
        if write is not None:
            os.close(write)


def test_handshake_deadline_expires_with_live_parent(supervisor, tmp_path):
    control_read, control_write = os.pipe()
    status_read, status_write = os.pipe()
    marker = tmp_path / "never-started"
    command = [
        sys.executable,
        supervisor.__file__,
        "--control-fd",
        str(control_read),
        "--status-fd",
        str(status_write),
        "--deadline",
        str(time.monotonic() + 0.2),
        "--",
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
        str(marker),
    ]
    process = subprocess.Popen(  # noqa: S603 - synthetic supervisor and marker
        command, pass_fds=(control_read, status_write), start_new_session=True
    )
    os.close(control_read)
    os.close(status_write)
    try:
        assert process.wait(timeout=3) == 0
        channel = supervisor.StatusChannel(status_read)
        channel.collect()
        assert [message["status"] for message in channel.messages] == ["timed_out"]
        assert not marker.exists()
    finally:
        os.close(control_write)
        os.close(status_read)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)


def test_close_failure_inside_callers_except_is_operational(supervisor, monkeypatch):
    pipe, close = os.pipe, os.close
    allocated = []

    def created():
        pair = pipe()
        allocated.extend(pair)
        return pair

    def failed(descriptor):
        close(descriptor)
        if len(allocated) >= 4 and descriptor == allocated[2]:
            raise OSError("injected status descriptor close failure")

    monkeypatch.setattr(supervisor.os, "pipe", created)
    monkeypatch.setattr(supervisor.os, "close", failed)
    outer = ValueError("unrelated caller exception")
    try:
        raise outer
    except (ValueError,):  # noqa: B013 -- exercise an active caller exception
        with pytest.raises(OSError, match="status descriptor close failure"):
            supervisor.run_supervised([sys.executable, "-c", "pass"], timeout=5)
    assert not getattr(outer, "__notes__", [])


def test_descriptor_cleanup_preserves_explicit_primary(supervisor, monkeypatch):
    read, write = os.pipe()
    close = os.close
    primary = KeyboardInterrupt("original cancellation")

    def failed(descriptor):
        close(descriptor)
        if descriptor == read:
            raise OSError("secondary close failure")

    monkeypatch.setattr(supervisor.os, "close", failed)
    supervisor.close_handles({read, write}, None, primary=primary)
    assert "secondary close failure" in " ".join(primary.__notes__)


def test_emergency_cleanup_attempts_every_step_and_keeps_first_failure(
    supervisor, monkeypatch
):
    first = PermissionError("worker signal denied")
    calls = []

    def signal_failed(pid, _signum):
        calls.append(pid)
        raise first

    def reap_failed(**_kwargs):
        calls.append("reap")
        raise OSError("reap failed")

    channel = supervisor.StatusChannel(-1, owned_pid=123456)
    channel.control = SimpleNamespace(
        cancel=lambda: calls.append("cancel"), send=lambda _: calls.append("release")
    )
    process = SimpleNamespace(
        pid=987654, returncode=None, communicate=reap_failed, wait=reap_failed
    )
    monkeypatch.setattr(supervisor, "stopped_supervisor", lambda *_: True)
    monkeypatch.setattr(
        supervisor.os, "kill", lambda pid, sig: calls.append((pid, sig))
    )
    monkeypatch.setattr(supervisor, "signal_group", signal_failed)
    with pytest.raises(PermissionError, match="worker signal denied") as caught:
        supervisor.emergency_stop(process, channel)
    assert caught.value is first
    assert calls == [
        123456,
        "cancel",
        "release",
        (987654, signal.SIGCONT),
        "reap",
        (987654, signal.SIGKILL),
        "reap",
    ]
    assert "reap failed" in " ".join(first.__notes__)
    assert "supervisor left unreaped" in " ".join(first.__notes__)


def test_fresh_cancellation_is_not_swallowed_by_cleanup(supervisor, monkeypatch):
    cancelled = KeyboardInterrupt("new operator cancellation")
    primary = OSError("earlier failure")

    def interrupted(*_args, **_kwargs):
        raise cancelled

    process = SimpleNamespace(communicate=interrupted)
    channel = SimpleNamespace(control=None)
    cleaned = []
    monkeypatch.setattr(supervisor, "communicate_until", interrupted)
    monkeypatch.setattr(supervisor, "emergency_stop", lambda *_: cleaned.append(True))
    with pytest.raises(KeyboardInterrupt) as caught:
        supervisor.cancel_supervisor(process, channel, primary)
    assert caught.value is cancelled
    assert cleaned == [True]
    assert caught.value.__cause__ is primary
