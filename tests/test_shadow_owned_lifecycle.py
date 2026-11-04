"""Admission, reserved ownership and cancellation against real POSIX children."""

import json
import os
import select
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from test_launchd import SCRIPT
from test_shadow_supervision import (
    supervisor as supervisor,  # noqa: PLC0414 -- expose pytest fixture
)
from test_shadow_supervision import unlocked, wait_for

pytestmark = pytest.mark.skipif(os.name != "posix", reason="native POSIX ownership")


@contextmanager
def raw_owner(supervisor, command, *, timeout=3):
    control_read, control_write = os.pipe()
    status_read, status_write = os.pipe()
    process = supervisor.start_supervisor(
        command,
        timeout=timeout,
        capture_output=False,
        control_read=control_read,
        status_write=status_write,
    )
    os.close(control_read)
    os.close(status_write)
    control = supervisor.ControlChannel(control_write)
    channel = supervisor.StatusChannel(status_read)
    try:
        yield process, control, channel
    finally:
        control.close()
        try:
            process.wait(timeout=8)
        except (subprocess.TimeoutExpired,):  # noqa: B013 -- synthetic child backstop
            process.kill()
            process.wait(timeout=3)
        os.close(status_read)


def statuses(channel):
    channel.collect()
    return [item["status"] for item in channel.messages]


@pytest.mark.parametrize("initial", [b"", b"S"])
def test_parent_eof_with_or_without_queued_token_never_admits(
    supervisor, tmp_path, initial
):
    marker = tmp_path / "executed"
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
        str(marker),
    ]
    with raw_owner(supervisor, command) as (process, control, channel):
        if initial:
            control.send(initial)
        control.close()
        assert process.wait(timeout=5) == 0
        assert statuses(channel) == ["cancelled"]
        assert not marker.exists()


def test_owned_bootstrap_waits_for_admission_and_exits_after_cancel(
    supervisor, tmp_path
):
    marker = tmp_path / "executed"
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
        str(marker),
    ]
    with raw_owner(supervisor, command) as (process, control, channel):
        control.send(b"S")
        wait_for(lambda: statuses(channel) == ["worker_ready"])
        assert channel.owned_pid is not None
        assert not marker.exists()
        control.cancel()
        wait_for(lambda: len(statuses(channel)) == 2)
        assert statuses(channel)[-1] == "cancelled"
        assert (
            process.poll() is None
        )  # Still owns its unreaped bootstrap until release.
        assert not marker.exists()
        control.send(b"R")
        assert process.wait(timeout=3) == 0


def test_terminal_ack_precedes_reaping_and_completion_wins_cancel(supervisor):
    with raw_owner(supervisor, [sys.executable, "-c", "pass"]) as (
        process,
        control,
        channel,
    ):
        control.send(b"S")
        wait_for(lambda: statuses(channel) == ["worker_ready"])
        control.send(b"A")
        wait_for(lambda: len(statuses(channel)) == 2)
        assert statuses(channel)[-1] == "completed"
        assert process.poll() is None
        control.cancel()
        control.send(b"R")
        assert process.wait(timeout=3) == 0


def test_completion_checked_before_cancellation(supervisor, monkeypatch):
    read, write = os.pipe()
    monkeypatch.setattr(supervisor, "observe_exit", lambda _process: 0)
    try:
        assert (
            supervisor.monitor(
                SimpleNamespace(),
                read,
                time.monotonic() + 1,
                supervisor.Cancellation(requested=True),
            )
            == "completed"
        )
    finally:
        os.close(read)
        os.close(write)


def test_cancel_during_token_read_is_not_ready(supervisor, monkeypatch):
    cancellation = supervisor.Cancellation()

    class Selector:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def register(self, *_):
            pass

        def select(self, *_):
            return [True]

    def read(*_):
        cancellation.requested = True
        return b"A"

    monkeypatch.setattr(supervisor.selectors, "DefaultSelector", Selector)
    monkeypatch.setattr(supervisor.os, "read", read)
    assert (
        supervisor.await_token(999999, b"A", time.monotonic() + 1, cancellation)
        == "cancelled"
    )


def test_observe_exit_keeps_direct_child_waitable(supervisor):
    process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])
    try:
        wait_for(lambda: supervisor.observe_exit(process) is not None)
        assert supervisor.observe_exit(process) == 7
        assert process.returncode is None
        assert process.wait(timeout=1) == 7
        with pytest.raises(OSError, match="ownership already released"):
            supervisor.observe_exit(process)
    finally:
        process.wait(timeout=3)


def test_no_historical_worker_signal_after_release(supervisor, monkeypatch):
    channel = supervisor.StatusChannel(-1, owned_pid=123456, released=True)
    process = SimpleNamespace(returncode=0, communicate=lambda **_: ("", ""))
    monkeypatch.setattr(supervisor, "stopped_supervisor", lambda *_: False)
    monkeypatch.setattr(
        supervisor, "signal_group", lambda *_: pytest.fail("signaled historical group")
    )
    supervisor.emergency_stop(process, channel)


def test_unknown_dead_supervisor_reports_uncertainty_without_signaling(
    supervisor, monkeypatch
):
    channel = supervisor.StatusChannel(-1, owned_pid=123456)
    process = SimpleNamespace(returncode=1, communicate=lambda **_: ("", ""))
    monkeypatch.setattr(supervisor, "stopped_supervisor", lambda *_: False)
    monkeypatch.setattr(
        supervisor, "signal_group", lambda *_: pytest.fail("signaled uncertain group")
    )
    with pytest.raises(
        OSError, match="cannot certify worker cleanup without live ownership"
    ):
        supervisor.emergency_stop(process, channel)


def test_second_cancellation_runs_emergency_before_propagating(supervisor, monkeypatch):
    primary = KeyboardInterrupt("first")
    calls = []

    def interrupted(*_):
        raise KeyboardInterrupt("second")

    monkeypatch.setattr(supervisor, "communicate_until", interrupted)
    monkeypatch.setattr(
        supervisor, "emergency_stop", lambda *_: calls.append("cleaned")
    )
    with pytest.raises(KeyboardInterrupt, match="second") as caught:
        supervisor.cancel_supervisor(
            SimpleNamespace(), SimpleNamespace(control=None), primary
        )
    assert calls == ["cleaned"]
    assert caught.value.__cause__ is primary
    assert "second" in " ".join(primary.__notes__)


def test_nested_emergency_notes_reach_primary(supervisor, monkeypatch):
    primary = OSError("primary")
    secondary = OSError("signal failed")
    secondary.add_note("supervisor left unreaped")

    def failed(*_):
        raise secondary

    monkeypatch.setattr(supervisor, "communicate_until", failed)
    monkeypatch.setattr(supervisor, "emergency_stop", failed)
    supervisor.cancel_supervisor(
        SimpleNamespace(), SimpleNamespace(control=None), primary
    )
    assert "supervisor left unreaped" in " ".join(primary.__notes__)


def test_closed_start_pipe_keeps_terminal_timeout(supervisor, monkeypatch):
    original = supervisor.ControlChannel.send

    def delayed(self, token):
        if token == b"S":
            time.sleep(0.2)
        return original(self, token)

    monkeypatch.setattr(supervisor.ControlChannel, "send", delayed)
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        supervisor.run_supervised(
            [sys.executable, "-c", "pass"], timeout=0.05, capture_output=True
        )
    assert caught.value.timeout == 0.05
    assert not getattr(caught.value, "__notes__", ())


def test_error_protocol_does_not_claim_process_cleanup_failed(supervisor, monkeypatch):
    read, write = os.pipe()
    channel = supervisor.StatusChannel(read, owned_pid=123456)
    try:
        os.write(write, b"malformed\n")
        with pytest.raises(OSError, match="malformed shadow supervisor status"):
            channel.collect()
        process = SimpleNamespace(
            pid=987654, returncode=None, communicate=lambda **_: ("", "")
        )
        monkeypatch.setattr(supervisor, "stopped_supervisor", lambda *_: True)
        monkeypatch.setattr(supervisor, "signal_group", lambda *_: None)
        monkeypatch.setattr(supervisor.os, "kill", lambda *_: None)
        supervisor.emergency_stop(process, channel)
    finally:
        os.close(read)
        os.close(write)


def test_real_uv_double_term_unwinds_runner_staging(supervisor, tmp_path):
    uv = shutil.which("uv")
    assert uv is not None
    output = tmp_path / "record"
    cleaned = tmp_path / "cleanup-finished"
    program = """
import sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from temperature_shadow import cli_signals,staged_directory
with cli_signals(),staged_directory(Path(sys.argv[2])) as folder:
 try:
  (folder/'partial').write_text('retain no accepted record')
  print('staging ready',flush=True)
  time.sleep(30)
 finally:
  time.sleep(.1)
  Path(sys.argv[3]).write_text('cleanup survived forwarded SIGTERM')
"""
    command = [
        uv,
        "run",
        "--no-project",
        "--offline",
        "--python",
        sys.executable,
        "python",
        "-c",
        program,
        str(SCRIPT.parents[2] / "scripts"),
        str(output),
        str(cleaned),
    ]
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        supervisor.run_supervised(command, timeout=2, capture_output=True)
    assert b"staging ready" in caught.value.output
    assert cleaned.read_text() == "cleanup survived forwarded SIGTERM"
    assert not output.exists()
    assert not list(tmp_path.glob("shadow-stage-*"))
    assert unlocked(tmp_path / ".record.publishing")


@pytest.mark.parametrize("damage", ["missing", "syntax", "permission"])
@pytest.mark.parametrize(
    "wrapper", ["daily-maintenance.py", "hourly-predict-publish.py"]
)
def test_optional_helper_damage_preserves_ordinary_wrapper_help(
    tmp_path, damage, wrapper
):
    for name in ("launchd_paths.py", wrapper):
        shutil.copy2(SCRIPT.with_name(name), tmp_path / name)
    helper = tmp_path / "shadow_supervisor.py"
    if damage != "missing":
        helper.write_text(
            "def broken("
            if damage == "syntax"
            else "raise AssertionError('must remain lazy')"
        )
    if damage == "permission":
        helper.chmod(0)
    try:
        result = subprocess.run(  # noqa: S603 -- reviewed wrapper copy with --help
            [sys.executable, str(tmp_path / wrapper), "--help"],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        code = "from launchd_paths import run_shadow_child; run_shadow_child(['unused'],timeout=1)"
        shadow = subprocess.run(  # noqa: S603 -- damaged helper prevents worker admission
            [sys.executable, "-c", code],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        assert shadow.returncode == 1
        assert "shadow supervisor helper unavailable" in shadow.stderr
    finally:
        if helper.exists():
            helper.chmod(0o600)


def test_emergency_rescues_real_stopped_supervisor_and_reserved_worker(
    supervisor, tmp_path
):
    marker = tmp_path / "holding"
    guard = tmp_path / ".guard"
    program = "import sys,time; from pathlib import Path; from filelock import FileLock; lock=FileLock(sys.argv[1]); lock.acquire(); Path(sys.argv[2]).touch(); time.sleep(30)"
    with raw_owner(
        supervisor, [sys.executable, "-c", program, str(guard), str(marker)], timeout=20
    ) as (process, control, channel):
        control.send(b"S")
        wait_for(lambda: statuses(channel) == ["worker_ready"])
        control.send(b"A")
        wait_for(marker.exists)
        assert not unlocked(guard)
        channel.control = control
        os.kill(process.pid, signal.SIGSTOP)
        started = time.monotonic()
        supervisor.emergency_stop(process, channel)
        assert (
            time.monotonic() - started
            < supervisor.REAP_TIMEOUT + supervisor.WAIT_MARGIN
        )
        assert process.returncode is not None
        with pytest.raises(ChildProcessError):
            os.waitpid(process.pid, os.WNOHANG)
        assert unlocked(guard)


def test_emergency_before_identity_read_does_not_admit_unknown_bootstrap(
    supervisor, tmp_path
):
    marker = tmp_path / "never-executed"
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
        str(marker),
    ]
    with raw_owner(supervisor, command, timeout=20) as (process, control, channel):
        control.send(b"S")
        assert select.select([channel.descriptor], [], [], 3)[0]
        assert channel.owned_pid is None
        channel.control = control
        supervisor.emergency_stop(process, channel)
        assert not marker.exists()
        assert process.returncode is not None


def test_worker_pipe_allocation_failure_closes_prior_descriptors(
    supervisor, monkeypatch
):
    pipe = os.pipe
    allocated = []

    def fail_second():
        if allocated:
            raise OSError("second pipe failed")
        pair = pipe()
        allocated.extend(pair)
        return pair

    monkeypatch.setattr(supervisor.os, "pipe", fail_second)
    deadline = time.monotonic() + 1
    with pytest.raises(OSError, match="second pipe failed"):
        supervisor.start_worker(["never-admitted"], deadline)
    for descriptor in allocated:
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(descriptor)


def test_repeated_real_ctrl_c_defers_until_worker_cleanup_finishes(tmp_path):
    guard = tmp_path / ".guard"
    ready = tmp_path / "ready"
    cleaning = tmp_path / "cleaning"
    cleaned = tmp_path / "cleaned"
    program = """
import sys,time
from pathlib import Path
from filelock import FileLock
sys.path.insert(0,sys.argv[1])
from temperature_shadow import cli_signals
with cli_signals(), FileLock(sys.argv[2]):
 try:
  Path(sys.argv[3]).touch()
  time.sleep(30)
 finally:
  Path(sys.argv[4]).touch()
  time.sleep(.7)
  Path(sys.argv[5]).touch()
"""
    wrapper = """
import sys
sys.path.insert(0,sys.argv[1])
from shadow_supervisor import run_supervised
run_supervised([sys.executable,'-c',sys.argv[2],*sys.argv[3:]],timeout=20)
"""
    owner = subprocess.Popen(  # noqa: S603 -- synthetic wrapper and worker
        [
            sys.executable,
            "-c",
            wrapper,
            str(SCRIPT.parent),
            program,
            str(SCRIPT.parents[2] / "scripts"),
            str(guard),
            str(ready),
            str(cleaning),
            str(cleaned),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for(ready.exists)
        owner.send_signal(signal.SIGINT)
        wait_for(cleaning.exists)
        owner.send_signal(signal.SIGINT)
        _, stderr = owner.communicate(timeout=9)
        assert owner.returncode != 0
        assert cleaned.exists()
        assert unlocked(guard)
        assert "additional cancellation deferred" in stderr
    finally:
        if owner.returncode is None:
            owner.kill()
            owner.communicate(timeout=3)


def test_waitid_stop_notification_is_not_worker_completion(supervisor, monkeypatch):
    info = SimpleNamespace(si_code=os.CLD_STOPPED, si_status=signal.SIGSTOP)
    monkeypatch.setattr(supervisor.os, "waitid", lambda *_: info)
    assert supervisor.observe_exit(SimpleNamespace(pid=123456, returncode=None)) is None


def test_status_handles_surrogate_paths_with_explicit_budget(supervisor):
    payload = supervisor.encode_status(
        "runtime_failure",
        {
            "detail": "permission: " + "\udcff" * 100000 + "/records/record.json",
            "notes": ["worker cleanup failed: " + "𝄞" * 100000 + "unreaped"],
            "cleanup_complete": True,
        },
    )
    assert len(payload) < supervisor.STATUS_LIMIT
    value = json.loads(payload)
    assert value["detail"].startswith("permission: ")
    assert value["detail"].endswith("/records/record.json")
    assert "[truncated]" in value["detail"]
    assert value["notes"][0].endswith("unreaped")
