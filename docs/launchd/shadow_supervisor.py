# ruff: noqa: INP001 -- installed beside the standalone launchd wrappers
"""Bound a shadow worker group to its wrapper's private pipe and deadline."""

from __future__ import annotations

import argparse
import json
import os
import selectors
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

TERM_GRACE = 2.0
REAP_TIMEOUT = 3.0
POLL_INTERVAL = 0.05
STATUS_LIMIT = 8192


def send_status(descriptor: int, status: str, **fields: Any) -> None:
    if "detail" in fields:
        fields["detail"] = str(fields["detail"])[:320]
    if "notes" in fields:
        fields["notes"] = [str(note)[:120] for note in fields["notes"][:2]]
    # At most two frames are sent. Fit both in the minimum atomic pipe capacity
    # so the producer cannot block while its parent is draining worker output.
    budget = min(2048, os.fpathconf(descriptor, "PC_PIPE_BUF") // 2)
    payload = encode_status(status, fields)
    while len(payload) > budget:
        shrink_status(fields)
        payload = encode_status(status, fields)
    with suppress(BrokenPipeError):
        os.write(descriptor, payload)


def encode_status(status: str, fields: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(kind="shadow_supervision", schema_version=1, status=status, **fields)
        ).encode()
        + b"\n"
    )


def shrink_status(fields: dict[str, Any]) -> None:
    notes = fields.get("notes", [])
    detail = fields.get("detail", "")
    if len(notes) > 1:
        notes.pop()
    elif detail and (not notes or len(detail) >= len(notes[0])):
        fields["detail"] = detail[: len(detail) // 2]
    elif notes and notes[0]:
        notes[0] = notes[0][: len(notes[0]) // 2]
    else:
        raise OSError("shadow supervisor status cannot fit pipe budget")


def signal_group(pid: int, signum: int) -> None:
    with suppress(ProcessLookupError):
        os.killpg(pid, signum)


def group_exists(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except (ProcessLookupError,):  # noqa: B013 -- project exception convention
        return False
    return True


def stop_worker(process: subprocess.Popen[str]) -> None:
    """Reap the launcher and terminate descendants even if their launcher exited."""
    # Darwin can transiently deny group probes while adopted zombies are reaped.
    # Do not resend TERM into a handler that is already unwinding its contexts.
    with suppress(PermissionError):
        signal_group(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + TERM_GRACE
    while time.monotonic() < deadline:
        process.poll()
        try:
            if not group_exists(process.pid):
                process.wait(timeout=REAP_TIMEOUT)
                return
        except (PermissionError,):  # noqa: B013 -- retry only within the grace period
            pass
        time.sleep(POLL_INTERVAL)
    signal_group(process.pid, signal.SIGKILL)
    process.wait(timeout=REAP_TIMEOUT)


@dataclass
class Cancellation:
    requested: bool = False

    def handle(self, _signum: int, _frame: object) -> None:
        self.requested = True


def monitor(
    process: subprocess.Popen[str],
    control: int,
    deadline: float,
    cancelled: Cancellation,
) -> str:
    with selectors.DefaultSelector() as selector:
        selector.register(control, selectors.EVENT_READ)
        while True:
            if cancelled.requested:
                return "cancelled"
            if process.poll() is not None:
                return "completed"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "timed_out"
            if selector.select(min(POLL_INTERVAL, remaining)):
                if os.read(control, 1):
                    raise OSError("unexpected parent-liveness data")
                return "cancelled"


def await_start(control: int, deadline: float, cancelled: Cancellation) -> str:
    with selectors.DefaultSelector() as selector:
        selector.register(control, selectors.EVENT_READ)
        while not cancelled.requested:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "timed_out"
            if selector.select(min(POLL_INTERVAL, remaining)):
                return "ready" if os.read(control, 1) == b"S" else "cancelled"
    return "cancelled"


def supervise(
    command: Sequence[str], control: int, status: int, deadline: float
) -> int:
    cancelled = Cancellation()
    previous = {
        signum: signal.signal(signum, cancelled.handle)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    process = None
    try:
        # No worker is started if its wrapper died before completing the handshake.
        readiness = await_start(control, deadline, cancelled)
        if readiness != "ready":
            send_status(status, readiness)
            return 0
        if time.monotonic() >= deadline:
            send_status(status, "timed_out")
            return 0
        process = subprocess.Popen(  # noqa: S603 -- fixed operator-selected shadow argv
            command, text=True, close_fds=True, process_group=0
        )
        send_status(status, "started", worker_pid=process.pid)
        outcome = monitor(process, control, deadline, cancelled)
        stop_worker(process)
        send_status(status, outcome, returncode=process.returncode)
        return 0
    except (Exception,) as error:  # noqa: B013 -- project exception convention
        if process is not None:
            try:
                signal_group(process.pid, signal.SIGKILL)
                process.wait(timeout=REAP_TIMEOUT)
            except (Exception,) as secondary:  # noqa: B013 -- project exception convention
                error.add_note(f"worker cleanup failed: {secondary}")
        send_status(
            status,
            "runtime_failure",
            detail=str(error),
            notes=getattr(error, "__notes__", []),
        )
        return 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        os.close(control)
        os.close(status)


@dataclass
class StatusChannel:
    descriptor: int
    payload: bytes = b""
    messages: list[dict[str, Any]] = field(default_factory=list)

    def collect(self) -> None:
        os.set_blocking(self.descriptor, False)
        while True:
            try:
                chunk = os.read(self.descriptor, STATUS_LIMIT)
            except (BlockingIOError,):  # noqa: B013 -- project exception convention
                break
            if not chunk:
                break
            self.payload += chunk
            if len(self.payload) > STATUS_LIMIT:
                raise OSError("shadow supervisor status exceeds limit")
        while b"\n" in self.payload:
            raw, self.payload = self.payload.split(b"\n", 1)
            try:
                value = json.loads(raw)
            except (ValueError,) as error:  # noqa: B013 -- JSON input boundary
                raise OSError("malformed shadow supervisor status") from error
            if (
                not isinstance(value, dict)
                or value.get("kind") != "shadow_supervision"
                or type(value.get("schema_version")) is not int
                or value["schema_version"] != 1
                or len(self.messages) >= 2
            ):
                raise OSError("invalid shadow supervisor status")
            self.messages.append(value)

    def worker_pid(self) -> int | None:
        self.collect()
        if self.messages and self.messages[0].get("status") == "started":
            value = self.messages[0].get("worker_pid")
            if type(value) is int and value > 1:
                return value
        return None


def emergency_stop(process: subprocess.Popen[str], channel: StatusChannel) -> None:
    worker = channel.worker_pid()
    if worker is not None:
        signal_group(worker, signal.SIGKILL)
    signal_group(process.pid, signal.SIGKILL)
    process.communicate(timeout=REAP_TIMEOUT)


def result(
    command: Sequence[str],
    timeout: float,
    channel: StatusChannel,
    stdout: str | None,
    stderr: str | None,
    *,
    supervisor_code: int | None = 0,
) -> subprocess.CompletedProcess[str]:
    channel.collect()
    if not channel.messages or channel.payload:
        raise OSError("missing complete shadow supervisor status")
    terminal = channel.messages[-1]
    status = terminal.get("status")
    if supervisor_code != 0 and status != "runtime_failure":
        raise OSError(f"shadow supervisor exited {supervisor_code} after {status}")
    if status == "timed_out":
        raise subprocess.TimeoutExpired(
            command,
            timeout,
            output=stdout.encode() if stdout is not None else None,
            stderr=stderr.encode() if stderr is not None else None,
        )
    if status == "completed" and type(terminal.get("returncode")) is int:
        return subprocess.CompletedProcess(
            command, terminal["returncode"], stdout, stderr
        )
    error = OSError(
        f"shadow supervisor {status}: {terminal.get('detail', 'worker cancelled')}"
    )
    for note in terminal.get("notes", []):
        error.add_note(str(note))
    raise error


def start_supervisor(
    command: Sequence[str],
    *,
    timeout: float,
    capture_output: bool,
    control_read: int,
    status_write: int,
) -> subprocess.Popen[str]:
    # The installed helper executes reviewed argv without shell parsing.
    return subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--control-fd",
            str(control_read),
            "--status-fd",
            str(status_write),
            "--deadline",
            repr(time.monotonic() + timeout),
            "--",
            *command,
        ],
        pass_fds=(control_read, status_write),
        start_new_session=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=True,
    )


def cancel_supervisor(
    process: subprocess.Popen[str], channel: StatusChannel, primary: BaseException
) -> None:
    try:
        process.communicate(timeout=TERM_GRACE + REAP_TIMEOUT)
        channel.collect()
        if channel.messages and channel.messages[-1].get("status") == "runtime_failure":
            primary.add_note(
                f"shadow child cleanup failed: {channel.messages[-1].get('detail')}"
            )
    except (Exception, KeyboardInterrupt, SystemExit):
        try:
            emergency_stop(process, channel)
        except (Exception, KeyboardInterrupt, SystemExit) as secondary:
            primary.add_note(f"shadow child cleanup failed: {secondary}")


def close_handles(
    descriptors: set[int],
    process: subprocess.Popen[str] | None,
    *,
    primary: BaseException | None,
) -> None:
    failures = []
    closers: list[Callable[[], None]] = [
        partial(os.close, descriptor) for descriptor in descriptors
    ]
    if process is not None:
        closers.extend(
            pipe.close for pipe in (process.stdout, process.stderr) if pipe is not None
        )
    for close in closers:
        try:
            close()
        except (OSError,) as error:  # noqa: B013 -- close every owned descriptor
            failures.append(error)
    if failures:
        if primary is None:
            raise failures[0]
        for error in failures:
            primary.add_note(f"shadow descriptor cleanup failed: {error}")


def run_supervised(
    command: Sequence[str], *, timeout: float, capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
    descriptors: set[int] = set()
    process = None
    primary = None
    try:
        control_read, control_write = os.pipe()
        descriptors.update((control_read, control_write))
        status_read, status_write = os.pipe()
        descriptors.update((status_read, status_write))
        channel = StatusChannel(status_read)
        process = start_supervisor(
            command,
            timeout=timeout,
            capture_output=capture_output,
            control_read=control_read,
            status_write=status_write,
        )
        for descriptor in (control_read, status_write):
            os.close(descriptor)
            descriptors.remove(descriptor)
        try:
            os.write(control_write, b"S")
            stdout, stderr = process.communicate(
                timeout=timeout + TERM_GRACE + REAP_TIMEOUT + 1
            )
        except (BaseException,) as error:  # noqa: B013 -- project exception convention
            try:
                os.close(control_write)
            except (OSError,) as secondary:  # noqa: B013 -- preserve cancellation
                error.add_note(f"shadow liveness-pipe cleanup failed: {secondary}")
            else:
                descriptors.remove(control_write)
            cancel_supervisor(process, channel, error)
            raise
        try:
            return result(
                command,
                timeout,
                channel,
                stdout,
                stderr,
                supervisor_code=process.returncode,
            )
        except (OSError,) as error:  # noqa: B013 -- failed supervision needs emergency cleanup
            if stderr:
                error.add_note(f"shadow supervisor stderr: {stderr[-400:]}")
            try:
                emergency_stop(process, channel)
            except (Exception,) as secondary:  # noqa: B013 -- preserve supervision failure
                error.add_note(f"shadow child cleanup failed: {secondary}")
            raise
    except (BaseException,) as error:  # noqa: B013 -- retain only this call's propagating failure
        primary = error
        raise
    finally:
        close_handles(descriptors, process, primary=primary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-fd", type=int, required=True)
    parser.add_argument("--status-fd", type=int, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return supervise(command, args.control_fd, args.status_fd, args.deadline)


if __name__ == "__main__":
    raise SystemExit(main())
