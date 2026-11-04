# ruff: noqa: INP001 -- installed beside the standalone launchd wrappers
"""Bound shadow command admission and owned process groups to a surviving supervisor."""

from __future__ import annotations

import argparse
import json
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, cast

TERM_GRACE = 2.0
REAP_TIMEOUT = 3.0
WAIT_MARGIN = 1.0
POLL_INTERVAL = 0.05
STATUS_LIMIT = 64 * 1024
TRUNCATED = " ... [truncated] ... "
TERMINAL = frozenset({"completed", "cancelled", "timed_out", "runtime_failure"})


@dataclass(frozen=True)
class Clock:
    now: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep


CLOCK = Clock()


def limited_text(value: str, budget: int) -> str:
    """Bound encoded JSON, retaining both the error prefix and its final path/context."""
    if len(json.dumps(value).encode()) <= budget:
        return value
    low, high = 0, len(value)
    while low < high:
        size = (low + high + 1) // 2
        candidate = (
            value[: (size + 1) // 2]
            + TRUNCATED
            + (value[-(size // 2) :] if size > 1 else "")
        )
        if len(json.dumps(candidate).encode()) <= budget:
            low = size
        else:
            high = size - 1
    return (
        value[: (low + 1) // 2] + TRUNCATED + (value[-(low // 2) :] if low > 1 else "")
    )


def encode_status(status: str, fields: dict[str, Any]) -> bytes:
    fields = dict(fields)
    if "detail" in fields:
        fields["detail"] = limited_text(str(fields["detail"]), STATUS_LIMIT * 3 // 4)
    if "notes" in fields:
        notes = [str(note) for note in fields["notes"]]
        if len(notes) > 32:
            notes = [*notes[:15], TRUNCATED, *notes[-16:]]
        fields["notes"] = [
            limited_text(note, (STATUS_LIMIT // 8) // max(1, len(notes)))
            for note in notes
        ]
    payload = (
        json.dumps(
            dict(kind="shadow_supervision", schema_version=2, status=status, **fields),
            ensure_ascii=True,
        ).encode()
        + b"\n"
    )
    if len(payload) > STATUS_LIMIT - 1024:
        raise OSError("shadow supervisor status exceeds its encoded budget")
    return payload


def send_status(descriptor: int, status: str, **fields: Any) -> None:
    payload = memoryview(encode_status(status, fields))
    with suppress(BrokenPipeError):
        while payload:
            written = os.write(descriptor, payload)
            if not written:
                raise OSError("shadow status write made no progress")
            payload = payload[written:]


def exception_notes(error: BaseException) -> list[str]:
    return [str(error), *getattr(error, "__notes__", ())]


def attach_failure(primary: BaseException, label: str, error: BaseException) -> None:
    for detail in exception_notes(error):
        primary.add_note(f"{label}: {detail}")


def signal_group(pid: int, signum: int) -> None:
    with suppress(ProcessLookupError):
        os.killpg(pid, signum)


def group_exists(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except (ProcessLookupError,):  # noqa: B013 -- project exception convention
        return False
    return True


def wait_notification(
    process: subprocess.Popen[str], flags: int
) -> os.waitid_result | None:
    """WNOHANG returns None on supported runtimes, despite narrower platform stubs."""
    return cast(
        "os.waitid_result | None",
        os.waitid(os.P_PID, process.pid, flags | os.WNOHANG | os.WNOWAIT),
    )


def observe_exit(process: subprocess.Popen[str]) -> int | None:
    """Inspect our direct child without releasing the PID that reserves its group ID."""
    if process.returncode is not None:
        raise OSError(
            "shadow ownership already released; refusing historical PID signaling"
        )
    info = wait_notification(process, os.WEXITED)
    if info is None or info.si_code not in (
        os.CLD_EXITED,
        os.CLD_KILLED,
        os.CLD_DUMPED,
    ):
        return None
    return info.si_status if info.si_code == os.CLD_EXITED else -info.si_status


def wait_exit(
    process: subprocess.Popen[str], deadline: float, *, clock: Clock = CLOCK
) -> int:
    while (code := observe_exit(process)) is None:
        if clock.now() >= deadline:
            raise OSError("owned shadow worker did not exit after forced termination")
        clock.sleep(POLL_INTERVAL)
    return code


def stop_worker(process: subprocess.Popen[str], *, clock: Clock = CLOCK) -> int:
    """End the group, retaining the launcher as a waitable child until release ACK."""
    with suppress(PermissionError):
        signal_group(process.pid, signal.SIGTERM)
    deadline = clock.now() + TERM_GRACE
    while clock.now() < deadline:
        try:
            if not group_exists(process.pid):
                return wait_exit(process, clock.now() + REAP_TIMEOUT, clock=clock)
        except (PermissionError,):  # noqa: B013 -- retry only during TERM grace
            # Darwin reports EPERM for a group containing only our reserved zombie.
            # A live same-user descendant remains signalable (covered through uv).
            if sys.platform == "darwin" and (code := observe_exit(process)) is not None:
                return code
        clock.sleep(POLL_INTERVAL)
    signal_group(process.pid, signal.SIGKILL)
    return wait_exit(process, clock.now() + REAP_TIMEOUT, clock=clock)


@dataclass
class Cancellation:
    requested: bool = False
    signum: int | None = None

    def handle(self, _signum: int, _frame: object) -> None:
        self.requested = True
        self.signum = _signum


def control_cancelled(control: int) -> bool:
    with selectors.DefaultSelector() as selector:
        selector.register(control, selectors.EVENT_READ)
        if not selector.select(0):
            return False
        token = os.read(control, 1)
        if token in (b"", b"C"):
            return True
        raise OSError("unexpected shadow control token")


def await_token(
    control: int,
    token: bytes,
    deadline: float,
    cancelled: Cancellation,
    *,
    clock: Clock = CLOCK,
) -> str:
    with selectors.DefaultSelector() as selector:
        selector.register(control, selectors.EVENT_READ)
        while not cancelled.requested:
            remaining = deadline - clock.now()
            if remaining <= 0:
                return "timed_out"
            if selector.select(min(POLL_INTERVAL, remaining)):
                received = os.read(control, 1)
                if received in (b"", b"C") or cancelled.requested:
                    return "cancelled"
                if received != token:
                    raise OSError("unexpected shadow admission token")
                return "cancelled" if control_cancelled(control) else "ready"
    return "cancelled"


def await_start(control: int, deadline: float, cancelled: Cancellation) -> str:
    return await_token(control, b"S", deadline, cancelled)


def monitor(
    process: subprocess.Popen[str],
    control: int,
    deadline: float,
    cancelled: Cancellation,
    *,
    clock: Clock = CLOCK,
) -> str:
    with selectors.DefaultSelector() as selector:
        selector.register(control, selectors.EVENT_READ)
        while True:
            if observe_exit(process) is not None:
                return "completed"
            if cancelled.requested:
                return "cancelled"
            remaining = deadline - clock.now()
            if remaining <= 0:
                return "timed_out"
            if selector.select(min(POLL_INTERVAL, remaining)):
                if os.read(control, 1) not in (b"", b"C"):
                    raise OSError("unexpected parent-liveness data")
                return "completed" if observe_exit(process) is not None else "cancelled"


def bootstrap(
    command: Sequence[str], admission: int, error_pipe: int, deadline: float
) -> int:
    """No command executes without admission; EOF after supervisor death aborts."""
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(admission, selectors.EVENT_READ)
            if (
                not selector.select(max(0, deadline - time.monotonic()))
                or os.read(admission, 1) != b"G"
            ):
                return 125
        os.close(admission)
        os.set_inheritable(error_pipe, False)  # noqa: FBT003 -- POSIX positional-only API
        # Reviewed operator-selected argv; no shell interpretation.
        os.execvp(command[0], list(command))  # noqa: S606
    except (OSError,) as error:  # noqa: B013 -- retain launcher failure semantics
        payload = json.dumps({"detail": str(error), "errno": error.errno}).encode()
        os.write(error_pipe, payload)
        return 127
    return 127


@dataclass
class Worker:
    process: subprocess.Popen[str]
    admission: int
    errors: int
    cleaned: bool = False
    released: bool = False

    def admit(self) -> None:
        os.write(self.admission, b"G")
        os.close(self.admission)
        self.admission = -1

    def close_admission(self) -> None:
        if self.admission >= 0:
            os.close(self.admission)
            self.admission = -1

    def exec_error(self) -> None:
        os.set_blocking(self.errors, False)
        try:
            raw = os.read(self.errors, STATUS_LIMIT)
        except (BlockingIOError,):  # noqa: B013 -- execution still starting
            return
        if raw:
            value = json.loads(raw)
            raise OSError(value["errno"], value["detail"])


def start_worker(command: Sequence[str], deadline: float) -> Worker:
    allocated: set[int] = set()
    process = None
    try:
        admission_read, admission_write = os.pipe()
        allocated.update((admission_read, admission_write))
        error_read, error_write = os.pipe()
        allocated.update((error_read, error_write))
        process = subprocess.Popen(  # noqa: S603 -- bootstrap gates reviewed argv
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--bootstrap",
                "--admission-fd",
                str(admission_read),
                "--error-fd",
                str(error_write),
                "--deadline",
                repr(deadline),
                "--",
                *command,
            ],
            text=True,
            close_fds=True,
            process_group=0,
            pass_fds=(admission_read, error_write),
        )
        for descriptor in (admission_read, error_write):
            os.close(descriptor)
            allocated.remove(descriptor)
        return Worker(process, admission_write, error_read)
    except (BaseException,) as error:  # noqa: B013 -- allocation never admits a command
        if process is not None:
            try:
                process.kill()
                process.wait(timeout=REAP_TIMEOUT)
            except (Exception,) as secondary:  # noqa: B013 -- retain allocation failure
                attach_failure(error, "bootstrap cleanup failed", secondary)
        close_handles(allocated, None, primary=error)
        raise


def await_release(control: int) -> None:
    """A live wrapper retains PID ownership until ACK; its EOF authorizes reaping."""
    while os.read(control, 1) not in (b"R", b""):
        pass  # Cancellation may precede the terminal ACK.


def worker_outcome(
    worker: Worker, control: int, status: int, deadline: float, cancelled: Cancellation
) -> str:
    send_status(status, "worker_ready", worker_pid=worker.process.pid)
    admitted = await_token(control, b"A", deadline, cancelled)
    if admitted != "ready":
        return admitted
    if cancelled.requested or control_cancelled(control):
        return "cancelled"
    if time.monotonic() >= deadline:
        return "timed_out"
    worker.admit()
    return monitor(worker.process, control, deadline, cancelled)


def finish_worker(worker: Worker, control: int, status: int, outcome: str) -> None:
    worker.close_admission()
    code = stop_worker(worker.process)
    worker.cleaned = True
    worker.exec_error()
    send_status(status, outcome, returncode=code, cleanup_complete=True)
    await_release(control)
    worker.released = True
    worker.process.wait(timeout=REAP_TIMEOUT)


def failed_worker(
    worker: Worker | None, control: int, status: int, error: Exception
) -> None:
    cleaned = worker is None or worker.cleaned
    if worker is not None and not cleaned:
        worker.close_admission()
        try:
            signal_group(worker.process.pid, signal.SIGKILL)
            wait_exit(worker.process, time.monotonic() + REAP_TIMEOUT)
            cleaned = True
        except (Exception,) as secondary:  # noqa: B013 -- keep the original supervisor failure
            attach_failure(error, "worker cleanup failed", secondary)
    send_status(
        status,
        "runtime_failure",
        detail=str(error),
        notes=getattr(error, "__notes__", []),
        cleanup_complete=cleaned,
    )
    if worker is not None:
        await_release(control)
        if cleaned:
            worker.process.wait(timeout=REAP_TIMEOUT)


def supervise(
    command: Sequence[str], control: int, status: int, deadline: float
) -> int:
    cancelled = Cancellation()
    previous = {
        sig: signal.signal(sig, cancelled.handle)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    worker = None
    try:
        readiness = await_start(control, deadline, cancelled)
        if readiness != "ready":
            send_status(status, readiness, cleanup_complete=True)
            return 0
        worker = start_worker(command, deadline)
        outcome = worker_outcome(worker, control, status, deadline, cancelled)
        finish_worker(worker, control, status, outcome)
        return 0
    except (Exception,) as error:  # noqa: B013 -- supervisor boundary
        if worker is not None and worker.released:
            print(
                f"shadow supervisor after ownership release: {error}", file=sys.stderr
            )
            return 1
        failed_worker(worker, control, status, error)
        return 1
    finally:
        if worker is not None:
            worker.close_admission()
            os.close(worker.errors)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        os.close(control)
        os.close(status)


@dataclass
class ControlChannel:
    descriptor: int
    cancelled: bool = False
    closed: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)

    def send(self, token: bytes) -> bool:
        with self.lock:
            if self.closed or (self.cancelled and token in (b"S", b"A")):
                return False
            try:
                os.write(self.descriptor, token)
            except (BrokenPipeError,):  # noqa: B013 -- terminal status remains authoritative
                return False
            return True

    def cancel(self) -> None:
        with self.lock:
            if not self.cancelled:
                self.cancelled = True
                self.send(b"C")

    def close(self) -> None:
        with self.lock:
            if not self.closed:
                self.closed = True
                os.close(self.descriptor)


def validate_diagnostic(value: dict[str, Any]) -> None:
    if not isinstance(value.get("status"), str) or not isinstance(
        value.get("detail", ""), str
    ):
        raise OSError("invalid shadow supervisor diagnostic fields")
    notes = value.get("notes", [])
    if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
        raise OSError("invalid shadow supervisor diagnostic fields")


def diagnostic_error(
    value: dict[str, Any], default: str = "worker cancelled"
) -> OSError:
    error = OSError(
        f"shadow supervisor {value['status']}: {value.get('detail', default)}"
    )
    for note in value.get("notes", []):
        error.add_note(str(note))
    return error


@dataclass
class StatusChannel:
    descriptor: int
    payload: bytes = b""
    messages: list[dict[str, Any]] = field(default_factory=list)
    owned_pid: int | None = None
    released: bool = False
    error: Exception | None = None
    total: int = 0
    control: ControlChannel | None = None
    thread: threading.Thread | None = None
    stopping: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def start(self, control: ControlChannel) -> None:
        self.control = control
        self.thread = threading.Thread(
            target=self.read_loop, name="shadow-status", daemon=True
        )
        self.thread.start()

    def read_loop(self) -> None:
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self.descriptor, selectors.EVENT_READ)
                while not self.stopping.is_set():
                    if selector.select(POLL_INTERVAL) and not self.drain():
                        return
        except (Exception,) as error:  # noqa: B013 -- transport errors reach wrapper thread
            with self.lock:
                self.error = self.error or error

    def drain(self) -> bool:
        os.set_blocking(self.descriptor, False)
        while not self.stopping.is_set():
            try:
                chunk = os.read(self.descriptor, STATUS_LIMIT)
            except (BlockingIOError,):  # noqa: B013 -- continuous nonblocking reader
                return True
            with self.lock:
                self.feed(chunk)
            if not chunk:
                return False
        return False

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            if self.payload and self.error is None:
                self.error = OSError("incomplete shadow supervisor status")
            return
        self.total += len(chunk)
        if self.total > STATUS_LIMIT:
            self.error = self.error or OSError("shadow supervisor status exceeds limit")
        if self.error is None:
            try:
                self.parse(chunk)
            except (Exception,) as error:  # noqa: B013 -- keep draining broken producers
                self.error = error

    def parse(self, chunk: bytes) -> None:
        self.payload += chunk
        while b"\n" in self.payload:
            raw, self.payload = self.payload.split(b"\n", 1)
            try:
                value = json.loads(raw)
            except (ValueError,) as error:  # noqa: B013 -- private protocol boundary
                raise OSError("malformed shadow supervisor status") from error
            if (
                not isinstance(value, dict)
                or value.get("kind") != "shadow_supervision"
                or type(value.get("schema_version")) is not int
                or value["schema_version"] != 2
                or len(self.messages) >= 2
            ):
                raise OSError("invalid shadow supervisor status")
            self.accept(value)
            self.messages.append(value)

    def accept(self, value: dict[str, Any]) -> None:
        validate_diagnostic(value)
        status = value["status"]
        if status == "worker_ready" and not self.messages:
            self.admit_identity(value)
        elif status in TERMINAL and type(value.get("cleanup_complete")) is bool:
            self.release_identity(value)
        else:
            raise OSError("invalid shadow supervision phase")

    def admit_identity(self, value: dict[str, Any]) -> None:
        pid = value.get("worker_pid")
        if type(pid) is not int or pid <= 1:
            raise OSError("invalid owned shadow worker identity")
        self.owned_pid = pid
        if self.control is not None:
            self.control.send(b"A")

    def release_identity(self, value: dict[str, Any]) -> None:
        if not value["cleanup_complete"]:
            raise diagnostic_error(value, "cleanup incomplete")
        self.released = True
        if self.control is not None:
            self.control.send(b"R")

    def collect(self) -> None:
        if self.thread is None:
            self.drain()
        with self.lock:
            if self.error is not None:
                raise self.error

    def finish(self) -> None:
        if self.thread is not None:
            self.thread.join(timeout=WAIT_MARGIN)
            if self.thread.is_alive():
                self.stopping.set()
                self.thread.join(timeout=WAIT_MARGIN)
                if self.thread.is_alive():
                    raise OSError("shadow status reader did not stop")
        self.collect()

    def stop(self) -> None:
        self.stopping.set()
        if self.thread is not None:
            self.thread.join(timeout=WAIT_MARGIN)
            if self.thread.is_alive():
                raise OSError("shadow status reader left running")


def stopped_supervisor(process: subprocess.Popen[str], deadline: float) -> bool:
    """Only our still-waitable, stopped child can vouch for its unreaped worker."""
    if process.returncode is not None or observe_exit(process) is not None:
        return False
    os.kill(process.pid, signal.SIGSTOP)
    while time.monotonic() < deadline:
        info = wait_notification(process, os.WEXITED | os.WSTOPPED)
        if info is not None:
            return info.si_code == os.CLD_STOPPED
        time.sleep(POLL_INTERVAL)
    raise OSError("could not verify stopped shadow supervisor")


def resume_supervisor(process: subprocess.Popen[str], channel: StatusChannel) -> None:
    if channel.control is not None:
        channel.control.cancel()
        channel.control.send(b"R")
    os.kill(process.pid, signal.SIGCONT)


def reap_supervisor(process: subprocess.Popen[str], deadline: float) -> None:
    try:
        process.communicate(timeout=max(0, deadline - time.monotonic()))
    except (Exception,) as error:  # noqa: B013 -- bounded final rescue
        # Popen.kill safely handles a direct child that exited during communicate.
        try:
            process.kill()
        except (Exception,) as secondary:  # noqa: B013 -- reaping must still be attempted
            attach_failure(error, "supervisor kill failed", secondary)
        try:
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except (Exception,) as secondary:  # noqa: B013 -- expose unreaped state
            attach_failure(error, "supervisor left unreaped", secondary)
        raise


def emergency_stop(process: subprocess.Popen[str], channel: StatusChannel) -> None:
    deadline = time.monotonic() + REAP_TIMEOUT
    failures: list[Exception] = []
    stopped = False
    try:
        stopped = stopped_supervisor(process, deadline)
        with channel.lock:
            worker, released = channel.owned_pid, channel.released
        if not stopped and not released:
            raise OSError(
                "supervisor exited; cannot certify worker cleanup without live ownership"
            )
        if stopped and worker is not None and not released:
            signal_group(worker, signal.SIGKILL)
    except (Exception,) as error:  # noqa: B013 -- still resume/reap the supervisor
        failures.append(error)
    finally:
        if stopped:
            try:
                resume_supervisor(process, channel)
            except (Exception,) as error:  # noqa: B013 -- retain independent failures
                failures.append(error)
        try:
            reap_supervisor(process, deadline)
        except (Exception,) as error:  # noqa: B013 -- retain independent reaping failure
            failures.append(error)
    if failures:
        primary, *secondary = failures
        for error in secondary:
            attach_failure(primary, "shadow emergency cleanup failed", error)
        raise primary


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
    if (
        not channel.messages
        or channel.payload
        or channel.messages[-1].get("status") not in TERMINAL
    ):
        raise OSError("missing complete shadow supervisor status")
    terminal = channel.messages[-1]
    status = terminal["status"]
    if supervisor_code != 0 and status != "runtime_failure":
        error = OSError(f"shadow supervisor exited {supervisor_code} after {status}")
        if stderr:
            error.add_note(f"shadow supervisor stderr: {limited_text(stderr, 8192)}")
        raise error
    if status == "timed_out":
        raise subprocess.TimeoutExpired(
            command,
            timeout,
            output=stdout.encode() if stdout is not None else None,
            stderr=stderr.encode() if stderr is not None else None,
        )
    if (
        status == "completed"
        and type(terminal.get("returncode")) is int
        and terminal["cleanup_complete"]
    ):
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
    if not all(hasattr(os, name) for name in ("waitid", "WNOWAIT", "WSTOPPED")):
        raise OSError("shadow supervision requires native waitid/WNOWAIT ownership")
    return subprocess.Popen(  # noqa: S603 -- reviewed operator-selected argv
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


def communicate_until(
    process: subprocess.Popen[str], channel: StatusChannel, deadline: float
) -> tuple[str | None, str | None]:
    while True:
        channel.collect()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise OSError("shadow supervisor did not finish within shutdown allowance")
        try:
            return process.communicate(timeout=min(POLL_INTERVAL, remaining))
        except (subprocess.TimeoutExpired,):  # noqa: B013 -- monitor protocol concurrently
            pass


@contextmanager
def deferred_cancellation() -> Iterator[Cancellation]:
    pending = Cancellation()
    if threading.current_thread() is not threading.main_thread():
        yield pending
        return
    previous = {
        sig: signal.signal(sig, pending.handle)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        yield pending
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def finish_cancellation(
    process: subprocess.Popen[str], channel: StatusChannel, primary: BaseException
) -> None:
    if channel.control is not None:
        channel.control.cancel()
    communicate_until(
        process, channel, time.monotonic() + TERM_GRACE + REAP_TIMEOUT + WAIT_MARGIN
    )
    channel.finish()
    if channel.messages and channel.messages[-1].get("status") == "runtime_failure":
        attach_failure(
            primary, "shadow supervisor failure", diagnostic_error(channel.messages[-1])
        )


def rescue_cancellation(
    process: subprocess.Popen[str], channel: StatusChannel, primary: BaseException
) -> None:
    try:
        emergency_stop(process, channel)
    except (Exception,) as error:  # noqa: B013 -- retain primary plus nested notes
        attach_failure(primary, "shadow child cleanup failed", error)


def propagate_cancellation(
    primary: BaseException,
    pending: Cancellation,
) -> None:
    if not pending.requested:
        return
    if isinstance(primary, (KeyboardInterrupt, SystemExit)):
        primary.add_note(
            "additional cancellation deferred until shadow cleanup completed"
        )
        return
    if pending.signum == signal.SIGTERM:
        raise SystemExit(128 + signal.SIGTERM) from primary
    raise KeyboardInterrupt("operator cancelled during shadow cleanup") from primary


def cancel_supervisor(
    process: subprocess.Popen[str], channel: StatusChannel, primary: BaseException
) -> None:
    completed = False
    with deferred_cancellation() as pending:
        try:
            finish_cancellation(process, channel, primary)
            completed = True
        except (KeyboardInterrupt, SystemExit) as error:
            attach_failure(primary, "shadow cancellation diagnostic", error)
            raise error from primary
        except (Exception,) as error:  # noqa: B013 -- operational diagnostic; caller retains primary
            attach_failure(primary, "shadow cancellation diagnostic", error)
        finally:
            if not completed:
                rescue_cancellation(process, channel, primary)
    propagate_cancellation(primary, pending)


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
        except (OSError,) as error:  # noqa: B013 -- close every descriptor
            failures.append(error)
    if failures:
        if primary is None:
            raise failures[0]
        for error in failures:
            attach_failure(primary, "shadow descriptor cleanup failed", error)


def run_supervised(
    command: Sequence[str], *, timeout: float, capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
    descriptors: set[int] = set()
    process = None
    primary = None
    channel = None
    control = None
    try:
        control_read, control_write = os.pipe()
        descriptors.update((control_read, control_write))
        status_read, status_write = os.pipe()
        descriptors.update((status_read, status_write))
        control = ControlChannel(control_write)
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
        channel.start(control)
        try:
            control.send(b"S")
            stdout, stderr = communicate_until(
                process,
                channel,
                time.monotonic() + timeout + TERM_GRACE + REAP_TIMEOUT + WAIT_MARGIN,
            )
            channel.finish()
            return result(
                command,
                timeout,
                channel,
                stdout,
                stderr,
                supervisor_code=process.returncode,
            )
        except (BaseException,) as error:  # noqa: B013 -- cancellation must complete cleanup
            cancel_supervisor(process, channel, error)
            raise
    except (BaseException,) as error:  # noqa: B013 -- retain this call's primary
        primary = error
        raise
    finally:
        cleanup_error = None
        if channel is not None:
            try:
                channel.stop()
            except (OSError,) as error:  # noqa: B013 -- preserve primary diagnostic
                if primary is None:
                    cleanup_error = error
                else:
                    attach_failure(
                        primary, "shadow status reader cleanup failed", error
                    )
        if control is not None:
            # Control closes only after cleanup, so EOF cannot release an identity mid-rescue.
            control.closed = True
        close_handles(descriptors, process, primary=primary or cleanup_error)
        if cleanup_error is not None:
            raise cleanup_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--admission-fd", type=int)
    parser.add_argument("--error-fd", type=int)
    parser.add_argument("--control-fd", type=int)
    parser.add_argument("--status-fd", type=int)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("shadow supervision requires a command")
    if args.bootstrap:
        return bootstrap(command, args.admission_fd, args.error_fd, args.deadline)
    return supervise(command, args.control_fd, args.status_fd, args.deadline)


if __name__ == "__main__":
    raise SystemExit(main())
