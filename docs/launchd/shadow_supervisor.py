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
PROTOCOL_VERSION = 4
TRUNCATED = " ... [truncated] ... "
TERMINAL = frozenset({"completed", "cancelled", "timed_out", "runtime_failure"})


@dataclass(frozen=True)
class Clock:
    now: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep


CLOCK = Clock()


def text_context(value: str, size: int) -> str:
    return (
        value[: (size + 1) // 2]
        + TRUNCATED
        + (value[-(size // 2) :] if size > 1 else "")
    )


def limited_text(value: str, budget: int) -> str:
    """Bound encoded JSON, retaining both the error prefix and its final path/context."""
    if len(json.dumps(value).encode()) <= budget:
        return value
    low, high = 0, len(value)
    while low < high:
        size = (low + high + 1) // 2
        candidate = text_context(value, size)
        if len(json.dumps(candidate).encode()) <= budget:
            low = size
        else:
            high = size - 1
    return text_context(value, low)


def encode_status(
    status: str, fields: dict[str, Any], *, kind: str = "shadow_supervision"
) -> bytes:
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
            dict(
                kind=kind,
                schema_version=PROTOCOL_VERSION,
                status=status,
                **fields,
            ),
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
    if primary is error:
        return
    for detail in exception_notes(error):
        note = f"{label}: {detail}"
        add_note_once(primary, note)


def add_note_once(error: BaseException, note: str) -> None:
    if note not in getattr(error, "__notes__", ()):
        error.add_note(note)


def signal_group(pid: int, signum: int) -> None:
    if pid <= 1:
        raise OSError("invalid owned worker group identity")
    with suppress(ProcessLookupError):
        os.killpg(pid, signum)


def group_exists(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except (ProcessLookupError,):  # noqa: B013
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


@dataclass(frozen=True)
class ShutdownBudget:
    started: float

    @property
    def term_until(self) -> float:
        return self.started + TERM_GRACE

    @property
    def deadline(self) -> float:
        return self.term_until + REAP_TIMEOUT

    @property
    def signal_until(self) -> float:
        return self.deadline - WAIT_MARGIN


@dataclass
class GroupShutdown:
    """Signal policy shared by normal supervision and parent-loss fallback."""

    pid: int
    budget: ShutdownBudget
    clock: Clock = CLOCK
    terminated: bool = False
    complete: bool = False
    denial: PermissionError | None = None

    def exists(self) -> bool:
        try:
            return group_exists(self.pid)
        except (PermissionError,) as error:  # noqa: B013
            self.denial = self.denial or error
            return True

    def signal_next(self) -> None:
        if self.clock.now() >= self.budget.term_until:
            signal_group(self.pid, signal.SIGKILL)
            self.complete = True
        elif not self.terminated:
            signal_group(self.pid, signal.SIGTERM)
            self.terminated = True

    def run(self, progress: Callable[[], None]) -> None:
        if self.complete:
            return
        while True:
            progress()
            if not self.exists():
                self.complete = True
                return
            if self.clock.now() >= self.budget.signal_until:
                break
            try:
                self.signal_next()
            except (PermissionError,) as error:  # noqa: B013
                self.denial = error
            if self.complete:
                return  # Collection has its own reserved final second.
            self.clock.sleep(POLL_INTERVAL)
        if self.denial is not None:
            raise self.denial
        raise OSError(
            "worker group termination was not certified before collection reserve"
        )


def stop_worker(
    process: subprocess.Popen[str],
    *,
    clock: Clock = CLOCK,
    started: float | None = None,
) -> int:
    """Compatibility helper for a still-owned, waitable group reservation."""
    budget = ShutdownBudget(clock.now() if started is None else started)
    GroupShutdown(process.pid, budget, clock).run(lambda: None)
    return wait_exit(process, budget.deadline, clock=clock)


def force_group(pid: int, deadline: float, *, clock: Clock = CLOCK) -> None:
    """Allow the outside guardian to reap a transient Darwin zombie; never ignore denial."""
    while not force_group_once(pid, deadline, clock=clock):
        clock.sleep(POLL_INTERVAL)


def force_group_once(pid: int, deadline: float, *, clock: Clock = CLOCK) -> bool:
    try:
        signal_group(pid, signal.SIGKILL)
        return True
    except (PermissionError,):  # noqa: B013
        if clock.now() >= deadline:
            raise
        return False


@dataclass
class Cancellation:
    requested: bool = False
    signum: int | None = None

    def handle(self, _signum: int, _frame: object) -> None:
        self.requested = True
        self.signum = _signum


def control_cancelled(control: int, selector: selectors.BaseSelector) -> bool:
    if not any(key.fd == control for key, _ in selector.select(0)):
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
                return "cancelled" if control_cancelled(control, selector) else "ready"
    return "cancelled"


def await_start(control: int, deadline: float, cancelled: Cancellation) -> str:
    return await_token(control, b"S", deadline, cancelled)


def monitor(
    worker: Worker,
    control: int,
    deadline: float,
    cancelled: Cancellation,
    *,
    clock: Clock = CLOCK,
) -> str:
    with selectors.DefaultSelector() as selector:
        selector.register(control, selectors.EVENT_READ)
        selector.register(worker.errors, selectors.EVENT_READ)
        while True:
            if (phase := worker_phase(worker, cancelled, deadline, clock)) is not None:
                return phase
            remaining = deadline - clock.now()
            for key, _ in selector.select(min(POLL_INTERVAL, remaining)):
                if key.fd != control:
                    continue
                if os.read(control, 1) not in (b"", b"C"):
                    raise OSError("unexpected parent-liveness data")
                worker.collect()
                worker.check_error()
                return (
                    "completed" if worker.events.returncode is not None else "cancelled"
                )


def worker_phase(
    worker: Worker, cancelled: Cancellation, deadline: float, clock: Clock
) -> str | None:
    worker.collect()
    if worker.events.returncode is not None:
        return "completed"
    worker.check_error()
    if cancelled.requested:
        return "cancelled"
    return "timed_out" if clock.now() >= deadline else None


def write_frame(descriptor: int, kind: str, event: str, **fields: Any) -> None:
    payload = memoryview(encode_status(event, fields, kind=kind))
    while payload:
        written = os.write(descriptor, payload)
        if not written:
            raise OSError("guardian channel write made no progress")
        payload = payload[written:]


def guardian_event(descriptor: int, event: str, **fields: Any) -> None:
    with suppress(BrokenPipeError):
        write_frame(descriptor, "shadow_guardian", event, **fields)


def guardian_error(descriptor: int, error: BaseException) -> None:
    guardian_event(
        descriptor,
        "error",
        detail=str(error),
        errno=getattr(error, "errno", None),
        notes=list(getattr(error, "__notes__", ())),
    )


@dataclass
class FrameStream:
    kind: str
    payload: bytes = b""
    total: int = 0
    eof: bool = False
    error: OSError | None = None

    def feed(self, chunk: bytes) -> list[dict[str, Any]]:
        self.total += len(chunk)
        if self.total > STATUS_LIMIT:
            self.error = self.error or OSError(f"{self.kind} channel exceeds limit")
            return []
        self.payload += chunk
        frames = []
        while b"\n" in self.payload:
            raw, self.payload = self.payload.split(b"\n", 1)
            try:
                value = json.loads(raw)
                if (
                    not isinstance(value, dict)
                    or value.get("kind") != self.kind
                    or type(value.get("schema_version")) is not int
                    or value["schema_version"] != PROTOCOL_VERSION
                ):
                    raise ValueError("invalid envelope")
                validate_diagnostic(value)
                frames.append(value)
            except (ValueError, OSError) as error:
                self.error = self.error or OSError(
                    f"malformed {self.kind} frame: {error}"
                )
        return frames

    def read(self, descriptor: int) -> list[dict[str, Any]]:
        os.set_blocking(descriptor, False)
        frames = []
        while not self.eof:
            try:
                chunk = os.read(descriptor, STATUS_LIMIT)
            except (BlockingIOError,):  # noqa: B013
                break
            if not chunk:
                self.eof = True
                if self.payload:
                    self.error = self.error or OSError(f"incomplete {self.kind} frame")
                break
            frames.extend(self.feed(chunk))
        return frames


@dataclass
class GuardianEvents:
    stream: FrameStream = field(default_factory=lambda: FrameStream("shadow_guardian"))
    ready: bool = False
    returncode: int | None = None
    error: OSError | None = None

    def collect(self, descriptor: int) -> None:
        for value in self.stream.read(descriptor):
            try:
                self.accept(value)
            except (OSError,) as error:  # noqa: B013
                self.error = self.error or error
        self.error = self.error or self.stream.error

    def accept(self, value: dict[str, Any]) -> None:
        match value["status"]:
            case "ready" if not self.ready and self.returncode is None:
                self.ready = True
            case "launcher_exit" if self.ready and self.returncode is None:
                if type(value.get("returncode")) is not int:
                    raise OSError(
                        "guardian launcher exit requires an integer returncode"
                    )
                self.returncode = value["returncode"]
            case "error":
                code, detail = value.get("errno"), value.get("detail")
                if (code is not None and type(code) is not int) or not isinstance(
                    detail, str
                ):
                    raise OSError("invalid guardian operational error")
                error = OSError(code, detail) if code is not None else OSError(detail)
                for note in value.get("notes", []):
                    error.add_note(note)
                if self.error is None:
                    self.error = error
                else:
                    attach_failure(self.error, "guardian failure", error)
            case _:
                raise OSError("invalid guardian lifecycle phase")


@dataclass
class OwnedLauncher:
    pid: int | None
    event_pipe: int
    returncode: int | None = None

    def __post_init__(self) -> None:
        if self.pid is None or self.pid <= 1:
            raise OSError("invalid owned launcher identity")

    def collect(self) -> None:
        if self.pid is None:
            return
        try:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
        except (ChildProcessError,) as error:  # noqa: B013
            self.pid = None  # An external reap ends ownership, even on failure.
            raise OSError("guardian lost waitable launcher ownership") from error
        if pid:
            self.pid = None  # Clear identity before any fallible reporting.
            self.returncode = os.waitstatus_to_exitcode(status)
            guardian_event(self.event_pipe, "launcher_exit", returncode=self.returncode)

    def wait(self, deadline: float, *, clock: Clock = CLOCK) -> None:
        while self.pid is not None:
            self.collect()
            if self.pid is None:
                return
            if clock.now() >= deadline:
                raise OSError(
                    "guardian could not reap launcher before shutdown deadline"
                )
            clock.sleep(POLL_INTERVAL)

    def stop_blocked(self, deadline: float) -> None:
        if self.pid is not None:
            with suppress(ProcessLookupError):
                os.kill(self.pid, signal.SIGKILL)
            self.wait(deadline)


@dataclass
class Guardian:
    launcher: OwnedLauncher
    parent: int
    admission: int
    reserved: bool = False
    admitted: bool = False
    budget: ShutdownBudget | None = None
    shutdown: GroupShutdown | None = None
    failed: bool = False

    def run(self) -> None:
        commands = FrameStream("shadow_guardian_control")
        with selectors.DefaultSelector() as selector:
            selector.register(self.parent, selectors.EVENT_READ)
            while True:
                try:
                    self.launcher.collect()
                except (OSError,) as error:  # noqa: B013
                    self.report_failure(error)
                for value in commands.read(self.parent):
                    try:
                        if self.command(value):
                            return
                    except (OSError,) as error:  # noqa: B013
                        self.report_failure(error)
                if commands.error is not None:
                    self.report_failure(commands.error)
                    commands.error = None
                if commands.eof:
                    self.fallback()
                    return
                selector.select(POLL_INTERVAL)

    def command(self, value: dict[str, Any]) -> bool:
        match value["status"]:
            case "admit" if (
                not self.failed and not self.admitted and self.budget is None
            ):
                os.write(self.admission, b"G")
                os.close(self.admission)
                self.admission = -1
                self.admitted = True
            case "shutdown" if self.budget is None:
                started = value.get("started")
                if type(started) not in (float, int) or not float(
                    "-inf"
                ) < started < float("inf"):
                    raise OSError("invalid guardian shutdown budget")
                self.budget = ShutdownBudget(float(started))
            case "release" if (
                self.launcher.returncode is not None
                and value.get("group_cleaned") is True
            ):
                return True
            case _:
                raise OSError("invalid guardian control phase")
        return False

    def report_failure(self, error: OSError) -> None:
        if not self.failed:
            guardian_error(self.launcher.event_pipe, error)
            self.failed = True

    def fallback(self) -> None:
        if self.shutdown is None:
            self.budget = self.budget or ShutdownBudget(CLOCK.now())
            self.shutdown = GroupShutdown(os.getpid(), self.budget)
        if self.reserved:
            self.shutdown.run(self.launcher.collect)
            self.launcher.wait(self.shutdown.budget.deadline)
        else:
            self.launcher.stop_blocked(self.shutdown.budget.deadline)


def launcher_main(
    command: Sequence[str],
    read: int,
    write: int,
    parent: int,
    events: int,
    deadline: float,
) -> None:
    """A forked launcher never enters guardian-owned exception handling."""
    code = 127
    try:
        os.close(write)
        os.close(parent)
        code = execute_admitted(command, read, events, deadline)
    except (BaseException,) as error:  # noqa: B013
        with suppress(Exception):
            guardian_error(events, error)
        raise  # The unconditional _exit below remains the fork-child boundary.
    finally:
        os._exit(code)


def bootstrap(
    command: Sequence[str], admission: int, error_pipe: int, deadline: float
) -> int:
    try:
        read, write = os.pipe()
        try:
            child = os.fork()
        except (BaseException,):  # noqa: B013
            os.close(read)
            os.close(write)
            raise
    except (OSError,) as error:  # noqa: B013
        guardian_error(error_pipe, error)
        return 127
    if child == 0:
        launcher_main(command, read, write, admission, error_pipe, deadline)
        raise AssertionError("forked launcher returned from os._exit")
    guardian = Guardian(OwnedLauncher(child, error_pipe), admission, write)
    try:
        os.close(read)
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, signal.SIG_IGN)
        os.setpgid(0, os.getpgid(os.getppid()))
        guardian.reserved = True
        guardian_event(error_pipe, "ready")
        guardian.run()
        return 0
    except (BaseException,) as error:  # noqa: B013
        recover_guardian(guardian, error, error_pipe)
        raise
    finally:
        for descriptor in (guardian.admission, admission, error_pipe):
            if descriptor >= 0:
                with suppress(OSError):
                    os.close(descriptor)


def recover_guardian(guardian: Guardian, error: BaseException, event_pipe: int) -> None:
    with suppress(Exception):
        guardian_error(event_pipe, error)
    try:
        if guardian.reserved:
            # Retain the supervisor's ordinary shutdown authority until EOF.
            guardian.failed = True
            guardian.run()
        else:
            guardian.fallback()  # Only its still-blocked direct child exists.
    except (Exception,) as secondary:  # noqa: B013
        attach_failure(error, "guardian recovery failed", secondary)


def execute_admitted(
    command: Sequence[str], admission: int, error_pipe: int, deadline: float
) -> int:
    try:
        for name in ("SIGINT", "SIGTERM", "SIGPIPE", "SIGXFZ", "SIGXFSZ"):
            if (signum := getattr(signal, name, None)) is not None:
                signal.signal(signum, signal.SIG_DFL)
        with selectors.DefaultSelector() as selector:
            selector.register(admission, selectors.EVENT_READ)
            if (
                not selector.select(max(0, deadline - CLOCK.now()))
                or os.read(admission, 1) != b"G"
            ):
                return 125
        os.close(admission)
        os.set_inheritable(error_pipe, False)  # noqa: FBT003
        os.execvp(command[0], list(command))  # noqa: S606
    except (OSError,) as error:  # noqa: B013
        guardian_error(error_pipe, error)
        return 127
    return 127


@dataclass
class Worker:
    process: subprocess.Popen[str]
    admission: int
    errors: int
    cleaned: bool = False
    released: bool = False
    admitted: bool = False
    release_sent: bool = False
    events: GuardianEvents = field(default_factory=GuardianEvents)
    shutdown: GroupShutdown | None = None
    stop_error: Exception | None = None

    def command(self, event: str, **fields: Any) -> None:
        write_frame(self.admission, "shadow_guardian_control", event, **fields)

    def admit(self) -> None:
        self.command("admit")
        self.admitted = True

    def collect(self) -> None:
        self.events.collect(self.errors)

    def check_error(self) -> None:
        if self.events.error is not None:
            raise self.events.error
        if not self.release_sent and observe_exit(self.process) is not None:
            raise OSError(
                "guardian exited without a valid cleanup release; worker outcome unknown"
            )

    def ready(self, control: int, deadline: float, cancelled: Cancellation) -> str:
        with selectors.DefaultSelector() as selector:
            selector.register(self.errors, selectors.EVENT_READ)
            selector.register(control, selectors.EVENT_READ)
            while True:
                self.collect()
                self.check_error()
                if self.events.ready:
                    return "ready"
                if cancelled.requested or control_cancelled(control, selector):
                    return "cancelled"
                remaining = deadline - CLOCK.now()
                if remaining <= 0:
                    return "timed_out"
                selector.select(min(POLL_INTERVAL, remaining))

    def stop(self) -> int:
        if self.stop_error is not None:
            raise self.stop_error
        if self.cleaned:
            return self.worker_code()
        try:
            return self.stop_owned()
        except (Exception,) as error:  # noqa: B013
            self.stop_error = error
            try:
                if (
                    self.shutdown is not None
                    and self.shutdown.complete
                    and observe_exit(self.process) is not None
                ):
                    self.cleaned = True  # Unknown result still allows safe collection.
            except (OSError,) as secondary:  # noqa: B013
                attach_failure(error, "guardian collection check failed", secondary)
            raise

    def stop_owned(self) -> int:
        if self.shutdown is None:
            self.shutdown = GroupShutdown(self.process.pid, ShutdownBudget(CLOCK.now()))
            try:
                self.command("shutdown", started=self.shutdown.budget.started)
            except (OSError,) as error:  # noqa: B013
                self.events.error = self.events.error or error
        self.await_reservation(self.shutdown.budget.term_until)
        self.shutdown.run(self.collect_for_cleanup)
        self.wait_launcher(self.shutdown.budget.deadline)
        self.command("release", group_cleaned=True)
        self.release_sent = True
        wait_exit(self.process, self.shutdown.budget.deadline)
        self.cleaned = True
        return self.worker_code()

    def worker_code(self) -> int:
        if self.events.returncode is None:
            raise OSError("guardian did not report a launcher exit")
        return self.events.returncode

    def collect_for_cleanup(self) -> None:
        try:
            self.collect()
        except (OSError,) as error:  # noqa: B013
            self.events.error = self.events.error or error

    def await_reservation(self, deadline: float) -> None:
        if self.events.ready:
            return
        with selectors.DefaultSelector() as selector:
            selector.register(self.errors, selectors.EVENT_READ)
            while not self.events.ready:
                self.collect()
                if self.events.ready or observe_exit(self.process) is not None:
                    return
                remaining = deadline - CLOCK.now()
                if remaining <= 0:
                    return
                selector.select(min(POLL_INTERVAL, remaining))

    def wait_launcher(self, deadline: float) -> None:
        if self.events.returncode is not None:
            return
        with selectors.DefaultSelector() as selector:
            selector.register(self.errors, selectors.EVENT_READ)
            while self.events.returncode is None:
                self.collect()
                if self.events.returncode is not None:
                    return
                if observe_exit(self.process) is not None:
                    self.check_error()
                remaining = deadline - CLOCK.now()
                if remaining <= 0:
                    raise OSError(
                        "guardian launcher collection exceeded shutdown deadline"
                    )
                selector.select(min(POLL_INTERVAL, remaining))

    def close_admission(self) -> None:
        if self.admission >= 0:
            os.close(self.admission)
            self.admission = -1

    def exec_error(self) -> None:
        self.collect()
        self.check_error()


def start_worker(command: Sequence[str], deadline: float) -> Worker:
    allocated: set[int] = set()
    process = None
    try:
        admission_read, admission_write = os.pipe()
        allocated.update((admission_read, admission_write))
        error_read, error_write = os.pipe()
        allocated.update((error_read, error_write))
        process = subprocess.Popen(  # noqa: S603
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
    except (BaseException,) as error:  # noqa: B013
        if process is not None:
            try:
                process.kill()
                process.wait(timeout=REAP_TIMEOUT)
            except (Exception,) as secondary:  # noqa: B013
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
    readiness = worker.ready(control, deadline, cancelled)
    if readiness != "ready":
        return readiness
    send_status(status, "worker_ready", worker_pid=worker.process.pid)
    admitted = await_token(control, b"A", deadline, cancelled)
    if admitted != "ready":
        return admitted
    with selectors.DefaultSelector() as selector:
        selector.register(control, selectors.EVENT_READ)
        if cancelled.requested or control_cancelled(control, selector):
            return "cancelled"
    if CLOCK.now() >= deadline:
        return "timed_out"
    worker.admit()
    return monitor(worker, control, deadline, cancelled)


def finish_worker(worker: Worker, control: int, status: int, outcome: str) -> None:
    code = worker.stop()
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
    if worker is not None and not cleaned and worker.stop_error is None:
        try:
            worker.stop()
            cleaned = True
        except (Exception,) as secondary:  # noqa: B013
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
    except (Exception,) as error:  # noqa: B013
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
            except (BrokenPipeError,):  # noqa: B013
                return False
            return True

    def cancel(self) -> None:
        with self.lock:
            if not self.cancelled:
                self.cancelled = True
                self.send(b"C")

    def close(self) -> None:
        if not self.lock.acquire(timeout=WAIT_MARGIN):
            raise OSError("shadow control sender still running; descriptor preserved")
        try:
            if not self.closed:
                self.closed = True
                os.close(self.descriptor)
        finally:
            self.lock.release()


def validate_diagnostic(value: dict[str, Any]) -> None:
    if not isinstance(value.get("status"), str) or not isinstance(
        value.get("detail", ""), str
    ):
        raise OSError("invalid shadow supervisor diagnostic fields")
    notes = value.get("notes", [])
    if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
        raise OSError("invalid shadow supervisor diagnostic fields")


def diagnostic_error(value: dict[str, Any], default: str | None = None) -> OSError:
    if default is None:
        default = {
            "cancelled": "worker cancelled",
            "timed_out": "worker execution deadline exceeded",
            "runtime_failure": "supervision failed",
        }.get(value["status"], "supervisor did not exit successfully")
    error = OSError(
        f"shadow supervisor {value['status']}: {value.get('detail', default)}"
    )
    for note in value.get("notes", []):
        error.add_note(str(note))
    return error


def attach_stderr(error: BaseException, stderr: str | bytes | None) -> None:
    if stderr:
        text = decoded_output(stderr) or ""
        add_note_once(error, f"shadow supervisor stderr: {limited_text(text, 8192)}")


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
    stderr: str | bytes | None = None
    stopping: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def start(self, control: ControlChannel) -> None:
        self.control = control
        thread = threading.Thread(
            target=self.read_loop, name="shadow-status", daemon=True
        )
        try:
            thread.start()
        except (RuntimeError,) as error:  # noqa: B013
            raise OSError(f"cannot start shadow status reader: {error}") from error
        self.thread = thread

    def read_loop(self) -> None:
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self.descriptor, selectors.EVENT_READ)
                while not self.stopping.is_set():
                    if selector.select(POLL_INTERVAL) and not self.drain():
                        return
        except (Exception,) as error:  # noqa: B013
            with self.lock:
                self.error = self.error or error

    def drain(self) -> bool:
        os.set_blocking(self.descriptor, False)
        while not self.stopping.is_set():
            try:
                chunk = os.read(self.descriptor, STATUS_LIMIT)
            except (BlockingIOError,):  # noqa: B013
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
        if self.total <= STATUS_LIMIT:
            try:
                self.parse(chunk)
            except (Exception,) as error:  # noqa: B013
                self.error = error

    def parse(self, chunk: bytes) -> None:
        self.payload += chunk
        while b"\n" in self.payload:
            raw, self.payload = self.payload.split(b"\n", 1)
            try:
                self.parse_message(raw)
            except (OSError,) as error:  # noqa: B013
                self.error = self.error or error

    def parse_message(self, raw: bytes) -> None:
        try:
            value = json.loads(raw)
        except (ValueError,) as error:  # noqa: B013
            raise OSError("malformed shadow supervisor status") from error
        if (
            not isinstance(value, dict)
            or value.get("kind") != "shadow_supervision"
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != PROTOCOL_VERSION
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
        if self.control is not None and self.error is None:
            self.control.send(b"A")

    def release_identity(self, value: dict[str, Any]) -> None:
        if not value["cleanup_complete"]:
            raise diagnostic_error(value, "cleanup incomplete")
        self.released = True
        if self.control is not None:
            self.control.send(b"R")

    def collect(self, *, check_errors: bool = True) -> None:
        if self.thread is None:
            self.drain()
        with self.lock:
            if check_errors and self.error is not None:
                raise self.error

    def finish(self, *, check_errors: bool = True) -> None:
        if self.thread is not None:
            self.thread.join(timeout=WAIT_MARGIN)
            if self.thread.is_alive():
                self.stopping.set()
                self.thread.join(timeout=WAIT_MARGIN)
                if self.thread.is_alive():
                    raise OSError("shadow status reader did not stop")
        self.collect(check_errors=check_errors)

    def stop(self) -> None:
        self.stopping.set()
        if self.thread is not None:
            self.thread.join(timeout=WAIT_MARGIN)
            if self.thread.is_alive():
                raise OSError("shadow status reader left running")


def stopped_supervisor(
    process: subprocess.Popen[str], deadline: float, *, clock: Clock = CLOCK
) -> bool:
    """Only our still-waitable, stopped child can vouch for its unreaped worker."""
    if process.returncode is not None or observe_exit(process) is not None:
        return False
    os.kill(process.pid, signal.SIGSTOP)
    while clock.now() < deadline:
        info = wait_notification(process, os.WEXITED | os.WSTOPPED)
        if info is not None:
            return info.si_code == os.CLD_STOPPED
        clock.sleep(POLL_INTERVAL)
    raise OSError("could not verify stopped shadow supervisor")


def resume_supervisor(process: subprocess.Popen[str], channel: StatusChannel) -> None:
    primary = None
    try:
        if channel.control is not None:
            channel.control.cancel()
            channel.control.send(b"R")
    except (OSError,) as error:  # noqa: B013
        primary = error
    try:
        os.kill(process.pid, signal.SIGCONT)
    except (OSError,) as error:  # noqa: B013
        if primary is None:
            raise
        attach_failure(primary, "supervisor resume failed", error)
    if primary is not None:
        raise primary


def reap_supervisor(
    process: subprocess.Popen[str], deadline: float, *, clock: Clock = CLOCK
) -> None:
    try:
        process.communicate(timeout=max(0, deadline - clock.now()))
    except (Exception,) as error:  # noqa: B013
        # Popen.kill safely handles a direct child that exited during communicate.
        try:
            process.kill()
        except (Exception,) as secondary:  # noqa: B013
            attach_failure(error, "supervisor kill failed", secondary)
        try:
            process.wait(timeout=max(0.01, deadline - clock.now()))
        except (Exception,) as secondary:  # noqa: B013
            attach_failure(error, "supervisor left unreaped", secondary)
        raise


def emergency_stop(process: subprocess.Popen[str], channel: StatusChannel) -> None:
    deadline = CLOCK.now() + REAP_TIMEOUT
    signal_deadline = deadline - WAIT_MARGIN
    failures: list[Exception] = []
    stopped = False
    try:
        stopped = stopped_supervisor(process, signal_deadline)
        with channel.lock:
            worker, released = channel.owned_pid, channel.released
        if not stopped and not released:
            raise OSError(
                "supervisor exited; cannot certify worker cleanup without live ownership"
            )
        if stopped and worker is not None and not released:
            if CLOCK.now() >= signal_deadline:
                raise OSError(
                    "emergency signal budget exhausted; collection reserve retained"
                )
            force_group(worker, signal_deadline)
    except (Exception,) as error:  # noqa: B013
        failures.append(error)
    finally:
        if process.returncode is None:
            try:
                resume_supervisor(process, channel)
            except (Exception,) as error:  # noqa: B013
                failures.append(error)
        try:
            reap_supervisor(process, deadline)
        except (Exception,) as error:  # noqa: B013
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
    try:
        return terminal_result(
            command, timeout, channel, stdout, stderr, supervisor_code
        )
    except (OSError,) as error:  # noqa: B013
        attach_stderr(error, stderr)
        raise


def terminal_result(
    command: Sequence[str],
    timeout: float,
    channel: StatusChannel,
    stdout: str | None,
    stderr: str | None,
    supervisor_code: int | None,
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
        raise diagnostic_error(
            {**terminal, "status": f"exited {supervisor_code} after {status}"}
        )
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
    raise diagnostic_error(terminal)


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
    return subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--control-fd",
            str(control_read),
            "--status-fd",
            str(status_write),
            "--deadline",
            repr(CLOCK.now() + timeout),
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
    process: subprocess.Popen[str],
    channel: StatusChannel,
    deadline: float,
    *,
    check_errors: bool = True,
    clock: Clock = CLOCK,
) -> tuple[str | None, str | None]:
    while True:
        channel.collect(check_errors=check_errors)
        remaining = deadline - clock.now()
        if remaining <= 0:
            raise OSError("shadow supervisor did not finish within shutdown allowance")
        try:
            stdout, stderr = process.communicate(timeout=min(POLL_INTERVAL, remaining))
            channel.stderr = stderr
            return stdout, stderr
        except (subprocess.TimeoutExpired,) as error:  # noqa: B013
            channel.stderr = error.stderr


def decoded_output(output: str | bytes | None) -> str | None:
    return output.decode(errors="replace") if isinstance(output, bytes) else output


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
        process,
        channel,
        CLOCK.now() + TERM_GRACE + REAP_TIMEOUT + WAIT_MARGIN,
        check_errors=False,
    )
    channel.finish(check_errors=False)
    if not channel.released:
        raise OSError("shadow cleanup lacks a valid terminal acknowledgment")
    if channel.messages and channel.messages[-1].get("status") == "runtime_failure":
        attach_failure(
            primary, "shadow supervisor failure", diagnostic_error(channel.messages[-1])
        )


def rescue_cancellation(
    process: subprocess.Popen[str], channel: StatusChannel, primary: BaseException
) -> None:
    try:
        emergency_stop(process, channel)
    except (Exception,) as error:  # noqa: B013
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
        except (Exception,) as error:  # noqa: B013
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
        except (OSError,) as error:  # noqa: B013
            failures.append(error)
    if failures:
        if primary is None:
            raise failures[0]
        for error in failures:
            attach_failure(primary, "shadow descriptor cleanup failed", error)


def stop_reader(channel: StatusChannel | None) -> OSError | None:
    if channel is not None:
        try:
            channel.stop()
        except (OSError,) as error:  # noqa: B013
            return error
    return None


def close_control(
    channel: StatusChannel | None,
    control: ControlChannel | None,
    descriptors: set[int],
    reader_error: OSError | None,
) -> OSError | None:
    if control is None:
        return None
    descriptors.discard(control.descriptor)
    if reader_error is not None:
        if channel is not None:
            descriptors.discard(channel.descriptor)
        return (
            None  # Still reachable by the failed reader; never race descriptor reuse.
        )
    try:
        control.close()
    except (OSError,) as error:  # noqa: B013
        return error
    return None


def close_supervision(
    descriptors: set[int],
    process: subprocess.Popen[str] | None,
    channel: StatusChannel | None,
    control: ControlChannel | None,
    primary: BaseException | None,
) -> None:
    reader_error = stop_reader(channel)
    control_error = close_control(channel, control, descriptors, reader_error)
    errors = [
        ("shadow status reader cleanup failed", reader_error),
        ("shadow control cleanup failed", control_error),
    ]
    first = primary or reader_error or control_error
    for label, error in errors:
        if first is not None and error is not None:
            attach_failure(first, label, error)
    close_handles(descriptors, process, primary=first)
    if primary is None and first is not None:
        raise first


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
        try:
            channel.start(control)
            control.send(b"S")
            stdout, stderr = communicate_until(
                process,
                channel,
                CLOCK.now() + timeout + TERM_GRACE + REAP_TIMEOUT + WAIT_MARGIN,
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
        except (BaseException,) as error:  # noqa: B013
            if process.returncode is None:
                cancel_supervisor(process, channel, error)
            elif not channel.released:
                error.add_note(
                    "supervisor exited; cannot certify worker cleanup without live ownership"
                )
            attach_stderr(error, channel.stderr)
            raise
    except (BaseException,) as error:  # noqa: B013
        primary = error
        raise
    finally:
        close_supervision(descriptors, process, channel, control, primary)


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
    names = (
        ("admission_fd", "error_fd") if args.bootstrap else ("control_fd", "status_fd")
    )
    for name in names:
        if getattr(args, name) is None or getattr(args, name) < 0:
            parser.error(
                f"{'bootstrap' if args.bootstrap else 'supervisor'} requires --{name.replace('_', '-')}"
            )
    if args.bootstrap:
        return bootstrap(command, args.admission_fd, args.error_fd, args.deadline)
    return supervise(command, args.control_fd, args.status_fd, args.deadline)


if __name__ == "__main__":
    raise SystemExit(main())
