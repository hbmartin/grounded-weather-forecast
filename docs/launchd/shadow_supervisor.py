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
LIFECYCLE_LIMIT = 8 * 1024
GUARDIAN_DIAGNOSTICS = 48 * 1024
LAUNCHER_DIAGNOSTICS = 8 * 1024
PROTOCOL_VERSION = 5
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
    status: str,
    fields: dict[str, Any],
    *,
    kind: str = "shadow_supervision",
    budget: int = STATUS_LIMIT - LIFECYCLE_LIMIT,
) -> bytes:
    fields = dict(fields)
    if "detail" in fields:
        fields["detail"] = limited_text(str(fields["detail"]), budget // 2)
    for name in ("strerror", "filename", "filename2"):
        if isinstance(fields.get(name), str):
            fields[name] = limited_text(fields[name], budget // 16)
    if "notes" in fields:
        notes = [str(note) for note in fields["notes"]]
        if len(notes) > 32:
            notes = [*notes[:15], TRUNCATED, *notes[-16:]]
        fields["notes"] = [
            limited_text(note, (budget // 8) // max(1, len(notes))) for note in notes
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
    if len(payload) > budget:
        raise OSError("shadow supervisor status exceeds its encoded budget")
    return payload


def send_status(descriptor: int, status: str, **fields: Any) -> None:
    with suppress(BrokenPipeError):
        write_frame(descriptor, "shadow_supervision", status, **fields)


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
    process: subprocess.Popen[str], deadline: float, *, clock: Clock | None = None
) -> int:
    clock = clock or CLOCK
    while (code := observe_exit(process)) is None:
        if clock.now() >= deadline:
            raise OSError("owned shadow worker did not exit after forced termination")
        clock.sleep(POLL_INTERVAL)
    return code


@dataclass(frozen=True)
class ShutdownBudget:
    started: float
    grace: float = TERM_GRACE

    @property
    def term_until(self) -> float:
        return self.started + self.grace

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
    clock: Clock = field(default_factory=lambda: CLOCK)
    terminated: bool = False
    killed: bool = False
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
            if not self.killed:
                signal_group(self.pid, signal.SIGKILL)
                self.killed = True
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
            self.clock.sleep(POLL_INTERVAL)
        if self.denial is not None:
            raise self.denial
        raise OSError(
            "worker group termination was not certified before collection reserve"
        )


def force_group(
    pid: int,
    deadline: float,
    *,
    clock: Clock | None = None,
    progress: Callable[[], None] = lambda: None,
) -> None:
    """Force-only policy; deadline ends signaling, reserving a final collection second."""
    clock = clock or CLOCK
    budget = ShutdownBudget(deadline - REAP_TIMEOUT + WAIT_MARGIN, grace=0)
    GroupShutdown(pid, budget, clock).run(progress)


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


@contextmanager
def control_selector(
    control: int, selector: selectors.BaseSelector | None
) -> Iterator[selectors.BaseSelector]:
    if selector is not None:
        yield selector
    else:
        with selectors.DefaultSelector() as owned:
            owned.register(control, selectors.EVENT_READ)
            yield owned


def await_token(
    control: int,
    token: bytes,
    deadline: float,
    cancelled: Cancellation,
    *,
    clock: Clock | None = None,
    selector: selectors.BaseSelector | None = None,
) -> str:
    clock = clock or CLOCK
    with control_selector(control, selector) as selected:
        while not cancelled.requested:
            remaining = deadline - clock.now()
            if remaining <= 0:
                return "timed_out"
            if selected.select(min(POLL_INTERVAL, remaining)):
                return receive_token(control, token, cancelled, selected)
    return "cancelled"


def receive_token(
    control: int,
    token: bytes,
    cancelled: Cancellation,
    selector: selectors.BaseSelector,
) -> str:
    received = os.read(control, 1)
    if received in (b"", b"C") or cancelled.requested:
        return "cancelled"
    if received != token:
        raise OSError("unexpected shadow admission token")
    return "cancelled" if control_cancelled(control, selector) else "ready"


def await_start(
    control: int,
    deadline: float,
    cancelled: Cancellation,
    *,
    clock: Clock | None = None,
) -> str:
    return await_token(control, b"S", deadline, cancelled, clock=clock)


def monitor(
    worker: Worker,
    control: int,
    deadline: float,
    cancelled: Cancellation,
    *,
    clock: Clock | None = None,
) -> str:
    clock = clock or CLOCK
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


def write_frame(
    descriptor: int,
    kind: str,
    event: str,
    *,
    budget: int = STATUS_LIMIT - LIFECYCLE_LIMIT,
    **fields: Any,
) -> int:
    payload = memoryview(encode_status(event, fields, kind=kind, budget=budget))
    size = len(payload)
    while payload:
        written = os.write(descriptor, payload)
        if not written:
            raise OSError("guardian channel write made no progress")
        payload = payload[written:]
    return size


def guardian_event(descriptor: int, event: str, **fields: Any) -> int:
    with suppress(BrokenPipeError):
        return write_frame(descriptor, "shadow_guardian", event, **fields)
    return 0


def guardian_error(
    descriptor: int,
    error: BaseException,
    *,
    budget: int = LAUNCHER_DIAGNOSTICS,
) -> int:
    return guardian_event(
        descriptor,
        "error",
        detail=str(error),
        errno=getattr(error, "errno", None),
        strerror=getattr(error, "strerror", None),
        filename=str(error.filename)
        if isinstance(error, OSError) and error.filename is not None
        else None,
        filename2=str(error.filename2)
        if isinstance(error, OSError) and error.filename2 is not None
        else None,
        notes=list(getattr(error, "__notes__", ())),
        budget=budget,
    )


@dataclass
class FrameStream:
    kind: str
    payload: bytes = b""
    total: int = 0
    eof: bool = False
    error: OSError | None = None
    diagnostic_bytes: int = 0
    lifecycle_bytes: int = 0
    discarding: bool = False
    configured: int | None = None
    pending: list[dict[str, Any]] = field(default_factory=list)

    def feed(self, chunk: bytes) -> list[dict[str, Any]]:
        self.total += len(chunk)
        if self.total > STATUS_LIMIT:
            self.error = self.error or OSError(f"{self.kind} channel exceeds limit")
        frames = []
        parts = chunk.split(b"\n")
        for index, part in enumerate(parts):
            ended = index < len(parts) - 1
            if (value := self.feed_piece(part, ended=ended)) is not None:
                frames.append(value)
        return frames

    def feed_piece(self, part: bytes, *, ended: bool) -> dict[str, Any] | None:
        if self.discarding or len(self.payload) + len(part) > STATUS_LIMIT:
            self.error = self.error or OSError(f"{self.kind} frame exceeds limit")
            self.payload = b""
            self.discarding = not ended
            return None
        self.payload += part
        if not ended:
            return None
        raw, self.payload = self.payload, b""
        try:
            return self.parse_frame(raw)
        except (ValueError, OSError, RecursionError) as error:
            label = (
                "shadow supervisor status"
                if self.kind == "shadow_supervision"
                else f"{self.kind} frame"
            )
            self.error = self.error or OSError(f"malformed {label}: {error}")
        return None

    def parse_frame(self, raw: bytes) -> dict[str, Any] | None:
        value = json.loads(raw)
        if (
            not isinstance(value, dict)
            or value.get("kind") != self.kind
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != PROTOCOL_VERSION
        ):
            raise ValueError("invalid envelope")
        validate_diagnostic(value)
        if value.get("detail") or value.get("notes"):
            self.diagnostic_bytes += len(raw) + 1
            if self.diagnostic_bytes > STATUS_LIMIT - LIFECYCLE_LIMIT:
                self.error = self.error or OSError(
                    f"{self.kind} diagnostic budget exceeded"
                )
                if value["status"] == "error":
                    return None
                value = {
                    k: v
                    for k, v in value.items()
                    if k not in {"detail", "notes", "strerror", "filename", "filename2"}
                }
                value["detail"] = "diagnostics omitted after channel budget exhaustion"
            else:
                return value
        self.lifecycle_bytes += len(json.dumps(value).encode()) + 1
        if self.lifecycle_bytes > LIFECYCLE_LIMIT:
            raise OSError(f"{self.kind} lifecycle budget exceeded")
        return value

    def read(self, descriptor: int) -> list[dict[str, Any]]:
        if self.configured is None:
            os.set_blocking(descriptor, False)
            self.configured = descriptor
        elif self.configured != descriptor:
            raise OSError("frame stream cannot change its owned descriptor")
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
    bootstrap_ready: bool = False
    startup_clean: bool = False
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
            case "bootstrap_ready" if not self.bootstrap_ready and not self.ready:
                self.bootstrap_ready = True
            case "startup_stopped" if (
                not self.ready and value.get("cleanup_complete") is True
            ):
                self.startup_clean = True
            case "ready" if not self.ready and self.returncode is None:
                self.ready = True
            case "launcher_exit" if self.ready and self.returncode is None:
                if type(value.get("returncode")) is not int:
                    raise OSError(
                        "guardian launcher exit requires an integer returncode"
                    )
                self.returncode = value["returncode"]
            case "error":
                error = guardian_operational_error(value)
                if self.error is None:
                    self.error = error
                else:
                    attach_failure(self.error, "guardian failure", error)
            case _:
                raise OSError("invalid guardian lifecycle phase")


def guardian_operational_error(value: dict[str, Any]) -> OSError:
    code, detail = value.get("errno"), value.get("detail")
    if (code is not None and type(code) is not int) or not isinstance(detail, str):
        raise OSError("invalid guardian operational error")
    if code is None:
        error = OSError(detail)
    else:
        message = value.get("strerror")
        if not isinstance(message, str):
            message = detail
        paths = (value.get("filename"), value.get("filename2"))
        if any(path is not None and not isinstance(path, str) for path in paths):
            raise OSError("invalid guardian error filename")
        error = OSError(code, message, paths[0], None, paths[1])
    for note in value.get("notes", []):
        error.add_note(note)
    return error


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

    def wait(self, deadline: float, *, clock: Clock | None = None) -> None:
        clock = clock or CLOCK
        while self.pid is not None:
            self.collect()
            if self.pid is None:
                return
            if clock.now() >= deadline:
                raise OSError(
                    "guardian could not reap launcher before shutdown deadline"
                )
            clock.sleep(POLL_INTERVAL)

    def stop_blocked(self, deadline: float, *, clock: Clock | None = None) -> None:
        if self.pid is not None:
            with suppress(ProcessLookupError):
                os.kill(self.pid, signal.SIGKILL)
            self.wait(deadline, clock=clock)


@dataclass
class Guardian:
    launcher: OwnedLauncher
    parent: int
    admission: int
    reserved: bool = False
    admitted: bool = False
    budget: ShutdownBudget | None = None
    failed: bool = False
    clock: Clock = field(default_factory=lambda: CLOCK)
    diagnostic_remaining: int = GUARDIAN_DIAGNOSTICS
    rescue: GroupShutdown | None = None
    commands: FrameStream = field(
        default_factory=lambda: FrameStream("shadow_guardian_control")
    )

    def run(self) -> None:
        commands = self.commands
        with selectors.DefaultSelector() as selector:
            selector.register(self.parent, selectors.EVENT_READ)
            while True:
                try:
                    self.launcher.collect()
                except (OSError,) as error:  # noqa: B013
                    self.report_failure(error)
                values, commands.pending = commands.pending, []
                values.extend(commands.read(self.parent))
                if commands.error is not None:
                    self.report_failure(commands.error)
                    commands.error = None
                for value in values:
                    try:
                        if self.command(value):
                            return
                    except (OSError,) as error:  # noqa: B013
                        self.report_failure(error)
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
            self.report_diagnostic(error)
            self.failed = True

    def report_diagnostic(self, error: BaseException) -> None:
        if self.diagnostic_remaining >= 1024:
            used = guardian_error(
                self.launcher.event_pipe, error, budget=self.diagnostic_remaining
            )
            self.diagnostic_remaining -= used

    def fallback(self) -> None:
        if self.rescue is None:
            self.rescue = GroupShutdown(
                os.getpid(), ShutdownBudget(self.clock.now(), grace=0), self.clock
            )
        if self.reserved:
            self.rescue.run(self.launcher.collect)
            self.launcher.wait(self.rescue.budget.deadline, clock=self.clock)
        else:
            self.launcher.stop_blocked(self.rescue.budget.deadline, clock=self.clock)


def authorize_fork(
    admission: int,
    events: int,
    deadline: float,
    clock: Clock,
    stream: FrameStream,
) -> bool:
    """No child may exist until the supervisor records and sends reserve."""
    guardian_event(events, "bootstrap_ready")
    with selectors.DefaultSelector() as selector:
        selector.register(admission, selectors.EVENT_READ)
        while clock.now() < deadline:
            values = stream.read(admission)
            if stream.error is not None:
                raise stream.error
            for index, value in enumerate(values):
                if value["status"] == "reserve":
                    stream.pending = values[index + 1 :]
                    return True
                if value["status"] == "shutdown":
                    return False
                raise OSError("invalid guardian pre-fork control phase")
            if stream.eof:
                return False
            selector.select(min(POLL_INTERVAL, max(0, deadline - clock.now())))
    return False


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
    clock = CLOCK
    commands = FrameStream("shadow_guardian_control")
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, signal.SIG_IGN)
    try:
        if not authorize_fork(admission, error_pipe, deadline, clock, commands):
            guardian_event(error_pipe, "startup_stopped", cleanup_complete=True)
            return 125
        read, write = os.pipe()
        try:
            child = os.fork()
        except (BaseException,):  # noqa: B013
            os.close(read)
            os.close(write)
            raise
    except (OSError,) as error:  # noqa: B013
        guardian_error(error_pipe, error)
        guardian_event(error_pipe, "startup_stopped", cleanup_complete=True)
        return 127
    if child == 0:
        launcher_main(command, read, write, admission, error_pipe, deadline)
        raise AssertionError("forked launcher returned from os._exit")
    guardian = Guardian(
        OwnedLauncher(child, error_pipe),
        admission,
        write,
        clock=clock,
        commands=commands,
    )
    try:
        os.close(read)
        os.setpgid(0, os.getpgid(os.getppid()))
        guardian.reserved = True
        guardian_event(error_pipe, "ready")
        guardian.run()
        return 0
    except (OSError,) as error:  # noqa: B013
        recover_guardian(guardian, error, error_pipe)
        return 127
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
        guardian.report_diagnostic(error)
    try:
        if guardian.reserved:
            # Retain the supervisor's ordinary shutdown authority until EOF.
            guardian.failed = True
            guardian.run()
        else:
            guardian.fallback()  # Only its still-blocked direct child exists.
            guardian_event(event_pipe, "startup_stopped", cleanup_complete=True)
    except (Exception,) as secondary:  # noqa: B013
        attach_failure(error, "guardian recovery failed", secondary)


def execute_admitted(
    command: Sequence[str],
    admission: int,
    error_pipe: int,
    deadline: float,
    *,
    clock: Clock | None = None,
) -> int:
    clock = clock or CLOCK
    try:
        for name in ("SIGINT", "SIGTERM"):
            if (signum := getattr(signal, name, None)) is not None:
                signal.signal(signum, signal.SIG_DFL)
        with selectors.DefaultSelector() as selector:
            selector.register(admission, selectors.EVENT_READ)
            if (
                not selector.select(max(0, deadline - clock.now()))
                or os.read(admission, 1) != b"G"
            ):
                return 125
        os.close(admission)
        os.set_inheritable(error_pipe, False)  # noqa: FBT003
        for name in ("SIGPIPE", "SIGXFZ", "SIGXFSZ"):
            if (signum := getattr(signal, name, None)) is not None:
                signal.signal(signum, signal.SIG_DFL)
        os.execvp(command[0], list(command))  # noqa: S606
    except (OSError,) as error:  # noqa: B013
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)
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
    fork_authorized: bool = False
    collected: bool = False
    release_sent: bool = False
    events: GuardianEvents = field(default_factory=GuardianEvents)
    shutdown: GroupShutdown | None = None
    stop_error: Exception | None = None
    clock: Clock = field(default_factory=lambda: CLOCK)

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
        if (
            not self.release_sent
            and not self.collected
            and observe_exit(self.process) is not None
        ):
            if not self.fork_authorized:
                raise OSError(
                    "guardian failed before fork authorization; native command never started"
                )
            if not self.events.ready:
                raise OSError(
                    "guardian failed before reserving worker group; blocked launcher cleanup is uncertain"
                )
            raise OSError(
                "guardian exited without a valid cleanup release; worker outcome unknown"
            )

    def ready(self, control: int, deadline: float, cancelled: Cancellation) -> str:
        with selectors.DefaultSelector() as selector:
            selector.register(self.errors, selectors.EVENT_READ)
            selector.register(control, selectors.EVENT_READ)
            while True:
                if (
                    phase := self.readiness_step(control, deadline, cancelled, selector)
                ) is not None:
                    return phase
                selector.select(min(POLL_INTERVAL, max(0, deadline - self.clock.now())))

    def readiness_step(
        self,
        control: int,
        deadline: float,
        cancelled: Cancellation,
        selector: selectors.BaseSelector,
    ) -> str | None:
        self.collect()
        if self.clock.now() >= deadline:
            return "timed_out"
        self.check_error()
        if cancelled.requested or control_cancelled(control, selector):
            return "cancelled"
        if (
            self.events.bootstrap_ready
            and not self.fork_authorized
            and (phase := self.reserve_current(deadline, cancelled)) is not None
        ):
            return phase
        return "ready" if self.events.ready else None

    def reserve_current(self, deadline: float, cancelled: Cancellation) -> str | None:
        if cancelled.requested:
            return "cancelled"
        if self.clock.now() >= deadline:
            return "timed_out"
        self.fork_authorized = True
        self.command("reserve")
        return None

    def stop(self) -> int | None:
        if self.stop_error is not None:
            raise self.stop_error
        if self.cleaned:
            return self.events.returncode
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

    def stop_owned(self) -> int | None:
        if not self.fork_authorized or self.events.startup_clean:
            return self.stop_unstarted()
        if self.shutdown is None:
            self.shutdown = GroupShutdown(
                self.process.pid, ShutdownBudget(self.clock.now()), self.clock
            )
            try:
                self.command("shutdown", started=self.shutdown.budget.started)
            except (OSError,) as error:  # noqa: B013
                self.collect_for_cleanup()
                try:
                    self.check_error()
                except (OSError,) as owner_error:  # noqa: B013
                    primary = owner_error
                else:
                    primary = OSError(
                        "guardian control lost; worker cleanup ownership is uncertain"
                    )
                attach_failure(primary, "guardian shutdown transport failed", error)
                self.events.error = primary
        self.await_reservation(self.shutdown.budget.term_until)
        if self.events.startup_clean:
            return self.stop_unstarted()
        if not self.events.ready:
            if observe_exit(self.process) is not None:
                self.process.wait(timeout=WAIT_MARGIN)
                self.collected = True
            raise OSError(
                "guardian lost before reservation; cannot certify worker cleanup"
            )
        self.shutdown.run(self.collect_for_cleanup)
        self.wait_launcher(self.shutdown.budget.deadline)
        self.command("release", group_cleaned=True)
        self.release_sent = True
        wait_exit(self.process, self.shutdown.budget.deadline, clock=self.clock)
        self.cleaned = True
        if self.admitted:
            return self.worker_code()
        return self.events.returncode

    def stop_unstarted(self) -> None:
        """Fork was never authorized, or the guardian certified no remaining child."""
        if not self.collected:
            if observe_exit(self.process) is None:
                self.process.kill()
            deadline = self.clock.now() + REAP_TIMEOUT
            wait_exit(self.process, deadline, clock=self.clock)
            self.process.wait(timeout=max(0.01, deadline - self.clock.now()))
            self.collected = True
        self.cleaned = True

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
                remaining = deadline - self.clock.now()
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
                remaining = deadline - self.clock.now()
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


def start_worker(
    command: Sequence[str], deadline: float, *, clock: Clock | None = None
) -> Worker:
    clock = clock or CLOCK
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
        return Worker(process, admission_write, error_read, clock=clock)
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
    with selectors.DefaultSelector() as selector:
        selector.register(control, selectors.EVENT_READ)
        admitted = await_token(
            control, b"A", deadline, cancelled, clock=worker.clock, selector=selector
        )
        if admitted != "ready":
            return admitted
        if cancelled.requested or control_cancelled(control, selector):
            return "cancelled"
    if worker.clock.now() >= deadline:
        return "timed_out"
    worker.admit()
    return monitor(worker, control, deadline, cancelled, clock=worker.clock)


def finish_worker(worker: Worker, control: int, status: int, outcome: str) -> None:
    notes: list[str] = []
    try:
        code = worker.stop()
        if outcome == "completed":
            worker.exec_error()
    except (Exception,) as error:  # noqa: B013
        if outcome not in {"timed_out", "cancelled"}:
            raise
        code = None
        notes = [f"worker cleanup failed: {text}" for text in exception_notes(error)]
    send_status(
        status, outcome, returncode=code, cleanup_complete=worker.cleaned, notes=notes
    )
    await_release(control)
    worker.released = True
    if worker.cleaned and not worker.collected:
        worker.process.wait(timeout=REAP_TIMEOUT)
        worker.collected = True


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
        finally:
            cleaned = worker.cleaned
    send_status(
        status,
        "runtime_failure",
        detail=str(error),
        notes=getattr(error, "__notes__", []),
        cleanup_complete=cleaned,
    )
    if worker is not None:
        await_release(control)
        if cleaned and not worker.collected:
            worker.process.wait(timeout=REAP_TIMEOUT)
            worker.collected = True


def supervise(
    command: Sequence[str], control: int, status: int, deadline: float
) -> int:
    clock = CLOCK
    cancelled = Cancellation()
    previous = {
        sig: signal.signal(sig, cancelled.handle)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    worker = None
    try:
        readiness = await_start(control, deadline, cancelled, clock=clock)
        if readiness != "ready":
            send_status(status, readiness, cleanup_complete=True)
            return 0
        worker = start_worker(command, deadline, clock=clock)
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
    stream: FrameStream = field(
        default_factory=lambda: FrameStream("shadow_supervision")
    )
    messages: list[dict[str, Any]] = field(default_factory=list)
    owned_pid: int | None = None
    released: bool = False
    error: Exception | None = None
    control: ControlChannel | None = None
    thread: threading.Thread | None = None
    stdout: str | bytes | None = None
    stderr: str | bytes | None = None
    stopping: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)
    clock: Clock = field(default_factory=lambda: CLOCK)

    @property
    def payload(self) -> bytes:
        return self.stream.payload

    @payload.setter
    def payload(self, value: bytes) -> None:
        self.stream.payload = value

    @property
    def total(self) -> int:
        return self.stream.total

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
        if self.stream.configured is None:
            os.set_blocking(self.descriptor, False)
            self.stream.configured = self.descriptor
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
        for value in self.stream.feed(chunk):
            try:
                self.accept_message(value)
            except (OSError,) as error:  # noqa: B013
                self.error = self.error or error
        self.error = self.error or self.stream.error

    def parse_message(self, raw: bytes) -> None:
        value = self.stream.parse_frame(raw)
        if value is not None:
            self.accept_message(value)

    def accept_message(self, value: dict[str, Any]) -> None:
        if len(self.messages) >= 2:
            raise OSError("invalid shadow supervisor status")
        # Keep an authoritative stop outcome even if cleanup requires rescue.
        if value["status"] in TERMINAL and type(value.get("cleanup_complete")) is bool:
            self.messages.append(value)
            self.accept(value)
        else:
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
    process: subprocess.Popen[str], deadline: float, *, clock: Clock | None = None
) -> bool:
    """Only our still-waitable, stopped child can vouch for its unreaped worker."""
    clock = clock or CLOCK
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
    process: subprocess.Popen[str], deadline: float, *, clock: Clock | None = None
) -> tuple[str | None, str | None]:
    clock = clock or CLOCK
    try:
        return process.communicate(timeout=max(0, deadline - clock.now()))
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


def collect_supervisor(
    process: subprocess.Popen[str], channel: StatusChannel, deadline: float
) -> None:
    try:
        captured = reap_supervisor(process, deadline, clock=channel.clock)
    except (subprocess.TimeoutExpired,) as error:  # noqa: B013
        retain_output(channel, error.output, error.stderr)
        raise
    if captured is not None:
        retain_output(channel, *captured)


def emergency_stop(process: subprocess.Popen[str], channel: StatusChannel) -> None:
    clock = channel.clock
    deadline = clock.now() + REAP_TIMEOUT
    signal_deadline = deadline - WAIT_MARGIN
    failures: list[Exception] = []
    stopped = False
    try:
        stopped = stopped_supervisor(process, signal_deadline, clock=clock)
        with channel.lock:
            worker, released = channel.owned_pid, channel.released
        if not stopped and not released:
            raise OSError(
                "supervisor exited; cannot certify worker cleanup without live ownership"
            )
        if stopped and worker is not None and not released:
            if clock.now() >= signal_deadline:
                raise OSError(
                    "emergency signal budget exhausted; collection reserve retained"
                )
            force_group(worker, signal_deadline, clock=clock)
    except (Exception,) as error:  # noqa: B013
        failures.append(error)
    finally:
        if process.returncode is None:
            try:
                resume_supervisor(process, channel)
            except (Exception,) as error:  # noqa: B013
                failures.append(error)
        try:
            collect_supervisor(process, channel, deadline)
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
    clock: Clock | None = None,
) -> subprocess.Popen[str]:
    clock = clock or CLOCK
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
            repr(clock.now() + timeout),
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
    clock: Clock | None = None,
) -> tuple[str | None, str | None]:
    clock = clock or CLOCK
    while True:
        channel.collect(check_errors=check_errors)
        remaining = deadline - clock.now()
        if remaining <= 0:
            raise OSError("shadow supervisor did not finish within shutdown allowance")
        try:
            stdout, stderr = process.communicate(timeout=min(POLL_INTERVAL, remaining))
            retain_output(channel, stdout, stderr)
            return stdout, stderr
        except (subprocess.TimeoutExpired,) as error:  # noqa: B013
            retain_output(channel, error.output, error.stderr)


def retain_output(
    channel: StatusChannel, stdout: str | bytes | None, stderr: str | bytes | None
) -> None:
    """Retain the latest cumulative snapshot without decoding it during polling."""
    if stdout is not None:
        channel.stdout = stdout
    if stderr is not None:
        channel.stderr = stderr


def attach_timeout_output(error: BaseException, channel: StatusChannel) -> None:
    if isinstance(error, subprocess.TimeoutExpired):
        if channel.stdout is not None:
            error.output = (
                channel.stdout.encode()
                if isinstance(channel.stdout, str)
                else channel.stdout
            )
        if channel.stderr is not None:
            error.stderr = (
                channel.stderr.encode()
                if isinstance(channel.stderr, str)
                else channel.stderr
            )


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
        channel.clock.now() + TERM_GRACE + REAP_TIMEOUT + WAIT_MARGIN,
        check_errors=False,
        clock=channel.clock,
    )
    attach_timeout_output(primary, channel)
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
            attach_timeout_output(primary, channel)
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
    clock = CLOCK
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
        channel = StatusChannel(status_read, clock=clock)
        process = start_supervisor(
            command,
            timeout=timeout,
            capture_output=capture_output,
            control_read=control_read,
            status_write=status_write,
            clock=clock,
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
                clock.now() + timeout + TERM_GRACE + REAP_TIMEOUT + WAIT_MARGIN,
                clock=clock,
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
            selected = selected_stop_error(command, timeout, channel, error)
            if process.returncode is None:
                cancel_supervisor(process, channel, selected)
            elif not channel.released:
                selected.add_note(
                    "supervisor exited; cannot certify worker cleanup without live ownership"
                )
            attach_stderr(selected, channel.stderr)
            attach_timeout_output(selected, channel)
            if (
                isinstance(selected, subprocess.TimeoutExpired)
                and selected is not error
            ):
                raise selected from error
            raise
    except (BaseException,) as error:  # noqa: B013
        primary = error
        raise
    finally:
        close_supervision(descriptors, process, channel, control, primary)


def selected_stop_error(
    command: Sequence[str],
    timeout: float,
    channel: StatusChannel,
    error: BaseException,
) -> BaseException:
    if isinstance(error, (KeyboardInterrupt, SystemExit, subprocess.TimeoutExpired)):
        return error
    with channel.lock:
        terminal = channel.messages[-1] if channel.messages else {}
    if terminal.get("status") != "timed_out":
        return error
    selected = subprocess.TimeoutExpired(command, timeout, stderr=channel.stderr)
    attach_timeout_output(selected, channel)
    attach_failure(selected, "shadow timeout cleanup", error)
    for note in terminal.get("notes", []):
        if note not in getattr(error, "__notes__", ()):
            add_note_once(selected, note)
    return selected


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
