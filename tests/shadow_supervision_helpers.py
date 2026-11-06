"""Bounded raw protocol ownership for synthetic readiness/cancellation tests."""

import os
import subprocess
from contextlib import contextmanager


def stop_worker(supervisor, process, *, clock=None, started=None):
    """Exercise the shared shutdown policy with a still-owned synthetic group."""
    clock = clock or supervisor.CLOCK
    budget = supervisor.ShutdownBudget(clock.now() if started is None else started)
    supervisor.GroupShutdown(process.pid, budget, clock).run(lambda: None)
    return supervisor.wait_exit(process, budget.deadline, clock=clock)


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
