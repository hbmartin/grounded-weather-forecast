# ruff: noqa: INP001 -- standalone installed launchd script
"""Daily grounded-weather-forecast maintenance, run by launchd.

Ingests the latest Open-Meteo ensemble statistics, rebuilds the dataset from
the live station + forecast archives, refreshes the truth-semantics alignment
artifact, attempts a live-source backtest (which promotes a model release once
enough live history exists), re-runs the synthetic backtest on Sundays (so
newly registered methods pick up backfill folds within a week), and
regenerates the leaderboard reports.

Runs daily at 06:30. Missed days require a manual launchctl kickstart. A stamp
file prevents a manual catch-up from re-running within 20 hours of the last
successful maintenance.

Every nonzero stage exit fails maintenance, including a backtest that cannot
produce folds. Failed runs never advance the successful-maintenance stamp.
"""

import fcntl
import os
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

REPO = "/Volumes/Pro2T/weather/grounded-weather-forecast"
UV = "/opt/homebrew/bin/uv"
MIN_INTERVAL_S = 20 * 3600


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def maintenance_steps(*, restore: bool) -> list[list[str]]:
    steps = [] if restore else [["ingest-ensembles"], ["build-dataset"], ["alignment"]]
    steps.append(["backtest", "--source", "live"])
    if not restore and datetime.now(UTC).weekday() == 6:
        steps.append(["backtest", "--source", "synthetic"])
    steps.extend([["truth-qc"], ["report"], ["prune-scores"]])
    return steps


def run_stage(step: list[str]) -> int:
    started = time.monotonic()
    name = step[0]
    print(f"{timestamp()} == {' '.join(step)} ==", flush=True)
    try:
        # Fresh interpreters release native allocator arenas between stages.
        code = subprocess.run(  # noqa: S603 -- fixed binary and stage arguments
            [
                "/usr/bin/time",
                "-l",
                UV,
                "--directory",
                REPO,
                "run",
                "--locked",
                "--project",
                REPO,
                "grounded-weather-forecast",
                "--config",
                f"{REPO}/config.toml",
                *step,
            ],
            check=False,
        ).returncode
        if code < 0:
            code = 128 - code
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        print(f"{name} crashed: {exc!r}", flush=True)
        code = 70
    print(
        f"{timestamp()} == {name} exited {code} duration_s={time.monotonic() - started:.1f} ==",
        flush=True,
    )
    return code


def maintain(stamp: Path, *, restore: bool) -> int:
    # Only actual maintenance owns this lock; readers never probe it exclusively.
    with stamp.with_name("maintenance-job.lock").open("a") as job_lock:
        try:
            fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError,):  # noqa: B013 -- project convention
            print("another maintenance job is running; skipping duplicate", flush=True)
            return 0
        if (
            not restore
            and stamp.exists()
            and time.time() - stamp.stat().st_mtime < MIN_INTERVAL_S
        ):
            print(
                "last successful maintenance is fresh; skipping catch-up run",
                flush=True,
            )
            return 0
        os.chdir(REPO)
        print(f"{timestamp()} PROJECT_CWD cwd={Path.cwd()}", flush=True)
        failures: list[str] = []
        for step in maintenance_steps(restore=restore):
            name = step[0]
            # Do not prune against a catalog that report failed to refresh.
            if name == "prune-scores" and "report" in failures:
                print("== prune-scores skipped: report failed this run ==", flush=True)
                continue
            code = run_stage(step)
            if code != 0:
                failures.append(" ".join(step))
        if failures:
            print(f"maintenance failed stages: {', '.join(failures)}", flush=True)
            return 1
        if not restore:
            stamp.touch()
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("POLARS_MAX_THREADS", "2")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    restore = "--restore" in (sys.argv[1:] if argv is None else argv)
    mode = "restore" if restore else "maintain"
    started = time.monotonic()
    code = 1
    print(
        f"{timestamp()} START mode={mode} pid={os.getpid()} ppid={os.getppid()} cwd={Path.cwd()} host={socket.gethostname()} project={REPO}",
        flush=True,
    )
    stamp = (
        Path.home()
        / "Library/Application Support/grounded-weather-forecast/last-maintenance"
    )
    try:
        code = maintain(stamp, restore=restore)
        return code  # noqa: RET504 -- finally logs the actual exit status
    finally:
        print(
            f"{timestamp()} END mode={mode} pid={os.getpid()} duration_s={time.monotonic() - started:.1f} exit={code}",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
