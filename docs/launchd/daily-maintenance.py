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

A backtest exit of 1 means "no folds yet" and is tolerated so the job stays
green while the archive matures.
"""

import atexit
import fcntl
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = "/Volumes/ExtStor/weather/grounded-weather-forecast"
UV = "/opt/homebrew/bin/uv"
os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
STAMP = (
    Path.home()
    / "Library/Application Support/grounded-weather-forecast/last-maintenance"
)
MIN_INTERVAL_S = 20 * 3600
RESTORE = "--restore" in sys.argv[1:]
MODE = "restore" if RESTORE else "maintain"
STARTED = time.monotonic()
EXIT_CODE = 1


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def log_finish() -> None:
    print(
        f"{timestamp()} END mode={MODE} pid={os.getpid()} "
        f"duration_s={time.monotonic() - STARTED:.1f} exit={EXIT_CODE}",
        flush=True,
    )


atexit.register(log_finish)
print(
    f"{timestamp()} START mode={MODE} pid={os.getpid()} ppid={os.getppid()} "
    f"cwd={Path.cwd()} host={socket.gethostname()} project={REPO}",
    flush=True,
)

JOB_LOCK = STAMP.with_name("maintenance-job.lock").open("a")
try:
    fcntl.flock(JOB_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print("another maintenance job is running; skipping duplicate", flush=True)
    EXIT_CODE = 0
    sys.exit(0)

if (
    not RESTORE
    and STAMP.exists()
    and (time.time() - STAMP.stat().st_mtime) < MIN_INTERVAL_S
):
    print("last successful maintenance is fresh; skipping catch-up run", flush=True)
    EXIT_CODE = 0
    sys.exit(0)

os.chdir(REPO)
print(f"{timestamp()} PROJECT_CWD cwd={Path.cwd()}", flush=True)

STEPS: list[list[str]] = []
if not RESTORE:
    STEPS.extend([["ingest-ensembles"], ["build-dataset"], ["alignment"]])
STEPS.append(["backtest", "--source", "live"])
if (
    not RESTORE and datetime.now(UTC).weekday() == 6
):  # Sunday: refresh synthetic evidence too
    STEPS.append(["backtest", "--source", "synthetic"])
STEPS.append(["truth-qc"])
STEPS.append(["report"])
# After report so every file is cataloged in the evaluations ledger before
# deletion is even considered; keeps newest 3 per group + release-referenced.
STEPS.append(["prune-scores"])

failures: list[str] = []
for step in STEPS:
    name = step[0]
    if name == "prune-scores" and "report" in failures:
        # Never arm the deleter on a stale catalog: if report crashed, the
        # evaluations ledger was not refreshed this run. (The catalog
        # precondition inside prune is the second line of defense.)
        print("== prune-scores skipped: report failed this run ==", flush=True)
        continue
    step_started = time.monotonic()
    print(f"{timestamp()} == {' '.join(step)} ==", flush=True)
    try:
        # Each stage gets a fresh interpreter so native allocator arenas from a
        # large backtest are released before report/selection starts. time -l
        # also records peak RSS and signal termination in the launchd log.
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
    except Exception as exc:  # a broken step must not stop the rest
        print(f"{name} crashed: {exc!r}", flush=True)
        code = 70
    print(
        f"{timestamp()} == {name} exited {code} "
        f"duration_s={time.monotonic() - step_started:.1f} ==",
        flush=True,
    )
    if code != 0 and not (name == "backtest" and code == 1):
        failures.append(name)

if failures:
    EXIT_CODE = 1
    sys.exit(1)
if not RESTORE:
    STAMP.touch()
EXIT_CODE = 0
sys.exit(0)
