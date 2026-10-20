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

Insufficient-history backtests remain pending while diagnostics continue.
Pending evaluations and failed runs never advance the success stamp.
"""

import fcntl
import os
import socket
import stat
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from launchd_paths import RuntimePathError, RuntimePaths, read_toml, resolve_runtime

from grounded_weather_forecast.status import EXIT_NO_FOLDS

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


def synthetic_mode(paths: RuntimePaths) -> tuple[str, Path]:
    raw = read_toml(paths.config)
    maintenance = raw.get("maintenance", {})
    dataset = raw.get("dataset", {})
    if not isinstance(maintenance, dict) or not isinstance(dataset, dict):
        raise RuntimePathError("maintenance and dataset must be TOML tables")
    mode = maintenance.get("synthetic_backtest", "auto")
    directory = dataset.get("dir", "data")
    if mode not in ("auto", "required", "off"):
        raise RuntimePathError(
            "maintenance.synthetic_backtest must be auto, required or off"
        )
    if not isinstance(directory, str) or not directory.strip():
        raise RuntimePathError("dataset.dir must be a nonempty path")
    return mode, paths.project / directory


def synthetic_enabled(mode: str, directory: Path) -> bool:
    if mode == "off":
        print("synthetic backtest skipped: scheduling is off", flush=True)
        return False
    present = []
    for product in ("hourly", "daily"):
        path = directory / f"{product}_matrix_synthetic.parquet"
        try:
            info = path.stat()
        except (FileNotFoundError,):
            present.append(False)
            continue
        if not stat.S_ISREG(info.st_mode):
            raise OSError(f"synthetic matrix is not a regular file: {path}")
        present.append(True)
    if mode == "auto" and not any(present):
        print(
            "synthetic backtest skipped: no backfill configured (both matrices absent)",
            flush=True,
        )
        return False
    if not all(present):
        raise OSError(
            f"synthetic backtest requires both hourly and daily matrices in {directory}"
        )
    return True


def run_stage(step: list[str], paths: RuntimePaths) -> int:
    started = time.monotonic()
    name = " ".join(step)
    print(f"{timestamp()} == {' '.join(step)} ==", flush=True)
    try:
        # Fresh interpreters release native allocator arenas between stages.
        code = subprocess.run(  # noqa: S603 -- fixed binary and stage arguments
            [
                "/usr/bin/time",
                "-l",
                paths.uv,
                "--directory",
                str(paths.project),
                "run",
                "--locked",
                "--project",
                str(paths.project),
                "grounded-weather-forecast",
                "--config",
                str(paths.config),
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


def maintain(stamp: Path, *, restore: bool, paths: RuntimePaths) -> int:
    mode, dataset_dir = synthetic_mode(paths)
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
        os.chdir(paths.project)
        print(f"{timestamp()} PROJECT_CWD cwd={Path.cwd()}", flush=True)
        failures: list[str] = []
        failed_stages: set[str] = set()
        pending: list[str] = []
        for step in maintenance_steps(restore=restore):
            name = step[0]
            label = " ".join(step)
            if step == ["backtest", "--source", "synthetic"]:
                try:
                    if not synthetic_enabled(mode, dataset_dir):
                        continue
                except (OSError,) as exc:
                    print(f"{label} failed: {exc}", flush=True)
                    failures.append(label)
                    failed_stages.add(name)
                    continue
            # Do not prune against a catalog that report failed to refresh.
            if name == "prune-scores" and "report" in failed_stages:
                print("== prune-scores skipped: report failed this run ==", flush=True)
                continue
            code = run_stage(step, paths)
            if name == "backtest" and code == EXIT_NO_FOLDS:
                pending.append(label)
                print(
                    f"evaluation pending: {label}; continuing diagnostics", flush=True
                )
                continue
            if code != 0:
                failures.append(label)
                failed_stages.add(name)
        if failures:
            print(f"maintenance failed stages: {', '.join(failures)}", flush=True)
            return 1
        if pending:
            print(
                f"maintenance pending evaluations: {', '.join(pending)}; success stamp unchanged",
                flush=True,
            )
            if restore:
                return EXIT_NO_FOLDS
        elif not restore:
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
        f"{timestamp()} START mode={mode} pid={os.getpid()} ppid={os.getppid()} cwd={Path.cwd()} host={socket.gethostname()} project={Path.cwd()}",
        flush=True,
    )
    stamp = (
        Path.home()
        / "Library/Application Support/grounded-weather-forecast/last-maintenance"
    )
    try:
        paths = resolve_runtime()
        code = maintain(stamp, restore=restore, paths=paths)
        return code  # noqa: RET504 -- finally logs the actual exit status
    except (RuntimePathError,) as exc:  # noqa: B013 -- project convention
        print(f"configuration error: {exc}", flush=True)
        code = 2
        return code  # noqa: RET504 -- finally logs the actual exit status
    finally:
        print(
            f"{timestamp()} END mode={mode} pid={os.getpid()} duration_s={time.monotonic() - started:.1f} exit={code}",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
