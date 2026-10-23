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

import argparse
import fcntl
import os
import socket
import stat
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grounded_weather_forecast.status import CommandOutcome

from launchd_paths import (
    RuntimePathError,
    RuntimePaths,
    resolve_runtime,
    working_directory,
)

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
    from grounded_weather_forecast.config import (  # noqa: PLC0415 -- guarded startup
        dataset_config,
    )

    raw = paths.raw_config
    maintenance = raw.get("maintenance", {})
    dataset = raw.get("dataset", {})
    if not isinstance(maintenance, dict) or not isinstance(dataset, dict):
        raise RuntimePathError("maintenance and dataset must be TOML tables")
    mode = maintenance.get("synthetic_backtest", "auto")
    if mode not in ("auto", "required", "off"):
        raise RuntimePathError(
            "maintenance.synthetic_backtest must be auto, required or off"
        )
    return mode, paths.project / dataset_config(raw).dir


def synthetic_enabled(mode: str, directory: Path) -> bool:
    from grounded_weather_forecast.dataset.matrix import (  # noqa: PLC0415 -- guarded startup
        matrix_path,
    )

    if mode == "off":
        print("synthetic backtest skipped: scheduling is off", flush=True)
        return False
    present = []
    if not stat.S_ISDIR(directory.stat().st_mode):
        raise OSError(f"dataset directory is not a directory: {directory}")
    # Opening the directory also detects access denial on otherwise absent leaves.
    with os.scandir(directory):
        pass
    for product in ("hourly", "daily"):
        path = matrix_path(directory, product, "synthetic")
        try:
            info = path.stat()
        except (FileNotFoundError,):  # noqa: B013 -- project convention
            # lstat distinguishes an absent entry from a dangling symlink.
            try:
                path.lstat()
            except (FileNotFoundError,):  # noqa: B013 -- project convention
                directory.stat()  # fail if storage vanished during discovery
            else:
                raise OSError(f"synthetic matrix is a dangling link: {path}") from None
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


def run_stage(
    step: list[str], paths: RuntimePaths, *, env: dict[str, str] | None = None
) -> int:
    started = time.monotonic()
    name = " ".join(step)
    print(f"{timestamp()} == {' '.join(step)} ==", flush=True)
    try:
        # Fresh interpreters release native allocator arenas between stages.
        code = subprocess.run(  # noqa: S603 -- fixed binary and stage arguments
            [
                "/usr/bin/time",
                "-l",
                *paths.command("grounded-weather-forecast"),
                "--config",
                str(paths.config),
                *step,
            ],
            check=False,
            env=env,
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


def backtest_stage(step: list[str], paths: RuntimePaths) -> "CommandOutcome":
    from grounded_weather_forecast.status import (  # noqa: PLC0415
        COMMAND_OUTCOME_PATH,
        read_outcome_hint,
    )

    with TemporaryDirectory(prefix="gwf-stage-") as directory:
        path = Path(directory) / "outcome.json"
        code = run_stage(
            step, paths, env={**os.environ, COMMAND_OUTCOME_PATH: str(path)}
        )
        return read_outcome_hint(path, code)


@contextmanager
def maintenance_lock(stamp: Path) -> Iterator[bool]:
    """Prepare startup state without reclassifying later stage exceptions."""
    try:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        lock = stamp.with_name("maintenance-job.lock").open("a")
    except (OSError,) as exc:  # noqa: B013 -- startup boundary
        raise RuntimePathError(f"cannot initialize maintenance lock: {exc}") from exc
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError,):  # noqa: B013 -- duplicate job remains nonfatal
            yield False
            return
        except (OSError,) as exc:  # noqa: B013 -- startup boundary
            raise RuntimePathError(f"cannot acquire maintenance lock: {exc}") from exc
        yield True


def maintain(stamp: Path, *, restore: bool, paths: RuntimePaths) -> int:
    try:
        from grounded_weather_forecast.status import (  # noqa: PLC0415 -- guarded startup
            EXIT_NO_FOLDS,
            PROMOTION_ONLY_FAILURE_KINDS,
            CommandOutcome,
        )
    except (ImportError,) as exc:  # noqa: B013 -- project convention
        raise RuntimePathError(f"maintenance runtime unavailable: {exc}") from exc

    # Only actual maintenance owns this lock; readers never probe it exclusively.
    with maintenance_lock(stamp) as acquired:
        if not acquired:
            print("another maintenance job is running; skipping duplicate", flush=True)
            return 0
        try:
            mode, dataset_dir = (
                ("off", paths.project) if restore else synthetic_mode(paths)
            )
        except (ImportError, ValueError) as exc:
            raise RuntimePathError(f"invalid maintenance configuration: {exc}") from exc
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
                except (OSError, ImportError, ValueError) as exc:
                    print(f"{label} failed: {exc}", flush=True)
                    failures.append(label)
                    failed_stages.add(name)
                    continue
            # Do not prune against a catalog that report failed to refresh.
            if name == "prune-scores" and "report" in failed_stages:
                print("== prune-scores skipped: report failed this run ==", flush=True)
                continue
            outcome = CommandOutcome.normalize(
                backtest_stage(step, paths)
                if name == "backtest"
                else run_stage(step, paths)
            )
            code = outcome.exit_code
            if name == "backtest" and code == EXIT_NO_FOLDS:
                pending.append(label)
                print(
                    f"evaluation pending: {label}; continuing diagnostics", flush=True
                )
                continue
            if code != 0:
                failures.append(label)
                failed_stages.add(name)
                if (
                    name == "backtest"
                    and outcome.failure_kind not in PROMOTION_ONLY_FAILURE_KINDS
                ):
                    print(f"maintenance stopped: {label} evaluation failed", flush=True)
                    break
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
        f"{timestamp()} START mode={mode} pid={os.getpid()} ppid={os.getppid()} cwd={working_directory()} host={socket.gethostname()}",
        flush=True,
    )
    try:
        try:
            from grounded_weather_forecast.support_paths import (  # noqa: PLC0415
                application_support_dir,
            )

            stamp = application_support_dir() / "last-maintenance"
        except (ImportError, ValueError, RuntimeError) as error:
            raise RuntimePathError(
                f"maintenance support path unavailable: {error}"
            ) from error
        parser = argparse.ArgumentParser(
            description="Daily maintenance or live-only restore"
        )
        parser.add_argument("--restore", action="store_true")
        args = parser.parse_args(argv)
        paths = resolve_runtime()
        print(
            f"{timestamp()} PROJECT project={paths.project} forecast={paths.forecast}",
            flush=True,
        )
        code = maintain(stamp, restore=args.restore, paths=paths)
        return code  # noqa: RET504 -- finally logs the actual exit status
    except (RuntimePathError,) as exc:  # noqa: B013 -- project convention
        print(f"configuration error: {exc}", flush=True)
        code = 2
        return code  # noqa: RET504 -- finally logs the actual exit status
    except (ValueError, OSError, ImportError) as exc:
        print(f"maintenance failed: {exc}", flush=True)
        code = 1
        return code  # noqa: RET504 -- finally logs the actual exit status
    except (SystemExit,) as exc:  # noqa: B013 -- project convention
        code = exc.code if isinstance(exc.code, int) else 1
        return code  # noqa: RET504 -- finally logs the actual exit status
    finally:
        print(
            f"{timestamp()} END mode={mode} pid={os.getpid()} duration_s={time.monotonic() - started:.1f} exit={code}",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
