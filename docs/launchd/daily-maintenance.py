# ruff: noqa: INP001 -- standalone installed launchd script
"""Daily grounded-weather-forecast maintenance, run by launchd.

Ingests the latest Open-Meteo ensemble statistics, rebuilds the dataset from
the live station + forecast archives, refreshes the truth-semantics alignment
artifact, attempts a live-source backtest (which promotes a model release once
enough live history exists), re-runs the synthetic backtest on Sundays (so
newly registered methods pick up backfill folds within a week), and
regenerates the leaderboard reports.

The supplied launchd schedule runs daily at 02:15 in the host's local calendar.
A success stamp retains the successful run's start. Catch-up skips only when
that start is between the latest scheduled slot and now. Custom schedules must
pass --scheduled-at HH:MM; this option does not install or alter a job.

Insufficient-history backtests remain pending while diagnostics continue.
Pending evaluations and failed runs never advance the success stamp.
"""

import argparse
import fcntl
import os
import re
import socket
import stat
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from datetime import UTC, date, datetime, timedelta
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

DEFAULT_SCHEDULE = "02:15"


def scheduled_time(value: str) -> tuple[int, int]:
    if re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value) is None:
        raise argparse.ArgumentTypeError("scheduled time must be HH:MM (00:00–23:59)")
    hour, minute = value.split(":")
    return int(hour), int(minute)


def local_slot(day: date, scheduled_at: tuple[int, int]) -> float:
    nominal = datetime(day.year, day.month, day.day, *scheduled_at)  # noqa: DTZ001 -- host calendar
    for offset in range(24 * 60 + 1):
        wall = nominal + timedelta(minutes=offset)
        parts = wall.timetuple()
        candidates = {
            value
            for is_dst in (-1, 0, 1)
            if time.localtime(value := time.mktime((*parts[:8], is_dst)))[:6]
            == parts[:6]
        }
        if candidates:
            return min(
                candidates
            )  # First fold; gaps advance to the first valid minute.
    raise ValueError("cannot resolve maintenance calendar slot")


def latest_scheduled_slot(now: float, scheduled_at: tuple[int, int]) -> float:
    today = datetime.fromtimestamp(now).date()  # noqa: DTZ006 -- launchd host calendar
    for offset in range(3):
        slot = local_slot(today - timedelta(days=offset), scheduled_at)
        if slot <= now:
            return slot
    raise ValueError("cannot locate the most recent maintenance slot")


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def stamp_success(stamp: Path, started_at: float) -> None:
    """Publish start-based freshness only after the whole run succeeds."""
    from grounded_weather_forecast.storage import (  # noqa: PLC0415
        output_target,
        stage_text,
    )

    candidate: Path | None = None
    try:
        target = output_target(stamp)
        candidate = stage_text("", target)
        os.utime(candidate, (started_at, started_at))
        os.replace(candidate, target)  # noqa: PTH105 -- fault-injected atomic publication boundary
    except OSError as error:
        raise type(error)(
            f"cannot publish maintenance stamp {stamp}: {error}"
        ) from error
    finally:
        if candidate is not None:
            with suppress(OSError):
                candidate.unlink(missing_ok=True)


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


def maintain(
    stamp: Path,
    *,
    restore: bool,
    paths: RuntimePaths,
    scheduled_at: tuple[int, int] = (2, 15),
) -> int:
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
        started_at = time.time()
        if (
            not restore
            and stamp.exists()
            and latest_scheduled_slot(started_at, scheduled_at)
            <= stamp.stat().st_mtime
            <= started_at
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
            # Only a successful start can satisfy the current calendar slot.
            stamp_success(stamp, started_at)
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
        parser.add_argument(
            "--scheduled-at", type=scheduled_time, default=DEFAULT_SCHEDULE
        )
        args = parser.parse_args(argv)
        paths = resolve_runtime()
        print(
            f"{timestamp()} PROJECT project={paths.project} forecast={paths.forecast}",
            flush=True,
        )
        code = maintain(
            stamp, restore=args.restore, paths=paths, scheduled_at=args.scheduled_at
        )
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
