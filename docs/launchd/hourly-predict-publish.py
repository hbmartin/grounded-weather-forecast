# ruff: noqa: INP001 -- standalone installed launchd script
"""Hourly grounded-weather-forecast predict + publish, run by launchd.

Renders the forecast document and the station documents, then publishes both to
the open-sun `data` branch in one commit. Replaces the plain `predict`
invocation that io.github.hbmartin.grounded-predict used to run directly.

One process rather than several LaunchAgents on purpose: `predict` writes
forecast.json with a non-atomic write_text, so a separately-scheduled reader can
catch a truncated file; StartInterval has no phase offset so two jobs would
drift into each other; and the `data` branch is one force-pushed unparented
commit, so two publishers would delete each other's files every hour.

If predict fails there is nothing new to publish -- it leaves the previous
forecast.json in place -- so the publish step is skipped and the already-served
document simply ages. open-sun renders that staleness from `issued_at` rather
than going blank.

The station step is more forgiving: because STAGED persists between runs and
publish_station.py writes atomically, a failed render republishes last hour's
observations rather than dropping them from the branch. Only when STAGED is
empty as well do we decline to publish at all, so that a bad render can never
strip files the branch is already serving.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from launchd_paths import (
    PublishingPaths,
    RuntimePathError,
    RuntimePaths,
    resolve_publishing,
    resolve_runtime,
    working_directory,
)

HERE = Path.home() / "Library/Application Support/grounded-weather-forecast"

# The station publisher runs under the collector's environment for its query API.
STATION_SCRIPT = HERE / "publish_station.py"
STAGED = HERE / "staged"
STATION_TIMEOUT_S = 180


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


# --config is a global flag: passing it after the subcommand is an argparse
# error, which is exactly how every scheduled run between 2026-07-27 and
# 2026-08-03 died silently.
def run_predict(paths: RuntimePaths) -> int:
    predict_started = time.monotonic()
    print(f"{timestamp()} == predict ==", flush=True)
    try:
        code = subprocess.run(  # noqa: S603 -- fixed binary and predict arguments
            [
                "/usr/bin/time",
                "-l",
                *paths.command("grounded-weather-forecast"),
                "--config",
                str(paths.config),
                "predict",
                "--out",
                str(paths.forecast),
            ],
            check=False,
        ).returncode
        if code < 0:
            code = 128 - code
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:  # report and skip publishing, never traceback out
        print(f"predict crashed: {exc!r}", flush=True)
        code = 70

    print(
        f"{timestamp()} == predict exited {code} "
        f"duration_s={time.monotonic() - predict_started:.1f} ==",
        flush=True,
    )
    if code != 0:
        print(
            "not publishing; the previously published document keeps serving",
            flush=True,
        )
        return 1
    return 0


RESTORE_LOG = HERE / "auto-restore.log"


def maybe_spawn_restore(paths: RuntimePaths) -> None:
    """One detached backtest+report when serving degraded on a code change.

    Only the "implementation changed" reason is restorable by a cycle:
    dataset/config-fingerprint changes resolve at the next 06:30 build, and
    cold starts have nothing to restore. The maintenance child acquires the
    job lock across all stages and skips duplicate jobs, so at most one
    restore runs however many hourly fires see the degraded document. Each
    child stage retains the CLI's pipeline lock as well.
    Publish still proceeds immediately; hold-last-good bridges the window.
    """
    try:
        document = json.loads(paths.forecast.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(document, dict):
        return
    reason = str(document.get("status_reason") or "")
    if document.get("status") != "degraded" or "implementation changed" not in reason:
        return
    env = {
        k: v
        for k, v in os.environ.items()
        if k != "VIRTUAL_ENV" and not k.startswith("UV_")
    }
    with RESTORE_LOG.open("a", encoding="utf-8") as log:
        log.write(f"\n== auto-restore spawned {datetime.now(UTC).isoformat()} ==\n")
        log.flush()
        process = subprocess.Popen(  # noqa: S603 - fixed uv binary and script
            [
                *paths.command(
                    "python", str(HERE / "daily-maintenance.py"), "--restore"
                ),
            ],
            stdout=log,
            stderr=log,
            start_new_session=True,  # survives this launchd job's exit
            env=env,
        )
    print(f"auto-restore spawned pid={process.pid} (see auto-restore.log)", flush=True)


def render_station(paths: PublishingPaths) -> int:
    """Refresh the staged station documents. Returns the child's exit code."""
    # This process is itself running inside the grounded project's virtualenv.
    # Leaving VIRTUAL_ENV set would make the nested `uv run` resolve against the
    # wrong environment, so hand the child a clean slate.
    env = {
        k: v
        for k, v in os.environ.items()
        if k != "VIRTUAL_ENV" and not k.startswith("UV_")
    }
    try:
        result = subprocess.run(  # noqa: S603 - fixed uv binary and script
            [
                paths.uv,
                "--directory",
                str(paths.station_project),
                "run",
                "--project",
                str(paths.station_project),
                "python",
                str(STATION_SCRIPT),
                "--out-dir",
                str(STAGED),
            ],
            env=env,
            timeout=STATION_TIMEOUT_S,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"station render failed: {exc}", flush=True)
        return 1
    return result.returncode


def publish(paths: RuntimePaths, station_project: Path | None = None) -> int:
    station_started = time.monotonic()
    print(f"{timestamp()} == station ==", flush=True)
    try:
        station_paths = resolve_publishing(station_project, runtime=paths)
        station_code = render_station(station_paths)
    except (RuntimePathError, OSError) as exc:
        print(f"station configuration failed: {exc}", flush=True)
        station_code = 2
    print(
        f"{timestamp()} == station exited {station_code} "
        f"duration_s={time.monotonic() - station_started:.1f} ==",
        flush=True,
    )
    staged_documents = sorted(STAGED.glob("*.json")) if STAGED.is_dir() else []

    if station_code != 0:
        print(f"station render exited {station_code}", flush=True)
        if not staged_documents:
            # Publishing now would push a tree with no station files, deleting any
            # the branch already serves. Leave the branch exactly as it is.
            print("no staged documents; not publishing at all this cycle", flush=True)
            return 1
        print(
            f"republishing {len(staged_documents)} staged document(s) from the last run",
            flush=True,
        )

    import publish_forecast  # noqa: PLC0415 -- installed publisher is on the runtime path

    publish_started = time.monotonic()
    print(f"{timestamp()} == publish ==", flush=True)
    # Degraded documents (every code-identity change causes one until the restore
    # cycle finishes) hold the last ready document for up to 6h instead of swapping
    # the public forecast's character for the integration window.
    source = publish_forecast.choose_source(paths.forecast)
    exit_code = publish_forecast.run(source=source, include_dir=STAGED)
    print(
        f"{timestamp()} == publish exited {exit_code} "
        f"duration_s={time.monotonic() - publish_started:.1f} ==",
        flush=True,
    )
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("POLARS_MAX_THREADS", "2")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    started = time.monotonic()
    code = 1
    print(
        f"{timestamp()} START mode=predict pid={os.getpid()} ppid={os.getppid()} cwd={working_directory()} host={socket.gethostname()}",
        flush=True,
    )
    try:
        parser = argparse.ArgumentParser(
            description="Hourly forecast and station publishing"
        )
        parser.add_argument(
            "--station-project",
            type=Path,
            help="collector project (default: sibling ambientweather2sqlite)",
        )
        args = parser.parse_args(argv)
        # History paths must resolve from the project before invoking the CLI.
        paths = resolve_runtime()
        os.chdir(paths.project)
        print(
            f"{timestamp()} PROJECT_CWD cwd={paths.project} project={paths.project} forecast={paths.forecast}",
            flush=True,
        )
        sys.path.insert(0, str(HERE))
        if run_predict(paths) != 0:
            return 1
        maybe_spawn_restore(paths)
        code = publish(paths, args.station_project)
        return code  # noqa: RET504 -- finally logs the actual exit status
    except (RuntimePathError,) as exc:  # noqa: B013 -- project convention
        print(f"configuration error: {exc}", flush=True)
        code = 2
        return code  # noqa: RET504 -- finally logs the actual exit status
    except (SystemExit,) as exc:  # noqa: B013 -- project convention
        code = exc.code if isinstance(exc.code, int) else 1
        return code  # noqa: RET504 -- finally logs the actual exit status
    except (OSError, ImportError) as exc:
        print(f"runtime configuration error: {exc}", flush=True)
        code = 2
        return code  # noqa: RET504 -- finally logs the actual exit status
    finally:
        print(
            f"{timestamp()} END mode=predict pid={os.getpid()} duration_s={time.monotonic() - started:.1f} exit={code}",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
