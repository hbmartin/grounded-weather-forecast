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

import atexit
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = "/Volumes/ExtStor/weather/grounded-weather-forecast"
CONFIG = f"{REPO}/config.toml"
OUT = f"{REPO}/forecast.json"
HERE = Path.home() / "Library/Application Support/grounded-weather-forecast"

# publish_station.py imports ambientweather2sqlite's own aggregation functions,
# so it runs under that project rather than this one. uv is also the binary
# holding the TCC grant for /Volumes/ExtStor.
UV = "/opt/homebrew/bin/uv"
os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
STATION_REPO = "/Volumes/ExtStor/weather/ambientweather2sqlite"
STATION_SCRIPT = HERE / "publish_station.py"
STAGED = HERE / "staged"
STATION_TIMEOUT_S = 180
STARTED = time.monotonic()
EXIT_CODE = 1


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def log_finish() -> None:
    print(
        f"{timestamp()} END mode=predict pid={os.getpid()} "
        f"duration_s={time.monotonic() - STARTED:.1f} exit={EXIT_CODE}",
        flush=True,
    )


atexit.register(log_finish)
print(
    f"{timestamp()} START mode=predict pid={os.getpid()} ppid={os.getppid()} "
    f"cwd={Path.cwd()} host={socket.gethostname()} project={REPO} "
    f"forecast={OUT}",
    flush=True,
)

# predict resolves config.predict.history_path relative to the working
# directory, so this must happen before the CLI is imported or the
# self-verification history lands somewhere else entirely.
os.chdir(REPO)
print(f"{timestamp()} PROJECT_CWD cwd={Path.cwd()}", flush=True)
sys.path.insert(0, str(HERE))


# --config is a global flag: passing it after the subcommand is an argparse
# error, which is exactly how every scheduled run between 2026-07-27 and
# 2026-08-03 died silently.
predict_started = time.monotonic()
print(f"{timestamp()} == predict ==", flush=True)
try:
    code = subprocess.run(  # noqa: S603 -- fixed binary and predict arguments
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
            CONFIG,
            "predict",
            "--out",
            OUT,
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
    print("not publishing; the previously published document keeps serving", flush=True)
    EXIT_CODE = 1
    sys.exit(1)

RESTORE_LOG = HERE / "auto-restore.log"


def maybe_spawn_restore() -> None:
    """One detached backtest+report when serving degraded on a code change.

    Only the "implementation changed" reason is restorable by a cycle:
    dataset/config-fingerprint changes resolve at the next 06:30 build, and
    cold starts have nothing to restore. The CLI's pipeline lock makes a
    second spawn exit 75 (EX_TEMPFAIL) instead of racing, so at most one
    restore runs however many hourly fires see the degraded document. The
    maintenance wrapper also holds the job lock across all child stages.
    Publish still proceeds immediately; hold-last-good bridges the window.
    """
    try:
        document = json.loads(Path(OUT).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    with (HERE / "maintenance-job.lock").open("a") as maintenance_lock:
        try:
            fcntl.flock(maintenance_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("maintenance is active; skipping automatic restore", flush=True)
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
                UV,
                "--directory",
                REPO,
                "run",
                "--locked",
                "--project",
                REPO,
                "python",
                str(HERE / "daily-maintenance.py"),
                "--restore",
            ],
            stdout=log,
            stderr=log,
            start_new_session=True,  # survives this launchd job's exit
            env=env,
        )
    print(f"auto-restore spawned pid={process.pid} (see auto-restore.log)", flush=True)


maybe_spawn_restore()


def render_station() -> int:
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
                UV,
                "run",
                "--project",
                STATION_REPO,
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


station_started = time.monotonic()
print(f"{timestamp()} == station ==", flush=True)
station_code = render_station()
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
        EXIT_CODE = 1
        sys.exit(1)
    print(
        f"republishing {len(staged_documents)} staged document(s) from the last run",
        flush=True,
    )

import publish_forecast  # noqa: E402

publish_started = time.monotonic()
print(f"{timestamp()} == publish ==", flush=True)
# Degraded documents (every code-identity change causes one until the restore
# cycle finishes) hold the last ready document for up to 6h instead of swapping
# the public forecast's character for the integration window.
source = publish_forecast.choose_source(Path(OUT))
EXIT_CODE = publish_forecast.run(source=source, include_dir=STAGED)
print(
    f"{timestamp()} == publish exited {EXIT_CODE} "
    f"duration_s={time.monotonic() - publish_started:.1f} ==",
    flush=True,
)
sys.exit(EXIT_CODE)
