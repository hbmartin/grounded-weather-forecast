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

HERE: Path | None = None

# The station publisher runs under the collector's environment for its query API.
STATION_SCRIPT: Path | None = None
STAGED: Path | None = None
STATION_TIMEOUT_S = 180


def support_dir() -> Path:
    if HERE is not None:
        return HERE
    try:
        from grounded_weather_forecast.support_paths import (  # noqa: PLC0415
            application_support_dir,
        )

        return application_support_dir()
    except (ImportError, ValueError, RuntimeError) as error:
        raise RuntimePathError(
            f"publishing support path unavailable: {error}"
        ) from error


def staged_dir() -> Path:
    return STAGED if STAGED is not None else support_dir() / "staged"


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


# --config is a global flag: passing it after the subcommand is an argparse
# error, which is exactly how every scheduled run between 2026-07-27 and
# 2026-08-03 died silently.
def run_predict(paths: RuntimePaths, shadow_capture: Path | None = None) -> int:
    from grounded_weather_forecast.status import CommandOutcome  # noqa: PLC0415

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
                *(["--shadow-capture", str(shadow_capture)] if shadow_capture else []),
            ],
            check=False,
        ).returncode
        code = CommandOutcome.from_subprocess(code).exit_code
    except (SystemExit,) as exc:  # noqa: B013 -- project exception convention
        code = CommandOutcome.normalize(exc.code).exit_code
    except (Exception,) as exc:  # noqa: B013 -- prediction boundary
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


RESTORE_LOG: Path | None = None


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
    with (
        RESTORE_LOG if RESTORE_LOG is not None else support_dir() / "auto-restore.log"
    ).open("a", encoding="utf-8") as log:
        log.write(f"\n== auto-restore spawned {datetime.now(UTC).isoformat()} ==\n")
        log.flush()
        process = subprocess.Popen(  # noqa: S603 - fixed uv binary and script
            [
                *paths.command(
                    "python", str(support_dir() / "daily-maintenance.py"), "--restore"
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
                str(
                    STATION_SCRIPT
                    if STATION_SCRIPT is not None
                    else support_dir() / "publish_station.py"
                ),
                "--out-dir",
                str(staged_dir()),
            ],
            env=env,
            timeout=STATION_TIMEOUT_S,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"station render failed: {exc}", flush=True)
        return 1
    return result.returncode


def publish(
    paths: RuntimePaths,
    station_project: Path | None = None,
    shadow: tuple[Path, Path] | None = None,
) -> int:
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
    staged_documents = (
        sorted(staged_dir().glob("*.json")) if staged_dir().is_dir() else []
    )

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
    receipt = None
    missing_receipt_api = shadow is not None and not hasattr(
        publish_forecast, "run_with_receipt"
    )
    if shadow is not None and not missing_receipt_api:
        result = publish_forecast.run_with_receipt(
            source=source, include_dir=staged_dir()
        )
        exit_code = result.exit_code
        receipt = result.receipt.to_dict() if result.receipt is not None else None
    else:
        exit_code = publish_forecast.run(source=source, include_dir=staged_dir())
    print(
        f"{timestamp()} == publish exited {exit_code} "
        f"duration_s={time.monotonic() - publish_started:.1f} ==",
        flush=True,
    )
    if exit_code == 0 and shadow is not None:
        if missing_receipt_api:
            print(
                "forecast published; shadow requires publisher run_with_receipt API (2.3+)",
                flush=True,
            )
            return 1
        root, protocol = shadow
        return finalize_shadow(
            paths,
            source,
            root,
            protocol,
            publication_receipt=receipt,
        )
    return exit_code


def finalize_shadow(
    paths: RuntimePaths,
    source: Path,
    root: Path,
    protocol: Path,
    *,
    publication_receipt: dict | None = None,
) -> int:
    """After successful publication, join its exact commit to captured inputs."""
    try:
        return _finalize_shadow(
            paths, source, root, protocol, publication_receipt=publication_receipt
        )
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"forecast published; shadow receipt/record failed: {error}", flush=True)
        return 1


def _finalize_shadow(
    paths: RuntimePaths,
    source: Path,
    root: Path,
    protocol: Path,
    *,
    publication_receipt: dict | None = None,
) -> int:
    from grounded_weather_forecast.shadow_artifacts import (  # noqa: PLC0415
        issuance_name,
        json_object,
        private_directory,
        sha256,
        write_bytes_new,
    )
    from grounded_weather_forecast.shadow_artifacts import (  # noqa: PLC0415
        timestamp as evidence_timestamp,
    )

    document = json_object(source.read_bytes(), "publisher source")
    name = issuance_name(evidence_timestamp(document["issued_at"]))
    receipt_file = None
    if publication_receipt is not None:
        raw = json.dumps(publication_receipt, allow_nan=False).encode()
        folder = root / "receipts" / name
        receipt_root = root / "receipts"
        if receipt_root.is_symlink() or receipt_root.is_junction():
            raise ValueError("receipt spool root must not be an alias")
        private_directory(folder)
        if (
            folder.is_symlink()
            or folder.is_junction()
            or folder.resolve().parent != (root / "receipts").resolve()
        ):
            raise ValueError("receipt spool must remain within the selected root")
        target = folder / (sha256(raw) + ".json")
        if target.exists():
            if (
                target.is_symlink()
                or not target.is_file()
                or target.read_bytes() != raw
            ):
                raise ValueError("receipt spool conflicts with existing evidence")
        else:
            write_bytes_new(target, raw)
        receipt_file = str(target)
    return _shadow_child(paths, source, root, protocol, name, receipt_file)


def _shadow_child(
    paths: RuntimePaths,
    source: Path,
    root: Path,
    protocol: Path,
    name: str,
    receipt_file: str | None,
) -> int:
    result = subprocess.run(  # noqa: S603 -- reviewed local shadow script, no shell
        [
            *paths.command("python"),
            str(paths.project / "scripts/temperature_shadow.py"),
            "finalize",
            "--input",
            str(root / "capsules" / name),
            "--out",
            str(root / "records"),
            "--protocol",
            str(protocol),
            *(["--publication-receipt", receipt_file] if receipt_file else []),
            "--source",
            str(source),
            "--mode",
            "prospective",
        ],
        check=False,
        timeout=300,
    )
    if result.returncode:
        print("forecast published; shadow receipt/record failed", flush=True)
    return 1 if result.returncode else 0


def shadow_preflight(paths: RuntimePaths, protocol: Path) -> tuple[bool, str | None]:
    """Shadow-specific faults cannot prevent ordinary production publication."""
    try:
        document = json.loads(protocol.read_text())
        start, end = (
            datetime.fromisoformat(document[key])
            for key in ("window_start", "window_end")
        )
        if start.utcoffset() is None or end.utcoffset() is None or start >= end:
            raise ValueError("shadow window requires ordered timezone-aware timestamps")
        if not start <= datetime.now(UTC) < end:
            print("shadow skipped: outside enrollment window", flush=True)
            return False, None
        started = time.monotonic()
        check = subprocess.run(  # noqa: S603 -- selected local runtime, no providers
            [
                *paths.command("python"),
                str(paths.project / "scripts/temperature_shadow.py"),
                "preflight",
                "--protocol",
                str(protocol),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        print(
            f"shadow preflight duration_s={time.monotonic() - started:.3f}", flush=True
        )
        if check.returncode:
            detail = (getattr(check, "stderr", "") or "").strip().splitlines()
            return (
                False,
                "shadow unavailable: selected prediction runtime lacks compatible v3 capture/enrollment"
                + (f": {detail[-1][:400]}" if detail else ""),
            )
        result = json.loads(check.stdout)
        if not isinstance(result, dict) or result.get("status") not in {
            "ready",
            "outside_window",
        }:
            raise ValueError("selected runtime returned an invalid preflight status")
        if result["status"] == "outside_window":
            print("shadow skipped: outside enrollment window", flush=True)
            return False, None
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RecursionError,
        subprocess.SubprocessError,
    ) as error:
        return False, f"shadow preflight failed: {error}"
    return True, None


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
        parser.add_argument(
            "--shadow-root", type=Path, help="opt-in private capsule/record directory"
        )
        parser.add_argument(
            "--shadow-protocol", type=Path, help="reviewed frozen enrollment protocol"
        )
        args = parser.parse_args(argv)
        if (args.shadow_root is None) != (args.shadow_protocol is None):
            parser.error(
                "--shadow-root and --shadow-protocol must be provided together"
            )
        # History paths must resolve from the project before invoking the CLI.
        paths = resolve_runtime()
        os.chdir(paths.project)
        print(
            f"{timestamp()} PROJECT_CWD cwd={paths.project} project={paths.project} forecast={paths.forecast}",
            flush=True,
        )
        sys.path.insert(0, str(support_dir()))
        shadow_enabled, shadow_error = (
            shadow_preflight(paths, args.shadow_protocol)
            if args.shadow_root is not None
            else (False, None)
        )
        predicted = (
            run_predict(paths, args.shadow_root / "capsules")
            if shadow_enabled
            else run_predict(paths)
        )
        if predicted != 0:
            return 1
        restore_failed = False
        try:
            maybe_spawn_restore(paths)
        except (Exception,) as exc:  # noqa: B013 -- optional launch faults cannot block publication
            restore_failed = True
            print(
                f"auto-restore launch failed: {exc}; continuing publication", flush=True
            )
        code = (
            publish(
                paths, args.station_project, (args.shadow_root, args.shadow_protocol)
            )
            if shadow_enabled
            else publish(paths, args.station_project)
        )
        if shadow_error:
            print(f"ordinary publication exited {code}; {shadow_error}", flush=True)
        if code == 0 and (restore_failed or shadow_error):
            code = 1
        return code
    except (RuntimePathError,) as exc:  # noqa: B013 -- project convention
        print(f"configuration error: {exc}", flush=True)
        code = 2
        return code  # noqa: RET504 -- finally logs the actual exit status
    except (SystemExit,) as exc:  # noqa: B013 -- project convention
        code = exc.code if isinstance(exc.code, int) else 1
        return code  # noqa: RET504 -- finally logs the actual exit status
    except (OSError, ImportError) as exc:
        print(f"runtime failure: {exc}", flush=True)
        code = 1
        return code  # noqa: RET504 -- finally logs the actual exit status
    finally:
        print(
            f"{timestamp()} END mode=predict pid={os.getpid()} duration_s={time.monotonic() - started:.1f} exit={code}",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
