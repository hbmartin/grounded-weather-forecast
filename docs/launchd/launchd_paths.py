# ruff: noqa: INP001 -- shared by the standalone installed launchd scripts
"""Resolve scheduled-job paths from the active project, without machine defaults."""

import os
import shutil
import signal
import stat
import subprocess
import tomllib
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def subprocess_exit_code(returncode: int) -> int:
    """Keep installed wrappers usable with the prior outcome runtime."""
    from grounded_weather_forecast.status import CommandOutcome  # noqa: PLC0415

    convert = getattr(CommandOutcome, "from_subprocess", CommandOutcome.normalize)
    return convert(returncode).exit_code


def run_shadow_child(
    command: Sequence[str], *, timeout: float, capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
    """Own the POSIX uv/Python process group so timeouts cannot orphan a runner."""
    # Reviewed shadow argv; shell parsing is never enabled.
    process = subprocess.Popen(  # noqa: S603
        command,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=True,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except (BaseException,) as error:  # noqa: B013 -- cancellation also releases children
        try:
            if os.name == "posix":
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.communicate(timeout=5)
        except (Exception,) as secondary:  # noqa: B013 -- preserve the triggering failure
            error.add_note(f"shadow child cleanup failed: {secondary}")
        raise
    finally:
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


class RuntimePathError(ValueError):
    """Scheduled-job configuration cannot safely identify an input."""


@dataclass(frozen=True)
class RuntimePaths:
    project: Path
    uv: str
    raw_config: dict[str, Any] = field(default_factory=dict, kw_only=True)

    @property
    def config(self) -> Path:
        return self.project / "config.toml"

    @property
    def forecast(self) -> Path:
        return self.project / "forecast.json"

    def command(self, *arguments: str) -> list[str]:
        return [
            self.uv,
            "--directory",
            str(self.project),
            "run",
            "--locked",
            "--project",
            str(self.project),
            *arguments,
        ]


@dataclass(frozen=True)
class PublishingPaths(RuntimePaths):
    station_project: Path


def read_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise RuntimePathError(f"cannot read configuration {path}: {exc}") from exc


def validate_project(path: Path, name: str) -> None:
    project = read_toml(path / "pyproject.toml").get("project")
    if not isinstance(project, dict) or project.get("name") != name:
        raise RuntimePathError(f"{path} is not a {name} project")


def executable_uv() -> str:
    supplied = os.environ.get("UV")
    executable = supplied if supplied is not None else shutil.which("uv")
    if not executable:
        raise RuntimePathError(
            "cannot find uv; set UV or include its installation directory in PATH"
        )
    try:
        path = Path(executable).expanduser().absolute()
        if not stat.S_ISREG(path.stat().st_mode) or not os.access(path, os.X_OK):
            raise RuntimePathError(f"uv is not an executable file: {path}")
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimePathError(f"invalid UV executable: {exc}") from exc
    # Keep the launcher path, including symlinks: macOS grants may name it.
    return str(path)


def resolve_runtime() -> RuntimePaths:
    try:
        project = Path.cwd().resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimePathError(f"cannot resolve working directory: {exc}") from exc
    try:
        validate_project(project, "grounded-weather-forecast")
    except (RuntimePathError,) as exc:  # noqa: B013 -- project convention
        raise RuntimePathError(
            f"{exc}; launch with uv --directory <forecast-project> run ..."
        ) from exc
    raw = read_toml(project / "config.toml")
    return RuntimePaths(project, executable_uv(), raw_config=raw)


def resolve_publishing(
    station_project: Path | None = None, *, runtime: RuntimePaths | None = None
) -> PublishingPaths:
    """Resolve station-only prerequisites after prediction has completed."""
    runtime = runtime or resolve_runtime()
    try:
        project = (
            station_project.expanduser()
            if station_project is not None
            else runtime.project.parent / "ambientweather2sqlite"
        ).resolve()
        validate_project(project, "ambientweather2sqlite")
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimePathError(
            f"cannot resolve collector project: {exc}; use --station-project PATH"
        ) from exc
    return PublishingPaths(
        runtime.project, runtime.uv, project, raw_config=runtime.raw_config
    )


def working_directory() -> str:
    """Logging must survive an unavailable current directory."""
    try:
        return str(Path.cwd())
    except (OSError,) as exc:  # noqa: B013 -- project convention
        return f"unavailable ({exc})"
