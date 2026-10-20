# ruff: noqa: INP001 -- shared by the standalone installed launchd scripts
"""Resolve scheduled-job paths from the active project, without machine defaults."""

import os
import shutil
import stat
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
