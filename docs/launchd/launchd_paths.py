# ruff: noqa: INP001 -- shared by the standalone installed launchd scripts
"""Resolve scheduled-job paths from the active project, without machine defaults."""

import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RuntimePathError(ValueError):
    """Scheduled-job configuration cannot safely identify an input."""


@dataclass(frozen=True)
class RuntimePaths:
    project: Path
    uv: str

    @property
    def config(self) -> Path:
        return self.project / "config.toml"

    @property
    def forecast(self) -> Path:
        return self.project / "forecast.json"


@dataclass(frozen=True)
class PublishingPaths(RuntimePaths):
    station_project: Path
    station_db: Path


def read_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise RuntimePathError(f"cannot read configuration {path}: {exc}") from exc


def validate_project(path: Path, name: str) -> None:
    project = read_toml(path / "pyproject.toml").get("project")
    if not isinstance(project, dict) or project.get("name") != name:
        raise RuntimePathError(f"{path} is not a {name} project")


def resolve_runtime() -> RuntimePaths:
    project = Path.cwd().resolve()
    try:
        validate_project(project, "grounded-weather-forecast")
        read_toml(project / "config.toml")
    except (RuntimePathError,) as exc:  # noqa: B013 -- project convention
        raise RuntimePathError(
            f"{exc}; launch with uv --directory <forecast-project> run ..."
        ) from exc
    executable = shutil.which("uv")
    if executable is None:
        raise RuntimePathError(
            "cannot find uv; include its installation directory in PATH"
        )
    # Preserve the discovered executable path: macOS may grant access to that
    # launcher path rather than to a versioned package-manager target.
    return RuntimePaths(project=project, uv=str(Path(executable).absolute()))


def resolve_publishing(station_project: Path | None = None) -> PublishingPaths:
    runtime = resolve_runtime()
    project = (
        station_project.expanduser()
        if station_project is not None
        else runtime.project.parent / "ambientweather2sqlite"
    ).resolve()
    try:
        validate_project(project, "ambientweather2sqlite")
    except (RuntimePathError,) as exc:  # noqa: B013 -- project convention
        raise RuntimePathError(
            f"{exc}; use --station-project <ambientweather2sqlite-project>"
        ) from exc
    station = read_toml(runtime.config).get("station")
    value = station.get("db_path") if isinstance(station, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise RuntimePathError(
            f"{runtime.config}: station.db_path must be a nonempty path"
        )
    try:
        database = Path(value).expanduser()
        if not database.is_absolute():
            database = runtime.project / database
        database = database.resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimePathError(
            f"{runtime.config}: invalid station.db_path: {exc}"
        ) from exc
    if not database.is_file():
        raise RuntimePathError(
            f"station database not found: {database}; check station.db_path"
        )
    return PublishingPaths(runtime.project, runtime.uv, project, database)
