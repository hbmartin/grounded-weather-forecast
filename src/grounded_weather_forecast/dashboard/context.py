"""Every filesystem read the dashboard makes, isolated in one collector.

Absence is data: a missing file loads as ``None`` (or an empty frame) and
drives the corresponding panel's loud "not yet" state instead of an error.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from grounded_weather_forecast.backtest.scores import load_scores
from grounded_weather_forecast.config import Config
from grounded_weather_forecast.contracts import MixedProvenanceError
from grounded_weather_forecast.dataset.matrix import DatasetPaths, matrix_path
from grounded_weather_forecast.dataset.providers import read_latest_archive_location
from grounded_weather_forecast.runs import RUNS_SCHEMA, read_runs, runs_path
from grounded_weather_forecast.serve.history import load_history
from grounded_weather_forecast.serve.observability import (
    OBSERVABILITY_HISTORY_SCHEMA,
    ObservabilitySnapshot,
    load_observability_history,
    load_observability_states_with_failures,
)
from grounded_weather_forecast.serve.schema import Forecast


@dataclass(frozen=True, slots=True)
class DashboardContext:
    """Everything the zone builders read; pure data, no I/O beyond here."""

    config: Config
    now: datetime
    manifest: Mapping[str, object] | None = None
    truth_minute: pl.DataFrame | None = None
    truth_hourly: pl.DataFrame | None = None
    truth_daily: pl.DataFrame | None = None
    qc: pl.DataFrame | None = None
    hourly_matrix: pl.DataFrame | None = None
    daily_matrix: pl.DataFrame | None = None
    synthetic_hourly: pl.DataFrame | None = None
    score_frames: Mapping[str, pl.DataFrame] = field(default_factory=dict)
    unreadable_scores: tuple[str, ...] = ()
    # Artifacts that exist but could not be read. A missing file is a young
    # archive; a corrupt or permission-denied one is a fault, and collapsing
    # both to None let a broken deployment render as merely early.
    unreadable_artifacts: tuple[str, ...] = ()
    history: pl.DataFrame | None = None
    latest_forecast: Forecast | None = None
    releases: tuple[Mapping[str, object], ...] = ()
    alignment: Mapping[str, object] | None = None
    drift: Mapping[str, object] | None = None
    observability_states: tuple[ObservabilitySnapshot, ...] = ()
    observability_history: pl.DataFrame = field(default_factory=pl.DataFrame)
    runs: pl.DataFrame = field(default_factory=pl.DataFrame)
    archive_location: tuple[float, float] | None = None


def _try_parquet(
    path: Path, failures: list[str] | None = None, *, label: str | None = None
) -> pl.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pl.read_parquet(path)
    except (OSError, pl.exceptions.PolarsError):
        if failures is not None:
            failures.append(label or path.name)
        return None


def _try_json(
    path: Path, failures: list[str] | None = None, *, label: str | None = None
) -> Mapping[str, object] | None:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        if failures is not None:
            failures.append(label or path.name)
        return None
    if not isinstance(loaded, dict):
        if failures is not None:
            failures.append(label or path.name)
        return None
    return loaded


def _score_frames(
    config: Config, failures: list[str]
) -> tuple[dict[str, pl.DataFrame], tuple[str, ...]]:
    """Readable score frames, plus the stems that exist but cannot be read.

    A file that is present but corrupt is *not* the same as a young archive
    with no file at all; the second return value keeps them distinguishable
    so zones can say "unreadable" instead of "not enough data yet".
    """
    frames: dict[str, pl.DataFrame] = {}
    unreadable: list[str] = []
    scores_dir = config.dataset.dir / "scores"
    for path in sorted(scores_dir.glob("scores_*.parquet")):
        try:
            frames[path.stem] = load_scores(path)
        except (OSError, MixedProvenanceError, pl.exceptions.PolarsError):
            unreadable.append(path.stem)
            failures.append(f"dataset/scores/{path.name}")
    return frames, tuple(unreadable)


def _latest_forecast(config: Config, failures: list[str]) -> Forecast | None:
    directory = config.predict.history_path.parent / "served_forecasts"
    if not directory.exists():
        return None
    documents = sorted(directory.glob("*.json"))
    if not documents:
        return None
    try:
        return Forecast.from_json(documents[-1].read_text(encoding="utf-8"))
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ):
        failures.append(f"predict/served_forecasts/{documents[-1].name}")
        return None


def _releases(config: Config, failures: list[str]) -> tuple[Mapping[str, object], ...]:
    directory = config.artifacts_dir / "releases"
    if not directory.exists():
        return ()
    loaded = (
        _try_json(
            path,
            failures,
            label=f"artifacts/releases/{path.name}",
        )
        for path in sorted(directory.glob("*.json"))
    )
    return tuple(release for release in loaded if release is not None)


def _qc_summary(config: Config) -> pl.DataFrame | None:
    from grounded_weather_forecast.dataset.qc import (  # noqa: PLC0415
        QC_FLATLINE,
        apply_causal_qc,
        apply_qc,
        qc_col,
        qc_summary,
    )
    from grounded_weather_forecast.dataset.station import (  # noqa: PLC0415
        read_observations,
    )

    channels = sorted(set(config.station.columns.values()))
    try:
        observations = read_observations(config.station)
        if observations.is_empty():
            return None
        summary = qc_summary(apply_qc(observations, config.qc, channels), channels)
        causal = apply_causal_qc(observations, config.qc, channels)
        latest = causal.row(causal.height - 1, named=True)
        active = pl.DataFrame(
            {
                "channel": channels,
                "active_flatline": [
                    bool(
                        (flag := latest.get(qc_col(channel))) is not None
                        and int(flag) & QC_FLATLINE
                    )
                    for channel in channels
                ],
            }
        )
        return summary.join(active, on="channel", how="left")
    except (OSError, ValueError, pl.exceptions.PolarsError):
        return None


def _history(config: Config, failures: list[str]) -> pl.DataFrame | None:
    if not config.predict.history_path.exists():
        return None
    try:
        frame = load_history(config.predict.history_path)
    except (OSError, pl.exceptions.PolarsError):
        failures.append(f"predict/{config.predict.history_path.name}")
        return None
    return None if frame.is_empty() else frame


def _observability_states(
    config: Config, failures: list[str]
) -> tuple[ObservabilitySnapshot, ...]:
    try:
        snapshots, unreadable = load_observability_states_with_failures(
            config.artifacts_dir
        )
    except (OSError, TypeError, ValueError):
        failures.append("artifacts/observability/latest.json")
        return ()
    for path in unreadable:
        try:
            relative = path.relative_to(config.artifacts_dir)
        except ValueError:
            relative = path
        failures.append(f"artifacts/{relative}")
    return snapshots


def _observability_history(config: Config, failures: list[str]) -> pl.DataFrame:
    path = config.artifacts_dir / "observability" / "history.parquet"
    try:
        return load_observability_history(config.artifacts_dir)
    except (OSError, ValueError, pl.exceptions.PolarsError):
        if path.exists():
            failures.append("artifacts/observability/history.parquet")
        return pl.DataFrame(schema=OBSERVABILITY_HISTORY_SCHEMA)


def _runs(config: Config, failures: list[str]) -> pl.DataFrame:
    path = runs_path(config)
    try:
        return read_runs(path)
    except (OSError, ValueError, pl.exceptions.PolarsError):
        if path.exists():
            failures.append("dataset/runs.parquet")
        return pl.DataFrame(schema=RUNS_SCHEMA)


def _archive_location(config: Config) -> tuple[float, float] | None:
    try:
        return read_latest_archive_location(config.forecasts)
    except (OSError,):  # noqa: B013 - project style requires tuple clauses
        return None


def collect_context(config: Config, *, now: datetime | None = None) -> DashboardContext:
    paths = DatasetPaths.in_dir(config.dataset.dir)
    unreadable: list[str] = []
    score_frames, unreadable_scores = _score_frames(config, unreadable)
    hourly_live = matrix_path(config.dataset.dir, "hourly", "live")
    daily_live = matrix_path(config.dataset.dir, "daily", "live")
    hourly_synthetic = matrix_path(config.dataset.dir, "hourly", "synthetic")
    return DashboardContext(
        config=config,
        now=now or datetime.now(tz=UTC),
        manifest=_try_json(
            paths.manifest, unreadable, label=f"dataset/{paths.manifest.name}"
        ),
        truth_minute=_try_parquet(
            paths.truth_minute,
            unreadable,
            label=f"dataset/{paths.truth_minute.name}",
        ),
        truth_hourly=_try_parquet(
            paths.truth_hourly,
            unreadable,
            label=f"dataset/{paths.truth_hourly.name}",
        ),
        truth_daily=_try_parquet(
            paths.truth_daily,
            unreadable,
            label=f"dataset/{paths.truth_daily.name}",
        ),
        qc=_qc_summary(config),
        hourly_matrix=_try_parquet(
            hourly_live, unreadable, label=f"dataset/{hourly_live.name}"
        ),
        daily_matrix=_try_parquet(
            daily_live, unreadable, label=f"dataset/{daily_live.name}"
        ),
        synthetic_hourly=_try_parquet(
            hourly_synthetic,
            unreadable,
            label=f"dataset/{hourly_synthetic.name}",
        ),
        score_frames=score_frames,
        unreadable_scores=unreadable_scores,
        history=_history(config, unreadable),
        latest_forecast=_latest_forecast(config, unreadable),
        releases=_releases(config, unreadable),
        alignment=_try_json(
            config.artifacts_dir / "alignment.json",
            unreadable,
            label="artifacts/alignment.json",
        ),
        drift=_try_json(
            config.artifacts_dir / "drift.json",
            unreadable,
            label="artifacts/drift.json",
        ),
        observability_states=_observability_states(config, unreadable),
        observability_history=_observability_history(config, unreadable),
        runs=_runs(config, unreadable),
        archive_location=_archive_location(config),
        unreadable_artifacts=tuple(sorted(set(unreadable))),
    )
