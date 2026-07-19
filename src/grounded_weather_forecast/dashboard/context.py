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
from grounded_weather_forecast.runs import RUNS_SCHEMA, load_runs, runs_path
from grounded_weather_forecast.serve.history import load_history
from grounded_weather_forecast.serve.observability import (
    OBSERVABILITY_HISTORY_SCHEMA,
    ObservabilitySnapshot,
    load_observability_history,
    load_observability_states,
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
    history: pl.DataFrame | None = None
    latest_forecast: Forecast | None = None
    releases: tuple[Mapping[str, object], ...] = ()
    alignment: Mapping[str, object] | None = None
    drift: Mapping[str, object] | None = None
    observability_states: tuple[ObservabilitySnapshot, ...] = ()
    observability_history: pl.DataFrame = field(default_factory=pl.DataFrame)
    runs: pl.DataFrame = field(default_factory=pl.DataFrame)
    archive_location: tuple[float, float] | None = None


def _try_parquet(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pl.read_parquet(path)
    except (OSError, pl.exceptions.PolarsError):
        return None


def _try_json(path: Path) -> Mapping[str, object] | None:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _score_frames(config: Config) -> dict[str, pl.DataFrame]:
    frames: dict[str, pl.DataFrame] = {}
    scores_dir = config.dataset.dir / "scores"
    for path in sorted(scores_dir.glob("scores_*.parquet")):
        try:
            frames[path.stem] = load_scores(path)
        except (OSError, MixedProvenanceError, pl.exceptions.PolarsError):
            continue
    return frames


def _latest_forecast(config: Config) -> Forecast | None:
    directory = config.predict.history_path.parent / "served_forecasts"
    if not directory.exists():
        return None
    documents = sorted(directory.glob("*.json"))
    if not documents:
        return None
    try:
        return Forecast.from_json(documents[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _releases(config: Config) -> tuple[Mapping[str, object], ...]:
    directory = config.artifacts_dir / "releases"
    if not directory.exists():
        return ()
    loaded = (_try_json(path) for path in sorted(directory.glob("*.json")))
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


def _history(config: Config) -> pl.DataFrame | None:
    try:
        frame = load_history(config.predict.history_path)
    except (OSError, pl.exceptions.PolarsError):
        return None
    return None if frame.is_empty() else frame


def _observability_states(config: Config) -> tuple[ObservabilitySnapshot, ...]:
    try:
        return load_observability_states(config.artifacts_dir)
    except (OSError, TypeError, ValueError):
        return ()


def _observability_history(config: Config) -> pl.DataFrame:
    try:
        return load_observability_history(config.artifacts_dir)
    except (OSError, ValueError, pl.exceptions.PolarsError):
        return pl.DataFrame(schema=OBSERVABILITY_HISTORY_SCHEMA)


def _runs(config: Config) -> pl.DataFrame:
    try:
        return load_runs(runs_path(config))
    except (OSError, ValueError, pl.exceptions.PolarsError):
        return pl.DataFrame(schema=RUNS_SCHEMA)


def _archive_location(config: Config) -> tuple[float, float] | None:
    try:
        return read_latest_archive_location(config.forecasts)
    except (OSError,):  # noqa: B013 - project style requires tuple clauses
        return None


def collect_context(config: Config, *, now: datetime | None = None) -> DashboardContext:
    paths = DatasetPaths.in_dir(config.dataset.dir)
    return DashboardContext(
        config=config,
        now=now or datetime.now(tz=UTC),
        manifest=_try_json(paths.manifest),
        truth_minute=_try_parquet(paths.truth_minute),
        truth_hourly=_try_parquet(paths.truth_hourly),
        truth_daily=_try_parquet(paths.truth_daily),
        qc=_qc_summary(config),
        hourly_matrix=_try_parquet(matrix_path(config.dataset.dir, "hourly", "live")),
        daily_matrix=_try_parquet(matrix_path(config.dataset.dir, "daily", "live")),
        synthetic_hourly=_try_parquet(
            matrix_path(config.dataset.dir, "hourly", "synthetic")
        ),
        score_frames=_score_frames(config),
        history=_history(config),
        latest_forecast=_latest_forecast(config),
        releases=_releases(config),
        alignment=_try_json(config.artifacts_dir / "alignment.json"),
        drift=_try_json(config.artifacts_dir / "drift.json"),
        observability_states=_observability_states(config),
        observability_history=_observability_history(config),
        runs=_runs(config),
        archive_location=_archive_location(config),
    )
