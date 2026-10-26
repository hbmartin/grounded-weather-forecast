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

from grounded_weather_forecast.artifact_labels import (
    CompatibilityNotice,
    artifact_label,
)
from grounded_weather_forecast.backtest.scores import load_scores
from grounded_weather_forecast.config import Config
from grounded_weather_forecast.contracts import MixedProvenanceError
from grounded_weather_forecast.dashboard.score_summary import (
    ScoreSummary,
    summarize_scores,
)
from grounded_weather_forecast.dataset.matrix import DatasetPaths, matrix_path
from grounded_weather_forecast.dataset.providers import read_latest_archive_location
from grounded_weather_forecast.evaluation import active_release_path
from grounded_weather_forecast.reports.eprocess import EProcessStore
from grounded_weather_forecast.runs import RUNS_SCHEMA, read_runs, runs_path
from grounded_weather_forecast.serve.approvals import (
    ApprovalSnapshot,
    read_approval_snapshot,
)
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
    """Everything zone builders read, with score summaries computed during collection."""

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
    score_summaries: Mapping[str, ScoreSummary] = field(default_factory=dict)
    unreadable_scores: tuple[str, ...] = ()
    # Artifacts that exist but could not be read. A missing file is a young
    # archive; a corrupt or permission-denied one is a fault, and collapsing
    # both to None let a broken deployment render as merely early.
    unreadable_artifacts: tuple[str, ...] = ()
    compatibility_notes: tuple[str, ...] = ()
    history: pl.DataFrame | None = None
    latest_forecast: Forecast | None = None
    latest_publish_attempt: Mapping[str, object] | None = None
    releases: tuple[Mapping[str, object], ...] = ()
    alignment: Mapping[str, object] | None = None
    drift: Mapping[str, object] | None = None
    truth_qc: Mapping[str, object] | None = None
    observability_states: tuple[ObservabilitySnapshot, ...] = ()
    observability_history: pl.DataFrame = field(default_factory=pl.DataFrame)
    runs: pl.DataFrame = field(default_factory=pl.DataFrame)
    archive_location: tuple[float, float] | None = None
    # The artifacts/history ledgers, keyed by ledger name; every key from
    # evidence.LEDGERS is always present (empty typed frame when absent).
    evidence_history: Mapping[str, pl.DataFrame] = field(default_factory=dict)
    compatibility_notices: tuple[CompatibilityNotice, ...] | None = None


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


def _score_summaries(
    config: Config, failures: list[str]
) -> tuple[Mapping[str, ScoreSummary], tuple[str, ...]]:
    """Decode each evaluation once and retain only small dashboard summaries."""
    summaries: dict[str, ScoreSummary] = {}
    unreadable: list[str] = []
    stores: dict[str, EProcessStore] = {}
    for path in sorted((config.dataset.dir / "scores").glob("scores_*.parquet")):
        try:
            frame = load_scores(path, compact=True)
        except (FileNotFoundError,):  # noqa: B013 -- project convention
            continue
        except (OSError, MixedProvenanceError, pl.exceptions.PolarsError):
            unreadable.append(path.stem)
            failures.append(f"dataset/scores/{path.name}")
            continue
        summaries[path.stem] = summarize_scores(config, path.stem, frame, stores)
        del frame
    return summaries, tuple(unreadable)


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


def _releases(
    config: Config,
    failures: list[str],
    *,
    now: datetime | None = None,
    snapshot: ApprovalSnapshot | None = None,
    compatibility_notes: list[str] | None = None,
    compatibility_notices: list[CompatibilityNotice] | None = None,
) -> tuple[Mapping[str, object], ...]:
    directory = config.artifacts_dir / "releases"
    snapshot = snapshot or read_approval_snapshot(directory, as_of=now)
    legacy = {
        str(item.path)
        for item in snapshot.diagnostics
        if item.category == "legacy_identity_missing"
    }
    for issue in snapshot.issues:
        label = artifact_label(Path(issue), config)
        if issue in legacy:
            notice = CompatibilityNotice(Path(issue))
            if compatibility_notes is not None:
                compatibility_notes.append(notice.label(config))
            if compatibility_notices is not None:
                compatibility_notices.append(notice)
            continue
        failures.append(label)
    from grounded_weather_forecast.reports.operations import (  # noqa: PLC0415
        _pointer_reference_ids,
        _ServedDocumentError,
    )

    try:
        _pointer_reference_ids(config)
    except (ValueError,) as error:  # noqa: B013
        path = (
            error.issue.path
            if isinstance(error, _ServedDocumentError)
            else active_release_path(config.artifacts_dir)
        )
        failures.append(artifact_label(path, config))
    return tuple(
        record.payload()
        for record in snapshot.records()
        if isinstance(record.release.get("selections"), dict)
    )


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


def _evidence_history(
    config: Config,
    failures: list[str],
    *,
    now: datetime | None = None,
    snapshot: ApprovalSnapshot | None = None,
) -> dict[str, pl.DataFrame]:
    """All five history ledgers, absent files as empty typed frames."""
    from grounded_weather_forecast.reports.evidence import (  # noqa: PLC0415
        LEDGERS,
        ledger_path,
        normalize_churn,
        read_ledger,
    )

    frames: dict[str, pl.DataFrame] = {}
    for name, spec in LEDGERS.items():
        path = ledger_path(config, spec)
        try:
            frame = read_ledger(path, spec.schema)
            frames[name] = (
                normalize_churn(
                    frame,
                    config.artifacts_dir / "releases",
                    as_of=now or datetime.now(UTC),
                    snapshot=snapshot,
                )
                if name == "churn"
                else frame
            )
        except (OSError, ValueError, pl.exceptions.PolarsError):
            if path.exists():
                failures.append(f"artifacts/history/{spec.name}.parquet")
            frames[name] = pl.DataFrame(schema=spec.schema)
    return frames


def collect_context(
    config: Config,
    *,
    now: datetime | None = None,
    score_summaries: Mapping[str, ScoreSummary] | None = None,
    score_read_failures: tuple[str, ...] = (),
) -> DashboardContext:
    """Collect artifacts, or reuse a complete supplied score-summary snapshot."""
    paths = DatasetPaths.in_dir(config.dataset.dir)
    approval_snapshot = read_approval_snapshot(
        config.artifacts_dir / "releases", as_of=now
    )
    unreadable: list[str] = []
    compatibility_notes: list[str] = []
    compatibility_notices: list[CompatibilityNotice] = []
    unreadable_scores: tuple[str, ...] = ()
    if score_summaries is None:
        score_summaries, unreadable_scores = _score_summaries(config, unreadable)
    else:
        score_summaries = dict(score_summaries)
    unreadable_scores = tuple(
        sorted({*unreadable_scores, *(Path(name).stem for name in score_read_failures)})
    )
    unreadable.extend(f"dataset/scores/{name}" for name in score_read_failures)
    hourly_live = matrix_path(config.dataset.dir, "hourly", "live")
    daily_live = matrix_path(config.dataset.dir, "daily", "live")
    hourly_synthetic = matrix_path(config.dataset.dir, "hourly", "synthetic")
    return DashboardContext(
        config=config,
        now=approval_snapshot.cutoff,
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
        score_summaries=score_summaries,
        unreadable_scores=unreadable_scores,
        history=_history(config, unreadable),
        latest_forecast=_latest_forecast(config, unreadable),
        latest_publish_attempt=_try_json(
            config.artifacts_dir / "latest_publish_attempt.json",
            unreadable,
            label="artifacts/latest_publish_attempt.json",
        ),
        releases=_releases(
            config,
            unreadable,
            now=now,
            snapshot=approval_snapshot,
            compatibility_notes=compatibility_notes,
            compatibility_notices=compatibility_notices,
        ),
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
        truth_qc=_try_json(
            config.artifacts_dir / "truth_qc.json",
            unreadable,
            label="artifacts/truth_qc.json",
        ),
        observability_states=_observability_states(config, unreadable),
        observability_history=_observability_history(config, unreadable),
        runs=_runs(config, unreadable),
        archive_location=_archive_location(config),
        evidence_history=_evidence_history(
            config, unreadable, now=now, snapshot=approval_snapshot
        ),
        unreadable_artifacts=tuple(sorted(set(unreadable))),
        compatibility_notes=tuple(compatibility_notes),
        compatibility_notices=tuple(compatibility_notices),
    )
