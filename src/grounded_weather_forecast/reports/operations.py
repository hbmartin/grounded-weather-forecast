"""Operational evidence collectors: the pipeline's edges, ledgered.

The quality ledgers (``reports/evidence.py``) watch the middle of the
pipeline — skill, churn, promises. This module watches the edges, where
the two historical week-long silent failures lived: a dead station logger
(Jul 20-27 truth hole, label-less forever) and a predict plist that failed
argparse for a week (stale published forecast, no served history). Every
collector is read-side over data the pipeline already stores; nothing here
adds collection, config keys, or serving-path risk.

Five concerns, each an append-only ledger written by ``report``:

- **pipeline** — one freshness row per day (truth/collector/serving/publish
  ages, run counts) plus hard-threshold alarm strings.
- **provider_health** — per provider per day: success rate, latency, point
  volumes, and the maximum stored lead; contraction against the provider's
  own trailing baseline is the plan-downgrade / quota-change detector.
- **build_funnel** — rows and max lead per source at each storage layer
  (collector -> long -> matrix), so silent loss between layers is a visible
  trend instead of an archaeology project.
- **changes** — config-fingerprint and code-version transitions with the
  changed config keys, so "did quality shift after X?" has a reliable X
  (config.toml is gitignored and otherwise has no history).
- **evaluations** — a catalog row per scores file (size, folds, spans) that
  outlives the file, making ``prune-scores`` safe for trend questions.

Every collector degrades to nulls rather than failing the report: a
missing collector database on a fresh deployment is a fact worth recording,
not a crash.
"""

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import polars as pl
from filelock import Timeout

from grounded_weather_forecast.config import Config
from grounded_weather_forecast.dataset.providers import source_slug
from grounded_weather_forecast.dataset.station import sqlite_uri
from grounded_weather_forecast.evaluation import code_identity, config_fingerprint
from grounded_weather_forecast.reports import evidence

_COLLECTOR_ERRORS = (OSError, ValueError, sqlite3.Error, pl.exceptions.PolarsError)

_FUNNEL_WINDOW_DAYS = 14
_BASELINE_WINDOW_DAYS = 14
_BASELINE_MIN_DAYS = 3

_TRUTH_AGE_ALARM_MINUTES = 120.0
_COLLECTOR_AGE_ALARM_MINUTES = 180.0
_SERVED_AGE_ALARM_MINUTES = 180.0
_DOCUMENT_AGE_ALARM_MINUTES = 180.0
# The hourly predict cadence yields ~24 runs/day; half that means the
# scheduler has been broken for at least half a day.
_MIN_PREDICT_RUNS_24H = 12
# A healthy station logs a sample about every 61 s (~1400-1900/day); the
# 2026-08-04..06 half-rate episode ran at ~810/day, so the floor sits above
# that. The baseline-relative check below covers milder degradation.
_MIN_TRUTH_SAMPLES_24H = 1000
_TRUTH_BASELINE_SHARE = 0.7
_SUCCESS_RATE_ALARM = 0.8
_DAILY_LEAD_CONTRACTION_DAYS = 1.0
_HOURLY_LEAD_CONTRACTION_HOURS = 12.0

_REDACTED_MARKERS = ("token", "secret", "password")
_DETAIL_MAX_CHARS = 400

# Shared with the dashboard operations zone so panel statuses and report
# alarms can never disagree about what "stale" means.
FRESHNESS_THRESHOLDS: Mapping[str, float] = {
    "truth_age_minutes": _TRUTH_AGE_ALARM_MINUTES,
    "collector_age_minutes": _COLLECTOR_AGE_ALARM_MINUTES,
    "served_history_age_minutes": _SERVED_AGE_ALARM_MINUTES,
    "forecast_document_age_minutes": _DOCUMENT_AGE_ALARM_MINUTES,
}

_SNAPSHOT_NAME = "identity_snapshot.json"


@dataclass(frozen=True, slots=True)
class OperationsReport:
    """Everything the pipeline-health report section renders."""

    freshness: pl.DataFrame
    alarms: tuple[str, ...]
    provider_health: pl.DataFrame
    contractions: tuple[str, ...]
    funnel: pl.DataFrame
    changes: pl.DataFrame
    catalog: pl.DataFrame


def _open_collector(config: Config) -> sqlite3.Connection:
    forecasts = config.forecasts
    if not forecasts.db_path.exists():
        msg = f"collector archive {forecasts.db_path} not found"
        raise OSError(msg)
    return sqlite3.connect(
        sqlite_uri(forecasts.db_path, immutable=forecasts.immutable), uri=True
    )


def _age_minutes(moment: datetime | None, now: datetime) -> float | None:
    if moment is None:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return (now - moment).total_seconds() / 60.0


def _truth_freshness(config: Config, now: datetime) -> tuple[float | None, int | None]:
    """Age and 24h sample count straight from the station database.

    The truth parquet only refreshes at build-dataset, so its newest row
    measures the build cadence, not the logger — and a dead logger is
    exactly the failure this row exists to catch on day one.
    """
    station = config.station
    if not station.db_path.exists():
        return None, None
    connection = sqlite3.connect(
        sqlite_uri(station.db_path, immutable=station.immutable), uri=True
    )
    try:
        newest = connection.execute("SELECT MAX(ts) FROM observations").fetchone()[0]
        cutoff = (now - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        count = connection.execute(
            "SELECT COUNT(*) FROM observations WHERE ts >= ?", (cutoff,)
        ).fetchone()[0]
    finally:
        connection.close()
    if newest is None:
        return None, 0
    # aw2sqlite ``ts`` text is UTC-naive with optional microseconds.
    moment = datetime.fromisoformat(str(newest)).replace(tzinfo=UTC)
    return _age_minutes(moment, now), int(count)


def _collector_freshness(
    config: Config, now: datetime
) -> tuple[float | None, int | None, float | None]:
    cutoff = now - timedelta(hours=24)
    connection = _open_collector(config)
    try:
        newest = connection.execute(
            "SELECT MAX(completed_at) FROM forecast_runs"
        ).fetchone()[0]
        runs = connection.execute(
            "SELECT COUNT(*) FROM forecast_runs WHERE completed_at >= ?",
            (cutoff.isoformat(),),
        ).fetchone()[0]
        attempts, ok = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(status = 'success'), 0) "
            "FROM provider_results WHERE fetched_at_unix >= ?",
            (int(cutoff.timestamp()),),
        ).fetchone()
    finally:
        connection.close()
    age = None
    if newest is not None:
        age = _age_minutes(datetime.fromisoformat(str(newest)), now)
    rate = (ok / attempts) if attempts else None
    return age, int(runs), rate


def _parquet_max(path: Path, column: str) -> datetime | None:
    if not path.exists():
        return None
    value = pl.scan_parquet(path).select(pl.col(column).max()).collect().item()
    return cast("datetime | None", value)


def _document_age(config: Config, now: datetime) -> float | None:
    directory = config.predict.history_path.parent / "served_forecasts"
    documents = sorted(directory.glob("*.json")) if directory.exists() else []
    if not documents:
        return None
    newest = max(document.stat().st_mtime for document in documents)
    return _age_minutes(datetime.fromtimestamp(newest, tz=UTC), now)


def _run_counts(runs_frame: pl.DataFrame, now: datetime) -> tuple[int, int] | None:
    """(ok predicts, failed commands) in 24h; None while the ledger is young."""
    if runs_frame.is_empty():
        return None
    recent = runs_frame.filter(pl.col("started_at") >= now - timedelta(hours=24))
    predicts = recent.filter(
        (pl.col("command") == "predict") & (pl.col("exit_code") == 0)
    ).height
    # A live backtest exiting 1 means "no folds yet" and is tolerated by the
    # maintenance job; counting it would alarm on every young archive.
    failed = recent.filter(
        pl.col("exit_code").is_not_null()
        & (pl.col("exit_code") != 0)
        & ~((pl.col("command") == "backtest") & (pl.col("exit_code") == 1))
    ).height
    return predicts, failed


def _stale(name: str, age: float | None, limit: float) -> list[str]:
    if age is None:
        return [f"{name} unavailable"]
    if age > limit:
        return [f"{name} stale ({age:.0f}m > {limit:.0f}m)"]
    return []


def _baseline_truth_alarm(
    truth_samples: int | None, history: pl.DataFrame | None, now: datetime
) -> list[str]:
    """Flag a sample rate well under the station's own recent norm.

    The fixed floor catches gross failure; this catches partial degradation
    (the 2026-08-04..06 half-rate episode ran at ~810 samples/day — above
    the old 720 floor, far below the ~1900 norm) without hardcoding the
    station's cadence.
    """
    if truth_samples is None or history is None or history.is_empty():
        return []
    window_start = (now - timedelta(days=_BASELINE_WINDOW_DAYS)).date()
    baseline = history.filter(
        (pl.col("as_of_date") >= window_start)
        & (pl.col("as_of_date") < now.date())
        & pl.col("truth_samples_24h").is_not_null()
    )
    if baseline["as_of_date"].n_unique() < _BASELINE_MIN_DAYS:
        return []
    median = cast("float", baseline["truth_samples_24h"].median())
    if median > 0 and truth_samples < _TRUTH_BASELINE_SHARE * median:
        return [
            f"thin truth vs baseline ({truth_samples} < "
            f"{_TRUTH_BASELINE_SHARE:.0%} of 14d median {median:.0f})"
        ]
    return []


def freshness_row(
    config: Config,
    runs_frame: pl.DataFrame,
    *,
    now: datetime,
    pipeline_history: pl.DataFrame | None = None,
) -> tuple[dict[str, object], tuple[str, ...]]:
    """One end-to-end freshness snapshot plus its threshold alarms."""
    try:
        truth_age, truth_samples = _truth_freshness(config, now)
    except _COLLECTOR_ERRORS:
        truth_age, truth_samples = None, None
    try:
        collector_age, collector_runs, success_rate = _collector_freshness(config, now)
    except _COLLECTOR_ERRORS:
        collector_age, collector_runs, success_rate = None, None, None
    try:
        served_age = _age_minutes(
            _parquet_max(config.predict.history_path, "issued_at"), now
        )
    except _COLLECTOR_ERRORS:
        served_age = None
    try:
        document_age = _document_age(config, now)
    except _COLLECTOR_ERRORS:
        document_age = None
    counts = _run_counts(runs_frame, now)

    alarms = _stale("truth", truth_age, _TRUTH_AGE_ALARM_MINUTES)
    alarms += _stale("collector", collector_age, _COLLECTOR_AGE_ALARM_MINUTES)
    alarms += _stale("served history", served_age, _SERVED_AGE_ALARM_MINUTES)
    alarms += _stale("forecast document", document_age, _DOCUMENT_AGE_ALARM_MINUTES)
    if truth_samples is not None and truth_samples < _MIN_TRUTH_SAMPLES_24H:
        alarms.append(
            f"thin truth ({truth_samples} samples/24h < {_MIN_TRUTH_SAMPLES_24H})"
        )
    alarms += _baseline_truth_alarm(truth_samples, pipeline_history, now)
    if counts is not None:
        predicts, failed = counts
        if predicts < _MIN_PREDICT_RUNS_24H:
            alarms.append(
                f"few predict runs ({predicts}/24h < {_MIN_PREDICT_RUNS_24H})"
            )
        if failed:
            alarms.append(f"failed cli runs in 24h: {failed}")
    row: dict[str, object] = {
        "recorded_at": now,
        "as_of_date": now.date(),
        "truth_age_minutes": truth_age,
        "truth_samples_24h": truth_samples,
        "collector_age_minutes": collector_age,
        "collector_runs_24h": collector_runs,
        "provider_success_rate_24h": success_rate,
        "served_history_age_minutes": served_age,
        "forecast_document_age_minutes": document_age,
        "predict_runs_24h": None if counts is None else counts[0],
        "failed_runs_24h": None if counts is None else counts[1],
        "alarms": ", ".join(alarms),
        "code_version": code_identity(),
        "config_fingerprint": config_fingerprint(config),
    }
    return row, tuple(alarms)


def _provider_results_24h(
    connection: sqlite3.Connection, cutoff_unix: int
) -> pl.DataFrame:
    rows = connection.execute(
        "SELECT provider, status, latency_ms FROM provider_results "
        "WHERE fetched_at_unix >= ?",
        (cutoff_unix,),
    ).fetchall()
    frame = pl.DataFrame(
        rows,
        schema={"provider": pl.String, "status": pl.String, "latency_ms": pl.Float64},
        orient="row",
    )
    return frame.group_by("provider").agg(
        pl.len().alias("runs_24h"),
        (pl.col("status") == "success").sum().alias("ok_24h"),
        pl.col("latency_ms").median().alias("median_latency_ms"),
    )


def _provider_leads_24h(
    connection: sqlite3.Connection, cutoff_unix: int
) -> pl.DataFrame:
    hourly = connection.execute(
        "SELECT sf.provider, COUNT(*), "
        "MAX((p.timestamp_unix - pr.fetched_at_unix) / 3600.0) "
        "FROM hourly_points AS p "
        "JOIN source_forecasts AS sf ON sf.id = p.source_forecast_id "
        "JOIN provider_results AS pr ON pr.id = sf.provider_result_id "
        "WHERE pr.fetched_at_unix >= ? GROUP BY sf.provider",
        (cutoff_unix,),
    ).fetchall()
    daily = connection.execute(
        "SELECT sf.provider, COUNT(*), "
        "MAX(julianday(date(p.forecast_date)) - julianday(date(pr.fetched_at))) "
        "FROM daily_points AS p "
        "JOIN source_forecasts AS sf ON sf.id = p.source_forecast_id "
        "JOIN provider_results AS pr ON pr.id = sf.provider_result_id "
        "WHERE pr.fetched_at_unix >= ? GROUP BY sf.provider",
        (cutoff_unix,),
    ).fetchall()
    hourly_frame = pl.DataFrame(
        hourly,
        schema={
            "provider": pl.String,
            "hourly_rows_24h": pl.Int64,
            "max_hourly_lead_hours": pl.Float64,
        },
        orient="row",
    )
    daily_frame = pl.DataFrame(
        daily,
        schema={
            "provider": pl.String,
            "daily_rows_24h": pl.Int64,
            "max_daily_lead_days": pl.Float64,
        },
        orient="row",
    )
    return hourly_frame.join(daily_frame, on="provider", how="full", coalesce=True)


def provider_health_rows(config: Config, *, now: datetime) -> pl.DataFrame:
    """Per-provider collector health over the trailing 24 hours."""
    cutoff_unix = int((now - timedelta(hours=24)).timestamp())
    try:
        connection = _open_collector(config)
        try:
            results = _provider_results_24h(connection, cutoff_unix)
            leads = _provider_leads_24h(connection, cutoff_unix)
        finally:
            connection.close()
    except _COLLECTOR_ERRORS:
        return pl.DataFrame(schema=evidence.PROVIDER_HEALTH_SCHEMA)
    if results.is_empty():
        return pl.DataFrame(schema=evidence.PROVIDER_HEALTH_SCHEMA)
    joined = results.join(leads, on="provider", how="full", coalesce=True)
    return (
        joined.with_columns(
            pl.lit(now).dt.replace_time_zone("UTC").alias("recorded_at"),
            pl.lit(now.date()).alias("as_of_date"),
            (pl.col("ok_24h") / pl.col("runs_24h")).alias("success_rate"),
            pl.lit(code_identity()).alias("code_version"),
        )
        .select(evidence.PROVIDER_HEALTH_SCHEMA.names())
        .cast(evidence.PROVIDER_HEALTH_SCHEMA, strict=False)
        .sort("provider")
    )


def provider_contractions(
    fresh: pl.DataFrame, history: pl.DataFrame, *, now: datetime
) -> tuple[str, ...]:
    """Today's rows against each provider's own trailing-median baseline."""
    notes: list[str] = []
    baseline = pl.DataFrame()
    if not history.is_empty():
        window_start = (now - timedelta(days=_BASELINE_WINDOW_DAYS)).date()
        baseline = (
            history.filter(
                (pl.col("as_of_date") >= window_start)
                & (pl.col("as_of_date") < now.date())
            )
            .group_by("provider")
            .agg(
                pl.col("as_of_date").n_unique().alias("days"),
                pl.col("max_daily_lead_days").median().alias("daily_median"),
                pl.col("max_hourly_lead_hours").median().alias("hourly_median"),
            )
            .filter(pl.col("days") >= _BASELINE_MIN_DAYS)
        )
    by_provider = (
        {str(r["provider"]): r for r in baseline.iter_rows(named=True)}
        if not baseline.is_empty()
        else {}
    )
    for row in fresh.iter_rows(named=True):
        provider = str(row["provider"])
        rate = row["success_rate"]
        if rate is not None and rate < _SUCCESS_RATE_ALARM:
            notes.append(f"{provider}: success rate {rate:.0%}")
        base = by_provider.get(provider)
        if base is None:
            continue
        daily, daily_median = row["max_daily_lead_days"], base["daily_median"]
        if (
            daily is not None
            and daily_median is not None
            and daily < daily_median - _DAILY_LEAD_CONTRACTION_DAYS
        ):
            notes.append(
                f"{provider}: daily lead {daily:.0f}d < median {daily_median:.0f}d"
            )
        hourly, hourly_median = row["max_hourly_lead_hours"], base["hourly_median"]
        if (
            hourly is not None
            and hourly_median is not None
            and hourly < hourly_median - _HOURLY_LEAD_CONTRACTION_HOURS
        ):
            notes.append(
                f"{provider}: hourly lead {hourly:.0f}h < median {hourly_median:.0f}h"
            )
    return tuple(notes)


def _collector_funnel_layer(config: Config, *, now: datetime) -> pl.DataFrame:
    cutoff_unix = int((now - timedelta(days=_FUNNEL_WINDOW_DAYS)).timestamp())
    lead_sql = {
        "hourly": (
            "SELECT sf.provider, sf.model, COUNT(*), "
            "MAX((p.timestamp_unix - pr.fetched_at_unix) / 3600.0) "
            "FROM hourly_points AS p "
        ),
        "daily": (
            "SELECT sf.provider, sf.model, COUNT(*), "
            "MAX(julianday(date(p.forecast_date)) - julianday(date(pr.fetched_at))) "
            "FROM daily_points AS p "
        ),
    }
    joins = (
        "JOIN source_forecasts AS sf ON sf.id = p.source_forecast_id "
        "JOIN provider_results AS pr ON pr.id = sf.provider_result_id "
        "WHERE pr.fetched_at_unix >= ? GROUP BY sf.provider, sf.model"
    )
    records: list[dict[str, object]] = []
    connection = _open_collector(config)
    try:
        for granularity, head in lead_sql.items():
            for provider, model, rows, lead in connection.execute(
                head + joins, (cutoff_unix,)
            ).fetchall():
                records.append(
                    {
                        "granularity": granularity,
                        "source": source_slug(str(provider), str(model)),
                        "collector_rows": int(rows),
                        "collector_max_lead": None if lead is None else float(lead),
                    }
                )
    finally:
        connection.close()
    return pl.DataFrame(
        records,
        schema={
            "granularity": pl.String,
            "source": pl.String,
            "collector_rows": pl.Int64,
            "collector_max_lead": pl.Float64,
        },
    )


def _long_funnel_layer(config: Config, *, now: datetime) -> pl.DataFrame:
    start = now - timedelta(days=_FUNNEL_WINDOW_DAYS)
    layers: list[pl.DataFrame] = []
    hourly_path = config.dataset.dir / "forecasts_long.parquet"
    if hourly_path.exists():
        layers.append(
            pl.scan_parquet(hourly_path)
            .filter(pl.col("fetched_at") >= start)
            .group_by("source")
            .agg(
                pl.len().alias("long_rows"),
                pl.col("lead_hours").max().alias("long_max_lead"),
            )
            .with_columns(pl.lit("hourly").alias("granularity"))
            .collect()
        )
    daily_path = config.dataset.dir / "daily_long.parquet"
    if daily_path.exists():
        layers.append(
            pl.scan_parquet(daily_path)
            .filter(pl.col("fetched_at") >= start)
            .group_by("source")
            .agg(
                pl.len().alias("long_rows"),
                (pl.col("forecast_date") - pl.col("fetched_at").dt.date())
                .dt.total_days()
                .max()
                .cast(pl.Float64)
                .alias("long_max_lead"),
            )
            .with_columns(pl.lit("daily").alias("granularity"))
            .collect()
        )
    if not layers:
        return pl.DataFrame(
            schema={
                "source": pl.String,
                "long_rows": pl.Int64,
                "long_max_lead": pl.Float64,
                "granularity": pl.String,
            }
        )
    return pl.concat(layers, how="vertical_relaxed")


def _matrix_source_slice(
    matrix: pl.DataFrame, columns: Sequence[str], lead_column: str
) -> tuple[int, float | None]:
    present = [column for column in columns if column in matrix.columns]
    if not present:
        return 0, None
    subset = matrix.filter(
        pl.any_horizontal([pl.col(column).is_not_null() for column in present])
    )
    if subset.is_empty():
        return 0, None
    lead = cast("float | None", subset[lead_column].max())
    return subset.height, None if lead is None else float(lead)


def _matrix_sources(matrix: pl.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    return sorted(
        {
            column.split("__")[1]
            for column in matrix.columns
            if column.startswith(prefixes) and len(column.split("__")) >= 3
        }
    )


def _hourly_matrix_layer(matrix: pl.DataFrame, now: datetime) -> pl.DataFrame:
    start = now - timedelta(days=_FUNNEL_WINDOW_DAYS)
    recent = matrix.filter(pl.col("issue_time") >= start)
    records = []
    for source in _matrix_sources(recent, ("fx__",)):
        columns = [c for c in recent.columns if c.startswith(f"fx__{source}__")]
        rows, lead = _matrix_source_slice(recent, columns, "lead_hours")
        records.append(
            {
                "granularity": "hourly",
                "source": source,
                "matrix_rows": rows,
                "matrix_max_lead": lead,
                "matrix_native_max_lead": None,
                "matrix_path_max_lead": None,
            }
        )
    return pl.DataFrame(records)


def _daily_matrix_layer(matrix: pl.DataFrame, now: datetime) -> pl.DataFrame:
    start = now - timedelta(days=_FUNNEL_WINDOW_DAYS)
    recent = matrix.filter(pl.col("issue_time") >= start)
    records = []
    for source in _matrix_sources(recent, ("fxd__", "path__")):
        native_columns = [c for c in recent.columns if c.startswith(f"fxd__{source}__")]
        path_columns = [c for c in recent.columns if c.startswith(f"path__{source}__")]
        rows, lead = _matrix_source_slice(
            recent, native_columns + path_columns, "lead_days"
        )
        _, native_lead = _matrix_source_slice(recent, native_columns, "lead_days")
        _, path_lead = _matrix_source_slice(recent, path_columns, "lead_days")
        records.append(
            {
                "granularity": "daily",
                "source": source,
                "matrix_rows": rows,
                "matrix_max_lead": lead,
                "matrix_native_max_lead": native_lead,
                "matrix_path_max_lead": path_lead,
            }
        )
    return pl.DataFrame(records)


def _matrix_funnel_layer(config: Config, *, now: datetime) -> pl.DataFrame:
    layers: list[pl.DataFrame] = []
    hourly_path = config.dataset.dir / "hourly_matrix_live.parquet"
    if hourly_path.exists():
        layers.append(_hourly_matrix_layer(pl.read_parquet(hourly_path), now))
    daily_path = config.dataset.dir / "daily_matrix_live.parquet"
    if daily_path.exists():
        layers.append(_daily_matrix_layer(pl.read_parquet(daily_path), now))
    layers = [layer for layer in layers if not layer.is_empty()]
    if not layers:
        return pl.DataFrame(
            schema={
                "granularity": pl.String,
                "source": pl.String,
                "matrix_rows": pl.Int64,
                "matrix_max_lead": pl.Float64,
                "matrix_native_max_lead": pl.Float64,
                "matrix_path_max_lead": pl.Float64,
            }
        )
    return pl.concat(layers, how="vertical_relaxed")


def build_funnel_rows(config: Config, *, now: datetime) -> pl.DataFrame:
    """Rows and max lead per source at each storage layer, trailing 14 days.

    Lead units follow the granularity: hours on hourly rows, days on daily.
    """
    try:
        collector = _collector_funnel_layer(config, now=now)
    except _COLLECTOR_ERRORS:
        collector = pl.DataFrame(
            schema={
                "granularity": pl.String,
                "source": pl.String,
                "collector_rows": pl.Int64,
                "collector_max_lead": pl.Float64,
            }
        )
    try:
        long_layer = _long_funnel_layer(config, now=now)
        matrix_layer = _matrix_funnel_layer(config, now=now)
    except _COLLECTOR_ERRORS:
        return pl.DataFrame(schema=evidence.BUILD_FUNNEL_SCHEMA)
    keys = ["granularity", "source"]
    spine = pl.concat(
        [collector.select(keys), long_layer.select(keys), matrix_layer.select(keys)]
    ).unique()
    if spine.is_empty():
        return pl.DataFrame(schema=evidence.BUILD_FUNNEL_SCHEMA)
    return (
        spine.join(collector, on=keys, how="left")
        .join(long_layer, on=keys, how="left")
        .join(matrix_layer, on=keys, how="left")
        .with_columns(
            pl.lit(now).dt.replace_time_zone("UTC").alias("recorded_at"),
            pl.lit(now.date()).alias("as_of_date"),
            pl.lit(code_identity()).alias("code_version"),
        )
        .select(evidence.BUILD_FUNNEL_SCHEMA.names())
        .cast(evidence.BUILD_FUNNEL_SCHEMA, strict=False)
        .sort(keys)
    )


def _flatten_config(value: object, prefix: str = "") -> dict[str, str]:
    if is_dataclass(value) and not isinstance(value, type):
        flat: dict[str, str] = {}
        for field in fields(value):
            flat |= _flatten_config(
                getattr(value, field.name), f"{prefix}{field.name}."
            )
        return flat
    if isinstance(value, Mapping):
        nested: dict[str, str] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            nested |= _flatten_config(item, f"{prefix}{key}.")
        return nested
    key = prefix.rstrip(".")
    leaf = key.rsplit(".", maxsplit=1)[-1]
    if any(marker in leaf for marker in _REDACTED_MARKERS):
        return {key: "<redacted>"}
    return {key: repr(value)}


def _changed_keys(before: Mapping[str, str], after: Mapping[str, str]) -> str:
    changed = sorted(
        set(before) ^ set(after)
        | {key for key in set(before) & set(after) if before[key] != after[key]}
    )
    detail = ", ".join(changed)
    if len(detail) > _DETAIL_MAX_CHARS:
        detail = detail[: _DETAIL_MAX_CHARS - 3] + "..."
    return detail


def identity_changes(config: Config, *, now: datetime) -> pl.DataFrame:
    """Config-fingerprint / code-version transitions since the last report.

    The first report writes the baseline snapshot and records nothing: with
    no prior state there is no transition to attribute evidence shifts to.
    """
    empty = pl.DataFrame(schema=evidence.CHANGES_SCHEMA)
    path = config.artifacts_dir / "observability" / _SNAPSHOT_NAME
    current = {
        "config_fingerprint": config_fingerprint(config),
        "code_version": code_identity(),
        "config": _flatten_config(config),
    }
    try:
        previous = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
    except _COLLECTOR_ERRORS:
        return empty
    if previous is None:
        return empty
    rows: list[dict[str, object]] = []
    if previous.get("config_fingerprint") != current["config_fingerprint"]:
        rows.append(
            {
                "kind": "config",
                "from_value": str(previous.get("config_fingerprint")),
                "to_value": current["config_fingerprint"],
                "detail": _changed_keys(
                    cast("Mapping[str, str]", previous.get("config", {})),
                    cast("Mapping[str, str]", current["config"]),
                ),
            }
        )
    if previous.get("code_version") != current["code_version"]:
        rows.append(
            {
                "kind": "code",
                "from_value": str(previous.get("code_version")),
                "to_value": current["code_version"],
                "detail": None,
            }
        )
    if not rows:
        return empty
    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.lit(now).dt.replace_time_zone("UTC").alias("recorded_at"),
            pl.lit(now.date()).alias("as_of_date"),
            pl.lit(code_identity()).alias("code_version"),
        )
        .select(evidence.CHANGES_SCHEMA.names())
        .cast(evidence.CHANGES_SCHEMA, strict=False)
    )


def evaluation_catalog_row(path: Path, scores: pl.DataFrame) -> dict[str, object]:
    """One immutable catalog row for a scores file the report just loaded.

    Missing columns degrade to nulls rather than failing the report: this is
    telemetry over a frame whose schema the report loop does not own.
    """
    parts = path.stem.split("_")
    usable = not scores.is_empty()
    fold_sizes = (
        scores.group_by("fold_origin").len()["len"]
        if usable and "fold_origin" in scores.columns
        else pl.Series("len", [], dtype=pl.UInt32)
    )
    issues = scores["issue_time"] if usable and "issue_time" in scores.columns else None
    return {
        "evaluation_id": parts[-1],
        "file_name": path.name,
        "product": parts[1] if len(parts) > 2 else None,
        "source_kind": parts[2] if len(parts) > 3 else None,
        "window": "_".join(parts[3:-1]) if len(parts) > 4 else None,
        "rows": scores.height,
        "n_methods": (
            scores["method_id"].n_unique()
            if usable and "method_id" in scores.columns
            else 0
        ),
        "n_folds": fold_sizes.len(),
        "fold_rows_min": (
            None if fold_sizes.is_empty() else cast("int", fold_sizes.min())
        ),
        "fold_rows_median": (
            None if fold_sizes.is_empty() else cast("float", fold_sizes.median())
        ),
        "issue_min": None if issues is None else issues.min(),
        "issue_max": None if issues is None else issues.max(),
        "file_size_mb": path.stat().st_size / 1e6 if path.exists() else None,
        "code_version": code_identity(),
    }


def record_operations(
    config: Config,
    *,
    runs_frame: pl.DataFrame,
    catalog_rows: Sequence[Mapping[str, object]],
    now: datetime | None = None,
) -> OperationsReport:
    """Collect every operational surface, append the ledgers, return the view."""
    moment = now or datetime.now(tz=UTC)
    pipeline_history = evidence.load_ledger(
        evidence.ledger_path(config, evidence.PIPELINE_LEDGER),
        evidence.PIPELINE_SCHEMA,
    )
    row, alarms = freshness_row(
        config, runs_frame, now=moment, pipeline_history=pipeline_history
    )
    freshness = pl.DataFrame([row]).cast(evidence.PIPELINE_SCHEMA, strict=False)
    provider = provider_health_rows(config, now=moment)
    history = evidence.load_ledger(
        evidence.ledger_path(config, evidence.PROVIDER_HEALTH_LEDGER),
        evidence.PROVIDER_HEALTH_SCHEMA,
    )
    contractions = provider_contractions(provider, history, now=moment)
    funnel = build_funnel_rows(config, now=moment)
    changes = identity_changes(config, now=moment)
    catalog = (
        pl.DataFrame(list(catalog_rows))
        .with_columns(pl.lit(moment).dt.replace_time_zone("UTC").alias("recorded_at"))
        .select(evidence.EVALUATIONS_SCHEMA.names())
        .cast(evidence.EVALUATIONS_SCHEMA, strict=False)
        if catalog_rows
        else pl.DataFrame(schema=evidence.EVALUATIONS_SCHEMA)
    )
    for fresh, spec in (
        (freshness, evidence.PIPELINE_LEDGER),
        (provider, evidence.PROVIDER_HEALTH_LEDGER),
        (funnel, evidence.BUILD_FUNNEL_LEDGER),
        (changes, evidence.CHANGES_LEDGER),
        (catalog, evidence.EVALUATIONS_LEDGER),
    ):
        evidence.append_ledger(
            fresh, evidence.ledger_path(config, spec), spec, now=moment
        )
    return OperationsReport(
        freshness=freshness,
        alarms=alarms,
        provider_health=provider,
        contractions=contractions,
        funnel=funnel,
        changes=changes,
        catalog=catalog,
    )


# --- scores-directory housekeeping -----------------------------------------

_KEEP_NEWEST_PER_GROUP = 3
_PROTECT_RELEASE_DAYS = 7


@dataclass(frozen=True, slots=True)
class PruneResult:
    """What prune-scores did (or would do, under --dry-run)."""

    deleted: tuple[Path, ...]
    kept: tuple[Path, ...]
    skipped: tuple[str, ...]
    freed_mb: float


def _protected_evaluations(config: Config, *, now: datetime) -> set[str]:
    """Evaluation ids referenced by any release promoted in the last 30 days."""
    directory = config.artifacts_dir / "releases"
    horizon = now - timedelta(days=_PROTECT_RELEASE_DAYS)
    protected: set[str] = set()
    for path in sorted(directory.glob("*.json")) if directory.exists() else []:
        try:
            release = json.loads(path.read_text(encoding="utf-8"))
            promoted_at = datetime.fromisoformat(str(release["promoted_at"]))
        except (OSError, ValueError, KeyError):
            continue
        if promoted_at.tzinfo is None:
            promoted_at = promoted_at.replace(tzinfo=UTC)
        if promoted_at >= horizon:
            protected |= {str(e) for e in release.get("evaluation_ids", [])}
    return protected


def prune_scores_files(
    config: Config, *, dry_run: bool, now: datetime | None = None
) -> PruneResult:
    """Delete superseded scores files; the catalog ledger keeps their story.

    Retention: the newest ``_KEEP_NEWEST_PER_GROUP`` files per
    (product, source_kind, window) group by mtime, plus anything referenced
    by a release promoted in the last ``_PROTECT_RELEASE_DAYS`` days —
    serving never reads superseded scores files (selections carry their own
    mae/n and archived documents replay without them), so a week of rollback
    candidates is ample and the directory rolls at ~7 days of evaluations
    instead of 30. A file
    the evaluations catalog has never seen is skipped, never deleted —
    pruning must not destroy evidence that was never summarized.
    """
    moment = now or datetime.now(tz=UTC)
    scores_dir = config.dataset.dir / "scores"
    files = sorted(scores_dir.glob("scores_*.parquet")) if scores_dir.exists() else []
    protected = _protected_evaluations(config, now=moment)
    try:
        cataloged = set(
            evidence.load_ledger(
                evidence.ledger_path(config, evidence.EVALUATIONS_LEDGER),
                evidence.EVALUATIONS_SCHEMA,
            )["evaluation_id"].to_list()
        )
    except (*_COLLECTOR_ERRORS, Timeout):
        cataloged = set()
    groups: dict[tuple[str, ...], list[Path]] = {}
    for path in files:
        parts = path.stem.split("_")
        groups.setdefault(tuple(parts[1:-1]), []).append(path)
    keep: set[Path] = set()
    for members in groups.values():
        members.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        keep |= set(members[:_KEEP_NEWEST_PER_GROUP])
    deleted: list[Path] = []
    skipped: list[str] = []
    freed = 0.0
    for path in files:
        evaluation_id = path.stem.split("_")[-1]
        if path in keep or evaluation_id in protected:
            continue
        if evaluation_id not in cataloged:
            skipped.append(f"{path.name}: not in evaluations catalog yet")
            continue
        freed += path.stat().st_size / 1e6
        deleted.append(path)
        if not dry_run:
            path.unlink()
    return PruneResult(
        deleted=tuple(deleted),
        kept=tuple(sorted(set(files) - set(deleted))),
        skipped=tuple(skipped),
        freed_mb=freed,
    )
