"""Version-1 daily read contract. No collectors, recorders, locks, or writers."""

import json
import math
import sqlite3
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import polars as pl

from grounded_weather_forecast.config import Config, config_fingerprint
from grounded_weather_forecast.contracts import HOURLY_VARIABLES, LOCATION_TOLERANCE
from grounded_weather_forecast.health_policy import (
    FRESHNESS_THRESHOLDS,
    truth_qc_freshness,
    truth_qc_refresh_time,
)
from grounded_weather_forecast.serve.provenance import (
    SelectionProvenance,
    selection_provenance,
)
from grounded_weather_forecast.status import PENDING_EXIT_CODES
from grounded_weather_forecast.support_paths import (
    published_station_path as published_station_path,  # noqa: PLC0414 -- preserve compatibility export
)


def evidence_age_limit() -> int:
    return int(FRESHNESS_THRESHOLDS["evidence_age_minutes"])


SCHEMA_VERSION = 1
WINDOW_HOURS = 12


class SourceError(ValueError):
    """A safe, machine-readable source limitation; never includes raw data."""


def timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        result = datetime.fromisoformat(value)
    else:
        raise SourceError("missing_timestamp")
    if result.tzinfo is None or result.utcoffset() is None:
        raise SourceError("timezone_required")
    return result.astimezone(UTC)


def freshness(value: object, now: datetime, limit: float) -> dict[str, Any]:
    try:
        moment = timestamp(value)
    except (ValueError, TypeError, OverflowError):
        return {
            "status": "unknown",
            "at": None,
            "age_minutes": None,
            "max_age_minutes": limit,
        }
    age = (now - moment).total_seconds() / 60
    status = "fresh"
    if age < 0:
        status = "unknown"
    elif age > limit:
        status = "stale"
    return {
        "status": status,
        "at": moment.isoformat(),
        "age_minutes": age,
        "max_age_minutes": limit,
    }


def _signature(path: Path) -> tuple[int, int, int, int] | None:
    try:
        info = path.stat()
    except (FileNotFoundError,):  # noqa: B013 - repository exception convention
        return None
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


@dataclass
class Snapshot:
    """Cached independent reads, with per-component replacement dependencies."""

    signatures: dict[Path, tuple[int, int, int, int] | None] = field(
        default_factory=dict
    )
    dependencies: dict[str, set[Path]] = field(default_factory=dict)
    _bytes: dict[Path, bytes] = field(default_factory=dict)
    _documents: dict[Path, dict[str, Any]] = field(default_factory=dict)
    _frames: dict[Path, pl.DataFrame] = field(default_factory=dict)
    _component: str | None = None

    @contextmanager
    def capture(self, name: str) -> Iterator[None]:
        self.dependencies[name] = set()
        self._component = name
        try:
            yield
        finally:
            self._component = None

    def track(self, path: Path) -> tuple[int, int, int, int] | None:
        if self._component is not None:
            self.dependencies[self._component].add(path)
        before = _signature(path)
        if path in self.signatures and self.signatures[path] != before:
            raise SourceError("concurrent_replacement")
        self.signatures[path] = before
        return before

    def read(self, path: Path) -> bytes:
        before = self.track(path)
        if path in self._bytes:
            return self._bytes[path]
        try:
            data = path.read_bytes()
        except (FileNotFoundError,) as exc:  # noqa: B013
            raise SourceError("missing_source") from exc
        if before != _signature(path):
            raise SourceError("concurrent_replacement")
        self._bytes[path] = data
        return data

    def document(self, path: Path, version: int | None = None) -> dict[str, Any]:
        self.track(path)
        if path not in self._documents:
            raw = json.loads(self.read(path))
            if not isinstance(raw, dict):
                raise SourceError("invalid_document")
            self._documents[path] = raw
        raw = self._documents[path]
        if version is not None and (
            type(raw.get("schema_version")) is not int
            or raw["schema_version"] != version
        ):
            raise SourceError("unsupported_source_schema")
        return raw

    def frame(self, path: Path, required: set[str]) -> pl.DataFrame:
        self.track(path)
        if path not in self._frames:
            self._frames[path] = pl.read_parquet(BytesIO(self.read(path)))
        frame = self._frames[path]
        if not required.issubset(frame.columns):
            raise SourceError("missing_required_columns")
        return frame

    def changed_paths(self) -> set[Path]:
        changed = set()
        for path, stamp in self.signatures.items():
            try:
                if _signature(path) != stamp:
                    changed.add(path)
            except (OSError,):  # noqa: B013
                changed.add(path)
        return changed

    def changed(self) -> bool:
        return bool(self.changed_paths())

    def invalidate(self, paths: set[Path]) -> None:
        for path in paths:
            self.signatures.pop(path, None)
            self._bytes.pop(path, None)
            self._documents.pop(path, None)
            self._frames.pop(path, None)


def component(
    reader: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any
) -> dict[str, Any]:
    try:
        return reader(*args, **kwargs)
    except (SourceError,) as exc:  # noqa: B013 - repository exception convention
        return {"status": "unknown", "limitations": [str(exc)]}
    except (
        OSError,
        ValueError,
        OverflowError,
        TypeError,
        KeyError,
        AttributeError,
        sqlite3.Error,
        pl.exceptions.PolarsError,
    ):
        return {"status": "unknown", "limitations": ["unreadable_or_invalid_source"]}


def _finite(value: object) -> bool:
    try:
        return (
            isinstance(value, (float, int))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    except (OverflowError,):  # noqa: B013 - repository exception convention
        return False


def _hour_index(hours: object, start: datetime) -> dict[datetime, list[dict[str, Any]]]:
    if not isinstance(hours, list):
        raise SourceError("missing_hourly_array")
    end = start + timedelta(hours=WINDOW_HOURS)
    index: dict[datetime, list[dict[str, Any]]] = {}
    for raw_point in hours:
        if not isinstance(raw_point, dict):
            raise SourceError("invalid_hourly_point")
        point = cast("dict[str, Any]", raw_point)
        valid = timestamp(point["valid_time"])
        if start <= valid < end:
            if point.get("lead_bucket") is not None and not isinstance(
                point["lead_bucket"], str
            ):
                raise SourceError("invalid_lead_bucket")
            if valid.minute or valid.second or valid.microsecond:
                raise SourceError("unaligned_hourly_timestamp")
            index.setdefault(valid, []).append(point)
    return index


def _hour(
    point: Mapping[str, Any] | None, valid: datetime, zone: ZoneInfo
) -> dict[str, Any]:
    values = point.get("values", {}) if point else {}
    if not isinstance(values, dict):
        raise SourceError("invalid_hourly_values")
    absent = [s.name for s in HOURLY_VARIABLES if s.name not in values]
    null = [
        s.name for s in HOURLY_VARIABLES if s.name in values and values[s.name] is None
    ]
    invalid = [
        s.name
        for s in HOURLY_VARIABLES
        if s.name in values
        and values[s.name] is not None
        and (
            not _finite(values[s.name])
            or (s.minimum is not None and values[s.name] < s.minimum)
            or (s.maximum is not None and values[s.name] > s.maximum)
        )
    ]
    maps = {}
    for key in ("methods", "release_ids", "truth_semantics", "selection_reasons"):
        value = point.get(key, {}) if point else {}
        if not isinstance(value, dict) or any(
            not isinstance(v, str) for v in value.values()
        ):
            raise SourceError("invalid_hourly_attribution")
        maps[key] = {s.name: value.get(s.name) for s in HOURLY_VARIABLES}
    return {
        "valid_time": valid.isoformat(),
        "local_time": valid.astimezone(zone).isoformat(),
        "lead_bucket": point.get("lead_bucket") if point else None,
        "values": {
            s.name: values.get(s.name) if s.name not in invalid else None
            for s in HOURLY_VARIABLES
        },
        "missing_variables": absent,
        "null_variables": null,
        "invalid_variables": invalid,
        **maps,
    }


def _forecast_metadata(
    raw: Mapping[str, Any], now: datetime
) -> tuple[datetime, list[str]]:
    issued = timestamp(raw["issued_at"])
    if issued > now:
        raise SourceError("forecast_issued_after_as_of")
    if not isinstance(raw.get("dataset_fingerprint"), str) or not isinstance(
        raw.get("status"), str
    ):
        raise SourceError("invalid_forecast_metadata")
    ids = raw.get("release_ids")
    if not isinstance(ids, list) or any(not isinstance(v, str) for v in ids):
        raise SourceError("invalid_release_ids")
    return issued, ids


def read_forecast(
    snapshot: Snapshot, path: Path, start: datetime, now: datetime, zone: ZoneInfo
) -> dict[str, Any]:
    raw = snapshot.document(path, 6)
    issued, ids = _forecast_metadata(raw, now)
    index = _hour_index(raw.get("hourly"), start)
    if any(
        raw.get(key) is not None and not isinstance(raw[key], str)
        for key in ("status_reason", "observation_at")
    ):
        raise SourceError("invalid_forecast_metadata")
    expected = [start + timedelta(hours=i) for i in range(WINDOW_HOURS)]
    missing = [t.isoformat() for t in expected if t not in index]
    duplicate = [t.isoformat() for t in expected if len(index.get(t, [])) > 1]
    rows = [
        _hour(index[t][0] if len(index.get(t, [])) == 1 else None, t, zone)
        for t in expected
    ]
    issues = []
    if missing:
        issues.append("missing_hours")
    if duplicate:
        issues.append("duplicate_hours")
    if any(
        r["missing_variables"] or r["null_variables"] or r["invalid_variables"]
        for r in rows
    ):
        issues.append("missing_null_or_invalid_values")
    age = freshness(issued, now, FRESHNESS_THRESHOLDS["forecast_document_age_minutes"])
    if age["status"] != "fresh":
        issues.append("forecast_stale")
    if raw.get("status") != "ready":
        issues.append("forecast_not_ready")
    return {
        "status": "partial" if issues else "ok",
        "limitations": issues,
        "source_schema_version": 6,
        "issued_at": issued.isoformat(),
        "observation_at": raw.get("observation_at"),
        "forecast_status": raw.get("status"),
        "status_reason": raw.get("status_reason"),
        "dataset_fingerprint": raw.get("dataset_fingerprint"),
        "release_ids": ids,
        "freshness": age,
        "observation_freshness": freshness(
            raw.get("observation_at"), now, FRESHNESS_THRESHOLDS["truth_age_minutes"]
        ),
        "units": {s.name: s.unit or "1 (probability)" for s in HOURLY_VARIABLES},
        "coverage": {
            "expected_hours": WINDOW_HOURS,
            "unique_hours": sum(len(index.get(t, [])) == 1 for t in expected),
            "missing_hours": missing,
            "duplicate_hours": duplicate,
        },
        "hourly": rows,
    }


def _open_readonly(path: Path, snapshot: Snapshot | None = None) -> sqlite3.Connection:
    """SQLite only sees an in-memory rollback snapshot, never the live filename."""
    snapshot = snapshot or Snapshot()
    for suffix in ("-wal", "-shm", "-journal"):
        associated = path.with_name(path.name + suffix)
        if snapshot.track(associated) is not None:
            raise SourceError("sqlite_journal_dependent_source")
        try:
            associated.lstat()
        except (FileNotFoundError,):  # noqa: B013
            pass
        else:
            raise SourceError("sqlite_journal_dependent_source")
    data = snapshot.read(path)
    if len(data) < 100 or data[:16] != b"SQLite format 3\x00":
        raise SourceError("invalid_sqlite_header")
    if data[18:20] != b"\x01\x01":
        raise SourceError("sqlite_wal_source_unsafe")
    if not hasattr(sqlite3.Connection, "deserialize"):
        raise SourceError("sqlite_memory_snapshot_unavailable")
    connection = sqlite3.connect(":memory:", timeout=1)
    try:
        connection.deserialize(data)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA temp_store=MEMORY")
    except (sqlite3.Error,):  # noqa: B013
        connection.close()
        raise
    return connection


def read_providers(
    path: Path,
    now: datetime,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    snapshot: Snapshot | None = None,
) -> dict[str, Any]:
    cutoff = now - timedelta(hours=24)
    connection = _open_readonly(path, snapshot)
    try:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(forecast_runs)")
        }
        if (
            not {"latitude", "longitude"} <= columns
            or latitude is None
            or longitude is None
        ):
            raise SourceError("provider_location_unavailable")
        rows = connection.execute(
            "SELECT pr.provider, pr.status, fr.completed_at "
            "FROM provider_results pr JOIN forecast_runs fr ON fr.id = pr.run_id "
            "WHERE julianday(fr.completed_at) >= julianday(?) "
            "AND julianday(fr.completed_at) <= julianday(?) "
            "AND ABS(fr.latitude - ?) <= ? AND ABS(fr.longitude - ?) <= ? "
            "ORDER BY julianday(fr.completed_at), fr.id, pr.id",
            (
                cutoff.isoformat(),
                now.isoformat(),
                latitude,
                LOCATION_TOLERANCE,
                longitude,
                LOCATION_TOLERANCE,
            ),
        ).fetchall()
    finally:
        connection.close()
    providers: dict[str, dict[str, Any]] = {}
    for name, status, completed in rows:
        if status not in {"success", "error"}:
            raise SourceError("unsupported_provider_status")
        row = providers.setdefault(
            name, {"provider": name, "attempts": 0, "successes": 0, "failures": 0}
        )
        row["attempts"] += 1
        row["successes"] += status == "success"
        row["failures"] += status == "error"
        row.update(latest_status=status, completed_at=timestamp(completed).isoformat())
    for row in providers.values():
        row["success_rate"] = row["successes"] / row["attempts"]
        row["freshness"] = freshness(
            row["completed_at"], now, FRESHNESS_THRESHOLDS["collector_age_minutes"]
        )
    return {
        "status": "ok" if rows else "unknown",
        "window_start": cutoff.isoformat(),
        "window_end": now.isoformat(),
        "providers": sorted(providers.values(), key=lambda r: r["provider"]),
        "limitations": [] if rows else ["no_provider_attempts_in_24h"],
    }


def read_station(
    snapshot: Snapshot, path: Path, published: Path, now: datetime
) -> dict[str, Any]:
    failure = None
    try:
        connection = _open_readonly(path, snapshot)
        try:
            value = connection.execute(
                "SELECT MAX(ts) FROM observations WHERE julianday(ts) <= julianday(?)",
                (now.isoformat(),),
            ).fetchone()[0]
        finally:
            connection.close()
        if value is None:
            raise SourceError("no_station_observations_as_of")
        moment = datetime.fromisoformat(value)
        # Station ts is documented as UTC-naive.
        moment = moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment
        source = "live_station_sqlite"
    except (OSError, ValueError, sqlite3.Error) as live_error:
        failure = "live_station_unavailable"
        try:
            moment = timestamp(snapshot.document(published)["metadata"]["observed_at"])
        except (OSError, ValueError, TypeError, KeyError):
            return {
                "status": "unknown",
                "limitations": ["live_station_and_published_snapshot_unavailable"],
                "published_station_path": str(published),
                "live_station_limitation": str(live_error)
                if isinstance(live_error, SourceError)
                else "live_station_unavailable",
            }
        source = "published_station_snapshot"
    age = freshness(moment, now, FRESHNESS_THRESHOLDS["truth_age_minutes"])
    return {
        "status": "ok" if age["status"] == "fresh" and failure is None else "partial",
        "source": source,
        "freshness": age,
        "limitations": [failure] if failure else [],
        "published_station_path": str(published),
    }


def _recent_rows(
    snapshot: Snapshot, path: Path, required: set[str], now: datetime
) -> list[dict[str, Any]]:
    frame = snapshot.frame(path, required | {"recorded_at"})
    return [r for r in frame.to_dicts() if timestamp(r["recorded_at"]) <= now]


def _recent_run(row: Mapping[str, Any], now: datetime) -> bool:
    if row["exit_code"] is not None and type(row["exit_code"]) is not int:
        raise SourceError("invalid_run_exit_code")
    if row.get("failure_kind") is not None and not isinstance(row["failure_kind"], str):
        raise SourceError("invalid_run_failure_kind")
    if (
        row.get("record_kind") not in (None, "invocation", "stage")
        or row["ended_at"] is None
    ):
        return False
    ended = timestamp(row["ended_at"])
    return ended <= now and timestamp(row["started_at"]) <= ended


def _run_status(row: Mapping[str, Any]) -> str:
    if (
        row.get("failure_kind") == "no_folds"
        and row["exit_code"] in PENDING_EXIT_CODES
        and not row.get("error")
    ):
        return "pending"
    if row["exit_code"] is None and not row.get("error"):
        return "unknown"
    if row["exit_code"] == 0 and not row.get("error"):
        return "ok"
    return "failed"


def read_runs(snapshot: Snapshot, path: Path, now: datetime) -> dict[str, Any]:
    frame = snapshot.frame(path, {"command", "started_at", "ended_at", "exit_code"})
    latest: dict[str, dict[str, Any]] = {}
    for row in frame.to_dicts():
        if not _recent_run(row, now):
            continue
        key = row["command"]
        if key not in {
            "predict",
            "publish",
            "maintain",
            "backtest",
            "report",
            "truth-qc",
            "build-dataset",
        }:
            continue
        if key in latest and timestamp(latest[key]["ended_at"]) >= timestamp(
            row["ended_at"]
        ):
            continue
        latest[key] = {
            "command": key,
            "ended_at": timestamp(row["ended_at"]).isoformat(),
            "status": _run_status(row),
            "exit_code": row["exit_code"],
            "failure_kind": row.get("failure_kind"),
            "freshness": freshness(row["ended_at"], now, evidence_age_limit()),
        }
    newest = max((r["ended_at"] for r in latest.values()), default=None)
    return {
        "status": "ok" if latest else "unknown",
        "latest_by_command": list(latest.values()),
        "freshness": freshness(newest, now, evidence_age_limit()),
        "limitations": [] if latest else ["no_completed_runs_as_of"],
    }


def read_pipeline(snapshot: Snapshot, path: Path, now: datetime) -> dict[str, Any]:
    rows = _recent_rows(snapshot, path, {"alarms"}, now)
    if not rows:
        raise SourceError("no_pipeline_evidence_as_of")
    row = max(rows, key=lambda r: timestamp(r["recorded_at"]))
    if row["alarms"] is not None and not isinstance(row["alarms"], str):
        raise SourceError("invalid_pipeline_alarms")
    return {
        "status": "historical",
        "freshness": freshness(row["recorded_at"], now, evidence_age_limit()),
        "historical_alarms": row["alarms"],
        "limitations": ["pipeline_alarms_are_historical_not_current_failures"],
    }


def read_qc(
    snapshot: Snapshot, path: Path, now: datetime, *, runs_path: Path | None = None
) -> dict[str, Any]:
    raw = snapshot.document(path, 3)
    flags = {
        k: raw.get(k) for k in ("drift_alert", "correlation_alert", "shield_alert")
    }
    if any(v is not None and type(v) is not bool for v in flags.values()):
        raise SourceError("invalid_truth_qc_alerts")
    runs = pl.DataFrame()
    if runs_path is not None:
        with suppress(OSError, ValueError, pl.exceptions.PolarsError):
            runs = snapshot.frame(
                runs_path, {"command", "started_at", "ended_at", "exit_code"}
            )
    assessment = truth_qc_freshness(raw, runs, now)
    refresh = truth_qc_refresh_time(raw, now)
    limit = int(FRESHNESS_THRESHOLDS["truth_qc_age_minutes"])
    age = freshness(raw.get("generated_at"), now, limit)
    if refresh is not None and refresh.date_only:
        minutes = max((now - refresh.latest).total_seconds() / 60, 0)
        age = {
            "status": "stale" if minutes > limit else "fresh",
            "at": None,
            "age_minutes": minutes,
            "max_age_minutes": limit,
        }
    limitations = []
    verdict = raw.get("drift_verdict", {})
    evaluation_date = None
    try:
        if not isinstance(verdict, Mapping):
            raise ValueError("invalid verdict")
        if verdict.get("as_of_date") is not None:
            evaluation_date = date.fromisoformat(str(verdict["as_of_date"])).isoformat()
    except (ValueError, TypeError):
        limitations.append("invalid_truth_qc_verdict")
    usable = age["status"] == "fresh" and any(type(v) is bool for v in flags.values())
    if not usable:
        limitations.append("insufficient_or_undated_truth_qc")
    if assessment.failed_attempt is not None:
        limitations.append("latest_truth_qc_refresh_failed")
    current = usable and assessment.failed_attempt is None
    return {
        "status": "ok"
        if current and not limitations
        else ("partial" if usable else "unknown"),
        "freshness": age,
        "evaluation_date": evaluation_date,
        "historical_alerts": flags,
        "current_alerts": flags if current else None,
        "limitations": limitations,
    }


def read_drift(snapshot: Snapshot, path: Path, now: datetime) -> dict[str, Any]:
    raw = snapshot.document(path, 2)
    alarms = raw.get("alarms")
    if not isinstance(alarms, list):
        raise SourceError("missing_drift_alarms")
    age = freshness(raw.get("generated_at"), now, evidence_age_limit())
    return {
        "status": "historical",
        "freshness": age,
        "historical_alarm_count": len(alarms),
        "current_alarm_count": None,
        "limitations": ["drift_has_no_dataset_attribution"],
    }


def _utc_column(frame: pl.DataFrame, name: str) -> pl.Expr:
    column = pl.col(name)
    if frame.schema[name] == pl.String:
        return column.str.to_datetime(format="%+", time_zone="UTC", strict=False)
    return column.cast(pl.Datetime("us", "UTC"), strict=False)


def _quality_rows(
    snapshot: Snapshot, config: Config, now: datetime, *, realized: bool
) -> pl.DataFrame:
    name = "served_quality" if realized else "quality"
    identity = [
        "product",
        "variable",
        "truth_semantics",
        "lead_bucket",
        "method_id",
        "dataset_fingerprint",
        "config_fingerprint",
        "code_version",
    ]
    extras = (
        ["release_id", "as_of_date", "live_mae"]
        if realized
        else ["evaluation_id", "evaluation_created_at", "source_kind", "mae"]
    )
    frame = snapshot.frame(
        config.artifacts_dir / "history" / f"{name}.parquet",
        set(identity + extras + ["n", "recorded_at"]),
    )
    frame = frame.with_columns(_utc_column(frame, "recorded_at").alias("recorded_at"))
    frame = frame.filter(
        (pl.col("recorded_at") <= now) & (pl.col("product") == "hourly")
    )
    if realized:
        frame = frame.with_columns(pl.col("as_of_date").cast(pl.Date, strict=False))
        frame = frame.filter(pl.col("as_of_date") <= now.date())
        clocks = ["as_of_date", "recorded_at"]
        keys = [*identity, "release_id"]
    else:
        frame = frame.with_columns(
            _utc_column(frame, "evaluation_created_at").alias("evaluation_created_at")
        )
        frame = frame.filter(
            (pl.col("evaluation_created_at") <= now) & (pl.col("source_kind") == "live")
        )
        clocks = ["evaluation_created_at", "recorded_at"]
        keys = [*identity, "evaluation_id"]
    # Reduce in Polars before any per-slice decoding or Python conversion.
    return (
        frame.sort(clocks)
        .unique(subset=keys, keep="last")
        .sort(clocks, descending=True)
    )


def _cohort_rows(
    frame: pl.DataFrame, cohort: tuple[str, ...], provenance: SelectionProvenance
) -> pl.DataFrame:
    variable, bucket, method, semantics, _ = cohort
    if provenance.code_version is None or provenance.config_fingerprint is None:
        return frame.clear()
    return frame.filter(
        (pl.col("variable") == variable)
        & (pl.col("lead_bucket") == bucket)
        & (pl.col("method_id") == method)
        & (pl.col("truth_semantics") == semantics)
        & (pl.col("config_fingerprint") == provenance.config_fingerprint)
        & (pl.col("code_version") == provenance.code_version)
    )


def _compatible_evidence(
    snapshot: Snapshot,
    config: Config,
    rows: pl.DataFrame,
    key: str,
    current: SelectionProvenance,
    now: datetime,
) -> dict[str, Any] | None:
    for row in rows.iter_rows(named=True):
        identifier = row["release_id"]
        if not isinstance(identifier, str) or not identifier.isalnum():
            continue
        try:
            release = snapshot.document(
                config.artifacts_dir / "releases" / f"{identifier}.json"
            )
            if (
                release.get("release_id") != identifier
                or timestamp(release.get("promoted_at")) > now
            ):
                continue
            candidate = selection_provenance(release, key)
            if current.compatible_with(candidate):
                return row
        except (OSError, ValueError, TypeError, OverflowError):
            continue
    return None


def _quality_value(
    row: Mapping[str, Any] | None,
    now: datetime,
    *,
    realized: bool,
    fallback: bool = False,
) -> dict[str, Any]:
    attribution = {
        "evidence_match": None,
        "evidence_release_id": None,
        "evidence_dataset_fingerprint": None,
    }
    if row is None:
        return {
            "status": "unknown",
            "limitations": ["no_matching_current_evidence"],
            **attribution,
        }
    age = freshness(
        row["recorded_at"] if realized else row["evaluation_created_at"],
        now,
        evidence_age_limit(),
    )
    score = row["live_mae" if realized else "mae"]
    enough = type(row["n"]) is int and row["n"] > 0 and _finite(score)
    status = "available" if enough else "insufficient"
    if enough and age["status"] != "fresh":
        status = "stale"
    limitations = [] if enough else ["no_finite_score_or_samples"]
    if fallback:
        limitations.append("older_compatible_release_evidence")
    if realized:
        attribution.update(
            evidence_match="compatible_release" if fallback else "exact_release",
            evidence_release_id=row["release_id"],
            evidence_dataset_fingerprint=row["dataset_fingerprint"],
        )
    return {
        "status": status,
        "n": row["n"] if type(row["n"]) is int else None,
        "mae": score if _finite(score) else None,
        "evaluation_id": row.get("evaluation_id"),
        "evaluation_created_at": row.get("evaluation_created_at"),
        "as_of_date": row.get("as_of_date"),
        "recorded_at": row["recorded_at"],
        "freshness": age,
        "limitations": limitations,
        **attribution,
    }


def _read_slice(
    snapshot: Snapshot,
    config: Config,
    forecast: Mapping[str, Any],
    rows: pl.DataFrame,
    cohort: tuple[str, ...],
    now: datetime,
    *,
    realized: bool,
) -> dict[str, Any]:
    variable, bucket, method, semantics, release_id = cohort
    if (
        not isinstance(release_id, str)
        or release_id not in forecast["release_ids"]
        or not release_id.isalnum()
    ):
        raise SourceError("slice_has_no_valid_forecast_release")
    release = snapshot.document(
        config.artifacts_dir / "releases" / f"{release_id}.json"
    )
    if release.get("release_id") != release_id or release.get(
        "config_fingerprint"
    ) != config_fingerprint(config):
        raise SourceError("release_current_identity_mismatch")
    key = f"hourly.{variable}.{bucket}"
    provenance = selection_provenance(release, key)
    if (
        not provenance.valid_context
        or provenance.method_id != method
        or provenance.truth_semantics != semantics
    ):
        raise SourceError("release_selection_mismatch")
    matching = _cohort_rows(rows, cohort, provenance)
    fallback = False
    if realized:
        exact = matching.filter(pl.col("release_id") == release_id)
        row = exact.row(0, named=True) if not exact.is_empty() else None
        if row is None:
            row = _compatible_evidence(snapshot, config, matching, key, provenance, now)
            fallback = row is not None
    else:
        if provenance.evaluation_id is None or provenance.dataset_fingerprint is None:
            return _quality_value(None, now, realized=False)
        matching = matching.filter(
            (pl.col("evaluation_id") == provenance.evaluation_id)
            & (pl.col("dataset_fingerprint") == provenance.dataset_fingerprint)
        )
        row = matching.row(0, named=True) if not matching.is_empty() else None
    return _quality_value(row, now, realized=realized, fallback=fallback)


def read_quality(
    snapshot: Snapshot,
    config: Config,
    forecast: Mapping[str, Any],
    now: datetime,
    *,
    realized: bool,
) -> dict[str, Any]:
    manifest = snapshot.document(config.dataset.dir / "manifest.json")
    dataset = manifest.get("fingerprint")
    if not dataset or dataset != forecast.get("dataset_fingerprint"):
        raise SourceError("forecast_current_dataset_mismatch")
    if not forecast.get("release_ids"):
        raise SourceError("forecast_has_no_release_ids")
    rows = _quality_rows(snapshot, config, now, realized=realized)
    cohorts = {
        (
            variable,
            point["lead_bucket"],
            method,
            point["truth_semantics"].get(variable),
            point["release_ids"].get(variable),
        )
        for point in forecast.get("hourly", [])
        for variable, method in point["methods"].items()
        if isinstance(method, str)
        and method
        and isinstance(point["release_ids"].get(variable), str)
        and point["release_ids"].get(variable)
    }
    result = []
    for cohort in sorted(cohorts, key=str):
        variable, bucket, method, semantics, release_id = cohort
        base = {
            "variable": variable,
            "unit": forecast["units"][variable],
            "lead_bucket": bucket,
            "method_id": method,
            "truth_semantics": semantics,
            "release_id": release_id,
        }
        result.append(
            base
            | component(
                _read_slice,
                snapshot,
                config,
                forecast,
                rows,
                cohort,
                now,
                realized=realized,
            )
        )
    return {
        "status": "available"
        if result and all(r["status"] == "available" for r in result)
        else "partial",
        "kind": "realized_live" if realized else "backtest",
        "product": "hourly",
        "dataset_fingerprint": dataset,
        "slices": result,
        "limitations": ["scores_describe_evidence_not_a_prediction_quality_verdict"],
    }


def _report_envelope(
    start: datetime,
    now: datetime,
    zone: ZoneInfo,
    forecast: dict[str, Any],
    operational: dict[str, Any],
    prediction: dict[str, Any],
    limitations: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "as_of": now.isoformat(),
        "window": {
            "start": start.isoformat(),
            "end": (start + timedelta(hours=WINDOW_HOURS)).isoformat(),
            "hours": WINDOW_HOURS,
            "timezone": str(zone),
            "local_start": start.astimezone(zone).isoformat(),
            "local_end": (start + timedelta(hours=WINDOW_HOURS))
            .astimezone(zone)
            .isoformat(),
            "end_exclusive": True,
        },
        "forecast": forecast,
        "operational_health": operational,
        "prediction_quality": prediction,
        "limitations": limitations,
    }


def _unstable_component() -> dict[str, Any]:
    return {"status": "unknown", "limitations": ["concurrent_replacement_after_retry"]}


def read_daily_report(
    config: Config,
    *,
    forecast_path: Path,
    start: datetime,
    as_of: datetime,
    hours: int = WINDOW_HOURS,
    published_station: Path | None = None,
) -> dict[str, Any]:
    """Retry changed dependency groups once while preserving independent reads."""
    start, now = timestamp(start), timestamp(as_of)
    if hours != WINDOW_HOURS or start.minute or start.second or start.microsecond:
        raise ValueError("start must be on an absolute hour and hours must equal 12")
    if now < start:
        raise ValueError("as-of must be at or after the scheduled start")
    zone = ZoneInfo(config.station.timezone)
    # Check all envelope arithmetic before reading components.
    (start + timedelta(hours=WINDOW_HOURS)).astimezone(zone)
    start.astimezone(zone)
    published_station = (
        published_station if published_station is not None else published_station_path()
    )
    snapshot = Snapshot()
    result: dict[str, dict[str, Any]] = {}
    readers: dict[str, Callable[[], dict[str, Any]]] = {
        "forecast": lambda: read_forecast(snapshot, forecast_path, start, now, zone),
        "station": lambda: read_station(
            snapshot, config.station.db_path, published_station, now
        ),
        "providers": lambda: read_providers(
            config.forecasts.db_path,
            now,
            latitude=config.forecasts.latitude,
            longitude=config.forecasts.longitude,
            snapshot=snapshot,
        ),
        "runs": lambda: read_runs(snapshot, config.dataset.dir / "runs.parquet", now),
        "pipeline": lambda: read_pipeline(
            snapshot, config.artifacts_dir / "history/pipeline.parquet", now
        ),
        "backtest": lambda: read_quality(
            snapshot, config, result["forecast"], now, realized=False
        ),
        "realized_live": lambda: read_quality(
            snapshot, config, result["forecast"], now, realized=True
        ),
        "truth_qc": lambda: read_qc(
            snapshot,
            config.artifacts_dir / "truth_qc.json",
            now,
            runs_path=config.dataset.dir / "runs.parquet",
        ),
        "drift": lambda: read_drift(snapshot, config.artifacts_dir / "drift.json", now),
    }
    pending = set(readers)
    unstable = False
    for attempt in range(2):
        for name, read in readers.items():
            if name in pending:
                with snapshot.capture(name):
                    result[name] = component(read)
        changed = snapshot.changed_paths()
        affected = {
            name for name, paths in snapshot.dependencies.items() if paths & changed
        }
        if "forecast" in affected:
            affected.update(("backtest", "realized_live"))
        if not affected:
            break
        if attempt == 0:
            snapshot.invalidate(changed)
            pending = affected
        else:
            unstable = True
            for name in affected:
                result[name] = _unstable_component()
    return _report_envelope(
        start,
        now,
        zone,
        result["forecast"],
        {key: result[key] for key in ("station", "providers", "runs", "pipeline")},
        {
            key: result[key]
            for key in ("backtest", "realized_live", "truth_qc", "drift")
        },
        ["concurrent_replacement_after_retry"]
        if unstable
        else ["sources_are_independent_snapshots_not_a_global_transaction"],
    )


def json_default(value: object) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(type(value).__name__)
