"""Version-1 daily read contract. No collectors, recorders, locks, or writers."""

import json
import math
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import polars as pl

from grounded_weather_forecast.config import Config, config_fingerprint
from grounded_weather_forecast.contracts import HOURLY_VARIABLES
from grounded_weather_forecast.health_policy import FRESHNESS_THRESHOLDS
from grounded_weather_forecast.status import PENDING_EXIT_CODES

SCHEMA_VERSION = 1
WINDOW_HOURS = 12
EVIDENCE_MAX_AGE_MINUTES = 2880
PUBLISHED_STATION = Path(
    "/Users/haroldmartin/Library/Application Support/grounded-weather-forecast"
    "/staged/current.json"
)


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
    except (ValueError, TypeError):
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
    """Detect replacements and in-place writes without creating lock files."""

    signatures: dict[Path, tuple[int, int, int, int] | None] = field(
        default_factory=dict
    )

    def read(self, path: Path) -> bytes:
        before = _signature(path)
        if path in self.signatures and self.signatures[path] != before:
            raise SourceError("concurrent_replacement")
        self.signatures[path] = before
        try:
            data = path.read_bytes()
        except (FileNotFoundError,) as exc:  # noqa: B013 - repository convention
            raise SourceError("missing_source") from exc
        if before != _signature(path):
            raise SourceError("concurrent_replacement")
        return data

    def document(self, path: Path, version: int | None = None) -> dict[str, Any]:
        raw = json.loads(self.read(path))
        if not isinstance(raw, dict):
            raise SourceError("invalid_document")
        if version is not None and (
            type(raw.get("schema_version")) is not int
            or raw["schema_version"] != version
        ):
            raise SourceError("unsupported_source_schema")
        return raw

    def frame(self, path: Path, required: set[str]) -> pl.DataFrame:
        frame = pl.read_parquet(BytesIO(self.read(path)))
        if not required.issubset(frame.columns):
            raise SourceError("missing_required_columns")
        return frame

    def changed(self) -> bool:
        return any(_signature(path) != stamp for path, stamp in self.signatures.items())


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


def _open_readonly(path: Path) -> sqlite3.Connection:
    # Ignore config.immutable: a live WAL must never be silently ignored.
    # Refuse to create missing WAL shared memory in a reader process.
    if (
        path.with_name(path.name + "-wal").exists()
        and not path.with_name(path.name + "-shm").exists()
    ):
        raise SourceError("wal_shared_memory_unavailable")
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)


def read_providers(path: Path, now: datetime) -> dict[str, Any]:
    cutoff = now - timedelta(hours=24)
    connection = _open_readonly(path)
    try:
        rows = connection.execute(
            "SELECT pr.provider, pr.status, fr.completed_at "
            "FROM provider_results pr JOIN forecast_runs fr ON fr.id = pr.run_id "
            "WHERE julianday(fr.completed_at) >= julianday(?) "
            "AND julianday(fr.completed_at) <= julianday(?) ORDER BY julianday(fr.completed_at), fr.id, pr.id",
            (cutoff.isoformat(), now.isoformat()),
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
        connection = _open_readonly(path)
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
    except (OSError, ValueError, sqlite3.Error):
        failure = "live_station_unavailable"
        try:
            moment = timestamp(snapshot.document(published)["metadata"]["observed_at"])
        except (OSError, ValueError, TypeError, KeyError):
            raise SourceError(
                "live_station_and_published_snapshot_unavailable"
            ) from None
        source = "published_station_snapshot"
    age = freshness(moment, now, FRESHNESS_THRESHOLDS["truth_age_minutes"])
    return {
        "status": "ok" if age["status"] == "fresh" and failure is None else "partial",
        "source": source,
        "freshness": age,
        "limitations": [failure] if failure else [],
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
    return (
        now - timedelta(hours=24) <= ended <= now
        and timestamp(row["started_at"]) <= ended
    )


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
        }
    newest = max((r["ended_at"] for r in latest.values()), default=None)
    return {
        "status": "ok" if latest else "unknown",
        "latest_by_command": list(latest.values()),
        "freshness": freshness(newest, now, EVIDENCE_MAX_AGE_MINUTES),
        "limitations": [] if latest else ["no_completed_runs_in_24h"],
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
        "freshness": freshness(row["recorded_at"], now, EVIDENCE_MAX_AGE_MINUTES),
        "historical_alarms": row["alarms"],
        "limitations": ["pipeline_alarms_are_historical_not_current_failures"],
    }


def read_qc(snapshot: Snapshot, path: Path, now: datetime) -> dict[str, Any]:
    raw = snapshot.document(path, 3)
    verdict = raw.get("drift_verdict", {})
    age = freshness(raw.get("generated_at"), now, EVIDENCE_MAX_AGE_MINUTES)
    flags = {
        k: raw.get(k) for k in ("drift_alert", "correlation_alert", "shield_alert")
    }
    if any(v is not None and type(v) is not bool for v in flags.values()):
        raise SourceError("invalid_truth_qc_alerts")
    # Legacy date-only verdicts cannot establish the precise refresh instant.
    return {
        "status": "ok"
        if age["status"] == "fresh" and all(type(v) is bool for v in flags.values())
        else "unknown",
        "freshness": age,
        "evaluation_date": date.fromisoformat(verdict["as_of_date"]).isoformat()
        if verdict.get("as_of_date") is not None
        else None,
        "historical_alerts": flags,
        "current_alerts": flags if age["status"] == "fresh" else None,
        "limitations": ["insufficient_or_undated_truth_qc"]
        if age["status"] != "fresh" or any(v is None for v in flags.values())
        else [],
    }


def read_drift(snapshot: Snapshot, path: Path, now: datetime) -> dict[str, Any]:
    raw = snapshot.document(path, 2)
    alarms = raw.get("alarms")
    if not isinstance(alarms, list):
        raise SourceError("missing_drift_alarms")
    age = freshness(raw.get("generated_at"), now, EVIDENCE_MAX_AGE_MINUTES)
    return {
        "status": "historical",
        "freshness": age,
        "historical_alarm_count": len(alarms),
        "current_alarm_count": None,
        "limitations": ["drift_has_no_dataset_attribution"],
    }


def _quality_rows(
    snapshot: Snapshot, config: Config, now: datetime, *, realized: bool
) -> list[dict[str, Any]]:
    name = "served_quality" if realized else "quality"
    needed = {
        "product",
        "variable",
        "truth_semantics",
        "lead_bucket",
        "method_id",
        "n",
        "dataset_fingerprint",
        "config_fingerprint",
    }
    needed |= (
        {"release_id", "as_of_date", "live_mae"}
        if realized
        else {"evaluation_id", "evaluation_created_at", "source_kind", "mae"}
    )
    return _recent_rows(
        snapshot, config.artifacts_dir / "history" / f"{name}.parquet", needed, now
    )


def _matches_score(
    row: Mapping[str, Any],
    cohort: tuple[str, ...],
    release: Mapping[str, Any],
    selection: Mapping[str, Any],
    now: datetime,
    *,
    realized: bool,
) -> bool:
    variable, bucket, method, semantics, release_id = cohort
    if (
        row["product"],
        row["variable"],
        row["lead_bucket"],
        row["method_id"],
        row["truth_semantics"],
    ) != ("hourly", variable, bucket, method, semantics):
        return False
    if (
        row["dataset_fingerprint"] != release["dataset_fingerprint"]
        or row["config_fingerprint"] != release["config_fingerprint"]
    ):
        return False
    if realized:
        return (
            row["release_id"] == release_id
            and row["as_of_date"] is not None
            and date.fromisoformat(str(row["as_of_date"])) <= now.date()
        )
    return (
        row["source_kind"] == "live"
        and row["evaluation_id"] == selection.get("evaluation_id")
        and timestamp(row["evaluation_created_at"]) <= now
    )


def _slice_score(
    rows: list[dict[str, Any]],
    cohort: tuple[str, ...],
    release: Mapping[str, Any],
    selection: Mapping[str, Any],
    now: datetime,
    *,
    realized: bool,
) -> dict[str, Any]:
    matching = [
        row
        for row in rows
        if _matches_score(row, cohort, release, selection, now, realized=realized)
    ]
    if not matching:
        return {"status": "unknown", "limitations": ["no_matching_current_evidence"]}
    row = max(
        matching,
        key=lambda r: (
            str(r.get("as_of_date", r.get("evaluation_created_at"))),
            timestamp(r["recorded_at"]),
        ),
    )
    age = freshness(
        row["recorded_at"] if realized else row["evaluation_created_at"],
        now,
        EVIDENCE_MAX_AGE_MINUTES,
    )
    score = row["live_mae" if realized else "mae"]
    enough = type(row["n"]) is int and row["n"] > 0 and _finite(score)
    status = "available"
    if not enough:
        status = "insufficient"
    elif age["status"] != "fresh":
        status = "stale"
    return {
        "status": status,
        "n": row["n"] if type(row["n"]) is int else None,
        "mae": score if _finite(score) else None,
        "evaluation_id": row.get("evaluation_id"),
        "evaluation_created_at": row.get("evaluation_created_at"),
        "as_of_date": row.get("as_of_date"),
        "recorded_at": row["recorded_at"],
        "freshness": age,
        "limitations": [] if enough else ["no_finite_score_or_samples"],
    }


def _read_slice(
    snapshot: Snapshot,
    config: Config,
    forecast: Mapping[str, Any],
    rows: list[dict[str, Any]],
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
    if (
        release.get("release_id") != release_id
        or release.get("dataset_fingerprint") != forecast["dataset_fingerprint"]
        or release.get("config_fingerprint") != config_fingerprint(config)
    ):
        raise SourceError("release_current_identity_mismatch")
    selection = release["selections"].get(f"hourly.{variable}.{bucket}", {})
    if (
        selection.get("method_id") != method
        or selection.get("truth_semantics") != semantics
    ):
        raise SourceError("release_selection_mismatch")
    return _slice_score(rows, cohort, release, selection, now, realized=realized)


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
    cohorts = set()
    for point in forecast.get("hourly", []):
        for variable, method in point["methods"].items():
            cohorts.add(
                (
                    variable,
                    point["lead_bucket"],
                    method,
                    point["truth_semantics"].get(variable),
                    point["release_ids"].get(variable),
                )
            )
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
    published_station: Path = PUBLISHED_STATION,
) -> dict[str, Any]:
    """Read existing artifacts, retrying once if any tracked file changes."""
    start, now = timestamp(start), timestamp(as_of)
    if hours != WINDOW_HOURS or start.minute or start.second or start.microsecond:
        raise ValueError("start must be on an absolute hour and hours must equal 12")
    if now < start:
        raise ValueError("as-of must be at or after the scheduled start")
    zone = ZoneInfo(config.station.timezone)
    for _attempt in range(2):
        snapshot = Snapshot()
        forecast = component(read_forecast, snapshot, forecast_path, start, now, zone)
        operational = {
            "station": component(
                read_station, snapshot, config.station.db_path, published_station, now
            ),
            "providers": component(read_providers, config.forecasts.db_path, now),
            "runs": component(
                read_runs, snapshot, config.dataset.dir / "runs.parquet", now
            ),
            "pipeline": component(
                read_pipeline,
                snapshot,
                config.artifacts_dir / "history/pipeline.parquet",
                now,
            ),
        }
        prediction = {
            "backtest": component(
                read_quality, snapshot, config, forecast, now, realized=False
            ),
            "realized_live": component(
                read_quality, snapshot, config, forecast, now, realized=True
            ),
            "truth_qc": component(
                read_qc, snapshot, config.artifacts_dir / "truth_qc.json", now
            ),
            "drift": component(
                read_drift, snapshot, config.artifacts_dir / "drift.json", now
            ),
        }
        if not snapshot.changed():
            return _report_envelope(
                start,
                now,
                zone,
                forecast,
                operational,
                prediction,
                ["sources_are_independent_snapshots_not_a_global_transaction"],
            )
    # Keep the success envelope stable, but discard any potentially mixed data.
    return _report_envelope(
        start,
        now,
        zone,
        _unstable_component(),
        {key: _unstable_component() for key in operational},
        {key: _unstable_component() for key in prediction},
        ["concurrent_replacement_after_retry"],
    )


def json_default(value: object) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(type(value).__name__)
