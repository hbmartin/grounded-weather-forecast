"""Read-only freshness policy shared by reporting and evidence collectors."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

import polars as pl

from grounded_weather_forecast.run_health import latest_attempt

FRESHNESS_THRESHOLDS: Mapping[str, float] = {
    "truth_age_minutes": 120.0,
    "collector_age_minutes": 180.0,
    "served_history_age_minutes": 180.0,
    "forecast_document_age_minutes": 180.0,
    "truth_qc_age_minutes": 2880.0,
    "evidence_age_minutes": 2880.0,
}


TRUTH_QC_MAX_AGE = timedelta(minutes=FRESHNESS_THRESHOLDS["truth_qc_age_minutes"])


def truth_qc_age_limit() -> str:
    """The duration used by both assessment warnings and alert provenance."""
    return f"{FRESHNESS_THRESHOLDS['truth_qc_age_minutes'] / 60:g} hours"


@dataclass(frozen=True, slots=True)
class RefreshTime:
    earliest: datetime
    latest: datetime
    label: str
    date_only: bool = False


@dataclass(frozen=True, slots=True)
class TruthQCFreshness:
    refreshed: str
    age: str
    warnings: tuple[str, ...]
    failed_attempt: str | None = None
    refresh_unknown: bool = False
    age_unusable: bool = False
    has_verdict: bool = False


def _refresh_time(payload: Mapping[str, object], now: datetime) -> RefreshTime | None:
    raw = payload.get("generated_at")
    if "generated_at" in payload:
        if not isinstance(raw, str):
            return None
        try:
            stamp = datetime.fromisoformat(raw)
        except (ValueError, OverflowError):
            return None
        try:
            if stamp.tzinfo is None or stamp > now:
                return None
            stamp = stamp.astimezone(UTC)
        except (ValueError, OverflowError):
            return None
        return RefreshTime(stamp, stamp, stamp.isoformat())
    verdict = payload.get("drift_verdict")
    raw_date = verdict.get("as_of_date") if isinstance(verdict, Mapping) else None
    if not isinstance(raw_date, str):
        return None
    try:
        day = date.fromisoformat(raw_date)
    except (ValueError, OverflowError):
        return None
    start = datetime.combine(day, time.min, UTC)
    if start > now:
        return None
    try:
        end = start + timedelta(days=1)
    except (OverflowError,):  # noqa: B013
        return None
    return RefreshTime(start, end, f"{day} (date only)", date_only=True)


def _latest_attempt(runs: pl.DataFrame, now: datetime) -> Mapping[str, object] | None:
    return latest_attempt(runs, "truth-qc", now)


def _failed_attempt(
    runs: pl.DataFrame, now: datetime, refresh: RefreshTime | None
) -> str | None:
    attempt = _latest_attempt(runs, now)
    if attempt is None or (attempt["exit_code"] == 0 and not attempt.get("error")):
        return None
    ended = attempt["ended_at"]
    if not isinstance(ended, datetime):  # normalized run ledgers carry UTC instants
        return None
    if refresh is not None and refresh.earliest > ended:
        return None
    outcome = (
        f"exit {attempt['exit_code']}"
        if attempt["exit_code"] is not None
        else "no successful exit recorded"
    )
    error = attempt.get("error")
    detail = attempt.get("error_detail")
    if detail and detail != error:
        error = f"{error}: {detail}" if error else detail
    return f"{ended.isoformat()} ({outcome}" + (f": {error})" if error else ")")


def truth_qc_freshness(
    payload: Mapping[str, object] | None, runs: pl.DataFrame, now: datetime
) -> TruthQCFreshness:
    payload = payload if payload is not None else {}
    refresh = _refresh_time(payload, now)
    failed = _failed_attempt(runs, now, refresh)
    verdict = payload.get("drift_verdict")
    has_verdict = isinstance(verdict, Mapping) and bool(verdict)
    warnings: list[str] = []
    age = "unknown"
    age_unusable = refresh is None
    if has_verdict:
        if refresh is None:
            warnings.append("truth-QC refresh time is unknown or invalid")
        else:
            elapsed = max(now - refresh.latest, timedelta())
            hours = elapsed.total_seconds() / 3600.0
            age = (
                f"at least {hours:.1f}h (date only)"
                if refresh.date_only
                else f"{hours:.1f}h"
            )
            age_unusable = elapsed > timedelta(
                minutes=FRESHNESS_THRESHOLDS["truth_qc_age_minutes"]
            )
            if age_unusable:
                warnings.append(f"truth-QC verdict is over {truth_qc_age_limit()} old")
    if failed is not None:
        availability = (
            "showing the available verdict"
            if has_verdict
            else "no truth-QC verdict is available"
        )
        warnings.append(f"latest truth-QC refresh failed at {failed}; {availability}")
    return TruthQCFreshness(
        refreshed=refresh.label if refresh is not None else "unknown",
        age=age,
        warnings=tuple(warnings),
        failed_attempt=failed,
        refresh_unknown=refresh is None,
        age_unusable=age_unusable,
        has_verdict=has_verdict,
    )


def truth_qc_refresh_time(
    payload: Mapping[str, object], now: datetime
) -> RefreshTime | None:
    return _refresh_time(payload, now)
