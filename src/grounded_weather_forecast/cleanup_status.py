"""Recorded cleanup evidence has its own validity and recovery boundary."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from grounded_weather_forecast.diagnostic_values import declared_types, field_valid
from grounded_weather_forecast.run_health import aware_moment
from grounded_weather_forecast.status import (
    ComponentDiagnostic,
    assess_progress_identity,
    is_verification_verdict,
)

type RecordedCleanupOutcome = Literal["completed", "incomplete", "unknown"]
_OUTCOMES: dict[str, RecordedCleanupOutcome] = {
    "audit_cleanup_complete": "completed",
    "audit_cleanup_issues": "completed",
    "audit_cleanup_aborted": "incomplete",
    "audit_cleanup_unverified": "unknown",
}
_COUNTS = (
    "issue_count",
    "omitted_count",
    "deleted_audit_count",
    "deleted_staging_count",
    "eligible_audit_count",
    "eligible_staging_count",
)


@dataclass(frozen=True, slots=True)
class CleanupSummary:
    outcome: RecordedCleanupOutcome
    recorded_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class CleanupAssessment:
    summaries: tuple[CleanupSummary, ...] = ()
    damaged: bool = False


def _valid_item(item: dict[str, Any]) -> bool:
    schema = declared_types(ComponentDiagnostic)
    marker = item.get("metadata_invalid")
    if (marker is not None and marker is not False) or item.get("exit_code") != 0:
        return False
    if not all(
        field_valid(name, value, schema)
        for name, value in item.items()
        if name in schema
    ):
        return False
    return all(item.get(name) is None or item[name] >= 0 for name in _COUNTS)


def _summary(items: list[dict[str, Any]], ended: datetime) -> CleanupSummary | None:
    check_ids = {item["check_id"] for item in items if item.get("check_id") is not None}
    if len(check_ids) > 1:
        raise ValueError("conflicting cleanup identities")
    summaries = [item for item in items if "issue_count" in item]
    if not summaries:
        return None  # Legacy sample-only records never establish a completed check.
    if len(summaries) != 1:
        raise ValueError("conflicting cleanup summaries")
    summary = summaries[0]
    if (
        summary.get("category") is not None
        or summary.get("failure_kind") not in _OUTCOMES
    ):
        raise ValueError("invalid cleanup summary")
    outcome = _OUTCOMES[summary["failure_kind"]]
    completed = summary.get("cleanup_completed")
    if completed is not None and completed != (outcome == "completed"):
        raise ValueError("contradictory cleanup outcome")
    _summary_counts(summary, len(items) - 1)
    return CleanupSummary(outcome, _summary_clock(items, ended))


def _summary_counts(summary: dict[str, Any], samples: int) -> None:
    if (
        summary["issue_count"] < samples
        or summary.get("omitted_count", summary["issue_count"] - samples)
        != summary["issue_count"] - samples
    ):
        raise ValueError("invalid cleanup sample count")
    if summary.get("dry_run") is True and any(
        summary.get(name) not in (None, 0)
        for name in ("deleted_audit_count", "deleted_staging_count")
    ):
        raise ValueError("dry cleanup recorded deletions")


def _summary_clock(items: list[dict[str, Any]], ended: datetime) -> datetime | None:
    clocks = [
        aware_moment(item["recorded_at"])
        if item.get("recorded_at") is not None
        else None
        for item in items
    ]
    if (
        any(
            item.get("recorded_at") is not None and clock is None
            for item, clock in zip(items, clocks, strict=True)
        )
        or len(set(clocks)) != 1
    ):
        raise ValueError("invalid cleanup clock")
    if clocks[0] is not None and clocks[0] > ended:
        raise ValueError("future cleanup clock")
    return clocks[0]


def assess_cleanup(row: Mapping[str, Any], ended: datetime) -> CleanupAssessment:
    try:
        raw = json.loads(row.get("failure_details_json") or "null")
    except (ValueError, TypeError):
        return CleanupAssessment()  # Unattributable damage belongs to verification.
    if not isinstance(raw, list):
        return CleanupAssessment()
    groups: dict[tuple[str, object], list[dict[str, Any]]] = {}
    bad_owners: set[str] = set()
    bad_checks: set[str] = set()
    damaged = False
    for item in raw:
        if (
            not isinstance(item, dict)
            or item.get("stage") != "retention_audit_cleanup"
            or is_verification_verdict(item.get("failure_kind"))
        ):
            continue
        identity = assess_progress_identity(
            item.get("progress_id"), item.get("check_id")
        )
        if not _valid_item(item) or not identity.valid:
            damaged = True
            if identity.progress_id is not None:
                bad_owners.add(identity.progress_id)
            if identity.check_id is not None:
                bad_checks.add(identity.check_id)
            continue
        key = (
            ("owner", identity.owner)
            if identity.owner
            else ("clock", item.get("recorded_at"))
        )
        groups.setdefault(key, []).append(item)
    summaries: list[CleanupSummary] = []
    for items in groups.values():
        try:
            if any(
                assess_progress_identity(
                    item.get("progress_id"), item.get("check_id")
                ).owner
                in bad_owners
                or item.get("check_id") in bad_checks
                for item in items
            ):
                raise ValueError("damaged cleanup group")
            if (value := _summary(items, ended)) is not None:
                summaries.append(value)
        except (ValueError, TypeError):
            damaged = True
    return CleanupAssessment(
        tuple(sorted(summaries, key=lambda s: s.recorded_at or ended)), damaged
    )


@dataclass(frozen=True, slots=True)
class CleanupStatus:
    latest: CleanupSummary | None = None
    uncertainty_until: datetime | None = None
    latest_order: datetime | None = None

    def advance(self, row: Mapping[str, Any], ended: datetime) -> "CleanupStatus":
        assessment = assess_cleanup(row, ended)
        boundary = ended if assessment.damaged else self.uncertainty_until
        latest = self.latest
        order = self.latest_order
        for summary in assessment.summaries:
            if order is None or (summary.recorded_at or ended) >= order:
                latest = summary
                order = summary.recorded_at or ended
            if (
                boundary is not None
                and summary.recorded_at is not None
                and summary.recorded_at > boundary
            ):
                boundary = None
        return CleanupStatus(latest, boundary, order)

    @property
    def warning(self) -> str | None:
        if self.uncertainty_until is None:
            return None
        return "Recorded audit-cleanup evidence is damaged or inconsistent; verification and execution recovery are assessed separately. A valid cleanup summary recorded strictly after the damaged run ends resolves this warning."
