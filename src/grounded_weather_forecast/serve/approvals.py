"""Ordered approval events for immutable release content.

Promotion holds the directory's approval-ledger lock across reading incumbents,
electing, publishing content, and committing an event. Readers need no lock:
event publication is atomic and event-managed content alone is not approved.
"""

import json
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from grounded_weather_forecast.evaluation import ModelRelease
from grounded_weather_forecast.storage import atomic_write_text


def aware_timestamp(value: object) -> datetime | None:
    """Accept only explicitly zoned clocks, comparing actual UTC instants."""
    if not isinstance(value, str):
        return None
    with suppress(ValueError):
        moment = datetime.fromisoformat(value)
        if moment.tzinfo is not None and moment.utcoffset() is not None:
            return moment.astimezone(UTC)
    return None


@dataclass(frozen=True, slots=True)
class ApprovalEvent:
    approval_id: str
    release_id: str
    approved_at: str
    sequence: int


@dataclass(frozen=True, slots=True)
class ApprovedRelease:
    release: dict[str, object]
    approved_at: datetime
    sequence: int = 0
    approval_id: str | None = None

    @property
    def order(self) -> tuple[datetime, int, str]:
        return self.approved_at, self.sequence, str(self.release["release_id"])


@dataclass(frozen=True, slots=True)
class ApprovalCommit:
    created: bool = False
    changed: bool = False
    approval_id: str | None = None


def _events(directory: Path) -> list[ApprovalEvent]:
    events = []
    for path in (directory / "approvals").glob("*.json"):
        with suppress(OSError, ValueError, TypeError):
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                continue
            event = ApprovalEvent(**raw)
            if (
                isinstance(event.sequence, int)
                and not isinstance(event.sequence, bool)
                and event.sequence > 0
                and isinstance(event.release_id, str)
                and event.approval_id == path.stem
                and event.approval_id == f"{event.sequence:020d}-{event.release_id}"
                and aware_timestamp(event.approved_at) is not None
            ):
                events.append(event)
    return events


def ordered_approvals(
    releases: Sequence[dict[str, object]], directory: Path, as_of: datetime
) -> list[ApprovedRelease]:
    """Newest approval per release, with implicit events for legacy content."""
    cutoff = min(as_of, datetime.now(UTC))
    latest: dict[str, ApprovalEvent] = {}
    for event in _events(directory):
        moment = aware_timestamp(event.approved_at)
        if moment is None or moment > cutoff:
            continue
        previous = latest.get(event.release_id)
        previous_moment = aware_timestamp(previous.approved_at) if previous else None
        if (
            previous_moment is None
            or previous is None
            or (moment, event.sequence) > (previous_moment, previous.sequence)
        ):
            latest[event.release_id] = event
    approved = []
    for raw in releases:
        if not isinstance(raw.get("release_id"), str):
            continue
        published = aware_timestamp(raw.get("promoted_at"))
        if published is None or published > cutoff:
            continue
        event = latest.get(str(raw.get("release_id")))
        if event is not None:
            moment = aware_timestamp(event.approved_at)
            if moment is not None and moment >= published:
                approved.append(
                    ApprovedRelease(raw, moment, event.sequence, event.approval_id)
                )
        elif raw.get("approval_required", False) is False:
            approved.append(ApprovedRelease(raw, published))
    return sorted(approved, key=lambda item: item.order, reverse=True)


def publish_approval(
    release: ModelRelease,
    directory: Path,
    current: ApprovedRelease | None,
) -> ApprovalCommit:
    """Commit a changed approval; caller must hold the approval-ledger lock."""
    if current is not None and current.release["release_id"] == release.release_id:
        return ApprovalCommit(approval_id=current.approval_id)
    _, created = replace(release, approval_required=True).write_once(directory)
    # Reserve names even for damaged events: append history, never overwrite it.
    sequences = (
        int(prefix)
        for path in (directory / "approvals").glob("*.json")
        if (prefix := path.stem.partition("-")[0]).isdigit()
    )
    sequence = max(sequences, default=0) + 1
    moment = datetime.now(UTC)
    # A clock stepping backwards must not make this decision older than the
    # one it replaces. Equal instants are ordered by the locked sequence.
    if current is not None:
        moment = max(moment, current.approved_at)
    approval_id = f"{sequence:020d}-{release.release_id}"
    event = ApprovalEvent(approval_id, release.release_id, moment.isoformat(), sequence)
    atomic_write_text(
        json.dumps(asdict(event), indent=2),
        directory / "approvals" / f"{approval_id}.json",
    )
    return ApprovalCommit(created=created, changed=True, approval_id=approval_id)


def approved_payloads(
    releases: Sequence[dict[str, object]], directory: Path, as_of: datetime
) -> list[dict[str, object]]:
    """Read models for reports without relabelling content publication time."""
    return [
        record.release
        | {
            "approved_at": record.approved_at.isoformat(),
            "approval_id": record.approval_id,
            "approval_sequence": record.sequence,
        }
        for record in ordered_approvals(releases, directory, as_of)
    ]


def approval_sort_key(release: Mapping[str, object]) -> tuple[datetime, int, str]:
    moment = aware_timestamp(release.get("approved_at", release.get("promoted_at")))
    sequence = release.get("approval_sequence")
    return (
        moment or datetime.min.replace(tzinfo=UTC),
        sequence if isinstance(sequence, int) else 0,
        str(release.get("release_id", "")),
    )


def read_approved_releases(
    directory: Path, *, legacy_references: bool = False
) -> list[dict[str, object]]:
    """Best-effort approved models for diagnostics and retention consumers."""
    releases = []
    for path in directory.glob("*.json"):
        with suppress(OSError, ValueError):
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                # Older retention ledgers recorded references without a serving
                # selection map or explicit id. Preserve their protection only;
                # they cannot justify a forecast or authorize managed content.
                if legacy_references and raw.get("approval_required", False) is False:
                    raw.setdefault("release_id", path.stem)
                if legacy_references or isinstance(raw.get("selections"), dict):
                    releases.append(raw)
    return approved_payloads(releases, directory, datetime.now(UTC))
