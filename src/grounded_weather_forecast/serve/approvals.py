"""Approval snapshots and sequence-ordered events for immutable release content.

Promotion holds the ledger lock from its snapshot read through event publication.
Readers require no lock: content alone cannot authorize an event-managed release.
"""

import json
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from grounded_weather_forecast.evaluation import ModelRelease
from grounded_weather_forecast.storage import atomic_write_text

type ApprovalDisposition = Literal["none", "first", "reapproval", "unchanged"]
type ApprovalMode = Literal["current", "historical"]


def aware_timestamp(value: object) -> datetime | None:
    """Reject missing zones and clocks that overflow during UTC conversion."""
    if isinstance(value, str):
        with suppress(ValueError, OverflowError):
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
    visible_at: str | None = None


@dataclass(frozen=True, slots=True)
class ApprovedRelease:
    release: dict[str, object]
    approved_at: datetime
    sequence: int = 0
    approval_id: str | None = None
    visible_at: datetime | None = None

    @property
    def order(self) -> tuple[bool, int, datetime, str]:
        return (
            bool(self.sequence),
            self.sequence,
            self.approved_at,
            str(self.release["release_id"]),
        )

    def payload(self) -> dict[str, object]:
        return self.release | {
            "approved_at": self.approved_at.isoformat(),
            "approval_id": self.approval_id,
            "approval_sequence": self.sequence,
            "visible_at": (self.visible_at or self.approved_at).isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ApprovalCommit:
    created: bool = False
    changed: bool = False
    approval_id: str | None = None
    disposition: ApprovalDisposition = "none"


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
                and (
                    event.visible_at is None
                    or aware_timestamp(event.visible_at) is not None
                )
            ):
                events.append(event)
    return events


def _reserved_sequence(directory: Path) -> int:
    prefixes = (
        path.stem.partition("-")[0] for path in (directory / "approvals").glob("*.json")
    )
    return max((int(p) for p in prefixes if p.isascii() and p.isdecimal()), default=0)


@dataclass(frozen=True, slots=True)
class ApprovalSnapshot:
    """One fixed ledger view and visibility cutoff, shared throughout a promotion."""

    releases: tuple[dict[str, object], ...]
    events: tuple[ApprovalEvent, ...]
    cutoff: datetime
    reserved_sequence: int
    issues: tuple[str, ...] = ()
    mode: ApprovalMode = "current"

    def visibility(self) -> dict[str, datetime]:
        """Conservative sequence-prefix floors, including pre-watermark events."""
        publications = {
            str(raw.get("release_id")): aware_timestamp(raw.get("promoted_at"))
            for raw in self.releases
        }
        floor = datetime.min.replace(tzinfo=UTC)
        visible = {}
        for event in sorted(self.events, key=lambda event: event.sequence):
            moment = aware_timestamp(event.approved_at)
            stored = aware_timestamp(event.visible_at)
            if moment is None or (event.visible_at is not None and stored is None):
                continue
            floor = max(
                floor,
                moment,
                stored or moment,
                publications.get(event.release_id) or moment,
            )
            visible[event.approval_id] = floor
        return visible

    def records(
        self, releases: Sequence[dict[str, object]] | None = None
    ) -> list[ApprovedRelease]:
        latest: dict[str, ApprovalEvent] = {}
        visibility = self.visibility()
        for event in self.events:
            visible = visibility.get(event.approval_id)
            if visible is None or (self.mode == "historical" and visible > self.cutoff):
                continue
            previous = latest.get(event.release_id)
            if previous is None or event.sequence > previous.sequence:
                latest[event.release_id] = event
        approved = []
        for raw in self.releases if releases is None else releases:
            published = aware_timestamp(raw.get("promoted_at"))
            if not isinstance(raw.get("release_id"), str) or published is None:
                continue
            event = latest.get(str(raw["release_id"]))
            if event is not None:
                moment = aware_timestamp(event.approved_at)
                if moment is not None:
                    approved.append(
                        ApprovedRelease(
                            raw,
                            moment,
                            event.sequence,
                            event.approval_id,
                            visibility[event.approval_id],
                        )
                    )
            elif (
                raw.get("approval_required", False) is False
                and published <= self.cutoff
            ):
                approved.append(ApprovedRelease(raw, published))
        return sorted(approved, key=lambda item: item.order, reverse=True)


def read_approval_snapshot(
    directory: Path,
    *,
    as_of: datetime | None = None,
    legacy_references: bool = False,
    mode: ApprovalMode | None = None,
) -> ApprovalSnapshot:
    moment = datetime.now(UTC)
    cutoff = min(as_of, moment) if as_of is not None else moment
    releases = []
    issues = []
    for path in sorted(directory.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            issues.append(str(path))
            continue
        if isinstance(raw, dict):
            if legacy_references and raw.get("approval_required", False) is False:
                raw.setdefault("release_id", path.stem)
            releases.append(raw)
        else:
            issues.append(str(path))
    return ApprovalSnapshot(
        tuple(releases),
        tuple(_events(directory)),
        cutoff,
        _reserved_sequence(directory),
        tuple(issues),
        mode or ("historical" if as_of is not None else "current"),
    )


def ordered_approvals(
    releases: Sequence[dict[str, object]], directory: Path, as_of: datetime
) -> list[ApprovedRelease]:
    """Compatibility adapter; sequence determines supersession, clocks eligibility."""
    return read_approval_snapshot(directory, as_of=as_of).records(releases)


def publish_approval(
    release: ModelRelease,
    directory: Path,
    current: ApprovedRelease | None,
    *,
    snapshot: ApprovalSnapshot | None = None,
) -> ApprovalCommit:
    """Caller holds the ledger lock; commit the event after immutable content."""
    snapshot = snapshot or read_approval_snapshot(directory)
    reserved = _reserved_sequence(directory)
    if reserved > snapshot.reserved_sequence:
        snapshot = read_approval_snapshot(directory)
        records = snapshot.records()
        current = records[0] if records else None
    if current is not None and current.release["release_id"] == release.release_id:
        return ApprovalCommit(approval_id=current.approval_id, disposition="unchanged")
    prior = any(
        record.release["release_id"] == release.release_id
        for record in replace(snapshot, mode="current").records()
    )
    path, created = replace(release, approval_required=True).write_once(directory)
    published = aware_timestamp(
        json.loads(path.read_text(encoding="utf-8")).get("promoted_at")
    )
    if published is None:
        raise ValueError(f"release {release.release_id}: invalid publication clock")
    moment = datetime.now(UTC)
    visible = max(moment, published, *snapshot.visibility().values())
    sequence = max(snapshot.reserved_sequence, reserved) + 1
    approval_id = f"{sequence:020d}-{release.release_id}"
    event = ApprovalEvent(
        approval_id,
        release.release_id,
        moment.isoformat(),
        sequence,
        visible.isoformat(),
    )
    atomic_write_text(
        json.dumps(asdict(event), indent=2),
        directory / "approvals" / f"{approval_id}.json",
    )
    return ApprovalCommit(
        created=created,
        changed=True,
        approval_id=approval_id,
        disposition="reapproval" if prior else "first",
    )


def approved_payloads(
    releases: Sequence[dict[str, object]], directory: Path, as_of: datetime
) -> list[dict[str, object]]:
    return [
        record.payload() for record in ordered_approvals(releases, directory, as_of)
    ]


def approval_sort_key(release: Mapping[str, object]) -> tuple[bool, int, datetime, str]:
    moment = aware_timestamp(release.get("approved_at", release.get("promoted_at")))
    raw_sequence = release.get("approval_sequence")
    sequence = (
        raw_sequence
        if isinstance(raw_sequence, int) and not isinstance(raw_sequence, bool)
        else 0
    )
    return (
        bool(sequence),
        sequence,
        moment or datetime.min.replace(tzinfo=UTC),
        str(release.get("release_id", "")),
    )


def read_approved_releases(
    directory: Path,
    *,
    as_of: datetime | None = None,
    legacy_references: bool = False,
    mode: ApprovalMode | None = None,
) -> list[dict[str, object]]:
    snapshot = read_approval_snapshot(
        directory, as_of=as_of, legacy_references=legacy_references, mode=mode
    )
    return [
        record.payload()
        for record in snapshot.records()
        if legacy_references or isinstance(record.release.get("selections"), dict)
    ]
