"""Sequence-ordered approvals with shared authorization and verification snapshots.

Promotion holds the ledger lock across elections and publication. Uncertain data
never authorizes an approval and is reported separately for conservative retention.
"""

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from grounded_weather_forecast.clocks import aware_utc
from grounded_weather_forecast.evaluation import ModelRelease
from grounded_weather_forecast.storage import atomic_write_text
from grounded_weather_forecast.verification_text import reason_for_error

type ApprovalDisposition = Literal["none", "first", "reapproval", "unchanged"]
type ApprovalMode = Literal["current", "historical"]


def aware_timestamp(value: object) -> datetime | None:
    """Reject missing zones and clocks that overflow during UTC conversion."""
    return aware_utc(value) if isinstance(value, str) else None


def approval_cutoff(as_of: datetime | None) -> datetime:
    moment = datetime.now(UTC)
    if as_of is None:
        return moment
    cutoff = aware_timestamp(as_of.isoformat())
    if cutoff is None:
        raise ValueError("approval cutoff must be a valid timezone-aware timestamp")
    return min(cutoff, moment)


@dataclass(frozen=True, slots=True)
class ApprovalIssue:
    path: Path
    category: str
    detail: str
    reason_code: str | None = None

    def diagnostic(self) -> str:
        return f"{self.path}: {self.category}: {self.detail}"


def _issue(issues: list[ApprovalIssue], path: Path, kind: str, detail: object) -> None:
    issues.append(
        ApprovalIssue(
            path, kind, str(detail).replace("\n", " ")[:400], reason_for_error(detail)
        )
    )


def json_entries(directory: Path, issues: list[ApprovalIssue]) -> tuple[Path, ...]:
    """A missing directory is empty; denied listings and broken links are unknown."""
    try:
        with os.scandir(directory) as entries:
            return tuple(
                sorted(directory / e.name for e in entries if e.name.endswith(".json"))
            )
    except (FileNotFoundError,):  # noqa: B013 -- distinguish a dangling directory link
        try:
            directory.lstat()
        except (FileNotFoundError,):  # noqa: B013
            return ()
        except (OSError,) as error:  # noqa: B013
            _issue(issues, directory, "directory_unreadable", error)
        else:
            _issue(issues, directory, "directory_unreadable", "dangling directory link")
    except (OSError,) as error:  # noqa: B013
        _issue(issues, directory, "directory_unreadable", error)
    return ()


def _object(path: Path, issues: list[ApprovalIssue]) -> dict[str, object] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw
        _issue(issues, path, "non_object", "expected a JSON object")
    except (OSError, ValueError) as error:
        _issue(issues, path, "unreadable", error)
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


class ApprovalPublicationError(ValueError):
    """Expected ledger/publication failure; completed evaluations remain recorded."""

    failure_kind = "promotion_publication_failed"


class StaleApprovalSnapshotError(ApprovalPublicationError):
    failure_kind = "promotion_snapshot_stale"


def _event(path: Path, issues: list[ApprovalIssue]) -> ApprovalEvent | None:
    if (raw := _object(path, issues)) is None:
        return None
    sequence = raw.get("sequence")
    release = raw.get("release_id")
    approved = raw.get("approved_at")
    visible = raw.get("visible_at")
    if (
        type(sequence) is not int
        or sequence <= 0
        or not isinstance(release, str)
        or not release
        or raw.get("approval_id") != path.stem
        or path.stem != f"{sequence:020d}-{release}"
        or aware_timestamp(approved) is None
        or (visible is not None and aware_timestamp(visible) is None)
    ):
        _issue(
            issues,
            path,
            "invalid_event",
            "invalid approval identity, sequence or clock",
        )
        return None
    # Read known fields explicitly: future additive fields must not hide an event.
    return ApprovalEvent(
        path.stem,
        release,
        str(approved),
        sequence,
        str(visible) if visible is not None else None,
    )


def _reservation(paths: Sequence[Path]) -> int:
    prefixes = (path.stem.partition("-")[0] for path in paths)
    return max((int(p) for p in prefixes if p.isascii() and p.isdecimal()), default=0)


def _reserved_sequence(directory: Path) -> int:
    issues: list[ApprovalIssue] = []
    value = _reservation(json_entries(directory / "approvals", issues))
    if issues:
        raise ApprovalPublicationError("; ".join(i.diagnostic() for i in issues))
    return value


def _visibility_floors(
    releases: Sequence[dict[str, object]], events: Sequence[ApprovalEvent]
) -> Mapping[str, datetime]:
    publications = {
        str(r.get("release_id")): aware_timestamp(r.get("promoted_at"))
        for r in releases
    }
    floor = datetime.min.replace(tzinfo=UTC)
    visible = {}
    for event in sorted(events, key=lambda event: event.sequence):
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
    return MappingProxyType(visible)


@dataclass(frozen=True, slots=True)
class ApprovalSnapshot:
    """One ledger view; derived visibility and authorization order are computed once."""

    releases: tuple[dict[str, object], ...]
    events: tuple[ApprovalEvent, ...]
    cutoff: datetime
    reserved_sequence: int
    issues: tuple[str, ...] = ()
    mode: ApprovalMode = "current"
    diagnostics: tuple[ApprovalIssue, ...] = ()
    _visibility: Mapping[str, datetime] = field(init=False, repr=False, compare=False)
    _latest: Mapping[ApprovalMode, Mapping[str, ApprovalEvent]] = field(
        init=False, repr=False, compare=False
    )
    _ordered: Mapping[ApprovalMode, tuple[ApprovedRelease, ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_visibility", _visibility_floors(self.releases, self.events)
        )
        latest: dict[ApprovalMode, dict[str, ApprovalEvent]] = {
            "current": {},
            "historical": {},
        }
        for event in sorted(self.events, key=lambda event: event.sequence):
            visible = self._visibility.get(event.approval_id)
            if visible is not None:
                latest["current"][event.release_id] = event
                if visible <= self.cutoff:
                    latest["historical"][event.release_id] = event
        object.__setattr__(
            self,
            "_latest",
            MappingProxyType({m: MappingProxyType(v) for m, v in latest.items()}),
        )
        object.__setattr__(
            self,
            "_ordered",
            MappingProxyType(
                {m: tuple(self._records(self.releases, m)) for m in latest}
            ),
        )

    def visibility(self) -> Mapping[str, datetime]:
        return self._visibility

    def _records(
        self, releases: Sequence[dict[str, object]], mode: ApprovalMode
    ) -> list[ApprovedRelease]:
        approved = []
        for raw in releases:
            published = aware_timestamp(raw.get("promoted_at"))
            if not isinstance(raw.get("release_id"), str) or published is None:
                continue
            if mode == "historical" and published > self.cutoff:
                continue
            event = self._latest[mode].get(str(raw["release_id"]))
            if event is not None:
                moment = aware_timestamp(event.approved_at)
                if moment is not None:
                    approved.append(
                        ApprovedRelease(
                            raw,
                            moment,
                            event.sequence,
                            event.approval_id,
                            self._visibility[event.approval_id],
                        )
                    )
            elif (
                raw.get("approval_required", False) is False
                and published <= self.cutoff
            ):
                approved.append(ApprovedRelease(raw, published))
        return sorted(approved, key=lambda item: item.order, reverse=True)

    def records(
        self,
        releases: Sequence[dict[str, object]] | None = None,
        *,
        mode: ApprovalMode | None = None,
    ) -> list[ApprovedRelease]:
        chosen = mode or self.mode
        return (
            list(self._ordered[chosen])
            if releases is None
            else self._records(releases, chosen)
        )


def _release(
    path: Path,
    issues: list[ApprovalIssue],
    *,
    cutoff: datetime,
    legacy_references: bool,
) -> dict[str, object] | None:
    if (raw := _object(path, issues)) is None:
        return None
    if legacy_references and raw.get("approval_required", False) is False:
        raw.setdefault("release_id", path.stem)
    if not isinstance(raw.get("release_id"), str) or raw["release_id"] != path.stem:
        _issue(
            issues,
            path,
            "legacy_identity_missing"
            if (
                "release_id" not in raw
                and raw.get("approval_required", False) is False
                and (clock := aware_timestamp(raw.get("promoted_at"))) is not None
                and clock <= cutoff
            )
            else "release_identity",
            f"release_id does not match filename {path.stem}",
        )
        return None
    published = aware_timestamp(raw.get("promoted_at"))
    if published is None:
        _issue(
            issues,
            path,
            "publication_clock",
            "missing or invalid timezone-aware publication clock",
        )
    elif raw.get("approval_required", False) is False and published > cutoff:
        _issue(
            issues,
            path,
            "legacy_future_clock",
            "legacy publication clock is beyond the cutoff",
        )
    return raw


def read_approval_snapshot(
    directory: Path,
    *,
    as_of: datetime | None = None,
    legacy_references: bool = False,
    mode: ApprovalMode | None = None,
) -> ApprovalSnapshot:
    cutoff = approval_cutoff(as_of)
    diagnostics: list[ApprovalIssue] = []
    releases = tuple(
        raw
        for path in json_entries(directory, diagnostics)
        if (
            raw := _release(
                path, diagnostics, cutoff=cutoff, legacy_references=legacy_references
            )
        )
        is not None
    )
    paths = json_entries(directory / "approvals", diagnostics)
    events = tuple(
        event for path in paths if (event := _event(path, diagnostics)) is not None
    )
    return ApprovalSnapshot(
        releases,
        events,
        cutoff,
        _reservation(paths),
        tuple(dict.fromkeys(str(i.path) for i in diagnostics)),
        mode or ("historical" if as_of is not None else "current"),
        tuple(diagnostics),
    )


def ordered_approvals(
    releases: Sequence[dict[str, object]], directory: Path, as_of: datetime
) -> list[ApprovedRelease]:
    """Historical compatibility adapter; clocks bound visibility, sequence orders it."""
    return read_approval_snapshot(directory, as_of=as_of).records(releases)


def _publish(
    release: ModelRelease,
    directory: Path,
    current: ApprovedRelease | None,
    snapshot: ApprovalSnapshot,
) -> ApprovalCommit:
    reserved = _reserved_sequence(directory)
    if reserved > snapshot.reserved_sequence:
        raise StaleApprovalSnapshotError(
            "approval ledger changed after election; rerun promotion"
        )
    if current is not None and current.release["release_id"] == release.release_id:
        return ApprovalCommit(approval_id=current.approval_id, disposition="unchanged")
    prior = any(
        record.release["release_id"] == release.release_id
        for record in snapshot.records(mode="current")
    )
    path, created = replace(release, approval_required=True).write_once(directory)
    raw = json.loads(path.read_text(encoding="utf-8"))
    published = (
        aware_timestamp(raw.get("promoted_at")) if isinstance(raw, dict) else None
    )
    if published is None:
        raise ApprovalPublicationError(
            f"release {release.release_id}: invalid publication clock"
        )
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
    destination = directory / "approvals" / f"{approval_id}.json"
    # All cooperating writers hold the ledger lock. Refuse even a damaged reserved entry.
    if destination.exists():
        raise StaleApprovalSnapshotError(
            f"approval entry already exists: {destination}"
        )
    atomic_write_text(json.dumps(asdict(event), indent=2), destination)
    return ApprovalCommit(
        created=created,
        changed=True,
        approval_id=approval_id,
        disposition="reapproval" if prior else "first",
    )


def publish_approval(
    release: ModelRelease,
    directory: Path,
    current: ApprovedRelease | None,
    *,
    snapshot: ApprovalSnapshot | None = None,
) -> ApprovalCommit:
    """Caller holds the ledger lock; expected publication failures are typed."""
    try:
        return _publish(
            release, directory, current, snapshot or read_approval_snapshot(directory)
        )
    except (ApprovalPublicationError,):  # noqa: B013
        raise
    except (OSError, ValueError) as error:
        raise ApprovalPublicationError(
            f"release {release.release_id}: {error}"
        ) from error


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
