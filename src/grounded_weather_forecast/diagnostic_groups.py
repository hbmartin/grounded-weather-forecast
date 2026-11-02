"""Shared journal decoding and lineage mechanics, independent of verdict policy."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from grounded_weather_forecast.status import (
    DiagnosticIdentity,
    assess_progress_identity,
)


@dataclass(frozen=True, slots=True)
class DiagnosticEntry:
    item: Any
    identity: DiagnosticIdentity


@dataclass(frozen=True, slots=True)
class DiagnosticDocument:
    entries: tuple[DiagnosticEntry, ...] = ()
    error: str | None = None


def decode_diagnostics(row: Mapping[str, Any]) -> DiagnosticDocument:
    try:
        raw = json.loads(row.get("failure_details_json") or "null")
    except (ValueError, TypeError):
        return DiagnosticDocument(error="invalid_diagnostic_json")
    if raw is None:
        return DiagnosticDocument()
    if not isinstance(raw, list):
        return DiagnosticDocument(error="invalid_diagnostic_container")
    return DiagnosticDocument(
        tuple(
            DiagnosticEntry(
                item,
                assess_progress_identity(
                    item.get("progress_id") if isinstance(item, dict) else None,
                    item.get("check_id") if isinstance(item, dict) else None,
                ),
            )
            for item in raw
        )
    )


@dataclass(slots=True)
class DiagnosticGroups:
    groups: dict[tuple[str, object], list[dict[str, Any]]] = field(default_factory=dict)
    identities: dict[tuple[str, object], DiagnosticIdentity] = field(
        default_factory=dict
    )
    check_owners: dict[str, set[str | None]] = field(default_factory=dict)
    bad_owners: set[str] = field(default_factory=set)
    bad_checks: set[str] = field(default_factory=set)

    def quarantine(self, identity: DiagnosticIdentity) -> None:
        if identity.progress_id is not None:
            self.bad_owners.add(identity.progress_id)
        if identity.check_id is not None:
            self.bad_checks.add(identity.check_id)

    def add_valid(self, item: dict[str, Any], identity: DiagnosticIdentity) -> None:
        if identity.check_id is not None:
            self.check_owners.setdefault(identity.check_id, set()).add(identity.owner)
        key = (
            ("owner", identity.owner)
            if identity.owner is not None
            else ("clock", item.get("recorded_at"))
        )
        self.groups.setdefault(key, []).append(item)
        self.identities[key] = identity

    def conflicts(self, key: tuple[str, object]) -> bool:
        return self.identities[key].owner in self.bad_owners or any(
            item.get("check_id") in self.bad_checks
            or len(self.check_owners.get(item.get("check_id"), ())) > 1
            for item in self.groups[key]
        )
