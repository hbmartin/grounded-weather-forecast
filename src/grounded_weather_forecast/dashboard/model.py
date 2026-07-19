"""Typed building blocks the dashboard page is rendered from."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

type PanelStatus = Literal["ok", "info", "amber", "red"]


@dataclass(frozen=True, slots=True)
class PanelCopy:
    """The explanatory prose every panel carries."""

    what: str
    why: str
    thresholds: str


@dataclass(frozen=True, slots=True)
class Stat:
    """One headline figure rendered as a chip above a panel's viz."""

    label: str
    value: str
    status: PanelStatus = "ok"


@dataclass(frozen=True, slots=True)
class ChartSpec:
    """A complete, JSON-safe Chart.js config; colors are role tokens."""

    config: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class TableSpec:
    """A table with optional per-cell CSS classes and hover titles."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    cell_classes: tuple[tuple[str, ...], ...] = ()
    cell_titles: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class Panel:
    panel_id: str
    title: str
    status: PanelStatus
    copy: PanelCopy
    stats: tuple[Stat, ...] = ()
    chart: ChartSpec | None = None
    table: TableSpec | None = None
    empty_reason: str | None = None
    intro: str | None = None
    raw_html: str | None = None
    """Pre-rendered safe HTML (static scaffolding only, never user data)."""


@dataclass(frozen=True, slots=True)
class Zone:
    zone_id: str
    title: str
    intro: str
    panels: tuple[Panel, ...]
