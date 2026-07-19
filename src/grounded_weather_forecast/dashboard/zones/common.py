"""Small helpers shared by the zone builders."""

from datetime import datetime

from grounded_weather_forecast.dashboard.copy import PANEL_COPY
from grounded_weather_forecast.dashboard.model import Panel, PanelStatus


def empty_panel(
    copy_key: str,
    panel_id: str,
    title: str,
    status: PanelStatus,
    reason: str,
) -> Panel:
    return Panel(
        panel_id=panel_id,
        title=title,
        status=status,
        copy=PANEL_COPY[copy_key],
        empty_reason=reason,
    )


def hours_ago(now: datetime, then: datetime) -> float:
    return (now - then).total_seconds() / 3600.0


def fmt(value: object, digits: int = 2) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return "—" if value is None else str(value)
