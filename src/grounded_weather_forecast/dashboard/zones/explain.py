"""Zone G: explainability drill-down over the latest served document."""

from grounded_weather_forecast.dashboard.context import DashboardContext
from grounded_weather_forecast.dashboard.copy import PANEL_COPY, ZONE_INTROS
from grounded_weather_forecast.dashboard.derive import Derived
from grounded_weather_forecast.dashboard.model import Panel, Stat, Zone
from grounded_weather_forecast.dashboard.zones.common import empty_panel

_PICKER_HTML = (
    '<div id="explain-root">'
    '<div class="explain-controls">'
    '<select id="explain-product" aria-label="product"></select>'
    '<select id="explain-time" aria-label="valid time"></select>'
    '<select id="explain-variable" aria-label="variable"></select>'
    "</div>"
    '<div id="explain-detail"><p class="empty-state">pick a served point to '
    "see its provenance</p></div>"
    "</div>"
)


def build(ctx: DashboardContext, derived: Derived) -> Zone:  # noqa: ARG001
    forecast = ctx.latest_forecast
    if forecast is None:
        panel = empty_panel(
            "g1",
            "g1",
            "Why is this value what it is?",
            "info",
            "no served document to explain yet — the drill-down lights up "
            "after the first `predict` run archives a forecast",
        )
    else:
        panel = Panel(
            panel_id="g1",
            title="Why is this value what it is?",
            status="ok",
            copy=PANEL_COPY["g1"],
            stats=(
                Stat("document", forecast.issued_at),
                Stat("hourly points", str(len(forecast.hourly))),
                Stat("daily points", str(len(forecast.daily))),
            ),
            raw_html=_PICKER_HTML,
        )
    return Zone(
        zone_id="G",
        title="Explainability",
        intro=ZONE_INTROS["G"],
        panels=(panel,),
    )
