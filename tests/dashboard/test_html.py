import json
import re

from grounded_weather_forecast.dashboard.copy import PANEL_COPY
from grounded_weather_forecast.dashboard.html import esc, render_page, render_panel
from grounded_weather_forecast.dashboard.model import Panel, TableSpec, Zone
from grounded_weather_forecast.reports.alerts import Alert


def make_panel(**overrides):
    defaults = {
        "panel_id": "a1",
        "title": "Test <panel>",
        "status": "red",
        "copy": PANEL_COPY["a1"],
        "empty_reason": "nothing here yet",
    }
    return Panel(**{**defaults, **overrides})


def make_page(payload=None, alerts=()):
    zone = Zone(zone_id="A", title="t", intro="i", panels=(make_panel(),))
    return render_page(
        title="Console",
        generated_at="2026-07-18T12:00:00+00:00",
        fingerprints={"dataset": "abc"},
        version="0.0.0",
        alerts=alerts,
        zones=(zone,),
        payload=payload if payload is not None else {"charts": {}},
    )


def test_esc_escapes_markup():
    assert esc("<script>") == "&lt;script&gt;"


def test_empty_panel_renders_loud_without_canvas():
    rendered = render_panel(make_panel())
    assert "nothing here yet" in rendered
    assert "<canvas" not in rendered
    assert "Test &lt;panel&gt;" in rendered


def test_table_cell_classes_and_titles():
    panel = make_panel(
        empty_reason=None,
        table=TableSpec(
            columns=("a",),
            rows=(("<b>",),),
            cell_classes=(("cell-bad",),),
            cell_titles=(("why",),),
        ),
    )
    rendered = render_panel(panel)
    assert '<td class="cell-bad" title="why">&lt;b&gt;</td>' in rendered


def test_page_has_no_external_references():
    page = make_page()
    assert 'src="http' not in page
    assert 'href="http' not in page
    assert "@import" not in page
    assert "Chart.js v" in page  # vendored library marker


def test_payload_round_trips_and_neutralizes_script_close():
    hostile = "</script><script>alert(1)</script>"
    page = make_page(payload={"charts": {}, "note": hostile})
    match = re.search(
        r'<script id="dashboard-data" type="application/json">(.*?)</script>',
        page,
        re.S,
    )
    assert match is not None
    raw = match.group(1)
    assert "</script" not in raw
    assert json.loads(raw)["note"] == hostile


def test_alert_strip_orders_and_links():
    alerts = (
        Alert("red", "A", "x", "boom <tag>", "threshold t"),
        Alert("info", "B", "y", "not evaluable yet: z", "t", evaluable=False),
    )
    page = make_page(alerts=alerts)
    assert 'href="#zone-A"' in page
    assert "boom &lt;tag&gt;" in page
    assert "alert-info" in page
