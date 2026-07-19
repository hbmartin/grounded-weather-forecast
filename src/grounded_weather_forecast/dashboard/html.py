"""String-building HTML renderers for the dashboard page.

Convention: every interpolated value passes through ``esc``; the only raw
concatenation is of fragments returned by the ``render_*`` functions in this
module, which are safe by construction.
"""

import html as html_module
import json
from collections.abc import Mapping, Sequence
from importlib import resources

from grounded_weather_forecast.dashboard.charts import json_safe
from grounded_weather_forecast.dashboard.model import Panel, TableSpec, Zone
from grounded_weather_forecast.reports.alerts import Alert

_STATUS_LABELS: Mapping[str, str] = {
    "ok": "ok",
    "info": "info",
    "amber": "attention",
    "red": "failing",
}


def esc(value: object) -> str:
    return html_module.escape(str(value), quote=True)


def format_cell(value: object) -> str:
    match value:
        case None:
            return ""
        case float():
            return f"{value:.3f}"
        case _:
            return str(value)


def load_asset(name: str) -> str:
    package = resources.files("grounded_weather_forecast.dashboard")
    return (package / "assets" / name).read_text(encoding="utf-8")


def render_table(table: TableSpec) -> str:
    head = "".join(f"<th>{esc(column)}</th>" for column in table.columns)
    body_rows: list[str] = []
    for row_index, row in enumerate(table.rows):
        cells: list[str] = []
        for column_index, cell in enumerate(row):
            css = _lookup(table.cell_classes, row_index, column_index)
            title = _lookup(table.cell_titles, row_index, column_index)
            attributes = (f' class="{esc(css)}"' if css else "") + (
                f' title="{esc(title)}"' if title else ""
            )
            cells.append(f"<td{attributes}>{esc(cell)}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<div class="table-scroll"><table>'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def _lookup(
    grid: tuple[tuple[str, ...], ...], row_index: int, column_index: int
) -> str:
    if row_index >= len(grid):
        return ""
    row = grid[row_index]
    return row[column_index] if column_index < len(row) else ""


def render_panel(panel: Panel) -> str:
    stats = "".join(
        f'<span class="stat stat-{esc(stat.status)}">'
        f'<span class="stat-label">{esc(stat.label)}</span> '
        f'<span class="stat-value">{esc(stat.value)}</span></span>'
        for stat in panel.stats
    )
    parts: list[str] = [
        f'<section class="panel panel-{esc(panel.status)}" '
        f'id="panel-{esc(panel.panel_id)}">',
        f'<h3><span class="dot dot-{esc(panel.status)}" '
        f'title="{esc(_STATUS_LABELS.get(panel.status, panel.status))}"></span>'
        f"{esc(panel.title)}</h3>",
    ]
    if stats:
        parts.append(f'<div class="stats">{stats}</div>')
    if panel.intro:
        parts.append(f'<p class="panel-intro">{esc(panel.intro)}</p>')
    if panel.chart is not None:
        parts.append(
            f'<div class="chart-box"><canvas id="chart-{esc(panel.panel_id)}">'
            "</canvas></div>"
        )
    if panel.table is not None:
        parts.append(render_table(panel.table))
    if panel.raw_html is not None:
        parts.append(panel.raw_html)
    if panel.chart is None and panel.table is None and panel.raw_html is None:
        reason = panel.empty_reason or "no data yet"
        parts.append(f'<p class="empty-state">{esc(reason)}</p>')
    parts.append(
        "<details open><summary>about this panel</summary>"
        f"<p><strong>What:</strong> {esc(panel.copy.what)}</p>"
        f"<p><strong>Why it matters:</strong> {esc(panel.copy.why)}</p>"
        f"<p><strong>Thresholds:</strong> {esc(panel.copy.thresholds)}</p>"
        "</details>"
    )
    parts.append("</section>")
    return "".join(parts)


def render_zone(zone: Zone) -> str:
    panels = "".join(render_panel(panel) for panel in zone.panels)
    return (
        f'<section class="zone" id="zone-{esc(zone.zone_id)}">'
        f"<h2>Zone {esc(zone.zone_id)} — {esc(zone.title)}</h2>"
        f'<p class="zone-intro">{esc(zone.intro)}</p>'
        f'<div class="panel-grid">{panels}</div>'
        "</section>"
    )


def render_alert_strip(alerts: Sequence[Alert]) -> str:
    if not alerts:
        return (
            '<div class="alert-strip"><span class="alert alert-ok">'
            "all alert families evaluated clean</span></div>"
        )
    rendered = []
    for alert in alerts:
        css = "alert-info" if not alert.evaluable else f"alert-{alert.severity}"
        rendered.append(
            f'<a class="alert {css}" href="#zone-{esc(alert.zone)}" '
            f'title="{esc(alert.threshold)}">'
            f'<span class="alert-id">{esc(alert.panel_id)}</span> '
            f"{esc(alert.message)}</a>"
        )
    return f'<div class="alert-strip">{"".join(rendered)}</div>'


def render_page(
    *,
    title: str,
    generated_at: str,
    fingerprints: Mapping[str, str],
    version: str,
    alerts: Sequence[Alert],
    zones: Sequence[Zone],
    payload: Mapping[str, object],
) -> str:
    payload_json = json.dumps(json_safe(payload), allow_nan=False).replace("</", "<\\/")
    nav = "".join(
        f'<a class="pill" href="#zone-{esc(zone.zone_id)}">{esc(zone.zone_id)} · '
        f"{esc(zone.title)}</a>"
        for zone in zones
    )
    prints = " · ".join(
        f"{esc(name)} <code>{esc(value)}</code>" for name, value in fingerprints.items()
    )
    body = "".join(render_zone(zone) for zone in zones)
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{esc(title)}</title>"
        f"<style>{load_asset('dashboard.css')}</style>"
        "</head><body>"
        "<header>"
        f"<h1>{esc(title)}</h1>"
        f'<p class="meta">generated {esc(generated_at)} · v{esc(version)} · '
        f"{prints}</p>"
        "</header>"
        f"{render_alert_strip(alerts)}"
        f'<nav class="pills">{nav}</nav>'
        f"<main>{body}</main>"
        '<footer><p>Self-contained static page written by the "report" '
        "command; every figure is a read-only projection of the on-disk "
        "artifacts.</p></footer>"
        f"<script>{load_asset('chart.umd.min.js')}</script>"
        f'<script id="dashboard-data" type="application/json">{payload_json}'
        "</script>"
        f"<script>{load_asset('dashboard.js')}</script>"
        "</body></html>"
    )
