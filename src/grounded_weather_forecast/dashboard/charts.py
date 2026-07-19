"""Chart.js config builders and heatmap-table helpers.

Configs carry color *role tokens* (``series-1`` … ``series-8``, status roles,
``muted``) instead of hex: the page script resolves tokens against the light
or dark palette at render time, because canvas cannot follow CSS variables.
Time axes are always ``category`` with labels pre-formatted here — the
Chart.js ``time`` scale needs a date adapter this page deliberately omits.
"""

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime

from grounded_weather_forecast.dashboard.model import ChartSpec

SERIES_TOKENS: tuple[str, ...] = tuple(f"series-{i}" for i in range(1, 9))
type Series = Sequence[tuple[str, Sequence[float | None]]]


def series_token(index: int) -> str:
    """Fixed-order categorical slot; callers cap series at eight."""
    return SERIES_TOKENS[index % len(SERIES_TOKENS)]


def json_safe(value: object) -> object:
    """Recursively make a payload ``json.dumps(allow_nan=False)``-safe."""
    match value:
        case float() if not math.isfinite(value):
            return None
        case bool() | int() | float() | str() | None:
            return value
        case datetime() | date():
            return value.isoformat()
        case Mapping():
            return {str(key): json_safe(item) for key, item in value.items()}
        case list() | tuple():
            return [json_safe(item) for item in value]
        case _:
            return str(value)


def _options(
    y_label: str, *, stacked: bool = False, horizontal: bool = False
) -> dict[str, object]:
    value_axis = {"title": {"display": bool(y_label), "text": y_label}}
    options: dict[str, object] = {
        "responsive": True,
        "maintainAspectRatio": False,
        "animation": False,
        "plugins": {"legend": {"position": "bottom"}},
        "scales": {
            "x": {"stacked": stacked},
            "y": {"stacked": stacked, **value_axis},
        },
    }
    if horizontal:
        options["indexAxis"] = "y"
        options["scales"] = {
            "x": {"stacked": stacked, **value_axis},
            "y": {"stacked": stacked},
        }
    return options


def line_chart(labels: Sequence[str], series: Series, *, y_label: str) -> ChartSpec:
    datasets = [
        {
            "label": name,
            "data": list(values),
            "borderColor": series_token(index),
            "backgroundColor": series_token(index),
            "borderWidth": 2,
            "pointRadius": 0,
            "pointHitRadius": 8,
            "spanGaps": True,
        }
        for index, (name, values) in enumerate(series)
    ]
    config = {
        "type": "line",
        "data": {"labels": list(labels), "datasets": datasets},
        "options": _options(y_label),
    }
    return ChartSpec(config=config)


def bar_chart(
    labels: Sequence[str],
    series: Series,
    *,
    y_label: str,
    stacked: bool = False,
    horizontal: bool = False,
    colors: Sequence[str] | None = None,
) -> ChartSpec:
    """Grouped/stacked bars; ``colors`` gives per-bar tokens for one series."""
    datasets = [
        {
            "label": name,
            "data": list(values),
            "backgroundColor": (
                list(colors)
                if colors is not None and len(series) == 1
                else series_token(index)
            ),
            "borderRadius": 4,
            "maxBarThickness": 40,
        }
        for index, (name, values) in enumerate(series)
    ]
    config = {
        "type": "bar",
        "data": {"labels": list(labels), "datasets": datasets},
        "options": _options(y_label, stacked=stacked, horizontal=horizontal),
    }
    return ChartSpec(config=config)


def stacked_area(labels: Sequence[str], series: Series, *, y_label: str) -> ChartSpec:
    datasets = [
        {
            "label": name,
            "data": list(values),
            "borderColor": series_token(index),
            "backgroundColor": series_token(index),
            "borderWidth": 1,
            "pointRadius": 0,
            "pointHitRadius": 8,
            "fill": True,
        }
        for index, (name, values) in enumerate(series)
    ]
    config = {
        "type": "line",
        "data": {"labels": list(labels), "datasets": datasets},
        "options": _options(y_label, stacked=True),
    }
    return ChartSpec(config=config)


def histogram(
    bin_labels: Sequence[str],
    counts: Sequence[int],
    *,
    y_label: str,
    label: str = "count",
) -> ChartSpec:
    return bar_chart(bin_labels, [(label, list(map(float, counts)))], y_label=y_label)


def reliability_scatter(
    points: Sequence[tuple[float, float]], *, counts: Sequence[int]
) -> ChartSpec:
    """PoP reliability: observed frequency vs forecast probability + diagonal."""
    config = {
        "type": "scatter",
        "data": {
            "datasets": [
                {
                    "label": "reliability",
                    "data": [
                        {"x": x, "y": y, "n": n}
                        for (x, y), n in zip(points, counts, strict=True)
                    ],
                    "backgroundColor": "series-1",
                    "pointRadius": 5,
                },
                {
                    "label": "perfect calibration",
                    "type": "line",
                    "data": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
                    "borderColor": "muted",
                    "borderDash": [4, 4],
                    "borderWidth": 1,
                    "pointRadius": 0,
                },
            ]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "animation": False,
            "plugins": {"legend": {"position": "bottom"}},
            "scales": {
                "x": {
                    "min": 0,
                    "max": 1,
                    "title": {"display": True, "text": "forecast probability"},
                },
                "y": {
                    "min": 0,
                    "max": 1,
                    "title": {"display": True, "text": "observed frequency"},
                },
            },
        },
    }
    return ChartSpec(config=config)


def sequential_class(value: float | None, lo: float, hi: float) -> str:
    """Five-bucket sequential heat class for a value in [lo, hi]."""
    if value is None or not math.isfinite(value):
        return "heat-none"
    if hi <= lo:
        return "heat-2"
    position = (value - lo) / (hi - lo)
    bucket = min(4, max(0, int(position * 5)))
    return f"heat-{bucket}"


def diverging_class(value: float | None, limit: float) -> str:
    """Five-bucket diverging class for a value in [-limit, +limit]."""
    if value is None or not math.isfinite(value):
        return "heat-none"
    if limit <= 0:
        return "div-0"
    position = max(-1.0, min(1.0, value / limit))
    bucket = round(position * 2)
    sign = "n" if bucket < 0 else "p" if bucket > 0 else ""
    return f"div-{sign}{abs(bucket)}" if bucket else "div-0"
