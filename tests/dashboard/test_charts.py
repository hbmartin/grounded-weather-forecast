import json
from datetime import UTC, datetime

from grounded_weather_forecast.dashboard.charts import (
    bar_chart,
    diverging_class,
    histogram,
    json_safe,
    line_chart,
    reliability_scatter,
    sequential_class,
    stacked_area,
)

ALL_SPECS = (
    line_chart(["a", "b"], [("s", [1.0, None])], y_label="y"),
    bar_chart(["a"], [("s", [float("nan")])], y_label="y", colors=["muted"]),
    bar_chart(["a", "b"], [("s1", [1.0, 2.0]), ("s2", [3.0, 4.0])], y_label="y"),
    stacked_area(["a"], [("s", [1.0])], y_label="y"),
    histogram(["0-1"], [3], y_label="rows"),
    reliability_scatter([(0.1, 0.2)], counts=[5]),
)


def _scale_types(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "type" and isinstance(value, str):
                yield value
            yield from _scale_types(value)
    elif isinstance(node, list):
        for item in node:
            yield from _scale_types(item)


def test_configs_serialize_without_nan():
    for spec in ALL_SPECS:
        json.dumps(json_safe(spec.config), allow_nan=False)


def test_no_time_scale_anywhere():
    for spec in ALL_SPECS:
        assert "time" not in set(_scale_types(dict(spec.config)))


def test_json_safe_conversions():
    assert json_safe(float("inf")) is None
    assert json_safe(float("nan")) is None
    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    assert json_safe(stamp) == "2026-01-01T00:00:00+00:00"
    assert json_safe({"k": (1, float("nan"))}) == {"k": [1, None]}


def test_sequential_class_buckets():
    assert sequential_class(0.0, 0.0, 1.0) == "heat-0"
    assert sequential_class(0.99, 0.0, 1.0) == "heat-4"
    assert sequential_class(None, 0.0, 1.0) == "heat-none"


def test_diverging_class_buckets():
    assert diverging_class(-2.0, 1.0) == "div-n2"
    assert diverging_class(0.0, 1.0) == "div-0"
    assert diverging_class(2.0, 1.0) == "div-p2"
    assert diverging_class(None, 1.0) == "heat-none"


def test_stacked_area_fills_and_stacks():
    spec = stacked_area(["a"], [("s", [1.0])], y_label="y")
    assert spec.config["data"]["datasets"][0]["fill"] is True
    assert spec.config["options"]["scales"]["y"]["stacked"] is True


def test_per_bar_colors_only_for_single_series():
    single = bar_chart(["a"], [("s", [1.0])], y_label="y", colors=["muted"])
    assert single.config["data"]["datasets"][0]["backgroundColor"] == ["muted"]
    grouped = bar_chart(
        ["a"], [("s1", [1.0]), ("s2", [2.0])], y_label="y", colors=["muted"]
    )
    assert grouped.config["data"]["datasets"][0]["backgroundColor"] == "series-1"
