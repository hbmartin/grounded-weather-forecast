from importlib import resources


def _asset(name: str) -> str:
    package = resources.files("grounded_weather_forecast.dashboard")
    return (package / "assets" / name).read_text(encoding="utf-8")


def test_vendored_chart_js_ships_with_the_package():
    text = _asset("chart.umd.min.js")
    assert "Chart.js v4.4.9" in text
    assert len(text) > 100_000


def test_css_and_js_assets_ship_with_the_package():
    assert "--surface" in _asset("dashboard.css")
    script = _asset("dashboard.js")
    assert "dashboard-data" in script
    assert 'product === "minutely"' in script
    assert "pointKey(productSelect.value, point)" in script
