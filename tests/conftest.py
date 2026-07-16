import sqlite3
from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from grounded_weather_forecast.config import load_config
from grounded_weather_forecast.leads import hourly_bucket
from grounded_weather_forecast.timeutil import utc

STATION_DB_COLUMNS = (
    "outTemp",
    "outHumi",
    "windir",
    "avgwind",
    "gustspeed",
    "eventrain",
    "AbsPress",
)


def make_station_db(path, rows):
    """Create an aw2sqlite-shaped observations DB.

    ``rows``: iterable of (ts_text, {column: value}).
    """
    connection = sqlite3.connect(path)
    try:
        columns_sql = ", ".join(f'"{c}" REAL' for c in STATION_DB_COLUMNS)
        connection.execute(
            "CREATE TABLE observations (ts TIMESTAMP DEFAULT "
            f"(STRFTIME('%Y-%m-%d %H:%M:%f', 'now')), {columns_sql})"
        )
        connection.execute(
            "CREATE UNIQUE INDEX idx_observations_ts ON observations(ts)"
        )
        for ts, values in rows:
            cols = ["ts", *values.keys()]
            placeholders = ", ".join("?" for _ in cols)
            connection.execute(
                f"INSERT INTO observations ({', '.join(cols)}) VALUES ({placeholders})",
                [ts, *values.values()],
            )
        connection.commit()
    finally:
        connection.close()
    return path


def write_config(
    tmp_path, station_db="station.db", forecasts_db="fx.sqlite", **overrides
):
    """Write a config.toml for tests; returns the loaded Config."""
    extra = overrides.get("extra_toml", "")
    sources = overrides.get("sources", [])
    sources_toml = ", ".join(f'"{s}"' for s in sources)
    text = f"""
[station]
db_path = "{tmp_path / station_db}"
timezone = "America/Los_Angeles"
latitude = 34.2768
longitude = -117.1692
elevation_m = 1400.0
immutable = true

[forecasts]
db_path = "{tmp_path / forecasts_db}"
sources = [{sources_toml}]

[dataset]
dir = "{tmp_path / "data"}"
min_hour_coverage = {overrides.get("min_hour_coverage", 0.8)}
min_day_coverage = {overrides.get("min_day_coverage", 0.8)}

[reports]
dir = "{tmp_path / "reports"}"

[artifacts]
dir = "{tmp_path / "artifacts"}"
{extra}
"""
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


@pytest.fixture
def la_config(tmp_path):
    return write_config(tmp_path)


def minute_series(start, count, step_seconds=60):
    return [start + timedelta(seconds=step_seconds * i) for i in range(count)]


def canonical_minute_frame(ts, **channels):
    """Build a truth-minute-shaped frame (canonical variables, metric units)."""
    n = len(ts)
    defaults = {
        "temp_c": [None] * n,
        "humidity_pct": [None] * n,
        "dew_point_c": [None] * n,
        "wind_speed_ms": [None] * n,
        "wind_gust_ms": [None] * n,
        "pressure_sea_hpa": [None] * n,
        "rain_counter_mm": [None] * n,
    }
    defaults |= channels
    return pl.DataFrame(
        {"ts": ts, **defaults},
        schema={
            "ts": pl.Datetime("us", "UTC"),
            **dict.fromkeys(defaults, pl.Float64),
        },
    )


FORECAST_DB_SCHEMA = """
CREATE TABLE forecast_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    latitude REAL NOT NULL, longitude REAL NOT NULL,
    granularity TEXT NOT NULL DEFAULT '["hourly", "daily"]',
    language TEXT NOT NULL DEFAULT 'en',
    completed_at TEXT NOT NULL,
    total_latency_ms REAL NOT NULL DEFAULT 0,
    total_results INTEGER NOT NULL DEFAULT 0,
    succeeded INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE provider_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    provider TEXT NOT NULL,
    status TEXT NOT NULL,
    fetched_at TEXT,
    fetched_at_unix INTEGER,
    run_cycle TEXT,
    latency_ms REAL NOT NULL DEFAULT 0
);
CREATE TABLE source_forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_result_id INTEGER NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL
);
CREATE TABLE hourly_points (
    source_forecast_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL DEFAULT '',
    timestamp_unix INTEGER NOT NULL,
    horizon_hours REAL,
    temperature REAL, humidity REAL, dew_point REAL,
    wind_speed REAL, wind_gust REAL, pressure_sea REAL,
    precipitation REAL, precipitation_probability REAL
);
CREATE TABLE daily_points (
    source_forecast_id INTEGER NOT NULL,
    forecast_date TEXT NOT NULL,
    temperature_max REAL, temperature_min REAL,
    precipitation_sum REAL, precipitation_probability_max REAL
);
CREATE TABLE minutely_points (
    source_forecast_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL DEFAULT '',
    timestamp_unix INTEGER NOT NULL,
    precipitation_intensity REAL, precipitation_probability REAL
);
"""

HOURLY_FIELDS = (
    "temperature",
    "humidity",
    "dew_point",
    "wind_speed",
    "wind_gust",
    "pressure_sea",
    "precipitation",
    "precipitation_probability",
)


def make_forecast_db(path, runs):
    """Create an omni-weather-forecast-apis-shaped archive.

    ``runs``: list of dicts with keys ``completed_at`` (ISO str) and
    ``results``: list of dicts with ``provider``, optional ``model`` (defaults
    to provider), ``status`` ('success'), ``fetched_at`` (ISO str or None —
    NULL mimics damage), and optional ``hourly``/``daily``/``minutely`` lists.
    Hourly: (valid_dt, {field: value}); daily: (date_str, {field: value});
    minutely: (valid_dt, intensity, probability).
    ``horizon_hours``/``fetched_at_unix``/``run_cycle`` are always NULL, as in
    the damaged sample archive.
    """
    connection = sqlite3.connect(path)
    try:
        connection.executescript(FORECAST_DB_SCHEMA)
        for run in runs:
            run_cursor = connection.execute(
                "INSERT INTO forecast_runs (latitude, longitude, completed_at)"
                " VALUES (?, ?, ?)",
                (
                    run.get("latitude", 34.2768),
                    run.get("longitude", -117.1692),
                    run["completed_at"],
                ),
            )
            run_id = run_cursor.lastrowid
            for result in run.get("results", []):
                pr_cursor = connection.execute(
                    "INSERT INTO provider_results (run_id, provider, status,"
                    " fetched_at) VALUES (?, ?, ?, ?)",
                    (
                        run_id,
                        result["provider"],
                        result.get("status", "success"),
                        result.get("fetched_at"),
                    ),
                )
                sf_cursor = connection.execute(
                    "INSERT INTO source_forecasts (provider_result_id, provider,"
                    " model) VALUES (?, ?, ?)",
                    (
                        pr_cursor.lastrowid,
                        result["provider"],
                        result.get("model", result["provider"]),
                    ),
                )
                sf_id = sf_cursor.lastrowid
                for valid_dt, fields in result.get("hourly", []):
                    columns = ["source_forecast_id", "timestamp_unix"]
                    values = [sf_id, int(valid_dt.timestamp())]
                    for field, value in fields.items():
                        columns.append(field)
                        values.append(value)
                    connection.execute(
                        f"INSERT INTO hourly_points ({', '.join(columns)})"
                        f" VALUES ({', '.join('?' for _ in values)})",
                        values,
                    )
                for date_str, fields in result.get("daily", []):
                    columns = ["source_forecast_id", "forecast_date"]
                    values = [sf_id, date_str]
                    for field, value in fields.items():
                        columns.append(field)
                        values.append(value)
                    connection.execute(
                        f"INSERT INTO daily_points ({', '.join(columns)})"
                        f" VALUES ({', '.join('?' for _ in values)})",
                        values,
                    )
                for valid_dt, intensity, probability in result.get("minutely", []):
                    connection.execute(
                        "INSERT INTO minutely_points (source_forecast_id,"
                        " timestamp_unix, precipitation_intensity,"
                        " precipitation_probability) VALUES (?, ?, ?, ?)",
                        (sf_id, int(valid_dt.timestamp()), intensity, probability),
                    )
        connection.commit()
    finally:
        connection.close()
    return path


def synthetic_hourly_matrix(
    days=30,
    sources=("alpha", "beta"),
    biases=None,
    noise_sd=0.5,
    seed=0,
    snapshots_per_day=2,
    max_lead=48,
    beta_max_lead=None,
    source_kind="live",
):
    """Deterministic sinusoidal-weather hourly matrix with known provider biases.

    Truth is exact (noise lives only in the forecasts), so directional tests
    can assert real wins. ``beta_max_lead`` truncates the second source's
    horizon to exercise sleeping-source raggedness.
    """
    rng = np.random.default_rng(seed)
    biases = biases or {}
    if not isinstance(noise_sd, dict):
        noise_sd = dict.fromkeys(sources, noise_sd)
    start = utc(2026, 1, 1)
    rows = []
    for day in range(days):
        for snap in range(snapshots_per_day):
            issue = start + timedelta(days=day, hours=snap * 24 / snapshots_per_day)
            for lead in range(1, max_lead + 1):
                valid = issue + timedelta(hours=lead)
                doy = valid.timetuple().tm_yday
                truth = (
                    10.0
                    + 8.0 * float(np.sin(2 * np.pi * (valid.hour - 15) / 24))
                    + 5.0 * float(np.sin(2 * np.pi * doy / 365))
                )
                row = {
                    "issue_time": issue,
                    "valid_time": valid,
                    "lead_hours": float(lead),
                    "source_kind": source_kind,
                    "valid_hour_local": valid.hour,
                    "valid_month": valid.month,
                    "obs__temp_c": 10.0
                    + 8.0 * float(np.sin(2 * np.pi * (issue.hour - 15) / 24))
                    + 5.0 * float(np.sin(2 * np.pi * doy / 365)),
                    "t__temp_c__inst": truth,
                    "t__temp_c__mean": truth,
                }
                for i, source in enumerate(sources):
                    limit = beta_max_lead if (i == 1 and beta_max_lead) else max_lead
                    if lead <= limit:
                        row[f"fx__{source}__temp_c"] = (
                            truth
                            + biases.get(source, 0.0)
                            + float(rng.normal(0.0, noise_sd.get(source, 0.5)))
                        )
                        row[f"age__{source}"] = 0.5
                    else:
                        row[f"fx__{source}__temp_c"] = None
                        row[f"age__{source}"] = None
                rows.append(row)
    frame = pl.DataFrame(rows).with_columns(
        pl.col("issue_time").dt.replace_time_zone(None).dt.replace_time_zone("UTC"),
        pl.col("valid_time").dt.replace_time_zone(None).dt.replace_time_zone("UTC"),
        pl.col("valid_hour_local").cast(pl.Int8),
        pl.col("valid_month").cast(pl.Int8),
    )
    buckets = pl.Series(
        "lead_bucket", [hourly_bucket(x) for x in frame["lead_hours"].to_list()]
    )
    return frame.with_columns(buckets)


__all__ = [
    "HOURLY_FIELDS",
    "canonical_minute_frame",
    "make_forecast_db",
    "make_station_db",
    "minute_series",
    "synthetic_hourly_matrix",
    "utc",
    "write_config",
]
