"""Read the omni-weather-forecast-apis archive into canonical long frames.

One row per (source, fetched_at, valid_time). Leads are ALWAYS recomputed from
the ISO ``fetched_at`` text: the archive's ``horizon_hours``, ``fetched_at_unix``
and ``run_cycle`` columns are NULL in damaged databases and are never trusted.
Missing tables, empty sources, and NULL fetch timestamps are tolerated.
"""

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

from grounded_weather_forecast.config import ForecastsConfig
from grounded_weather_forecast.contracts import SourceKind
from grounded_weather_forecast.dataset.station import sqlite_uri
from grounded_weather_forecast.timeutil import parse_iso_utc_expr

HOURLY_COLUMN_MAP: Mapping[str, str] = {
    "temperature": "temp_c",
    "humidity": "humidity_pct",
    "dew_point": "dew_point_c",
    "wind_speed": "wind_speed_ms",
    "wind_gust": "wind_gust_ms",
    "pressure_sea": "pressure_sea_hpa",
    "precipitation": "precip_mm",
    "precipitation_probability": "pop",
}

DAILY_COLUMN_MAP: Mapping[str, str] = {
    "temperature_max": "temp_max_c",
    "temperature_min": "temp_min_c",
    "precipitation_probability_max": "pop",
    "precipitation_sum": "precip_sum_mm",
}

_SECONDS_PER_HOUR = 3600.0
LOCATION_TOLERANCE = 1e-4


@dataclass(frozen=True, slots=True)
class ForecastArchive:
    """One transactionally consistent view of every forecast product."""

    hourly: pl.DataFrame
    daily: pl.DataFrame
    minutely: pl.DataFrame
    completions: pl.DataFrame


def source_slug(provider: str, model: str) -> str:
    """One stable slug per forecast stream."""
    if model in (provider, "best_match", ""):
        return provider
    return f"{provider}_{model}"


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    return row is not None


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }


def _fetch_points(
    connection: sqlite3.Connection,
    forecasts: ForecastsConfig,
    points_table: str,
    point_columns: list[str],
) -> list[tuple[object, ...]]:
    for required in (
        points_table,
        "source_forecasts",
        "provider_results",
        "forecast_runs",
    ):
        if not _table_exists(connection, required):
            return []
    if not {"latitude", "longitude"} <= _table_columns(connection, "forecast_runs"):
        return []
    columns = ", ".join(f"p.{column}" for column in point_columns)
    query = (
        f"SELECT pr.run_id, sf.provider, sf.model, pr.fetched_at, {columns} "
        f"FROM {points_table} AS p "
        "JOIN source_forecasts AS sf ON sf.id = p.source_forecast_id "
        "JOIN provider_results AS pr ON pr.id = sf.provider_result_id "
        "JOIN forecast_runs AS fr ON fr.id = pr.run_id "
        "WHERE pr.status = 'success' "
        "AND ABS(fr.latitude - ?) <= ? "
        "AND ABS(fr.longitude - ?) <= ?"
    )
    return connection.execute(
        query,
        (
            forecasts.latitude,
            LOCATION_TOLERANCE,
            forecasts.longitude,
            LOCATION_TOLERANCE,
        ),
    ).fetchall()


def _base_schema(point_columns: Mapping[str, pl.DataType]) -> dict[str, pl.DataType]:
    return {
        "run_id": pl.Int64(),
        "provider": pl.String(),
        "model": pl.String(),
        "fetched_at_raw": pl.String(),
        **dict(point_columns),
    }


def _with_provenance(frame: pl.DataFrame, forecasts: ForecastsConfig) -> pl.DataFrame:
    slugged = frame.with_columns(
        pl.struct("provider", "model")
        .map_elements(
            lambda row: source_slug(row["provider"], row["model"]),
            return_dtype=pl.String,
        )
        .alias("source"),
        parse_iso_utc_expr(pl.col("fetched_at_raw")).alias("fetched_at"),
        pl.lit(SourceKind.LIVE.value).alias("source_kind"),
    ).drop_nulls("fetched_at")
    if forecasts.sources:
        slugged = slugged.filter(pl.col("source").is_in(list(forecasts.sources)))
    return slugged


def _open(forecasts: ForecastsConfig) -> sqlite3.Connection:
    if not forecasts.db_path.exists():
        msg = f"cannot open forecast archive {forecasts.db_path}: file not found"
        raise OSError(msg)
    return sqlite3.connect(
        sqlite_uri(forecasts.db_path, immutable=forecasts.immutable), uri=True
    )


def _read_hourly_long(
    connection: sqlite3.Connection, forecasts: ForecastsConfig
) -> pl.DataFrame:
    point_columns = ["timestamp_unix", *HOURLY_COLUMN_MAP.keys()]
    rows = _fetch_points(connection, forecasts, "hourly_points", point_columns)
    schema = _base_schema(
        {"timestamp_unix": pl.Int64()}
        | {column: pl.Float64() for column in HOURLY_COLUMN_MAP}
    )
    raw = pl.DataFrame(rows, schema=schema, orient="row")
    return (
        _with_provenance(raw.drop_nulls("timestamp_unix"), forecasts)
        .with_columns(
            pl.from_epoch("timestamp_unix", time_unit="s")
            .cast(pl.Datetime("us"))
            .dt.replace_time_zone("UTC")
            .alias("valid_time"),
        )
        .with_columns(
            (
                (pl.col("valid_time") - pl.col("fetched_at")).dt.total_seconds()
                / _SECONDS_PER_HOUR
            ).alias("lead_hours")
        )
        .rename(dict(HOURLY_COLUMN_MAP))
        .sort("source", "fetched_at", "valid_time")
        .unique(subset=["source", "fetched_at", "valid_time"], keep="last")
        .sort("source", "fetched_at", "valid_time")
        .select(
            "run_id",
            "source",
            "source_kind",
            "fetched_at",
            "valid_time",
            "lead_hours",
            *HOURLY_COLUMN_MAP.values(),
        )
    )


def read_hourly_long(forecasts: ForecastsConfig) -> pl.DataFrame:
    """Hourly forecast points as a canonical long frame."""
    connection = _open(forecasts)
    try:
        return _read_hourly_long(connection, forecasts)
    finally:
        connection.close()


def _read_daily_long(
    connection: sqlite3.Connection, forecasts: ForecastsConfig
) -> pl.DataFrame:
    point_columns = ["forecast_date", *DAILY_COLUMN_MAP.keys()]
    rows = _fetch_points(connection, forecasts, "daily_points", point_columns)
    schema = _base_schema(
        {"forecast_date": pl.String()}
        | {column: pl.Float64() for column in DAILY_COLUMN_MAP}
    )
    raw = pl.DataFrame(rows, schema=schema, orient="row")
    return (
        _with_provenance(raw.drop_nulls("forecast_date"), forecasts)
        .with_columns(pl.col("forecast_date").str.to_date("%Y-%m-%d", strict=False))
        .drop_nulls("forecast_date")
        .rename(dict(DAILY_COLUMN_MAP))
        .sort("source", "fetched_at", "forecast_date")
        .unique(subset=["source", "fetched_at", "forecast_date"], keep="last")
        .sort("source", "fetched_at", "forecast_date")
        .select(
            "run_id",
            "source",
            "source_kind",
            "fetched_at",
            "forecast_date",
            *DAILY_COLUMN_MAP.values(),
        )
    )


def read_daily_long(forecasts: ForecastsConfig) -> pl.DataFrame:
    """Daily forecast points as a canonical long frame keyed by forecast date."""
    connection = _open(forecasts)
    try:
        return _read_daily_long(connection, forecasts)
    finally:
        connection.close()


def _read_minutely_long(
    connection: sqlite3.Connection, forecasts: ForecastsConfig
) -> pl.DataFrame:
    point_columns = [
        "timestamp_unix",
        "precipitation_intensity",
        "precipitation_probability",
    ]
    rows = _fetch_points(connection, forecasts, "minutely_points", point_columns)
    schema = _base_schema(
        {
            "timestamp_unix": pl.Int64(),
            "precipitation_intensity": pl.Float64(),
            "precipitation_probability": pl.Float64(),
        }
    )
    raw = pl.DataFrame(rows, schema=schema, orient="row")
    return (
        _with_provenance(raw.drop_nulls("timestamp_unix"), forecasts)
        .with_columns(
            pl.from_epoch("timestamp_unix", time_unit="s")
            .cast(pl.Datetime("us"))
            .dt.replace_time_zone("UTC")
            .alias("valid_time")
        )
        .rename(
            {
                "precipitation_intensity": "precip_intensity_mmh",
                "precipitation_probability": "pop",
            }
        )
        .sort("source", "fetched_at", "valid_time")
        .unique(subset=["source", "fetched_at", "valid_time"], keep="last")
        .sort("source", "fetched_at", "valid_time")
        .select(
            "run_id",
            "source",
            "source_kind",
            "fetched_at",
            "valid_time",
            "precip_intensity_mmh",
            "pop",
        )
    )


def read_minutely_long(forecasts: ForecastsConfig) -> pl.DataFrame:
    """Minutely precipitation points (the only minutely content providers emit)."""
    connection = _open(forecasts)
    try:
        return _read_minutely_long(connection, forecasts)
    finally:
        connection.close()


def _read_run_completions(
    connection: sqlite3.Connection, forecasts: ForecastsConfig
) -> pl.DataFrame:
    if not _table_exists(connection, "forecast_runs"):
        return pl.DataFrame(schema={"completed_at": pl.Datetime("us", "UTC")})
    if not {"completed_at", "latitude", "longitude"} <= _table_columns(
        connection, "forecast_runs"
    ):
        return pl.DataFrame(schema={"completed_at": pl.Datetime("us", "UTC")})
    rows = connection.execute(
        "SELECT completed_at FROM forecast_runs "
        "WHERE completed_at IS NOT NULL "
        "AND ABS(latitude - ?) <= ? AND ABS(longitude - ?) <= ?",
        (
            forecasts.latitude,
            LOCATION_TOLERANCE,
            forecasts.longitude,
            LOCATION_TOLERANCE,
        ),
    ).fetchall()
    raw = pl.DataFrame(rows, schema={"completed_at_raw": pl.String()}, orient="row")
    return (
        raw.with_columns(
            parse_iso_utc_expr(pl.col("completed_at_raw")).alias("completed_at")
        )
        .drop_nulls("completed_at")
        .select("completed_at")
        .sort("completed_at")
    )


def read_run_completions(forecasts: ForecastsConfig) -> pl.DataFrame:
    """``forecast_runs.completed_at`` instants (snapshot anchors)."""
    connection = _open(forecasts)
    try:
        return _read_run_completions(connection, forecasts)
    finally:
        connection.close()


def read_latest_archive_location(
    forecasts: ForecastsConfig,
) -> tuple[float, float] | None:
    """Coordinates recorded by the newest forecast run, without filtering."""
    try:
        connection = _open(forecasts)
        try:
            if not _table_exists(connection, "forecast_runs"):
                return None
            if not {"latitude", "longitude"} <= _table_columns(
                connection, "forecast_runs"
            ):
                return None
            row = connection.execute(
                "SELECT latitude, longitude FROM forecast_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        finally:
            connection.close()
    except (  # noqa: B013 - project style requires tuple clauses
        sqlite3.Error,
    ) as exc:
        msg = f"cannot read forecast archive location {forecasts.db_path}: {exc}"
        raise OSError(msg) from exc
    if (
        row is None
        or len(row) != 2
        or not all(isinstance(value, (int, float)) for value in row)
    ):
        return None
    return float(row[0]), float(row[1])


def read_forecast_archive(forecasts: ForecastsConfig) -> ForecastArchive:
    """Read all products inside one SQLite snapshot transaction."""
    connection = _open(forecasts)
    try:
        connection.execute("BEGIN")
        return ForecastArchive(
            hourly=_read_hourly_long(connection, forecasts),
            daily=_read_daily_long(connection, forecasts),
            minutely=_read_minutely_long(connection, forecasts),
            completions=_read_run_completions(connection, forecasts),
        )
    finally:
        connection.close()
