"""Read the aw2sqlite observations database into a canonical minute frame.

Output columns: ``ts`` (UTC microsecond datetime) plus one metric-unit column
per configured canonical channel (``temp`` in °C, ``wind_speed`` in m/s,
``rain_counter`` in mm, ``pressure_station`` in hPa, ...). Missing station
columns are tolerated (all-null channel); a missing table yields an empty
frame — the sample databases are damaged and the reader must not crash.
"""

import sqlite3
from pathlib import Path

import polars as pl

from omni_forecast.config import StationConfig
from omni_forecast.timeutil import parse_station_ts_expr
from omni_forecast.units import convert_expr


def sqlite_uri(path: Path, *, immutable: bool) -> str:
    """Read-only SQLite URI; immutable only for static snapshot files."""
    mode = "immutable=1" if immutable else "mode=ro"
    return f"file:{path}?{mode}"


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row[1]) for row in rows}


def _empty_frame(channels: list[str]) -> pl.DataFrame:
    schema: dict[str, pl.DataType] = {"ts": pl.Datetime("us", "UTC")}
    schema |= {channel: pl.Float64() for channel in channels}
    return pl.DataFrame(schema=schema)


def read_observations(station: StationConfig) -> pl.DataFrame:
    """Load, unit-convert, sort, and dedupe the station's minute samples."""
    channels = sorted(set(station.columns.values()))
    if len(channels) != len(station.columns.values()):
        msg = "station column mapping contains duplicate canonical channels"
        raise ValueError(msg)
    if not station.db_path.exists():
        msg = f"cannot open station database {station.db_path}: file not found"
        raise OSError(msg)
    uri = sqlite_uri(station.db_path, immutable=station.immutable)
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        msg = f"cannot open station database {station.db_path}: {exc}"
        raise OSError(msg) from exc
    try:
        available = _table_columns(connection, "observations")
        if not available:
            return _empty_frame(channels)
        selected = {
            db_column: channel
            for db_column, channel in station.columns.items()
            if db_column in available
        }
        columns = ", ".join(["ts", *(f'"{c}"' for c in selected)])
        cursor = connection.execute(f"SELECT {columns} FROM observations ORDER BY ts")
        rows = cursor.fetchall()
    finally:
        connection.close()
    raw = pl.DataFrame(
        rows,
        schema={"ts": pl.String} | dict.fromkeys(selected.values(), pl.Float64),
        orient="row",
    )
    missing = [channel for channel in channels if channel not in selected.values()]
    return (
        raw.with_columns(
            parse_station_ts_expr(pl.col("ts")).alias("ts"),
            *(
                convert_expr(pl.col(channel), station.units.get(channel, "degC")).alias(
                    channel
                )
                for channel in selected.values()
            ),
        )
        .with_columns(
            *(pl.lit(None, dtype=pl.Float64).alias(channel) for channel in missing)
        )
        .drop_nulls("ts")
        .sort("ts")
        .unique(subset="ts", keep="first", maintain_order=True)
        .select("ts", *channels)
    )
