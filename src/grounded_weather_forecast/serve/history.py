"""Archive of everything this system has itself emitted.

Every served forecast is appended here so it can later be scored against the
truth that arrives afterwards. Backtest skill is an estimate of live skill; the
history is what turns that estimate into a measurement, and it is the only way
to catch a live path that has quietly diverged from the backtested one.
"""

from datetime import date
from pathlib import Path

import polars as pl

from grounded_weather_forecast.serve.schema import Forecast
from grounded_weather_forecast.storage import (
    atomic_write_parquet,
    atomic_write_text,
    locked_path,
)
from grounded_weather_forecast.timeutil import local_day_start_utc

HISTORY_SCHEMA: pl.Schema = pl.Schema(
    {
        "issued_at": pl.Datetime("us", "UTC"),
        "product": pl.String(),
        "variable": pl.String(),
        "valid_time": pl.Datetime("us", "UTC"),
        "valid_date": pl.Date(),
        "lead_hours": pl.Float64(),
        "method_id": pl.String(),
        "y_pred": pl.Float64(),
        "dataset_fingerprint": pl.String(),
        "release_id": pl.String(),
        "selection_reason": pl.String(),
        "quantiles_json": pl.String(),
    }
)

_HOURS_PER_DAY = 24.0


def forecast_to_rows(forecast: Forecast) -> pl.DataFrame:
    """Flatten an emitted document into scoreable rows."""
    rows: list[dict[str, object]] = []
    release_id = ",".join(forecast.release_ids)
    for point in forecast.minutely:
        values = {
            "temp_c": point.temp_c,
            "humidity_pct": point.humidity_pct,
            "dew_point_c": point.dew_point_c,
            "wind_speed_ms": point.wind_speed_ms,
            "precip_intensity_mmh": point.precip_intensity_mmh,
            "pop": point.pop,
        }
        for variable, value in values.items():
            if value is None:
                continue
            rows.append(
                {
                    "issued_at": forecast.issued_at,
                    "product": "minutely",
                    "variable": variable,
                    "valid_time": point.valid_time,
                    "valid_date": None,
                    "lead_hours": point.minutes_ahead / 60.0,
                    "method_id": point.methods.get(variable, "native_or_anchored"),
                    "y_pred": value,
                    "dataset_fingerprint": forecast.dataset_fingerprint,
                    "release_id": release_id,
                    "selection_reason": None,
                    "quantiles_json": _quantiles_json(point.quantiles.get(variable)),
                }
            )
    for point in forecast.hourly:
        for variable, value in point.values.items():
            if value is None:
                continue
            rows.append(
                {
                    "issued_at": forecast.issued_at,
                    "product": "hourly",
                    "variable": variable,
                    "valid_time": point.valid_time,
                    "valid_date": None,
                    "lead_hours": point.lead_hours,
                    "method_id": point.methods.get(variable, "unknown"),
                    "y_pred": value,
                    "dataset_fingerprint": forecast.dataset_fingerprint,
                    "release_id": release_id,
                    "selection_reason": point.selection_reasons.get(variable),
                    "quantiles_json": _quantiles_json(point.quantiles.get(variable)),
                }
            )
    for daily in forecast.daily:
        for variable, value in daily.values.items():
            if value is None:
                continue
            rows.append(
                {
                    "issued_at": forecast.issued_at,
                    "product": "daily",
                    "variable": variable,
                    "valid_time": local_day_start_utc(
                        date.fromisoformat(daily.date_local), forecast.timezone
                    ).isoformat(),
                    "valid_date": daily.date_local,
                    "lead_hours": daily.lead_days * _HOURS_PER_DAY,
                    "method_id": daily.methods.get(variable, "unknown"),
                    "y_pred": value,
                    "dataset_fingerprint": forecast.dataset_fingerprint,
                    "release_id": release_id,
                    "selection_reason": daily.selection_reasons.get(variable),
                    "quantiles_json": _quantiles_json(daily.quantiles.get(variable)),
                }
            )
    if not rows:
        return pl.DataFrame(schema=HISTORY_SCHEMA)
    return (
        pl.DataFrame(
            rows,
            schema_overrides={
                "valid_date": pl.String(),
                "selection_reason": pl.String(),
                "quantiles_json": pl.String(),
            },
        )
        .with_columns(
            pl.col("issued_at").str.to_datetime(time_unit="us", time_zone="UTC"),
            pl.col("valid_time").str.to_datetime(time_unit="us", time_zone="UTC"),
            pl.col("valid_date").str.to_date(strict=False),
        )
        .cast(HISTORY_SCHEMA)
    )


def _quantiles_json(quantiles: dict[str, float] | None) -> str | None:
    if not quantiles:
        return None
    import json  # noqa: PLC0415

    return json.dumps(quantiles, sort_keys=True)


def _archive_document(forecast: Forecast, history_path: Path) -> None:
    directory = history_path.parent / "served_forecasts"
    safe_issue = forecast.issued_at.replace(":", "-").replace("+", "_")
    destination = directory / f"{safe_issue}.json"
    atomic_write_text(forecast.to_json(), destination)


def load_archived_forecast(history_path: Path, issued_at: str) -> Forecast | None:
    """Return the exact document served at an issue time, when archived."""
    safe_issue = issued_at.replace(":", "-").replace("+", "_")
    path = history_path.parent / "served_forecasts" / f"{safe_issue}.json"
    if not path.exists():
        return None
    return Forecast.from_json(path.read_text(encoding="utf-8"))


def append_history(forecast: Forecast, path: Path) -> int:
    """Append an emitted forecast; returns the number of rows added."""
    fresh = forecast_to_rows(forecast)
    if fresh.is_empty():
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with locked_path(path):
        combined = pl.concat([load_history(path), fresh]) if path.exists() else fresh
        atomic_write_parquet(combined, path)
        _archive_document(forecast, path)
    return fresh.height


def load_history(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema=HISTORY_SCHEMA)
    frame = pl.read_parquet(path)
    missing = [
        pl.lit(None, dtype=dtype).alias(column)
        for column, dtype in HISTORY_SCHEMA.items()
        if column not in frame.columns
    ]
    return (
        frame.with_columns(*missing)
        .select(HISTORY_SCHEMA.names())
        .cast(HISTORY_SCHEMA, strict=False)
    )
