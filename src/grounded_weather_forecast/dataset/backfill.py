"""Open-Meteo Previous Runs backfill: the cold-start escape hatch.

The live archive is the binding constraint on this whole project — every method
needs months of stored forecast *vintages*, and ground truth alone is not
enough. Open-Meteo's Previous Runs API serves archived forecasts at fixed
day-offsets (``temperature_2m_previous_day1`` is what the model predicted for
this valid hour one day earlier), which is exactly a synthetic forecast archive
with lead as a controlled variable.

Three properties are load-bearing:

- Rows are tagged ``synthetic`` and never silently pooled with ``live`` ones
  (:class:`MixedProvenanceError` enforces this downstream). Commercial
  providers cannot be backfilled, so a leaderboard built on synthetic data
  says nothing about them.
- Only offsets 1..7 are requested. The unsuffixed field is the *latest* run for
  a past hour, whose effective lead is near zero; treating it as a forecast
  would fill the short-lead buckets with what is essentially an analysis and
  make anchoring look miraculous. Excluding it means synthetic leads are exact
  24-hour multiples.
- Leads are therefore quantized to whole days, so synthetic matrices populate
  only the buckets at and beyond 24 h. Reports surface bucket coverage per
  source kind rather than interpolating across the gap.

The HTTP fetcher is injected, so tests never touch the network.
"""

import json
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Any

import polars as pl

from grounded_weather_forecast.config import Config
from grounded_weather_forecast.contracts import SourceKind
from grounded_weather_forecast.dataset.providers import HOURLY_COLUMN_MAP

type Fetcher = Callable[[str], Mapping[str, Any]]

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# Open-Meteo hourly variable -> canonical variable. Only the fields Previous
# Runs exposes with per-day offsets are requested.
BACKFILL_VARIABLES: Mapping[str, str] = {
    "temperature_2m": "temp_c",
    "relative_humidity_2m": "humidity_pct",
    "dew_point_2m": "dew_point_c",
    "wind_speed_10m": "wind_speed_ms",
    "wind_gusts_10m": "wind_gust_ms",
    "pressure_msl": "pressure_sea_hpa",
    "precipitation": "precip_mm",
}
MIN_PREVIOUS_DAY = 1
MAX_PREVIOUS_DAYS = 7
_SECONDS_PER_HOUR = 3600.0
_HTTP_TIMEOUT_SECONDS = 60


def http_fetcher(url: str) -> Mapping[str, Any]:  # pragma: no cover - network
    with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT_SECONDS) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        msg = f"unexpected Previous Runs payload type: {type(payload).__name__}"
        raise ValueError(msg)
    return payload


def _offset_field(variable: str, day: int) -> str:
    return f"{variable}_previous_day{day}"


def _offsets(previous_days: int) -> range:
    return range(MIN_PREVIOUS_DAY, previous_days + 1)


def build_url(
    config: Config,
    model: str,
    start: date,
    end: date,
    previous_days: int = MAX_PREVIOUS_DAYS,
) -> str:
    """Previous Runs request for one model over one date range."""
    fields = [
        _offset_field(variable, day)
        for variable in BACKFILL_VARIABLES
        for day in _offsets(previous_days)
    ]
    query = urllib.parse.urlencode(
        {
            "latitude": config.station.latitude,
            "longitude": config.station.longitude,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(fields),
            "models": model,
            "timezone": "UTC",
            "wind_speed_unit": "ms",
        }
    )
    return f"{PREVIOUS_RUNS_URL}?{query}"


class BackfillError(RuntimeError):
    """The Previous Runs response was missing or malformed."""


def parse_previous_runs(
    payload: Mapping[str, Any], model: str, previous_days: int = MAX_PREVIOUS_DAYS
) -> pl.DataFrame:
    """Payload -> canonical long frame, one row per (day-offset, valid hour).

    Each day-offset becomes a synthetic issue time: the forecast for this valid
    hour as it stood ``day`` days earlier, at the same clock time.
    """
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        msg = "Previous Runs payload has no hourly.time block"
        raise BackfillError(msg)
    times = [datetime.fromisoformat(t) for t in hourly["time"]]
    frames: list[pl.DataFrame] = []
    for day in _offsets(previous_days):
        columns: dict[str, list[Any]] = {}
        for source_field, canonical in BACKFILL_VARIABLES.items():
            values = hourly.get(_offset_field(source_field, day))
            if isinstance(values, list) and len(values) == len(times):
                columns[canonical] = values
        if not columns:
            continue
        offset = timedelta(days=day)
        frames.append(
            pl.DataFrame(
                {
                    "valid_time": times,
                    "fetched_at": [t - offset for t in times],
                    **columns,
                },
                schema_overrides={
                    "valid_time": pl.Datetime("us"),
                    "fetched_at": pl.Datetime("us"),
                }
                | dict.fromkeys(columns, pl.Float64),
            )
        )
    if not frames:
        msg = "Previous Runs payload contained no usable day offsets"
        raise BackfillError(msg)
    combined = pl.concat(frames, how="diagonal")
    missing = [
        canonical
        for canonical in HOURLY_COLUMN_MAP.values()
        if canonical not in combined.columns
    ]
    return (
        combined.with_columns(
            pl.col("valid_time").dt.replace_time_zone("UTC"),
            pl.col("fetched_at").dt.replace_time_zone("UTC"),
            pl.lit(None, dtype=pl.Int64).alias("run_id"),
            pl.lit(f"open_meteo_{model}").alias("source"),
            pl.lit(SourceKind.SYNTHETIC.value).alias("source_kind"),
            *(pl.lit(None, dtype=pl.Float64).alias(name) for name in missing),
        )
        .with_columns(
            (
                (pl.col("valid_time") - pl.col("fetched_at")).dt.total_seconds()
                / _SECONDS_PER_HOUR
            ).alias("lead_hours")
        )
        .drop_nulls("valid_time")
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


def _chunks(start: date, end: date, days: int) -> list[tuple[date, date]]:
    if days <= 0:
        msg = f"chunk_days must be a positive integer, got {days}"
        raise BackfillError(msg)
    spans: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=days - 1), end)
        spans.append((cursor, stop))
        cursor = stop + timedelta(days=1)
    return spans


def backfill_long(
    config: Config,
    end: date,
    models: Sequence[str] | None = None,
    fetcher: Fetcher = http_fetcher,
    chunk_days: int = 90,
    *,
    start: date | None = None,
) -> pl.DataFrame:
    """Fetch models over an inclusive valid-date range as one long frame."""
    start = start or config.backfill.start_date
    if start is None:
        msg = "set [backfill.open_meteo].start_date to backfill"
        raise BackfillError(msg)
    selected = tuple(models) if models else config.backfill.models
    if not selected:
        msg = "set [backfill.open_meteo].models to backfill"
        raise BackfillError(msg)
    if end < start:
        msg = f"backfill end {end} precedes start {start}"
        raise BackfillError(msg)
    frames: list[pl.DataFrame] = []
    for model in selected:
        for span_start, span_end in _chunks(start, end, chunk_days):
            url = build_url(config, model, span_start, span_end)
            frames.append(parse_previous_runs(fetcher(url), model))
    return pl.concat(frames).sort("source", "fetched_at", "valid_time")
