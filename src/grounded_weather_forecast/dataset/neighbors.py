"""Neighbor-station truth cross-checks via the Synoptic Data API.

Everything upstream optimizes toward the station, so a drifting sensor makes
the whole system calibrate to a broken thermometer — and a failing radiation
shield "can look plausible for months" (Limitations §7). Two defenses, both
needing only a free-signup Synoptic token and a daily cron:

- **Slow bias drift**: the 30-day rolling median of station-minus-consensus,
  where the consensus is the median of 3+ lapse-adjusted neighbors inside an
  elevation band. An alert past ~1 °C is a sensor conversation, not a
  weather event.
- **Decorrelation** (the CrowdQC+ m4 idea, single-station form): a rolling
  ~72 h correlation of hourly station temperature against the consensus.
  Correlation collapsing below ~0.9 flags a failing sensor even while its
  values stay inside every plausibility bound.

The fetcher is injected so tests never touch the network, mirroring the
backfill modules.
"""

import os
import urllib.parse
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from grounded_weather_forecast.config import Config
from grounded_weather_forecast.dataset.backfill import Fetcher, http_fetcher

SYNOPTIC_URL = "https://api.synopticdata.com/v2/stations/timeseries"
_KM_PER_MILE = 1.609344
_FEET_PER_METER = 3.28084
_MIN_NEIGHBORS = 3
_DRIFT_ALERT_C = 1.0
_CORRELATION_FLOOR = 0.9
_CORRELATION_WINDOW_HOURS = 72
_DRIFT_WINDOW_DAYS = 30
_MIN_DRIFT_DAYS = 7
_DEFAULT_HISTORY_HOURS = 30 * 24


class NeighborError(RuntimeError):
    """The Synoptic response was missing, malformed, or unauthorized."""


@dataclass(frozen=True, slots=True)
class NeighborChecks:
    """The two cross-check series plus their current verdicts."""

    daily_drift: pl.DataFrame  # date, station_minus_consensus_c
    rolling_correlation: pl.DataFrame  # valid_hour, correlation
    comparison: pl.DataFrame  # station, consensus, wind, and difference by hour
    drift_alert: bool | None
    correlation_alert: bool | None
    n_neighbors: int
    overlap_hours: int
    drift_reason: str
    correlation_reason: str


def resolve_token(raw: str) -> str:
    """A leading ``$`` reads the token from the environment, never the file."""
    if raw.startswith("$"):
        return os.environ.get(raw.lstrip("$"), "")
    return raw


def build_neighbors_url(config: Config, hours: int) -> str:
    token = resolve_token(config.truth_qc.synoptic_token)
    if not token:
        msg = "set [truth_qc].synoptic_token (or the referenced env var)"
        raise NeighborError(msg)
    radius_miles = config.truth_qc.radius_km / _KM_PER_MILE
    query = urllib.parse.urlencode(
        {
            "token": token,
            "radius": (
                f"{config.station.latitude},{config.station.longitude},"
                f"{radius_miles:.1f}"
            ),
            "vars": "air_temp",
            "recent": hours * 60,
            "units": "metric",
            "obtimezone": "UTC",
        }
    )
    return f"{SYNOPTIC_URL}?{query}"


def parse_neighbors(
    payload: dict[str, object],
    site_elevation_m: float,
    elevation_band_m: float,
    lapse_k_per_km: float,
) -> pl.DataFrame:
    """Payload -> long (stid, ts, temp_c) lapse-adjusted to the site elevation."""
    stations = payload.get("STATION")
    if not isinstance(stations, list):
        msg = "Synoptic payload has no STATION list"
        raise NeighborError(msg)
    frames: list[pl.DataFrame] = []
    for station in stations:
        if not isinstance(station, dict):
            continue
        elevation_ft = station.get("ELEVATION")
        match elevation_ft:
            case int() | float() | str() if str(elevation_ft).strip():
                try:
                    elevation_m = float(elevation_ft) / _FEET_PER_METER
                except ValueError:
                    continue
            case _:
                continue
        if abs(elevation_m - site_elevation_m) > elevation_band_m:
            continue
        observations = station.get("OBSERVATIONS")
        if not isinstance(observations, dict):
            continue
        times = observations.get("date_time")
        temps = observations.get("air_temp_set_1")
        if not isinstance(times, list) or not isinstance(temps, list):
            continue
        adjustment = lapse_k_per_km * (elevation_m - site_elevation_m) / 1000.0
        rows = [
            (str(station.get("STID", "?")), t, float(v) + adjustment)
            for t, v in zip(times, temps, strict=False)
            if isinstance(t, str) and isinstance(v, (int, float))
        ]
        if rows:
            frames.append(
                pl.DataFrame(
                    {
                        "stid": [r[0] for r in rows],
                        "ts": [datetime.fromisoformat(r[1]) for r in rows],
                        "temp_c": [r[2] for r in rows],
                    },
                    schema_overrides={"ts": pl.Datetime("us", "UTC")},
                )
            )
    if not frames:
        return pl.DataFrame(
            schema={
                "stid": pl.String(),
                "ts": pl.Datetime("us", "UTC"),
                "temp_c": pl.Float64(),
            }
        )
    return pl.concat(frames).sort("stid", "ts")


def neighbor_consensus(neighbors: pl.DataFrame) -> pl.DataFrame:
    """Hourly median across neighbors; hours with < 3 reporters are dropped."""
    if neighbors.is_empty():
        return pl.DataFrame(
            schema={
                "valid_hour": pl.Datetime("us", "UTC"),
                "consensus_c": pl.Float64(),
            }
        )
    return (
        neighbors.with_columns(pl.col("ts").dt.truncate("1h").alias("valid_hour"))
        .group_by("valid_hour", "stid")
        .agg(pl.col("temp_c").mean())
        .group_by("valid_hour")
        .agg(
            pl.col("temp_c").median().alias("consensus_c"),
            pl.col("stid").n_unique().alias("n"),
        )
        .filter(pl.col("n") >= _MIN_NEIGHBORS)
        .drop("n")
        .sort("valid_hour")
    )


def cross_check(truth_hourly: pl.DataFrame, consensus: pl.DataFrame) -> NeighborChecks:
    """Station-vs-consensus drift and decorrelation verdicts."""
    truth_columns = ["valid_hour", "t__temp_c__inst"]
    if "t__wind_speed_ms__inst" in truth_hourly.columns:
        truth_columns.append("t__wind_speed_ms__inst")
    joined = (
        truth_hourly.select(truth_columns)
        .drop_nulls(["valid_hour", "t__temp_c__inst"])
        .join(consensus, on="valid_hour", how="inner")
        .with_columns(
            (pl.col("t__temp_c__inst") - pl.col("consensus_c")).alias("difference")
        )
        .sort("valid_hour")
    )
    if joined.is_empty():
        empty_daily = pl.DataFrame(
            schema={"date": pl.Date(), "station_minus_consensus_c": pl.Float64()}
        )
        empty_corr = pl.DataFrame(
            schema={"valid_hour": pl.Datetime("us", "UTC"), "correlation": pl.Float64()}
        )
        reason = "no overlapping station and neighbor-consensus hours"
        return NeighborChecks(
            daily_drift=empty_daily,
            rolling_correlation=empty_corr,
            comparison=joined,
            drift_alert=None,
            correlation_alert=None,
            n_neighbors=0,
            overlap_hours=0,
            drift_reason=reason,
            correlation_reason=reason,
        )
    daily = (
        joined.group_by(pl.col("valid_hour").dt.date().alias("date"))
        .agg(pl.col("difference").median().alias("station_minus_consensus_c"))
        .sort("date")
    )
    recent = daily.tail(_DRIFT_WINDOW_DAYS)
    recent_median = recent["station_minus_consensus_c"].median()
    drift = (
        float(recent_median)
        if isinstance(recent_median, (int, float))
        else float("nan")
    )
    drift_evaluable = recent.height >= _MIN_DRIFT_DAYS and np.isfinite(drift)
    correlation = joined.with_columns(
        pl.rolling_corr(
            pl.col("t__temp_c__inst"),
            pl.col("consensus_c"),
            window_size=_CORRELATION_WINDOW_HOURS,
            min_samples=_CORRELATION_WINDOW_HOURS // 2,
        ).alias("correlation")
    ).select("valid_hour", "correlation")
    latest = correlation.drop_nulls("correlation").tail(1)
    latest_correlation = (
        float(latest["correlation"][0]) if latest.height else float("nan")
    )
    correlation_evaluable = np.isfinite(latest_correlation)
    return NeighborChecks(
        daily_drift=daily,
        rolling_correlation=correlation,
        comparison=joined,
        drift_alert=abs(drift) > _DRIFT_ALERT_C if drift_evaluable else None,
        correlation_alert=(
            latest_correlation < _CORRELATION_FLOOR if correlation_evaluable else None
        ),
        n_neighbors=0,
        overlap_hours=joined.height,
        drift_reason=(
            f"{recent.height} daily comparisons; recent median bias {drift:+.2f} C"
            if drift_evaluable
            else f"need at least {_MIN_DRIFT_DAYS} daily comparisons; got {recent.height}"
        ),
        correlation_reason=(
            f"latest rolling correlation {latest_correlation:.3f}"
            if correlation_evaluable
            else "need at least 36 overlapping hourly comparisons"
        ),
    )


def fetch_neighbor_checks(
    config: Config,
    truth_hourly: pl.DataFrame,
    fetcher: Fetcher = http_fetcher,
    hours: int = _DEFAULT_HISTORY_HOURS,
    now: datetime | None = None,
) -> NeighborChecks:
    """One cron pass: fetch, adjust, consense, verdict."""
    del now  # reserved for future request shaping; keeps call sites stable
    payload = dict(fetcher(build_neighbors_url(config, hours)))
    neighbors = parse_neighbors(
        payload,
        config.station.elevation_m,
        config.truth_qc.elevation_band_m,
        config.truth_qc.lapse_k_per_km,
    )
    checks = cross_check(truth_hourly, neighbor_consensus(neighbors))
    n_neighbors = int(neighbors["stid"].n_unique()) if not neighbors.is_empty() else 0
    return NeighborChecks(
        daily_drift=checks.daily_drift,
        rolling_correlation=checks.rolling_correlation,
        comparison=checks.comparison,
        drift_alert=checks.drift_alert,
        correlation_alert=checks.correlation_alert,
        n_neighbors=n_neighbors,
        overlap_hours=checks.overlap_hours,
        drift_reason=checks.drift_reason,
        correlation_reason=checks.correlation_reason,
    )
