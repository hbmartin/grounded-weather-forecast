"""Neighbor-station truth cross-checks: NWS METAR by default, Synoptic opt-in.

Everything upstream optimizes toward the station, so a drifting sensor makes
the whole system calibrate to a broken thermometer — and a failing radiation
shield "can look plausible for months" (Limitations §7). Two defenses:

- **Slow bias drift**: the 30-day rolling median of station-minus-consensus,
  where the consensus is the median of 3+ lapse-adjusted neighbors inside an
  elevation band. An alert past ~1 °C is a sensor conversation, not a
  weather event.
- **Decorrelation** (the CrowdQC+ m4 idea, single-station form): a rolling
  ~72 h correlation of hourly station temperature against the consensus.
  Correlation collapsing below ~0.9 flags a failing sensor even while its
  values stay inside every plausibility bound.

Sources (future-work #19: Synoptic's free tier went .edu-only, leaving
drift detection blind): the keyless default discovers nearby observation
stations through the NWS ``points`` API and pulls each station's METAR
history — fully generic, no station ids in config, elevations delivered in
meters. Configuring ``[truth_qc].synoptic_token`` still routes through the
Synoptic timeseries API for deployments that hold a paid/edu token. The
NWS observations endpoint caps a request at 500 records, so sub-hourly
reporters may cover fewer than the requested days — the drift check only
needs 7 daily comparisons, and the cap comfortably clears it.

The fetcher is injected so tests never touch the network, mirroring the
backfill modules.
"""

import json as _json
import math
import os
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

from grounded_weather_forecast.config import Config
from grounded_weather_forecast.dataset.backfill import Fetcher, http_fetcher

SYNOPTIC_URL = "https://api.synopticdata.com/v2/stations/timeseries"
NWS_POINTS_URL = "https://api.weather.gov/points"
_NWS_HEADERS = {
    # api.weather.gov rejects requests without a User-Agent outright.
    "User-Agent": "grounded-weather-forecast (github.com/hbmartin/grounded-weather-forecast)",
    "Accept": "application/geo+json",
}
_NWS_OBSERVATION_LIMIT = 500
_MAX_NWS_STATIONS = 8
_HTTP_TIMEOUT_SECONDS = 60
_EARTH_RADIUS_KM = 6371.0
_KM_PER_MILE = 1.609344
_FEET_PER_METER = 3.28084
_MIN_NEIGHBORS = 3
_DRIFT_ALERT_C = 1.0
_CORRELATION_FLOOR = 0.9
_CORRELATION_WINDOW_HOURS = 72
_MIN_CORRELATION_SAMPLES = _CORRELATION_WINDOW_HOURS // 2
_DRIFT_WINDOW_DAYS = 30
_MIN_DRIFT_DAYS = 7
_DEFAULT_HISTORY_HOURS = 30 * 24


class NeighborError(RuntimeError):
    """The neighbor-source response was missing, malformed, or unauthorized."""


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


def nws_http_fetcher(url: str) -> Mapping[str, Any]:  # pragma: no cover - network
    """GET with the User-Agent api.weather.gov requires."""
    request = urllib.request.Request(url, headers=dict(_NWS_HEADERS))  # noqa: S310
    with urllib.request.urlopen(  # noqa: S310
        request, timeout=_HTTP_TIMEOUT_SECONDS
    ) as response:
        payload = _json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        msg = f"unexpected NWS payload type: {type(payload).__name__}"
        raise NeighborError(msg)
    return payload


def build_points_url(config: Config) -> str:
    return (
        f"{NWS_POINTS_URL}/{config.station.latitude:.4f},{config.station.longitude:.4f}"
    )


def _haversine_km(
    latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float
) -> float:
    phi_a, phi_b = math.radians(latitude_a), math.radians(latitude_b)
    d_phi = phi_b - phi_a
    d_lambda = math.radians(longitude_b - longitude_a)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def stations_url_from_points(payload: Mapping[str, Any]) -> str:
    """The gridpoint's observation-stations URL from a points response."""
    properties = payload.get("properties")
    if isinstance(properties, dict):
        stations = properties.get("observationStations")
        if isinstance(stations, str) and stations.startswith("https://"):
            return stations
    msg = "NWS points payload has no observationStations URL"
    raise NeighborError(msg)


def parse_station_directory(
    payload: Mapping[str, Any],
    *,
    site_latitude: float,
    site_longitude: float,
    site_elevation_m: float,
    elevation_band_m: float,
    radius_km: float,
) -> list[tuple[str, float]]:
    """(station id, elevation m) of usable neighbors, proximity-ordered.

    The NWS station list arrives ordered by distance from the gridpoint;
    filtering preserves that order, so the cap keeps the nearest survivors.
    """
    features = payload.get("features")
    if not isinstance(features, list):
        msg = "NWS stations payload has no features list"
        raise NeighborError(msg)
    stations: list[tuple[str, float]] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties")
        geometry = feature.get("geometry")
        if not isinstance(properties, dict) or not isinstance(geometry, dict):
            continue
        station_id = properties.get("stationIdentifier")
        elevation = properties.get("elevation")
        coordinates = geometry.get("coordinates")
        if (
            not isinstance(station_id, str)
            or not isinstance(elevation, dict)
            or not isinstance(coordinates, list)
            or len(coordinates) < 2
        ):
            continue
        elevation_m = elevation.get("value")
        if not isinstance(elevation_m, (int, float)):
            continue
        if abs(float(elevation_m) - site_elevation_m) > elevation_band_m:
            continue
        longitude, latitude = float(coordinates[0]), float(coordinates[1])
        if (
            _haversine_km(site_latitude, site_longitude, latitude, longitude)
            > radius_km
        ):
            continue
        stations.append((station_id, float(elevation_m)))
        if len(stations) >= _MAX_NWS_STATIONS:
            break
    return stations


def build_station_observations_url(station_id: str, hours: int, now: datetime) -> str:
    start = (now - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    query = urllib.parse.urlencode(
        {"start": start, "end": end, "limit": _NWS_OBSERVATION_LIMIT}
    )
    quoted = urllib.parse.quote(station_id, safe="")
    return f"https://api.weather.gov/stations/{quoted}/observations?{query}"


def parse_station_observations(
    payload: Mapping[str, Any],
    station_id: str,
    elevation_m: float,
    site_elevation_m: float,
    lapse_k_per_km: float,
) -> pl.DataFrame:
    """One station's METAR history -> (stid, ts, temp_c) lapse-adjusted."""
    features = payload.get("features")
    if not isinstance(features, list):
        return _empty_neighbors()
    adjustment = lapse_k_per_km * (elevation_m - site_elevation_m) / 1000.0
    timestamps: list[datetime] = []
    temperatures: list[float] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            continue
        timestamp = properties.get("timestamp")
        temperature = properties.get("temperature")
        if not isinstance(timestamp, str) or not isinstance(temperature, dict):
            continue
        value = temperature.get("value")
        if not isinstance(value, (int, float)):
            continue
        try:
            moment = datetime.fromisoformat(timestamp)
        except ValueError:
            continue
        moment = moment if moment.tzinfo else moment.replace(tzinfo=UTC)
        timestamps.append(moment.astimezone(UTC))
        temperatures.append(float(value) + adjustment)
    if not timestamps:
        return _empty_neighbors()
    return pl.DataFrame(
        {
            "stid": [station_id] * len(timestamps),
            "ts": timestamps,
            "temp_c": temperatures,
        },
        schema_overrides={"ts": pl.Datetime("us", "UTC")},
    )


def _empty_neighbors() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "stid": pl.String(),
            "ts": pl.Datetime("us", "UTC"),
            "temp_c": pl.Float64(),
        }
    )


def fetch_nws_neighbors(
    config: Config,
    fetcher: Fetcher,
    hours: int,
    now: datetime,
) -> pl.DataFrame:
    """Discover nearby NWS stations and pull their lapse-adjusted METARs.

    No station inside the elevation band and radius is a young/remote-site
    condition, not an error — the consensus simply stays empty and every
    verdict reads "not evaluable".
    """
    points = dict(fetcher(build_points_url(config)))
    directory = dict(fetcher(stations_url_from_points(points)))
    stations = parse_station_directory(
        directory,
        site_latitude=config.station.latitude,
        site_longitude=config.station.longitude,
        site_elevation_m=config.station.elevation_m,
        elevation_band_m=config.truth_qc.elevation_band_m,
        radius_km=config.truth_qc.radius_km,
    )
    frames = [
        observations
        for station_id, elevation_m in stations
        if not (
            observations := parse_station_observations(
                dict(fetcher(build_station_observations_url(station_id, hours, now))),
                station_id,
                elevation_m,
                config.station.elevation_m,
                config.truth_qc.lapse_k_per_km,
            )
        ).is_empty()
    ]
    if not frames:
        return _empty_neighbors()
    return pl.concat(frames).sort("stid", "ts")


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


def cross_check(
    truth_hourly: pl.DataFrame,
    consensus: pl.DataFrame,
    *,
    as_of: datetime | None = None,
) -> NeighborChecks:
    """Station-vs-consensus drift and decorrelation verdicts."""
    if as_of is not None and as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)
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
            min_samples=_MIN_CORRELATION_SAMPLES,
        ).alias("correlation")
    ).select("valid_hour", "correlation")
    reference_time = as_of or truth_hourly.select(pl.col("valid_hour").max()).item()
    if not isinstance(reference_time, datetime):  # pragma: no cover - joined proves it
        msg = "truth valid_hour must contain datetimes"
        raise NeighborError(msg)
    correlation_window = joined.filter(
        pl.col("valid_hour").is_between(
            reference_time - timedelta(hours=_CORRELATION_WINDOW_HOURS),
            reference_time,
            closed="right",
        )
    )
    finite_pairs = correlation_window.filter(
        pl.col("t__temp_c__inst").is_not_null()
        & pl.col("consensus_c").is_not_null()
        & pl.col("t__temp_c__inst").is_finite()
        & pl.col("consensus_c").is_finite()
    )
    finite_pair_count = finite_pairs.height
    if finite_pair_count < _MIN_CORRELATION_SAMPLES:
        latest_correlation = float("nan")
        unavailable_correlation_reason = (
            f"need at least {_MIN_CORRELATION_SAMPLES} finite paired hourly "
            f"comparisons; got {finite_pair_count}"
        )
    elif (
        finite_pairs["t__temp_c__inst"].n_unique() <= 1
        or finite_pairs["consensus_c"].n_unique() <= 1
    ):
        latest_correlation = float("nan")
        unavailable_correlation_reason = (
            "rolling correlation is undefined because the station or "
            f"neighbor consensus has zero variance across {finite_pair_count} "
            "finite paired hourly comparisons"
        )
    else:
        raw_correlation = finite_pairs.select(
            pl.corr("t__temp_c__inst", "consensus_c")
        ).item()
        latest_correlation = (
            float(raw_correlation)
            if isinstance(raw_correlation, (int, float))
            else float("nan")
        )
        unavailable_correlation_reason = (
            "rolling correlation is undefined despite "
            f"{finite_pair_count} finite, nonconstant paired hourly comparisons"
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
            else unavailable_correlation_reason
        ),
    )


def fetch_neighbor_checks(
    config: Config,
    truth_hourly: pl.DataFrame,
    fetcher: Fetcher | None = None,
    hours: int = _DEFAULT_HISTORY_HOURS,
    now: datetime | None = None,
) -> NeighborChecks:
    """One cron pass: fetch, adjust, consense, verdict.

    A configured Synoptic token selects the legacy timeseries API; the
    keyless default discovers and pulls NWS METAR neighbors.
    """
    moment = now or datetime.now(tz=UTC)
    token = resolve_token(config.truth_qc.synoptic_token)
    if token:
        synoptic_fetcher = fetcher or http_fetcher
        payload = dict(synoptic_fetcher(build_neighbors_url(config, hours)))
        neighbors = parse_neighbors(
            payload,
            config.station.elevation_m,
            config.truth_qc.elevation_band_m,
            config.truth_qc.lapse_k_per_km,
        )
    else:
        neighbors = fetch_nws_neighbors(
            config, fetcher or nws_http_fetcher, hours, moment
        )
    checks = cross_check(
        truth_hourly,
        neighbor_consensus(neighbors),
        as_of=moment,
    )
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
