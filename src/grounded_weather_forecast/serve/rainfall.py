"""Causal wet-evidence guards and reconciliation of precipitation products."""

import math
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from itertools import pairwise
from zoneinfo import ZoneInfo

import polars as pl

from grounded_weather_forecast.leads import daily_bucket, hourly_bucket
from grounded_weather_forecast.serve.schema import (
    DailyPoint,
    HourlyPoint,
    MinutelyPoint,
)
from grounded_weather_forecast.serve.selection import Selection
from grounded_weather_forecast.timeutil import local_day_start_utc

MIN_WET_DAYS = 5
MAX_NATIVE_GAP = timedelta(minutes=15)
RAIN_VARIABLES = frozenset({"pop", "precip_mm", "precip_sum_mm"})
HOURLY_FALLBACK = "hourly_rainfall_fallback"
MINUTELY_FLOOR = "minutely_rainfall_floor"
HOURLY_SUM = "hourly_rainfall_sum"
ACCUMULATION_FLOOR = "rainfall_accumulation_floor"
PROBABILITY_FLOOR = "hourly_rainfall_probability_floor"
OBSERVED_OCCURRENCE = "observed_daily_occurrence"


@dataclass(frozen=True, slots=True)
class RainfallTimeline:
    """Parsed times and point indices, safe across minute replacements."""

    hour_starts: tuple[datetime, ...]
    minute_times: tuple[datetime, ...]
    minute_indices: tuple[int, ...]
    hour_minutes: tuple[tuple[int, ...], ...]

    @classmethod
    def build(
        cls, hourly: list[HourlyPoint], minutely: list[MinutelyPoint]
    ) -> "RainfallTimeline":
        starts = tuple(datetime.fromisoformat(h.valid_time) for h in hourly)
        minutes = sorted(
            (datetime.fromisoformat(m.valid_time), index)
            for index, m in enumerate(minutely)
        )
        times = tuple(time for time, _ in minutes)
        indices = tuple(index for _, index in minutes)
        children = tuple(
            tuple(
                sorted(
                    indices[
                        bisect_right(times, start) : bisect_right(
                            times, start + timedelta(hours=1)
                        )
                    ]
                )
            )
            for start in starts
        )
        return cls(starts, times, indices, children)

    def minutes_between(self, start: datetime, end: datetime) -> tuple[int, ...]:
        """Original point order in an interval whose minute labels end it."""
        return tuple(
            sorted(
                self.minute_indices[
                    bisect_right(self.minute_times, start) : bisect_right(
                        self.minute_times, end
                    )
                ]
            )
        )

    def hours_between(self, start: datetime, end: datetime) -> tuple[int, ...]:
        return tuple(
            i for i, time in enumerate(self.hour_starts) if start <= time < end
        )


@dataclass(slots=True)
class RainfallEffects:
    """Prediction-local attribution losses; never part of forecast JSON."""

    guarded_release_removed: bool = False
    reconciled_release_removed: bool = False


def rainfall_selections(
    train: pl.DataFrame,
    chosen: list[Selection],
    buckets: list[str | None],
    *,
    daily: bool,
    timezone: str,
) -> list[Selection]:
    """Repeated vintages of one storm cannot qualify a dry-season rain fit."""
    counts: dict[str | None, set[object]] = defaultdict(set)
    if "t__pop" in train.columns:
        wet = train.filter(pl.col("t__pop").is_finite() & (pl.col("t__pop") >= 0.5))
        columns = (
            ("lead_days", "forecast_date") if daily else ("lead_hours", "valid_time")
        )
        for row in wet.select(columns).iter_rows(named=True):
            bucket = (
                daily_bucket(row["lead_days"])
                if daily
                else hourly_bucket(row["lead_hours"])
            )
            day = (
                row["forecast_date"]
                if daily
                else row["valid_time"].astimezone(ZoneInfo(timezone)).date()
            )
            counts[bucket].add(day)
    return [
        Selection(
            "equal_weight",
            reason=f"rainfall fallback: {len(counts[bucket])} wet local days in {bucket}; need {MIN_WET_DAYS}; uncalibrated provider mean; original selection: {selection.reason}",
            degraded=True,
            truth_semantics=selection.truth_semantics,
        )
        if len(counts[bucket]) < MIN_WET_DAYS and not selection.pinned
        else selection
        for selection, bucket in zip(chosen, buckets, strict=True)
    ]


def native_rainfall(
    frame: pl.DataFrame,
) -> dict[datetime, tuple[float | None, float | None]]:
    """Densify each latest native path inside its support, then average sources."""
    if frame.is_empty():
        return {}
    latest = frame.filter(
        pl.col("fetched_at") == pl.col("fetched_at").max().over("source")
    )
    values: dict[datetime, list[list[float]]] = defaultdict(lambda: [[], []])
    for source in latest.partition_by("source"):
        source_name = str(source["source"][0])
        interval_rate = source_name == "open_meteo" or source_name.startswith(
            "open_meteo_"
        )
        rows = (
            source.sort("valid_time")
            .unique("valid_time", keep="last", maintain_order=True)
            .to_dicts()
        )
        for field, name in enumerate(("precip_intensity_mmh", "pop")):
            usable = [
                (r["valid_time"], _finite(r[name], probability=field == 1))
                for r in rows
            ]
            usable = [(t, v) for t, v in usable if v is not None]
            for t, value in usable:
                # The adapter converted a preceding-15-minute sum to mm/hour.
                # Constant disaggregation preserves that sum exactly.
                offsets = range(15) if interval_rate and field == 0 else range(1)
                for offset in offsets:
                    values[t - timedelta(minutes=offset)][field].append(value)
            if interval_rate and field == 0:
                continue
            for (left, a), (right, b) in pairwise(usable):
                gap = right - left
                if gap > MAX_NATIVE_GAP:
                    continue
                t = left + timedelta(minutes=1)
                while t < right:
                    fraction = (t - left) / gap
                    values[t][field].append(a + fraction * (b - a))
                    t += timedelta(minutes=1)
    return {
        t: (
            sum(fields[0]) / len(fields[0]) if fields[0] else None,
            sum(fields[1]) / len(fields[1]) if fields[1] else None,
        )
        for t, fields in values.items()
    }


def _finite(value: float | None, *, probability: bool = False) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return min(1.0, max(0.0, value)) if probability else max(0.0, value)


def _set_rain(
    point: HourlyPoint | DailyPoint,
    variable: str,
    value: float,
    method: str,
    reason: str,
    *,
    effects: RainfallEffects | None = None,
) -> None:
    old = point.values.get(variable)
    if old is not None and math.isclose(old, value, rel_tol=0.0, abs_tol=1e-12):
        return
    previous = point.selection_reasons.get(variable)
    point.values[variable] = value
    point.methods[variable] = method
    point.selection_reasons[variable] = (
        f"{reason}; original selection: {previous}" if previous else reason
    )
    # The original distribution/release did not produce this reconciled value.
    point.quantiles.pop(variable, None)
    point.quantiles_source.pop(variable, None)
    release = point.release_ids.pop(variable, None)
    if effects is not None and release and _finite(old) is not None:
        effects.reconciled_release_removed = True


def _native_value(point: MinutelyPoint, field: str) -> float | None:
    if point.methods.get(field) == HOURLY_FALLBACK:
        return None
    return _finite(getattr(point, field), probability=field == "pop")


def fill_minutely_rain(
    minutely: list[MinutelyPoint],
    hourly: list[HourlyPoint],
    *,
    timeline: RainfallTimeline | None = None,
) -> None:
    """Distribute only the original parent hour's remaining rainfall budget."""
    timeline = timeline or RainfallTimeline.build(hourly, minutely)
    for hour, indices in zip(hourly, timeline.hour_minutes, strict=True):
        native = [
            value
            for index in indices
            if (value := _native_value(minutely[index], "precip_intensity_mmh"))
            is not None
        ]
        amount = _finite(hour.values.get("precip_mm"))
        missing = 60 - len(native)
        # Unshown slots consume their share too; a truncated horizon must not
        # concentrate the whole hour's budget into its displayed minutes.
        rate = (
            max(amount - sum(native) / 60.0, 0.0) * 60.0 / missing
            if amount is not None and missing > 0
            else None
        )
        probability = _finite(hour.values.get("pop"), probability=True)
        for index in indices:
            point = minutely[index]
            for field, value in (("precip_intensity_mmh", rate), ("pop", probability)):
                if _native_value(point, field) is not None or value is None:
                    continue
                point = replace(point, **{field: value})
                point.methods[field] = HOURLY_FALLBACK
                point.selection_reasons[field] = (
                    "remaining hourly accumulation distributed uniformly across missing minute slots; assumed fallback shape"
                    if field == "precip_intensity_mmh"
                    else "native minute probability missing; matching hourly probability"
                )
            minutely[index] = point


def reconcile_hourly_rain(
    hourly: list[HourlyPoint],
    minutely: list[MinutelyPoint],
    *,
    effects: RainfallEffects | None = None,
    timeline: RainfallTimeline | None = None,
) -> None:
    """A parent hour contains at least its forecast minute accumulation."""
    timeline = timeline or RainfallTimeline.build(hourly, minutely)
    for hour, indices in zip(hourly, timeline.hour_minutes, strict=True):
        minutes = [minutely[index] for index in indices]
        amount = sum(
            value / 60.0
            for m in minutes
            if (value := _native_value(m, "precip_intensity_mmh")) is not None
        )
        probability = max(
            (value for m in minutes if (value := _native_value(m, "pop")) is not None),
            default=0.0,
        )
        for variable, floor in (("precip_mm", amount), ("pop", probability)):
            old = _finite(hour.values.get(variable), probability=variable == "pop")
            if floor > 0.0 and (old is None or floor > old + 1e-12):
                _set_rain(
                    hour,
                    variable,
                    floor,
                    MINUTELY_FLOOR,
                    "parent hour reconciled to contained native minute forecasts; probability uses max, without independence assumptions",
                    effects=effects,
                )


def _complete_day(
    hours: list[HourlyPoint],
    starts: tuple[datetime, ...],
    start: datetime,
    end: datetime,
) -> bool:
    expected = {
        start + timedelta(hours=i)
        for i in range(int((end - start).total_seconds() / 3600))
    }
    return set(starts) == expected and all(
        _finite(h.values.get("precip_mm")) is not None for h in hours
    )


def _today_prefix(
    minutely: list[MinutelyPoint],
    first_hour: datetime,
    issue_time: datetime,
    end: datetime,
    timeline: RainfallTimeline,
) -> float:
    return sum(
        value / 60.0
        for index in timeline.minutes_between(issue_time, min(first_hour, end))
        if (value := _native_value(minutely[index], "precip_intensity_mmh")) is not None
    )


def _daily_probability(
    point: DailyPoint,
    hours: list[HourlyPoint],
    *,
    observed: bool,
    minute_pop: float,
    effects: RainfallEffects | None = None,
) -> None:
    probability = (
        1.0
        if observed
        else max(
            minute_pop,
            max(
                (_finite(h.values.get("pop"), probability=True) or 0.0 for h in hours),
                default=0.0,
            ),
        )
    )
    old_pop = _finite(point.values.get("pop"), probability=True)
    if probability > (old_pop or 0.0) + 1e-12:
        _set_rain(
            point,
            "pop",
            probability,
            OBSERVED_OCCURRENCE if observed else PROBABILITY_FLOOR,
            "measurable rain already observed in this local calendar day"
            if observed
            else "daily probability at least the largest contained native minute or hourly probability; intervals are not assumed independent",
            effects=effects,
        )


def reconcile_daily_rain(
    daily: list[DailyPoint],
    hourly: list[HourlyPoint],
    minutely: list[MinutelyPoint],
    *,
    issue_time: datetime,
    timezone: str,
    observed_today_mm: float | None,
    threshold_mm: float,
    effects: RainfallEffects | None = None,
    timeline: RainfallTimeline | None = None,
) -> None:
    """Exact sums for complete local days; explicit lower bounds otherwise."""
    zone = ZoneInfo(timezone)
    today = issue_time.astimezone(zone).date()
    timeline = timeline or RainfallTimeline.build(hourly, minutely)
    for point in daily:
        day = date.fromisoformat(point.date_local)
        start, end = (
            local_day_start_utc(day, timezone),
            local_day_start_utc(day + timedelta(days=1), timezone),
        )
        indices = timeline.hours_between(start, end)
        hours = [hourly[index] for index in indices]
        starts = tuple(timeline.hour_starts[index] for index in indices)
        amounts = [_finite(h.values.get("precip_mm")) for h in hours]
        complete = _complete_day(hours, starts, start, end)
        total = sum(v for v in amounts if v is not None)
        if day == today:
            total += observed_today_mm or 0.0
            total += _today_prefix(
                minutely, min(starts, default=end), issue_time, end, timeline
            )
        old = _finite(point.values.get("precip_sum_mm"))
        if complete or total > (old or 0.0) + 1e-12:
            _set_rain(
                point,
                "precip_sum_mm",
                total,
                HOURLY_SUM if complete else ACCUMULATION_FLOOR,
                "sum of complete local-day hourly path"
                if complete
                else "at least observed rain today plus remaining hourly/minute rain; partial coverage is a lower bound",
                effects=effects,
            )
        _daily_probability(
            point,
            hours,
            minute_pop=max(
                (
                    value
                    for index in timeline.minutes_between(start, end)
                    if (value := _native_value(minutely[index], "pop")) is not None
                ),
                default=0.0,
            ),
            observed=day == today
            and observed_today_mm is not None
            and observed_today_mm >= threshold_mm,
            effects=effects,
        )


def rainfall_status_reason(effects: RainfallEffects) -> str | None:
    """Name only safeguards that actually removed served attribution."""
    if effects.guarded_release_removed:
        return f"rainfall safeguard: fewer than {MIN_WET_DAYS} wet local days in one or more served lead buckets; using uncalibrated rainfall"
    if effects.reconciled_release_removed:
        return "rainfall reconciliation changed served values; original release attribution no longer applies"
    return None
