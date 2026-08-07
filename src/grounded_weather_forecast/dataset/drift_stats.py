"""Change-point statistics for station-vs-neighbor drift attribution.

The homogenization literature frames "did my station drift?" as relative
testing on difference series: a candidate is judged only against the
spatial context of its neighbors, never against forecasts. The primary
statistic is SNHT (Alexandersson 1986) — most sensitive to breaks near the
series end, which is exactly the monitoring case — with Pettitt as the
robust rank-based alternative and the Craddock cusum as the operator's
corroborating visual. Attribution follows Menne & Williams' pairwise
logic in miniature: the break belongs to the series common to all breaking
pairs, so a station drift shows coincident breaks in most
station-minus-neighbor series while neighbor-minus-neighbor pairs stay
quiet, and a regime event breaks the neighbor network too.

Daily difference series are autocorrelated, which inflates end-of-series
test statistics; callers guard with persistence (consecutive evaluations
above threshold) rather than prewhitening — drifts persist, regimes revert.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from grounded_weather_forecast.contracts import FloatArray

MIN_SERIES_DAYS = 7
_SCALE_EPSILON = 1e-9
# 95% critical values for max-T SNHT on n iid daily values, Monte Carlo
# (20k replicates against THIS implementation; the n=30 value of 8.05
# matches the published Khaliq & Ouarda 2007 scale). The multi-year
# T > 100 convention does not apply at monitoring windows.
_SNHT_CRITICAL_95 = (
    (7, 5.5),
    (10, 6.29),
    (14, 6.93),
    (21, 7.51),
    (30, 8.05),
    (45, 8.46),
    (60, 8.72),
)
_MIN_ATTRIBUTION_SERIES = 3
_BREAK_SHARE = 2.0 / 3.0
_QUIET_SHARE = 1.0 / 3.0
_COINCIDENCE_DAYS = 2


@dataclass(frozen=True, slots=True)
class BreakTest:
    """One change-point test: the statistic and where it peaked."""

    statistic: float
    break_index: int
    p_value: float | None = None


def snht(values: FloatArray | Sequence[float]) -> BreakTest:
    """Max-T SNHT: T(a) = a*mean(z[:a])^2 + (n-a)*mean(z[a:])^2."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.shape[0]
    if n < 4:
        return BreakTest(0.0, 0)
    scale = float(np.std(x))
    if scale < _SCALE_EPSILON:
        return BreakTest(0.0, 0)
    z = (x - float(np.mean(x))) / scale
    best_statistic, best_index = 0.0, 0
    for split in range(1, n):
        left = float(np.mean(z[:split]))
        right = float(np.mean(z[split:]))
        statistic = split * left * left + (n - split) * right * right
        if statistic > best_statistic:
            best_statistic, best_index = statistic, split
    return BreakTest(best_statistic, best_index)


def snht_critical_value(n: int) -> float:
    """95% critical value for ``snht`` at n daily values, interpolated."""
    sizes = np.asarray([size for size, _ in _SNHT_CRITICAL_95], dtype=np.float64)
    criticals = np.asarray([value for _, value in _SNHT_CRITICAL_95])
    return float(np.interp(float(n), sizes, criticals))


def pettitt(values: FloatArray | Sequence[float]) -> BreakTest:
    """Pettitt's rank-based test with its asymptotic p-value."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.shape[0]
    if n < 4:
        return BreakTest(0.0, 0, p_value=None)
    signs = np.sign(x[:, np.newaxis] - x[np.newaxis, :])
    u = np.asarray(
        [float(signs[: split + 1, split + 1 :].sum()) for split in range(n - 1)]
    )
    k = float(np.max(np.abs(u)))
    break_index = int(np.argmax(np.abs(u))) + 1
    p_value = min(1.0, 2.0 * math.exp(-6.0 * k * k / (n**3 + n**2)))
    return BreakTest(k, break_index, p_value=p_value)


def craddock_cusum(values: FloatArray | Sequence[float]) -> list[float]:
    """Cumulative sum of anomalies; a step change is a sustained slope change."""
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    if not bool(finite.any()):
        return [0.0] * x.shape[0]
    anomalies = np.where(finite, x - float(np.mean(x[finite])), 0.0)
    return [float(value) for value in np.cumsum(anomalies)]


@dataclass(frozen=True, slots=True)
class BreakAttribution:
    """Mini pairwise-PHA verdict over the neighbor network."""

    verdict: str  # "station_drift" | "regime" | "inconclusive"
    station_series: int
    station_breaks: int
    coincident_breaks: int
    neighbor_pairs: int
    neighbor_breaks: int
    break_date: date | None


def _series_break(frame: pl.DataFrame) -> tuple[bool, date | None]:
    """(broke, break date) for one (date, difference) daily series."""
    ordered = frame.drop_nulls("difference").sort("date")
    if ordered.height < MIN_SERIES_DAYS:
        return False, None
    values = ordered["difference"].to_numpy().astype(np.float64)
    test = snht(values)
    if test.statistic <= snht_critical_value(values.shape[0]):
        return False, None
    return True, ordered["date"][min(test.break_index, ordered.height - 1)]


def attribute_break(
    station_minus_neighbor: Mapping[str, pl.DataFrame],
    neighbor_minus_neighbor: Mapping[str, pl.DataFrame],
) -> BreakAttribution:
    """Attribute a detected break: the station, the regime, or unclear.

    Station drift: >= 2/3 of the station-minus-neighbor series break at a
    coincident date (+-2 days) while the neighbor-vs-neighbor pairs stay
    quiet. Neighbor pairs breaking alongside — or scattered break dates —
    read as a regional regime event instead: NWP is wrong everywhere and
    the station is fine.
    """
    station_results = [
        _series_break(frame) for frame in station_minus_neighbor.values()
    ]
    evaluable = [
        result
        for result, frame in zip(
            station_results, station_minus_neighbor.values(), strict=True
        )
        if frame.drop_nulls("difference").height >= MIN_SERIES_DAYS
    ]
    breaks = [break_date for broke, break_date in evaluable if broke]
    neighbor_results = [
        _series_break(frame) for frame in neighbor_minus_neighbor.values()
    ]
    neighbor_evaluable = [
        result
        for result, frame in zip(
            neighbor_results, neighbor_minus_neighbor.values(), strict=True
        )
        if frame.drop_nulls("difference").height >= MIN_SERIES_DAYS
    ]
    neighbor_breaks = sum(1 for broke, _ in neighbor_evaluable if broke)
    coincident = 0
    consensus_date: date | None = None
    if breaks:
        ordered_dates = sorted(break_dates for break_dates in breaks if break_dates)
        if ordered_dates:
            consensus_date = ordered_dates[len(ordered_dates) // 2]
            window = timedelta(days=_COINCIDENCE_DAYS)
            coincident = sum(
                1
                for break_date in ordered_dates
                if abs(break_date - consensus_date) <= window
            )
    counts = {
        "station_series": len(evaluable),
        "station_breaks": len(breaks),
        "coincident_breaks": coincident,
        "neighbor_pairs": len(neighbor_evaluable),
        "neighbor_breaks": neighbor_breaks,
    }
    if len(evaluable) < _MIN_ATTRIBUTION_SERIES:
        return BreakAttribution(
            verdict="inconclusive", break_date=consensus_date, **counts
        )
    breaking_share = coincident / len(evaluable)
    neighbor_share = (
        neighbor_breaks / len(neighbor_evaluable) if neighbor_evaluable else 0.0
    )
    if breaking_share >= _BREAK_SHARE and neighbor_share <= _QUIET_SHARE:
        verdict = "station_drift"
    elif breaking_share >= _BREAK_SHARE or neighbor_share > _QUIET_SHARE:
        verdict = "regime"
    else:
        verdict = "inconclusive"
    return BreakAttribution(verdict=verdict, break_date=consensus_date, **counts)
