"""Change-point statistics: detection, criticals, and PHA attribution."""

from datetime import date, timedelta

import numpy as np
import polars as pl

from grounded_weather_forecast.dataset.drift_stats import (
    attribute_break,
    craddock_cusum,
    pettitt,
    snht,
    snht_critical_value,
)


def stepped(n=30, break_at=22, size=-1.5, noise=0.15, seed=3):
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, noise, n)
    values[break_at:] += size
    return values


def daily_frame(values, start=date(2026, 7, 1)):
    return pl.DataFrame(
        {
            "date": [start + timedelta(days=index) for index in range(len(values))],
            "difference": list(values),
        }
    )


class TestSnht:
    def test_detects_a_step_at_the_right_place(self):
        test = snht(stepped())
        assert test.statistic > snht_critical_value(30)
        assert abs(test.break_index - 22) <= 1

    def test_stationary_noise_stays_below_critical(self):
        rng = np.random.default_rng(0)
        test = snht(rng.normal(0.0, 1.0, 30))
        assert test.statistic < snht_critical_value(30)

    def test_constant_series_is_degenerate_zero(self):
        assert snht(np.ones(20)).statistic == 0.0

    def test_short_series_is_zero(self):
        assert snht(np.asarray([1.0, 2.0, 3.0])).statistic == 0.0

    def test_critical_values_interpolate_the_table(self):
        assert snht_critical_value(7) == 5.5
        assert snht_critical_value(30) == 8.05
        assert 5.5 < snht_critical_value(9) < 6.29
        # clamped at the table's ends
        assert snht_critical_value(5) == 5.5
        assert snht_critical_value(90) == 8.72


class TestPettitt:
    def test_small_p_on_a_step(self):
        test = pettitt(stepped())
        assert test.p_value is not None and test.p_value < 0.01
        assert abs(test.break_index - 22) <= 1

    def test_large_p_on_noise(self):
        rng = np.random.default_rng(12)
        test = pettitt(rng.normal(0.0, 1.0, 30))
        assert test.p_value is not None and test.p_value > 0.05


class TestCraddock:
    def test_cusum_kinks_at_the_break(self):
        cusum = craddock_cusum(stepped(noise=0.01))
        # anomalies are positive before the (negative) break, so the cusum
        # rises to its extreme exactly at the step, then falls
        assert cusum.index(max(cusum)) == 21
        assert abs(cusum[-1]) < 1e-6


class TestAttribution:
    def test_coincident_station_breaks_with_quiet_pairs_is_station_drift(self):
        station = {f"N{index}": daily_frame(stepped(seed=index)) for index in range(5)}
        rng = np.random.default_rng(40)
        pairs = {
            f"N{a}~N{b}": daily_frame(rng.normal(0.0, 0.15, 30))
            for a in range(3)
            for b in range(a + 1, 3)
        }
        result = attribute_break(station, pairs)
        assert result.verdict == "station_drift"
        assert result.break_date == date(2026, 7, 1) + timedelta(days=22)

    def test_breaking_neighbor_pairs_read_as_regime(self):
        station = {f"N{index}": daily_frame(stepped(seed=index)) for index in range(5)}
        pairs = {
            f"N{a}~N{b}": daily_frame(stepped(seed=10 + a * 3 + b))
            for a in range(3)
            for b in range(a + 1, 3)
        }
        result = attribute_break(station, pairs)
        assert result.verdict == "regime"

    def test_too_few_series_is_inconclusive(self):
        station = {"N0": daily_frame(stepped()), "N1": daily_frame(stepped(seed=9))}
        result = attribute_break(station, {})
        assert result.verdict == "inconclusive"

    def test_scattered_break_dates_do_not_localize(self):
        rng = np.random.default_rng(21)
        station = {
            "N0": daily_frame(stepped(break_at=5, seed=1)),
            "N1": daily_frame(stepped(break_at=15, seed=2)),
            "N2": daily_frame(stepped(break_at=25, seed=3)),
            "N3": daily_frame(rng.normal(0.0, 0.15, 30)),
            "N4": daily_frame(rng.normal(0.0, 0.15, 30)),
        }
        result = attribute_break(station, {})
        assert result.verdict != "station_drift"
