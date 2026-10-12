from datetime import timedelta
from copy import deepcopy

import polars as pl
import pytest
from conftest import canonical_minute_frame, utc, write_config

from grounded_weather_forecast.dataset.truth import observed_rainfall
from grounded_weather_forecast.serve.rainfall import (
    RainfallEffects,
    fill_minutely_rain,
    native_rainfall,
    rainfall_selections,
    rainfall_status_reason,
    reconcile_daily_rain,
    reconcile_hourly_rain,
)
from grounded_weather_forecast.serve.schema import (
    DailyPoint,
    HourlyPoint,
    MinutelyPoint,
)
from grounded_weather_forecast.serve.selection import Selection
from grounded_weather_forecast.timeutil import local_day_start_utc

NOW = utc(2026, 10, 11, 14)


def hour(t, amount=1.0, pop=0.6):
    return HourlyPoint(
        t.isoformat(),
        1.0,
        values={"precip_mm": amount, "pop": pop},
        methods={"precip_mm": "damped", "pop": "damped"},
        quantiles={"precip_mm": {"0.5": 0.0}},
        quantiles_source={"precip_mm": "residual"},
        release_ids={"precip_mm": "old", "temp_c": "valid"},
    )


def minutes(start, count=60, rate=6.0, pop=0.8):
    return [
        MinutelyPoint(
            (start + timedelta(minutes=i)).isoformat(),
            i,
            precip_intensity_mmh=rate,
            pop=pop,
        )
        for i in range(1, count + 1)
    ]


def daily(day, amount=0.0, pop=0.0):
    return DailyPoint(str(day), 1, values={"precip_sum_mm": amount, "pop": pop})


def reconcile(days, hours, mins=(), *, issue=NOW, observed=None):
    reconcile_daily_rain(
        days,
        hours,
        list(mins),
        issue_time=issue,
        timezone="America/Los_Angeles",
        observed_today_mm=observed,
        threshold_mm=0.254,
    )


@pytest.mark.parametrize("is_daily", [False, True])
def test_wet_evidence_counts_days_not_repeated_forecast_vintages(is_daily):
    dates = [NOW + timedelta(days=i) for i in range(4)]
    data = {"t__pop": [1.0] * 40}
    if is_daily:
        data.update(lead_days=[1] * 40, forecast_date=[d.date() for d in dates] * 10)
        bucket = "D1"
    else:
        data.update(lead_hours=[2.0] * 40, valid_time=dates * 10)
        bucket = "1-3h"
    selection = Selection("damped_equal_weight", reason="selected", release_id="old")
    guarded = rainfall_selections(
        pl.DataFrame(data),
        [selection],
        [bucket],
        daily=is_daily,
        timezone="America/Los_Angeles",
    )[0]
    assert guarded.method_id == "equal_weight"
    assert guarded.release_id is None
    assert "4 wet local days" in guarded.reason
    pinned = Selection("damped_equal_weight", reason="pinned", pinned=True)
    assert (
        rainfall_selections(
            pl.DataFrame(data),
            [pinned],
            [bucket],
            daily=is_daily,
            timezone="America/Los_Angeles",
        )[0]
        is pinned
    )


def test_five_wet_days_allow_calibrated_selection():
    frame = pl.DataFrame(
        {
            "t__pop": [1.0] * 5,
            "lead_hours": [2.0] * 5,
            "valid_time": [NOW + timedelta(days=i) for i in range(5)],
        }
    )
    choice = Selection("grounded_equal_weight", reason="selected")
    assert rainfall_selections(
        frame, [choice], ["1-3h"], daily=False, timezone="America/Los_Angeles"
    ) == [choice]


def test_native_paths_interpolate_latest_vintage_without_extrapolation():
    frame = pl.DataFrame(
        {
            "source": ["a"] * 5 + ["b"] * 2,
            "fetched_at": [NOW - timedelta(hours=1)] * 2 + [NOW] * 3 + [NOW] * 2,
            "valid_time": [
                NOW,
                NOW + timedelta(minutes=30),
                NOW,
                NOW + timedelta(minutes=15),
                NOW + timedelta(minutes=40),
                NOW,
                NOW + timedelta(minutes=15),
            ],
            "precip_intensity_mmh": [99.0, 99.0, 0.0, 6.0, 3.0, 2.0, 2.0],
            "pop": [1.0, 1.0, 0.4, 0.8, None, 0.6, 0.6],
        }
    )
    path = native_rainfall(frame)
    assert path[NOW + timedelta(minutes=5)][0] == pytest.approx(2.0)
    assert path[NOW + timedelta(minutes=5)][1] == pytest.approx(17 / 30)
    assert NOW + timedelta(minutes=16) not in path
    assert NOW + timedelta(minutes=30) not in path
    assert NOW - timedelta(minutes=1) not in path
    assert path[NOW + timedelta(minutes=40)] == (3.0, None)
    assert native_rainfall(pl.DataFrame()) == {}


def test_native_invalid_values_are_missing_and_bounds_are_clipped():
    frame = pl.DataFrame(
        {
            "source": ["a", "a"],
            "fetched_at": [NOW, NOW],
            "valid_time": [NOW, NOW + timedelta(minutes=1)],
            "precip_intensity_mmh": [float("nan"), -1.0],
            "pop": [float("inf"), 2.0],
        }
    )
    assert native_rainfall(frame) == {NOW + timedelta(minutes=1): (0.0, 1.0)}


@pytest.mark.parametrize("source", ["open_meteo", "open_meteo_ecmwf_ifs025"])
def test_open_meteo_minute_disaggregation_preserves_each_preceding_interval(source):
    frame = pl.DataFrame(
        {
            "source": [source] * 2,
            "fetched_at": [NOW] * 2,
            "valid_time": [NOW + timedelta(minutes=15), NOW + timedelta(minutes=30)],
            "precip_intensity_mmh": [4.0, 8.0],
            "pop": [0.5, 0.8],
        }
    )
    path = native_rainfall(frame)
    assert sum(
        path[NOW + timedelta(minutes=i)][0] / 60 for i in range(1, 16)
    ) == pytest.approx(1.0)
    assert sum(
        path[NOW + timedelta(minutes=i)][0] / 60 for i in range(16, 31)
    ) == pytest.approx(2.0)
    assert NOW not in path
    assert NOW + timedelta(minutes=31) not in path


def test_hourly_floor_integrates_all_sixty_minutes_and_clears_old_distribution():
    parent = hour(NOW, 0.001, 0.0)
    reconcile_hourly_rain([parent], minutes(NOW))
    assert parent.values["precip_mm"] == pytest.approx(6.0)
    assert parent.values["pop"] == 0.8
    assert parent.methods["precip_mm"] == "minutely_rainfall_floor"
    assert "precip_mm" not in parent.quantiles
    assert "precip_mm" not in parent.quantiles_source
    assert parent.release_ids == {"temp_c": "valid"}
    higher = hour(NOW, 10.0, 0.9)
    reconcile_hourly_rain([higher], minutes(NOW))
    assert higher.values == {"precip_mm": 10.0, "pop": 0.9}


def test_fallback_keeps_native_fields_and_uses_correct_hour_at_boundary():
    points = minutes(NOW, count=61, rate=None, pop=None)
    points[0] = MinutelyPoint(points[0].valid_time, 1, pop=0.9)
    fill_minutely_rain(
        points, [hour(NOW, 2.0, 0.5), hour(NOW + timedelta(hours=1), 4.0)]
    )
    assert points[0].pop == 0.9
    assert points[0].precip_intensity_mmh == 2.0
    assert points[59].precip_intensity_mmh == 2.0
    assert points[60].precip_intensity_mmh == 4.0
    assert points[0].methods["precip_intensity_mmh"] == "hourly_rainfall_fallback"
    fill_minutely_rain(points, [])


@pytest.mark.parametrize(
    ("day", "count"), [(utc(2026, 3, 8).date(), 23), (utc(2026, 11, 1).date(), 25)]
)
def test_complete_daily_sum_uses_dst_calendar_and_max_probability(day, count):
    start = local_day_start_utc(day, "America/Los_Angeles")
    points = [hour(start + timedelta(hours=i), 1.0, 0.6) for i in range(count)]
    parent = daily(day, amount=100.0)
    reconcile([parent], points, issue=start - timedelta(days=1))
    assert parent.values == {"precip_sum_mm": float(count), "pop": 0.6}
    assert parent.methods["precip_sum_mm"] == "hourly_rainfall_sum"


def test_current_day_includes_observed_rain_and_partial_hour_once():
    issue = NOW + timedelta(minutes=30)
    mins = minutes(issue, rate=6.0)
    hours = [hour(NOW + timedelta(hours=1), 2.0)]
    parent = daily(NOW.date())
    reconcile([parent], hours, mins, issue=issue, observed=3.0)
    assert parent.values["precip_sum_mm"] == pytest.approx(8.0)
    assert parent.values["pop"] == 1.0
    assert parent.methods["pop"] == "observed_daily_occurrence"
    # Partial hourly coverage is a lower bound; it cannot reduce a provider total.
    parent = daily(NOW.date(), amount=12.0)
    reconcile([parent], hours, mins, issue=issue, observed=3.0)
    assert parent.values["precip_sum_mm"] == 12.0


def test_missing_hours_do_not_become_a_complete_zero_day():
    tomorrow = (NOW + timedelta(days=1)).date()
    parent = daily(tomorrow, amount=5.0)
    reconcile(
        [parent],
        [hour(local_day_start_utc(tomorrow, "America/Los_Angeles"), None, None)],
    )
    assert parent.values["precip_sum_mm"] == 5.0
    assert parent.values["pop"] == 0.0


def test_observed_rain_is_causal_local_day_sum_not_event_counter(tmp_path):
    config = write_config(tmp_path)
    midnight = utc(2026, 10, 11, 7)
    ts = [
        midnight - timedelta(minutes=1),
        midnight,
        midnight + timedelta(minutes=1),
        midnight + timedelta(minutes=2),
        midnight + timedelta(minutes=3),
        midnight + timedelta(minutes=4),
    ]
    frame = canonical_minute_frame(
        ts, rain_counter_mm=[10.0, 10.2, 10.1, 10.2, 0.1, 100.0]
    )
    result = observed_rainfall(frame, ts[4], config)
    assert result["rain_observed_today_mm"] == pytest.approx(0.3)
    stale = observed_rainfall(
        frame.filter(pl.col("ts") <= ts[4]), ts[4] + timedelta(hours=2), config
    )
    assert stale["rain_observed_today_mm"] == pytest.approx(0.3)
    assert observed_rainfall(frame.head(0), ts[4], config) == {}
    assert observed_rainfall(frame.drop("rain_counter_mm"), ts[4], config) == {}


def test_observed_rain_does_not_attribute_counter_jumps_across_missing_samples(
    tmp_path,
):
    config = write_config(tmp_path)
    frame = canonical_minute_frame(
        [NOW - timedelta(minutes=20), NOW], rain_counter_mm=[0.0, 5.0]
    )
    assert observed_rainfall(frame, NOW, config)["rain_observed_today_mm"] == 0.0


def test_observed_rain_retains_counter_maximum_from_before_midnight(tmp_path):
    config = write_config(tmp_path)
    midnight = utc(2026, 10, 11, 7)
    frame = canonical_minute_frame(
        [midnight - timedelta(minutes=1), midnight, midnight + timedelta(minutes=1)],
        rain_counter_mm=[10.0, 9.8, 10.0],
    )
    assert observed_rainfall(frame, midnight + timedelta(minutes=1), config) == {
        "rain_observed_today_mm": 0.0
    }


@pytest.mark.parametrize(
    ("amount", "rate", "expected", "fallback"),
    [
        (2.0, 4.0, 2.0, 0.0),
        (4.0, 4.0, 4.0, 4.0),
        (2.0, 6.0, 3.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    ],
)
def test_partial_native_path_cannot_inflate_its_own_parent(
    amount, rate, expected, fallback
):
    parent = hour(NOW, amount)
    points = minutes(NOW, count=30, rate=rate) + minutes(
        NOW + timedelta(minutes=30), count=30, rate=None, pop=None
    )
    fill_minutely_rain(points, [parent])
    assert [p.precip_intensity_mmh for p in points[:30]] == [rate] * 30
    assert [p.precip_intensity_mmh for p in points[30:]] == [fallback] * 30
    reconcile_hourly_rain([parent], points)
    assert parent.values["precip_mm"] == pytest.approx(expected)
    assert sum(p.precip_intensity_mmh / 60 for p in points) == pytest.approx(expected)


def test_partial_output_horizon_keeps_unshown_slots_share_of_budget():
    parent = hour(NOW, 3.0)
    points = minutes(NOW, count=10, rate=6.0) + minutes(
        NOW + timedelta(minutes=10), count=10, rate=None, pop=None
    )
    fill_minutely_rain(points, [parent])
    assert points[-1].precip_intensity_mmh == pytest.approx(2.4)
    assert sum(p.precip_intensity_mmh / 60 for p in points) == pytest.approx(1.4)
    reconcile_hourly_rain([parent], points)
    assert parent.values["precip_mm"] == 3.0


def test_missing_parent_amount_does_not_impute_missing_minutes_as_dry():
    parent = hour(NOW, None, None)
    points = minutes(NOW, count=30) + minutes(
        NOW + timedelta(minutes=30), count=30, rate=None, pop=None
    )
    fill_minutely_rain(points, [parent])
    assert points[-1].precip_intensity_mmh is None
    assert points[-1].pop is None
    reconcile_hourly_rain([parent], points)
    assert parent.values["precip_mm"] == 3.0


def test_hourly_derived_minutes_are_not_independent_floor_evidence():
    parent = hour(NOW, 2.0, 0.2)
    points = minutes(NOW, rate=10.0, pop=0.9)
    for point in points:
        point.methods.update(
            precip_intensity_mmh="hourly_rainfall_fallback",
            pop="hourly_rainfall_fallback",
        )
    reconcile_hourly_rain([parent], points)
    assert parent.values == {"precip_mm": 2.0, "pop": 0.2}


def test_future_hour_is_not_borrowed_into_the_daily_prefix():
    issue = NOW + timedelta(minutes=30)
    points = minutes(issue, rate=None, pop=None)
    hours = [hour(NOW + timedelta(hours=1), 2.0, 0.2)]
    fill_minutely_rain(points, hours)
    assert all(point.precip_intensity_mmh is None for point in points[:30])
    assert all(point.pop is None for point in points[:30])
    parent = daily(issue.date(), amount=2.0, pop=0.2)
    reconcile_hourly_rain(hours, points)
    reconcile([parent], hours, points, issue=issue)
    assert parent.values == {"precip_sum_mm": 2.0, "pop": 0.2}


def test_native_prefix_probability_is_a_daily_floor_even_without_hourly_pop():
    issue = NOW + timedelta(minutes=30)
    points = minutes(issue, count=30, rate=None, pop=0.9)
    parent = daily(issue.date(), pop=0.2)
    reconcile(
        [parent], [hour(NOW + timedelta(hours=1), 2.0, None)], points, issue=issue
    )
    assert parent.values["pop"] == 0.9
    assert "native minute" in parent.selection_reasons["pop"]
    # A minute ending at local midnight belongs to the preceding day.
    midnight = local_day_start_utc(
        issue.date() + timedelta(days=1), "America/Los_Angeles"
    )
    following = daily(issue.date() + timedelta(days=1), pop=0.2)
    reconcile(
        [following],
        [],
        minutes(midnight - timedelta(minutes=1), count=1, pop=0.9),
        issue=issue,
    )
    assert following.values["pop"] == 0.2


@pytest.mark.parametrize("difference", [0.0, 5e-13])
def test_unchanged_complete_day_keeps_original_distribution_and_release(difference):
    day = (NOW + timedelta(days=1)).date()
    start = local_day_start_utc(day, "America/Los_Angeles")
    parent = DailyPoint(
        str(day),
        1,
        values={"precip_sum_mm": 24.0 + difference, "pop": 0.9},
        methods={"precip_sum_mm": "csgd"},
        quantiles={"precip_sum_mm": {"0.5": 24.0}},
        quantiles_source={"precip_sum_mm": "native"},
        release_ids={"precip_sum_mm": "daily-release"},
        selection_reasons={"precip_sum_mm": "promoted"},
    )
    before = deepcopy(parent)
    reconcile([parent], [hour(start + timedelta(hours=i)) for i in range(24)])
    assert parent == before


def test_changed_daily_amount_clears_only_its_own_attribution():
    day = (NOW + timedelta(days=1)).date()
    start = local_day_start_utc(day, "America/Los_Angeles")
    parent = DailyPoint(
        str(day),
        1,
        values={"precip_sum_mm": 25.0, "pop": 0.9},
        methods={"precip_sum_mm": "csgd", "pop": "pop_calibration"},
        quantiles={"precip_sum_mm": {"0.5": 25.0}, "pop": {"0.5": 0.9}},
        quantiles_source={"precip_sum_mm": "native", "pop": "native"},
        release_ids={"precip_sum_mm": "daily-release", "pop": "pop-release"},
    )
    reconcile([parent], [hour(start + timedelta(hours=i)) for i in range(24)])
    assert parent.values["precip_sum_mm"] == 24.0
    assert parent.release_ids == {"pop": "pop-release"}
    assert parent.quantiles == {"pop": {"0.5": 0.9}}
    assert parent.quantiles_source == {"pop": "native"}
    assert parent.methods["pop"] == "pop_calibration"


def test_status_names_wet_guard_before_reconciliation_and_otherwise_abstains():
    parent = hour(NOW, 0.0)
    effects = RainfallEffects()
    assert rainfall_status_reason(effects) is None
    reconcile_hourly_rain([parent], minutes(NOW), effects=effects)
    assert "reconciliation changed served values" in rainfall_status_reason(effects)
    effects.guarded_release_removed = True
    assert "fewer than 5 wet local days" in rainfall_status_reason(effects)


@pytest.mark.parametrize(
    ("original", "release", "removed"),
    [
        (0.0, "old", True),
        (0.0, None, False),
        (None, "old", False),
        (float("nan"), "old", False),
        (6.0 + 5e-13, "old", False),
    ],
)
def test_effects_require_changed_finite_released_values(original, release, removed):
    parent = hour(NOW, original)
    parent.release_ids.clear()
    if release:
        parent.release_ids["precip_mm"] = release
    effects = RainfallEffects()
    reconcile_hourly_rain([parent], minutes(NOW), effects=effects)
    assert effects.reconciled_release_removed is removed
    assert bool(rainfall_status_reason(effects)) is removed


def test_daily_reconciliation_records_only_actual_attribution_loss():
    parent = daily(NOW.date())
    parent.release_ids["precip_sum_mm"] = "daily-release"
    effects = RainfallEffects()
    reconcile_daily_rain(
        [parent],
        [hour(NOW, 2.0)],
        [],
        issue_time=NOW,
        timezone="America/Los_Angeles",
        observed_today_mm=None,
        threshold_mm=0.254,
        effects=effects,
    )
    assert effects.reconciled_release_removed
    assert parent.values["precip_sum_mm"] == 2.0
    assert "precip_sum_mm" not in parent.release_ids


def test_native_probability_crossing_half_hour_midnight_floors_following_day():
    issue = utc(2026, 10, 11, 17, 45)
    hours = [hour(utc(2026, 10, 11, h), 0.0, 0.2) for h in (18, 19)]
    points = minutes(utc(2026, 10, 11, 18, 34), count=1, rate=None, pop=0.9)
    following = daily("2026-10-12", pop=0.2)
    reconcile_hourly_rain(hours, points)
    reconcile_daily_rain(
        [following],
        hours,
        points,
        issue_time=issue,
        timezone="Asia/Kolkata",
        observed_today_mm=None,
        threshold_mm=0.254,
    )
    assert hours[0].values["pop"] == 0.9
    assert hours[1].values["pop"] == 0.2
    assert following.values["pop"] == 0.9
