"""Promoted failures belong to the product snapshot, not its latest attempt."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from grounded_weather_forecast.serve import diagnostics as module
from grounded_weather_forecast.serve.diagnostics import (
    FitDiagnostics,
    FitUnavailableBranch,
    FitUnavailableCause,
)

NOW = datetime(2026, 10, 13, 12, tzinfo=UTC)


def history():
    return pl.DataFrame(
        {"issue_time": [NOW - timedelta(hours=1)], "t__temp_c__inst": [1.0]}
    )


def attempt(diagnostics, branch, *, promoted=True, product="hourly", variable="temp_c"):
    diagnostics.start_fit(product, variable, "inst")
    if branch is not None:
        diagnostics.note_branch(product, variable, "inst", branch)
    diagnostics.record_unavailable(
        ["release" if promoted else None],
        product=product,
        variable=variable,
        truth_semantics="inst",
        truth_column="t__temp_c__inst",
    )


@pytest.mark.parametrize("later", [None, *FitUnavailableBranch])
def test_later_unpromoted_attempt_preserves_original_pending_branch(later):
    raw = history()
    diagnostics = FitDiagnostics()
    diagnostics.capture_training("hourly", raw)
    attempt(diagnostics, FitUnavailableBranch.EMPTY_SLICE)
    attempt(diagnostics, later, promoted=False)
    diagnostics.resolve_training("hourly", raw, NOW)
    assert len(diagnostics.failures) == 1
    assert diagnostics.failures[0].cause == FitUnavailableCause.CAUSAL_EXCLUSION


def test_successful_attempt_does_not_erase_promoted_failure():
    diagnostics = FitDiagnostics()
    diagnostics.capture_training("hourly", history())
    attempt(diagnostics, FitUnavailableBranch.NO_FORECAST_COLUMNS)
    previous = diagnostics.failures
    diagnostics.start_fit("hourly", "temp_c", "inst")
    assert diagnostics.failures == previous


def test_distinct_promoted_branches_survive_and_sort_deterministically():
    raw = history()
    diagnostics = FitDiagnostics()
    diagnostics.capture_training("hourly", raw)
    for branch in [
        FitUnavailableBranch.EMPTY_SLICE,
        FitUnavailableBranch.NO_FORECAST_COLUMNS,
        None,
    ]:
        attempt(diagnostics, branch)
    diagnostics.resolve_training("hourly", raw, NOW)
    assert [failure.cause for failure in diagnostics.failures] == [
        FitUnavailableCause.CAUSAL_EXCLUSION,
        FitUnavailableCause.NO_FORECAST_COLUMNS,
        FitUnavailableCause.UNKNOWN_HISTORY,
    ]


def test_identical_failures_before_and_after_resolution_scan_once(monkeypatch):
    raw = history()
    diagnostics = FitDiagnostics()
    diagnostics.capture_training("hourly", raw)
    real = module._truth_availability
    calls = []

    def count(frame, columns, issue):
        calls.append(tuple(columns))
        return real(frame, columns, issue)

    monkeypatch.setattr(module, "_truth_availability", count)
    for _ in range(2):
        attempt(diagnostics, FitUnavailableBranch.EMPTY_SLICE)
    diagnostics.resolve_training("hourly", raw, NOW)
    attempt(diagnostics, FitUnavailableBranch.EMPTY_SLICE)
    diagnostics.resolve_training("hourly", raw, NOW)
    assert calls == [("t__temp_c__inst",)]
    assert len(diagnostics.failures) == 1


def test_product_capture_resets_only_its_own_snapshot():
    raw = history()
    diagnostics = FitDiagnostics()
    for product in ("hourly", "daily"):
        diagnostics.capture_training(product, raw)
        attempt(diagnostics, FitUnavailableBranch.EMPTY_SLICE, product=product)
    diagnostics.capture_training("hourly", raw.head(0))
    diagnostics.resolve_training("daily", raw, NOW)
    assert [failure.product for failure in diagnostics.failures] == ["daily"]
    assert diagnostics.training["hourly"].raw_rows == 0
    assert not diagnostics.pending
    assert all(key[0] != "hourly" for key in diagnostics.branches)


def test_different_branches_with_same_scoped_cause_are_deduplicated():
    raw = history().head(0)
    diagnostics = FitDiagnostics()
    diagnostics.capture_training("hourly", raw)
    for branch in FitUnavailableBranch:
        attempt(diagnostics, branch)
    diagnostics.resolve_training("hourly", raw, NOW)
    assert len(diagnostics.failures) == 1
    assert diagnostics.failures[0].cause == FitUnavailableCause.EMPTY_MATRIX


def test_pending_is_not_a_completed_unknown_history_diagnosis():
    raw = history()
    diagnostics = FitDiagnostics()
    for product in ("hourly", "daily"):
        diagnostics.capture_training(product, raw)
        attempt(diagnostics, FitUnavailableBranch.EMPTY_SLICE, product=product)
    assert diagnostics.promoted_fit_unavailable
    assert not diagnostics.unavailable
    with pytest.raises(RuntimeError, match="daily, hourly; call resolve_training"):
        _ = diagnostics.failures
    diagnostics.resolve_training("hourly", raw, NOW)
    with pytest.raises(RuntimeError, match="pending for daily; call resolve_training"):
        _ = diagnostics.failures
    diagnostics.resolve_training("daily", raw, NOW)
    assert len(diagnostics.failures) == 2
    assert diagnostics.promoted_fit_unavailable


def test_reset_discards_pending_only_for_that_product():
    raw = history()
    diagnostics = FitDiagnostics()
    for product in ("hourly", "daily"):
        diagnostics.capture_training(product, raw)
        attempt(diagnostics, FitUnavailableBranch.EMPTY_SLICE, product=product)
    diagnostics.capture_training("hourly", raw)
    with pytest.raises(RuntimeError, match="pending for daily;"):
        _ = diagnostics.failures
    diagnostics.capture_training("daily", raw)
    assert not diagnostics.promoted_fit_unavailable
    assert diagnostics.failures == ()


def test_branchless_unknown_history_is_already_finalized():
    diagnostics = FitDiagnostics()
    diagnostics.capture_training("hourly", history())
    attempt(diagnostics, None)
    assert not diagnostics.pending
    assert diagnostics.failures[0].cause == FitUnavailableCause.UNKNOWN_HISTORY


@pytest.mark.parametrize("empty", [False, True])
def test_incremental_resolution_accumulates_facts_and_reuses_columns(
    monkeypatch, empty
):
    raw = history().with_columns(pl.lit(None, dtype=pl.Float64).alias("t__rain__inst"))
    if empty:
        raw = raw.head(0)
    diagnostics = FitDiagnostics()
    diagnostics.capture_training("hourly", raw)
    real = module._truth_availability
    calls = []

    def count(frame, columns, issue):
        assert frame is raw
        assert issue == NOW
        calls.append(tuple(columns))
        return real(frame, columns, issue)

    monkeypatch.setattr(module, "_truth_availability", count)
    attempt(diagnostics, FitUnavailableBranch.EMPTY_SLICE)
    diagnostics.resolve_training("hourly", raw, NOW)
    temp_facts = diagnostics.training["hourly"].truth["t__temp_c__inst"]
    diagnostics.start_fit("hourly", "rain", "inst")
    diagnostics.note_branch("hourly", "rain", "inst", FitUnavailableBranch.EMPTY_SLICE)
    diagnostics.record_unavailable(
        ["release"],
        product="hourly",
        variable="rain",
        truth_semantics="inst",
        truth_column="t__rain__inst",
    )
    diagnostics.resolve_training("hourly", raw, NOW)
    assert diagnostics.training["hourly"].truth["t__temp_c__inst"] == temp_facts
    # A distinct target sharing an already resolved column needs no new scan.
    attempt(diagnostics, FitUnavailableBranch.EMPTY_SLICE, variable="other_temp")
    diagnostics.resolve_training("hourly", raw, NOW)
    expected_temp = (
        FitUnavailableCause.EMPTY_MATRIX
        if empty
        else FitUnavailableCause.CAUSAL_EXCLUSION
    )
    assert [(failure.variable, failure.cause) for failure in diagnostics.failures] == [
        ("other_temp", expected_temp),
        (
            "rain",
            FitUnavailableCause.EMPTY_MATRIX
            if empty
            else FitUnavailableCause.NO_FINITE_TRUTH,
        ),
        ("temp_c", expected_temp),
    ]
    assert calls == [("t__temp_c__inst",), ("t__rain__inst",)]
    assert set(diagnostics.training["hourly"].truth) == {
        "t__temp_c__inst",
        "t__rain__inst",
    }
    diagnostics.capture_training("hourly", raw)
    attempt(diagnostics, FitUnavailableBranch.EMPTY_SLICE)
    diagnostics.resolve_training("hourly", raw, NOW)
    assert calls[-1] == ("t__temp_c__inst",)
    assert len(calls) == 3
