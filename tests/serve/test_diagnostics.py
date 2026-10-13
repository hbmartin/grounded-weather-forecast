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


def attempt(diagnostics, branch, *, promoted=True, product="hourly"):
    diagnostics.start_fit(product, "temp_c", "inst")
    if branch is not None:
        diagnostics.note_branch(product, "temp_c", "inst", branch)
    diagnostics.record_unavailable(
        ["release" if promoted else None],
        product=product,
        variable="temp_c",
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
