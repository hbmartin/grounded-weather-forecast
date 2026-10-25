"""A fresh dashboard must not make retained QC evidence look freshly checked."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from conftest import write_config

from grounded_weather_forecast.dashboard.context import DashboardContext
from grounded_weather_forecast.dashboard.zones.data_trust import _truth_cross_check

NOW = datetime(2026, 10, 12, 12, tzinfo=UTC)


def context(tmp_path, *, generated_at=NOW.isoformat(), verdict=None, runs=()):
    payload = {
        "schema_version": 3,
        "drift_verdict": verdict
        if verdict is not None
        else {
            "exceeded": False,
            "latched": False,
            "reason": "previous quiet verdict",
            "days_used": 28,
            "series": [{"date": "2026-10-10", "difference": 0.1, "cusum": 0.2}],
        },
    }
    if generated_at != "absent":
        payload["generated_at"] = generated_at
    return DashboardContext(
        config=write_config(tmp_path),
        now=NOW,
        truth_qc=payload,
        runs=pl.DataFrame(
            list(runs),
            schema={
                "command": pl.String,
                "record_kind": pl.String,
                "started_at": pl.Datetime("us", "UTC"),
                "ended_at": pl.Datetime("us", "UTC"),
                "exit_code": pl.Int64,
                "error": pl.String,
                "error_detail": pl.String,
            },
        ),
    )


def attempt(
    *, kind="stage", ended=NOW, code=1, error=None, detail=None, command="truth-qc"
):
    return {
        "command": command,
        "record_kind": kind,
        "started_at": NOW - timedelta(minutes=10),
        "ended_at": ended,
        "exit_code": code,
        "error": error,
        "error_detail": detail,
    }


@pytest.mark.parametrize("hours,stale", [(47.9, False), (48, False), (48.01, True)])
def test_qc_age_uses_48_hour_boundary(tmp_path, hours, stale):
    ctx = context(tmp_path, generated_at=(NOW - timedelta(hours=hours)).isoformat())
    panel = _truth_cross_check(ctx)
    assert panel.status == ("amber" if stale else "ok")
    assert ("over 48 hours" in panel.intro) == stale
    assert any(stat.label == "refreshed" for stat in panel.stats)
    from grounded_weather_forecast.dashboard.derive import derive
    from grounded_weather_forecast.dashboard.writer import _alert_inputs
    from grounded_weather_forecast.reports.alerts import evaluate_alerts

    alerts = [
        a
        for a in evaluate_alerts(_alert_inputs(ctx, derive(ctx)))
        if a.panel_id == "truth-qc-freshness"
    ]
    assert bool(alerts) == stale


@pytest.mark.parametrize(
    "stamp",
    [
        "absent",
        None,
        "invalid",
        "2026-10-12T10:00:00",
        (NOW + timedelta(seconds=1)).isoformat(),
    ],
)
def test_unknown_or_invalid_refresh_time_is_amber(tmp_path, stamp):
    panel = _truth_cross_check(context(tmp_path, generated_at=stamp))
    assert panel.status == "amber"
    assert "refresh time is unknown or invalid" in panel.intro
    assert panel.chart is not None


@pytest.mark.parametrize(
    "date,now,stale",
    [
        ("2026-10-10", NOW, False),
        ("2026-10-09", NOW, True),
        ("2026-10-09", NOW.replace(hour=0), False),
    ],
)
def test_legacy_date_uses_latest_possible_refresh_bound(tmp_path, date, now, stale):
    ctx = context(tmp_path, generated_at="absent")
    ctx.truth_qc["drift_verdict"]["as_of_date"] = date
    panel = _truth_cross_check(replace(ctx, now=now))
    assert panel.status == ("amber" if stale else "ok")
    assert any(stat.value == f"{date} (date only)" for stat in panel.stats)
    assert any(
        stat.label == "verdict age" and "at least" in stat.value for stat in panel.stats
    )


@pytest.mark.parametrize("date", ["invalid", "2026-10-13", None])
def test_invalid_or_future_legacy_dates_do_not_claim_freshness(tmp_path, date):
    ctx = context(tmp_path, generated_at="absent")
    ctx.truth_qc["drift_verdict"]["as_of_date"] = date
    assert _truth_cross_check(ctx).status == "amber"


@pytest.mark.parametrize("kind", ["stage", "invocation", None])
@pytest.mark.parametrize(
    "code,error", [(1, None), (None, None), (0, "QC write failed")]
)
def test_failed_refresh_keeps_verdict_and_chart_with_warning(
    tmp_path, kind, code, error
):
    ctx = context(tmp_path, generated_at=(NOW - timedelta(hours=1)).isoformat())
    original = _truth_cross_check(ctx)
    failed = context(
        tmp_path,
        generated_at=(NOW - timedelta(hours=1)).isoformat(),
        runs=[attempt(kind=kind, code=code, error=error)],
    )
    panel = _truth_cross_check(failed)
    assert panel.status == "amber"
    assert "previous quiet verdict" in panel.intro
    assert "latest truth-QC refresh failed" in panel.intro
    assert panel.chart == original.chart
    assert any(stat.label == "failed refresh" for stat in panel.stats)


@pytest.mark.parametrize(
    "row",
    [
        attempt(ended=None),
        attempt(ended=NOW + timedelta(seconds=1)),
        attempt(command="maintain"),
        attempt(kind="other"),
    ],
)
def test_unfinished_future_and_unrelated_attempts_are_ignored(tmp_path, row):
    panel = _truth_cross_check(context(tmp_path, runs=[row]))
    assert panel.status == "ok"
    assert "failed" not in panel.intro


def test_reversed_interval_retains_verdict_with_unknown_attempt_warning(tmp_path):
    panel = _truth_cross_check(
        context(tmp_path, runs=[attempt(ended=NOW - timedelta(minutes=11))])
    )
    assert panel.status == "amber"
    assert "outcome is unknown" in panel.intro
    assert "previous quiet verdict" in panel.intro
    assert "refresh failed" not in panel.intro


@pytest.mark.parametrize("new_ledger_success", [False, True])
def test_newer_successful_refresh_clears_old_failure(tmp_path, new_ledger_success):
    rows = [attempt(ended=NOW - timedelta(minutes=5))]
    if new_ledger_success:
        rows.append(attempt(code=0))
    panel = _truth_cross_check(context(tmp_path, runs=rows))
    assert panel.status == "ok"
    assert "failed" not in panel.intro


def test_legacy_refresh_on_later_day_supersedes_failed_attempt(tmp_path):
    row = attempt(ended=NOW - timedelta(days=1))
    row["started_at"] = row["ended_at"] - timedelta(minutes=10)
    ctx = context(tmp_path, generated_at="absent", runs=[row])
    ctx.truth_qc["drift_verdict"]["as_of_date"] = NOW.date().isoformat()
    assert _truth_cross_check(ctx).status == "ok"


@pytest.mark.parametrize("has_failure", [False, True])
def test_no_verdict_is_young_or_failed_refresh(tmp_path, has_failure):
    ctx = context(tmp_path, runs=[attempt()] if has_failure else [])
    panel = _truth_cross_check(replace(ctx, truth_qc=None))
    assert panel.status == ("amber" if has_failure else "info")
    assert panel.empty_reason
    assert ("failed" in panel.empty_reason) == has_failure
    from grounded_weather_forecast.dashboard.derive import derive
    from grounded_weather_forecast.dashboard.writer import _alert_inputs
    from grounded_weather_forecast.reports.alerts import evaluate_alerts

    empty_ctx = replace(ctx, truth_qc=None)
    (alert,) = [
        a
        for a in evaluate_alerts(_alert_inputs(empty_ctx, derive(empty_ctx)))
        if a.panel_id == "truth-qc-freshness"
    ]
    assert alert.severity == ("amber" if has_failure else "info")
    assert alert.evaluable == has_failure


def test_latched_drift_remains_red_with_staleness_and_failed_refresh(tmp_path):
    ctx = context(
        tmp_path,
        generated_at=(NOW - timedelta(days=3)).isoformat(),
        verdict={"latched": True, "exceeded": True, "attribution": "station_drift"},
        runs=[attempt()],
    )
    panel = _truth_cross_check(ctx)
    assert panel.status == "red"
    assert "over 48 hours" in panel.intro
    assert "failed" in panel.intro


@pytest.mark.parametrize(
    "hours,unknown,failed", [(1, False, True), (49, False, False), (1, True, False)]
)
def test_qc_chips_describe_their_own_conditions(tmp_path, hours, unknown, failed):
    ctx = context(
        tmp_path,
        generated_at=None if unknown else (NOW - timedelta(hours=hours)).isoformat(),
        runs=[attempt(detail="neighbor HTTP unavailable")] if failed else [],
    )
    panel = _truth_cross_check(ctx)
    chips = {stat.label: stat for stat in panel.stats}
    assert panel.status == "amber"
    assert chips["verdict"].status == "ok"
    assert chips["refreshed"].status == ("amber" if unknown else "ok")
    assert chips["verdict age"].status == ("amber" if unknown or hours > 48 else "ok")
    if failed:
        assert "neighbor HTTP unavailable" in chips["failed refresh"].value


def test_shared_qc_assessment_drives_panel_and_single_alert(tmp_path, monkeypatch):
    from grounded_weather_forecast.dashboard import derive as derivation
    from grounded_weather_forecast.dashboard import writer
    from grounded_weather_forecast.dashboard.truth_qc import truth_qc_freshness
    from grounded_weather_forecast.dashboard.derive import derive
    from grounded_weather_forecast.dashboard.writer import _alert_inputs
    from grounded_weather_forecast.dashboard.zones import data_trust
    from grounded_weather_forecast.reports.alerts import evaluate_alerts

    ctx = context(
        tmp_path,
        generated_at=(NOW - timedelta(hours=49)).isoformat(),
        runs=[attempt(detail="HTTP unavailable")],
    )
    calls = []

    def assess(received):
        assert received is ctx
        calls.append(received)
        return truth_qc_freshness(received)

    monkeypatch.setattr(derivation, "truth_qc_freshness", assess)
    derived = derive(ctx)
    assert calls == [ctx]
    monkeypatch.setattr(
        writer, "truth_qc_freshness", lambda _ctx: pytest.fail("assessment recomputed")
    )
    monkeypatch.setattr(
        data_trust,
        "truth_qc_freshness",
        lambda _ctx: pytest.fail("assessment recomputed"),
    )
    zone = data_trust.build(ctx, derived)
    panel = next(panel for panel in zone.panels if panel.panel_id == "b5")
    inputs = _alert_inputs(ctx, derived)
    assert inputs.truth_qc_freshness is derived.truth_qc_freshness
    alerts = [
        alert
        for alert in evaluate_alerts(inputs)
        if alert.panel_id == "truth-qc-freshness"
    ]
    assert len(alerts) == 1
    assert alerts[0].severity == "amber"
    for warning in derived.truth_qc_freshness.warnings:
        assert warning in alerts[0].message
        assert warning in panel.intro


def test_error_details_are_html_escaped(tmp_path):
    from grounded_weather_forecast.dashboard.html import render_panel

    panel = _truth_cross_check(
        context(
            tmp_path,
            generated_at=(NOW - timedelta(hours=1)).isoformat(),
            runs=[attempt(detail='<script>alert("x")</script>')],
        )
    )
    html = render_panel(panel)
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


@pytest.mark.parametrize("state", ["fresh", "stale", "unknown", "failed", "missing"])
def test_direct_alert_inputs_fallback_agrees_with_panel(tmp_path, monkeypatch, state):
    from grounded_weather_forecast.dashboard import writer
    from grounded_weather_forecast.dashboard.truth_qc import truth_qc_freshness
    from grounded_weather_forecast.dashboard.derive import Derived
    from grounded_weather_forecast.dashboard.writer import _alert_inputs
    from grounded_weather_forecast.reports.alerts import evaluate_alerts

    ctx = context(
        tmp_path,
        generated_at=None
        if state == "unknown"
        else (NOW - timedelta(hours=49 if state == "stale" else 1)).isoformat(),
        runs=[attempt(detail="neighbor unavailable")] if state == "failed" else [],
    )
    if state == "missing":
        ctx = replace(ctx, truth_qc=None)
    calls = []

    def assess(received):
        assert received is ctx
        calls.append(received)
        return truth_qc_freshness(received)

    monkeypatch.setattr(writer, "truth_qc_freshness", assess)
    inputs = _alert_inputs(ctx, Derived())
    assert calls == [ctx]
    assert inputs.truth_qc_freshness is not None
    panel = _truth_cross_check(ctx)
    alerts = [a for a in evaluate_alerts(inputs) if a.panel_id == "truth-qc-freshness"]
    if state == "fresh":
        assert panel.status == "ok" and not alerts
    else:
        (alert,) = alerts
        assert alert.severity == panel.status
        for warning in inputs.truth_qc_freshness.warnings:
            assert warning in (panel.intro or panel.empty_reason)
            assert warning in alert.message


def test_unsupplied_assessment_does_not_assert_missing_verdict(tmp_path):
    from grounded_weather_forecast.reports.alerts import AlertInputs, evaluate_alerts

    ctx = context(tmp_path)
    (alert,) = [
        a
        for a in evaluate_alerts(AlertInputs(ctx.config, NOW))
        if a.panel_id == "truth-qc-freshness"
    ]
    assert alert.severity == "info" and not alert.evaluable
    assert "assessment was not supplied" in alert.message
    assert "no truth-QC verdict" not in alert.message


def test_threshold_decision_warning_and_alert_provenance_share_policy(
    tmp_path, monkeypatch
):
    from grounded_weather_forecast import health_policy as truth_qc
    from grounded_weather_forecast.dashboard.derive import derive
    from grounded_weather_forecast.dashboard.writer import _alert_inputs
    from grounded_weather_forecast.reports.alerts import evaluate_alerts

    monkeypatch.setitem(truth_qc.FRESHNESS_THRESHOLDS, "truth_qc_age_minutes", 1440.0)
    ctx = context(tmp_path, generated_at=(NOW - timedelta(hours=30)).isoformat())
    panel = _truth_cross_check(ctx)
    (alert,) = [
        a
        for a in evaluate_alerts(_alert_inputs(ctx, derive(ctx)))
        if a.panel_id == "truth-qc-freshness"
    ]
    assert panel.status == alert.severity == "amber"
    assert "over 24 hours old" in panel.intro
    assert "over 24 hours old" in alert.message
    assert "24 hours" in alert.threshold
    assert "48 hours" not in alert.message + alert.threshold


@pytest.mark.parametrize(
    "error,detail,expected",
    [
        ("RuntimeError", "neighbor unavailable", "RuntimeError: neighbor unavailable"),
        (None, "handled failure", "handled failure"),
        ("OSError", None, "OSError"),
        ("OSError", "", "OSError"),
        ("OSError", "OSError", "OSError"),
        (None, None, "exit 1"),
    ],
)
def test_failed_refresh_preserves_class_and_detail(tmp_path, error, detail, expected):
    from grounded_weather_forecast.dashboard.derive import derive
    from grounded_weather_forecast.dashboard.writer import _alert_inputs
    from grounded_weather_forecast.reports.alerts import evaluate_alerts

    ctx = context(
        tmp_path,
        generated_at=(NOW - timedelta(hours=1)).isoformat(),
        runs=[attempt(error=error, detail=detail)],
    )
    panel = _truth_cross_check(ctx)
    chip = next(stat for stat in panel.stats if stat.label == "failed refresh")
    (alert,) = [
        a
        for a in evaluate_alerts(_alert_inputs(ctx, derive(ctx)))
        if a.panel_id == "truth-qc-freshness"
    ]
    assert expected in chip.value and expected in alert.message
    if error is not None and error == detail:
        assert chip.value.count(error) == 1
