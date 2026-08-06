"""Zone H: quality-over-time panels reading the evidence ledgers."""

import math
from datetime import UTC, datetime, timedelta

import polars as pl
from conftest import write_config

from grounded_weather_forecast.dashboard.context import collect_context
from grounded_weather_forecast.dashboard.derive import derive
from grounded_weather_forecast.dashboard.zones import quality_history
from grounded_weather_forecast.reports.evidence import (
    CHURN_LEDGER,
    EPROCESS_WEALTH_LEDGER,
    QUALITY_LEDGER,
    SERVED_QUALITY_LEDGER,
    VERDICTS_LEDGER,
    append_ledger,
    ledger_path,
)

NOW = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)


def context_with(tmp_path, spec, rows):
    config = write_config(tmp_path)
    frame = pl.DataFrame(rows, schema_overrides=dict(spec.schema)).select(
        spec.schema.names()
    )
    append_ledger(frame, ledger_path(config, spec), spec, now=NOW)
    return collect_context(config, now=NOW)


def build_zone(ctx):
    return quality_history.build(ctx, derive(ctx))


def quality_rows(n_variables=1, evaluations=2, worsen=False):
    rows = []
    for evaluation in range(evaluations):
        for variable_index in range(n_variables):
            mae = 1.0
            if worsen and evaluation == evaluations - 1:
                mae = 5.0
            rows.append(
                {
                    "recorded_at": NOW,
                    "evaluation_id": f"eval{evaluation}",
                    "evaluation_created_at": NOW
                    - timedelta(days=evaluations - evaluation),
                    "product": "hourly",
                    "source_kind": "live",
                    "variable": f"var{variable_index}",
                    "truth_semantics": "mean",
                    "lead_bucket": "0-1h",
                    "method_id": "equal_weight",
                    "n": 100,
                    "n_valid_times": 90,
                    "coverage": 1.0,
                    "mae": mae,
                    "rmse": mae,
                    "bias": 0.0,
                    "coverage80": None,
                    "coverage90": None,
                    "crps": None,
                    "pinball": None,
                    "recent_mae": mae,
                    "recent_n": 40,
                    "code_version": "code1",
                    "config_fingerprint": "cfg1",
                    "dataset_fingerprint": "ds1",
                }
            )
    return rows


class TestColdContext:
    def test_all_panels_render_young_history(self, tmp_path):
        ctx = collect_context(write_config(tmp_path), now=NOW)
        zone = build_zone(ctx)
        assert zone.zone_id == "H"
        assert len(zone.panels) == 5
        for panel in zone.panels:
            assert panel.status == "info"
            assert panel.empty_reason is not None


class TestBacktestTrend:
    def test_series_capped_at_eight(self, tmp_path):
        ctx = context_with(tmp_path, QUALITY_LEDGER, quality_rows(n_variables=12))
        panel = build_zone(ctx).panels[0]
        assert panel.chart is not None
        datasets = panel.chart.config["data"]["datasets"]
        assert len(datasets) == 8

    def test_worsening_series_goes_amber(self, tmp_path):
        ctx = context_with(
            tmp_path, QUALITY_LEDGER, quality_rows(evaluations=4, worsen=True)
        )
        panel = build_zone(ctx).panels[0]
        assert panel.status == "amber"

    def test_stable_series_is_ok(self, tmp_path):
        ctx = context_with(tmp_path, QUALITY_LEDGER, quality_rows(evaluations=4))
        assert build_zone(ctx).panels[0].status == "ok"


def churn_rows(changed_count=2, total=4):
    rows = []
    for index in range(total):
        rows.append(
            {
                "recorded_at": NOW,
                "from_release_id": "relA",
                "to_release_id": "relB",
                "from_promoted_at": NOW - timedelta(days=1),
                "to_promoted_at": NOW,
                "slice_key": f"hourly.var{index}.0-1h",
                "product": "hourly",
                "variable": f"var{index}",
                "lead_bucket": "0-1h",
                "from_method": "equal_weight",
                "to_method": "harmonic" if index < changed_count else "equal_weight",
                "changed": index < changed_count,
                "from_mae": 1.0,
                "to_mae": 0.9,
                "from_n": 100,
                "to_n": 100,
                "dataset_fingerprint": "ds1",
                "config_fingerprint": "cfg1",
                "code_version": "code1",
            }
        )
    return rows


class TestChurnPanel:
    def test_rate_and_diff_table(self, tmp_path):
        ctx = context_with(tmp_path, CHURN_LEDGER, churn_rows())
        panel = build_zone(ctx).panels[1]
        assert panel.status == "amber"  # 2/4 = 0.5
        assert panel.table is not None
        assert len(panel.table.rows) == 2  # changed slices only
        assert panel.stats[0].value == "50%"

    def test_low_churn_is_ok(self, tmp_path):
        ctx = context_with(
            tmp_path, CHURN_LEDGER, churn_rows(changed_count=1, total=10)
        )
        assert build_zone(ctx).panels[1].status == "ok"


def served_rows(days=7, live=2.0, backtest=1.0):
    rows = []
    for day in range(days):
        rows.append(
            {
                "recorded_at": NOW - timedelta(days=days - day),
                "as_of_date": (NOW - timedelta(days=days - day)).date(),
                "product": "hourly",
                "variable": "temp_c",
                "truth_semantics": "mean",
                "lead_bucket": "0-1h",
                "method_id": "equal_weight",
                "release_id": "relA",
                "dataset_fingerprint": "ds1",
                "n": 24,
                "live_mae": live,
                "live_rmse": live,
                "live_bias": 0.0,
                "backtest_mae": backtest,
                "mae_gap": live - backtest,
                "code_version": "code1",
                "config_fingerprint": "cfg1",
            }
        )
    return rows


class TestServedPromise:
    def test_red_past_gap_factor(self, tmp_path):
        # live 2.0 > live_gap_factor (1.5) * backtest 1.0
        ctx = context_with(tmp_path, SERVED_QUALITY_LEDGER, served_rows())
        assert build_zone(ctx).panels[2].status == "red"

    def test_balanced_history_is_ok(self, tmp_path):
        ctx = context_with(
            tmp_path, SERVED_QUALITY_LEDGER, served_rows(live=1.0, backtest=1.0)
        )
        panel = build_zone(ctx).panels[2]
        assert panel.status == "ok"
        assert panel.chart is not None


def verdict_rows(agree=0.7, days=3):
    rows = []
    for day in range(days):
        moment = NOW - timedelta(days=days - day)
        for name, value in (
            ("recalib_win_share_cqr", 0.6),
            ("gate_agree_rate", agree),
        ):
            rows.append(
                {
                    "recorded_at": moment,
                    "evaluation_id": f"eval{day}",
                    "product": "hourly",
                    "source_kind": "live",
                    "name": name,
                    "value": value,
                    "code_version": "code1",
                    "config_fingerprint": "cfg1",
                    "dataset_fingerprint": "ds1",
                }
            )
    return rows


class TestVerdictsPanel:
    def test_amber_below_agree_threshold(self, tmp_path):
        ctx = context_with(tmp_path, VERDICTS_LEDGER, verdict_rows(agree=0.7))
        panel = build_zone(ctx).panels[3]
        assert panel.status == "amber"

    def test_healthy_agreement_is_ok(self, tmp_path):
        ctx = context_with(tmp_path, VERDICTS_LEDGER, verdict_rows(agree=1.0))
        assert build_zone(ctx).panels[3].status == "ok"


def wealth_rows_fixture(pairs=10, resets=0):
    rows = []
    for index in range(pairs):
        for step in (1, 2):
            rows.append(
                {
                    "recorded_at": NOW - timedelta(days=2 - step),
                    "product": "hourly",
                    "pair_key": f"hourly|temp_c|mean|0-1h|cand{index}|ref",
                    "resets": resets,
                    "t": step * 10,
                    "log_e": float(index) + step * 0.1,
                    "lam": 0.3,
                    "scale": 1.0,
                    "config_fingerprint": "cfg1",
                    "code_version": "code1",
                }
            )
    return rows


class TestWealthPanel:
    def test_top_pairs_and_threshold_reference(self, tmp_path):
        ctx = context_with(tmp_path, EPROCESS_WEALTH_LEDGER, wealth_rows_fixture())
        panel = build_zone(ctx).panels[4]
        assert panel.chart is not None
        datasets = panel.chart.config["data"]["datasets"]
        # 6 pairs + the dashed threshold guide
        assert len(datasets) == 7
        guide = datasets[-1]
        assert guide["borderDash"] == [4, 4]
        alpha = ctx.config.promotion.alpha
        assert guide["data"][0] == math.log(1.0 / alpha)

    def test_only_newest_reset_era_plotted(self, tmp_path):
        old_era = wealth_rows_fixture(pairs=1, resets=0)
        new_era = [
            {**row, "resets": 1, "log_e": 9.0} for row in wealth_rows_fixture(pairs=1)
        ]
        ctx = context_with(tmp_path, EPROCESS_WEALTH_LEDGER, old_era + new_era)
        panel = build_zone(ctx).panels[4]
        series = panel.chart.config["data"]["datasets"][0]
        finite = [value for value in series["data"] if value is not None]
        assert all(value == 9.0 for value in finite)
