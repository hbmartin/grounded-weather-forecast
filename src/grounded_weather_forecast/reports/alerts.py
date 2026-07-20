"""Operator alerts evaluated from already-loaded artifacts.

Pure functions: every input is a frame or mapping the caller has loaded,
plus the ``Config``. Each alert's ``threshold`` string names the real config
knob or module constant that defines it — the alerting invents no policy.
Families that cannot be evaluated yet (young deployment: nothing served,
nothing promoted) return one non-evaluable info alert instead of silence or
a false alarm.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import pairwise
from typing import Literal
from zoneinfo import ZoneInfo

import polars as pl

from grounded_weather_forecast.config import Config
from grounded_weather_forecast.contracts import (
    age_col,
    finite_number,
    provider_age_is_fresh,
)
from grounded_weather_forecast.evaluation import config_fingerprint
from grounded_weather_forecast.leads import (
    DAILY_BUCKET_LABELS,
    HOURLY_BUCKET_LABELS,
)
from grounded_weather_forecast.reports.leaderboard import CONSUMER_TOLERANCES

type Severity = Literal["red", "amber", "info"]

_SEVERITY_ORDER: Mapping[str, int] = {"red": 0, "amber": 1, "info": 2}
_MIN_BIAS_SAMPLES = 8  # mirrors leaderboard._MIN_DM_SAMPLES
_SWAP_WINDOW_DAYS = 3.0  # mirrors drift._FAST_WINDOW_DAYS
_SWAP_MIN_HOLD_DAYS = 0.5  # a crossing shorter than this is a tie, not a swap
_TRUTH_WINDOW_DAYS = 7.0  # the "trailing week" the message promises


@dataclass(frozen=True, slots=True)
class Alert:
    """One operator-facing condition, with the provenance of its threshold."""

    severity: Severity
    zone: str
    panel_id: str
    message: str
    threshold: str
    evaluable: bool = True


@dataclass(frozen=True, slots=True)
class AlertInputs:
    """Pre-loaded evidence; empty frames and None mean 'not on disk'."""

    config: Config
    now: datetime
    manifest: Mapping[str, object] | None = None
    runs: pl.DataFrame = field(default_factory=pl.DataFrame)
    minute_truth: pl.DataFrame = field(default_factory=pl.DataFrame)
    hourly_truth: pl.DataFrame = field(default_factory=pl.DataFrame)
    daily_truth: pl.DataFrame = field(default_factory=pl.DataFrame)
    qc: pl.DataFrame = field(default_factory=pl.DataFrame)
    hourly_matrix: pl.DataFrame = field(default_factory=pl.DataFrame)
    board: pl.DataFrame = field(default_factory=pl.DataFrame)
    live_vs_backtest: pl.DataFrame = field(default_factory=pl.DataFrame)
    drift: Mapping[str, object] | None = None
    latest_status: tuple[str, str | None] | None = None
    releases: tuple[Mapping[str, object], ...] = ()
    observability_history: pl.DataFrame = field(default_factory=pl.DataFrame)
    archive_location: tuple[float, float] | None = None
    unreadable_artifacts: tuple[str, ...] = ()


def _not_evaluable(zone: str, panel_id: str, message: str, threshold: str) -> Alert:
    return Alert(
        severity="info",
        zone=zone,
        panel_id=panel_id,
        message=f"not evaluable yet: {message}",
        threshold=threshold,
        evaluable=False,
    )


def _station_alerts(inputs: AlertInputs) -> tuple[Alert, ...]:
    from grounded_weather_forecast.serve.predict import OBS_STALENESS  # noqa: PLC0415

    threshold = (
        "serve/predict.py::OBS_STALENESS = 30 min; "
        "config [forecasts].max_forecast_age_hours = "
        f"{inputs.config.forecasts.max_forecast_age_hours}"
    )
    if inputs.minute_truth.is_empty() or "ts" not in inputs.minute_truth.columns:
        return (
            _not_evaluable("A", "anchor-lost", "no station observations", threshold),
        )
    newest = inputs.minute_truth["ts"].max()
    if not isinstance(newest, datetime):
        return (
            _not_evaluable("A", "anchor-lost", "no station observations", threshold),
        )
    lag = inputs.now - newest
    stalled = timedelta(hours=inputs.config.forecasts.max_forecast_age_hours)
    if lag > stalled:
        return (
            Alert(
                severity="red",
                zone="A",
                panel_id="ingestion-stalled",
                message=(
                    f"station ingestion stalled: last observation {newest.isoformat()}"
                    f" is {lag.total_seconds() / 3600:.1f}h old"
                ),
                threshold=threshold,
            ),
        )
    if lag > OBS_STALENESS:
        return (
            Alert(
                severity="amber",
                zone="A",
                panel_id="anchor-lost",
                message=(
                    "anchor lost: last observation is "
                    f"{lag.total_seconds() / 60:.0f} min old, beyond the serve "
                    "staleness cap; predictions run unanchored"
                ),
                threshold=threshold,
            ),
        )
    return ()


def _provider_alerts(inputs: AlertInputs) -> tuple[Alert, ...]:
    cap = inputs.config.forecasts.max_forecast_age_hours
    threshold = f"config [forecasts].max_forecast_age_hours = {cap}; manifest.sources"
    expected = _manifest_sources(inputs.manifest)
    matrix = inputs.hourly_matrix
    if not expected or matrix.is_empty() or "issue_time" not in matrix.columns:
        return (
            _not_evaluable(
                "A", "provider-dropped", "no live matrix snapshot", threshold
            ),
        )
    newest = matrix.filter(pl.col("issue_time") == pl.col("issue_time").max())
    row = newest.row(0, named=True)
    ages = {source: finite_number(row.get(age_col(source))) for source in expected}
    missing = [source for source, age in ages.items() if age is None]
    aged = [
        source
        for source, age in ages.items()
        if age is not None and not provider_age_is_fresh(age, cap)
    ]
    alerts: list[Alert] = []
    if missing:
        alerts.append(
            Alert(
                severity="red" if len(missing) == len(expected) else "amber",
                zone="A",
                panel_id="provider-dropped",
                message=(
                    "providers absent from the newest snapshot or carrying "
                    "non-finite ages: "
                    f"{', '.join(sorted(missing))}"
                ),
                threshold=threshold,
            )
        )
    if aged:
        alerts.append(
            Alert(
                severity="amber",
                zone="A",
                panel_id="provider-aged-out",
                message=(
                    "providers past the freshness cap (silently dropped from "
                    f"snapshots): {', '.join(sorted(aged))}"
                ),
                threshold=threshold,
            )
        )
    return tuple(alerts)


def _serving_alerts(inputs: AlertInputs) -> tuple[Alert, ...]:
    alerts: list[Alert] = []
    refusal_threshold = "serve/predict.py::NoForecastDataError; runs ledger exit codes"
    predicts = (
        inputs.runs.filter(pl.col("command") == "predict")
        if not inputs.runs.is_empty() and "command" in inputs.runs.columns
        else pl.DataFrame()
    )
    if predicts.is_empty():
        alerts.append(
            _not_evaluable(
                "F", "serving-refused", "no predict runs recorded", refusal_threshold
            )
        )
    else:
        last = predicts.sort("started_at").row(predicts.height - 1, named=True)
        if last["error"] is not None or (
            last["exit_code"] is not None and last["exit_code"] != 0
        ):
            failure = last["error"] or f"exit code {last['exit_code']}"
            alerts.append(
                Alert(
                    severity="red",
                    zone="F",
                    panel_id="serving-refused",
                    message=f"last predict run failed: {failure}",
                    threshold=refusal_threshold,
                )
            )
    status_threshold = "Forecast.status / selection.no_evidence_reason"
    if inputs.latest_status is None:
        alerts.append(
            _not_evaluable(
                "F", "serving-degraded", "nothing served yet", status_threshold
            )
        )
    else:
        status, reason = inputs.latest_status
        if status == "degraded":
            alerts.append(
                Alert(
                    severity="amber",
                    zone="F",
                    panel_id="serving-degraded",
                    message=f"serving degraded: {reason or 'no promoted release'}",
                    threshold=status_threshold,
                )
            )
    return tuple(alerts)


def _truth_alerts(inputs: AlertInputs) -> tuple[Alert, ...]:
    hour_floor = inputs.config.dataset.min_hour_coverage
    day_floor = inputs.config.dataset.min_day_coverage
    threshold = (
        f"config [dataset].min_hour_coverage = {hour_floor}, "
        f"min_day_coverage = {day_floor}"
    )
    thin: dict[str, float] = {}
    unusable: list[str] = []
    evaluated = False

    hourly = inputs.hourly_truth
    hourly_columns = [c for c in hourly.columns if c.endswith("_cov")]
    if not hourly.is_empty() and hourly_columns and "valid_hour" in hourly.columns:
        # A row count is not a week. `tail(24 * 7)` on a gappy archive spanned
        # 49 days and reported healthy mean coverage while the actual trailing
        # week sat far below the floor.
        recent_hourly = hourly.filter(
            pl.col("valid_hour").is_between(
                inputs.now - timedelta(days=_TRUTH_WINDOW_DAYS),
                inputs.now,
                closed="both",
            )
        )
        if not recent_hourly.is_empty():
            evaluated = True
            for column in hourly_columns:
                if (mean := finite_number(recent_hourly[column].mean())) is None:
                    # Stripped to match the `thin` detail below: an operator
                    # reading both lines should see one name per channel.
                    unusable.append(column.removesuffix("_cov"))
                elif mean < hour_floor:
                    thin[column] = mean

    daily = inputs.daily_truth
    daily_columns = [
        column
        for column in ("coverage_frac", "rain_coverage")
        if column in daily.columns
    ]
    # Judged independently of hourly: returning early when hourly truth was
    # missing silently discarded perfectly good daily coverage.
    if not daily.is_empty() and daily_columns and "date_local" in daily.columns:
        local_today = inputs.now.astimezone(
            ZoneInfo(inputs.config.station.timezone)
        ).date()
        horizon = local_today - timedelta(days=_TRUTH_WINDOW_DAYS)
        recent_daily = daily.filter(
            pl.col("date_local").is_between(horizon, local_today, closed="both")
        )
        if not recent_daily.is_empty():
            evaluated = True
            daily_labels = {
                "coverage_frac": "daily.temperature",
                "rain_coverage": "daily.rain",
            }
            for column in daily_columns:
                if (mean := finite_number(recent_daily[column].mean())) is None:
                    unusable.append(daily_labels[column])
                elif mean < day_floor:
                    thin[daily_labels[column]] = mean

    if not evaluated:
        return (
            _not_evaluable(
                "B",
                "truth-thinning",
                f"no truth in the trailing {_TRUTH_WINDOW_DAYS:.0f} days",
                threshold,
            ),
        )
    alerts: list[Alert] = []
    if unusable:
        alerts.append(
            Alert(
                severity="red",
                zone="B",
                panel_id="truth-thinning",
                message=(
                    "no usable truth-coverage samples: "
                    f"{', '.join(sorted(unusable))}; coverage is absent, not high"
                ),
                threshold=threshold,
            )
        )
    if not thin:
        return tuple(alerts)
    detail = ", ".join(
        f"{column.removesuffix('_cov')}={value:.2f}"
        for column, value in sorted(thin.items())
    )
    alerts.append(
        Alert(
            severity="amber",
            zone="B",
            panel_id="truth-thinning",
            message=(
                f"trailing-week truth coverage below the floor: {detail}; "
                "affected periods are nulled out of training"
            ),
            threshold=threshold,
        )
    )
    return tuple(alerts)


def _sensor_alerts(inputs: AlertInputs) -> tuple[Alert, ...]:
    threshold = "dataset/qc.py::QC_FLATLINE; config [qc].flatline_minutes"
    needed = {"channel", "flatline", "active_flatline"}
    if inputs.qc.is_empty() or not needed <= set(inputs.qc.columns):
        return (_not_evaluable("B", "stuck-sensor", "no active QC state", threshold),)
    flat = inputs.qc.filter(pl.col("active_flatline"))
    if flat.is_empty():
        return ()
    detail = ", ".join(
        f"{row['channel']} ({row['flatline']} flagged samples in history)"
        for row in flat.iter_rows(named=True)
    )
    return (
        Alert(
            severity="amber",
            zone="B",
            panel_id="stuck-sensor",
            message=f"flatline-flagged channels (possible stuck sensor): {detail}",
            threshold=threshold,
        ),
    )


def _drift_alerts(inputs: AlertInputs) -> tuple[Alert, ...]:
    threshold = "reports/drift.py::_FAST_Z = 6.0, Page-Hinkley lambda floor 25.0"
    if inputs.drift is None:
        return (
            _not_evaluable("B", "provider-drifting", "no drift artifact", threshold),
        )
    alarms = inputs.drift.get("alarms")
    if not isinstance(alarms, list) or not alarms:
        return ()
    alerts: list[Alert] = []
    for alarm in alarms:
        if not isinstance(alarm, Mapping):
            continue
        tier = str(alarm.get("tier", ""))
        alerts.append(
            Alert(
                severity="red" if tier == "residual" else "amber",
                zone="B",
                panel_id="provider-drifting",
                message=(
                    f"{alarm.get('source')} [{alarm.get('variable')}"
                    f" {alarm.get('lead_bucket')}] {tier} drift: "
                    f"{alarm.get('detail')}"
                ),
                threshold=threshold,
            )
        )
    return tuple(alerts)


def _bias_alerts(inputs: AlertInputs) -> tuple[Alert, ...]:
    threshold = "reports/leaderboard.py::CONSUMER_TOLERANCES per variable"
    board = inputs.board
    needed = {"variable", "method_id", "lead_bucket", "bias", "n"}
    if board.is_empty() or not needed <= set(board.columns):
        return (
            _not_evaluable("D", "grounding-bias", "no leaderboard rows", threshold),
        )
    alerts: list[Alert] = []
    for row in board.iter_rows(named=True):
        tolerance = CONSUMER_TOLERANCES.get(str(row["variable"]))
        bias = row["bias"]
        if (
            tolerance is None
            or bias is None
            or row["n"] is None
            or row["n"] < _MIN_BIAS_SAMPLES
            or abs(bias) <= tolerance
        ):
            continue
        alerts.append(
            Alert(
                severity="amber",
                zone="D",
                panel_id="grounding-bias",
                message=(
                    f"{row['method_id']} [{row['variable']} {row['lead_bucket']}] "
                    f"bias {bias:+.2f} exceeds the consumer tolerance "
                    f"{tolerance:.2f} — the §4.1 grounding-tilt signature"
                ),
                threshold=threshold,
            )
        )
    return tuple(alerts)


def _baseline_alerts(inputs: AlertInputs) -> tuple[Alert, ...]:
    threshold = (
        "structural heuristic, no config knob: climatology beating "
        "best_provider at the shortest lead is a leakage signature"
    )
    board = inputs.board
    needed = {"product", "variable", "method_id", "lead_bucket", "mae"}
    if board.is_empty() or not needed <= set(board.columns):
        return (
            _not_evaluable(
                "D", "baseline-implausible", "no leaderboard rows", threshold
            ),
        )
    alerts: list[Alert] = []
    keys = board.select("product", "variable").unique().to_dicts()
    for key in keys:
        group = board.filter(
            (pl.col("product") == key["product"])
            & (pl.col("variable") == key["variable"])
        )
        available = set(group["lead_bucket"].drop_nulls().to_list())
        order = (
            DAILY_BUCKET_LABELS if key["product"] == "daily" else HOURLY_BUCKET_LABELS
        )
        shortest = next((label for label in order if label in available), None)
        if shortest is None:
            continue
        group = group.filter(pl.col("lead_bucket") == shortest)
        mae = {
            row["method_id"]: row["mae"]
            for row in group.iter_rows(named=True)
            if row["mae"] is not None
        }
        climatology = mae.get("climatology")
        reference = mae.get("best_provider")
        if climatology is None or reference is None or climatology >= reference:
            continue
        alerts.append(
            Alert(
                severity="amber",
                zone="D",
                panel_id="baseline-implausible",
                message=(
                    f"climatology MAE {climatology:.2f} beats best_provider "
                    f"{reference:.2f} for {key['variable']} {shortest} — "
                    "check the baseline floor before trusting anything above it"
                ),
                threshold=threshold,
            )
        )
    return tuple(alerts)


def _argmax_source(state: Mapping[str, object]) -> dict[str, str]:
    """Per lead bucket, the source currently carrying the most weight."""
    sources = state.get("sources")
    buckets = state.get("buckets")
    if not isinstance(sources, list) or not isinstance(buckets, Mapping):
        return {}
    leaders: dict[str, str] = {}
    for label, bucket in buckets.items():
        weights = bucket.get("weights") if isinstance(bucket, Mapping) else None
        if not isinstance(weights, list) or len(weights) != len(sources):
            continue
        leaders[str(label)] = str(sources[weights.index(max(weights))])
    return leaders


def _leader_sequences(window: pl.DataFrame) -> dict[str, list[tuple[datetime, str]]]:
    """Per-label ``(issue_time, leading source)`` samples, oldest first."""
    sequences: dict[str, list[tuple[datetime, str]]] = {}
    for row in window.iter_rows(named=True):
        try:
            leaders = _argmax_source(json.loads(row["state_json"]))
        except (TypeError, ValueError):
            continue
        for label, leader in leaders.items():
            sequences.setdefault(label, []).append((row["issue_time"], leader))
    return sequences


def _held_leaders(
    samples: list[tuple[datetime, str]], min_hold: timedelta
) -> list[str]:
    """Leaders that actually held ``min_hold``, in order, brief crossings dropped.

    A regime is measured only across actual observations. The newest sample
    contributes no speculative future interval, so a last-sample crossing stays
    pending until another snapshot observes that leader.
    """
    if len(samples) < 2:
        return [leader for _moment, leader in samples]
    starts: list[tuple[datetime, str]] = []
    for moment, leader in samples:
        if not starts or starts[-1][1] != leader:
            starts.append((moment, leader))
    horizon = samples[-1][0]
    held: list[str] = []
    for index, (start, leader) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else horizon
        if end - start >= min_hold and (not held or held[-1] != leader):
            held.append(leader)
    return held


def _confirmed_transitions(
    window: pl.DataFrame,
) -> tuple[bool, dict[str, tuple[str, str, int]]]:
    """Latest sustained leader flip per label, with how many flips held.

    ``_argmax_source`` crosses whenever two experts' weights are near-tied,
    and at the documented 10-minute cadence a 3-day window holds several
    hundred samples — so counting every crossing would alert on arithmetic
    noise. A leader must hold ``_SWAP_MIN_HOLD_DAYS`` to count as a regime.
    """
    min_hold = timedelta(days=_SWAP_MIN_HOLD_DAYS)
    comparable = False
    transitions: dict[str, tuple[str, str, int]] = {}
    for label, samples in _leader_sequences(window).items():
        if len(samples) < 2:
            continue
        comparable = True
        held = _held_leaders(samples, min_hold)
        if flips := [(a, b) for a, b in pairwise(held) if a != b]:
            previous, leader = flips[-1]
            transitions[label] = (previous, leader, len(flips))
    return comparable, transitions


def _backend_swap_alerts(inputs: AlertInputs) -> tuple[Alert, ...]:
    threshold = (
        f"leader change held {_SWAP_MIN_HOLD_DAYS:.1f}d+ across a "
        f"{_SWAP_WINDOW_DAYS:.0f}d window (reports/drift.py::_FAST_WINDOW_DAYS); "
        "expert weights from artifacts/observability/history.parquet"
    )
    history = inputs.observability_history
    if history.is_empty() or "state_json" not in history.columns:
        return (
            _not_evaluable(
                "E", "backend-swap", "no expert-weight trajectory yet", threshold
            ),
        )
    alerts: list[Alert] = []
    comparable = False
    for _key, group in history.sort("issue_time").group_by(
        ["method_id", "product", "variable"], maintain_order=True
    ):
        newest = group.row(group.height - 1, named=True)
        edge = newest["issue_time"] - timedelta(days=_SWAP_WINDOW_DAYS)
        window = group.filter(pl.col("issue_time") >= edge)
        group_comparable, transitions = _confirmed_transitions(window)
        comparable = comparable or group_comparable
        for label, (previous, leader, flips) in transitions.items():
            flapping = f", {flips} times" if flips > 1 else ""
            alerts.append(
                Alert(
                    severity="amber",
                    zone="E",
                    panel_id="backend-swap",
                    message=(
                        f"{newest['method_id']} [{newest['product']}."
                        f"{newest['variable']} {label}] leading expert flipped "
                        f"{previous} -> {leader} within "
                        f"{_SWAP_WINDOW_DAYS:.0f}d{flapping} — "
                        "possible provider regime change"
                    ),
                    threshold=threshold,
                )
            )
    if not comparable and not alerts:
        return (
            _not_evaluable(
                "E",
                "backend-swap",
                "trajectory spans less than the comparison window",
                threshold,
            ),
        )
    return tuple(alerts)


def _divergence_alerts(inputs: AlertInputs) -> tuple[Alert, ...]:
    promotion = inputs.config.promotion
    threshold = (
        f"config [promotion].live_gap_factor = {promotion.live_gap_factor}, "
        f"min_live_n = {promotion.min_live_n}"
    )
    live = inputs.live_vs_backtest
    needed = {"product", "variable", "method_id", "n", "live_mae", "backtest_mae"}
    if live.is_empty() or not needed <= set(live.columns):
        return (
            _not_evaluable(
                "F", "serving-diverged", "no realized served forecasts", threshold
            ),
        )
    alerts: list[Alert] = []
    for row in live.iter_rows(named=True):
        if (
            row["live_mae"] is None
            or row["backtest_mae"] is None
            or row["n"] is None
            or row["n"] < promotion.min_live_n
            or row["live_mae"] <= promotion.live_gap_factor * row["backtest_mae"]
        ):
            continue
        alerts.append(
            Alert(
                severity="red",
                zone="F",
                panel_id="serving-diverged",
                message=(
                    f"{row['method_id']} [{row['product']}.{row['variable']}] "
                    f"live MAE {row['live_mae']:.2f} vs backtest "
                    f"{row['backtest_mae']:.2f} (n={row['n']}) — the serving "
                    "path has diverged from what the backtest promised"
                ),
                threshold=threshold,
            )
        )
    return tuple(alerts)


def _lineage_alerts(inputs: AlertInputs) -> tuple[Alert, ...]:
    threshold = "manifest.fingerprint vs release dataset/config fingerprints"
    if not inputs.releases:
        return (
            _not_evaluable(
                "F", "artifacts-stale", "no release has been promoted", threshold
            ),
        )
    manifest_print = (
        str(inputs.manifest.get("fingerprint", "unknown"))
        if inputs.manifest
        else "unknown"
    )
    newest = max(inputs.releases, key=lambda r: str(r.get("promoted_at", "")))
    alerts: list[Alert] = []
    if str(newest.get("dataset_fingerprint")) != manifest_print:
        alerts.append(
            Alert(
                severity="amber",
                zone="F",
                panel_id="artifacts-stale",
                message=(
                    "the newest release was promoted against dataset "
                    f"{newest.get('dataset_fingerprint')} but the manifest is "
                    f"{manifest_print}; a rebuild invalidated promoted evidence"
                ),
                threshold=threshold,
            )
        )
    if str(newest.get("config_fingerprint")) != config_fingerprint(inputs.config):
        alerts.append(
            Alert(
                severity="amber",
                zone="F",
                panel_id="artifacts-stale",
                message=(
                    "the newest release was promoted under a different config "
                    "fingerprint; re-run `backtest --source live` then `report`"
                ),
                threshold=threshold,
            )
        )
    return tuple(alerts)


def _archive_alerts(inputs: AlertInputs) -> tuple[Alert, ...]:
    max_age = inputs.config.forecasts.max_forecast_age_hours
    threshold = (
        f"config [forecasts].max_forecast_age_hours = {max_age} "
        "(snapshot cadence bound)"
    )
    matrix = inputs.hourly_matrix
    if matrix.is_empty() or "issue_time" not in matrix.columns:
        return (_not_evaluable("C", "archive-stalled", "no live matrix", threshold),)
    newest = matrix["issue_time"].max()
    if not isinstance(newest, datetime):
        return (_not_evaluable("C", "archive-stalled", "no live matrix", threshold),)
    lag_hours = (inputs.now - newest).total_seconds() / 3600
    if lag_hours <= max_age:
        return ()
    return (
        Alert(
            severity="red" if lag_hours > 3 * max_age else "amber",
            zone="C",
            panel_id="archive-stalled",
            message=(
                f"archive growth stalled: newest snapshot is {lag_hours:.1f}h old; "
                "every missed snapshot is training data that can never be recovered"
            ),
            threshold=threshold,
        ),
    )


def _manifest_sources(manifest: Mapping[str, object] | None) -> tuple[str, ...]:
    if manifest is None:
        return ()
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        return ()
    return tuple(str(source) for source in sources)


def _silent_empty_alerts(inputs: AlertInputs) -> tuple[Alert, ...]:
    from grounded_weather_forecast.dataset.providers import (  # noqa: PLC0415
        LOCATION_TOLERANCE,
    )

    threshold = (
        "manifest sources/snapshots/file rows; "
        f"dataset/providers.py::LOCATION_TOLERANCE = {LOCATION_TOLERANCE}"
    )
    if inputs.manifest is None:
        return (
            Alert(
                severity="red",
                zone="A",
                panel_id="silent-empty",
                message="no dataset manifest: `build-dataset` has never succeeded",
                threshold=threshold,
            ),
        )
    alerts: list[Alert] = []
    if not _manifest_sources(inputs.manifest):
        alerts.append(
            Alert(
                severity="red",
                zone="A",
                panel_id="silent-empty",
                message="manifest lists zero sources — the archive is empty",
                threshold=threshold,
            )
        )
    if inputs.manifest.get("snapshots") in (0, None):
        alerts.append(
            Alert(
                severity="red",
                zone="A",
                panel_id="silent-empty",
                message="manifest records zero snapshots — nothing was ingested",
                threshold=threshold,
            )
        )
    files = inputs.manifest.get("files")
    if isinstance(files, Mapping):
        empty = sorted(
            str(name)
            for name, info in files.items()
            if isinstance(info, Mapping) and info.get("rows") == 0
        )
        if empty:
            alerts.append(
                Alert(
                    severity="red",
                    zone="A",
                    panel_id="silent-empty",
                    message=f"zero-row dataset files: {', '.join(empty)}",
                    threshold=threshold,
                )
            )
    if inputs.archive_location is not None:
        station = inputs.config.station
        latitude = finite_number(inputs.archive_location[0])
        longitude = finite_number(inputs.archive_location[1])
        if latitude is None or longitude is None:
            alerts.append(
                Alert(
                    severity="red",
                    zone="A",
                    panel_id="silent-empty",
                    message=(
                        "forecast archive records a non-finite location "
                        f"{inputs.archive_location}; it cannot be checked against "
                        "the configured station"
                    ),
                    threshold=threshold,
                )
            )
        elif (
            abs(latitude - station.latitude) > LOCATION_TOLERANCE
            or abs(longitude - station.longitude) > LOCATION_TOLERANCE
        ):
            alerts.append(
                Alert(
                    severity="red",
                    zone="A",
                    panel_id="silent-empty",
                    message=(
                        f"archive location ({latitude}, {longitude}) does not match "
                        f"the configured station ({station.latitude}, "
                        f"{station.longitude}); forecasts silently filter to zero rows"
                    ),
                    threshold=threshold,
                )
            )
    return tuple(alerts)


def _unreadable_artifact_alerts(inputs: AlertInputs) -> tuple[Alert, ...]:
    """Artifacts that exist on disk but could not be read.

    Every loader falls back to "absent" on failure, which is right for a young
    archive and wrong for a corrupt or permission-denied file: the dashboard
    would render "not yet" over a broken deployment. Name them instead.
    """
    if not inputs.unreadable_artifacts:
        return ()
    return (
        Alert(
            severity="red",
            zone="G",
            panel_id="unreadable-artifacts",
            message=(
                "artifacts exist but could not be read: "
                f"{', '.join(inputs.unreadable_artifacts)}; the panels that "
                "depend on them are showing absence, not health"
            ),
            threshold="any artifact present on disk must be readable",
        ),
    )


def evaluate_alerts(inputs: AlertInputs) -> tuple[Alert, ...]:
    """Every alert family, most severe first; evaluable ones before info."""
    alerts = (
        *_silent_empty_alerts(inputs),
        *_station_alerts(inputs),
        *_provider_alerts(inputs),
        *_archive_alerts(inputs),
        *_truth_alerts(inputs),
        *_sensor_alerts(inputs),
        *_drift_alerts(inputs),
        *_bias_alerts(inputs),
        *_baseline_alerts(inputs),
        *_backend_swap_alerts(inputs),
        *_divergence_alerts(inputs),
        *_serving_alerts(inputs),
        *_lineage_alerts(inputs),
        *_unreadable_artifact_alerts(inputs),
    )
    return tuple(
        sorted(
            alerts,
            key=lambda alert: (
                _SEVERITY_ORDER[alert.severity],
                not alert.evaluable,
                alert.zone,
                alert.panel_id,
            ),
        )
    )
