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
from typing import Literal

import polars as pl

from grounded_weather_forecast.config import Config
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
    threshold = (
        "config [forecasts].max_forecast_age_hours = "
        f"{inputs.config.forecasts.max_forecast_age_hours}; manifest.sources"
    )
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
    missing = [
        source
        for source in expected
        if not isinstance(row.get(f"age__{source}"), (int, float))
    ]
    aged = [
        source
        for source in expected
        if isinstance(age := row.get(f"age__{source}"), (int, float))
        and age > inputs.config.forecasts.max_forecast_age_hours
    ]
    alerts: list[Alert] = []
    if missing:
        alerts.append(
            Alert(
                severity="red" if len(missing) == len(expected) else "amber",
                zone="A",
                panel_id="provider-dropped",
                message=(
                    "providers absent from the newest snapshot: "
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
    floor = inputs.config.dataset.min_hour_coverage
    threshold = (
        f"config [dataset].min_hour_coverage = {floor}, "
        f"min_day_coverage = {inputs.config.dataset.min_day_coverage}"
    )
    frame = inputs.hourly_truth
    coverage_columns = [c for c in frame.columns if c.endswith("_cov")]
    if frame.is_empty() or not coverage_columns:
        return (_not_evaluable("B", "truth-thinning", "no hourly truth", threshold),)
    recent = frame.tail(24 * 7)
    thin: dict[str, float] = {}
    for column in coverage_columns:
        mean = recent[column].mean()
        if isinstance(mean, (int, float)) and float(mean) < floor:
            thin[column] = float(mean)
    if not thin:
        return ()
    detail = ", ".join(
        f"{column.removesuffix('_cov')}={value:.2f}"
        for column, value in sorted(thin.items())
    )
    return (
        Alert(
            severity="amber",
            zone="B",
            panel_id="truth-thinning",
            message=(
                f"trailing-week truth coverage below the floor: {detail}; "
                "affected hours are nulled out of training"
            ),
            threshold=threshold,
        ),
    )


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


def _backend_swap_alerts(inputs: AlertInputs) -> tuple[Alert, ...]:
    threshold = (
        f"leader change across a {_SWAP_WINDOW_DAYS:.0f}d window "
        "(reports/drift.py::_FAST_WINDOW_DAYS); expert weights from "
        "artifacts/observability/history.parquet"
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
        earlier = group.filter(pl.col("issue_time") <= edge)
        if earlier.is_empty():
            continue
        comparable = True
        baseline = earlier.row(earlier.height - 1, named=True)
        try:
            now_leaders = _argmax_source(json.loads(newest["state_json"]))
            then_leaders = _argmax_source(json.loads(baseline["state_json"]))
        except (TypeError, ValueError):
            continue
        for label, leader in now_leaders.items():
            previous = then_leaders.get(label)
            if previous is None or previous == leader:
                continue
            alerts.append(
                Alert(
                    severity="amber",
                    zone="E",
                    panel_id="backend-swap",
                    message=(
                        f"{newest['method_id']} [{newest['product']}."
                        f"{newest['variable']} {label}] leading expert flipped "
                        f"{previous} -> {leader} within {_SWAP_WINDOW_DAYS:.0f}d — "
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
        latitude, longitude = inputs.archive_location
        station = inputs.config.station
        if (
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
