"""Two-tier provider drift detection: consensus-fast, truth-slow.

A provider silently swapping its backend model is the event the online
experts exist for — and the trap is that truth-based detection lags by the
lead time (a 7-day forecast's error resolves a week late). So:

- **Fast tier** (issue time, no truth needed): each source's deviation from
  the cross-source consensus median. A backend swap is visible against the
  other providers within hours. Alarm on a robust z-score of the recent mean
  deviation against the source's own trailing baseline.
- **Slow tier** (truth-based confirmation): Page-Hinkley on the source's
  grounded-residual series — the sequential change detector that accumulates
  drift beyond a dead-band and alarms when the excursion exceeds lambda.

Alarms are written to a report section and ``artifacts/drift.json``; state
resets and automated down-weighting stay manual until alarm precision has a
track record (fixed share already gives graceful re-entry either way).
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from grounded_weather_forecast.contracts import (
    FloatArray,
    TruthSemantics,
    VariableSpec,
    fx_col,
    truth_col,
)
from grounded_weather_forecast.leads import hourly_bucket_expr

_FAST_WINDOW_DAYS = 3.0
_FAST_BASELINE_DAYS = 21.0
_FAST_Z = 6.0
_PH_DELTA = 0.1
_PH_LAMBDA_FLOOR = 25.0
_PH_LAMBDA_SCALE = 4.0  # a driftless walk's excursion range grows like sqrt(n)
_MIN_ROWS = 48


@dataclass(frozen=True, slots=True)
class DriftAlarm:
    source: str
    lead_bucket: str
    tier: str  # "consensus" | "residual"
    statistic: float
    detail: str


def _upward_page_hinkley(
    values: FloatArray, delta: float = _PH_DELTA, lam: float | None = None
) -> tuple[bool, float]:
    if values.shape[0] < 2:
        return False, 0.0
    if lam is None:
        lam = max(_PH_LAMBDA_FLOOR, _PH_LAMBDA_SCALE * float(np.sqrt(values.shape[0])))
    cumulative = 0.0
    minimum = 0.0
    maximum_excursion = 0.0
    running_mean = values[0]
    for index, value in enumerate(values[1:], start=2):
        running_mean += (value - running_mean) / index
        cumulative += value - running_mean - delta
        minimum = min(minimum, cumulative)
        maximum_excursion = max(maximum_excursion, cumulative - minimum)
    return maximum_excursion > lam, float(maximum_excursion)


def page_hinkley(
    values: FloatArray, delta: float = _PH_DELTA, lam: float | None = None
) -> tuple[bool, float]:
    """Two-sided sequential mean-shift detector.

    Values should be standardized (unit-scale residuals); ``delta`` is the
    dead-band. Running the upward statistic on both signs gives falling and
    rising provider bias equal treatment.
    """
    upward, upward_excursion = _upward_page_hinkley(values, delta, lam)
    downward, downward_excursion = _upward_page_hinkley(-values, delta, lam)
    return upward or downward, max(upward_excursion, downward_excursion)


def _with_lead_bucket(matrix: pl.DataFrame) -> pl.DataFrame:
    if "lead_bucket" in matrix.columns:
        return matrix
    return matrix.with_columns(
        hourly_bucket_expr(pl.col("lead_hours")).alias("lead_bucket")
    )


def _fast_deviations(
    matrix: pl.DataFrame, variable: VariableSpec
) -> pl.DataFrame | None:
    matrix = _with_lead_bucket(matrix)
    columns = [c for c in matrix.columns if c.startswith("fx__")]
    sources = sorted(
        {c.split("__")[1] for c in columns if c.endswith(f"__{variable.name}")}
    )
    if len(sources) < 4:  # a consensus needs a crowd
        return None
    frame = matrix.select(
        "issue_time",
        "lead_bucket",
        *(pl.col(fx_col(source, variable.name)).alias(source) for source in sources),
    )
    values = frame.select(sources).to_numpy().astype(np.float64)
    with np.errstate(invalid="ignore"):
        consensus = np.nanmedian(values, axis=1)
    deviations = values - consensus[:, np.newaxis]
    return (
        pl.DataFrame(
            {
                "issue_time": frame["issue_time"],
                "lead_bucket": frame["lead_bucket"],
            }
            | {source: deviations[:, index] for index, source in enumerate(sources)}
        )
        .group_by("issue_time", "lead_bucket")
        .agg(*(pl.col(source).mean() for source in sources))
        .sort("issue_time", "lead_bucket")
    )


def consensus_alarms(matrix: pl.DataFrame, variable: VariableSpec) -> list[DriftAlarm]:
    """Fast tier: recent deviation-from-consensus vs the trailing baseline."""
    deviations = _fast_deviations(matrix, variable)
    if deviations is None or deviations.height < _MIN_ROWS:
        return []
    alarms: list[DriftAlarm] = []
    sources = deviations.columns[2:]
    for bucket_key, bucket_frame in deviations.partition_by(
        "lead_bucket", as_dict=True
    ).items():
        lead_bucket = str(bucket_key[0])
        newest = bucket_frame["issue_time"].max()
        if not isinstance(newest, datetime):
            continue
        recent_edge = newest - timedelta(days=_FAST_WINDOW_DAYS)
        baseline_edge = recent_edge - timedelta(days=_FAST_BASELINE_DAYS)
        for source in sources:
            recent = (
                bucket_frame.filter(pl.col("issue_time") > recent_edge)[source]
                .drop_nulls()
                .to_numpy()
            )
            baseline = (
                bucket_frame.filter(
                    (pl.col("issue_time") <= recent_edge)
                    & (pl.col("issue_time") > baseline_edge)
                )[source]
                .drop_nulls()
                .to_numpy()
            )
            if recent.size < 4 or baseline.size < 24:
                continue
            center = float(np.median(baseline))
            scale = float(np.median(np.abs(baseline - center))) * 1.4826
            scale = max(scale, 1e-6)
            z = (float(np.mean(recent)) - center) / (scale / np.sqrt(recent.size))
            if abs(z) > _FAST_Z:
                alarms.append(
                    DriftAlarm(
                        source=source,
                        lead_bucket=lead_bucket,
                        tier="consensus",
                        statistic=round(float(z), 2),
                        detail=(
                            f"recent {_FAST_WINDOW_DAYS:.0f}d deviation from consensus "
                            f"shifted {float(np.mean(recent)) - center:+.2f} "
                            f"{variable.unit} vs its "
                            f"{_FAST_BASELINE_DAYS:.0f}d baseline"
                        ),
                    ),
                )
    return alarms


def residual_alarms(
    matrix: pl.DataFrame,
    variable: VariableSpec,
    semantics: TruthSemantics = TruthSemantics.INSTANTANEOUS,
) -> list[DriftAlarm]:
    """Slow tier: Page-Hinkley on each source's standardized residuals."""
    truth_column = (
        truth_col(variable.name, semantics)
        if variable.has_dual_semantics
        else truth_col(variable.name)
    )
    if truth_column not in matrix.columns:
        return []
    alarms: list[DriftAlarm] = []
    sources = sorted(
        {
            c.split("__")[1]
            for c in matrix.columns
            if c.startswith("fx__") and c.endswith(f"__{variable.name}")
        }
    )
    scored = (
        _with_lead_bucket(matrix)
        .drop_nulls(truth_column)
        .sort("issue_time", "valid_time")
    )
    for source in sources:
        column = fx_col(source, variable.name)
        if column not in scored.columns:
            continue
        issue_residuals = (
            scored.select(
                "issue_time",
                "lead_bucket",
                (pl.col(column) - pl.col(truth_column)).alias("residual"),
            )
            .drop_nulls("residual")
            .group_by("issue_time", "lead_bucket")
            .agg(pl.col("residual").mean())
            .sort("issue_time", "lead_bucket")
        )
        for bucket_key, bucket_frame in issue_residuals.partition_by(
            "lead_bucket", as_dict=True
        ).items():
            residuals = (
                bucket_frame["residual"].drop_nulls().to_numpy().astype(np.float64)
            )
            if residuals.shape[0] < _MIN_ROWS:
                continue
            center = float(np.median(residuals))
            scale = float(np.median(np.abs(residuals - center))) * 1.4826
            standardized = (residuals - center) / max(scale, 1e-6)
            alarmed, excursion = page_hinkley(standardized)
            if alarmed:
                alarms.append(
                    DriftAlarm(
                        source=source,
                        lead_bucket=str(bucket_key[0]),
                        tier="residual",
                        statistic=round(excursion, 2),
                        detail=(
                            f"two-sided Page-Hinkley excursion {excursion:.1f} on "
                            f"{residuals.shape[0]} issue-level residuals"
                        ),
                    ),
                )
    return alarms


def drift_report(
    matrix: pl.DataFrame, variables: tuple[VariableSpec, ...]
) -> pl.DataFrame:
    """All alarms across both tiers, one row each; empty when all is calm."""
    rows = [
        {
            "variable": variable.name,
            "source": alarm.source,
            "lead_bucket": alarm.lead_bucket,
            "tier": alarm.tier,
            "statistic": alarm.statistic,
            "detail": alarm.detail,
        }
        for variable in variables
        for alarm in (
            *consensus_alarms(matrix, variable),
            *residual_alarms(matrix, variable),
        )
    ]
    return pl.DataFrame(
        rows,
        schema={
            "variable": pl.String,
            "source": pl.String,
            "lead_bucket": pl.String,
            "tier": pl.String,
            "statistic": pl.Float64,
            "detail": pl.String,
        },
    )


def write_drift_artifact(alarms: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 2, "alarms": alarms.to_dicts()}, indent=2),
        encoding="utf-8",
    )
