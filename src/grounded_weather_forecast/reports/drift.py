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
  Residuals are standardized by a floored robust scale first, and
  near-constant windows (zero-inflated precip in a dry month) are skipped
  with a ``skipped_degenerate`` note row instead of an alarm, so silence
  stays distinguishable from "checked and quiet".

Alarms are written to a report section and ``artifacts/drift.json``; state
resets and automated down-weighting stay manual until alarm precision has a
track record (fixed share already gives graceful re-entry either way).
"""

import json
from collections.abc import Mapping
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
# With the lambda floor at 25, no fewer than ~4 consecutive clipped samples
# can alarm — a lone monsoon burst cannot masquerade as a mean shift.
_PH_CLIP_SIGMA = 8.0
_MAD_TO_SIGMA = 1.4826  # sigma-equivalent MAD for a normal distribution
_SCALE_ABS_EPSILON = 1e-3  # metric milliunits: below any weather-scale signal
_SCALE_IQR_FRACTION = 0.1  # per-variable floor as a share of the window IQR
_DEGENERATE_NEAR_SHARE = 0.9  # residual share at the median that disables PH
_MIN_ROWS = 48

RESIDUAL_SKIPPED_TIER = "residual_skipped"
CONSENSUS_SKIPPED_TIER = "consensus_skipped"
COMMON_MODE_TIER = "common_mode"
# Per-source residual rows folded into a common-mode headline keep their
# data but leave the pageable-alert stream.
RESIDUAL_COMMON_TIER = "residual_common"
_COMMON_MODE_SHARE = 2.0 / 3.0
_MIN_COMMON_MODE_SOURCES = 3


@dataclass(frozen=True, slots=True)
class DriftAlarm:
    source: str
    lead_bucket: str
    tier: str  # "consensus" | "residual" | "residual_skipped"
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


@dataclass(frozen=True, slots=True)
class _ResidualScale:
    """Robust center/scale for one residual window, with degeneracy verdict."""

    center: float
    scale: float
    mad_sigma: float
    near_median_share: float

    @property
    def degenerate(self) -> bool:
        """Near-constant residuals cannot support a mean-shift statistic."""
        return (
            self.near_median_share > _DEGENERATE_NEAR_SHARE
            or self.mad_sigma < _SCALE_ABS_EPSILON
        )


def _residual_scale(residuals: FloatArray) -> _ResidualScale:
    """MAD scale with a floor: a fraction of the window IQR, then epsilon.

    The floor keeps a near-zero MAD (zero-inflated precip) from exploding
    standardized residuals; the IQR term makes the floor track each
    variable's own spread instead of a one-size absolute constant.
    """
    center = float(np.median(residuals))
    distances = np.abs(residuals - center)
    mad_sigma = float(np.median(distances)) * _MAD_TO_SIGMA
    quartiles = np.quantile(residuals, (0.25, 0.75))
    floor = max(
        _SCALE_IQR_FRACTION * float(quartiles[1] - quartiles[0]), _SCALE_ABS_EPSILON
    )
    return _ResidualScale(
        center=center,
        scale=max(mad_sigma, floor),
        mad_sigma=mad_sigma,
        near_median_share=float(np.mean(distances <= _SCALE_ABS_EPSILON)),
    )


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
            # The floored robust scale the residual tier already uses: a raw
            # MAD on a near-constant deviation series (zero-inflated precip)
            # produced z-statistics in the tens of thousands for +0.01 mm
            # shifts (future-work #22a).
            robust = _residual_scale(baseline)
            if robust.degenerate:
                alarms.append(
                    _skipped_degenerate_row(
                        source,
                        lead_bucket,
                        baseline,
                        robust,
                        tier=CONSENSUS_SKIPPED_TIER,
                    )
                )
                continue
            center = robust.center
            scale = robust.scale
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


def _skipped_degenerate_row(
    source: str,
    lead_bucket: str,
    residuals: FloatArray,
    robust: _ResidualScale,
    tier: str = RESIDUAL_SKIPPED_TIER,
) -> DriftAlarm:
    test_name = "Page-Hinkley" if tier == RESIDUAL_SKIPPED_TIER else "consensus z-test"
    return DriftAlarm(
        source=source,
        lead_bucket=lead_bucket,
        tier=tier,
        statistic=round(robust.near_median_share, 3),
        detail=(
            f"skipped_degenerate: {robust.near_median_share:.0%} of "
            f"{residuals.shape[0]} residuals within {_SCALE_ABS_EPSILON:g} of "
            f"the median (MAD-sigma {robust.mad_sigma:.3g}); {test_name} "
            "not run on a near-constant series"
        ),
    )


def _residual_row(
    source: str, lead_bucket: str, residuals: FloatArray
) -> DriftAlarm | None:
    """One report row for a bucket: alarm, skip note, or ``None`` when quiet."""
    robust = _residual_scale(residuals)
    if robust.degenerate:
        return _skipped_degenerate_row(source, lead_bucket, residuals, robust)
    standardized: FloatArray = np.clip(
        (residuals - robust.center) / robust.scale, -_PH_CLIP_SIGMA, _PH_CLIP_SIGMA
    )
    alarmed, excursion = page_hinkley(standardized)
    if not alarmed:
        return None
    return DriftAlarm(
        source=source,
        lead_bucket=lead_bucket,
        tier="residual",
        statistic=round(excursion, 2),
        detail=(
            f"two-sided Page-Hinkley excursion {excursion:.1f} on "
            f"{residuals.shape[0]} issue-level residuals"
        ),
    )


def residual_alarms(
    matrix: pl.DataFrame,
    variable: VariableSpec,
    semantics: TruthSemantics = TruthSemantics.INSTANTANEOUS,
) -> list[DriftAlarm]:
    """Slow tier: Page-Hinkley on each source's robustly standardized residuals.

    Residuals are standardized by a floored MAD scale and winsorized at
    ``_PH_CLIP_SIGMA`` before accumulation; near-constant windows are
    reported as ``skipped_degenerate`` rows instead of being scored.
    """
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
            if (
                row := _residual_row(source, str(bucket_key[0]), residuals)
            ) is not None:
                alarms.append(row)
    return alarms


def _variable_source_count(matrix: pl.DataFrame, variable_name: str) -> int:
    return len(
        {
            column.split("__")[1]
            for column in matrix.columns
            if column.startswith("fx__") and column.endswith(f"__{variable_name}")
        }
    )


def _common_mode_detail(truth_qc: Mapping[str, object] | None) -> str:
    """One interpretable line in place of a wall of per-source alarms."""
    if isinstance(truth_qc, Mapping):
        block = truth_qc.get("drift_verdict")
        if isinstance(block, Mapping):
            attribution = block.get("attribution")
            latched = bool(block.get("latched"))
            if latched and attribution == "station_drift":
                return (
                    "suspect station truth: the neighbor cross-check holds a "
                    "latched station-drift verdict — do not trust these "
                    "residual alarms as provider failures"
                )
            if attribution == "regime":
                return (
                    "regime event: the neighbor network shifted too — expect "
                    "NWP to catch up and the alarms to clear"
                )
            if block.get("exceeded") is False:
                return (
                    "neighbor cross-check is clean — a shared forecast-model "
                    "regression or a local regime NWP misses; watch for "
                    "reversion (regimes revert, drifts persist)"
                )
    return (
        "no neighbor cross-check verdict available to arbitrate — run "
        "truth-qc for attribution"
    )


def _collapse_common_mode(
    rows: list[dict[str, object]],
    matrix: pl.DataFrame,
    truth_qc: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    """Fold per-source residual walls into one attributed headline per variable.

    When most sources alarm on the same variable at once, the per-source
    rows say nothing about which source broke — the common factor is the
    verifying station or the regional regime, and the truth-QC neighbor
    verdict is the arbiter (future-work #24's cross-method invariant).
    """
    by_variable: dict[str, set[str]] = {}
    for row in rows:
        if row["tier"] == "residual":
            by_variable.setdefault(str(row["variable"]), set()).add(str(row["source"]))
    collapsed = list(rows)
    for variable_name, alarmed in sorted(by_variable.items()):
        n_sources = _variable_source_count(matrix, variable_name)
        if n_sources < _MIN_COMMON_MODE_SOURCES:
            continue
        if len(alarmed) / n_sources < _COMMON_MODE_SHARE:
            continue
        for row in collapsed:
            if row["variable"] == variable_name and row["tier"] == "residual":
                row["tier"] = RESIDUAL_COMMON_TIER
        collapsed.insert(
            0,
            {
                "variable": variable_name,
                "source": "(common)",
                "lead_bucket": "(all)",
                "tier": COMMON_MODE_TIER,
                "statistic": float(len(alarmed)),
                "detail": (
                    f"{len(alarmed)}/{n_sources} sources threw residual "
                    f"alarms together; {_common_mode_detail(truth_qc)}"
                ),
            },
        )
    return collapsed


def drift_report(
    matrix: pl.DataFrame,
    variables: tuple[VariableSpec, ...],
    truth_qc: Mapping[str, object] | None = None,
) -> pl.DataFrame:
    """All alarms and skip notes across both tiers, one row each.

    Empty only when every evaluated bucket ran and stayed quiet — degenerate
    buckets still produce a ``residual_skipped`` row, so an operator can tell
    "checked but unscoreable" apart from silence. When most sources alarm on
    one variable simultaneously the per-source rows collapse into a single
    ``common_mode`` headline attributed by the truth-QC neighbor verdict.
    """
    rows: list[dict[str, object]] = [
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
    rows = _collapse_common_mode(rows, matrix, truth_qc)
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
    """Write alarm rows under ``alarms`` and skip notes under ``notes``.

    The split keeps ``skipped_degenerate`` rows out of the alert pipeline
    (every ``alarms`` entry becomes a pageable alert) while the dashboard —
    which treats absence as failure — still sees that the tier was checked.
    """
    is_note = pl.col("tier").is_in(
        [RESIDUAL_SKIPPED_TIER, CONSENSUS_SKIPPED_TIER, RESIDUAL_COMMON_TIER]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "alarms": alarms.filter(~is_note).to_dicts(),
                "notes": alarms.filter(is_note).to_dicts(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
