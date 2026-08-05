"""Offline quantile-recalibration A/B: PIT remapping vs CQR margins.

The 2026-08-05 audit found nearly every quantile emitter fails coverage on
the live archive (2 of 311 hourly rows pass PIT). Two mutually exclusive
post-hoc repairs exist and this module fits and scores both, side by side,
from the stored scores alone — no backtest re-run — so evidence, not taste,
picks the serving mode (``[predict].quantile_recalibration``):

- ``pit`` (Kuleshov-style): learn the map from nominal to realized quantile
  levels from the PIT sample, then read the native grid at the adjusted
  levels. Interpolation cannot escape the grid's outermost levels, so it
  repairs shape errors but not severe under-dispersion.
- ``cqr`` (Romano-style): per-level additive conformal margins on the
  residual ``y - q_tau``. Margins can widen past the native grid, so it can
  repair severe under-dispersion but shifts every row by the same offset.

Leakage rule: fit rows strictly precede evaluation rows in ``valid_time``
(chronological split on unique times). Only the newest evaluation is used —
expanding windows re-score overlapping folds, so pooling across evaluations
double-counts cases (the ``serve.dressing`` rule).
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import polars as pl

from grounded_weather_forecast.contracts import FloatArray
from grounded_weather_forecast.metrics.probabilistic import (
    pinball_loss,
    pit_from_quantiles,
)
from grounded_weather_forecast.serve.dressing import corrected_error_quantiles

MIN_FIT_ROWS = 50
_MIN_EVAL_ROWS = 20
_FIT_FRACTION = 0.7
_CASE_KEYS = ("product", "variable", "semantics", "lead_bucket", "method_id")

_REPORT_SCHEMA: dict[str, pl.DataType] = {
    "product": pl.String(),
    "variable": pl.String(),
    "semantics": pl.String(),
    "lead_bucket": pl.String(),
    "method_id": pl.String(),
    "fit_scope": pl.String(),
    "n_fit": pl.Int64(),
    "n_eval": pl.Int64(),
    "transform": pl.String(),
    "coverage80": pl.Float64(),
    "coverage90": pl.Float64(),
    "pinball": pl.Float64(),
}


@dataclass(frozen=True)
class QuantileCase:
    """One (slice x method) block of scored quantile forecasts, time-sorted."""

    keys: dict[str, str | None]
    valid_time_us: np.ndarray
    y: FloatArray
    grids: FloatArray
    levels: tuple[float, ...]


def _parse_grids(rows: list[str]) -> FloatArray:
    return np.asarray(
        [
            [np.nan if value is None else float(value) for value in json.loads(row)]
            for row in rows
        ]
    )


def newest_evaluation(scores: pl.DataFrame) -> pl.DataFrame:
    if scores.is_empty() or "evaluation_id" not in scores.columns:
        return scores
    newest = scores.select(
        pl.col("evaluation_id").sort_by("evaluation_created_at").last()
    ).item()
    return scores.filter(pl.col("evaluation_id") == newest)


def quantile_cases(scores: pl.DataFrame) -> Iterator[QuantileCase]:
    """Usable quantile rows of the newest evaluation, one case per slice."""
    if "quantiles_json" not in scores.columns:
        return
    frame = newest_evaluation(scores).drop_nulls(
        ["quantiles_json", "y_true", "valid_time"]
    )
    if frame.is_empty():
        return
    for keys, group in frame.sort("valid_time").group_by(
        list(_CASE_KEYS), maintain_order=True
    ):
        levels = tuple(json.loads(group["quantile_levels_json"][0]))
        if not levels:
            continue
        grids = _parse_grids(group["quantiles_json"].to_list())
        y = group["y_true"].to_numpy().astype(np.float64)
        usable = np.isfinite(grids).all(axis=1) & np.isfinite(y)
        if not usable.any():
            continue
        yield QuantileCase(
            keys=dict(
                zip(
                    _CASE_KEYS,
                    (None if k is None else str(k) for k in keys),
                    strict=False,
                )
            ),
            valid_time_us=group["valid_time"].cast(pl.Int64).to_numpy()[usable],
            y=y[usable],
            grids=grids[usable],
            levels=levels,
        )


def fit_pit_levels(
    y: FloatArray, grids: FloatArray, levels: tuple[float, ...]
) -> FloatArray:
    """Adjusted levels tau' such that reading the grid at tau' calibrates."""
    pit = pit_from_quantiles(y, grids, levels)
    return np.clip(corrected_error_quantiles(pit, levels), 0.0, 1.0)


def apply_pit_levels(
    grids: FloatArray, levels: tuple[float, ...], adjusted: FloatArray
) -> FloatArray:
    level_array = np.asarray(levels)
    recalibrated = np.empty_like(grids)
    for row in range(grids.shape[0]):
        recalibrated[row] = np.interp(adjusted, level_array, grids[row])
    return np.sort(recalibrated, axis=1)


def fit_cqr_margins(
    y: FloatArray, grids: FloatArray, levels: tuple[float, ...]
) -> FloatArray:
    """Per-level additive margins: conservative tau-quantile of y - q_tau."""
    return np.asarray(
        [
            corrected_error_quantiles(y - grids[:, index], (level,))[0]
            for index, level in enumerate(levels)
        ]
    )


def apply_cqr_margins(grids: FloatArray, margins: FloatArray) -> FloatArray:
    return np.sort(grids + margins[np.newaxis, :], axis=1)


def _interval_coverage(
    y: FloatArray,
    grids: FloatArray,
    levels: tuple[float, ...],
    lower: float,
    upper: float,
) -> float | None:
    level_array = np.asarray(levels)
    lower_hits = np.flatnonzero(np.isclose(level_array, lower))
    upper_hits = np.flatnonzero(np.isclose(level_array, upper))
    if lower_hits.shape[0] == 0 or upper_hits.shape[0] == 0:
        return None
    low = grids[:, int(lower_hits[0])]
    high = grids[:, int(upper_hits[0])]
    return float(np.mean((y >= low) & (y <= high)))


def _metrics(
    y: FloatArray, grids: FloatArray, levels: tuple[float, ...]
) -> dict[str, float | None]:
    pinball = float(
        np.mean([pinball_loss(y, grids[:, i], level) for i, level in enumerate(levels)])
    )
    return {
        "coverage80": _interval_coverage(y, grids, levels, 0.1, 0.9),
        "coverage90": _interval_coverage(y, grids, levels, 0.05, 0.95),
        "pinball": pinball,
    }


def _split_index(valid_time_us: np.ndarray, fit_fraction: float) -> int:
    """First row of the evaluation era: a strict chronological time split."""
    unique_times = np.unique(valid_time_us)
    if unique_times.shape[0] < 2:
        return valid_time_us.shape[0]
    cutoff = unique_times[
        min(
            max(int(np.floor(unique_times.shape[0] * fit_fraction)), 1),
            unique_times.shape[0] - 1,
        )
    ]
    return int(np.searchsorted(valid_time_us, cutoff, side="left"))


def recalibration_report(
    scores: pl.DataFrame,
    *,
    fit_fraction: float = _FIT_FRACTION,
    min_fit_rows: int = MIN_FIT_ROWS,
) -> pl.DataFrame:
    """Raw vs pit vs cqr holdout calibration per quantile-emitting slice."""
    cases = list(quantile_cases(scores))
    by_variable: dict[tuple[str | None, ...], list[QuantileCase]] = {}
    for case in cases:
        scope_key = tuple(
            case.keys[key] for key in ("product", "variable", "semantics", "method_id")
        )
        by_variable.setdefault(scope_key, []).append(case)
    rows: list[dict[str, object]] = []
    for case in cases:
        split = _split_index(case.valid_time_us, fit_fraction)
        eval_y = case.y[split:]
        eval_grids = case.grids[split:]
        if eval_y.shape[0] < _MIN_EVAL_ROWS:
            continue
        fit_y, fit_grids = case.y[:split], case.grids[:split]
        fit_scope = "bucket"
        if fit_y.shape[0] < min_fit_rows:
            # variable-global fallback: pool the method's other buckets, still
            # strictly before this bucket's evaluation era
            cutoff = case.valid_time_us[split]
            scope_key = tuple(
                case.keys[key]
                for key in ("product", "variable", "semantics", "method_id")
            )
            pooled = [
                sibling
                for sibling in by_variable[scope_key]
                if sibling.levels == case.levels
            ]
            mask_parts = [(sibling.valid_time_us < cutoff) for sibling in pooled]
            fit_y = np.concatenate(
                [
                    sibling.y[mask]
                    for sibling, mask in zip(pooled, mask_parts, strict=True)
                ]
            )
            fit_grids = np.concatenate(
                [
                    sibling.grids[mask]
                    for sibling, mask in zip(pooled, mask_parts, strict=True)
                ]
            )
            fit_scope = "variable"
            if fit_y.shape[0] < min_fit_rows:
                continue
        adjusted = fit_pit_levels(fit_y, fit_grids, case.levels)
        margins = fit_cqr_margins(fit_y, fit_grids, case.levels)
        variants = {
            "raw": eval_grids,
            "pit": apply_pit_levels(eval_grids, case.levels, adjusted),
            "cqr": apply_cqr_margins(eval_grids, margins),
        }
        for transform, grids in variants.items():
            rows.append(
                {
                    **case.keys,
                    "fit_scope": fit_scope,
                    "n_fit": int(fit_y.shape[0]),
                    "n_eval": int(eval_y.shape[0]),
                    "transform": transform,
                    **_metrics(eval_y, grids, case.levels),
                }
            )
    if not rows:
        return pl.DataFrame(schema=_REPORT_SCHEMA)
    return pl.DataFrame(rows, schema_overrides=_REPORT_SCHEMA).sort(
        ["product", "variable", "lead_bucket", "method_id", "transform"]
    )
