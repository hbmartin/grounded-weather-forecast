"""Empirical truth-semantics calibration (ADR 0003).

Providers don't document whether an hourly value is an instantaneous state or
an hour-interval mean, and misalignment masquerades as provider bias. This
study correlates every source's forecasts against BOTH truth semantics, per
dual-semantics variable, and recommends the semantics preferred by the
weighted majority of sources. The result is stored as a JSON artifact and a
markdown report; the mixed-semantics default applies until data suffices.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from grounded_weather_forecast.contracts import (
    HOURLY_VARIABLES,
    TruthSemantics,
    fx_col,
    truth_col,
)
from grounded_weather_forecast.dataset.matrix import matrix_sources

_MIN_ROWS = 72


def _correlation(
    a: np.ndarray, b: np.ndarray, overlap: np.ndarray | None = None
) -> tuple[float | None, int]:
    overlap = ~(np.isnan(a) | np.isnan(b)) if overlap is None else overlap
    n = int(overlap.sum())
    if n < _MIN_ROWS:
        return None, n
    with np.errstate(invalid="ignore"):
        correlation = float(np.corrcoef(a[overlap], b[overlap])[0, 1])
    return (correlation if np.isfinite(correlation) else None), n


def alignment_study(matrix: pl.DataFrame) -> pl.DataFrame:
    """Per (variable, source): correlation with each semantics + preference."""
    rows: list[dict[str, object]] = []
    sources = matrix_sources(matrix)
    for spec in HOURLY_VARIABLES:
        if not spec.has_dual_semantics:
            continue
        inst_column = truth_col(spec.name, TruthSemantics.INSTANTANEOUS)
        mean_column = truth_col(spec.name, TruthSemantics.INTERVAL_MEAN)
        if inst_column not in matrix.columns or mean_column not in matrix.columns:
            continue
        inst_truth = matrix[inst_column].to_numpy()
        mean_truth = matrix[mean_column].to_numpy()
        for source in sources:
            column = fx_col(source, spec.name)
            if column not in matrix.columns:
                continue
            forecast = matrix[column].to_numpy()
            overlap = ~(
                np.isnan(forecast) | np.isnan(inst_truth) | np.isnan(mean_truth)
            )
            r_inst, n_inst = _correlation(forecast, inst_truth, overlap)
            r_mean, n_mean = _correlation(forecast, mean_truth, overlap)
            preferred: str | None = None
            if r_inst is not None and r_mean is not None:
                preferred = (
                    TruthSemantics.INSTANTANEOUS.value
                    if r_inst >= r_mean
                    else TruthSemantics.INTERVAL_MEAN.value
                )
            rows.append(
                {
                    "variable": spec.name,
                    "source": source,
                    "r_inst": r_inst,
                    "r_mean": r_mean,
                    "n": min(n_inst, n_mean),
                    "preferred": preferred,
                }
            )
    return pl.DataFrame(rows)


def recommended_semantics(study: pl.DataFrame) -> dict[str, str]:
    """Per variable: the n-weighted majority preference (default inst)."""
    recommendations: dict[str, str] = {}
    if study.is_empty():
        return recommendations
    for variable, group in study.partition_by("variable", as_dict=True).items():
        decided = group.drop_nulls("preferred")
        if decided.is_empty():
            recommendations[str(variable[0])] = TruthSemantics.INSTANTANEOUS.value
            continue
        weights = {
            str(row["preferred"]): int(row["n"])
            for row in decided.group_by("preferred")
            .agg(pl.col("n").sum())
            .iter_rows(named=True)
        }
        inst = TruthSemantics.INSTANTANEOUS.value
        mean = TruthSemantics.INTERVAL_MEAN.value
        recommendations[str(variable[0])] = (
            mean if weights.get(mean, 0) > weights.get(inst, 0) else inst
        )
    return recommendations


def write_alignment(study: pl.DataFrame, path: Path) -> dict[str, object]:
    """Persist the study + recommendations; returns the artifact dict."""
    sanitized = study.with_columns(
        *(
            pl.when(pl.col(column).is_finite())
            .then(pl.col(column))
            .otherwise(None)
            .alias(column)
            for column in ("r_inst", "r_mean")
            if column in study.columns
        )
    )
    artifact: dict[str, object] = {
        "recommended": recommended_semantics(study),
        "study": sanitized.to_dicts(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    return artifact


def load_recommended(path: Path) -> dict[str, str]:
    """Recommended semantics per variable; empty when no artifact exists."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    match raw.get("recommended"):
        case dict() as recommended:
            return {str(k): str(v) for k, v in recommended.items()}
        case _:
            return {}
