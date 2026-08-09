"""Serve-layer quantile completion: dress point-only winners with residuals.

Only a handful of registered methods emit native quantiles, so most served
slices ship bare points. When the winning method for a slice is point-only,
this module attaches the empirical quantiles of that method's own live
backtest errors for the same (product, variable, lead bucket) — asymmetric
by construction, so a systematic bias lands inside the interval instead of
being hidden by a symmetric band. A finite-sample split-conformal rank rule
keeps thin cells honest; a bucket below the floor borrows the
variable-global pool, and when even that is thin the row stays undressed.

Provenance guards mirror ``serve.selection._compatible_scores``: only live
rows whose dataset/config fingerprints and code identity match the running
process may contribute, and only the newest evaluation is pooled — the
expanding backtest re-scores overlapping folds, so pooling across
evaluations would double-count cases. Dressing never raises out of
``predict``: the first IO or schema failure disables it for the run.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from grounded_weather_forecast.blenders.protocol import finalize_quantiles
from grounded_weather_forecast.config import Config
from grounded_weather_forecast.contracts import (
    FloatArray,
    TruthSemantics,
    VariableSpec,
)
from grounded_weather_forecast.serve.score_pool import collect_live_pool

if TYPE_CHECKING:
    from grounded_weather_forecast.serve.selection import Selection

# Conformal's grid: coverage80 needs 0.1/0.9 and coverage90 needs 0.05/0.95
# exactly, and six levels stay estimable at the pool floor where a 19-level
# grid would not.
DRESSING_LEVELS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.75, 0.9, 0.95)
MIN_POOL_ROWS = 24

_REQUIRED_COLUMNS = frozenset(
    {
        "source_kind",
        "dataset_fingerprint",
        "config_fingerprint",
        "code_version",
        "evaluation_id",
        "evaluation_created_at",
        "method_id",
        "variable",
        "semantics",
        "lead_bucket",
        "y_pred",
        "y_true",
    }
)
_POOL_COLUMNS = (
    "method_id",
    "lead_bucket",
    "evaluation_id",
    "evaluation_created_at",
    "y_pred",
    "y_true",
)


def corrected_error_quantiles(
    errors: FloatArray, levels: tuple[float, ...] = DRESSING_LEVELS
) -> FloatArray:
    """Finite-sample empirical quantiles of ``error = y_true - y_pred``.

    Upper levels take the split-conformal order statistic ``ceil((n+1)*tau)``
    clipped to ``n``; lower levels the mirrored ``floor((n+1)*tau)`` clipped
    to 1 — slightly conservative bands instead of the plain plug-in
    quantile's systematic under-coverage on small pools.
    """
    ordered = np.sort(errors)
    n = ordered.shape[0]
    ranks = [
        min(n, math.ceil((n + 1) * tau))
        if tau >= 0.5
        else max(1, math.floor((n + 1) * tau))
        for tau in levels
    ]
    return ordered[np.asarray(ranks) - 1]


@dataclass
class ResidualDresser:
    """Per-run cache of fingerprint-compatible live residual pools."""

    scores_dir: Path
    product: str
    config: Config
    _cache: dict[tuple[str, str, frozenset[str]], pl.DataFrame | None] = field(
        default_factory=dict
    )
    _disabled: bool = False

    def errors_for(
        self,
        variable: VariableSpec,
        semantics: TruthSemantics,
        method_ids: frozenset[str],
        method_id: str,
        lead_bucket: str | None,
    ) -> FloatArray | None:
        """This method's error pool for a bucket (or variable-global)."""
        if self._disabled:
            return None
        frame = self._frame(variable, semantics, method_ids)
        if frame is None or frame.is_empty():
            return None
        subset = frame.filter(pl.col("method_id") == method_id)
        if lead_bucket is not None:
            subset = subset.filter(pl.col("lead_bucket") == lead_bucket)
        errors = _newest_evaluation_errors(subset)
        if errors is None or errors.shape[0] < MIN_POOL_ROWS:
            return None
        return errors

    def _frame(
        self,
        variable: VariableSpec,
        semantics: TruthSemantics,
        method_ids: frozenset[str],
    ) -> pl.DataFrame | None:
        key = (variable.name, semantics.value, method_ids)
        if key not in self._cache:
            try:
                self._cache[key] = self._collect(variable, semantics, method_ids)
            except (OSError, ValueError, pl.exceptions.PolarsError):
                self._disabled = True
                self._cache[key] = None
        return self._cache[key]

    def _collect(
        self,
        variable: VariableSpec,
        semantics: TruthSemantics,
        method_ids: frozenset[str],
    ) -> pl.DataFrame | None:
        return collect_live_pool(
            self.scores_dir,
            self.product,
            self.config,
            variable,
            semantics,
            method_ids,
            required_columns=_REQUIRED_COLUMNS,
            pool_columns=_POOL_COLUMNS,
            value_predicate=pl.col("y_pred").is_not_null()
            & pl.col("y_true").is_not_null(),
        )


def _newest_evaluation_errors(subset: pl.DataFrame) -> FloatArray | None:
    if subset.is_empty():
        return None
    newest = subset.select(
        pl.col("evaluation_id").sort_by("evaluation_created_at").last()
    ).item()
    rows = subset.filter(pl.col("evaluation_id") == newest)
    errors = (rows["y_true"] - rows["y_pred"]).to_numpy().astype(np.float64)
    return errors[np.isfinite(errors)]


def dress_variable(
    dresser: ResidualDresser,
    variable: VariableSpec,
    semantics: TruthSemantics,
    chosen: "list[Selection]",
    lead_buckets: list[str | None],
    point: FloatArray,
    quantiles: list[dict[str, float]],
) -> tuple[str | None, ...]:
    """Fill empty per-row quantile dicts in place; return provenance labels.

    Rows with native quantiles are structurally untouched — only empty dicts
    are candidates. The label records which ladder rung supplied the pool.
    """
    sources: list[str | None] = [None] * len(quantiles)
    undressed: dict[tuple[str, str | None], list[int]] = {}
    for row, selection in enumerate(chosen):
        if quantiles[row] or not np.isfinite(point[row]):
            continue
        undressed.setdefault((selection.method_id, lead_buckets[row]), []).append(row)
    if not undressed:
        return tuple(sources)
    method_ids = frozenset(method_id for method_id, _ in undressed)
    for (method_id, bucket), rows in undressed.items():
        errors = dresser.errors_for(variable, semantics, method_ids, method_id, bucket)
        label = "dressed_bucket"
        if errors is None:
            errors = dresser.errors_for(
                variable, semantics, method_ids, method_id, None
            )
            label = "dressed_variable"
        if errors is None:
            continue
        offsets = corrected_error_quantiles(errors)
        row_indices = np.asarray(rows)
        bands = finalize_quantiles(
            point[row_indices][:, np.newaxis] + offsets[np.newaxis, :],
            variable.kind,
            variable,
        )
        for offset, row in enumerate(row_indices):
            quantiles[row] = {
                str(level): float(bands[offset, index])
                for index, level in enumerate(DRESSING_LEVELS)
            }
            sources[row] = label
    return tuple(sources)
