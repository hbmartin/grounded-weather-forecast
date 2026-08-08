"""Serve-layer application of the configured quantile-recalibration mode.

The offline report (``reports.recalibration``) fits PIT remapping and CQR
margins side by side on a holdout and shows which one repairs each emitter;
``[predict].quantile_recalibration`` applies the chosen transform to
natively-emitted quantiles at serve time, routed per product — a bare
string routes every product, a ``[predict.quantile_recalibration]`` table
sets each product's mode (the A/B found daily favors cqr while hourly
favors raw, the documented norm across aggregation levels). Dressed rows are
untouched — their bands are already split-conformal, and correcting them
twice would double-count the residual archive.

Provenance guards mirror ``serve.dressing.ResidualDresser``: only live rows
whose dataset/config fingerprints and code identity match the running
process contribute, only the newest evaluation is pooled, and the first IO
or schema failure disables the transform for the run — recalibration can
never fail ``predict``.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from grounded_weather_forecast.blenders.protocol import finalize_quantiles
from grounded_weather_forecast.config import Config
from grounded_weather_forecast.contracts import (
    FloatArray,
    TruthSemantics,
    VariableSpec,
)
from grounded_weather_forecast.reports.recalibration import (
    MIN_FIT_ROWS,
    apply_cqr_margins,
    apply_pit_levels,
    fit_cqr_margins,
    fit_pit_levels,
)
from grounded_weather_forecast.serve.score_pool import collect_live_pool
from grounded_weather_forecast.serve.selection import Selection

MODES = ("none", "pit", "cqr")

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
        "y_true",
        "quantiles_json",
        "quantile_levels_json",
    }
)
_POOL_COLUMNS = (
    "method_id",
    "lead_bucket",
    "evaluation_id",
    "evaluation_created_at",
    "y_true",
    "quantiles_json",
    "quantile_levels_json",
)

# fitted transform: (levels, per-level parameter vector, fit scope)
_Fitted = tuple[tuple[float, ...], FloatArray, str]


@dataclass
class QuantileRecalibrator:
    """Per-run cache of fingerprint-compatible fitted level transforms."""

    scores_dir: Path
    product: str
    config: Config
    mode: str
    _pools: dict[tuple[str, str, frozenset[str]], pl.DataFrame | None] = field(
        default_factory=dict
    )
    _fits: dict[tuple[str, str, str, str | None], _Fitted | None] = field(
        default_factory=dict
    )
    _disabled: bool = False

    def fitted_for(
        self,
        variable: VariableSpec,
        semantics: TruthSemantics,
        method_ids: frozenset[str],
        method_id: str,
        lead_bucket: str | None,
    ) -> _Fitted | None:
        if self._disabled:
            return None
        key = (variable.name, semantics.value, method_id, lead_bucket)
        if key not in self._fits:
            self._fits[key] = self._fit(
                variable, semantics, method_ids, method_id, lead_bucket
            )
        return self._fits[key]

    def _fit(
        self,
        variable: VariableSpec,
        semantics: TruthSemantics,
        method_ids: frozenset[str],
        method_id: str,
        lead_bucket: str | None,
    ) -> _Fitted | None:
        frame = self._pool(variable, semantics, method_ids)
        if frame is None or frame.is_empty():
            return None
        subset = frame.filter(pl.col("method_id") == method_id)
        scope = "bucket"
        rows = (
            subset.filter(pl.col("lead_bucket") == lead_bucket)
            if lead_bucket is not None
            else subset
        )
        cases = _newest_evaluation_cases(rows)
        if cases is None or cases[0].shape[0] < MIN_FIT_ROWS:
            scope = "variable"
            cases = _newest_evaluation_cases(subset)
        if cases is None or cases[0].shape[0] < MIN_FIT_ROWS:
            return None
        y, grids, levels = cases
        parameters = (
            fit_pit_levels(y, grids, levels)
            if self.mode == "pit"
            else fit_cqr_margins(y, grids, levels)
        )
        return (levels, parameters, scope)

    def _pool(
        self,
        variable: VariableSpec,
        semantics: TruthSemantics,
        method_ids: frozenset[str],
    ) -> pl.DataFrame | None:
        key = (variable.name, semantics.value, method_ids)
        if key not in self._pools:
            try:
                self._pools[key] = self._collect(variable, semantics, method_ids)
            except (OSError, ValueError, pl.exceptions.PolarsError):
                self._disabled = True
                self._pools[key] = None
        return self._pools[key]

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
            value_predicate=pl.col("y_true").is_not_null()
            & pl.col("quantiles_json").is_not_null(),
        )


def _newest_evaluation_cases(
    subset: pl.DataFrame,
) -> tuple[FloatArray, FloatArray, tuple[float, ...]] | None:
    if subset.is_empty():
        return None
    newest = subset.select(
        pl.col("evaluation_id").sort_by("evaluation_created_at").last()
    ).item()
    rows = subset.filter(pl.col("evaluation_id") == newest)
    levels = tuple(json.loads(rows["quantile_levels_json"][0]))
    if not levels:
        return None
    grids = np.asarray(
        [
            [np.nan if value is None else float(value) for value in json.loads(row)]
            for row in rows["quantiles_json"].to_list()
        ]
    )
    y = rows["y_true"].to_numpy().astype(np.float64)
    usable = np.isfinite(grids).all(axis=1) & np.isfinite(y)
    if not usable.any():
        return None
    return y[usable], grids[usable], levels


def _row_levels(quantiles: dict[str, float]) -> tuple[float, ...]:
    return tuple(sorted(float(level) for level in quantiles))


def recalibrate_variable(
    recalibrator: QuantileRecalibrator,
    variable: VariableSpec,
    semantics: TruthSemantics,
    chosen: "list[Selection]",
    lead_buckets: list[str | None],
    quantiles: list[dict[str, float]],
    existing_sources: tuple[str | None, ...] | None,
) -> tuple[str | None, ...]:
    """Transform native quantile rows in place; return provenance labels.

    Rows already labeled by dressing keep their label and values — dressed
    bands are split-conformal already and must not be corrected twice.
    """
    method_ids = frozenset(selection.method_id for selection in chosen)
    labels: list[str | None] = (
        list(existing_sources) if existing_sources is not None else [None] * len(chosen)
    )
    for row, (selection, bucket) in enumerate(zip(chosen, lead_buckets, strict=True)):
        if labels[row] is not None or not quantiles[row]:
            continue
        fitted = recalibrator.fitted_for(
            variable, semantics, method_ids, selection.method_id, bucket
        )
        if fitted is None:
            continue
        levels, parameters, scope = fitted
        row_levels = _row_levels(quantiles[row])
        if len(row_levels) != len(levels) or not np.allclose(row_levels, levels):
            continue
        values = np.asarray([quantiles[row][str(level)] for level in row_levels])[
            np.newaxis, :
        ]
        transformed = (
            apply_pit_levels(values, levels, parameters)
            if recalibrator.mode == "pit"
            else apply_cqr_margins(values, parameters)
        )
        finalized = finalize_quantiles(transformed, variable.kind, variable)
        quantiles[row] = {
            str(level): float(finalized[0, index])
            for index, level in enumerate(row_levels)
        }
        labels[row] = f"recalibrated_{recalibrator.mode}_{scope}"
    return tuple(labels)
