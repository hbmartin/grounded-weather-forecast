"""The scores frame: one row per (method, variable, test case).

All evaluation happens downstream of this frame via group_by; the engine never
declares winners itself. Provenance (live vs synthetic) travels with every row
and mixed frames are rejected on load unless explicitly allowed.
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from grounded_weather_forecast.contracts import FloatArray, MixedProvenanceError

if TYPE_CHECKING:
    from grounded_weather_forecast.evaluation import EvaluationRun

SCORES_SCHEMA: pl.Schema = pl.Schema(
    {
        "method_id": pl.String(),
        "variable": pl.String(),
        "product": pl.String(),
        "source_kind": pl.String(),
        "evaluation_id": pl.String(),
        "evaluation_created_at": pl.Datetime("us", "UTC"),
        "dataset_fingerprint": pl.String(),
        "source_set_json": pl.String(),
        "feature_set_json": pl.String(),
        "semantics": pl.String(),
        "code_version": pl.String(),
        "config_fingerprint": pl.String(),
        "window": pl.String(),
        "fold_origin": pl.Datetime("us", "UTC"),
        "issue_time": pl.Datetime("us", "UTC"),
        "valid_time": pl.Datetime("us", "UTC"),
        "lead_hours": pl.Float64(),
        "lead_bucket": pl.String(),
        "y_pred": pl.Float64(),
        "y_true": pl.Float64(),
        "quantile_levels_json": pl.String(),
        "quantiles_json": pl.String(),
    }
)


def empty_scores() -> pl.DataFrame:
    return pl.DataFrame(schema=SCORES_SCHEMA)


def stamped_scores(
    keys: pl.DataFrame,
    evaluation: "EvaluationRun",
    *,
    method_id: str,
    variable: str,
    product: str,
    source_kind: str,
    semantics: str,
    window: str,
    fold_origin: datetime,
    y_pred: FloatArray,
    y_true: FloatArray,
    quantile_levels: tuple[float, ...] | None = None,
    quantiles: FloatArray | None = None,
) -> pl.DataFrame:
    """One method-fold's predictions stamped into the full scores schema.

    ``keys`` carries issue_time/valid_time/lead_hours/lead_bucket; every
    runner (hourly/daily engine, minutely path runner) shares this stamping
    so a schema change cannot silently fork between products.
    """
    return keys.with_columns(
        pl.lit(method_id).alias("method_id"),
        pl.lit(variable).alias("variable"),
        pl.lit(product).alias("product"),
        pl.lit(source_kind).alias("source_kind"),
        pl.lit(evaluation.evaluation_id).alias("evaluation_id"),
        pl.lit(datetime.fromisoformat(evaluation.created_at)).alias(
            "evaluation_created_at"
        ),
        pl.lit(evaluation.dataset_fingerprint).alias("dataset_fingerprint"),
        pl.lit(json.dumps(evaluation.source_set)).alias("source_set_json"),
        pl.lit(json.dumps(evaluation.feature_set)).alias("feature_set_json"),
        pl.lit(semantics).alias("semantics"),
        pl.lit(evaluation.code_version).alias("code_version"),
        pl.lit(evaluation.config_fingerprint).alias("config_fingerprint"),
        pl.lit(window).alias("window"),
        pl.lit(fold_origin).alias("fold_origin"),
        pl.Series("y_pred", y_pred, dtype=pl.Float64).fill_nan(None).alias("y_pred"),
        pl.Series("y_true", y_true, dtype=pl.Float64).fill_nan(None).alias("y_true"),
        pl.lit(json.dumps(quantile_levels)).alias("quantile_levels_json"),
        pl.Series(
            "quantiles_json",
            [
                json.dumps(
                    [value if math.isfinite(value) else None for value in row.tolist()]
                )
                for row in quantiles
            ]
            if quantiles is not None
            else [None] * keys.height,
            dtype=pl.String,
        ),
    ).select(list(SCORES_SCHEMA))


def scores_path(
    directory: Path,
    product: str,
    source_kind: str,
    window: str | None = None,
    evaluation_id: str | None = None,
) -> Path:
    """Path that preserves distinct windows/evaluations instead of overwriting."""
    suffix = "_".join(part for part in (window, evaluation_id) if part)
    tail = f"_{suffix}" if suffix else ""
    return directory / f"scores_{product}_{source_kind}{tail}.parquet"


def write_scores(scores: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.write_parquet(path)


def load_scores(path: Path, *, allow_mixed: bool = False) -> pl.DataFrame:
    scores = pl.read_parquet(path)
    kinds = scores["source_kind"].unique().to_list()
    if len(kinds) > 1 and not allow_mixed:
        msg = f"scores at {path} mix source kinds {sorted(map(str, kinds))}"
        raise MixedProvenanceError(msg)
    return scores
