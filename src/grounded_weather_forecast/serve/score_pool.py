"""Fingerprint-guarded live score pooling shared by dressing and recalibration.

Only live rows whose dataset/config fingerprints and code identity match the
running process may contribute, and hourly pools are additionally scoped to a
single truth semantics — the provenance rules both consumers document.
"""

from pathlib import Path

import polars as pl

from grounded_weather_forecast.config import Config
from grounded_weather_forecast.contracts import TruthSemantics, VariableSpec
from grounded_weather_forecast.evaluation import (
    code_identity,
    config_fingerprint,
    dataset_fingerprint,
)


def collect_live_pool(
    scores_dir: Path,
    product: str,
    config: Config,
    variable: VariableSpec,
    semantics: TruthSemantics,
    method_ids: frozenset[str],
    *,
    required_columns: frozenset[str],
    pool_columns: tuple[str, ...],
    value_predicate: pl.Expr,
) -> pl.DataFrame | None:
    """Provenance-compatible live rows for one variable, or ``None`` if empty.

    ``value_predicate`` is the consumer's row guard — e.g. non-null point
    predictions for residual dressing, non-null quantile grids for
    recalibration.
    """
    current_dataset = dataset_fingerprint(config)
    current_config = config_fingerprint(config)
    current_code = code_identity()
    frames: list[pl.DataFrame] = []
    for path in sorted(scores_dir.glob(f"scores_{product}_live*.parquet")):
        lazy = pl.scan_parquet(path)
        if not set(lazy.collect_schema().names()) >= required_columns:
            continue
        lazy = lazy.filter(
            (pl.col("source_kind") == "live")
            & (pl.col("dataset_fingerprint") == current_dataset)
            & (pl.col("config_fingerprint") == current_config)
            & (pl.col("code_version") == current_code)
            & (pl.col("variable") == variable.name)
            & pl.col("method_id").is_in(sorted(method_ids))
            & value_predicate
        )
        if product == "hourly":
            lazy = lazy.filter(pl.col("semantics") == semantics.value)
        frames.append(lazy.select(pool_columns).collect())
    if not frames:
        return None
    return pl.concat(frames)
