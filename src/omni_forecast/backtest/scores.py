"""The scores frame: one row per (method, variable, test case).

All evaluation happens downstream of this frame via group_by; the engine never
declares winners itself. Provenance (live vs synthetic) travels with every row
and mixed frames are rejected on load unless explicitly allowed.
"""

from pathlib import Path

import polars as pl

from omni_forecast.contracts import MixedProvenanceError

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
