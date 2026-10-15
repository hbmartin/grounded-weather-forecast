"""The scores frame: one row per (method, variable, test case).

All evaluation happens downstream of this frame via group_by; the engine never
declares winners itself. Provenance (live vs synthetic) travels with every row
and mixed frames are rejected on load unless explicitly allowed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime
from itertools import chain, pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from grounded_weather_forecast.backtest.score_storage import score_parquet_destination
from grounded_weather_forecast.contracts import FloatArray, MixedProvenanceError

if TYPE_CHECKING:
    import pyarrow.parquet as pq

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

# Operational limits must not enter Config, whose fingerprint identifies fits.
SCORE_BATCH_ROWS = 65_536
CATEGORICAL_SCORE_COLUMNS = tuple(
    name
    for name, dtype in SCORES_SCHEMA.items()
    if dtype == pl.String and name != "quantiles_json"
)
COMPACT_SCORES_SCHEMA = pl.Schema(
    {
        name: pl.Categorical() if name in CATEGORICAL_SCORE_COLUMNS else dtype
        for name, dtype in SCORES_SCHEMA.items()
    }
)


@dataclass(frozen=True, slots=True)
class ScoreWriteResult:
    """Scalar-only outcome; never retains score or training buffers."""

    path: Path
    row_count: int
    evaluation_id: str | None


def empty_scores() -> pl.DataFrame:
    return pl.DataFrame(schema=SCORES_SCHEMA)


def _validate_score_shapes(
    row_count: int,
    y_pred: FloatArray,
    y_true: FloatArray,
    quantiles: FloatArray | None,
    quantile_levels: tuple[float, ...] | None,
) -> None:
    for name, values in (("y_pred", y_pred), ("y_true", y_true)):
        if values.shape != (row_count,):
            msg = f"{name} shape {values.shape} != expected {(row_count,)}"
            raise pl.exceptions.ShapeError(msg)
    if quantiles is not None and (
        quantiles.ndim != 2 or quantiles.shape[0] != row_count
    ):
        msg = (
            f"quantiles shape {quantiles.shape} != expected ({row_count}, n_quantiles)"
        )
        raise pl.exceptions.ShapeError(msg)
    if quantiles is None:
        if quantile_levels:
            raise ValueError("quantile_levels require quantiles")
        return
    if not quantile_levels:
        raise ValueError("quantiles require nonempty quantile_levels")
    if quantiles.shape[1] != len(quantile_levels):
        msg = f"quantiles shape {quantiles.shape} != expected {(row_count, len(quantile_levels))}"
        raise pl.exceptions.ShapeError(msg)
    if any(
        not math.isfinite(level) or not 0 < level < 1 for level in quantile_levels
    ) or any(left >= right for left, right in pairwise(quantile_levels)):
        raise ValueError(
            "quantile_levels must be finite, strictly increasing and within (0, 1)"
        )


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
    compact: bool = False,
) -> pl.DataFrame:
    """One method-fold's predictions stamped into the full scores schema.

    ``keys`` carries issue_time/valid_time/lead_hours/lead_bucket; every
    runner (hourly/daily engine, minutely path runner) shares this stamping
    so a schema change cannot silently fork between products.
    """
    _validate_score_shapes(keys.height, y_pred, y_true, quantiles, quantile_levels)
    scores = keys.with_columns(
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
    if compact:
        return scores.cast(COMPACT_SCORES_SCHEMA)
    return scores


def stamped_score_batches(
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
) -> Iterator[pl.DataFrame]:
    """Slice predictions before allocating score columns or quantile JSON."""
    _validate_score_shapes(keys.height, y_pred, y_true, quantiles, quantile_levels)
    for offset in range(0, keys.height, SCORE_BATCH_ROWS):
        selection = slice(offset, offset + SCORE_BATCH_ROWS)
        yield stamped_scores(
            keys.slice(offset, SCORE_BATCH_ROWS),
            evaluation,
            method_id=method_id,
            variable=variable,
            product=product,
            source_kind=source_kind,
            semantics=semantics,
            window=window,
            fold_origin=fold_origin,
            y_pred=y_pred[selection],
            y_true=y_true[selection],
            quantile_levels=quantile_levels,
            quantiles=quantiles[selection] if quantiles is not None else None,
            compact=True,
        )


@contextmanager
def _close_callback(close: Callable[[], object]) -> Iterator[None]:
    """Close on every exit without losing the original iteration/write error."""
    try:
        yield
    except (BaseException,) as error:  # noqa: B013 -- project tuple convention
        try:
            close()
        except (Exception,) as secondary:  # noqa: B013
            error.add_note(f"score stream cleanup failed: {secondary}")
        raise
    else:
        close()


@contextmanager
def _closing_batches(
    batches: Iterable[pl.DataFrame],
) -> Iterator[Iterator[pl.DataFrame]]:
    iterator = iter(batches)
    close = getattr(iterator, "close", lambda: None)
    with _close_callback(close):
        yield iterator


def _validated_batches(
    batches: Iterator[pl.DataFrame],
    product: str,
    source_kind: str,
    window: str,
) -> Iterator[pl.DataFrame]:
    evaluation_id: str | None = None
    for batch in batches:
        if batch.schema != COMPACT_SCORES_SCHEMA:
            raise ValueError("score batch must use COMPACT_SCORES_SCHEMA")
        if batch.is_empty():
            continue
        identities = (
            batch.select("evaluation_id", "product", "source_kind", "window")
            .unique()
            .rows()
        )
        if len(identities) != 1:
            raise ValueError("score batch mixes evaluation identities")
        identity = identities[0]
        if not isinstance(identity[0], str) or not identity[0]:
            raise ValueError("score batch has no evaluation ID")
        if evaluation_id is None:
            evaluation_id = identity[0]
        if identity != (evaluation_id, product, source_kind, window):
            raise ValueError("score batch identity does not match its destination")
        yield batch


def _write_buffer(writer: pq.ParquetWriter, frames: list[pl.DataFrame]) -> None:
    frame = pl.concat(frames, rechunk=False)
    writer.write_table(frame.to_arrow(), row_group_size=SCORE_BATCH_ROWS)
    frames.clear()


def _write_batches(writer: pq.ParquetWriter, batches: Iterable[pl.DataFrame]) -> int:
    frames: list[pl.DataFrame] = []
    buffered = total = 0
    for batch in batches:
        offset = 0
        while offset < batch.height:
            take = min(SCORE_BATCH_ROWS - buffered, batch.height - offset)
            frames.append(batch.slice(offset, take))
            buffered += take
            offset += take
            total += take
            if buffered == SCORE_BATCH_ROWS:
                _write_buffer(writer, frames)
                buffered = 0
    if frames:
        _write_buffer(writer, frames)
    return total


def write_score_batches(
    batches: Iterable[pl.DataFrame],
    directory: Path,
    product: str,
    source_kind: str,
    window: str,
) -> ScoreWriteResult:
    """Consume compact batches, close their producer, and atomically publish.

    Omitting Arrow's schema metadata lets existing readers see String columns,
    while dictionary arrays avoid expanding repeated metadata during encoding.
    Even an empty stream produces the historical empty score-file schema.
    """
    with ExitStack() as publication, _closing_batches(batches) as incoming:
        import pyarrow.parquet as pq  # noqa: PLC0415 -- writer-only dependency

        validated = _validated_batches(incoming, product, source_kind, window)
        first = next(validated, None)
        evaluation_id = str(first["evaluation_id"][0]) if first is not None else None
        path = scores_path(directory, product, source_kind, window, evaluation_id)
        temporary = publication.enter_context(score_parquet_destination(path))
        schema = pl.DataFrame(schema=COMPACT_SCORES_SCHEMA).to_arrow().schema
        writer = pq.ParquetWriter(
            temporary,
            schema,
            compression="zstd",
            compression_level=3,
            use_dictionary=list(CATEGORICAL_SCORE_COLUMNS),
            store_schema=False,
        )
        with _close_callback(writer.close):
            pending = chain((first,), validated) if first is not None else validated
            del first
            row_count = _write_batches(writer, pending)
    return ScoreWriteResult(path, row_count, evaluation_id)


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
    # Atomic so a concurrent reader (predict's lock-free scan) never sees a
    # half-written file.
    with score_parquet_destination(path) as temporary:
        scores.write_parquet(temporary)


def scan_score_file(path: Path, *, compact: bool = True) -> pl.LazyFrame:
    """Scan stored scores, categorizing repeated text when ``compact=True``.

    Feature descriptions can be kilobytes long and repeat tens of millions of
    times. Categorizing them in the streaming scan keeps one copy per value,
    rather than retaining the Parquet decoder's string buffers for every row.
    Numeric scores and per-case quantiles retain their original types.
    With ``compact=False``, all columns retain their stored types. The scan
    defaults to compact storage; ``load_scores`` defaults to stored types for
    compatibility with existing joins and concatenations.
    """
    scan = pl.scan_parquet(path, parallel="none", low_memory=True)
    if not compact:
        return scan
    categorical = [
        name
        for name, dtype in scan.collect_schema().items()
        if dtype == pl.String and name != "quantiles_json"
    ]
    return scan.with_columns(pl.col(categorical).cast(pl.Categorical))


def load_scores(
    path: Path, *, allow_mixed: bool = False, compact: bool = False
) -> pl.DataFrame:
    """Read score values with their stored types, or opt into compact categories."""
    scores = scan_score_file(path, compact=compact).collect(engine="streaming")
    kinds = scores["source_kind"].unique().to_list()
    if len(kinds) > 1 and not allow_mixed:
        msg = f"scores at {path} mix source kinds {sorted(map(str, kinds))}"
        raise MixedProvenanceError(msg)
    return scores
