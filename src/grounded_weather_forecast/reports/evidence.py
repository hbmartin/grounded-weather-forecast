"""Append-only quality-evidence ledgers under ``artifacts/history/``.

Every nightly ``report`` renders a point-in-time view and discards the
trend; the ledgers keep it. The quality-side files record, per cycle: the
leaderboard's quality metrics (plus a 14-day recent-window MAE so genuine
movement is not diluted by months of expanding-window history), selection
churn between consecutive releases, A/B verdict summaries (quantile
recalibration and promotion-gate agreement), e-process wealth snapshots,
and realized served quality against its backtest promise. The operations
side (collectors in ``reports/operations.py``) records pipeline freshness,
per-provider collector health, the collector-to-matrix build funnel,
config/code identity changes, and a catalog of every scores file — the
edges of the pipeline, where the two historical week-long silent failures
(the Jul 2026 truth hole and the predict plist misfire) lived.

Contracts, all inherited from the ``runs.py`` run ledger:

- **Idempotent**: appends anti-join on natural keys (``nulls_equal`` — a
  nullable ``release_id`` would otherwise duplicate forever), so re-running
  ``report`` — and the expanding backtest re-rendering old score files
  every night — appends nothing.
- **Bounded**: wall-clock age horizon plus a row-count burst backstop.
- **Forward-compatible**: older files load with missing columns null-filled.
- **Harmless**: a recorder can never fail the report; the corrupt-ledger
  trade (read empty, rewrite with fresh) is the documented telemetry cost.
- **No config keys**: paths derive from ``artifacts_dir`` and constants are
  module-level, so this module adds zero config-fingerprint churn.

Provenance columns (code version, config/dataset fingerprints, and the
e-process ``resets`` era) ride on every row so trends stay segmentable
across fingerprint resets instead of silently mixing incomparable eras.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
from filelock import Timeout

from grounded_weather_forecast.config import Config
from grounded_weather_forecast.evaluation import code_identity, config_fingerprint
from grounded_weather_forecast.metrics.probabilistic import (
    crps_ensemble,
    empirical_coverage,
)
from grounded_weather_forecast.storage import atomic_write_parquet, locked_path

if TYPE_CHECKING:
    from grounded_weather_forecast.reports.eprocess import EProcessStore

_LOCK_TIMEOUT_SECONDS = 5.0
RECENT_WINDOW_DAYS = 14
RETENTION_DAYS = 730
_MIN_RECENT_QUANTILE_ROWS = 20

_RECORDER_ERRORS = (OSError, ValueError, Timeout, pl.exceptions.PolarsError)


@dataclass(frozen=True, slots=True)
class LedgerSpec:
    """One ledger's identity: file name, schema, dedupe keys, bounds."""

    name: str
    schema: pl.Schema
    dedupe_keys: tuple[str, ...]
    max_rows: int
    time_column: str = "recorded_at"


QUALITY_SCHEMA = pl.Schema(
    {
        "recorded_at": pl.Datetime("us", "UTC"),
        "evaluation_id": pl.String(),
        "evaluation_created_at": pl.Datetime("us", "UTC"),
        "product": pl.String(),
        "source_kind": pl.String(),
        "variable": pl.String(),
        "truth_semantics": pl.String(),
        "lead_bucket": pl.String(),
        "method_id": pl.String(),
        "n": pl.Int64(),
        "n_valid_times": pl.Int64(),
        "coverage": pl.Float64(),
        "mae": pl.Float64(),
        "rmse": pl.Float64(),
        "bias": pl.Float64(),
        "coverage80": pl.Float64(),
        "coverage90": pl.Float64(),
        "crps": pl.Float64(),
        "pinball": pl.Float64(),
        "recent_mae": pl.Float64(),
        "recent_n": pl.Int64(),
        "recent_coverage80": pl.Float64(),
        "recent_coverage90": pl.Float64(),
        "recent_crps": pl.Float64(),
        "code_version": pl.String(),
        "config_fingerprint": pl.String(),
        "dataset_fingerprint": pl.String(),
    }
)

CHURN_SCHEMA = pl.Schema(
    {
        "recorded_at": pl.Datetime("us", "UTC"),
        "from_release_id": pl.String(),
        "to_release_id": pl.String(),
        "from_promoted_at": pl.Datetime("us", "UTC"),
        "to_promoted_at": pl.Datetime("us", "UTC"),
        "slice_key": pl.String(),
        "product": pl.String(),
        "variable": pl.String(),
        "lead_bucket": pl.String(),
        "from_method": pl.String(),
        "to_method": pl.String(),
        "changed": pl.Boolean(),
        "from_mae": pl.Float64(),
        "to_mae": pl.Float64(),
        "from_n": pl.Int64(),
        "to_n": pl.Int64(),
        "dataset_fingerprint": pl.String(),
        "config_fingerprint": pl.String(),
        "code_version": pl.String(),
    }
)

VERDICTS_SCHEMA = pl.Schema(
    {
        "recorded_at": pl.Datetime("us", "UTC"),
        "evaluation_id": pl.String(),
        "product": pl.String(),
        "source_kind": pl.String(),
        "name": pl.String(),
        "value": pl.Float64(),
        "code_version": pl.String(),
        "config_fingerprint": pl.String(),
        "dataset_fingerprint": pl.String(),
    }
)

EPROCESS_WEALTH_SCHEMA = pl.Schema(
    {
        "recorded_at": pl.Datetime("us", "UTC"),
        "product": pl.String(),
        "pair_key": pl.String(),
        "resets": pl.Int64(),
        "t": pl.Int64(),
        "log_e": pl.Float64(),
        "lam": pl.Float64(),
        "scale": pl.Float64(),
        "config_fingerprint": pl.String(),
        "code_version": pl.String(),
    }
)

SERVED_QUALITY_SCHEMA = pl.Schema(
    {
        "recorded_at": pl.Datetime("us", "UTC"),
        "as_of_date": pl.Date(),
        "product": pl.String(),
        "variable": pl.String(),
        "truth_semantics": pl.String(),
        "lead_bucket": pl.String(),
        "method_id": pl.String(),
        "release_id": pl.String(),
        "dataset_fingerprint": pl.String(),
        "n": pl.Int64(),
        "live_mae": pl.Float64(),
        "live_rmse": pl.Float64(),
        "live_bias": pl.Float64(),
        "backtest_mae": pl.Float64(),
        "mae_gap": pl.Float64(),
        "code_version": pl.String(),
        "config_fingerprint": pl.String(),
    }
)

QUALITY_LEDGER = LedgerSpec(
    name="quality",
    schema=QUALITY_SCHEMA,
    dedupe_keys=(
        "evaluation_id",
        "product",
        "source_kind",
        "variable",
        "truth_semantics",
        "lead_bucket",
        "method_id",
    ),
    max_rows=2_000_000,
)
CHURN_LEDGER = LedgerSpec(
    name="churn",
    schema=CHURN_SCHEMA,
    dedupe_keys=("from_release_id", "to_release_id", "slice_key"),
    max_rows=100_000,
)
VERDICTS_LEDGER = LedgerSpec(
    name="verdicts",
    schema=VERDICTS_SCHEMA,
    dedupe_keys=("evaluation_id", "product", "name"),
    max_rows=100_000,
)
EPROCESS_WEALTH_LEDGER = LedgerSpec(
    name="eprocess_wealth",
    schema=EPROCESS_WEALTH_SCHEMA,
    # `resets` in the key is load-bearing: the store zeroes wealth on a
    # config/code identity change, and a pair re-reaching an old ``t`` in the
    # new era must not be anti-joined away as a duplicate.
    dedupe_keys=("pair_key", "resets", "t"),
    max_rows=1_000_000,
)
SERVED_QUALITY_LEDGER = LedgerSpec(
    name="served_quality",
    schema=SERVED_QUALITY_SCHEMA,
    dedupe_keys=(
        "as_of_date",
        "product",
        "variable",
        "truth_semantics",
        "lead_bucket",
        "method_id",
        "release_id",
    ),
    max_rows=500_000,
)

PIPELINE_SCHEMA = pl.Schema(
    {
        "recorded_at": pl.Datetime("us", "UTC"),
        "as_of_date": pl.Date(),
        "truth_age_minutes": pl.Float64(),
        "truth_samples_24h": pl.Int64(),
        "collector_age_minutes": pl.Float64(),
        "collector_runs_24h": pl.Int64(),
        "provider_success_rate_24h": pl.Float64(),
        "served_history_age_minutes": pl.Float64(),
        "forecast_document_age_minutes": pl.Float64(),
        "predict_runs_24h": pl.Int64(),
        "failed_runs_24h": pl.Int64(),
        "alarms": pl.String(),
        "code_version": pl.String(),
        "config_fingerprint": pl.String(),
    }
)

PROVIDER_HEALTH_SCHEMA = pl.Schema(
    {
        "recorded_at": pl.Datetime("us", "UTC"),
        "as_of_date": pl.Date(),
        "provider": pl.String(),
        "runs_24h": pl.Int64(),
        "ok_24h": pl.Int64(),
        "success_rate": pl.Float64(),
        "median_latency_ms": pl.Float64(),
        "hourly_rows_24h": pl.Int64(),
        "daily_rows_24h": pl.Int64(),
        "max_hourly_lead_hours": pl.Float64(),
        "max_daily_lead_days": pl.Float64(),
        "code_version": pl.String(),
    }
)

# ``*_max_lead`` units follow the granularity: hours on "hourly" rows, days
# on "daily" rows. The native/path split exists because a daily value can
# reach the matrix two ways (a provider's own daily product vs an aggregate
# of its hourly path) with very different horizons — collapsing them once
# hid a 7-day native feed behind a 2-day path gate.
BUILD_FUNNEL_SCHEMA = pl.Schema(
    {
        "recorded_at": pl.Datetime("us", "UTC"),
        "as_of_date": pl.Date(),
        "granularity": pl.String(),
        "source": pl.String(),
        "collector_rows": pl.Int64(),
        "collector_max_lead": pl.Float64(),
        "long_rows": pl.Int64(),
        "long_max_lead": pl.Float64(),
        "matrix_rows": pl.Int64(),
        "matrix_max_lead": pl.Float64(),
        "matrix_native_max_lead": pl.Float64(),
        "matrix_path_max_lead": pl.Float64(),
        "code_version": pl.String(),
    }
)

CHANGES_SCHEMA = pl.Schema(
    {
        "recorded_at": pl.Datetime("us", "UTC"),
        "as_of_date": pl.Date(),
        "kind": pl.String(),
        "from_value": pl.String(),
        "to_value": pl.String(),
        "detail": pl.String(),
        "code_version": pl.String(),
    }
)

EVALUATIONS_SCHEMA = pl.Schema(
    {
        "recorded_at": pl.Datetime("us", "UTC"),
        "evaluation_id": pl.String(),
        "file_name": pl.String(),
        "product": pl.String(),
        "source_kind": pl.String(),
        "window": pl.String(),
        "rows": pl.Int64(),
        "n_methods": pl.Int64(),
        "n_folds": pl.Int64(),
        "fold_rows_min": pl.Int64(),
        "fold_rows_median": pl.Float64(),
        "issue_min": pl.Datetime("us", "UTC"),
        "issue_max": pl.Datetime("us", "UTC"),
        "file_size_mb": pl.Float64(),
        "code_version": pl.String(),
    }
)

PIPELINE_LEDGER = LedgerSpec(
    name="pipeline",
    schema=PIPELINE_SCHEMA,
    # One row per day, first writer wins: a manual afternoon report must not
    # shadow the scheduled morning snapshot the trend lines are built from.
    dedupe_keys=("as_of_date",),
    max_rows=10_000,
)
PROVIDER_HEALTH_LEDGER = LedgerSpec(
    name="provider_health",
    schema=PROVIDER_HEALTH_SCHEMA,
    dedupe_keys=("as_of_date", "provider"),
    max_rows=100_000,
)
BUILD_FUNNEL_LEDGER = LedgerSpec(
    name="build_funnel",
    schema=BUILD_FUNNEL_SCHEMA,
    dedupe_keys=("as_of_date", "granularity", "source"),
    max_rows=200_000,
)
CHANGES_LEDGER = LedgerSpec(
    name="changes",
    schema=CHANGES_SCHEMA,
    # ``as_of_date`` in the key is load-bearing: a flip A->B months after an
    # earlier A->B is a new event, not a duplicate.
    dedupe_keys=("as_of_date", "kind", "from_value", "to_value"),
    max_rows=10_000,
)
EVALUATIONS_LEDGER = LedgerSpec(
    name="evaluations",
    schema=EVALUATIONS_SCHEMA,
    # Scores files are write-once (a new evaluation mints a new id), so the
    # catalog row is immutable — and it outlives the file itself, which is
    # what makes pruning the 600+ MB scores directory safe.
    dedupe_keys=("evaluation_id",),
    max_rows=100_000,
)

LEDGERS: Mapping[str, LedgerSpec] = {
    spec.name: spec
    for spec in (
        QUALITY_LEDGER,
        CHURN_LEDGER,
        VERDICTS_LEDGER,
        EPROCESS_WEALTH_LEDGER,
        SERVED_QUALITY_LEDGER,
        PIPELINE_LEDGER,
        PROVIDER_HEALTH_LEDGER,
        BUILD_FUNNEL_LEDGER,
        CHANGES_LEDGER,
        EVALUATIONS_LEDGER,
    )
}


def ledger_path(config: Config, spec: LedgerSpec) -> Path:
    return config.artifacts_dir / "history" / f"{spec.name}.parquet"


def read_ledger(path: Path, schema: pl.Schema) -> pl.DataFrame:
    """Read and normalize a ledger, raising when an existing file is unusable."""
    if not path.exists():
        return pl.DataFrame(schema=schema)
    frame = pl.read_parquet(path)
    missing = [
        pl.lit(None, dtype=dtype).alias(column)
        for column, dtype in schema.items()
        if column not in frame.columns
    ]
    return (
        frame.with_columns(*missing).select(schema.names()).cast(schema, strict=False)
    )


def load_ledger(path: Path, schema: pl.Schema) -> pl.DataFrame:
    """Read a ledger; a corrupt file reads empty (the runs-ledger trade)."""
    try:
        return read_ledger(path, schema)
    except (OSError, pl.exceptions.PolarsError):
        return pl.DataFrame(schema=schema)


def prune_ledger(
    frame: pl.DataFrame, spec: LedgerSpec, *, now: datetime
) -> pl.DataFrame:
    """Bound by wall-clock age, then by row count as a burst backstop."""
    if frame.is_empty():
        return frame
    horizon = now - timedelta(days=RETENTION_DAYS)
    return frame.filter(
        pl.col(spec.time_column).is_null() | (pl.col(spec.time_column) >= horizon)
    ).tail(spec.max_rows)


def append_ledger(
    fresh: pl.DataFrame, path: Path, spec: LedgerSpec, *, now: datetime | None = None
) -> None:
    """Append novel rows only; a ledger failure never reaches the report."""
    try:
        if fresh.is_empty():
            return
        # Same forward-compatibility as read_ledger, on the write side: a
        # builder predating a schema column null-fills it instead of failing.
        missing = [
            pl.lit(None, dtype=dtype).alias(column)
            for column, dtype in spec.schema.items()
            if column not in fresh.columns
        ]
        fresh = (
            fresh.with_columns(*missing)
            .select(spec.schema.names())
            .cast(spec.schema, strict=False)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with locked_path(path, timeout=_LOCK_TIMEOUT_SECONDS):
            existing = load_ledger(path, spec.schema)
            novel = fresh.join(
                existing,
                on=list(spec.dedupe_keys),
                how="anti",
                nulls_equal=True,
            )
            atomic_write_parquet(
                prune_ledger(
                    pl.concat([existing, novel]),
                    spec,
                    now=now or datetime.now(tz=UTC),
                ),
                path,
            )
    except _RECORDER_ERRORS:
        return


def _first(scores: pl.DataFrame, column: str) -> object:
    return scores[column][0] if column in scores.columns else None


def _interval_coverage(
    y: np.ndarray,
    grids: np.ndarray,
    levels: list[float],
    lower: float,
    upper: float,
) -> float | None:
    lower_matches = np.flatnonzero(np.isclose(np.asarray(levels), lower))
    upper_matches = np.flatnonzero(np.isclose(np.asarray(levels), upper))
    if not lower_matches.size or not upper_matches.size:
        return None
    return empirical_coverage(
        y, grids[:, int(lower_matches[0])], grids[:, int(upper_matches[0])]
    )


def _recent_quantile_metrics(
    scores: pl.DataFrame, anchor: datetime | None
) -> pl.DataFrame:
    """Recent-window coverage and CRPS per slice x method, from stored grids.

    The pooled expanding-evaluation coverage is frozen by history: with
    hundreds of thousands of pre-repair rows, a repaired interval mechanism
    moves it by tenths of a point per day. The recent window is the only
    place a calibration change is visible on a useful timescale.
    """
    identity = ["product", "variable", "truth_semantics", "lead_bucket", "method_id"]
    empty = pl.DataFrame(
        schema={
            **dict.fromkeys(identity, pl.String()),
            "recent_coverage80": pl.Float64(),
            "recent_coverage90": pl.Float64(),
            "recent_crps": pl.Float64(),
        }
    )
    if anchor is None or {"quantiles_json", "quantile_levels_json"} - set(
        scores.columns
    ):
        return empty
    recent = scores.filter(
        pl.col("valid_time") > anchor - timedelta(days=RECENT_WINDOW_DAYS)
    ).drop_nulls(["quantiles_json", "quantile_levels_json", "y_true"])
    if recent.is_empty():
        return empty
    rows: list[dict[str, object]] = []
    group_keys = ["product", "variable", "semantics", "lead_bucket", "method_id"]
    for key, group in recent.group_by(group_keys):
        levels = [
            float(level) for level in json.loads(group["quantile_levels_json"][0])
        ]
        if not levels:
            continue
        grids = np.asarray(
            [
                [np.nan if value is None else float(value) for value in json.loads(row)]
                for row in group["quantiles_json"].to_list()
            ]
        )
        y = group["y_true"].to_numpy().astype(np.float64)
        usable = np.isfinite(grids).all(axis=1) & np.isfinite(y)
        if int(usable.sum()) < _MIN_RECENT_QUANTILE_ROWS:
            continue
        grids, y = grids[usable], y[usable]
        rows.append(
            {
                **dict(zip(identity, (str(part) for part in key), strict=True)),
                "recent_coverage80": _interval_coverage(y, grids, levels, 0.1, 0.9),
                "recent_coverage90": _interval_coverage(y, grids, levels, 0.05, 0.95),
                "recent_crps": crps_ensemble(y, np.sort(grids, axis=1)),
            }
        )
    if not rows:
        return empty
    return pl.DataFrame(rows, schema_overrides=dict(empty.schema)).select(empty.columns)


def quality_rows(
    scores: pl.DataFrame, board: pl.DataFrame, *, now: datetime
) -> pl.DataFrame:
    """One quality row per board slice x method, with the recent-window MAE.

    A scores file normally holds exactly one evaluation, so the in-scope
    board is reused verbatim; a pooled multi-evaluation frame is filtered to
    its newest evaluation and re-boarded (without references — this ledger
    records no skill columns) so metrics are never stamped with an
    evaluation id they did not come from.
    """
    from grounded_weather_forecast.reports.leaderboard import (  # noqa: PLC0415
        leaderboard,
    )
    from grounded_weather_forecast.reports.recalibration import (  # noqa: PLC0415
        newest_evaluation,
    )

    if scores.is_empty() or board.is_empty():
        return pl.DataFrame(schema=QUALITY_SCHEMA)
    if scores["evaluation_id"].n_unique() > 1:
        scores = newest_evaluation(scores)
        board = leaderboard(scores, references=())
        if board.is_empty():
            return pl.DataFrame(schema=QUALITY_SCHEMA)
    identity = ["product", "variable", "truth_semantics", "lead_bucket", "method_id"]
    metric_columns = [
        "n",
        "n_valid_times",
        "coverage",
        "mae",
        "rmse",
        "bias",
        "coverage80",
        "coverage90",
        "crps",
        "pinball",
    ]
    rows = board.select(*identity, *metric_columns)
    anchor = scores["valid_time"].max()
    recent = pl.DataFrame(
        schema={
            **dict.fromkeys(identity, pl.String()),
            "recent_mae": pl.Float64(),
            "recent_n": pl.Int64(),
        }
    )
    recent_quantile = _recent_quantile_metrics(
        scores, anchor if isinstance(anchor, datetime) else None
    )
    if isinstance(anchor, datetime):
        # half-open (anchor - window, anchor]: exactly the last 14 days
        recent = (
            scores.filter(
                pl.col("valid_time") > anchor - timedelta(days=RECENT_WINDOW_DAYS)
            )
            .drop_nulls(["y_pred", "y_true"])
            .group_by(["product", "variable", "semantics", "lead_bucket", "method_id"])
            .agg(
                (pl.col("y_pred") - pl.col("y_true")).abs().mean().alias("recent_mae"),
                pl.len().cast(pl.Int64).alias("recent_n"),
            )
            .rename({"semantics": "truth_semantics"})
        )
    return (
        rows.join(recent, on=identity, how="left")
        .join(recent_quantile, on=identity, how="left")
        .with_columns(
            pl.lit(now).alias("recorded_at"),
            pl.lit(_first(scores, "evaluation_id")).alias("evaluation_id"),
            pl.lit(scores["evaluation_created_at"].max()).alias(
                "evaluation_created_at"
            ),
            pl.lit(_first(scores, "source_kind")).alias("source_kind"),
            pl.lit(_first(scores, "code_version")).alias("code_version"),
            pl.lit(_first(scores, "config_fingerprint")).alias("config_fingerprint"),
            pl.lit(_first(scores, "dataset_fingerprint")).alias("dataset_fingerprint"),
        )
        .select(QUALITY_SCHEMA.names())
        .cast(QUALITY_SCHEMA, strict=False)
    )


def _selection_field(
    selections: Mapping[object, object], key: str, field_name: str
) -> object:
    entry = selections.get(key)
    if isinstance(entry, Mapping):
        return entry.get(field_name)
    return None


def selection_churn(
    previous: Mapping[str, object],
    current: Mapping[str, object],
    *,
    now: datetime,
    code_version: str,
) -> pl.DataFrame:
    """Per-slice diff of two release payloads, added/removed rows included."""
    previous_selections = previous.get("selections")
    current_selections = current.get("selections")
    if not isinstance(previous_selections, Mapping) or not isinstance(
        current_selections, Mapping
    ):
        return pl.DataFrame(schema=CHURN_SCHEMA)
    previous_map = cast("Mapping[object, object]", previous_selections)
    current_map = cast("Mapping[object, object]", current_selections)
    rows: list[dict[str, object]] = []
    for key in sorted(str(raw) for raw in set(previous_map) | set(current_map)):
        product, _, rest = key.partition(".")
        variable, _, lead_bucket = rest.partition(".")
        from_method = _selection_field(previous_map, key, "method_id")
        to_method = _selection_field(current_map, key, "method_id")
        rows.append(
            {
                "recorded_at": now,
                "from_release_id": str(previous.get("release_id")),
                "to_release_id": str(current.get("release_id")),
                "from_promoted_at": _parse_promoted_at(previous),
                "to_promoted_at": _parse_promoted_at(current),
                "slice_key": key,
                "product": product,
                "variable": variable,
                "lead_bucket": lead_bucket,
                "from_method": from_method,
                "to_method": to_method,
                "changed": from_method != to_method,
                "from_mae": _selection_field(previous_map, key, "mae"),
                "to_mae": _selection_field(current_map, key, "mae"),
                "from_n": _selection_field(previous_map, key, "n"),
                "to_n": _selection_field(current_map, key, "n"),
                "dataset_fingerprint": current.get("dataset_fingerprint"),
                "config_fingerprint": current.get("config_fingerprint"),
                "code_version": code_version,
            }
        )
    return pl.DataFrame(rows, schema_overrides=dict(CHURN_SCHEMA)).select(
        CHURN_SCHEMA.names()
    )


def _parse_promoted_at(release: Mapping[str, object]) -> datetime | None:
    raw = release.get("promoted_at")
    if not isinstance(raw, str):
        return None
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


_TRANSFORM_ORDER = {"raw": 0, "pit": 1, "cqr": 2}


def recalibration_verdicts(recalib: pl.DataFrame) -> dict[str, float]:
    """Win shares per transform: coverage80 closest to 0.8, pinball tiebreak."""
    if recalib.is_empty() or "transform" not in recalib.columns:
        return {}
    cell = ["product", "variable", "semantics", "lead_bucket", "method_id"]
    winners = (
        recalib.drop_nulls("coverage80")
        .with_columns(
            (pl.col("coverage80") - 0.8).abs().alias("distance"),
            pl.col("transform")
            .replace_strict(_TRANSFORM_ORDER, default=99)
            .alias("order"),
        )
        .sort(["distance", "pinball", "order"])
        .group_by(cell)
        .first()
    )
    total = winners.height
    if total == 0:
        return {}
    counts = {
        str(transform): int(count)
        for transform, count in winners.group_by("transform").len().iter_rows()
    }
    verdicts: dict[str, float] = {
        f"recalib_win_share_{transform}": counts.get(transform, 0) / total
        for transform in ("raw", "pit", "cqr")
    }
    verdicts["recalib_cells"] = float(total)
    return verdicts


def gate_verdicts(comparison: pl.DataFrame) -> dict[str, float]:
    """Agreement metrics between the mcs and seq_mcs promotion rules."""
    if comparison.is_empty() or "agree" not in comparison.columns:
        return {}
    agree = comparison["agree"].drop_nulls()
    if agree.is_empty():
        return {}
    return {
        "gate_agree_rate": float(cast("float", agree.mean())),
        "gate_disagreements": float(cast("int", (~agree).sum())),
        "gate_slices": float(agree.len()),
    }


def discovery_verdicts(board: pl.DataFrame, *, alpha: float) -> dict[str, float]:
    """p-BH vs e-BH discovery counts — the two corrections' running A/B.

    A discovery is one (row, reference) pair the correction flags; the pair
    totals differ between the families (DM tests every scored pair, the
    e-process only tracked ones), so both denominators are recorded too.
    """
    if board.is_empty():
        return {}
    verdicts: dict[str, float] = {}
    p_columns = [c for c in board.columns if c.startswith("dm_q_vs_")]
    if p_columns:
        q_values = board.select(p_columns)
        verdicts["pbh_discoveries"] = float(
            sum(
                int(cast("int", (q_values[column] <= alpha).sum()))
                for column in p_columns
            )
        )
        verdicts["pbh_pairs"] = float(
            sum(q_values[column].drop_nulls().len() for column in p_columns)
        )
    e_columns = [c for c in board.columns if c.startswith("ebh_sig_vs_")]
    if e_columns:
        verdicts["ebh_discoveries"] = float(
            sum(int(cast("int", board[column].sum())) for column in e_columns)
        )
        verdicts["ebh_pairs"] = float(
            sum(
                board[column.replace("ebh_sig_vs_", "e_vs_")].drop_nulls().len()
                for column in e_columns
            )
        )
    return verdicts


def wealth_rows(product: str, store: "EProcessStore", *, now: datetime) -> pl.DataFrame:
    """One snapshot row per e-process pair; unchanged wealth appends nothing."""
    rows = [
        {
            "recorded_at": now,
            "product": product,
            "pair_key": key,
            "resets": store.resets,
            "t": entry.t,
            "log_e": entry.log_e,
            "lam": entry.lam,
            "scale": entry.scale,
            "config_fingerprint": store.config_fingerprint,
            "code_version": store.code_version,
        }
        for key, entry in sorted(store.entries.items())
    ]
    if not rows:
        return pl.DataFrame(schema=EPROCESS_WEALTH_SCHEMA)
    return pl.DataFrame(rows, schema_overrides=dict(EPROCESS_WEALTH_SCHEMA)).select(
        EPROCESS_WEALTH_SCHEMA.names()
    )


def served_quality_rows(
    served: pl.DataFrame,
    *,
    config_fingerprint_value: str,
    code_version: str,
    now: datetime,
) -> pl.DataFrame:
    """Daily sample of realized served quality vs its backtest promise."""
    if served.is_empty():
        return pl.DataFrame(schema=SERVED_QUALITY_SCHEMA)
    frame = served
    for column, dtype in (("backtest_mae", pl.Float64()), ("mae_gap", pl.Float64())):
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=dtype).alias(column))
    return (
        frame.with_columns(
            pl.lit(now).alias("recorded_at"),
            pl.lit(now.date()).alias("as_of_date"),
            pl.lit(config_fingerprint_value).alias("config_fingerprint"),
            pl.lit(code_version).alias("code_version"),
        )
        .select(SERVED_QUALITY_SCHEMA.names())
        .cast(SERVED_QUALITY_SCHEMA, strict=False)
    )


def record_quality(
    config: Config,
    scores: pl.DataFrame,
    board: pl.DataFrame,
    *,
    now: datetime | None = None,
) -> None:
    try:
        moment = now or datetime.now(tz=UTC)
        fresh = quality_rows(scores, board, now=moment)
        append_ledger(
            fresh, ledger_path(config, QUALITY_LEDGER), QUALITY_LEDGER, now=moment
        )
    except _RECORDER_ERRORS:
        return


def record_selection_churn(
    config: Config, *, now: datetime | None = None
) -> pl.DataFrame:
    """Diff the two newest releases; returns the churn frame for rendering."""
    try:
        moment = now or datetime.now(tz=UTC)
        releases = _readable_releases(config.artifacts_dir / "releases")
        if len(releases) < 2:
            return pl.DataFrame(schema=CHURN_SCHEMA)
        fresh = selection_churn(
            releases[-2], releases[-1], now=moment, code_version=code_identity()
        )
        append_ledger(
            fresh, ledger_path(config, CHURN_LEDGER), CHURN_LEDGER, now=moment
        )
    except _RECORDER_ERRORS:
        return pl.DataFrame(schema=CHURN_SCHEMA)
    return fresh


def _readable_releases(directory: Path) -> list[Mapping[str, object]]:
    releases: list[Mapping[str, object]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            isinstance(raw, dict)
            and isinstance(raw.get("release_id"), str)
            and isinstance(raw.get("promoted_at"), str)
            and isinstance(raw.get("selections"), dict)
        ):
            releases.append(raw)
    releases.sort(key=lambda release: str(release["promoted_at"]))
    return releases


def record_verdicts(
    config: Config,
    product: str,
    scores: pl.DataFrame,
    recalib: pl.DataFrame,
    comparison: pl.DataFrame,
    *,
    board: pl.DataFrame | None = None,
    alpha: float = 0.05,
    now: datetime | None = None,
) -> None:
    from grounded_weather_forecast.reports.recalibration import (  # noqa: PLC0415
        newest_evaluation,
    )

    try:
        moment = now or datetime.now(tz=UTC)
        verdicts = {**recalibration_verdicts(recalib), **gate_verdicts(comparison)}
        if board is not None:
            verdicts.update(discovery_verdicts(board, alpha=alpha))
        if not verdicts or scores.is_empty():
            return
        newest = newest_evaluation(scores)
        rows = [
            {
                "recorded_at": moment,
                "evaluation_id": _first(newest, "evaluation_id"),
                "product": product,
                "source_kind": _first(newest, "source_kind"),
                "name": name,
                "value": float(value),
                "code_version": _first(newest, "code_version"),
                "config_fingerprint": _first(newest, "config_fingerprint"),
                "dataset_fingerprint": _first(newest, "dataset_fingerprint"),
            }
            for name, value in sorted(verdicts.items())
        ]
        fresh = pl.DataFrame(rows, schema_overrides=dict(VERDICTS_SCHEMA)).select(
            VERDICTS_SCHEMA.names()
        )
        append_ledger(
            fresh, ledger_path(config, VERDICTS_LEDGER), VERDICTS_LEDGER, now=moment
        )
    except _RECORDER_ERRORS:
        return


def record_eprocess_wealth(
    config: Config,
    product: str,
    store: "EProcessStore",
    *,
    now: datetime | None = None,
) -> None:
    try:
        moment = now or datetime.now(tz=UTC)
        fresh = wealth_rows(product, store, now=moment)
        append_ledger(
            fresh,
            ledger_path(config, EPROCESS_WEALTH_LEDGER),
            EPROCESS_WEALTH_LEDGER,
            now=moment,
        )
    except _RECORDER_ERRORS:
        return


def record_served_quality(
    config: Config, served: pl.DataFrame, *, now: datetime | None = None
) -> None:
    try:
        moment = now or datetime.now(tz=UTC)
        fresh = served_quality_rows(
            served,
            config_fingerprint_value=config_fingerprint(config),
            code_version=code_identity(),
            now=moment,
        )
        append_ledger(
            fresh,
            ledger_path(config, SERVED_QUALITY_LEDGER),
            SERVED_QUALITY_LEDGER,
            now=moment,
        )
    except _RECORDER_ERRORS:
        return


def record_winner_curse(
    config: Config,
    product: str,
    scores: pl.DataFrame,
    winners: pl.DataFrame,
    *,
    now: datetime | None = None,
) -> None:
    """Winner-bias scalars into the verdicts ledger (same keys, new names)."""
    from grounded_weather_forecast.reports.recalibration import (  # noqa: PLC0415
        newest_evaluation,
    )
    from grounded_weather_forecast.reports.winner_curse import (  # noqa: PLC0415
        winner_curse_verdicts,
    )

    try:
        moment = now or datetime.now(tz=UTC)
        verdicts = winner_curse_verdicts(winners)
        if not verdicts or scores.is_empty():
            return
        newest = newest_evaluation(scores)
        rows = [
            {
                "recorded_at": moment,
                "evaluation_id": _first(newest, "evaluation_id"),
                "product": product,
                "source_kind": _first(newest, "source_kind"),
                "name": name,
                "value": float(value),
                "code_version": _first(newest, "code_version"),
                "config_fingerprint": _first(newest, "config_fingerprint"),
                "dataset_fingerprint": _first(newest, "dataset_fingerprint"),
            }
            for name, value in sorted(verdicts.items())
        ]
        fresh = pl.DataFrame(rows, schema_overrides=dict(VERDICTS_SCHEMA)).select(
            VERDICTS_SCHEMA.names()
        )
        append_ledger(
            fresh, ledger_path(config, VERDICTS_LEDGER), VERDICTS_LEDGER, now=moment
        )
    except _RECORDER_ERRORS:
        return


def _winner_bias_note(config: Config, product: str) -> str:
    """The newest recorded winner bias, as a suffix for the delta line.

    The delta line's "best mae" is an argmin over every method per slice —
    the most curse-contaminated headline in the system; the note keeps the
    uncorrected-argmin caveat attached to it.
    """
    try:
        verdicts = load_ledger(
            ledger_path(config, VERDICTS_LEDGER), VERDICTS_SCHEMA
        ).filter(
            (pl.col("product") == product) & (pl.col("name") == "winner_bias_mean")
        )
        if verdicts.is_empty():
            return ""
        newest = verdicts.sort("recorded_at").row(-1, named=True)
        return f" (argmin winner bias ~ {float(newest['value']):+.3f}, uncorrected)"
    except _RECORDER_ERRORS:
        return ""


def quality_delta_line(config: Config) -> str | None:
    """One line comparing the two newest live evaluations' best-slice MAE."""
    try:
        frame = load_ledger(ledger_path(config, QUALITY_LEDGER), QUALITY_SCHEMA).filter(
            pl.col("source_kind") == "live"
        )
        if frame.is_empty():
            return None
        product_counts = frame.group_by("product").len().sort("len", descending=True)
        product = str(product_counts["product"][0])
        rows = frame.filter(pl.col("product") == product)
        evaluations = (
            rows.select("evaluation_id", "evaluation_created_at")
            .unique()
            .sort("evaluation_created_at")
        )
        if evaluations.height < 2:
            return None

        def weighted_best(evaluation_id: str) -> tuple[float, int]:
            best = (
                rows.filter(pl.col("evaluation_id") == evaluation_id)
                .group_by(["variable", "lead_bucket"])
                .agg(
                    pl.col("mae").min().alias("best_mae"),
                    pl.col("n").max().alias("slice_n"),
                )
                .drop_nulls(["best_mae", "slice_n"])
            )
            total = float(cast("float", best["slice_n"].sum()))
            if best.is_empty() or total <= 0.0:
                return float("nan"), 0
            weighted_sum = float(
                cast("float", (best["best_mae"] * best["slice_n"]).sum())
            )
            return weighted_sum / total, best.height

        previous_id = str(evaluations["evaluation_id"][-2])
        current_id = str(evaluations["evaluation_id"][-1])
        previous_mae, _ = weighted_best(previous_id)
        current_mae, slices = weighted_best(current_id)
        if not (previous_mae == previous_mae and current_mae == current_mae):
            return None
        percent = (
            (current_mae - previous_mae) / previous_mae * 100.0
            if previous_mae > 0.0
            else 0.0
        )
        return (
            f"quality delta ({product} live): n-weighted best mae "
            f"{previous_mae:.3f} -> {current_mae:.3f} ({percent:+.1f}%) "
            f"over {slices} slices{_winner_bias_note(config, product)}"
        )
    except _RECORDER_ERRORS:
        return None
