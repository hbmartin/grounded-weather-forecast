"""Append-only quality-evidence ledgers under ``artifacts/history/``.

Every nightly ``report`` renders a point-in-time view and discards the
trend; the ledgers keep it. Five parquet files record, per cycle: the
leaderboard's quality metrics (plus a 14-day recent-window MAE so genuine
movement is not diluted by months of expanding-window history), selection
churn between consecutive releases, A/B verdict summaries (quantile
recalibration and promotion-gate agreement), e-process wealth snapshots,
and realized served quality against its backtest promise.

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

import polars as pl
from filelock import Timeout

from grounded_weather_forecast.config import Config
from grounded_weather_forecast.evaluation import code_identity, config_fingerprint
from grounded_weather_forecast.storage import atomic_write_parquet, locked_path

if TYPE_CHECKING:
    from grounded_weather_forecast.reports.eprocess import EProcessStore

_LOCK_TIMEOUT_SECONDS = 5.0
RECENT_WINDOW_DAYS = 14
RETENTION_DAYS = 730

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

LEDGERS: Mapping[str, LedgerSpec] = {
    spec.name: spec
    for spec in (
        QUALITY_LEDGER,
        CHURN_LEDGER,
        VERDICTS_LEDGER,
        EPROCESS_WEALTH_LEDGER,
        SERVED_QUALITY_LEDGER,
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
        fresh = fresh.select(spec.schema.names()).cast(spec.schema, strict=False)
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
    now: datetime | None = None,
) -> None:
    from grounded_weather_forecast.reports.recalibration import (  # noqa: PLC0415
        newest_evaluation,
    )

    try:
        moment = now or datetime.now(tz=UTC)
        verdicts = {**recalibration_verdicts(recalib), **gate_verdicts(comparison)}
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
            f"over {slices} slices"
        )
    except _RECORDER_ERRORS:
        return None
