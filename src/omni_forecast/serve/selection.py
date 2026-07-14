"""Method selection: which blender serves which (product, variable, lead bucket).

The backtest leaderboard is the only thing allowed to declare winners, so
selection reads the persisted scores rather than re-deciding anything. Config
pins override, and any slice with no evidence falls back to a named default —
never to silence.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from omni_forecast.backtest.scores import load_scores
from omni_forecast.config import Config
from omni_forecast.evaluation import (
    ModelRelease,
    config_fingerprint,
    dataset_fingerprint,
)
from omni_forecast.reports.leaderboard import leaderboard, slice_winners

FALLBACK_METHOD = "equal_weight"


@dataclass(frozen=True, slots=True)
class Selection:
    """The chosen method for one slice, and why."""

    method_id: str
    reason: str
    n: int = 0
    mae: float | None = None
    evaluation_id: str | None = None
    dataset_fingerprint: str | None = None
    release_id: str | None = None


type SelectionMap = Mapping[tuple[str, str, str], Selection]


def _pins(config: Config) -> dict[tuple[str, str], str]:
    """``[predict.methods]`` keys are ``"<product>.<variable>"``."""
    pinned: dict[tuple[str, str], str] = {}
    for key, method_id in config.predict.methods.items():
        product, _, variable = key.partition(".")
        if product and variable:
            pinned[(product, variable)] = method_id
    return pinned


def _compatible_scores(
    config: Config, scores_dir: Path, as_of: datetime | None
) -> list[pl.DataFrame]:
    current_dataset = dataset_fingerprint(config)
    candidates: list[pl.DataFrame] = []
    for path in sorted(scores_dir.glob("scores_*.parquet")):
        scores = load_scores(path)
        if scores.is_empty() or set(scores["source_kind"].unique()) != {"live"}:
            continue
        required_identity = {
            "dataset_fingerprint",
            "config_fingerprint",
            "evaluation_id",
            "evaluation_created_at",
        }
        if not required_identity <= set(scores.columns):
            continue
        scores = scores.filter(pl.col("dataset_fingerprint") == current_dataset)
        scores = scores.filter(
            pl.col("config_fingerprint") == config_fingerprint(config)
        )
        if as_of is not None:
            scores = scores.filter(pl.col("evaluation_created_at") <= as_of)
        if as_of is not None:
            scores = scores.filter(pl.col("valid_time") <= as_of)
        if not scores.is_empty():
            candidates.append(scores)
    return candidates


def _release_as_of(config: Config, as_of: datetime) -> SelectionMap | None:
    """Newest release that genuinely existed by a historical issue time."""
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)
    candidates: list[tuple[datetime, dict[str, object]]] = []
    for path in (config.artifacts_dir / "releases").glob("*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("dataset_fingerprint") != dataset_fingerprint(config):
            continue
        if raw.get("config_fingerprint") != config_fingerprint(config):
            continue
        promoted_at = datetime.fromisoformat(str(raw["promoted_at"]))
        if promoted_at <= as_of:
            candidates.append((promoted_at, raw))
    if not candidates:
        return None
    _, release = max(candidates, key=lambda item: item[0])
    release_id = str(release["release_id"])
    dataset = str(release["dataset_fingerprint"])
    selections: dict[tuple[str, str, str], Selection] = {}
    raw_selections = release.get("selections", {})
    if not isinstance(raw_selections, dict):
        return None
    for raw_key, raw_selection in raw_selections.items():
        if not isinstance(raw_selection, dict):
            continue
        parts = str(raw_key).split(".", maxsplit=2)
        if len(parts) != 3:
            continue
        method_id = raw_selection.get("method_id")
        reason = raw_selection.get("reason")
        raw_n = raw_selection.get("n", 0)
        raw_mae = raw_selection.get("mae")
        if not isinstance(method_id, str) or not isinstance(reason, str):
            continue
        n = int(raw_n) if isinstance(raw_n, (int, float)) else 0
        mae = float(raw_mae) if isinstance(raw_mae, (int, float)) else None
        product, variable, bucket = parts
        selections[(product, variable, bucket)] = Selection(
            method_id=method_id,
            reason=reason,
            n=n,
            mae=mae,
            evaluation_id=str(raw_selection.get("evaluation_id") or "unknown"),
            dataset_fingerprint=dataset,
            release_id=release_id,
        )
    return selections


def select_methods(
    config: Config, scores_dir: Path, as_of: datetime | None = None
) -> SelectionMap:
    """Select only live evidence compatible with this dataset and issue time."""
    if as_of is not None:
        return _release_as_of(config, as_of) or {}
    pinned = _pins(config)
    selections: dict[tuple[str, str, str], Selection] = {}
    compatible = _compatible_scores(config, scores_dir, as_of)
    selected_scores: list[pl.DataFrame] = []
    for product in ("hourly", "daily"):
        product_frames = [
            frame.filter(pl.col("product") == product) for frame in compatible
        ]
        product_frames = [frame for frame in product_frames if not frame.is_empty()]
        if not product_frames:
            continue
        combined = pl.concat(product_frames, how="diagonal_relaxed")
        if "evaluation_created_at" in combined.columns:
            latest = combined["evaluation_created_at"].max()
            combined = combined.filter(pl.col("evaluation_created_at") == latest)
        selected_scores.append(combined)
        winners = slice_winners(leaderboard(combined))
        evaluation_id = (
            str(combined["evaluation_id"][0])
            if "evaluation_id" in combined.columns
            else "legacy"
        )
        for row in winners.iter_rows(named=True):
            key = (row["product"], row["variable"], row["lead_bucket"])
            selections[key] = Selection(
                method_id=row["method_id"],
                reason="lowest backtest MAE among promotable common-case methods",
                n=int(row["n"]),
                mae=float(row["mae"]),
                evaluation_id=evaluation_id,
                dataset_fingerprint=dataset_fingerprint(config),
            )
    for (product, variable), method_id in pinned.items():
        for key in [k for k in selections if k[:2] == (product, variable)]:
            selections[key] = replace(
                selections[key], method_id=method_id, reason="pinned in config"
            )
    if selections:
        evaluation_ids = tuple(
            sorted(
                {
                    selected.evaluation_id
                    for selected in selections.values()
                    if selected.evaluation_id is not None
                }
            )
        )
        release = ModelRelease.create(
            dataset=dataset_fingerprint(config),
            configuration=config_fingerprint(config),
            evaluation_ids=evaluation_ids,
            evaluation_contexts=tuple(
                {
                    "evaluation_id": str(frame["evaluation_id"][0]),
                    "source_kind": str(frame["source_kind"][0]),
                    "source_set_json": str(frame["source_set_json"][0]),
                    "semantics": {
                        str(row["variable"]): str(row["semantics"])
                        for row in frame.select("variable", "semantics")
                        .unique()
                        .iter_rows(named=True)
                    },
                    "window": str(frame["window"][0]),
                    "code_version": str(frame["code_version"][0]),
                    "config_fingerprint": str(frame["config_fingerprint"][0]),
                }
                for frame in selected_scores
            ),
            training_cutoff=max(
                (frame["valid_time"].max() for frame in selected_scores), default=None
            ),
            selections={
                ".".join(key): {
                    "method_id": selected.method_id,
                    "reason": selected.reason,
                    "evaluation_id": selected.evaluation_id,
                    "n": selected.n,
                    "mae": selected.mae,
                }
                for key, selected in selections.items()
            },
        )
        release.write(config.artifacts_dir / "releases")
        selections = {
            key: replace(selected, release_id=release.release_id)
            for key, selected in selections.items()
        }
    return selections


def method_for(
    selections: SelectionMap,
    product: str,
    variable: str,
    lead_bucket: str | None,
    config: Config | None = None,
) -> Selection:
    """The selected method, falling back explicitly when a slice has no scores."""
    if config is not None:
        pinned = _pins(config).get((product, variable))
        if pinned is not None:
            return Selection(pinned, reason="pinned in config")
    if lead_bucket is not None:
        found = selections.get((product, variable, lead_bucket))
        if found is not None:
            return found
    return Selection(FALLBACK_METHOD, reason="no backtest evidence for this slice")


def selection_report(selections: SelectionMap) -> pl.DataFrame:
    rows = [
        {
            "product": product,
            "variable": variable,
            "lead_bucket": bucket,
            "method_id": chosen.method_id,
            "n": chosen.n,
            "mae": chosen.mae,
            "reason": chosen.reason,
            "evaluation_id": chosen.evaluation_id,
            "release_id": chosen.release_id,
        }
        for (product, variable, bucket), chosen in sorted(selections.items())
    ]
    return pl.DataFrame(rows)
