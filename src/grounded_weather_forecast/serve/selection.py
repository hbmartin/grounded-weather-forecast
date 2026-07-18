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

from grounded_weather_forecast.backtest.scores import load_scores
from grounded_weather_forecast.config import Config
from grounded_weather_forecast.evaluation import (
    ModelRelease,
    config_fingerprint,
    dataset_fingerprint,
)
from grounded_weather_forecast.reports.leaderboard import (
    DEFAULT_REFERENCES,
    leaderboard,
    slice_winners,
)

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
type SliceKey = tuple[str, str, str]


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


def _newest_complete_slices(
    candidates: list[pl.DataFrame],
) -> dict[SliceKey, pl.DataFrame]:
    """Newest atomic evaluation per slice containing both reference methods."""
    if not candidates:
        return {}
    combined = pl.concat(candidates, how="diagonal_relaxed")
    selected: dict[SliceKey, tuple[datetime, str, pl.DataFrame]] = {}
    for evaluation_key, evaluation in combined.partition_by(
        "evaluation_id", as_dict=True
    ).items():
        evaluation_id = str(evaluation_key[0])
        created_at = evaluation["evaluation_created_at"].max()
        if not isinstance(created_at, datetime):
            continue
        for slice_key, frame in evaluation.partition_by(
            ["product", "variable", "lead_bucket"], as_dict=True
        ).items():
            methods = {str(method) for method in frame["method_id"].unique().to_list()}
            if not set(DEFAULT_REFERENCES) <= methods:
                continue
            product, variable, lead_bucket = (str(part) for part in slice_key)
            key: SliceKey = (product, variable, lead_bucket)
            previous = selected.get(key)
            marker = (created_at, evaluation_id)
            if previous is None or marker > previous[:2]:
                selected[key] = (created_at, evaluation_id, frame)
    return {key: selected[key][2] for key in sorted(selected)}


def _selection_payload(
    selections: Mapping[SliceKey, Selection],
) -> dict[str, dict[str, object]]:
    return {
        ".".join(key): {
            "method_id": selected.method_id,
            "reason": selected.reason,
            "evaluation_id": selected.evaluation_id,
            "n": selected.n,
            "mae": selected.mae,
        }
        for key, selected in sorted(selections.items())
    }


def _evaluation_contexts(
    selected_scores: list[pl.DataFrame],
) -> tuple[dict[str, object], ...]:
    if not selected_scores:
        return ()
    combined = pl.concat(selected_scores, how="diagonal_relaxed")
    contexts: list[dict[str, object]] = []
    for evaluation_key, frame in combined.partition_by(
        "evaluation_id", as_dict=True
    ).items():
        contexts.append(
            {
                "evaluation_id": str(evaluation_key[0]),
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
        )
    return tuple(sorted(contexts, key=lambda context: str(context["evaluation_id"])))


def _make_release(
    config: Config,
    selections: Mapping[SliceKey, Selection],
    selected_scores: list[pl.DataFrame],
) -> ModelRelease:
    evaluation_ids = tuple(
        sorted(
            {
                selected.evaluation_id
                for selected in selections.values()
                if selected.evaluation_id is not None
            }
        )
    )
    return ModelRelease.create(
        dataset=dataset_fingerprint(config),
        configuration=config_fingerprint(config),
        evaluation_ids=evaluation_ids,
        evaluation_contexts=_evaluation_contexts(selected_scores),
        training_cutoff=max(
            (frame["valid_time"].max() for frame in selected_scores), default=None
        ),
        selections=_selection_payload(selections),
    )


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
    slices = _newest_complete_slices(compatible)
    selected_scores: list[pl.DataFrame] = []
    fallbacks: dict[SliceKey, Selection] = {}
    for key, frame in slices.items():
        board = leaderboard(frame)
        winners = slice_winners(
            board,
            scores=frame,
            rule=config.promotion.rule,
            alpha=config.promotion.alpha,
        )
        if winners.is_empty():
            continue
        selected_scores.append(frame)
        evaluation_id = str(frame["evaluation_id"][0])
        row = winners.row(0, named=True)
        selections[key] = Selection(
            method_id=str(row["method_id"]),
            reason="lowest backtest MAE among promotable common-case methods",
            n=int(row["n"]),
            mae=float(row["mae"]),
            evaluation_id=evaluation_id,
            dataset_fingerprint=dataset_fingerprint(config),
        )
        fallback_rows = board.filter(pl.col("method_id") == FALLBACK_METHOD).sort("mae")
        if not fallback_rows.is_empty():
            fallback = fallback_rows.row(0, named=True)
            fallbacks[key] = Selection(
                method_id=FALLBACK_METHOD,
                reason="reference fallback",
                n=int(fallback["n"]),
                mae=float(fallback["mae"]),
                evaluation_id=evaluation_id,
                dataset_fingerprint=dataset_fingerprint(config),
            )
    for (product, variable), method_id in pinned.items():
        for key in [key for key in selections if key[:2] == (product, variable)]:
            selections[key] = replace(
                selections[key], method_id=method_id, reason="pinned in config"
            )
    if not selections:
        return selections
    prospective = _make_release(config, selections, selected_scores)
    selections = {
        key: replace(selected, release_id=prospective.release_id)
        for key, selected in selections.items()
    }
    fallbacks = {
        key: replace(selected, release_id=prospective.release_id)
        for key, selected in fallbacks.items()
    }
    selections = apply_live_gate(
        selections,
        _live_verification(config),
        factor=config.promotion.live_gap_factor,
        min_n=config.promotion.min_live_n,
        fallbacks=fallbacks,
    )
    release = _make_release(config, selections, selected_scores)
    release.write(config.artifacts_dir / "releases")
    return {
        key: replace(selected, release_id=release.release_id)
        for key, selected in selections.items()
    }


def _live_verification(config: Config) -> pl.DataFrame:
    """Realized served-forecast skill, empty until history and truth exist."""
    if not config.predict.history_path.exists():
        return pl.DataFrame()
    from grounded_weather_forecast.dataset.matrix import build_truth  # noqa: PLC0415
    from grounded_weather_forecast.reports.verification import (  # noqa: PLC0415
        verify_history,
    )

    minute, hourly, daily = build_truth(config)
    return verify_history(config.predict.history_path, hourly, minute, daily)


def apply_live_gate(
    selections: dict[tuple[str, str, str], Selection],
    live: pl.DataFrame,
    *,
    factor: float,
    min_n: int,
    fallbacks: Mapping[SliceKey, Selection] | None = None,
) -> dict[tuple[str, str, str], Selection]:
    """Close the self-verification loop: demote methods that underdeliver live.

    A selected method whose realized served MAE is materially worse than its
    backtest promise (``live_mae > factor * backtest_mae`` at ``n >= min_n``)
    falls back to the reference method — the one failure a backtest can never
    catch by itself. The verdict travels in the selection reason, so the
    release ledger records every demotion.
    """
    identity_columns = {"lead_bucket", "dataset_fingerprint", "release_id"}
    if live.is_empty() or not identity_columns <= set(live.columns):
        return selections
    by_key = {
        (
            str(row["product"]),
            str(row["variable"]),
            str(row["lead_bucket"]),
            str(row["method_id"]),
            str(row["dataset_fingerprint"]),
            str(row["release_id"]),
        ): row
        for row in live.iter_rows(named=True)
        if row["dataset_fingerprint"] is not None and row["release_id"] is not None
    }
    gated = dict(selections)
    for key, selected in selections.items():
        product, variable, bucket = key
        if (
            selected.method_id == FALLBACK_METHOD
            or selected.reason == "pinned in config"
            or selected.mae is None
            or selected.dataset_fingerprint is None
            or selected.release_id is None
        ):
            continue
        row = by_key.get(
            (
                product,
                variable,
                bucket,
                selected.method_id,
                selected.dataset_fingerprint,
                selected.release_id,
            )
        )
        if row is None:
            continue
        live_mae = float(row["live_mae"])
        if int(row["n"]) >= min_n and live_mae > factor * selected.mae:
            fallback = (
                fallbacks.get(key)
                if fallbacks is not None
                else Selection(
                    FALLBACK_METHOD,
                    reason="reference fallback",
                    dataset_fingerprint=selected.dataset_fingerprint,
                )
            )
            if fallback is None:
                continue
            gated[key] = replace(
                fallback,
                reason=(
                    f"demoted {selected.method_id}: live MAE {live_mae:.3f} vs "
                    f"backtest {selected.mae:.3f} (factor {factor})"
                ),
                dataset_fingerprint=selected.dataset_fingerprint,
                release_id=selected.release_id,
            )
    return gated


def no_evidence_reason(config: Config, scores_dir: Path) -> str:
    """Why serving is degraded: cold start vs invalidated evidence.

    A rebuild that adds matrix columns changes the dataset fingerprint, which
    correctly invalidates promoted evidence — but that failure reads exactly
    like a cold start unless it is named. The distinction decides the fix:
    keep polling, or just re-run the backtest.
    """
    paths = sorted(scores_dir.glob("scores_*.parquet"))
    if not paths:
        return (
            "cold start: no backtest scores exist yet; run "
            "`backtest --source live` then `report` once the archive has folds"
        )
    live_datasets: set[str] = set()
    live_configs: set[str] = set()
    for path in paths:
        scores = load_scores(path)
        if scores.is_empty() or set(scores["source_kind"].unique()) != {"live"}:
            continue
        if "dataset_fingerprint" in scores.columns:
            live_datasets |= {
                str(value) for value in scores["dataset_fingerprint"].unique().to_list()
            }
        if "config_fingerprint" in scores.columns:
            live_configs |= {
                str(value) for value in scores["config_fingerprint"].unique().to_list()
            }
    if not live_datasets:
        return (
            "no live backtest evidence yet (synthetic evidence is never "
            "promoted); keep polling and run `backtest --source live`"
        )
    if dataset_fingerprint(config) not in live_datasets:
        return (
            "dataset fingerprint changed since the last backtest (a rebuild "
            "invalidates promoted evidence); re-run `backtest --source live` "
            "then `report`"
        )
    if config_fingerprint(config) not in live_configs:
        return (
            "config changed since the last backtest; re-run "
            "`backtest --source live` then `report`"
        )
    return "live evidence exists but no slice met the promotion gates; keep polling"


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
