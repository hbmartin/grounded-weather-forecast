"""Method selection: which blender serves which (product, variable, lead bucket).

The backtest leaderboard is the only thing allowed to declare winners, so
selection reads the persisted scores rather than re-deciding anything. Config
pins override, and any slice with no evidence falls back to a named default —
never to silence.
"""

import json
import math
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import polars as pl

from grounded_weather_forecast.backtest.scores import load_scores
from grounded_weather_forecast.config import Config
from grounded_weather_forecast.evaluation import (
    ModelRelease,
    code_identity,
    config_fingerprint,
    dataset_fingerprint,
)
from grounded_weather_forecast.reports.leaderboard import (
    DEFAULT_REFERENCES,
    leaderboard,
    slice_winners,
)

FALLBACK_METHOD = "equal_weight"
_LIVE_KEY = ("product", "variable", "lead_bucket", "method_id")
# How long a served row may still testify. A module constant rather than a
# config field on purpose: a new field changes `repr(config)`, hence
# `config_fingerprint`, which would invalidate every score and release once.
# The window also gives the gate hysteresis — a demoted method eventually ages
# out of its own bad evidence and gets another hearing.
_LIVE_EVIDENCE_WINDOW = timedelta(days=14)


@dataclass(frozen=True, slots=True)
class Selection:
    """The chosen method for one slice, and why.

    ``reason`` is display text. Anything that changes behaviour reads the
    flags instead: a reworded message must never silently turn a pinned
    method into a degradable one.
    """

    method_id: str
    reason: str
    n: int = 0
    mae: float | None = None
    evaluation_id: str | None = None
    dataset_fingerprint: str | None = None
    release_id: str | None = None
    code_version: str | None = None
    # Appended to preserve the positional meaning of all fields that existed
    # before behaviour flags were introduced.
    pinned: bool = False
    degraded: bool = False


type SelectionMap = Mapping[tuple[str, str, str], Selection]
type SliceKey = tuple[str, str, str]
type LiveKey = tuple[str, str, str, str]
type EligibleReleases = Mapping[LiveKey, frozenset[str]]


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
    current_code = code_identity()
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
            "code_version",
        }
        if not required_identity <= set(scores.columns):
            continue
        scores = scores.filter(pl.col("dataset_fingerprint") == current_dataset)
        scores = scores.filter(
            pl.col("config_fingerprint") == config_fingerprint(config)
        )
        scores = scores.filter(pl.col("code_version") == current_code)
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
            "code_version": selected.code_version,
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


def _compatible_releases(
    config: Config, *, match_dataset: bool
) -> list[dict[str, object]]:
    """Ledger entries written under a config this system can still act on.

    ``match_dataset`` additionally pins the dataset fingerprint, which is what
    a historical replay needs. The live gate deliberately leaves it off — see
    ``_eligible_release_ids``.
    """
    configuration = config_fingerprint(config)
    dataset = dataset_fingerprint(config) if match_dataset else None
    releases: list[dict[str, object]] = []
    for path in sorted((config.artifacts_dir / "releases").glob("*.json")):
        # One unreadable ledger entry must not take down nightly selection.
        with suppress(OSError, ValueError):
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                continue
            required = (
                "release_id",
                "promoted_at",
                "dataset_fingerprint",
                "config_fingerprint",
                "selections",
            )
            if any(field not in raw for field in required):
                continue
            if not isinstance(raw["selections"], dict):
                continue
            if raw.get("config_fingerprint") != configuration:
                continue
            if dataset is not None and raw.get("dataset_fingerprint") != dataset:
                continue
            releases.append(raw)
    return releases


def _selections_from_release(release: Mapping[str, object]) -> SelectionMap | None:
    """Rehydrate a ledger entry's per-slice selections."""
    raw_selections = release.get("selections", {})
    if not isinstance(raw_selections, dict):
        return None
    release_id = release.get("release_id")
    dataset = release.get("dataset_fingerprint")
    if not isinstance(release_id, str) or not isinstance(dataset, str):
        return None
    selections: dict[tuple[str, str, str], Selection] = {}
    for raw_key, raw_selection in raw_selections.items():
        if not isinstance(raw_selection, dict):
            continue
        selection_mapping = cast(Mapping[str, object], raw_selection)
        parts = str(raw_key).split(".", maxsplit=2)
        if len(parts) != 3:
            continue
        method_id = selection_mapping.get("method_id")
        reason = selection_mapping.get("reason")
        raw_n = selection_mapping.get("n", 0)
        raw_mae = selection_mapping.get("mae")
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
            evaluation_id=str(selection_mapping.get("evaluation_id") or "unknown"),
            dataset_fingerprint=dataset,
            release_id=release_id,
            code_version=_release_selection_code_version(release, selection_mapping),
        )
    return selections


def _release_selection_code_version(
    release: Mapping[str, object], selection: Mapping[str, object]
) -> str | None:
    """Implementation identity for one persisted selection, including v1 ledgers."""
    direct = selection.get("code_version")
    if isinstance(direct, str) and direct:
        return direct
    evaluation_id = selection.get("evaluation_id")
    contexts = release.get("evaluation_contexts")
    if not isinstance(contexts, list):
        return None
    versions: set[str] = set()
    for raw_context in contexts:
        if not isinstance(raw_context, dict):
            continue
        context = cast(Mapping[str, object], raw_context)
        version = context.get("code_version")
        if isinstance(version, str) and (
            evaluation_id is None or context.get("evaluation_id") == evaluation_id
        ):
            versions.add(version)
    return next(iter(versions)) if len(versions) == 1 else None


def _eligible_release_ids(
    config: Config, selections: Mapping[SliceKey, Selection]
) -> EligibleReleases:
    """Release cohorts matching each selected method and implementation.

    Deliberately ignores the dataset fingerprint. That fingerprint hashes row
    counts over every parquet artifact, so it rotates on *every ingested
    observation* — keying live evidence to it means any slice whose truth
    arrives after one rebuild cycle can never be scored, which is every lead
    bucket at or beyond 24h. ``config_fingerprint`` pins settings while
    ``code_version`` includes a source digest, so evidence cannot cross an
    implementation change. Recency is applied to served rows rather than to
    immutable release creation times.
    """
    eligible: dict[LiveKey, set[str]] = {
        (*key, selected.method_id): set() for key, selected in selections.items()
    }
    for raw in _compatible_releases(config, match_dataset=False):
        release_id = raw.get("release_id")
        raw_selections = raw.get("selections")
        if not isinstance(release_id, str) or not isinstance(raw_selections, dict):
            continue
        for key, selected in selections.items():
            raw_selected = raw_selections.get(".".join(key))
            if not isinstance(raw_selected, dict):
                continue
            selection_mapping = cast(Mapping[str, object], raw_selected)
            live_key = (*key, selected.method_id)
            if (
                selection_mapping.get("method_id") == selected.method_id
                and selected.code_version is not None
                and _release_selection_code_version(raw, selection_mapping)
                == selected.code_version
            ):
                eligible[live_key].add(release_id)
    return {key: frozenset(releases) for key, releases in eligible.items()}


def _release_as_of(config: Config, as_of: datetime) -> SelectionMap | None:
    """Newest release that genuinely existed by a historical issue time."""
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)
    candidates: list[tuple[datetime, dict[str, object]]] = []
    for raw in _compatible_releases(config, match_dataset=True):
        with suppress(TypeError, ValueError):
            promoted_at = datetime.fromisoformat(str(raw["promoted_at"]))
            if promoted_at <= as_of:
                candidates.append((promoted_at, raw))
    if not candidates:
        return None
    _, release = max(candidates, key=lambda item: item[0])
    return _selections_from_release(release)


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
            code_version=str(frame["code_version"][0]),
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
                code_version=str(frame["code_version"][0]),
            )
    for (product, variable), method_id in pinned.items():
        for key in [key for key in selections if key[:2] == (product, variable)]:
            selections[key] = replace(
                selections[key],
                method_id=method_id,
                reason="pinned in config",
                pinned=True,
            )
    if not selections:
        return selections
    # Gate before minting the release. Stamping a prospective id first only
    # ever orphaned it: a demotion rewrites `selections`, which the release
    # hash covers, so the rows served under the acting release ended up
    # attributed to an id that release does not have.
    now = datetime.now(tz=UTC)
    selections = apply_live_gate(
        selections,
        _live_verification(config, now),
        factor=config.promotion.live_gap_factor,
        min_n=config.promotion.min_live_n,
        fallbacks=fallbacks,
        eligible_releases=_eligible_release_ids(config, selections),
    )
    release = _make_release(config, selections, selected_scores)
    release.write(config.artifacts_dir / "releases")
    return {
        key: replace(selected, release_id=release.release_id)
        for key, selected in selections.items()
    }


def _live_verification(config: Config, now: datetime) -> pl.DataFrame:
    """Realized served-forecast skill, empty until history and truth exist."""
    if not config.predict.history_path.exists():
        return pl.DataFrame()
    from grounded_weather_forecast.dataset.matrix import build_truth  # noqa: PLC0415
    from grounded_weather_forecast.reports.verification import (  # noqa: PLC0415
        verify_history,
    )

    minute, hourly, daily = build_truth(config)
    return verify_history(
        config.predict.history_path,
        hourly,
        minute,
        daily,
        issued_after=now - _LIVE_EVIDENCE_WINDOW,
        issued_before=now,
    )


def _pooled_live_skill(
    live: pl.DataFrame,
    eligible_releases: frozenset[str] | EligibleReleases | None,
) -> dict[tuple[str, str, str, str], tuple[int, float]]:
    """Realized skill per (product, variable, bucket, method), pooled by identity.

    ``verify_history`` scores each release cohort separately, which is right
    for a report but fatal for a gate: a release lives about a day, so a
    24-48h forecast's truth always arrives after its own cohort has been
    retired. Pooling the eligible cohorts is what lets long leads accumulate
    evidence at all. The n-weighted mean is exact for MAE.
    """
    required = {"product", "variable", "lead_bucket", "method_id", "release_id", "n"}
    if live.is_empty() or not required <= set(live.columns):
        return {}
    totals: dict[LiveKey, tuple[int, float]] = {}
    for row in live.iter_rows(named=True):
        release_id = row.get("release_id")
        key: LiveKey = (
            str(row["product"]),
            str(row["variable"]),
            str(row["lead_bucket"]),
            str(row["method_id"]),
        )
        if eligible_releases is None:
            release_is_eligible = True
        elif isinstance(eligible_releases, frozenset):
            release_is_eligible = release_id in eligible_releases
        else:
            release_is_eligible = release_id in eligible_releases.get(key, frozenset())
        n = row.get("n")
        live_mae = row.get("live_mae")
        if (
            not isinstance(release_id, str)
            or not release_is_eligible
            or not isinstance(n, int)
            or n <= 0
            or not isinstance(live_mae, (int, float))
            or not math.isfinite(float(live_mae))
        ):
            continue
        old_n, old_error = totals.get(key, (0, 0.0))
        totals[key] = (old_n + n, old_error + float(live_mae) * n)
    return {
        key: (n, weighted_error / n)
        for key, (n, weighted_error) in totals.items()
        if n > 0 and math.isfinite(weighted_error)
    }


def _live_verdict(
    pooled: Mapping[tuple[str, str, str, str], tuple[int, float]],
    key: SliceKey,
    selected: Selection,
    *,
    factor: float,
    min_n: int,
) -> str | None:
    """The demotion reason for this slice, or ``None`` to leave it standing."""
    if (
        selected.method_id == FALLBACK_METHOD
        or selected.pinned
        or selected.mae is None
        or not math.isfinite(selected.mae)
    ):
        return None
    product, variable, bucket = key
    measured = pooled.get((product, variable, bucket, selected.method_id))
    if measured is None:
        return None
    n, live_mae = measured
    if not math.isfinite(live_mae) or n < min_n or live_mae <= factor * selected.mae:
        return None
    return (
        f"demoted {selected.method_id}: live MAE {live_mae:.3f} vs "
        f"backtest {selected.mae:.3f} (factor {factor}, n={n})"
    )


def _gate_fallback(
    key: SliceKey,
    selected: Selection,
    fallbacks: Mapping[SliceKey, Selection] | None,
) -> Selection | None:
    if fallbacks is not None:
        return fallbacks.get(key)
    return Selection(
        FALLBACK_METHOD,
        reason="reference fallback",
        dataset_fingerprint=selected.dataset_fingerprint,
    )


def apply_live_gate(
    selections: dict[tuple[str, str, str], Selection],
    live: pl.DataFrame,
    *,
    factor: float,
    min_n: int,
    fallbacks: Mapping[SliceKey, Selection] | None = None,
    eligible_releases: frozenset[str] | EligibleReleases | None = None,
) -> dict[tuple[str, str, str], Selection]:
    """Close the self-verification loop: demote methods that underdeliver live.

    A selected method whose realized served MAE is materially worse than its
    backtest promise (``live_mae > factor * backtest_mae`` at ``n >= min_n``)
    falls back to the reference method — the one failure a backtest can never
    catch by itself. The verdict travels in the selection reason, so the
    release ledger records every demotion.

    Evidence is pooled across ``eligible_releases`` rather than matched to one
    release, mirroring ``ArtifactStore.load_latest_state``: identity that
    rotates with data volume cannot be a precondition for scoring, or the
    slices that most need watching are the ones that never get watched.
    ``None`` means no ledger filter — every release-tagged row is eligible.
    """
    pooled = _pooled_live_skill(live, eligible_releases)
    if not pooled:
        return selections
    gated = dict(selections)
    for key, selected in selections.items():
        verdict = _live_verdict(pooled, key, selected, factor=factor, min_n=min_n)
        if verdict is None:
            continue
        fallback = _gate_fallback(key, selected, fallbacks)
        if fallback is None:
            continue
        gated[key] = replace(
            fallback,
            reason=verdict,
            dataset_fingerprint=selected.dataset_fingerprint,
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
    live_code_versions: set[str] = set()
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
        if "code_version" in scores.columns:
            live_code_versions |= {
                str(value) for value in scores["code_version"].unique().to_list()
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
    if code_identity() not in live_code_versions:
        return (
            "implementation changed since the last backtest; re-run "
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
            found = (
                selections.get((product, variable, lead_bucket))
                if lead_bucket is not None
                else None
            )
            if found is not None and found.method_id == pinned:
                return replace(found, reason="pinned in config", pinned=True)
            return Selection(pinned, reason="pinned in config", pinned=True)
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
