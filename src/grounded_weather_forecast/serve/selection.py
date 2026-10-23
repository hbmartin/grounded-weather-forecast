"""Method selection: which blender serves which (product, variable, lead bucket).

The backtest leaderboard is the only thing allowed to declare winners, so
selection reads the persisted scores rather than re-deciding anything. Config
pins override, and any slice with no evidence falls back to a named default —
never to silence.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import polars as pl

from grounded_weather_forecast.config import Config
from grounded_weather_forecast.contracts import TruthSemantics
from grounded_weather_forecast.evaluation import (
    ModelRelease,
    code_identity,
    config_fingerprint,
    dataset_fingerprint,
)
from grounded_weather_forecast.reports.eprocess import (
    EProcessStore,
    product_eprocess_store,
)
from grounded_weather_forecast.reports.leaderboard import (
    _DAILY_GATE_POOL,
    eligible_board_rows,
    gate_references,
    leaderboard,
    slice_winners,
    union_references,
)
from grounded_weather_forecast.reports.winner_curse import should_retain_incumbent
from grounded_weather_forecast.serve.approvals import (
    ApprovalDisposition,
    ApprovalSnapshot,
    ApprovedRelease,
    aware_timestamp,
    publish_approval,
    read_approval_snapshot,
)
from grounded_weather_forecast.serve.diagnostics import (
    FitFailure,
    describe_fit_failures,
)
from grounded_weather_forecast.serve.evidence import (
    EvaluationContext,
    EvidenceSummary,
    GateVerdict,
    PromotionEvidenceError,
    ValidatedContexts,
)
from grounded_weather_forecast.storage import locked_path

from grounded_weather_forecast.serve.score_scan import (  # isort: skip
    ScoreIdentity,
    ScoreScanResult,
    compatible_slices,
    live_identity_sets as _live_identity_sets,
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
    truth_semantics: str | None = None
    # True when the winner-curse guard kept the previous release's method
    # over a statistically indistinguishable new argmin. Appended last —
    # field order is a compatibility contract.
    retained: bool = False
    training_cutoff: datetime | None = None
    legacy_context_missing: bool = False


@dataclass(frozen=True, slots=True)
class PromotionProvenance:
    recorded_at: datetime
    dataset_fingerprint: str
    config_fingerprint: str
    code_version: str


@dataclass(slots=True)
class _PromotionTelemetry:
    provenance: PromotionProvenance | None = None
    verdicts: tuple[GateVerdict, ...] = ()


type SelectionMap = Mapping[tuple[str, str, str], Selection]
type SliceKey = tuple[str, str, str]
type LiveKey = tuple[str, str, str, str]
type EligibleReleases = Mapping[LiveKey, frozenset[str]]
type EvaluationCompatibility = tuple[str, frozenset[str], str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class PromotionResult:
    selections: SelectionMap
    release_id: str | None = None
    created: bool = False
    approval_id: str | None = None
    approval_changed: bool = False
    disposition: ApprovalDisposition = "none"
    gate_verdicts: tuple[GateVerdict, ...] = ()
    provenance: PromotionProvenance | None = None


@dataclass(frozen=True, slots=True)
class _ApprovedChoice:
    selection: Selection
    context: Mapping[str, object] | None


def load_approved_selections(
    config: Config,
    *,
    as_of: datetime | None = None,
    semantics: Mapping[str, TruthSemantics] | None = None,
    matrices: Mapping[str, pl.DataFrame] | None = None,
    dataset_id: str | None = None,
) -> SelectionMap:
    """Read approved choices; current serving never elects or writes a release."""
    if as_of is not None:
        return _release_as_of(config, as_of, semantics) or {}
    return {
        key: choice.selection
        for key, choice in _approved_choices(
            config, semantics, dataset_id, matrices
        ).items()
    }


def _product_signatures(
    config: Config, matrices: Mapping[str, pl.DataFrame] | None
) -> dict[str, tuple[frozenset[str], tuple[str, ...]]]:
    from grounded_weather_forecast.dataset.matrix import (  # noqa: PLC0415
        matrix_feature_columns,
        matrix_path,
        matrix_sources,
    )

    signatures = {}
    for product in ("hourly", "daily"):
        frame = (matrices or {}).get(product)
        if frame is None:
            path = matrix_path(config.dataset.dir, product, "live")
            if path.exists():
                try:
                    frame = pl.DataFrame(schema=pl.read_parquet_schema(path))
                except (OSError, pl.exceptions.PolarsError):
                    signatures[product] = (frozenset(), ())
                    continue
        if frame is not None:
            signatures[product] = (
                frozenset(matrix_sources(frame)),
                matrix_feature_columns(frame),
            )
    if "hourly" in signatures:
        signatures["minutely"] = signatures["hourly"]
    return signatures


def _approved_choice_compatible(
    key: SliceKey,
    selected: Selection,
    context: Mapping[str, object] | None,
    signatures: Mapping[str, tuple[frozenset[str], tuple[str, ...]]],
    dataset: str,
    code: str,
    semantics: Mapping[str, TruthSemantics] | None,
) -> bool:
    expected = (
        (semantics or {}).get(key[1], TruthSemantics.INSTANTANEOUS)
        if key[0] == "hourly"
        else TruthSemantics.INSTANTANEOUS
    )
    if selected.code_version != code or selected.truth_semantics != expected.value:
        return False
    signature = signatures.get(key[0])
    compatibility = _evaluation_compatibility(context, key[1])
    if context is None:
        return selected.dataset_fingerprint == dataset
    if signature is not None:
        return (
            compatibility is not None
            and (compatibility[1], compatibility[3]) == signature
        )
    # Legacy ledgers can still justify the identical dataset. Without a current
    # matrix signature they cannot authorize a different dataset.
    return selected.dataset_fingerprint == dataset


def _approved_context_matches(
    context: Mapping[str, object] | None,
    selected: Selection,
    key: SliceKey,
    configuration: object,
    code: str,
) -> bool:
    if context is None:
        return True
    semantic_map = context.get("semantics")
    return (
        context.get("config_fingerprint") == configuration
        and context.get("code_version") == code
        and context.get("dataset_fingerprint", selected.dataset_fingerprint)
        == selected.dataset_fingerprint
        and isinstance(semantic_map, Mapping)
        and semantic_map.get(key[1]) == selected.truth_semantics
    )


def _approved_records(
    config: Config,
    as_of: datetime | None = None,
    *,
    match_dataset: bool = False,
    snapshot: ApprovalSnapshot | None = None,
) -> list[ApprovedRelease]:
    snapshot = snapshot or read_approval_snapshot(
        config.artifacts_dir / "releases", as_of=as_of
    )
    return snapshot.records(
        _compatible_releases(config, match_dataset=match_dataset, snapshot=snapshot)
    )


def _approved_choices(
    config: Config,
    semantics: Mapping[str, TruthSemantics] | None,
    dataset_id: str | None = None,
    matrices: Mapping[str, pl.DataFrame] | None = None,
    *,
    snapshot: ApprovalSnapshot | None = None,
) -> dict[SliceKey, _ApprovedChoice]:
    dataset = dataset_id or dataset_fingerprint(config)
    signatures = _product_signatures(config, matrices)
    code = code_identity()
    pins = _pins(config)
    selections: dict[SliceKey, _ApprovedChoice] = {}
    for record in _approved_records(config, snapshot=snapshot):
        raw = record.release
        approved = _selections_from_release(raw) or {}
        recorded = _contexts_by_evaluation_id(raw.get("evaluation_contexts"))
        for key, selected in approved.items():
            context = (
                None
                if selected.legacy_context_missing
                else recorded.get(selected.evaluation_id or "")
            )
            if not _approved_context_matches(
                context, selected, key, raw["config_fingerprint"], code
            ):
                continue
            if key in selections or not _approved_choice_compatible(
                key, selected, context, signatures, dataset, code, semantics
            ):
                continue
            selections[key] = _ApprovedChoice(
                replace(
                    selected,
                    pinned=pins.get(key[:2]) == selected.method_id,
                    legacy_context_missing=context is None
                    and selected.evaluation_id is not None,
                ),
                context,
            )
    return selections


def _pins(config: Config) -> dict[tuple[str, str], str]:
    """``[predict.methods]`` keys are ``"<product>.<variable>"``."""
    pinned: dict[tuple[str, str], str] = {}
    for key, method_id in config.predict.methods.items():
        product, _, variable = key.partition(".")
        if product and variable:
            pinned[(product, variable)] = method_id
    return pinned


def _load_evidence(
    config: Config,
    scores_dir: Path,
    as_of: datetime | None,
    semantics: Mapping[str, TruthSemantics] | None,
    dataset_id: str | None = None,
) -> ScoreScanResult:
    identity = ScoreIdentity(
        dataset_id or dataset_fingerprint(config),
        config_fingerprint(config),
        code_identity(),
        as_of,
        semantics,
    )
    return compatible_slices(scores_dir, identity, config.promotion)


def _pool_sibling_scores(
    key: SliceKey,
    frame: pl.DataFrame,
    slices: Mapping[SliceKey, pl.DataFrame],
) -> pl.DataFrame:
    """Widen a far daily slice's scores to its D3-10 pool siblings.

    The score scanner chooses evidence per fine bucket, but the
    pooled far-bucket gate can only pool what it can see — without the
    siblings of the same evaluation, serving would never reach the pooled
    promotions the report shows.
    """
    product, variable, bucket = key
    pool = _DAILY_GATE_POOL.get(bucket) if product == "daily" else None
    if not pool:
        return frame
    evaluation_id = str(frame["evaluation_id"][0])
    siblings = [
        sibling
        for sibling_bucket in pool
        if (sibling := slices.get((product, variable, sibling_bucket))) is not None
        and str(sibling["evaluation_id"][0]) == evaluation_id
    ]
    return pl.concat(siblings, how="diagonal_relaxed") if siblings else frame


def _eprocess_store_for(
    config: Config, product: str, cache: dict[str, EProcessStore]
) -> EProcessStore | None:
    """Read-only e-process wealth for the ``seq_mcs`` rule.

    Only ``report`` persists the store; selection's in-memory updates are
    cursor-deduped and deterministic, so a reader that runs before the
    nightly writer reaches the same wealth the writer will record.
    """
    if config.promotion.rule != "seq_mcs":
        return None
    return product_eprocess_store(config, product, cache)


def _selection_payload(
    selections: Mapping[SliceKey, Selection],
) -> dict[str, dict[str, object]]:
    return {
        ".".join(key): {
            "method_id": selected.method_id,
            "reason": selected.reason,
            "evaluation_id": selected.evaluation_id,
            "dataset_fingerprint": selected.dataset_fingerprint,
            "code_version": selected.code_version,
            "n": selected.n,
            "mae": selected.mae,
            "truth_semantics": selected.truth_semantics,
            "retained": selected.retained,
            **(
                {"legacy_context_missing": True}
                if selected.legacy_context_missing
                else {}
            ),
            "training_cutoff": selected.training_cutoff.isoformat()
            if selected.training_cutoff is not None
            else None,
        }
        for key, selected in sorted(selections.items())
    }


def _contexts_by_evaluation_id(
    raw_contexts: object,
) -> dict[str, Mapping[str, object]]:
    """Well-formed release contexts keyed by their evaluation id."""
    if not isinstance(raw_contexts, (list, tuple)):
        return {}
    contexts: dict[str, Mapping[str, object]] = {}
    for raw_context in raw_contexts:
        if not isinstance(raw_context, dict):
            continue
        context = cast(Mapping[str, object], raw_context)
        evaluation_id = context.get("evaluation_id")
        if isinstance(evaluation_id, str) and evaluation_id:
            contexts[evaluation_id] = context
    return contexts


def _evaluation_compatibility(
    context: Mapping[str, object] | None, variable: str
) -> EvaluationCompatibility | None:
    """The serving-relevant identity shared by live release cohorts."""
    if context is None or context.get("source_kind") != "live":
        return None
    raw_source_set = context.get("source_set_json")
    raw_feature_set = context.get("feature_set_json")
    semantics = context.get("semantics")
    if (
        not isinstance(raw_source_set, str)
        or not isinstance(raw_feature_set, str)
        or not isinstance(semantics, Mapping)
    ):
        return None
    try:
        sources = json.loads(raw_source_set)
        features = json.loads(raw_feature_set)
    except (json.JSONDecodeError, TypeError):
        return None
    semantic = semantics.get(variable)
    if (
        not isinstance(sources, list)
        or not sources
        or not all(isinstance(source, str) and source for source in sources)
        or not isinstance(features, list)
        or not all(isinstance(feature, str) and feature for feature in features)
        or not isinstance(semantic, str)
        or not semantic
    ):
        return None
    return (
        "live",
        frozenset(cast(str, source) for source in sources),
        semantic,
        tuple(cast(str, feature) for feature in features),
    )


def _make_release(
    config: Config,
    selections: Mapping[SliceKey, Selection],
    evidence: EvidenceSummary,
    dataset_id: str | None = None,
    *,
    promoted_at: datetime | None = None,
) -> ModelRelease:
    return ModelRelease.create(
        dataset=dataset_id or dataset_fingerprint(config),
        configuration=config_fingerprint(config),
        evaluation_ids=evidence.evaluation_ids,
        evaluation_contexts=evidence.contexts,
        training_cutoff=evidence.training_cutoff,
        selections=_selection_payload(selections),
        promoted_at=promoted_at,
    )


def _compatible_releases(
    config: Config, *, match_dataset: bool, snapshot: ApprovalSnapshot | None = None
) -> list[dict[str, object]]:
    """Ledger entries written under a config this system can still act on.

    ``match_dataset`` additionally pins the dataset fingerprint, which is what
    a historical replay needs. The live gate deliberately leaves it off — see
    ``_eligible_release_ids``.
    """
    configuration = config_fingerprint(config)
    dataset = dataset_fingerprint(config) if match_dataset else None
    snapshot = snapshot or read_approval_snapshot(config.artifacts_dir / "releases")
    required = (
        "release_id",
        "promoted_at",
        "dataset_fingerprint",
        "config_fingerprint",
        "selections",
    )
    return [
        raw
        for raw in snapshot.releases
        if all(field in raw for field in required)
        and isinstance(raw["selections"], dict)
        and raw.get("config_fingerprint") == configuration
        and (dataset is None or raw.get("dataset_fingerprint") == dataset)
    ]


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
            evaluation_id=str(selection_mapping["evaluation_id"])
            if selection_mapping.get("evaluation_id")
            and selection_mapping["evaluation_id"] != "unknown"
            else None,
            dataset_fingerprint=str(
                selection_mapping.get("dataset_fingerprint") or dataset
            ),
            release_id=release_id,
            code_version=_release_selection_code_version(release, selection_mapping),
            truth_semantics=_release_selection_truth_semantics(
                release, selection_mapping, variable
            ),
            retained=bool(selection_mapping.get("retained", False)),
            training_cutoff=aware_timestamp(selection_mapping.get("training_cutoff")),
            legacy_context_missing=selection_mapping.get("legacy_context_missing")
            is True,
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


def _release_selection_truth_semantics(
    release: Mapping[str, object],
    selection: Mapping[str, object],
    variable: str,
) -> str | None:
    """Truth target used to evaluate one persisted serving selection."""
    direct = selection.get("truth_semantics")
    if isinstance(direct, str) and direct:
        return direct
    evaluation_id = selection.get("evaluation_id")
    contexts = release.get("evaluation_contexts")
    if not isinstance(contexts, list):
        return None
    values: set[str] = set()
    for raw_context in contexts:
        if not isinstance(raw_context, dict):
            continue
        context = cast(Mapping[str, object], raw_context)
        raw_semantics = context.get("semantics")
        semantic = (
            raw_semantics.get(variable) if isinstance(raw_semantics, Mapping) else None
        )
        if isinstance(semantic, str) and (
            evaluation_id is None or context.get("evaluation_id") == evaluation_id
        ):
            values.add(semantic)
    return next(iter(values)) if len(values) == 1 else None


def _eligible_release_ids(
    config: Config,
    selections: Mapping[SliceKey, Selection],
    evaluation_contexts: tuple[dict[str, object], ...] = (),
    *,
    candidates: Mapping[SliceKey, tuple[Selection, ...]] | None = None,
    snapshot: ApprovalSnapshot | None = None,
    choice_contexts: Mapping[LiveKey, Mapping[str, object] | None] | None = None,
) -> EligibleReleases:
    """Release cohorts matching each selected method and evaluation context.

    Deliberately ignores the dataset fingerprint. That fingerprint hashes row
    counts over every parquet artifact, so it rotates on *every ingested
    observation* — keying live evidence to it means any slice whose truth
    arrives after one rebuild cycle can never be scored, which is every lead
    bucket at or beyond 24h. ``config_fingerprint`` pins settings while
    ``code_version`` includes a source digest. Source set and per-variable
    truth semantics are matched from the release contexts because
    auto-discovered providers can change without either identity changing.
    Recency is applied to served rows rather than to immutable release creation
    times.
    """
    choices = {
        (*key, selected.method_id): selected for key, selected in selections.items()
    }
    for key, references in (candidates or {}).items():
        choices.update(
            {(*key, reference.method_id): reference for reference in references}
        )
    eligible: dict[LiveKey, set[str]] = {key: set() for key in choices}
    current_contexts = _contexts_by_evaluation_id(evaluation_contexts)
    for approved in _approved_records(config, snapshot=snapshot):
        raw = approved.release
        release_id = raw.get("release_id")
        raw_selections = raw.get("selections")
        if not isinstance(release_id, str) or not isinstance(raw_selections, dict):
            continue
        release_contexts = _contexts_by_evaluation_id(raw.get("evaluation_contexts"))
        for live_key, selected in choices.items():
            key = live_key[:3]
            raw_selected = raw_selections.get(".".join(key))
            if not isinstance(raw_selected, dict):
                continue
            selection_mapping = cast(Mapping[str, object], raw_selected)
            current_context = (
                choice_contexts.get(live_key)
                if choice_contexts is not None
                else current_contexts.get(selected.evaluation_id or "")
            )
            release_context = release_contexts.get(
                str(selection_mapping.get("evaluation_id") or "")
            )
            if selection_mapping.get("legacy_context_missing") is True:
                release_context = None
            if (
                selection_mapping.get("method_id") == selected.method_id
                and selected.code_version is not None
                and _release_selection_code_version(raw, selection_mapping)
                == selected.code_version
                and (
                    current_compatibility := _evaluation_compatibility(
                        current_context, key[1]
                    )
                )
                is not None
                and _evaluation_compatibility(release_context, key[1])
                == current_compatibility
            ):
                eligible[live_key].add(release_id)
    return {key: frozenset(releases) for key, releases in eligible.items()}


def _release_as_of(
    config: Config,
    as_of: datetime,
    semantics: Mapping[str, TruthSemantics] | None,
) -> SelectionMap | None:
    """Newest release that genuinely existed by a historical issue time."""
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)
    current_code = code_identity()
    for approved in _approved_records(config, as_of, match_dataset=True):
        raw = approved.release
        selections = _selections_from_release(raw)
        raw_selections = raw.get("selections")
        if (
            not selections
            or not isinstance(raw_selections, dict)
            or len(selections) != len(raw_selections)
            or any(
                selected.code_version != current_code
                for selected in selections.values()
            )
            or any(
                key[0] == "hourly"
                and key[1] in (semantics or {})
                and selected.truth_semantics != (semantics or {})[key[1]].value
                for key, selected in selections.items()
            )
        ):
            continue
        return selections
    return None


_RETAINED_REASON = "retained incumbent within one SE of best (winner-curse guard)"
# Retention only ever overrides an argmin pick; a gate-forced reference or a
# pooled winner was not selected on the minimum, so there is no curse to guard.
_RETENTION_GATES = (None, "eligibility")


def _incumbent_methods(
    config: Config, *, snapshot: ApprovalSnapshot | None = None
) -> dict[SliceKey, tuple[str, str | None]]:
    """The previous release's (method_id, truth_semantics) per slice.

    Canonical approval sequence among config-compatible releases, with the dataset
    and code filters deliberately off: the dataset fingerprint rotates on
    every ingest and the code digest on every deploy, so an exact-match
    lookup would make retention dead code on precisely the re-runs it exists
    for. Only the method identity is read — every statistical input comes
    from the current board.
    """

    releases = _approved_records(config, snapshot=snapshot)
    if not releases:
        return {}
    newest = releases[0].release
    selections = _selections_from_release(newest) or {}
    return {
        key: (selected.method_id, selected.truth_semantics)
        for key, selected in selections.items()
    }


def _retained_incumbent(
    config: Config,
    winner_row: Mapping[str, object],
    board: pl.DataFrame,
    frame: pl.DataFrame,
    incumbent: tuple[str, str | None] | None,
    evaluation_id: str,
    dataset_id: str | None = None,
) -> Selection | None:
    """The winner-curse guard: keep a statistically indistinguishable incumbent.

    Near-ties churn the served method while flattering the report's argmin
    number. When the previous release served a *different* method for this
    slice and that method is still eligible on the current board, with the
    same truth semantics, and within one bootstrap SE of today's winner,
    keep serving it. The live demotion gate still runs afterwards and keeps
    the last word; the config-pin pass overwrites retention entirely.
    """
    if incumbent is None:
        return None
    incumbent_method, incumbent_semantics = incumbent
    if (
        incumbent_method == str(winner_row["method_id"])
        or winner_row.get("gate") not in _RETENTION_GATES
        or incumbent_method == FALLBACK_METHOD
        # A demoted slice's persisted method is already the fallback;
        # retaining it by name would outlast the demotion's own re-hearing.
        or incumbent_method
        in gate_references(
            str(winner_row["variable"]),
            config.promotion,
            product=str(winner_row["product"]),
        )
    ):
        return None
    slice_semantics = str(frame["semantics"][0])
    if incumbent_semantics is not None and incumbent_semantics != slice_semantics:
        return None
    candidates = eligible_board_rows(
        board.filter(pl.col("method_id") == incumbent_method)
    ).sort("mae")
    if candidates.is_empty():
        return None
    incumbent_row = candidates.row(0, named=True)
    if incumbent_row.get("mae") is None:
        return None
    if not should_retain_incumbent(
        frame,
        winner_row,
        incumbent_method,
        promotion=config.promotion,
    ):
        return None
    return Selection(
        method_id=incumbent_method,
        reason=_RETAINED_REASON,
        n=int(incumbent_row["n"]),
        mae=float(incumbent_row["mae"]),
        evaluation_id=evaluation_id,
        dataset_fingerprint=dataset_id or dataset_fingerprint(config),
        code_version=str(frame["code_version"][0]),
        truth_semantics=slice_semantics,
        retained=True,
    )


def _selection_from_row(
    row: Mapping[str, object],
    frame: pl.DataFrame,
    dataset: str,
    reason: str,
) -> Selection:
    return Selection(
        method_id=str(row["method_id"]),
        reason=reason,
        n=int(cast(int, row["n"])),
        mae=float(cast(float, row["mae"])),
        evaluation_id=str(frame["evaluation_id"][0]),
        dataset_fingerprint=dataset,
        code_version=str(frame["code_version"][0]),
        truth_semantics=str(
            row.get("truth_semantics") or frame["semantics"][0] or "inst"
        ),
    )


def _pinned_selection(
    method: str, board: pl.DataFrame, frame: pl.DataFrame, dataset: str
) -> Selection:
    rows = (
        board.filter(pl.col("method_id") == method) if not board.is_empty() else board
    )
    selected = (
        _selection_from_row(rows.row(0, named=True), frame, dataset, "pinned in config")
        if not rows.is_empty()
        else Selection(
            method,
            "pinned in config",
            dataset_fingerprint=dataset,
            code_version=str(frame["code_version"][0]),
            truth_semantics=str(frame["semantics"][0] or "inst"),
        )
    )
    return replace(selected, pinned=True, retained=False)


def _reference_candidates(
    key: SliceKey,
    board: pl.DataFrame,
    frame: pl.DataFrame,
    config: Config,
    dataset: str,
) -> tuple[Selection, ...]:
    references = gate_references(key[1], config.promotion, product=key[0])
    eligible = eligible_board_rows(board)
    return (
        tuple(
            _selection_from_row(row, frame, dataset, "reference fallback")
            for method in references
            for row in eligible.filter(pl.col("method_id") == method).iter_rows(
                named=True
            )
        )
        if not eligible.is_empty()
        else ()
    )


def select_methods(
    config: Config,
    scores_dir: Path,
    as_of: datetime | None = None,
    *,
    semantics: Mapping[str, TruthSemantics] | None = None,
    dataset_id: str | None = None,
    products: frozenset[str] | None = None,
) -> SelectionMap:
    """Compatibility API: explicit promotion, or strict historical release lookup."""
    if as_of is not None:
        return load_approved_selections(config, as_of=as_of, semantics=semantics)
    return promote_methods(
        config,
        scores_dir,
        semantics=semantics,
        dataset_id=dataset_id,
        products=products,
    ).selections


def promote_methods(
    config: Config,
    scores_dir: Path,
    *,
    semantics: Mapping[str, TruthSemantics] | None = None,
    dataset_id: str | None = None,
    products: frozenset[str] | None = None,
) -> PromotionResult:
    """Serialize elections and approval commits against the same incumbent state."""
    from grounded_weather_forecast.reports.evidence import (  # noqa: PLC0415
        record_live_gate,
    )

    telemetry = _PromotionTelemetry()
    try:
        with locked_path(config.artifacts_dir / "releases" / ".approval-ledger"):
            return _promote_methods(
                config, scores_dir, semantics, dataset_id, products, telemetry
            )
    finally:
        if telemetry.provenance is not None:
            record_live_gate(
                config,
                telemetry.verdicts,
                now=telemetry.provenance.recorded_at,
                provenance=telemetry.provenance,
            )


def _elect_slices(
    config: Config,
    slices: Mapping[SliceKey, pl.DataFrame],
    dataset: str,
    cutoffs: Mapping[SliceKey, datetime | None],
    snapshot: ApprovalSnapshot,
) -> tuple[dict[SliceKey, Selection], dict[SliceKey, tuple[Selection, ...]]]:
    pinned = _pins(config)
    incumbents = _incumbent_methods(config, snapshot=snapshot)
    selected: dict[SliceKey, Selection] = {}
    candidates: dict[SliceKey, tuple[Selection, ...]] = {}
    references = union_references(config.promotion)
    stores: dict[str, EProcessStore] = {}
    for key, frame in slices.items():
        board = leaderboard(frame, references=references)
        # A carried choice must still have scored replacements available when
        # no new winner qualifies. Missing candidates never synthesize a fallback.
        candidates[key] = tuple(
            replace(candidate, training_cutoff=cutoffs.get(key))
            for candidate in _reference_candidates(key, board, frame, config, dataset)
        )
        if (method := pinned.get(key[:2])) is not None:
            pinned_choice = _pinned_selection(method, board, frame, dataset)
            selected[key] = replace(
                pinned_choice,
                training_cutoff=cutoffs.get(key)
                if pinned_choice.evaluation_id is not None
                else None,
            )
            continue
        winners = slice_winners(
            board,
            scores=_pool_sibling_scores(key, frame, slices),
            rule=config.promotion.rule,
            alpha=config.promotion.alpha,
            promotion=config.promotion,
            eprocess_store=_eprocess_store_for(config, key[0], stores),
        )
        if winners.is_empty():
            continue
        row = winners.row(0, named=True)
        reason = "lowest backtest MAE among promotable common-case methods"
        if row.get("gate") == "pooled_D3-10":
            reason += " (gated on pooled D3-10 evidence)"
        choice = _retained_incumbent(
            config,
            row,
            board,
            frame,
            incumbents.get(key),
            str(frame["evaluation_id"][0]),
            dataset,
        ) or _selection_from_row(row, frame, dataset, reason)
        selected[key] = replace(choice, training_cutoff=cutoffs.get(key))
    return selected, candidates


def _evaluated_context(
    contexts: Mapping[str, EvaluationContext], choice: Selection, key: SliceKey
) -> Mapping[str, object] | None:
    if choice.evaluation_id is None:
        return None  # Unscored explicit pins make no evaluation claim.
    context = contexts.get(choice.evaluation_id)
    if context is None:
        raise PromotionEvidenceError(choice.evaluation_id, "context")
    return _project_context(context.payload(choice.evaluation_id), key)


def _gate_completed(
    config: Config,
    selections: dict[SliceKey, Selection],
    contexts: dict[SliceKey, Mapping[str, object] | None],
    candidates: Mapping[SliceKey, tuple[Selection, ...]],
    products: frozenset[str] | None,
    scanned: ScoreScanResult,
    snapshot: ApprovalSnapshot,
) -> tuple[GateVerdict, ...]:
    pinned = _pins(config)
    in_scope = {
        key: replace(choice, pinned=pinned.get(key[:2]) == choice.method_id)
        for key, choice in selections.items()
        if products is None or key[0] in products
    }
    diagnostics: list[GateVerdict] = []
    if in_scope:
        gate_contexts = {
            (*key, choice.method_id): contexts.get(key)
            for key, choice in in_scope.items()
        }
        gate_contexts.update(
            {
                (*key, c.method_id): _evaluated_context(scanned.contexts, c, key)
                for key, refs in candidates.items()
                for c in refs
            }
        )
        gated = apply_live_gate(
            in_scope,
            _live_verification(config, snapshot.cutoff),
            factor=config.promotion.live_gap_factor,
            min_n=config.promotion.min_live_n,
            candidates=candidates,
            eligible_releases=_eligible_release_ids(
                config,
                in_scope,
                candidates=candidates,
                snapshot=snapshot,
                choice_contexts=gate_contexts,
            ),
            diagnostics=diagnostics,
        )
        for key, choice in gated.items():
            if (
                choice.evaluation_id != in_scope[key].evaluation_id
                or choice.method_id != in_scope[key].method_id
            ):
                contexts[key] = _evaluated_context(scanned.contexts, choice, key)
        selections.update(gated)
    return tuple(diagnostics)


def _promote_methods(
    config: Config,
    scores_dir: Path,
    semantics: Mapping[str, TruthSemantics] | None,
    dataset_id: str | None,
    products: frozenset[str] | None,
    telemetry: _PromotionTelemetry,
) -> PromotionResult:
    snapshot = read_approval_snapshot(config.artifacts_dir / "releases")
    dataset = dataset_id or dataset_fingerprint(config)
    provenance = PromotionProvenance(
        snapshot.cutoff, dataset, config_fingerprint(config), code_identity()
    )
    telemetry.provenance = provenance
    scanned = _load_evidence(config, scores_dir, None, semantics, dataset)
    slices = {
        key: frame
        for key, frame in scanned.slices.items()
        if products is None or key[0] in products
    }
    approved = _approved_choices(config, semantics, dataset, snapshot=snapshot)
    carried = {key: value.selection for key, value in approved.items()}
    elected, candidates = _elect_slices(
        config, slices, dataset, scanned.cutoffs, snapshot
    )
    selections = carried | elected
    contexts: dict[SliceKey, Mapping[str, object] | None] = {
        key: choice.context for key, choice in approved.items()
    }

    contexts.update(
        {
            key: _evaluated_context(scanned.contexts, choice, key)
            for key, choice in elected.items()
        }
    )
    diagnostics = _gate_completed(
        config, selections, contexts, candidates, products, scanned, snapshot
    )
    telemetry.verdicts = diagnostics
    final_contexts = ValidatedContexts(
        tuple(
            _project_context(contexts[key], key)
            for key in selections
            if contexts.get(key) is not None
        )
    )
    evidence = EvidenceSummary.from_validated_contexts(
        final_contexts,
        {key: choice.training_cutoff for key, choice in selections.items()},
        {key: choice.evaluation_id for key, choice in selections.items()},
        legacy_missing=frozenset(
            key for key, choice in selections.items() if choice.legacy_context_missing
        ),
    )
    records = _approved_records(config, snapshot=snapshot)
    current = records[0] if records else None
    if not selections or _selection_payload(selections) == _selection_payload(carried):
        return PromotionResult(
            carried,
            str(current.release["release_id"]) if current else None,
            approval_id=current.approval_id if current else None,
            disposition="unchanged" if current else "none",
            gate_verdicts=tuple(diagnostics),
            provenance=provenance,
        )
    release = _make_release(config, selections, evidence, dataset)
    commit = publish_approval(
        release, config.artifacts_dir / "releases", current, snapshot=snapshot
    )
    return PromotionResult(
        {
            key: replace(choice, release_id=release.release_id)
            for key, choice in selections.items()
        },
        release.release_id,
        commit.created,
        commit.approval_id,
        commit.changed,
        commit.disposition,
        tuple(diagnostics),
        provenance,
    )


def _project_context(
    context: Mapping[str, object] | None, key: SliceKey
) -> dict[str, object]:
    if context is None:
        raise PromotionEvidenceError("unknown", "context")
    semantics = context.get("semantics")
    if not isinstance(semantics, Mapping) or key[1] not in semantics:
        raise PromotionEvidenceError(
            str(context.get("evaluation_id")), f"semantics.{key[1]}"
        )
    return dict(context) | {
        "semantics": {key[1]: cast(Mapping[str, object], semantics)[key[1]]}
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
    if selected.pinned or selected.mae is None or not math.isfinite(selected.mae):
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


def _replacement_rank(
    key: SliceKey,
    candidate: Selection,
    pooled: Mapping[LiveKey, tuple[int, float]],
    winner_live_mae: float,
    factor: float,
    min_n: int,
    order: int,
) -> tuple[float, int, int] | None:
    if (
        candidate.mae is None
        or not math.isfinite(candidate.mae)
        or candidate.mae < 0
        or candidate.mae >= winner_live_mae
    ):
        return None
    score = candidate.mae
    measured = pooled.get((*key, candidate.method_id))
    if measured is not None and measured[0] >= min_n:
        live_mae = measured[1]
        if live_mae >= winner_live_mae or live_mae > factor * candidate.mae:
            return None
        score = max(score, live_mae)
    return score, 0 if candidate.method_id == FALLBACK_METHOD else 1, order


def _gate_fallback(
    key: SliceKey,
    selected: Selection,
    choices: tuple[Selection, ...],
    pooled: Mapping[LiveKey, tuple[int, float]],
    *,
    factor: float,
    min_n: int,
) -> Selection | None:
    winner_live_mae = pooled[(*key, selected.method_id)][1]
    ranked = [
        (rank, candidate)
        for order, candidate in enumerate(choices)
        if candidate.method_id != selected.method_id
        and (
            rank := _replacement_rank(
                key, candidate, pooled, winner_live_mae, factor, min_n, order
            )
        )
        is not None
    ]
    return min(ranked, key=lambda pair: pair[0])[1] if ranked else None


def apply_live_gate(
    selections: dict[tuple[str, str, str], Selection],
    live: pl.DataFrame,
    *,
    factor: float,
    min_n: int,
    fallbacks: Mapping[SliceKey, Selection] | None = None,
    candidates: Mapping[SliceKey, tuple[Selection, ...]] | None = None,
    eligible_releases: frozenset[str] | EligibleReleases | None = None,
    diagnostics: list[GateVerdict] | None = None,
) -> dict[tuple[str, str, str], Selection]:
    """Close the self-verification loop: demote methods that underdeliver live.

    A selected method whose realized served MAE exceeds its backtest promise
    can move to an eligible reference only when the quality guard passes.
    Current backtest error is a proxy; mature live reference error can veto
    it. Failed gates with no qualified replacement keep serving and record
    the blocked verdict separately through diagnostics and the live-gate ledger.

    Evidence is pooled across ``eligible_releases`` rather than matched to one
    release, mirroring ``ArtifactStore.load_latest_state``: identity that
    rotates with data volume cannot be a precondition for scoring, or the
    slices that most need watching are the ones that never get watched.
    ``None`` means no ledger filter — every release-tagged row is eligible.
    ``fallbacks`` is a compatibility adapter: its scored candidates obey the
    same quality guard as ``candidates``; unscored fallbacks are rejected.
    """
    pooled = _pooled_live_skill(live, eligible_releases)
    if not pooled:
        return selections
    gated = dict(selections)
    for key, selected in selections.items():
        verdict = _live_verdict(pooled, key, selected, factor=factor, min_n=min_n)
        if verdict is None:
            continue
        choices = (candidates or {}).get(key, ())
        if fallbacks is not None and key in fallbacks:
            choices = (*choices, fallbacks[key])
        fallback = _gate_fallback(
            key, selected, choices, pooled, factor=factor, min_n=min_n
        )
        if diagnostics is not None:
            n, live_mae = pooled[(*key, selected.method_id)]
            diagnostics.append(
                GateVerdict(
                    *key,
                    selected.method_id,
                    selected.evaluation_id,
                    selected.release_id,
                    "blocked" if fallback is None else "replaced",
                    verdict
                    if fallback is not None
                    else verdict.replace("demoted", "live gate failed for", 1)
                    + "; replacement blocked: no reference passes quality guard",
                    live_mae,
                    float(selected.mae or 0.0),
                    n,
                )
            )
        if fallback is None:
            continue
        gated[key] = replace(
            fallback,
            reason=verdict,
        )
    return gated


def no_evidence_reason(
    config: Config,
    scores_dir: Path,
    *,
    dataset_id: str | None = None,
    promoted_fit_unavailable: bool = False,
    fit_failures: Sequence[FitFailure] = (),
    approved_selections: bool = False,
    missing_product_approvals: bool = False,
) -> str:
    """Why serving is degraded: cold start vs invalidated evidence.

    Approved compatible releases survive data-only rebuilds. Without an approval,
    distinguish absent evidence from changed scientific identity so recovery has
    an actionable cause. Caller-supplied selections alone cannot establish approval.

    Only compatible live identities may use the prediction-local fitting
    outcome; missing or invalidated evidence keeps its more actionable reason.
    """
    if missing_product_approvals:
        return "no approved hourly or daily selections; run `backtest --source live` then `report`"
    if approved_selections:
        return (
            describe_fit_failures(fit_failures)
            if fit_failures
            else "approved selections produced no finite hourly or daily forecast from training available at issuance; keep polling"
        )
    # Retry the identity diagnosis once against a fresh directory listing
    # before tolerating missing files, so the diagnosis never reflects a
    # half-pruned directory.
    try:
        found_any, live_datasets, live_configs, live_code_versions = (
            _live_identity_sets(scores_dir, skip_missing=False)
        )
    except FileNotFoundError:
        found_any, live_datasets, live_configs, live_code_versions = (
            _live_identity_sets(scores_dir, skip_missing=True)
        )
    if not found_any:
        return (
            "cold start: no backtest scores exist yet; run "
            "`backtest --source live` then `report` once the archive has folds"
        )
    if not live_datasets:
        return (
            "no live backtest evidence yet (synthetic evidence is never "
            "promoted); keep polling and run `backtest --source live`"
        )
    if code_identity() not in live_code_versions:
        return (
            "implementation changed since the last backtest; re-run "
            "`backtest --source live` then `report`"
        )
    if config_fingerprint(config) not in live_configs:
        return (
            "config changed since the last backtest; re-run "
            "`backtest --source live` then `report`"
        )
    if (dataset_id or dataset_fingerprint(config)) not in live_datasets:
        return (
            "dataset fingerprint changed since the last backtest and no compatible "
            "approval exists; re-run `backtest --source live` "
            "then `report`"
        )
    if fit_failures:
        return describe_fit_failures(fit_failures)
    if promoted_fit_unavailable:
        return (
            "at least one compatible promoted selection could not be fitted "
            "from training available at issuance; keep polling as live issuance "
            "advances, and rebuild "
            "only when newly available truth has not yet been incorporated"
        )
    return (
        "live evidence exists but no promoted release produced a finite hourly "
        "or daily forecast; keep polling"
    )


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
                return replace(
                    found, reason="pinned in config", pinned=True, retained=False
                )
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
            "truth_semantics": chosen.truth_semantics,
            "retained": chosen.retained,
        }
        for (product, variable, bucket), chosen in sorted(selections.items())
    ]
    return pl.DataFrame(rows)
