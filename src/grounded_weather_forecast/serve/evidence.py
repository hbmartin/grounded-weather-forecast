"""Compact, validated evidence for immutable serving releases."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Literal, cast

import polars as pl

CONTEXT_COLUMNS = (
    "source_kind",
    "source_set_json",
    "feature_set_json",
    "window",
    "code_version",
    "config_fingerprint",
)
_CONTEXT_IDENTITY_COLUMNS = ("dataset_fingerprint", "evaluation_created_at")
_LOGGER = logging.getLogger(__name__)


class PromotionEvidenceError(ValueError):
    """Invalid final release provenance, with diagnostics safe for command outcomes."""

    def __init__(
        self,
        evaluation: str,
        field: str,
        left: object = None,
        right: object = None,
        *,
        conflict: bool = False,
    ) -> None:
        self.evaluation_id = evaluation
        self.field = field
        self.left = left
        self.right = right
        self.failure_kind = (
            "promotion_evidence_conflict" if conflict else "promotion_evidence_invalid"
        )
        kind = "conflicting" if conflict else "missing or invalid"
        super().__init__(
            f"{kind} evaluation evidence {evaluation}: {field}; recorded={left!r}, proposed={right!r}"
        )


@dataclass(frozen=True, slots=True)
class GateVerdict:
    product: str
    variable: str
    lead_bucket: str
    method_id: str
    evaluation_id: str | None
    release_id: str | None
    verdict: Literal["blocked", "replaced"]
    detail: str
    live_mae: float
    backtest_mae: float
    n: int


@dataclass(slots=True)
class EvaluationContext:
    metadata: tuple[str, ...]
    identity: tuple[object, ...]
    semantics: dict[str, str] = field(default_factory=dict)

    def payload(self, evaluation_id: str) -> dict[str, object]:
        source, sources, features, window, code, configuration = self.metadata
        payload: dict[str, object] = {
            "evaluation_id": evaluation_id,
            "source_kind": source,
            "source_set_json": sources,
            "feature_set_json": features,
            "semantics": dict(self.semantics),
            "window": window,
            "code_version": code,
            "config_fingerprint": configuration,
        }
        for name, value in zip(_CONTEXT_IDENTITY_COLUMNS, self.identity, strict=False):
            if value is not None:
                payload[name] = (
                    value.isoformat() if isinstance(value, datetime) else value
                )
        return payload


@dataclass(slots=True)
class ContextAccumulator:
    contexts: dict[str, EvaluationContext] = field(default_factory=dict)
    invalid: set[str] = field(default_factory=set)

    def reject(self, evaluation_id: str) -> None:
        if evaluation_id not in self.invalid:
            _LOGGER.warning(
                "Ignoring conflicting evaluation evidence: %s", evaluation_id
            )
        self.invalid.add(evaluation_id)

    def add(self, frame: pl.DataFrame) -> None:
        identity_columns = [
            name for name in _CONTEXT_IDENTITY_COLUMNS if name in frame.columns
        ]
        compact = frame.select(
            "evaluation_id",
            *CONTEXT_COLUMNS,
            *identity_columns,
            "variable",
            "semantics",
        ).unique(maintain_order=True)
        for row in compact.iter_rows(named=True):
            evaluation_id = str(row["evaluation_id"])
            if evaluation_id in self.invalid:
                continue
            metadata = tuple(str(row[column]) for column in CONTEXT_COLUMNS)
            identity = tuple(row.get(column) for column in _CONTEXT_IDENTITY_COLUMNS)
            context = self.contexts.get(evaluation_id)
            if context is None:
                context = EvaluationContext(metadata, identity)
                self.contexts[evaluation_id] = context
            variable = str(row["variable"])
            semantic = str(row["semantics"] or "inst")
            if (
                context.metadata != metadata
                or context.identity != identity
                or (
                    variable in context.semantics
                    and context.semantics[variable] != semantic
                )
            ):
                self.reject(evaluation_id)
            else:
                context.semantics[variable] = semantic

    def payloads(self) -> tuple[dict[str, object], ...]:
        return tuple(
            self.contexts[key].payload(key)
            for key in sorted(self.contexts)
            if key not in self.invalid
        )


def evaluation_contexts(
    frames: Sequence[pl.DataFrame],
) -> tuple[dict[str, object], ...]:
    accumulator = ContextAccumulator()
    for frame in frames:
        accumulator.add(frame)
    return accumulator.payloads()


def merge_contexts(
    payloads: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Union attributed semantics without overwriting an evaluation's identity."""
    merged: dict[str, dict[str, object]] = {}
    for payload in payloads:
        evaluation = str(payload["evaluation_id"])
        identity = {
            k: v
            for k, v in payload.items()
            if k not in ("semantics", "training_cutoff")
        }
        semantics = payload.get("semantics")
        if not isinstance(semantics, Mapping):
            raise PromotionEvidenceError(evaluation, "semantics")
        current = merged.setdefault(evaluation, {})
        old_semantics = current.get("semantics", {})
        if not isinstance(old_semantics, Mapping):
            raise PromotionEvidenceError(evaluation, "semantics", old_semantics)
        semantics = cast(Mapping[str, object], semantics)
        old_semantics = cast(Mapping[str, object], old_semantics)
        for key, value in identity.items():
            if key in current and current[key] != value:
                raise PromotionEvidenceError(
                    evaluation, key, current[key], value, conflict=True
                )
        for variable, semantic in semantics.items():
            if variable in old_semantics and old_semantics[variable] != semantic:
                raise PromotionEvidenceError(
                    evaluation,
                    variable,
                    old_semantics[variable],
                    semantic,
                    conflict=True,
                )
        current.update(identity)
        current["semantics"] = dict(old_semantics) | dict(semantics)
    return merged


@dataclass(frozen=True, slots=True, init=False)
class ValidatedContexts:
    """A merged, read-only context set; construction always validates raw input."""

    contexts: Mapping[str, Mapping[str, object]]

    def __init__(self, payloads: Sequence[Mapping[str, object]]) -> None:
        contexts = {
            evaluation: MappingProxyType(
                context
                | {
                    "semantics": MappingProxyType(
                        cast(Mapping[str, object], context["semantics"])
                    )
                }
            )
            for evaluation, context in merge_contexts(payloads).items()
        }
        object.__setattr__(self, "contexts", MappingProxyType(contexts))


@dataclass(frozen=True, slots=True)
class EvidenceSummary:
    evaluation_ids: tuple[str, ...]
    contexts: tuple[dict[str, object], ...]
    training_cutoff: datetime | None

    @classmethod
    def from_payloads(
        cls,
        payloads: Sequence[Mapping[str, object]],
        cutoffs: Mapping[tuple[str, str, str], datetime | None],
        evaluations: Mapping[tuple[str, str, str], str | None],
        *,
        legacy_missing: frozenset[tuple[str, str, str]] = frozenset(),
    ) -> "EvidenceSummary":
        return cls.from_validated_contexts(
            ValidatedContexts(payloads),
            cutoffs,
            evaluations,
            legacy_missing=legacy_missing,
        )

    @classmethod
    def from_validated_contexts(
        cls,
        validated: ValidatedContexts,
        cutoffs: Mapping[tuple[str, str, str], datetime | None],
        evaluations: Mapping[tuple[str, str, str], str | None],
        *,
        legacy_missing: frozenset[tuple[str, str, str]] = frozenset(),
    ) -> "EvidenceSummary":
        contexts = validated.contexts
        projected: dict[str, dict[str, object]] = {}
        by_evaluation: dict[str, list[datetime]] = {}
        for key, evaluation in sorted(evaluations.items()):
            if evaluation is None or key in legacy_missing:
                continue
            if evaluation not in contexts:
                raise PromotionEvidenceError(evaluation, "context")
            context = dict(contexts[evaluation])
            semantics = context["semantics"]
            if not isinstance(semantics, Mapping) or key[1] not in semantics:
                raise PromotionEvidenceError(evaluation, f"semantics.{key[1]}")
            semantics = cast(Mapping[str, object], semantics)
            selected = projected.setdefault(evaluation, context | {"semantics": {}})
            selected_semantics = cast(dict[str, object], selected["semantics"])
            selected_semantics[key[1]] = semantics[key[1]]
            if cutoff := cutoffs.get(key):
                by_evaluation.setdefault(evaluation, []).append(cutoff)
        for evaluation, context in projected.items():
            cutoff = max(by_evaluation.get(evaluation, []), default=None)
            context["training_cutoff"] = cutoff.isoformat() if cutoff else None
        return cls(
            tuple(
                sorted({value for value in evaluations.values() if value is not None})
            ),
            tuple(projected[key] for key in sorted(projected)),
            max(
                (value for values in by_evaluation.values() for value in values),
                default=None,
            ),
        )

    @classmethod
    def from_contexts(
        cls,
        contexts: Mapping[str, EvaluationContext],
        cutoffs: Mapping[tuple[str, str, str], datetime | None],
        evaluations: Mapping[tuple[str, str, str], str | None],
    ) -> "EvidenceSummary":
        return cls.from_payloads(
            tuple(
                context.payload(evaluation) for evaluation, context in contexts.items()
            ),
            cutoffs,
            evaluations,
        )
