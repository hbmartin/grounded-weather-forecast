"""Compact, validated evidence for immutable serving releases."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

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


@dataclass(slots=True)
class EvaluationContext:
    metadata: tuple[str, ...]
    identity: tuple[object, ...]
    semantics: dict[str, str] = field(default_factory=dict)

    def payload(self, evaluation_id: str) -> dict[str, object]:
        source, sources, features, window, code, configuration = self.metadata
        return {
            "evaluation_id": evaluation_id,
            "source_kind": source,
            "source_set_json": sources,
            "feature_set_json": features,
            "semantics": dict(self.semantics),
            "window": window,
            "code_version": code,
            "config_fingerprint": configuration,
        }


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
            identity = tuple(row[column] for column in identity_columns)
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


@dataclass(frozen=True, slots=True)
class EvidenceSummary:
    evaluation_ids: tuple[str, ...]
    contexts: tuple[dict[str, object], ...]
    training_cutoff: datetime | None

    @classmethod
    def from_contexts(
        cls,
        contexts: Mapping[str, EvaluationContext],
        cutoffs: Mapping[tuple[str, str, str], datetime | None],
        evaluations: Mapping[tuple[str, str, str], str | None],
    ) -> "EvidenceSummary":
        attributed = {
            key: evaluation_id
            for key, evaluation_id in sorted(evaluations.items())
            if evaluation_id is not None
        }
        projected: dict[str, EvaluationContext] = {}
        for key, evaluation_id in attributed.items():
            context = contexts[evaluation_id]
            selected = projected.setdefault(
                evaluation_id, EvaluationContext(context.metadata, context.identity)
            )
            selected.semantics[key[1]] = context.semantics[key[1]]
        return cls(
            evaluation_ids=tuple(sorted(projected)),
            contexts=tuple(projected[key].payload(key) for key in sorted(projected)),
            training_cutoff=max(
                (
                    value
                    for key in attributed
                    if isinstance(value := cutoffs[key], datetime)
                ),
                default=None,
            ),
        )
