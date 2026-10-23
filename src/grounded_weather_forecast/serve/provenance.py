"""Pure decoding of recorded selection evidence, shared by serving and readers.

No filesystem, persistence, clock, or approval mutations belong in this module.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import cast

type EvaluationCompatibility = tuple[str, frozenset[str], str, tuple[str, ...]]


def evaluation_id(value: object) -> str | None:
    return value if isinstance(value, str) and value and value != "unknown" else None


def contexts_by_evaluation_id(raw_contexts: object) -> dict[str, Mapping[str, object]]:
    if not isinstance(raw_contexts, (list, tuple)):
        return {}
    return {
        identifier: cast(Mapping[str, object], context)
        for context in raw_contexts
        if isinstance(context, Mapping)
        and (identifier := evaluation_id(context.get("evaluation_id"))) is not None
    }


def selection_context(
    release: Mapping[str, object], selection: Mapping[str, object]
) -> Mapping[str, object] | None:
    if selection.get("legacy_context_missing") is True:
        return None
    return contexts_by_evaluation_id(release.get("evaluation_contexts")).get(
        evaluation_id(selection.get("evaluation_id")) or ""
    )


def _context_values(
    release: Mapping[str, object],
    selection: Mapping[str, object],
    field: str,
    variable: str | None = None,
) -> str | None:
    if (
        selection.get("evaluation_id") == "unknown"
        or selection.get("legacy_context_missing") is True
    ):
        return None
    identifier = evaluation_id(selection.get("evaluation_id"))
    contexts = contexts_by_evaluation_id(release.get("evaluation_contexts"))
    values = set()
    for key, context in contexts.items():
        if identifier is not None and key != identifier:
            continue
        value = context.get(field)
        if variable is not None:
            value = value.get(variable) if isinstance(value, Mapping) else None
        if isinstance(value, str) and value:
            values.add(value)
    return next(iter(values)) if len(values) == 1 else None


def selection_code_version(
    release: Mapping[str, object], selection: Mapping[str, object]
) -> str | None:
    direct = selection.get("code_version")
    return (
        direct
        if isinstance(direct, str) and direct
        else _context_values(release, selection, "code_version")
    )


def selection_truth_semantics(
    release: Mapping[str, object], selection: Mapping[str, object], variable: str
) -> str | None:
    direct = selection.get("truth_semantics")
    return (
        direct
        if isinstance(direct, str) and direct
        else _context_values(release, selection, "semantics", variable)
    )


def evaluation_compatibility(
    context: Mapping[str, object] | None, variable: str
) -> EvaluationCompatibility | None:
    if context is None or context.get("source_kind") != "live":
        return None
    semantics = context.get("semantics")
    if not isinstance(semantics, Mapping):
        return None
    try:
        sources = json.loads(str(context.get("source_set_json")))
        features = json.loads(str(context.get("feature_set_json")))
    except (ValueError, TypeError):
        return None
    semantic = semantics.get(variable)
    if (
        not isinstance(sources, list)
        or not sources
        or not all(isinstance(s, str) and s for s in sources)
        or not isinstance(features, list)
        or not all(isinstance(f, str) and f for f in features)
        or not isinstance(semantic, str)
        or not semantic
    ):
        return None
    return (
        "live",
        frozenset(cast(str, s) for s in sources),
        semantic,
        tuple(cast(str, f) for f in features),
    )


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


@dataclass(frozen=True, slots=True)
class SelectionProvenance:
    method_id: str | None
    evaluation_id: str | None
    dataset_fingerprint: str | None
    config_fingerprint: str | None
    code_version: str | None
    truth_semantics: str | None
    compatibility: EvaluationCompatibility | None
    valid_context: bool = True

    def compatible_with(self, other: "SelectionProvenance") -> bool:
        """Only complete recorded modern contexts can justify cross-release fallback."""
        return (
            self.valid_context
            and other.valid_context
            and self.compatibility is not None
            and self.compatibility == other.compatibility
            and self.method_id == other.method_id
            and self.code_version is not None
            and self.code_version == other.code_version
            and self.config_fingerprint is not None
            and self.config_fingerprint == other.config_fingerprint
            and self.truth_semantics == other.truth_semantics
        )


def selection_provenance(
    release: Mapping[str, object], key: str
) -> SelectionProvenance:
    selections = release.get("selections")
    selected = selections.get(key) if isinstance(selections, Mapping) else None
    selection: Mapping[str, object] = (
        cast(Mapping[str, object], selected) if isinstance(selected, Mapping) else {}
    )
    variable = key.split(".", maxsplit=2)[1]
    context = selection_context(release, selection)
    dataset = (
        selection.get("dataset_fingerprint")
        or (context or {}).get("dataset_fingerprint")
        or release.get("dataset_fingerprint")
    )
    code = selection_code_version(release, selection)
    semantics = selection_truth_semantics(release, selection, variable)
    signature = evaluation_compatibility(context, variable)
    valid = context is None or (
        signature is not None
        and signature[2] == semantics
        and context.get("code_version") == code
        and context.get("config_fingerprint") == release.get("config_fingerprint")
        and context.get("dataset_fingerprint", dataset) == dataset
    )
    return SelectionProvenance(
        _string(selection.get("method_id")),
        evaluation_id(selection.get("evaluation_id")),
        _string(dataset),
        _string(release.get("config_fingerprint")),
        code,
        semantics,
        signature,
        valid,
    )


@dataclass(frozen=True, slots=True)
class PromotionProvenance:
    recorded_at: datetime
    dataset_fingerprint: str
    config_fingerprint: str
    code_version: str
