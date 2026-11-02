"""Decision semantics shared by reports and serving consumers."""

from collections.abc import Mapping

ELIGIBILITY = "eligibility"
POOLED_SCOPE = "pooled_D3-10"
SLICE_SCOPE = "slice"
EXCLUSION_REASON = "board minimum excluded by eligibility requirements"


def decision_gate(row: Mapping[str, object]) -> str | None:
    value = row.get("decision_gate")
    if value is None:
        value = row.get("gate")
    return (
        value
        if (isinstance(value, str) and bool(value)) or value is None
        else "invalid_selection_decision"
    )


def argmin_selected(row: Mapping[str, object]) -> bool:
    return (
        decision_gate(row) in (None, ELIGIBILITY)
        and row.get("evidence_scope", SLICE_SCOPE) == SLICE_SCOPE
    )
