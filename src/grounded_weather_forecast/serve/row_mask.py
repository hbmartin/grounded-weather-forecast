"""Shared eligibility normalization for serve-time uncertainty diagnostics."""

from collections.abc import Sequence


def normalize_eligible_rows(
    eligible_rows: Sequence[bool] | None, count: int
) -> tuple[bool, ...]:
    eligible = tuple(eligible_rows) if eligible_rows is not None else (True,) * count
    if len(eligible) != count:
        msg = "eligible_rows must match the number of chosen rows"
        raise ValueError(msg)
    return eligible
