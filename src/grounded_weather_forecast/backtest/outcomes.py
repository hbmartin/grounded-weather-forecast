"""Product-level outcomes, independent of the number of emitted score rows."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import polars as pl


@dataclass(slots=True)
class BacktestDiagnostics:
    """Filled by the score producer using its actual fold-planning clock."""

    history_too_short: bool = False
    snapshot_at: datetime | None = None
    awaiting_tests: int = 0
    overdue_tests: int = 0

    def observe_tests(self, known_at: pl.Series, *, usable_training: bool) -> None:
        """Only future test truth, with usable training, can be expected pending."""
        if not usable_training:
            return
        if self.snapshot_at is not None:
            future = known_at > self.snapshot_at
            self.awaiting_tests += int(future.sum())
            self.overdue_tests += len(known_at) - int(future.sum())

    @property
    def pending_reason(
        self,
    ) -> Literal["insufficient_history", "awaiting_truth"] | None:
        if self.history_too_short:
            return "insufficient_history"
        if self.awaiting_tests and not self.overdue_tests:
            return "awaiting_truth"
        return None


@dataclass(frozen=True, slots=True)
class ProductBacktestOutcome:
    product: str
    status: Literal["evaluated", "pending", "not_applicable", "failed"]
    row_count: int = 0
    pending_reason: Literal["insufficient_history", "awaiting_truth"] | None = None
    error_detail: str | None = None
    exit_code: int = 1

    @classmethod
    def from_scores(
        cls, product: str, row_count: int, diagnostics: BacktestDiagnostics
    ) -> "ProductBacktestOutcome":
        if row_count:
            return cls(product, "evaluated", row_count)
        if reason := diagnostics.pending_reason:
            return cls(product, "pending", pending_reason=reason)
        return cls(product, "failed")
