"""Product-level outcomes, independent of the number of emitted score rows."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import polars as pl

from grounded_weather_forecast.status import (
    EXIT_NO_FOLDS,
    CommandOutcome,
    ComponentDiagnostic,
)


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


@dataclass(slots=True)
class BacktestResult:
    """Keep evaluation failures even if a later promotion fails independently."""

    products: list[ProductBacktestOutcome] = field(default_factory=list)
    promotion: CommandOutcome = field(default_factory=CommandOutcome)

    @property
    def row_count(self) -> int:
        return sum(product.row_count for product in self.products)

    def product_diagnostics(self) -> tuple[ComponentDiagnostic, ...]:
        return tuple(
            ComponentDiagnostic(
                product.product,
                product.exit_code,
                "backtest_product_failed",
                product.error_detail or "evaluation produced no usable scores",
            )
            for product in self.products
            if product.status == "failed"
        )

    def command(self) -> CommandOutcome:
        diagnostics = self.product_diagnostics() + self.promotion.diagnostics
        failed = next((p for p in self.products if p.status == "failed"), None)
        if failed is not None:
            return CommandOutcome(
                failed.exit_code,
                "backtest_product_failed",
                failed.error_detail,
                diagnostics,
            )
        if self.promotion.exit_code:
            return CommandOutcome(
                self.promotion.exit_code,
                self.promotion.failure_kind,
                self.promotion.error_detail,
                diagnostics,
            )
        if any(p.status == "pending" for p in self.products):
            return CommandOutcome(EXIT_NO_FOLDS, "no_folds")
        return CommandOutcome(0 if self.row_count else 2)
