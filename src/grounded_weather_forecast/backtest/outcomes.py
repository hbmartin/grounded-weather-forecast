"""Product-level outcomes, independent of the number of emitted score rows."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import polars as pl

from grounded_weather_forecast.config import BacktestConfig


@dataclass(slots=True)
class BacktestDiagnostics:
    """Filled by the score producer using its actual fold-planning clock."""

    history_too_short: bool = False

    def observe_clock(self, issues: pl.Series, config: BacktestConfig) -> None:
        if issues.is_empty() or issues.null_count():
            return
        first, last = issues.min(), issues.max()
        if not isinstance(first, datetime) or not isinstance(last, datetime):
            return
        self.history_too_short = last <= first + timedelta(
            days=config.initial_train_days
        )


@dataclass(frozen=True, slots=True)
class ProductBacktestOutcome:
    product: str
    status: Literal["evaluated", "insufficient_history", "not_applicable", "failed"]
    row_count: int = 0

    @classmethod
    def from_scores(
        cls, product: str, row_count: int, diagnostics: BacktestDiagnostics
    ) -> "ProductBacktestOutcome":
        if row_count:
            return cls(product, "evaluated", row_count)
        if diagnostics.history_too_short:
            return cls(product, "insufficient_history")
        return cls(product, "failed")
