"""Small, prediction-local facts explaining unavailable promoted fits."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import polars as pl


class FitUnavailableCause(StrEnum):
    EMPTY_MATRIX = "empty_matrix"
    NO_FORECAST_COLUMNS = "no_forecast_columns"
    CAUSAL_EXCLUSION = "causal_exclusion"
    NO_FINITE_TRUTH = "no_finite_truth"
    UNKNOWN_HISTORY = "unknown_history"


_CAUSE_TEXT = {
    FitUnavailableCause.EMPTY_MATRIX: (
        "training matrix is missing or empty; build the dataset once truth is available"
    ),
    FitUnavailableCause.NO_FORECAST_COLUMNS: (
        "training matrix has no provider forecast columns; rebuild after forecasts are archived"
    ),
    FitUnavailableCause.CAUSAL_EXCLUSION: (
        "finite training truth is excluded by the issue-time cutoff; keep polling"
    ),
    FitUnavailableCause.NO_FINITE_TRUTH: (
        "no finite training truth; wait for valid truth and rebuild once it is incorporated"
    ),
    FitUnavailableCause.UNKNOWN_HISTORY: (
        "no scoreable training history available at issuance; keep polling and "
        "rebuild only when new truth has not yet been incorporated"
    ),
}


def _finite_truth_counts(frame: pl.DataFrame) -> dict[str, int]:
    columns = [
        name
        for name, dtype in frame.schema.items()
        if name.startswith("t__") and dtype.is_numeric()
    ]
    if not columns:
        return {}
    counts = frame.select(pl.col(name).is_finite().sum() for name in columns)
    return {name: int(value or 0) for name, value in counts.row(0, named=True).items()}


@dataclass(frozen=True, slots=True)
class TrainingFacts:
    raw_rows: int
    has_forecasts: bool
    raw_truth: Mapping[str, int]
    eligible_truth: Mapping[str, int]

    def unavailable_cause(self, truth_column: str) -> FitUnavailableCause:
        if self.raw_rows == 0:
            return FitUnavailableCause.EMPTY_MATRIX
        if not self.has_forecasts:
            return FitUnavailableCause.NO_FORECAST_COLUMNS
        if self.raw_truth.get(truth_column, 0) > 0 and not self.eligible_truth.get(
            truth_column, 0
        ):
            return FitUnavailableCause.CAUSAL_EXCLUSION
        return FitUnavailableCause.NO_FINITE_TRUTH


@dataclass(frozen=True, slots=True, order=True)
class FitFailure:
    product: str
    variable: str
    truth_semantics: str
    cause: FitUnavailableCause


@dataclass(slots=True)
class FitDiagnostics:
    """Retain counts and scoped failures, never the full training frames."""

    training: dict[str, TrainingFacts] = field(default_factory=dict)
    unavailable: dict[tuple[str, str, str], FitFailure] = field(default_factory=dict)

    @property
    def promoted_fit_unavailable(self) -> bool:
        return bool(self.unavailable)

    @property
    def failures(self) -> tuple[FitFailure, ...]:
        return tuple(sorted(self.unavailable.values()))

    def capture_training(
        self,
        product: str,
        raw: pl.DataFrame,
        eligible: pl.DataFrame,
        *,
        has_forecasts: bool,
    ) -> None:
        self.training[product] = TrainingFacts(
            raw_rows=raw.height,
            has_forecasts=has_forecasts,
            raw_truth=_finite_truth_counts(raw),
            eligible_truth=_finite_truth_counts(eligible),
        )

    def record_unavailable(
        self,
        release_ids: Iterable[str | None],
        *,
        product: str,
        variable: str,
        truth_semantics: str,
        truth_column: str,
    ) -> None:
        if not any(release_id is not None for release_id in release_ids):
            return
        facts = self.training.get(product)
        cause = (
            facts.unavailable_cause(truth_column)
            if facts is not None
            else FitUnavailableCause.UNKNOWN_HISTORY
        )
        key = product, variable, truth_semantics
        self.unavailable[key] = FitFailure(*key, cause=cause)


def describe_fit_failures(failures: Sequence[FitFailure]) -> str:
    details = "; ".join(
        f"{failure.product}.{failure.variable} "
        f"({failure.truth_semantics}: {_CAUSE_TEXT[failure.cause]})"
        for failure in sorted(set(failures))
    )
    return (
        "some compatible promoted selections could not be fitted at issuance: "
        + details
    )
