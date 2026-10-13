"""Small, prediction-local facts explaining unavailable promoted fits."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

import polars as pl


class FitUnavailableCause(StrEnum):
    EMPTY_MATRIX = "empty_matrix"
    NO_FORECAST_COLUMNS = "no_forecast_columns"
    CAUSAL_EXCLUSION = "causal_exclusion"
    LATER_ISSUANCE = "later_issuance"
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
        "training truth on already-issued rows is not yet available at the requested issuance; "
        "keep polling as live issuance advances; repeating a fixed historical --now "
        "cannot admit future evidence"
    ),
    FitUnavailableCause.LATER_ISSUANCE: (
        "finite training truth exists only on later-issued archive rows, unavailable at "
        "the requested issuance; advance issuance for a live forecast; repeating or "
        "rebuilding a fixed historical --now cannot admit future evidence"
    ),
    FitUnavailableCause.NO_FINITE_TRUTH: (
        "no finite training truth; wait for valid truth and rebuild once it is incorporated"
    ),
    FitUnavailableCause.UNKNOWN_HISTORY: (
        "training availability could not be established at issuance; inspect the "
        "training history and timestamps before rebuilding"
    ),
}


class FitUnavailableBranch(StrEnum):
    NO_FORECAST_COLUMNS = "no_forecast_columns"
    EMPTY_SLICE = "empty_slice"


type FitKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class TruthAvailability:
    total: int
    issued: int
    future: int


def _truth_availability(
    raw: pl.DataFrame, columns: Sequence[str], issue_time: datetime
) -> dict[str, TruthAvailability]:
    if raw.is_empty() or not columns:
        return {}
    finite = raw.select(
        pl.col("issue_time"),
        *(pl.col(name).is_finite().alias(name) for name in columns),
    )
    counts = finite.select(
        pl.struct(
            pl.col(name).sum().alias("total"),
            (pl.col(name) & (pl.col("issue_time") <= issue_time)).sum().alias("issued"),
            (pl.col(name) & (pl.col("issue_time") > issue_time)).sum().alias("future"),
        ).alias(name)
        for name in columns
    ).row(0, named=True)
    return {name: TruthAvailability(**values) for name, values in counts.items()}


@dataclass(frozen=True, slots=True)
class TrainingFacts:
    raw_rows: int
    truth: Mapping[str, TruthAvailability] = field(default_factory=dict)

    def unavailable_cause(self, truth_column: str) -> FitUnavailableCause:
        facts = self.truth.get(truth_column)
        if facts is None:
            return FitUnavailableCause.UNKNOWN_HISTORY
        if facts.total == 0:
            return FitUnavailableCause.NO_FINITE_TRUTH
        if facts.issued > 0:
            return FitUnavailableCause.CAUSAL_EXCLUSION
        if facts.future > 0:
            return FitUnavailableCause.LATER_ISSUANCE
        return FitUnavailableCause.UNKNOWN_HISTORY


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
    unavailable: dict[FitKey, FitFailure] = field(default_factory=dict)
    branches: dict[FitKey, FitUnavailableBranch] = field(default_factory=dict)
    pending: dict[FitKey, str] = field(default_factory=dict)

    @property
    def promoted_fit_unavailable(self) -> bool:
        return bool(self.unavailable)

    @property
    def failures(self) -> tuple[FitFailure, ...]:
        return tuple(sorted(self.unavailable.values()))

    def capture_training(self, product: str, raw: pl.DataFrame) -> None:
        self.training[product] = TrainingFacts(raw_rows=raw.height)
        for records in (self.branches, self.unavailable, self.pending):
            for key in tuple(records):
                if key[0] == product:
                    del records[key]

    def start_fit(self, product: str, variable: str, semantics: str) -> None:
        key = product, variable, semantics
        self.branches.pop(key, None)
        self.unavailable.pop(key, None)
        self.pending.pop(key, None)

    def resolve_training(
        self, product: str, raw: pl.DataFrame, issue_time: datetime
    ) -> None:
        targets = {
            key: column for key, column in self.pending.items() if key[0] == product
        }
        if not targets:
            return
        columns = sorted(set(targets.values()))
        self.training[product] = TrainingFacts(
            raw_rows=raw.height, truth=_truth_availability(raw, columns, issue_time)
        )
        for key, column in targets.items():
            self.unavailable[key] = FitFailure(*key, cause=self._cause(key, column))
            del self.pending[key]

    def _cause(self, key: FitKey, truth_column: str) -> FitUnavailableCause:
        branch = self.branches.get(key)
        facts = self.training.get(key[0])
        if branch is None:
            return FitUnavailableCause.UNKNOWN_HISTORY
        if facts is not None and facts.raw_rows == 0:
            return FitUnavailableCause.EMPTY_MATRIX
        if branch == FitUnavailableBranch.NO_FORECAST_COLUMNS:
            return FitUnavailableCause.NO_FORECAST_COLUMNS
        return (
            facts.unavailable_cause(truth_column)
            if facts is not None
            else FitUnavailableCause.UNKNOWN_HISTORY
        )

    def note_branch(
        self, product: str, variable: str, semantics: str, branch: FitUnavailableBranch
    ) -> None:
        self.branches[product, variable, semantics] = branch

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
        key = product, variable, truth_semantics
        self.unavailable[key] = FitFailure(*key, cause=self._cause(key, truth_column))
        if self.branches.get(key) == FitUnavailableBranch.EMPTY_SLICE:
            self.pending[key] = truth_column


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
