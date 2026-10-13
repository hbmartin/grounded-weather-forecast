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
        return {name: TruthAvailability(0, 0, 0) for name in columns}
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


@dataclass(frozen=True, slots=True)
class UnavailableFit:
    """The unavailable branch of a promoted attempt, independent of later fits."""

    product: str
    variable: str
    truth_semantics: str
    truth_column: str
    branch: FitUnavailableBranch | None


@dataclass(slots=True)
class FitDiagnostics:
    """Capture a product snapshot, fit its targets, then resolve its failures.

    Starting an attempt resets only its branch. Promoted failures remain until
    capture_training begins a new snapshot for that product. Resolve training
    after all fits, while the raw matrix is still local to the caller, before
    reading failures. Incremental resolution caches facts within the same raw
    matrix and issuance cutoff; capture a new snapshot before either changes.
    """

    training: dict[str, TrainingFacts] = field(default_factory=dict)
    unavailable: dict[UnavailableFit, FitFailure] = field(default_factory=dict)
    branches: dict[FitKey, FitUnavailableBranch] = field(default_factory=dict)
    pending: set[UnavailableFit] = field(default_factory=set)

    @property
    def promoted_fit_unavailable(self) -> bool:
        return bool(self.unavailable or self.pending)

    @property
    def failures(self) -> tuple[FitFailure, ...]:
        if self.pending:
            products = ", ".join(sorted({attempt.product for attempt in self.pending}))
            raise RuntimeError(
                f"fit diagnostics pending for {products}; call resolve_training "
                "before reading failures"
            )
        return tuple(sorted(set(self.unavailable.values())))

    def capture_training(self, product: str, raw: pl.DataFrame) -> None:
        self.training[product] = TrainingFacts(raw_rows=raw.height)
        self.branches = {
            key: branch for key, branch in self.branches.items() if key[0] != product
        }
        self.unavailable = {
            attempt: failure
            for attempt, failure in self.unavailable.items()
            if attempt.product != product
        }
        self.pending = {
            attempt for attempt in self.pending if attempt.product != product
        }

    def start_fit(self, product: str, variable: str, semantics: str) -> None:
        key = product, variable, semantics
        self.branches.pop(key, None)

    def resolve_training(
        self, product: str, raw: pl.DataFrame, issue_time: datetime
    ) -> None:
        targets = tuple(
            attempt for attempt in self.pending if attempt.product == product
        )
        if not targets:
            return
        facts = self.training.get(product)
        truth = dict(facts.truth) if facts is not None else {}
        columns = sorted({attempt.truth_column for attempt in targets} - truth.keys())
        if columns:
            truth.update(_truth_availability(raw, columns, issue_time))
        self.training[product] = TrainingFacts(raw_rows=raw.height, truth=truth)
        for attempt in targets:
            self.unavailable[attempt] = self._failure(attempt)
            self.pending.remove(attempt)

    def _cause(self, attempt: UnavailableFit) -> FitUnavailableCause:
        branch = attempt.branch
        facts = self.training.get(attempt.product)
        if branch is None:
            return FitUnavailableCause.UNKNOWN_HISTORY
        if facts is not None and facts.raw_rows == 0:
            return FitUnavailableCause.EMPTY_MATRIX
        if branch == FitUnavailableBranch.NO_FORECAST_COLUMNS:
            return FitUnavailableCause.NO_FORECAST_COLUMNS
        return (
            facts.unavailable_cause(attempt.truth_column)
            if facts is not None
            else FitUnavailableCause.UNKNOWN_HISTORY
        )

    def _failure(self, attempt: UnavailableFit) -> FitFailure:
        return FitFailure(
            attempt.product,
            attempt.variable,
            attempt.truth_semantics,
            self._cause(attempt),
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
        attempt = UnavailableFit(*key, truth_column, self.branches.get(key))
        if attempt in self.unavailable or attempt in self.pending:
            return
        if attempt.branch == FitUnavailableBranch.EMPTY_SLICE:
            self.pending.add(attempt)
        else:
            self.unavailable[attempt] = self._failure(attempt)


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
