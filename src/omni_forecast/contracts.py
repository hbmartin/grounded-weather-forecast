"""Frozen contracts shared by every layer: variables, matrices, and the Blender protocol.

This module and ``leads`` are the only modules other packages may deep-import from.
Blenders import contracts only, never the dataset layer.
"""

from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Protocol, Self

import numpy as np
import numpy.typing as npt
import polars as pl

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]

COLUMN_SEPARATOR = "__"


class TargetKind(StrEnum):
    CONTINUOUS = "continuous"
    PROBABILITY = "probability"


class SourceKind(StrEnum):
    LIVE = "live"
    SYNTHETIC = "synthetic"


class Product(StrEnum):
    MINUTELY = "minutely"
    HOURLY = "hourly"
    DAILY = "daily"


class LeadUnit(StrEnum):
    MINUTES = "minutes"
    HOURS = "hours"
    LOCAL_DAYS = "local_days"


class TruthSemantics(StrEnum):
    INSTANTANEOUS = "inst"
    INTERVAL_MEAN = "mean"


@dataclass(frozen=True, slots=True)
class VariableSpec:
    """One canonical forecast variable in normalized metric units."""

    name: str
    kind: TargetKind
    unit: str
    has_dual_semantics: bool = False
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True, slots=True)
class ProductSpec:
    """Temporal contract for one emitted product."""

    product: Product
    lead_unit: LeadUnit
    horizon: int


PRODUCT_SPECS: tuple[ProductSpec, ...] = (
    ProductSpec(Product.MINUTELY, LeadUnit.MINUTES, 60),
    ProductSpec(Product.HOURLY, LeadUnit.HOURS, 48),
    ProductSpec(Product.DAILY, LeadUnit.LOCAL_DAYS, 10),
)


HOURLY_VARIABLES: tuple[VariableSpec, ...] = (
    VariableSpec("temp_c", TargetKind.CONTINUOUS, "°C", has_dual_semantics=True),
    VariableSpec(
        "humidity_pct",
        TargetKind.CONTINUOUS,
        "%",
        has_dual_semantics=True,
        minimum=0.0,
        maximum=100.0,
    ),
    VariableSpec("dew_point_c", TargetKind.CONTINUOUS, "°C", has_dual_semantics=True),
    VariableSpec(
        "wind_speed_ms",
        TargetKind.CONTINUOUS,
        "m/s",
        has_dual_semantics=True,
        minimum=0.0,
    ),
    VariableSpec("wind_gust_ms", TargetKind.CONTINUOUS, "m/s", minimum=0.0),
    VariableSpec(
        "pressure_sea_hpa", TargetKind.CONTINUOUS, "hPa", has_dual_semantics=True
    ),
    VariableSpec("precip_mm", TargetKind.CONTINUOUS, "mm", minimum=0.0),
    VariableSpec("pop", TargetKind.PROBABILITY, "", minimum=0.0, maximum=1.0),
)

DAILY_VARIABLES: tuple[VariableSpec, ...] = (
    VariableSpec("temp_max_c", TargetKind.CONTINUOUS, "°C"),
    VariableSpec("temp_min_c", TargetKind.CONTINUOUS, "°C"),
    VariableSpec("pop", TargetKind.PROBABILITY, "", minimum=0.0, maximum=1.0),
    VariableSpec("precip_sum_mm", TargetKind.CONTINUOUS, "mm", minimum=0.0),
)


def hourly_variable(name: str) -> VariableSpec:
    for spec in HOURLY_VARIABLES:
        if spec.name == name:
            return spec
    msg = f"unknown hourly variable: {name!r}"
    raise KeyError(msg)


def daily_variable(name: str) -> VariableSpec:
    for spec in DAILY_VARIABLES:
        if spec.name == name:
            return spec
    msg = f"unknown daily variable: {name!r}"
    raise KeyError(msg)


def _encode_column_segment(segment: str) -> str:
    if not segment:
        raise ValueError("column-name segments must not be empty")
    return segment.replace("%", "%25").replace(COLUMN_SEPARATOR, "%5F%5F")


def _decode_column_segment(segment: str) -> str:
    return segment.replace("%5F%5F", COLUMN_SEPARATOR).replace("%25", "%")


def fx_col(source: str, variable: str) -> str:
    """Forecast column for one source and canonical variable."""
    return f"fx{COLUMN_SEPARATOR}{_encode_column_segment(source)}{COLUMN_SEPARATOR}{_encode_column_segment(variable)}"


def fxd_col(source: str, variable: str) -> str:
    """Daily-forecast column for one source and canonical daily variable."""
    return f"fxd{COLUMN_SEPARATOR}{_encode_column_segment(source)}{COLUMN_SEPARATOR}{_encode_column_segment(variable)}"


def age_col(source: str) -> str:
    """Hours-since-fetch column for one source."""
    return f"age{COLUMN_SEPARATOR}{source}"


def obs_col(variable: str) -> str:
    """Station observation at issue time (leakage-safe past data)."""
    return f"obs{COLUMN_SEPARATOR}{variable}"


def truth_col(variable: str, semantics: TruthSemantics | None = None) -> str:
    """Truth column, optionally with dual-semantics suffix."""
    base = f"t{COLUMN_SEPARATOR}{variable}"
    if semantics is None:
        return base
    return f"{base}{COLUMN_SEPARATOR}{semantics.value}"


def parse_fx_col(column: str) -> tuple[str, str]:
    """Invert :func:`fx_col`/:func:`fxd_col` into (source, variable)."""
    match column.split(COLUMN_SEPARATOR):
        case ["fx" | "fxd", source, variable] if source and variable:
            return _decode_column_segment(source), _decode_column_segment(variable)
        case _:
            msg = f"not a forecast column: {column!r}"
            raise ValueError(msg)


def is_truth_col(column: str) -> bool:
    return column.startswith(f"t{COLUMN_SEPARATOR}")


class ContractViolationError(ValueError):
    """A dataclass invariant in this module was violated."""


class MixedProvenanceError(ValueError):
    """Live and synthetic rows were combined without explicit consent."""


@dataclass(frozen=True)
class ForecastMatrix:
    """Aligned per-row source forecasts handed to a blender.

    ``values`` is ``(n, k)`` float64 with ``NaN`` where a source is unavailable;
    ``availability`` is its explicit ``~isnan`` so blenders never re-derive it
    inconsistently. ``features`` carries aligned context columns (calendar,
    lead, source ages, issue-time observations) and never truth columns.
    """

    sources: tuple[str, ...]
    values: FloatArray
    availability: BoolArray
    lead_hours: FloatArray
    features: pl.DataFrame
    product: Product = Product.HOURLY

    def __post_init__(self) -> None:
        n, k = self.values.shape
        if k != len(self.sources):
            msg = f"values has {k} columns for {len(self.sources)} sources"
            raise ContractViolationError(msg)
        if self.availability.shape != (n, k):
            msg = (
                f"availability shape {self.availability.shape} != values shape {(n, k)}"
            )
            raise ContractViolationError(msg)
        if self.lead_hours.shape != (n,):
            msg = f"lead_hours shape {self.lead_hours.shape} != ({n},)"
            raise ContractViolationError(msg)
        if self.features.height != n:
            msg = f"features has {self.features.height} rows, expected {n}"
            raise ContractViolationError(msg)
        if leaked := [c for c in self.features.columns if is_truth_col(c)]:
            msg = f"truth columns leaked into features: {leaked}"
            raise ContractViolationError(msg)

    @property
    def n_rows(self) -> int:
        return self.values.shape[0]

    @classmethod
    def build(
        cls,
        sources: tuple[str, ...],
        values: FloatArray,
        lead_hours: FloatArray,
        features: pl.DataFrame,
        product: Product = Product.HOURLY,
    ) -> Self:
        """Construct with availability derived from the NaN pattern."""
        return cls(
            sources=sources,
            values=values,
            availability=~np.isnan(values),
            lead_hours=lead_hours,
            features=features,
            product=product,
        )


@dataclass(frozen=True)
class SupervisedSlice:
    """Training data for one variable: a matrix plus non-null truth."""

    x: ForecastMatrix
    y: FloatArray
    variable: VariableSpec
    source_kind: SourceKind

    def __post_init__(self) -> None:
        if self.y.shape != (self.x.n_rows,):
            msg = f"y shape {self.y.shape} != ({self.x.n_rows},)"
            raise ContractViolationError(msg)
        if np.isnan(self.y).any():
            msg = "y contains NaN; null-truth rows must be excluded upstream"
            raise ContractViolationError(msg)


@dataclass(frozen=True)
class BlendResult:
    """A blender's output: points, optionally with predictive quantiles."""

    point: FloatArray
    quantiles: FloatArray | None = None
    quantile_levels: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        match (self.quantiles, self.quantile_levels):
            case (None, ()):
                pass
            case (np.ndarray() as q, levels) if len(levels) > 0:
                expected = (self.point.shape[0], len(levels))
                if q.shape != expected:
                    msg = f"quantiles shape {q.shape} != {expected}"
                    raise ContractViolationError(msg)
                if any(not 0.0 < level < 1.0 for level in levels) or any(
                    left >= right for left, right in pairwise(levels)
                ):
                    msg = "quantile_levels must be strictly increasing inside (0, 1)"
                    raise ContractViolationError(msg)
            case _:
                msg = "quantiles and quantile_levels must be provided together"
                raise ContractViolationError(msg)


class Blender(Protocol):
    """One forecasting method; baselines included. Fresh instance per fit."""

    method_id: str

    def fit(self, train: SupervisedSlice) -> Self: ...

    def predict(self, x: ForecastMatrix) -> BlendResult: ...
