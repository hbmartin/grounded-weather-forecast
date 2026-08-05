"""Blender registry: method_id -> factory. Factories, never instances —
the backtest engine constructs a fresh blender per fold (a leakage defense)."""

from collections.abc import Callable

from grounded_weather_forecast.contracts import (
    Blender,
    Product,
    TargetKind,
    VariableSpec,
)

type BlenderFactory = Callable[[], Blender]

_REGISTRY: dict[str, BlenderFactory] = {}
# Methods restricted to specific variables (a PoP calibrator, a precip
# distribution head) are simply never fitted off-scope; the engine and
# serving already consult supports_product per (product, variable) pair.
_VARIABLE_SCOPE: dict[str, frozenset[str]] = {}


class UnknownMethodError(KeyError):
    """No blender is registered under the requested method_id."""


def register(
    method_id: str,
    factory: BlenderFactory,
    *,
    variables: frozenset[str] | None = None,
) -> None:
    if method_id in _REGISTRY:
        msg = f"method_id already registered: {method_id!r}"
        raise ValueError(msg)
    _REGISTRY[method_id] = factory
    if variables is not None:
        _VARIABLE_SCOPE[method_id] = variables


def get_factory(method_id: str) -> BlenderFactory:
    try:
        return _REGISTRY[method_id]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        msg = f"unknown method {method_id!r}; available: {available}"
        raise UnknownMethodError(msg) from exc


def available_methods() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def supports_product(
    method_id: str, product: Product, variable: VariableSpec | None = None
) -> bool:
    """Whether a method's feature assumptions match the product contract."""
    hourly_only = method_id == "persistence" or method_id.startswith("anchored_")
    if product is Product.DAILY and hourly_only:
        return False
    if variable is None:
        return True
    scope = _VARIABLE_SCOPE.get(method_id)
    if scope is not None and variable.name not in scope:
        return False
    if variable.name in {"precip_mm", "pop"} and hourly_only:
        return False
    # Gaussian/isotonic distribution heads make no sense for probabilities or
    # zero-inflated precipitation; a dedicated occurrence/amount split owns
    # those (improvement plan section 4 P2).
    distributional = method_id in {"emos", "idr"}
    skewed = variable.name in {"precip_mm", "precip_sum_mm"}
    return not (distributional and (variable.kind is TargetKind.PROBABILITY or skewed))
