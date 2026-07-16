"""Blender registry: method_id -> factory. Factories, never instances —
the backtest engine constructs a fresh blender per fold (a leakage defense)."""

from collections.abc import Callable

from grounded_weather_forecast.contracts import Blender, Product, VariableSpec

type BlenderFactory = Callable[[], Blender]

_REGISTRY: dict[str, BlenderFactory] = {}


class UnknownMethodError(KeyError):
    """No blender is registered under the requested method_id."""


def register(method_id: str, factory: BlenderFactory) -> None:
    if method_id in _REGISTRY:
        msg = f"method_id already registered: {method_id!r}"
        raise ValueError(msg)
    _REGISTRY[method_id] = factory


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
    return not (
        variable is not None and variable.name in {"precip_mm", "pop"} and hourly_only
    )
