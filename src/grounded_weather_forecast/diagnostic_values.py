"""Non-raising primitive checks for untrusted optional journal metadata."""

from functools import cache
from inspect import get_annotations
from types import UnionType
from typing import Union, cast, get_args, get_origin, get_type_hints

_PRIMITIVES = (str, int, bool, type(None))


def actual_string(value: object) -> str | None:
    """Normalize real subclasses, never objects spoofing ``__class__``."""
    return str.__str__(cast(str, value)) if issubclass(type(value), str) else None


def actual_integer(value: object) -> int | None:
    if type(value) is bool or not issubclass(type(value), int):
        return None
    return int.__int__(cast(int, value))


def primitive(value: object) -> object:
    if (text := actual_string(value)) is not None:
        return text
    if (number := actual_integer(value)) is not None:
        return number
    return value


def _allowed_types(annotation: object) -> tuple[type, ...]:
    if any(annotation is kind for kind in _PRIMITIVES):
        return (cast(type, annotation),)
    if get_origin(annotation) not in (UnionType, Union):
        return ()
    parts = get_args(annotation)
    return parts if all(any(p is kind for kind in _PRIMITIVES) for p in parts) else ()


@cache
def declared_types(cls: type) -> dict[str, tuple[type, ...]]:
    """Unsupported extensions do not erase the inherited diagnostic schema."""
    result: dict[str, tuple[type, ...]] = {}
    for base in reversed(cls.__mro__):
        # Python 3.14 can defer annotations; resolve through the public APIs.
        try:
            resolved = get_type_hints(base)
        except (NameError, TypeError, ValueError):
            resolved = get_annotations(base, eval_str=False)
        result.update(
            (name, _allowed_types(annotation)) for name, annotation in resolved.items()
        )
    return result


def field_valid(name: str, value: object, schema: dict[str, tuple[type, ...]]) -> bool:
    allowed = schema.get(name, ())
    # Only supported declared primitives can reach a runtime class check.
    if not allowed or not all(any(t is p for p in _PRIMITIVES) for t in allowed):
        return False
    if type(value) is bool:
        valid = bool in allowed
    elif value is None:
        valid = type(None) in allowed
    elif actual_string(value) is not None:
        valid = str in allowed
    elif actual_integer(value) is not None:
        valid = int in allowed
    else:
        valid = False
    return valid and (name != "stage" or bool((actual_string(value) or "").strip()))
