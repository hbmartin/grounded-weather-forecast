"""Non-raising primitive checks for untrusted optional journal metadata."""

import sys
from importlib import import_module
from inspect import get_annotations
from types import SimpleNamespace, UnionType
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


_SCHEMAS: dict[type, dict[str, tuple[type, ...]]] = {}


def _annotations(base: type) -> dict[str, object]:
    if sys.version_info >= (3, 14):
        # This is a 3.14 standard-library module, not a 3.13 PyPI dependency.
        annotations = import_module("annotationlib")
        return annotations.get_annotations(base, format=annotations.Format.STRING)
    return get_annotations(base, eval_str=False)


def _resolve(base: type, name: str, annotation: object) -> object:
    namespace = vars(sys.modules[base.__module__])
    holder = SimpleNamespace(__annotations__={name: annotation})
    return get_type_hints(holder, globalns=namespace, localns=dict(vars(base)))[name]


def declared_types(cls: type) -> dict[str, tuple[type, ...]]:
    """Unsupported extensions do not erase the inherited diagnostic schema."""
    if cls in _SCHEMAS:
        return _SCHEMAS[cls]
    result: dict[str, tuple[type, ...]] = {}
    complete = True
    for base in reversed(cls.__mro__):
        try:
            annotations = _annotations(base)
        except (Exception,):  # noqa: B013 -- optional class metadata is untrusted
            complete = False
            continue
        for name, annotation in annotations.items():
            if result.get(name):
                continue  # Narrowed annotations cannot erase inherited wire fields.
            try:
                result[name] = _allowed_types(_resolve(base, name, annotation))
            except (Exception,):  # noqa: B013 -- contain resolution to one field
                result[name] = ()
                complete = False
    if complete:
        _SCHEMAS[cls] = result
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
