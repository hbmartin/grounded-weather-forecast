import pathlib
from collections.abc import Mapping
from pathlib import Path


def invalid_division(root: Path, raw_pointer: Mapping[str, str]) -> Path:
    fingerprint = raw_pointer["fingerprint"]
    # ruleid: artifact-pointer-fields-must-not-feed-paths-directly
    return root / fingerprint


def invalid_joinpath(root: Path, raw_pointer: Mapping[str, str]) -> Path:
    # ruleid: artifact-pointer-fields-must-not-feed-paths-directly
    return root.joinpath(raw_pointer.get("method_id"), "state.json")


def invalid_constructor(raw_pointer: Mapping[str, str]) -> Path:
    # ruleid: artifact-pointer-fields-must-not-feed-paths-directly
    return Path(raw_pointer["fingerprint"])


def invalid_multiarg_constructor(root: Path, raw_pointer: Mapping[str, str]) -> Path:
    # ruleid: artifact-pointer-fields-must-not-feed-paths-directly
    return Path(root, raw_pointer["fingerprint"], "state.json")


def invalid_qualified_constructor(raw_pointer: Mapping[str, str]) -> pathlib.Path:
    # ruleid: artifact-pointer-fields-must-not-feed-paths-directly
    return pathlib.Path(raw_pointer["fingerprint"])


def valid_division(root: Path, validated_pointer: Mapping[str, str]) -> Path:
    # ok: artifact-pointer-fields-must-not-feed-paths-directly
    return root / validated_pointer["fingerprint"]


def valid_joinpath(root: Path, fingerprint: str) -> Path:
    # ok: artifact-pointer-fields-must-not-feed-paths-directly
    return root.joinpath(fingerprint, "state.json")


def valid_constructor(validated_pointer: Mapping[str, str]) -> Path:
    # ok: artifact-pointer-fields-must-not-feed-paths-directly
    return Path(validated_pointer["fingerprint"])


def valid_qualified_constructor(validated_pointer: Mapping[str, str]) -> pathlib.Path:
    # ok: artifact-pointer-fields-must-not-feed-paths-directly
    return pathlib.Path(validated_pointer["fingerprint"])
