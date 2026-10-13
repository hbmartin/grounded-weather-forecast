"""Shared output safety for benchmark runners; failures retain available evidence."""

import json
import sys
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self
from uuid import uuid4


def output_path(repository: Path, supplied: Path | None, *, prefix: str) -> Path:
    if supplied is not None:
        return supplied
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return repository / "reports/benchmarks" / f"{prefix}-{stamp}-{uuid4().hex}.jsonl"


def artifact_paths(
    destination: Path, *, include_payloads: bool = False
) -> dict[str, Path]:
    if not destination.name:
        raise ValueError(f"benchmark output must name a file: {destination}")
    paths = {
        "samples": destination,
        "summary": destination.with_suffix(".summary.json"),
    }
    if include_payloads:
        paths["payloads"] = destination.with_suffix(".payloads.json")
    return paths


def preflight_artifacts(paths: Mapping[str, Path]) -> None:
    if len(set(paths.values())) != len(paths):
        raise ValueError("benchmark artifact destinations must be distinct")
    for path in paths.values():
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"benchmark output already exists: {path}")
    for parent in dict.fromkeys(path.parent for path in paths.values()):
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=parent) as probe:
            probe.write(b"benchmark output write probe\n")
            probe.flush()


def publish_artifacts(paths: Mapping[str, Path], contents: Mapping[str, str]) -> None:
    """Open all required paths exclusively, retaining every file after a failure."""
    with ExitStack() as stack:
        streams = {
            name: stack.enter_context(paths[name].open("x", encoding="utf-8"))
            for name in contents
        }
        for name, stream in streams.items():
            stream.write(contents[name])


def result_contents(
    records: list[dict[str, Any]],
    *,
    summary: list[dict[str, Any]] | None = None,
    payloads: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    contents = {
        "samples": "".join(
            json.dumps(record, sort_keys=True) + "\n" for record in records
        )
    }
    if summary is not None:
        contents["summary"] = json.dumps(summary, indent=2) + "\n"
    if payloads:
        contents["payloads"] = json.dumps(payloads, indent=2, sort_keys=True) + "\n"
    return contents


@dataclass(frozen=True, slots=True)
class BenchmarkOutputs:
    repository: Path
    prefix: str
    paths: Mapping[str, Path]

    @classmethod
    def prepare(
        cls,
        repository: Path,
        prefix: str,
        supplied: Path | None,
        *,
        include_payloads: bool = False,
    ) -> Self:
        destination = output_path(repository, supplied, prefix=prefix)
        paths = artifact_paths(destination, include_payloads=include_payloads)
        preflight_artifacts(paths)
        return cls(repository, prefix, paths)

    def rescue(self, contents: Mapping[str, str]) -> None:
        if not contents.get("samples"):
            return
        try:
            recovery = self.prepare(
                self.repository,
                f"{self.prefix}-recovery",
                None,
                include_payloads="payloads" in contents,
            )
            publish_artifacts(recovery.paths, contents)
        except (OSError, ValueError) as error:
            print(f"could not rescue benchmark outputs: {error}", file=sys.stderr)
            print("available benchmark samples (JSONL):", file=sys.stderr)
            print(contents["samples"], end="", file=sys.stderr)
        else:
            for name in contents:
                print(f"rescued {recovery.paths[name]}", file=sys.stderr)

    @contextmanager
    def measurements(
        self,
        records: list[dict[str, Any]],
        *,
        payloads: Mapping[str, Any] | None = None,
    ) -> Iterator[None]:
        try:
            yield
        except (BaseException,) as error:  # noqa: B013 -- project tuple convention
            print(f"benchmark measurements failed: {error}", file=sys.stderr)
            self.rescue(result_contents(records, payloads=payloads))
            raise

    def publish(self, contents: Mapping[str, str]) -> None:
        try:
            publish_artifacts(self.paths, contents)
        except (OSError,) as error:  # noqa: B013 -- project tuple convention
            print(f"could not publish benchmark outputs: {error}", file=sys.stderr)
            print(
                "destination artifacts may be incomplete; no files were removed: "
                + ", ".join(str(self.paths[name]) for name in contents),
                file=sys.stderr,
            )
            self.rescue(contents)
            raise SystemExit(1) from error
        for name in contents:
            print(f"wrote {self.paths[name]}", flush=True)
