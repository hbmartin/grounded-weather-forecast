"""Shared output safety for benchmark runners; failures retain available evidence."""

import json
import sys
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self, TextIO
from uuid import uuid4

THREADS = {
    "POLARS_MAX_THREADS": "2",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}


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


def _report(*values: object) -> None:
    """Failure reporting must never displace the failure being reported."""
    with suppress(BaseException):
        print(*values, file=sys.stderr, flush=True)


class JournalPreparationError(OSError):
    """The checkpoint destination could not be prepared before measuring."""


@dataclass(slots=True)
class BenchmarkResults:
    journal_path: Path
    journal: TextIO
    records: list[dict[str, Any]] = field(default_factory=list)
    payloads: dict[str, Any] = field(default_factory=dict)

    def add_sample(self, record: dict[str, Any]) -> None:
        """Checkpoint outside worker timings; flush does not promise power-loss safety."""
        self.records.append(record)
        self.journal.write(json.dumps(record, sort_keys=True) + "\n")
        self.journal.flush()

    def record_payloads(self, payloads: Mapping[str, Any]) -> None:
        self.payloads.update(payloads)

    def contents(
        self, *, summary: list[dict[str, Any]] | None = None
    ) -> dict[str, str]:
        return result_contents(self.records, summary=summary, payloads=self.payloads)

    def close(self) -> None:
        self.journal.close()


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
            _report("could not rescue benchmark outputs:", error)
            _report("available benchmark samples (JSONL):")
            _report(contents["samples"].rstrip("\n"))
        else:
            for name in contents:
                _report("rescued", recovery.paths[name])

    def open_journal(self) -> BenchmarkResults:
        path = output_path(self.repository, None, prefix=f"{self.prefix}-journal")
        journal = None
        try:
            preflight_artifacts({"journal": path})
            journal = path.open("x", encoding="utf-8")
            print(f"journal {path}", flush=True)
        except (BaseException,) as error:  # noqa: B013 -- preserve preparation failures
            if journal is not None:
                try:
                    journal.close()
                except (BaseException,) as secondary:  # noqa: B013
                    _report("could not close benchmark journal:", secondary)
            if isinstance(error, (OSError, ValueError)):
                raise JournalPreparationError(
                    f"could not prepare journal {path}: {error}"
                ) from error
            raise
        return BenchmarkResults(path, journal)

    def _rescue_measurements(self, results: BenchmarkResults) -> None:
        try:
            self.rescue(results.contents())
        except (BaseException,) as error:  # noqa: B013 -- rescue is best effort
            _report("could not rescue benchmark outputs:", error)
        _report("retained benchmark journal:", results.journal_path)

    @contextmanager
    def measurements(self) -> Iterator[BenchmarkResults]:
        results = self.open_journal()
        try:
            yield results
            results.close()
        except (BaseException,) as error:  # noqa: B013 -- project tuple convention
            _report("benchmark measurements failed:", error)
            self._rescue_measurements(results)
            try:
                results.close()
            except (BaseException,) as secondary:  # noqa: B013
                _report("could not close benchmark journal:", secondary)
            raise

    def publish(
        self, contents: Mapping[str, str], *, journal_path: Path | None = None
    ) -> None:
        try:
            publish_artifacts(self.paths, contents)
        except (OSError,) as error:  # noqa: B013 -- project tuple convention
            _report("could not publish benchmark outputs:", error)
            _report(
                "destination artifacts may be incomplete; no files were removed: "
                + ", ".join(str(self.paths[name]) for name in contents),
            )
            try:
                self.rescue(contents)
            except (BaseException,) as secondary:  # noqa: B013
                _report("could not rescue benchmark outputs:", secondary)
            if journal_path is not None:
                _report("retained benchmark journal:", journal_path)
            raise SystemExit(1) from error
        for name in contents:
            print(f"wrote {self.paths[name]}", flush=True)
