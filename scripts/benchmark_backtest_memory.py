"""Compare score generation and persistence in immutable, isolated workers.

The fixed workload mixes point and 19-level quantile scores with repeated
8.4 KB evaluation metadata. No live dataset, API, or promoted release is changed.
"""

from __future__ import annotations

# Workers select their source snapshot before importing first-party modules.
# All subprocess calls use constructed argv for local tools and source workers.
# ruff: noqa: PLC0415, S603, S607
import argparse
import hashlib
import io
import json
import os
import platform
import re
import resource
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_outputs import (
    THREADS,
    BenchmarkOutputs,
    BenchmarkResults,
    JournalPreparationError,
)
from benchmark_score_readers import tree_hash

FIXTURE_ROWS = 16_384
FEATURE_BYTES = 8449
ROW_COUNTS = (500_000, 2_000_000)

if TYPE_CHECKING:
    import polars as pl


def fixture_batches(
    module: ModuleType, count: int, *, compact: bool
) -> Iterator[pl.DataFrame]:
    import numpy as np
    import polars as pl

    from grounded_weather_forecast.evaluation import EvaluationRun

    moment = datetime(2026, 10, 14, tzinfo=UTC)
    evaluation = EvaluationRun(
        evaluation_id="fixed",
        created_at=moment.isoformat(),
        dataset_fingerprint="fixed-dataset",
        source_kind="live",
        source_set=("nws", "open_meteo"),
        feature_set=("x" * (FEATURE_BYTES - 4),),
        product="hourly",
        window="expanding",
        semantics={"temp_c": "inst"},
        methods=("point", "quantile"),
        code_version="fixed-code",
        config_fingerprint="fixed-config",
    )
    levels = tuple(i / 20 for i in range(1, 20))
    for offset in range(0, count, FIXTURE_ROWS):
        n = min(FIXTURE_ROWS, count - offset)
        position = np.arange(offset, offset + n)
        point = np.sin(position / 100)
        lead = (position % 48 + 1).astype(float)
        keys = pl.DataFrame(
            {
                "issue_time": pl.repeat(moment, n, eager=True),
                "valid_time": pl.Series(
                    (int(moment.timestamp()) + position) * 1_000_000
                ).cast(pl.Datetime("us", "UTC")),
                "lead_hours": lead,
                "lead_bucket": np.where(lead <= 24, "0-24h", "24-48h"),
            }
        )
        quantile = (offset // FIXTURE_ROWS) % 2 == 1
        kwargs = {"compact": True} if compact else {}
        yield module.stamped_scores(
            keys,
            evaluation,
            method_id="quantile" if quantile else "point",
            variable="temp_c",
            product="hourly",
            source_kind="live",
            semantics="inst",
            window="expanding",
            fold_origin=moment,
            y_pred=point,
            y_true=np.cos(position / 100),
            quantile_levels=levels if quantile else None,
            quantiles=point[:, None] + np.array(levels)[None, :] if quantile else None,
            **kwargs,
        )


def write_workload(module: ModuleType, count: int, directory: Path) -> Path:
    import polars as pl

    if hasattr(module, "write_score_batches"):
        result = module.write_score_batches(
            fixture_batches(module, count, compact=True),
            directory,
            "hourly",
            "live",
            "expanding",
        )
        return result.path
    frames = list(fixture_batches(module, count, compact=False))
    scores = pl.concat(frames).cast(module.SCORES_SCHEMA)
    del frames
    path = module.scores_path(directory, "hourly", "live", "expanding", "fixed")
    module.write_scores(scores, path)
    return path


def footprint_metrics() -> dict[str, float]:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metrics = {"peak_rss_mib": peak / (1024**2 if sys.platform == "darwin" else 1024)}
    if sys.platform == "darwin":
        result = subprocess.run(
            ["/usr/bin/vmmap", "-summary", str(os.getpid())],
            capture_output=True,
            text=True,
            check=True,
        )
        for label, key in (
            ("Physical footprint:", "physical_footprint_mib"),
            ("Physical footprint (peak):", "peak_physical_footprint_mib"),
        ):
            match = re.search(re.escape(label) + r"\s+([\d.]+)([KMG])", result.stdout)
            if match is None:
                raise RuntimeError(f"vmmap did not report {label}")
            metrics[key] = (
                float(match[1]) * {"K": 1 / 1024, "M": 1, "G": 1024}[match[2]]
            )
        output = subprocess.run(
            [
                "/usr/bin/top",
                "-l",
                "1",
                "-pid",
                str(os.getpid()),
                "-stats",
                "pid,cmprs",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        match = re.search(rf"(?m)^\s*{os.getpid()}\s+([\d.]+)([BKMG])", output)
        if match is None:
            raise RuntimeError("top did not report compressed memory")
        metrics["compressed_mib"] = (
            float(match[1])
            * {"B": 1 / 1024**2, "K": 1 / 1024, "M": 1, "G": 1024}[match[2]]
        )
    return metrics


def value_digest(path: Path, categorical_columns: Sequence[str]) -> str:
    """Hash logical values and row order without expanding repeated metadata.

    Dictionary values get stable SHA-256 tokens, independent of their encoding
    or row-group boundaries. Validity and numeric bits are checked separately.
    """
    import numpy as np
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pl.scan_parquet(path).collect_schema()
    digests = [(hashlib.sha256(), hashlib.sha256()) for _ in schema]
    parquet = pq.ParquetFile(path, read_dictionary=list(categorical_columns))
    for batch in parquet.iter_batches(batch_size=4096):
        for column, (validity_digest, digest) in zip(
            batch.columns, digests, strict=True
        ):
            valid = column.is_valid().to_numpy(zero_copy_only=False)
            validity_digest.update(valid.tobytes())
            if pa.types.is_dictionary(column.type):
                tokens = np.asarray(
                    [
                        hashlib.sha256(value.encode()).digest()
                        for value in column.dictionary.to_pylist()
                    ],
                    dtype="S32",
                )
                values = np.zeros(len(column), dtype="S32")
                values[valid] = tokens[
                    column.indices.filter(column.is_valid()).to_numpy(
                        zero_copy_only=False
                    )
                ]
                digest.update(values.tobytes())
            elif pa.types.is_string(column.type) or pa.types.is_large_string(
                column.type
            ):
                for value in column.to_pylist():
                    encoded = value.encode() if value is not None else b""
                    digest.update(len(encoded).to_bytes(8, "little"))
                    digest.update(encoded)
            else:
                numeric = (
                    column.cast(pa.int64())
                    if pa.types.is_timestamp(column.type)
                    else column
                )
                digest.update(
                    numeric.fill_null(0).to_numpy(zero_copy_only=False).tobytes()
                )
    return hashlib.sha256(
        json.dumps(
            {
                "schema": [(name, str(dtype)) for name, dtype in schema.items()],
                "columns": [
                    (validity.hexdigest(), digest.hexdigest())
                    for validity, digest in digests
                ],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def worker(source: Path, count: int) -> dict[str, Any]:
    sys.path.insert(0, str(source))
    import polars as pl
    import pyarrow as pa

    from grounded_weather_forecast.backtest import scores as module

    if not Path(module.__file__).is_relative_to(source):
        raise RuntimeError("worker imported the wrong source tree")
    with tempfile.TemporaryDirectory(prefix="backtest-memory-worker-") as directory:
        started = time.perf_counter()
        path = write_workload(module, count, Path(directory))
        seconds = time.perf_counter() - started
        # Verification happens after the measurement so reader allocations cannot
        # be mistaken for score-generation memory.
        metrics = footprint_metrics()
        categories = [
            name
            for name, dtype in module.SCORES_SCHEMA.items()
            if dtype == pl.String and name != "quantiles_json"
        ]
        return {
            "rows": count,
            "seconds": seconds,
            "file_bytes": path.stat().st_size,
            "payload_sha256": value_digest(path, categories),
            "python": platform.python_version(),
            "polars": pl.__version__,
            "pyarrow": pa.__version__,
            **metrics,
        }


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for count in sorted({record["rows"] for record in records}):
        for variant in ("baseline", "updated"):
            group = [
                record
                for record in records
                if record["rows"] == count and record["variant"] == variant
            ]
            item = {
                "rows": count,
                "variant": variant,
                "runs": len(group),
                "payload_sha256": group[0]["payload_sha256"],
            }
            for metric in (
                "seconds",
                "file_bytes",
                "peak_rss_mib",
                "peak_physical_footprint_mib",
                "compressed_mib",
            ):
                if metric in group[0]:
                    values = [record[metric] for record in group]
                    item[f"{metric}_median"] = statistics.median(values)
                    item[f"{metric}_range"] = [min(values), max(values)]
            summary.append(item)
    return summary


def acceptance(summary: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {(item["rows"], item["variant"]): item for item in summary}
    exact = all(
        lookup[count, "baseline"]["payload_sha256"]
        == lookup[count, "updated"]["payload_sha256"]
        for count in {item["rows"] for item in summary}
    )
    result = {"exact_output_agreement": exact}
    if all(
        (count, variant) in lookup
        for count in ROW_COUNTS
        for variant in ("baseline", "updated")
    ):
        metric = "peak_physical_footprint_mib_median"
        if metric not in lookup[ROW_COUNTS[-1], "updated"]:
            result["memory_acceptance"] = (
                "requires macOS physical-footprint measurements"
            )
        else:
            ratio = (
                lookup[ROW_COUNTS[-1], "updated"][metric]
                / lookup[ROW_COUNTS[-1], "baseline"][metric]
            )
            growth = (
                lookup[ROW_COUNTS[-1], "updated"][metric]
                - lookup[ROW_COUNTS[0], "updated"][metric]
            )
            result.update(
                memory_ratio=ratio,
                candidate_growth_mib=growth,
                memory_acceptance=ratio <= 0.5 and growth <= 128,
            )
    return result


def collect(
    repository: Path, args: argparse.Namespace, results: BenchmarkResults
) -> list[dict[str, Any]]:
    runner = Path(__file__).resolve()
    runner_hash = hashlib.sha256(runner.read_bytes()).hexdigest()
    baseline_sha = subprocess.check_output(
        ["git", "rev-parse", args.baseline], cwd=repository, text=True
    ).strip()
    with tempfile.TemporaryDirectory(prefix="backtest-memory-benchmark-") as directory:
        root = Path(directory)
        baseline = root / "baseline"
        baseline.mkdir()
        archive = subprocess.check_output(
            ["git", "archive", baseline_sha, "src"], cwd=repository
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
            contents.extractall(baseline, filter="data")
        updated = root / "updated" / "src"
        shutil.copytree(
            repository / "src",
            updated,
            ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"),
        )
        sources = {"baseline": baseline / "src", "updated": updated}
        hashes = {variant: tree_hash(source) for variant, source in sources.items()}
        environment = os.environ | THREADS
        for count in args.rows:
            for repetition in range(args.runs):
                for variant, source in sources.items():
                    record = json.loads(
                        subprocess.check_output(
                            [
                                sys.executable,
                                str(runner),
                                "--worker",
                                "--source",
                                str(source),
                                "--rows",
                                str(count),
                            ],
                            env=environment,
                            text=True,
                        )
                    )
                    record.update(
                        variant=variant,
                        run=repetition + 1,
                        baseline_sha=baseline_sha,
                        source_sha256=hashes[variant],
                        runner_sha256=runner_hash,
                        feature_json_bytes=FEATURE_BYTES,
                        fixture_batch_rows=FIXTURE_ROWS,
                        threads=THREADS,
                        platform=platform.platform(),
                    )
                    results.add_sample(record)
                    peak = record.get(
                        "peak_physical_footprint_mib", record["peak_rss_mib"]
                    )
                    print(
                        f"rows={count} {variant} run={repetition + 1}: {record['seconds']:.2f}s peak={peak:.1f}MiB",
                        flush=True,
                    )
            if (
                len(
                    {
                        record["payload_sha256"]
                        for record in results.records
                        if record["rows"] == count
                    }
                )
                != 1
            ):
                raise RuntimeError(f"score values differ at {count} rows")
        if hashes != {
            variant: tree_hash(source) for variant, source in sources.items()
        }:
            raise RuntimeError("source snapshots changed during benchmark")
        if runner_hash != hashlib.sha256(runner.read_bytes()).hexdigest():
            raise RuntimeError("benchmark runner changed during benchmark")
    return summarize(results.records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="ee93d83")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--rows", type=int, nargs="+", default=list(ROW_COUNTS))
    parser.add_argument(
        "--output",
        type=Path,
        help="new JSONL destination; defaults to unique ignored reports/benchmarks files",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.source, args.rows[0])))
        return
    if (
        args.runs < 3
        or any(count <= 0 for count in args.rows)
        or len(set(args.rows)) != len(args.rows)
    ):
        parser.error("use at least three runs and distinct positive row counts")
    repository = Path(__file__).resolve().parents[1]
    try:
        outputs = BenchmarkOutputs.prepare(repository, "backtest-memory", args.output)
        with outputs.measurements() as results:
            summary = collect(repository, args, results)
            gates = acceptance(summary)
            contents = results.contents(summary=[*summary, {"acceptance": gates}])
        outputs.publish(contents, journal_path=results.journal_path)
    except (OSError, ValueError, JournalPreparationError) as error:
        parser.error(str(error))
    print(json.dumps(gates, sort_keys=True))
    if not gates["exact_output_agreement"] or gates.get("memory_acceptance") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
