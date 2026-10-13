"""Measure diagnostic collection and fitting in isolated immutable source trees.

Use uv run python scripts/benchmark_fit_diagnostics.py. All temporary configs and
matrix fixtures stay outside the checkout; raw samples identify both source trees.
"""

# Worker source selection precedes package imports; asserts validate measured results.
# Git/tar are fixed argv commands; worker source imports intentionally happen later.
# ruff: noqa: PLC0415, S101, S607

import argparse
import hashlib
import importlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

THREADS = {
    "POLARS_MAX_THREADS": "2",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}


def source_hash(source: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        digest.update(path.relative_to(source).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def module_origins(source: Path) -> dict[str, str]:
    """Reject cached or editable imports outside the requested immutable tree."""
    origins = {}
    for name, module in tuple(sys.modules.items()):
        if name != "grounded_weather_forecast" and not name.startswith(
            "grounded_weather_forecast."
        ):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            raise RuntimeError(f"worker cannot verify module origin: {name}")
        path = Path(filename).resolve(strict=True)
        if not path.is_relative_to(source):
            raise RuntimeError(
                f"worker imported the wrong source tree: {name} at {path}"
            )
        origins[name] = str(path)
    return origins


def output_path(repository: Path, supplied: Path | None) -> Path:
    return (
        repository / "docs/research/fit-diagnostics-2026-10-13.jsonl"
        if supplied is None
        else supplied
    )


def worker(args: argparse.Namespace) -> dict[str, object]:
    source = args.source.resolve(strict=True)
    if not source.is_dir():
        raise ValueError(f"worker source must be a directory: {source}")
    sys.path.insert(0, str(source))
    import numpy as np
    import polars as pl

    from grounded_weather_forecast.config import load_config
    from grounded_weather_forecast.contracts import TruthSemantics, hourly_variable
    from grounded_weather_forecast.leads import hourly_bucket
    from grounded_weather_forecast.serve.diagnostics import FitDiagnostics
    from grounded_weather_forecast.serve.selection import Selection

    module = importlib.import_module("grounded_weather_forecast.serve.predict")
    module_origins(source)
    now = datetime(2026, 10, 12, 12, tzinfo=UTC)
    truth_names = ["t__temp_c__inst", *(f"t__unused{i}" for i in range(12))]
    raw = pl.DataFrame(
        {name: np.linspace(0, 1, args.rows) for name in truth_names}
    ).with_columns(
        pl.lit(now - timedelta(hours=3)).alias("issue_time"),
        pl.lit(
            now if args.workload == "pending_fit" else now - timedelta(hours=3)
        ).alias("valid_time"),
        pl.lit(1.0).alias("lead_hours"),
        pl.lit("live").alias("source_kind"),
        pl.lit(2.0).alias("fx__nws__temp_c"),
        pl.lit(3.0).alias("fx__open_meteo__temp_c"),
    )
    future = pl.DataFrame(
        {
            "issue_time": [now],
            "valid_time": [now + timedelta(hours=1)],
            "lead_hours": [1.0],
            "fx__nws__temp_c": [2.0],
            "fx__open_meteo__temp_c": [3.0],
        }
    )
    selection = {
        ("hourly", "temp_c", hourly_bucket(1.0)): Selection(
            "equal_weight", "promoted", release_id="release", truth_semantics="inst"
        )
    }
    updated = hasattr(FitDiagnostics, "resolve_training")
    with tempfile.TemporaryDirectory(prefix="fit-worker-") as temporary:
        directory = Path(temporary)
        config_path = directory / "config.toml"
        config_path.write_text(f"""[station]
db_path = "{directory / "station.db"}"
timezone = "UTC"
latitude = 34.0
longitude = -117.0
elevation_m = 1400.0
[forecasts]
db_path = "{directory / "forecasts.db"}"
[dataset]
dir = "{directory / "data"}"
""")
        config = load_config(config_path)
        eligible = raw.head(int(args.rows * 0.9))

        def run() -> dict[str, object]:
            diagnostics = FitDiagnostics()
            if args.workload == "aggregation":
                if updated:
                    diagnostics.capture_training("hourly", raw)
                else:
                    diagnostics.capture_training(
                        "hourly", raw, eligible, has_forecasts=True
                    )
                return {"raw_rows": diagnostics.training["hourly"].raw_rows}
            train = module._training_matrix(
                config, "hourly", now, frame=raw, fit_diagnostics=diagnostics
            )
            result = module._blend_variable(
                train,
                future,
                hourly_variable("temp_c"),
                selection,
                config,
                daily=False,
                semantics=TruthSemantics.INSTANTANEOUS,
                fit_diagnostics=diagnostics,
            )
            if updated:
                diagnostics.resolve_training("hourly", raw, now)
            if args.workload == "successful_fit":
                assert result.release_ids == ["release"]
                assert not diagnostics.failures
            else:
                assert result.release_ids == [None]
                assert len(diagnostics.failures) == 1
            return {
                "point": result.point.tolist(),
                "release_ids": result.release_ids,
                "methods": result.methods,
                "quantiles": result.quantiles,
            }

        payload = run()
        origins = module_origins(source)
        times = []
        for _ in range(9):
            started = time.perf_counter()
            assert run() == payload
            times.append((time.perf_counter() - started) * 1000)
    return {
        "workload": args.workload,
        "rows": args.rows,
        "truth_columns": 13,
        "median_ms": statistics.median(times),
        "samples_ms": times,
        "payload": payload,
        "python": platform.python_version(),
        "polars": pl.__version__,
        "threads": THREADS,
        "source_root": str(source),
        "module_origins": origins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="2ccd7e0")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        help="output JSONL path; defaults to the repository's dated experiment path",
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--rows", type=int)
    parser.add_argument(
        "--workload", choices=("aggregation", "successful_fit", "pending_fit")
    )
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args)))
        return
    repo = Path(__file__).resolve().parents[1]
    destination = output_path(repo, args.output)
    baseline_sha = subprocess.check_output(  # noqa: S603 -- Git argv, no shell
        ["git", "rev-parse", args.baseline],
        cwd=repo,
        text=True,
    ).strip()
    records = []
    with tempfile.TemporaryDirectory(prefix="fit-benchmark-") as temporary:
        root = Path(temporary)
        baseline = root / "baseline"
        baseline.mkdir()
        archive = subprocess.check_output(  # noqa: S603 -- resolved Git commit
            ["git", "archive", baseline_sha, "src"],
            cwd=repo,
        )
        subprocess.run(  # noqa: S603 -- archive generated by Git above
            ["tar", "-xf", "-", "-C", str(baseline)],
            input=archive,
            check=True,
        )
        updated = root / "updated"
        shutil.copytree(
            repo / "src", updated / "src", ignore=shutil.ignore_patterns("__pycache__")
        )
        sources = {"baseline": baseline / "src", "updated": updated / "src"}
        hashes = {name: source_hash(source) for name, source in sources.items()}
        for workload in ("aggregation", "successful_fit", "pending_fit"):
            for rows in (100_000, 1_000_000):
                for repetition in range(args.runs):
                    for variant, source in sources.items():
                        command = [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker",
                            "--source",
                            str(source),
                            "--workload",
                            workload,
                            "--rows",
                            str(rows),
                        ]
                        record = json.loads(
                            subprocess.check_output(  # noqa: S603 -- isolated worker argv
                                command, env=os.environ | THREADS, text=True
                            )
                        )
                        record.update(
                            variant=variant,
                            repetition=repetition + 1,
                            source_sha256=hashes[variant],
                            baseline_sha=baseline_sha,
                            runner_sha256=hashlib.sha256(
                                Path(__file__).read_bytes()
                            ).hexdigest(),
                        )
                        records.append(record)
                        print(
                            f"{workload} {rows} {variant} run {repetition + 1}: {record['median_ms']:.3f} ms",
                            flush=True,
                        )
        assert hashes["updated"] == source_hash(repo / "src"), (
            "source changed during benchmark"
        )
    summary = []
    for workload in ("aggregation", "successful_fit", "pending_fit"):
        for rows in (100_000, 1_000_000):
            group = [
                record
                for record in records
                if record["workload"] == workload and record["rows"] == rows
            ]
            assert (
                len({json.dumps(record["payload"], sort_keys=True) for record in group})
                == 1
            )
            medians = {
                variant: statistics.median(
                    record["median_ms"]
                    for record in group
                    if record["variant"] == variant
                )
                for variant in ("baseline", "updated")
            }
            summary.append(
                {
                    "workload": workload,
                    "rows": rows,
                    "median_ms": medians,
                    "payloads_equal": True,
                }
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    destination.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
