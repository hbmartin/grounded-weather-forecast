"""Benchmark immutable source snapshots and isolated score-reader workloads.

All report/artifact writes stay inside temporary directories. Each measurement
runs in a fresh process. Source-tree hashes identify uncommitted implementations.
"""

# Workers must select their immutable source tree before importing the package.
# ruff: noqa: PLC0415

import argparse
import contextlib
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
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from benchmark_selection_memory import prepare

POLICIES = ("bounded", "auto_low_memory", "auto")
WORKLOADS = ("many_evaluations", "mostly_incompatible", "dashboard", "report")
SCENARIOS = (*WORKLOADS[:2], "reporting")


def prepare_reporting(fixtures: Path) -> None:
    """Keep the report's sequential wealth finite and populate both chart types."""
    import polars as pl

    target = fixtures / "reporting"
    target.mkdir()
    for path in sorted((fixtures / "many_evaluations").glob("*.parquet")):
        scores = pl.read_parquet(path)
        alternating = (
            pl.when(pl.int_range(pl.len()) % 2 == 0).then(0.25).otherwise(-0.25)
        )
        point = scores.with_columns(
            pl.when(pl.col("method_id") == "challenger")
            .then(5.0 + alternating)
            .otherwise(pl.col("y_pred"))
            .alias("y_pred"),
            pl.lit("[0.1, 0.5, 0.9]").alias("quantile_levels_json"),
            pl.lit("[4.0, 5.0, 6.0]").alias("quantiles_json"),
        )
        probability = scores.with_columns(
            pl.lit("pop").alias("variable"),
            pl.lit(0.5).alias("y_pred"),
            (pl.int_range(pl.len()) % 2).cast(pl.Float64).alias("y_true"),
            pl.lit("[0.1, 0.5, 0.9]").alias("quantile_levels_json"),
            pl.lit("[0.0, 0.5, 1.0]").alias("quantiles_json"),
        )
        pl.concat([point, probability]).write_parquet(target / path.name)


def tree_hash(source: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*")):
        if (
            path.is_file()
            and "__pycache__" not in path.parts
            and path.name != ".DS_Store"
        ):
            digest.update(path.relative_to(source).as_posix().encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def digest_payload(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def dashboard_payload(path: Path) -> object:
    match = re.search(
        r'<script[^>]*id="dashboard-data"[^>]*>(.*?)</script>',
        path.read_text(),
        re.DOTALL,
    )
    if match is None:
        raise RuntimeError("dashboard payload missing")
    return json.loads(match.group(1))


def canonical_markdown(markdown: str) -> str:
    """Compare table contents independently of undefined aggregate tie order."""

    def table(match: re.Match) -> str:
        rows = match.group().splitlines()
        return "\n".join([*rows[:2], *sorted(rows[2:])]) + "\n"

    return re.sub(r"(?m)(?:^\|.*\n)+", table, markdown)


def worker(source: Path, fixtures: Path, workload: str, policy: str) -> dict:
    sys.path.insert(0, str(source))
    import polars as pl

    import grounded_weather_forecast.evaluation as evaluation
    from grounded_weather_forecast.config import load_config

    evaluation.code_identity = lambda: "code-fixed"
    evaluation.config_fingerprint = lambda _: "config-fixed"
    evaluation.dataset_fingerprint = lambda _: "dataset-fixed"
    import grounded_weather_forecast.serve.selection as selection

    if not Path(selection.__file__).is_relative_to(source):
        raise RuntimeError("worker imported the wrong source tree")
    selection.code_identity = evaluation.code_identity
    selection.config_fingerprint = evaluation.config_fingerprint
    selection.dataset_fingerprint = evaluation.dataset_fingerprint
    real_scan = pl.scan_parquet
    reads = []
    parallel, low_memory = (
        ("none", True) if policy == "bounded" else ("auto", policy == "auto_low_memory")
    )

    def scan(path: Path | str, **kwargs: Any) -> pl.LazyFrame:
        kwargs.update(parallel=parallel, low_memory=low_memory)
        return real_scan(path, **kwargs)

    pl.scan_parquet = scan
    from grounded_weather_forecast.backtest import scores as score_module

    real_load = score_module.load_scores

    def load(path: Path | str, **kwargs: Any) -> pl.DataFrame:
        reads.append(Path(path).name)
        return real_load(path, **kwargs)

    score_module.load_scores = load
    config = load_config(fixtures / "config.toml")
    with tempfile.TemporaryDirectory(prefix="score-reader-worker-") as temporary:
        directory = Path(temporary)
        dataset = directory / "data"
        dataset.mkdir()
        config = replace(
            config,
            dataset=replace(config.dataset, dir=dataset),
            artifacts_dir=directory / "artifacts",
            reports_dir=directory / "reports",
            predict=replace(config.predict, history_path=directory / "history.parquet"),
        )
        scenario = workload if workload in WORKLOADS[:2] else "reporting"
        if workload in WORKLOADS[2:]:
            archive = dataset / "scores"
            archive.mkdir()
            for index, path in enumerate(
                sorted((fixtures / scenario).glob("*.parquet"))
            ):
                os.link(
                    path,
                    archive / f"scores_hourly_live_expanding_eval-{index:03}.parquet",
                )
        # Import the measured entry point before starting its wall clock.
        moment = datetime(2026, 10, 10, tzinfo=UTC)
        if workload == "dashboard":
            from grounded_weather_forecast.dashboard import write_dashboard
        elif workload == "report":
            from grounded_weather_forecast import dashboard
            from grounded_weather_forecast.cli import _cmd_report
            from grounded_weather_forecast.reports import evidence, operations

            # Fix recorder and dashboard clocks without changing datetime type checks.
            for name in dir(evidence):
                if name.startswith("record_"):
                    setattr(
                        evidence, name, partial(getattr(evidence, name), now=moment)
                    )
            operations.record_operations = partial(
                operations.record_operations, now=moment
            )
            dashboard.write_dashboard = partial(dashboard.write_dashboard, now=moment)
        started = time.perf_counter()
        if workload in WORKLOADS[:2]:
            selected = selection.select_methods(config, fixtures / scenario)
            payload = selection._selection_payload(selected)
        elif workload == "dashboard":
            path = write_dashboard(config, now=moment)
            payload = dashboard_payload(path)
        else:
            # Report output is diagnostic noise, not part of the measurement.
            with contextlib.redirect_stdout(io.StringIO()):
                code = _cmd_report(config)
            if code != 0:
                raise RuntimeError(f"report exited {code}")
            payload = {
                "tables": {
                    path.name: path.read_text()
                    for path in sorted(config.reports_dir.glob("leaderboard_*.md"))
                },
                "charts": dashboard_payload(config.reports_dir / "dashboard.html")[
                    "charts"
                ],
            }
        seconds = time.perf_counter() - started
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rendered_digest = digest_payload(payload)
        if workload == "report":
            payload["tables"] = {
                name: canonical_markdown(markdown)
                for name, markdown in payload["tables"].items()
            }
        return {
            "workload": workload,
            "policy": policy,
            "seconds": seconds,
            "peak_rss_mib": peak / (1024**2 if sys.platform == "darwin" else 1024),
            "payload_sha256": digest_payload(payload),
            "rendered_sha256": rendered_digest,
            "full_decodes": len(reads),
            "python": platform.python_version(),
            "polars": pl.__version__,
            "payload": payload,
        }


def summarize(records: list[dict]) -> list[dict]:
    groups = {}
    for record in records:
        groups.setdefault(
            (record["workload"], record["variant"], record["policy"]), []
        ).append(record)
    summaries = []
    for (workload, variant, policy), group in groups.items():
        seconds = [record["seconds"] for record in group]
        rss = [record["peak_rss_mib"] for record in group]
        summaries.append(
            {
                "workload": workload,
                "variant": variant,
                "policy": policy,
                "runs": len(group),
                "seconds_median": statistics.median(seconds),
                "seconds_range": [min(seconds), max(seconds)],
                "peak_rss_median_mib": statistics.median(rss),
                "peak_rss_range_mib": [min(rss), max(rss)],
                "full_decodes": sorted({record["full_decodes"] for record in group}),
                "payload_sha256": group[0]["payload_sha256"],
            }
        )
    return summaries


def run(args: argparse.Namespace) -> None:
    repository = Path(__file__).resolve().parents[1]
    baseline_sha = subprocess.check_output(  # noqa: S603 -- Git argv, no shell
        ["git", "rev-parse", args.baseline],  # noqa: S607
        cwd=repository,
        text=True,
    ).strip()
    with tempfile.TemporaryDirectory(prefix="score-reader-benchmark-") as temporary:
        directory = Path(temporary)
        fixtures = directory / "fixtures"
        fixtures.mkdir()
        prepare(fixtures, args.evaluations, args.cases)
        prepare_reporting(fixtures)
        for path in fixtures.rglob("*.parquet"):
            path.chmod(0o444)
        baseline = directory / "baseline"
        baseline.mkdir()
        archive = subprocess.check_output(  # noqa: S603 -- resolved Git commit
            ["git", "archive", baseline_sha, "src"],  # noqa: S607
            cwd=repository,
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
            contents.extractall(baseline, filter="data")
        updated = directory / "updated" / "src"
        shutil.copytree(
            repository / "src",
            updated,
            ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"),
        )
        sources = {"baseline": baseline / "src", "updated": updated}
        hashes = {variant: tree_hash(source) for variant, source in sources.items()}
        # Exclude the location-dependent config from the stable fixture hash.
        fixture_hashes = {name: tree_hash(fixtures / name) for name in SCENARIOS}
        fixture_hash = digest_payload(fixture_hashes)
        fixture_details = {
            name: {
                "sha256": fixture_hashes[name],
                "files": args.evaluations,
                "compressed_bytes": sum(
                    path.stat().st_size for path in (fixtures / name).glob("*.parquet")
                ),
                "rows": (
                    args.evaluations * args.cases * 4
                    if name == "many_evaluations"
                    else (args.evaluations - 1) * args.cases * 4 + 80 * 4
                    if name == "mostly_incompatible"
                    else args.evaluations * args.cases * 8
                ),
            }
            for name in SCENARIOS
        }
        environment = os.environ | {
            "POLARS_MAX_THREADS": "2",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
        records = []
        payloads = {}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("")
        for workload in WORKLOADS:
            for repetition in range(args.runs):
                for variant, policies in (
                    ("baseline", ("bounded",)),
                    ("updated", POLICIES),
                ):
                    for policy in policies:
                        command = [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker",
                            "--source",
                            str(sources[variant]),
                            "--fixtures",
                            str(fixtures),
                            "--workload",
                            workload,
                            "--policy",
                            policy,
                        ]
                        record = json.loads(
                            subprocess.check_output(  # noqa: S603 -- this script, no shell
                                command, env=environment, text=True
                            )
                        )
                        payload = record.pop("payload")
                        if repetition == 0:
                            payloads[f"{workload}/{variant}/{policy}"] = payload
                        record.update(
                            variant=variant,
                            run=repetition + 1,
                            source_sha256=hashes[variant],
                            runner_sha256=hashlib.sha256(
                                Path(__file__).read_bytes()
                            ).hexdigest(),
                            baseline_sha=baseline_sha,
                            fixture_sha256=fixture_hash,
                            fixture_details=fixture_details,
                            evaluations=args.evaluations,
                            cases=args.cases,
                            feature_json_bytes=8196,
                            threads={
                                key: environment[key]
                                for key in (
                                    "POLARS_MAX_THREADS",
                                    "OMP_NUM_THREADS",
                                    "OPENBLAS_NUM_THREADS",
                                    "MKL_NUM_THREADS",
                                )
                            },
                            platform=platform.platform(),
                        )
                        records.append(record)
                        with args.output.open("a") as output:
                            output.write(json.dumps(record, sort_keys=True) + "\n")
                        print(
                            f"{workload} {variant} {policy} run={repetition + 1}: {record['seconds']:.3f}s {record['peak_rss_mib']:.1f}MiB",
                            flush=True,
                        )
            hashes_for_workload = {
                record["payload_sha256"]
                for record in records
                if record["workload"] == workload
            }
            if len(hashes_for_workload) != 1:
                args.output.with_suffix(".payloads.json").write_text(
                    json.dumps(payloads, indent=2, sort_keys=True) + "\n"
                )
                raise RuntimeError(f"payloads differ for {workload}")
        if fixture_hashes != {name: tree_hash(fixtures / name) for name in SCENARIOS}:
            raise RuntimeError("benchmark inputs changed")
        if hashes != {
            variant: tree_hash(source) for variant, source in sources.items()
        }:
            raise RuntimeError("benchmark source snapshots changed")
        summary_path = args.output.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summarize(records), indent=2) + "\n")
        print(f"Saved {args.output} and {summary_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="6a4933f")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--evaluations", type=int, default=12)
    parser.add_argument("--cases", type=int, default=4000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/research/score-readers-2026-10-10.jsonl"),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--fixtures", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--workload", choices=WORKLOADS, help=argparse.SUPPRESS)
    parser.add_argument("--policy", choices=POLICIES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(
            json.dumps(worker(args.source, args.fixtures, args.workload, args.policy))
        )
    elif args.runs < 3 or args.evaluations < 1 or args.cases < 1:
        parser.error("use at least three runs and positive fixture sizes")
    else:
        run(args)


if __name__ == "__main__":
    main()
