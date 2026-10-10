"""Compare complete selection runs in fresh processes using fixed evidence.

Run with uv run python scripts/benchmark_selection_memory.py.
Fixtures, extracted baseline sources, and run artifacts live in a temporary
directory. The output reports total process peak RSS, including score loading.
"""

import argparse
import hashlib
import io
import json
import os
import resource
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path


def prepare(directory: Path, evaluations: int, cases: int) -> None:
    import polars as pl  # noqa: PLC0415 - workers choose their source before imports

    from grounded_weather_forecast.backtest.scores import SCORES_SCHEMA  # noqa: PLC0415

    for scenario in ("many_evaluations", "mostly_incompatible"):
        target = directory / scenario
        target.mkdir()
        for index in range(evaluations):
            current = scenario == "many_evaluations" or index == evaluations - 1
            n_cases = cases if scenario == "many_evaluations" or not current else 80
            valid = [
                datetime(2020, 1, 1, tzinfo=UTC) + timedelta(hours=i)
                for i in range(n_cases)
            ]
            frames = []
            for method in (
                "challenger",
                "equal_weight",
                "best_provider",
                "damped_grounded_equal_weight",
            ):
                n = n_cases
                frame = pl.DataFrame(
                    {
                        "method_id": [method] * n,
                        "variable": ["temp_c"] * n,
                        "product": ["hourly"] * n,
                        "source_kind": ["live"] * n,
                        "evaluation_id": [f"eval-{index:03}"] * n,
                        "evaluation_created_at": [
                            datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index)
                        ]
                        * n,
                        "dataset_fingerprint": ["dataset-fixed"] * n,
                        "config_fingerprint": [
                            "config-fixed" if current else "obsolete"
                        ]
                        * n,
                        "code_version": ["code-fixed"] * n,
                        "source_set_json": ['["nws"]'] * n,
                        "feature_set_json": [json.dumps(["x" * 8192])] * n,
                        "semantics": ["inst"] * n,
                        "window": ["expanding"] * n,
                        "fold_origin": [datetime(2019, 1, 1, tzinfo=UTC)] * n,
                        "issue_time": [value - timedelta(hours=30) for value in valid],
                        "valid_time": valid,
                        "lead_hours": [30.0] * n,
                        "lead_bucket": ["24-48h"] * n,
                        "y_pred": [1.0 if method == "challenger" else 5.0] * n,
                        "y_true": [0.0] * n,
                        "quantile_levels_json": ["[]"] * n,
                        "quantiles_json": [None] * n,
                    },
                    schema=SCORES_SCHEMA,
                )
                frames.append(frame)
            pl.concat(frames).write_parquet(target / f"scores_{index:03}.parquet")
    (directory / "config.toml").write_text(f"""[station]
db_path = "{directory / "station.db"}"
timezone = "UTC"
latitude = 0.0
longitude = 0.0
elevation_m = 0.0
[forecasts]
db_path = "{directory / "forecasts.db"}"
[dataset]
dir = "{directory / "data"}"
""")


def worker(source: Path, directory: Path, scenario: str) -> None:
    sys.path.insert(0, str(source))
    import grounded_weather_forecast.serve.selection as selection  # noqa: PLC0415
    from grounded_weather_forecast.config import load_config  # noqa: PLC0415

    config = load_config(directory / "config.toml")
    with tempfile.TemporaryDirectory(prefix="selection-run-") as run:
        config = replace(config, artifacts_dir=Path(run))
        selection.dataset_fingerprint = lambda _: "dataset-fixed"
        selection.config_fingerprint = lambda _: "config-fixed"
        selection.code_identity = lambda: "code-fixed"
        started = time.perf_counter()
        result = selection.select_methods(config, directory / scenario)
        elapsed = time.perf_counter() - started
        payload = selection._selection_payload(result)
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(
            json.dumps(
                {
                    "source": selection.__file__,
                    "scenario": scenario,
                    "peak_rss_mib": peak
                    / (1024**2 if sys.platform == "darwin" else 1024),
                    "seconds": elapsed,
                    "selection_sha256": digest,
                }
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="a4ec76d")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--evaluations", type=int, default=12)
    parser.add_argument("--cases", type=int, default=4000)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--fixtures", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--scenario", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.source, args.fixtures, args.scenario)
        return
    if args.runs < 3 or args.evaluations < 1 or args.cases < 1:
        parser.error("use at least three runs and positive evaluation/case counts")
    repository = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="selection-benchmark-") as temporary:
        directory = Path(temporary)
        prepare(directory, args.evaluations, args.cases)
        baseline = directory / "baseline"
        baseline.mkdir()
        archive = subprocess.check_output(  # noqa: S603 - trusted Git argv, no shell
            ["git", "archive", args.baseline, "src"],  # noqa: S607
            cwd=repository,
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
            contents.extractall(baseline, filter="data")
        records = []
        for scenario in ("many_evaluations", "mostly_incompatible"):
            for variant, source in (
                ("baseline", baseline / "src"),
                ("updated", repository / "src"),
            ):
                for _ in range(args.runs):
                    environment = os.environ | {"PYTHONPATH": str(source)}
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--source",
                        str(source),
                        "--fixtures",
                        str(directory),
                        "--scenario",
                        scenario,
                    ]
                    record = json.loads(
                        subprocess.check_output(  # noqa: S603 - this script, no shell
                            command, env=environment, text=True
                        )
                    )
                    if not Path(record["source"]).is_relative_to(source):
                        raise RuntimeError("worker imported the wrong source tree")
                    records.append(record | {"variant": variant})
        for scenario in ("many_evaluations", "mostly_incompatible"):
            selected = [record for record in records if record["scenario"] == scenario]
            if len({record["selection_sha256"] for record in selected}) != 1:
                raise RuntimeError("selection payloads differ between variants/runs")
            for variant in ("baseline", "updated"):
                group = [record for record in selected if record["variant"] == variant]
                rss = [record["peak_rss_mib"] for record in group]
                seconds = [record["seconds"] for record in group]
                print(
                    json.dumps(
                        {
                            "scenario": scenario,
                            "variant": variant,
                            "peak_rss_median_mib": statistics.median(rss),
                            "peak_rss_range_mib": [min(rss), max(rss)],
                            "seconds_median": statistics.median(seconds),
                            "seconds_range": [min(seconds), max(seconds)],
                            "runs": args.runs,
                            "selection_sha256": group[0]["selection_sha256"],
                            "samples": [
                                {
                                    "peak_rss_mib": record["peak_rss_mib"],
                                    "seconds": record["seconds"],
                                }
                                for record in group
                            ],
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
