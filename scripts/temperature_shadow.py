"""Inactive local temperature shadow: record immutable issues and score later.

Consumes supplied files only. It never calls the production CLI or a provider,
opens SQLite, publishes, promotes, or installs a job. See docs/reference/cli.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from shadow_candidate_v1 import (
    bias_corrections,
    fit_simplex,
    groups,
    independent_training,
    observed_climate,
    safe_blend,
)

from grounded_weather_forecast.blenders.baselines import BestProvider
from grounded_weather_forecast.contracts import TruthSemantics, hourly_variable
from grounded_weather_forecast.dataset.matrix import (
    to_forecast_matrix,
    to_supervised_slice,
)
from grounded_weather_forecast.evaluation import code_identity

METHODS = (
    "production",
    "candidate",
    "decay21",
    "best_provider",
    "persistence",
    "climatology",
)
HORIZONS = ("0-6h", "6-24h", "24-72h", "72-96h")
TARGET = "t__temp_c__mean"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("timestamps must include timezone")
    return result.astimezone(UTC)


def write_new(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False, default=str)


def number(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def implementation_lock(protocol_path: Path) -> dict[str, str]:
    return {
        "protocol_sha256": digest(protocol_path),
        "runner_sha256": digest(Path(__file__)),
        "candidate_sha256": digest(Path(__file__).with_name("shadow_candidate_v1.py")),
        "code_identity": code_identity(),
    }


def freeze(protocol_path: Path, output: Path) -> Path:
    """Seal a reviewed, future enrollment without scheduling any work."""
    protocol = json.loads(protocol_path.read_text())
    template = json.loads(
        (
            Path(__file__).parents[1] / "examples/temperature-shadow/protocol.json"
        ).read_text()
    )
    fixed = template.keys() - {"window_start", "window_end", "status"}
    if any(protocol.get(key) != template[key] for key in fixed):
        raise ValueError(
            "candidate and evaluation rules are fixed; review a new protocol version"
        )
    if timestamp(protocol["window_start"]) <= datetime.now(UTC):
        raise ValueError(
            "enrollment window has started; explicit review of a new window is required"
        )
    if timestamp(protocol["window_end"]) - timestamp(
        protocol["window_start"]
    ) != timedelta(days=42):
        raise ValueError("the predeclared initial window must be six weeks")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    target = output / "protocol.json"
    target.write_bytes(protocol_path.read_bytes())
    write_new(target.with_suffix(".lock.json"), implementation_lock(target))
    return target


def capsule_metadata(capsule: Path) -> dict[str, Any]:
    meta = json.loads((capsule / "capsule.json").read_text())
    receipt = capsule / "publication.json"
    if receipt.exists():
        meta["publication"] = json.loads(receipt.read_text())
    for name, expected in meta.get("input_sha256", {}).items():
        if digest(capsule / name) != expected:
            raise ValueError(f"captured input changed: {name}")
    return meta


def fetch_publication(remote: str, branch: str) -> tuple[str, bytes]:
    """Read exact committed bytes into an isolated temporary Git object store."""
    git = shutil.which("git")
    if git is None or remote.startswith("-") or not branch.startswith("refs/heads/"):
        raise ValueError("Git and an explicit remote/branch ref are required")
    with tempfile.TemporaryDirectory(prefix="shadow-publication-") as folder:

        def run(*args: str) -> bytes:
            return subprocess.run(  # noqa: S603 -- explicit Git reads, no shell
                [git, "-C", folder, *args],
                check=True,
                capture_output=True,
                timeout=30,
            ).stdout

        run("init", "--bare", "--quiet")
        run("fetch", "--quiet", "--depth=1", "--no-tags", remote, branch)
        commit = run("rev-parse", "FETCH_HEAD").decode().strip()
        payload = run("show", f"{commit}:forecast.json")
    return commit, payload


def finalize(
    capsule: Path,
    output: Path,
    protocol_path: Path,
    *,
    remote: str,
    branch: str,
    source: Path | None,
    mode: str,
) -> Path:
    """Seal a successful public commit and score the original captured issue."""
    meta = capsule_metadata(capsule)
    if source is not None and digest(source) != digest(capsule / "production.json"):
        raise ValueError(
            "publisher chose a different source; do not relabel held forecasts"
        )
    commit, payload = fetch_publication(remote, branch)
    published = json.loads(payload)
    if published["issued_at"] != meta["input_as_of"]:
        raise ValueError("public commit belongs to a different issuance")
    with (capsule / "published.json").open("xb") as stream:
        stream.write(payload)
    write_new(
        capsule / "publication.json",
        {
            "commit": commit,
            "issued_at": published["issued_at"],
            "published_at": published["published_at"],
            "source_sha256": digest(capsule / "production.json"),
            "published_sha256": digest(capsule / "published.json"),
            "remote": remote,
            "branch": branch,
        },
    )
    return record(capsule, output, protocol_path, mode=mode)


def validate_provenance(meta: dict[str, Any], document: dict[str, Any]) -> None:
    provenance = meta.get("provenance", {})
    required = (
        "production_code_identity",
        "config_fingerprint",
        "dataset_fingerprint",
        "availability_assumptions",
        "provider_completion_clock",
        "station_receipt_clock",
    )
    if any(
        not isinstance(provenance.get(k), str) or not provenance[k] for k in required
    ):
        raise ValueError(
            "prospective requires explicit code/config/data and availability provenance"
        )
    if provenance["dataset_fingerprint"] != document["dataset_fingerprint"]:
        raise ValueError("input data provenance differs from production")


def production_rows(
    capsule: Path, meta: dict[str, Any], *, prospective: bool
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    document = json.loads((capsule / "production.json").read_text())
    rows = {row["valid_time"]: row for row in document["hourly"]}
    if any(
        row.get("truth_semantics", {}).get("temp_c") != "mean" for row in rows.values()
    ):
        raise ValueError("production temperature must target interval mean truth")
    if not prospective:
        return document, rows
    if "publication" not in meta or not (capsule / "published.json").is_file():
        raise ValueError(
            "prospective requires a committed payload and publication receipt"
        )
    validate_provenance(meta, document)
    receipt = meta["publication"]
    published_path = capsule / "published.json"
    published = json.loads(published_path.read_text())
    commit = receipt["commit"]
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("prospective publication needs full commit identity")
    if (
        receipt["published_sha256"] != digest(published_path)
        or receipt["source_sha256"] != digest(capsule / "production.json")
        or receipt["issued_at"] != document["issued_at"]
        or published["issued_at"] != document["issued_at"]
        or published["dataset_fingerprint"] != document["dataset_fingerprint"]
        or set(published["release_ids"]) != set(document["release_ids"])
        or published.get("units", {}).get("temp_f") != "F"
        or timestamp(receipt["published_at"]) != timestamp(published["published_at"])
    ):
        raise ValueError("publication receipt/source/payload mismatch")
    for row in published["hourly"]:
        original = rows[row["valid_time"]]
        original_value = original["values"].get("temp_c")
        value_f = row["values"].get("temp_f")
        if (original_value is None) != (value_f is None):
            raise ValueError("published temperature availability differs")
        if value_f is not None:
            value_c = (value_f - 32) / 1.8
            if abs(value_c - original_value) > 0.004:
                raise ValueError("published temperature differs from source")
            # Evaluate the actual public, rounded value.
            original["values"]["temp_c"] = value_c
    if {row["valid_time"] for row in published["hourly"]} != set(rows):
        raise ValueError("published timestamp grid differs")
    return document, rows


def predictions(
    train: pl.DataFrame, issue: pl.DataFrame, truth: pl.DataFrame, origin: datetime
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    train = independent_training(
        train.filter(
            (pl.col("issue_time") <= origin)
            & (pl.col("valid_time") + timedelta(hours=2) <= origin)
            & pl.col(TARGET).is_finite()
        )
    )
    if train["valid_time"].n_unique() < 48:
        raise ValueError("need at least 48 unique known training hours")
    spec = hourly_variable("temp_c")
    supervised = to_supervised_slice(
        train, spec, semantics=TruthSemantics.INTERVAL_MEAN
    )
    x = to_forecast_matrix(issue, spec, sources=supervised.x.sources)
    correction, row_weights = bias_corrections(train, "temp_c", origin, "decay21")
    train_corrected = supervised.x.values - correction[groups(supervised.x.lead_hours)]
    weights = fit_simplex(train_corrected, supervised.y, row_weights)
    corrected = x.values - correction[groups(x.lead_hours)]
    best = BestProvider().fit(supervised)
    obs = (
        issue["obs__temp_c"].to_numpy()
        if "obs__temp_c" in issue.columns
        else np.full(issue.height, np.nan)
    )
    values = {
        "candidate": safe_blend(corrected, weights),
        "decay21": safe_blend(corrected, np.ones(len(x.sources))),
        "best_provider": best.predict(x).point,
        "persistence": obs,
        "climatology": observed_climate(truth, issue, "temp_c", origin),
    }
    return values, {
        "sources": x.sources,
        "source_weights": weights.tolist(),
        "corrections": correction.tolist(),
        "unique_train_hours": train["valid_time"].n_unique(),
        "unique_train_dates": train["valid_time"].dt.date().n_unique(),
        "train_rows": train.height,
        "last_train_valid": train["valid_time"].max(),
        "best_provider_state": best.to_state(),
        "optimizer_equal_result": bool(
            np.allclose(weights, np.full(len(weights), 1 / len(weights)))
        ),
        "optimizer_equal_note": "equal result can mean thin fit, failed solve, or an optimal equal vector; do not infer failure from weights alone",
    }


def record(capsule: Path, output: Path, protocol_path: Path, *, mode: str) -> Path:
    prospective = mode == "prospective"
    now = datetime.now(UTC)
    meta = capsule_metadata(capsule)
    protocol = json.loads(protocol_path.read_text())
    lock = json.loads(protocol_path.with_suffix(".lock.json").read_text())
    if lock != implementation_lock(protocol_path):
        raise ValueError(
            "frozen protocol or implementation changed; review a new enrollment"
        )
    document, production = production_rows(capsule, meta, prospective=prospective)
    origin = timestamp(document["issued_at"])
    if timestamp(meta["input_as_of"]) != origin:
        raise ValueError("features must have the production issuance time")
    if prospective and (
        meta["capture_kind"] != "serving_snapshot"
        or not timestamp(protocol["window_start"])
        <= origin
        < timestamp(protocol["window_end"])
        or not origin <= now <= origin + timedelta(minutes=30)
        or not origin <= timestamp(meta["publication"]["published_at"]) <= now
        or document.get("timezone") != "America/Los_Angeles"
    ):
        raise ValueError(
            "prospective requires a contemporaneous serving snapshot in the frozen window"
        )
    issue = pl.read_parquet(capsule / "issue.parquet").sort("valid_time")
    if issue["issue_time"].n_unique() != 1 or issue["issue_time"][0] != origin:
        raise ValueError("issue matrix must contain exactly this issuance")
    issue = issue.filter((pl.col("lead_hours") >= 0) & (pl.col("lead_hours") <= 96))
    if issue.is_empty() or issue["valid_time"].n_unique() != issue.height:
        raise ValueError("empty or duplicated forecast grid")
    if {v.isoformat() for v in issue["valid_time"]} != set(production):
        raise ValueError("input matrix must cover exactly the published hourly grid")
    observation_at = document.get("observation_at")
    if (
        observation_at is None
        or not origin - timedelta(minutes=30) <= timestamp(observation_at) <= origin
    ):
        issue = issue.with_columns(pl.lit(None, dtype=pl.Float64).alias("obs__temp_c"))
    for row in issue.select("valid_time", "lead_hours").iter_rows(named=True):
        expected = (row["valid_time"] - origin).total_seconds() / 3600
        if abs(expected - row["lead_hours"]) > 0.001:
            raise ValueError("lead hours must agree with issuance and valid time")
    if prospective and now >= cast(datetime, issue["valid_time"].min()) + timedelta(
        hours=1
    ):
        raise ValueError("refusing forecasts recorded after outcomes")
    values, fitted = predictions(
        pl.read_parquet(capsule / "training.parquet"),
        issue,
        pl.read_parquet(capsule / "truth_hourly.parquet"),
        origin,
    )
    rows = []
    for index, valid in enumerate(issue["valid_time"].to_list()):
        official = production[valid.isoformat()]
        lead = issue["lead_hours"][index]
        rows.append(
            dict(
                issued_at=origin.isoformat(),
                valid_time=valid.isoformat(),
                lead_hours=lead,
                horizon=HORIZONS[min(int(np.digitize(lead, [6, 24, 72])), 3)],
                production=official["values"].get("temp_c"),
                production_method=official["methods"].get("temp_c"),
                production_release=official.get("release_ids", {}).get("temp_c"),
                production_reason=official.get("selection_reasons", {}).get("temp_c"),
                effective_sources=[
                    s
                    for s, v in zip(
                        fitted["sources"],
                        to_forecast_matrix(
                            issue.slice(index, 1),
                            hourly_variable("temp_c"),
                            sources=tuple(fitted["sources"]),
                        ).values[0],
                        strict=True,
                    )
                    if np.isfinite(v)
                ],
                **{name: number(v[index]) for name, v in values.items()},
            )
        )
    # Exclusive directory creation makes an issue immutable, including after partial failure.
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = output / origin.strftime("%Y%m%dT%H%M%SZ")
    target.mkdir(exist_ok=False, mode=0o700)
    hashes = {p.name: digest(p) for p in capsule.iterdir() if p.is_file()}
    payload = {
        "mode": mode,
        "recorded_at": now.isoformat(),
        "issued_at": origin.isoformat(),
        "protocol_sha256": digest(protocol_path),
        "code_identity": code_identity(),
        "runner_sha256": digest(Path(__file__)),
        "candidate_sha256": digest(Path(__file__).with_name("shadow_candidate_v1.py")),
        "capsule": meta,
        "input_sha256": hashes,
        "fitted": fitted,
        "predictions": rows,
    }
    write_new(target / "record.json", payload)
    return target


def interval(
    daily: pl.DataFrame, column: str, *, block: int, protocol: dict[str, Any]
) -> list[float]:
    dates = daily["day"].to_list()
    n = (dates[-1] - dates[0]).days + 1
    values = np.full(n, np.nan)
    for day, value in zip(dates, daily[column].to_list(), strict=True):
        values[(day - dates[0]).days] = value
    rule = protocol["uncertainty"]
    rng = np.random.default_rng(rule["seed"])
    starts = rng.integers(0, n, size=(rule["bootstrap_samples"], math.ceil(n / block)))
    indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(len(starts), -1)[
        :, :n
    ]
    samples = values[indices]
    count = np.isfinite(samples).sum(axis=1)
    means = np.divide(
        np.nansum(samples, axis=1),
        count,
        out=np.full(len(count), np.nan),
        where=count > 0,
    )
    tail = rule["two_sided_family_alpha"] / (2 * rule["family_comparisons"])
    return np.nanquantile(means, [tail, 1 - tail]).tolist()


def quality_pass(
    daily: pl.DataFrame,
    common: pl.DataFrame,
    eligible: pl.DataFrame,
    issue_coverage: float,
    protocol: dict[str, Any],
) -> bool:
    q = protocol["quality"]
    return (
        issue_coverage >= q["minimum_common_coverage"]
        and daily.height >= q["minimum_matched_utc_dates_per_horizon"]
        and common["valid_time"].n_unique()
        >= q["minimum_matched_valid_hours_per_horizon"]
        and common.height / max(eligible.height, 1) >= q["minimum_common_coverage"]
        and daily["day"].dt.week().n_unique() >= q["minimum_weeks_with_data"]
    )


def score_horizon(
    frame: pl.DataFrame,
    horizon: str,
    mode: str,
    issue_coverage: float,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    eligible = frame.filter(pl.col("horizon") == horizon)
    availability = cohort_counts(eligible)
    common = eligible.filter(
        pl.all_horizontal([pl.col(c).is_finite() for c in (*METHODS, "truth")])
    )
    if common.is_empty():
        return {
            "horizon": horizon,
            "eligible_rows": eligible.height,
            "availability": availability,
            "decision": "inconclusive: no common cases",
        }
    losses = common.with_columns(
        *[(pl.col(c) - pl.col("truth")).abs().alias(c) for c in METHODS]
    )
    hourly = losses.group_by("valid_time").agg(*[pl.col(c).mean() for c in METHODS])
    signed = (
        common.with_columns(*[(pl.col(c) - pl.col("truth")).alias(c) for c in METHODS])
        .group_by("valid_time")
        .agg(*[pl.col(c).mean() for c in METHODS])
    )
    daily = (
        hourly.with_columns(pl.col("valid_time").dt.date().alias("day"))
        .group_by("day")
        .agg(*[pl.col(c).mean() for c in METHODS])
        .sort("day")
    )
    paired = daily.with_columns(
        *[
            (pl.col("candidate") - pl.col(c)).alias(c + "_diff")
            for c in ("production", "decay21", "best_provider")
        ]
    )
    intervals = {
        f"{c}_{block}d": interval(paired, c + "_diff", block=block, protocol=protocol)
        for c in ("production", "decay21", "best_provider")
        for block in (3, 7)
    }
    quality = quality_pass(daily, common, eligible, issue_coverage, protocol)
    prod = float(cast(float, daily["production"].mean()))
    cand = float(cast(float, daily["candidate"].mean()))
    improved = prod - cand >= max(0.1, 0.05 * prod)
    passed = all(
        v[1] < 0.0 if k.startswith("production") else v[1] <= 0.1
        for k, v in intervals.items()
    )
    completed = datetime.now(UTC) >= timestamp(protocol["window_end"]) + timedelta(
        hours=98
    )
    decision = (
        "review candidate"
        if quality and improved and passed and completed and mode == "prospective"
        else "retain production / inconclusive"
    )
    return {
        "horizon": horizon,
        "mode": mode,
        "eligible_rows": eligible.height,
        "availability": availability,
        "common_rows": common.height,
        "unique_hours": common["valid_time"].n_unique(),
        "utc_dates": daily.height,
        "quality_pass": quality,
        "window_complete": completed,
        "daily_mae": {c: daily[c].mean() for c in METHODS},
        "hourly_mae": {c: hourly[c].mean() for c in METHODS},
        "bias": {c: signed[c].mean() for c in METHODS},
        "intervals": intervals,
        "decision": decision,
    }


def cohort_counts(eligible: pl.DataFrame) -> dict[str, Any]:
    realized = eligible.filter(pl.col("truth").is_finite())
    return {
        "finite_truth_rows": realized.height,
        "per_method": {
            c: realized.filter(pl.col(c).is_finite()).height for c in METHODS
        },
        "paired_with_candidate": {
            c: realized.filter(
                pl.col(c).is_finite() & pl.col("candidate").is_finite()
            ).height
            for c in METHODS
        },
    }


def evaluate(
    store: Path, truth_path: Path, protocol_path: Path, output: Path, *, mode: str
) -> None:
    protocol = json.loads(protocol_path.read_text())
    records = [json.loads(p.read_text()) for p in sorted(store.glob("*/record.json"))]
    records = [
        r
        for r in records
        if r["mode"] == mode and r["protocol_sha256"] == digest(protocol_path)
    ]
    if not records:
        raise ValueError(
            "no matching records; replay is excluded from prospective analysis"
        )
    implementations = {
        (r["code_identity"], r["runner_sha256"], r["candidate_sha256"]) for r in records
    }
    if len(implementations) != 1:
        raise ValueError("cannot pool changed candidate implementations")
    production_codes = {
        r["capsule"].get("provenance", {}).get("production_code_identity")
        for r in records
    }
    if mode == "prospective" and len(production_codes) != 1:
        raise ValueError(
            "production implementation changed; a new reviewed enrollment is required"
        )
    expected_issues = (
        timestamp(protocol["window_end"]) - timestamp(protocol["window_start"])
    ).total_seconds() / 3600
    issue_coverage = len({r["issued_at"] for r in records}) / expected_issues
    frame = pl.DataFrame(
        [row for r in records for row in r["predictions"]], infer_schema_length=None
    ).with_columns(
        pl.col("valid_time").str.to_datetime(time_zone="UTC"),
        pl.col("issued_at").str.to_datetime(time_zone="UTC"),
    )
    truth = pl.read_parquet(truth_path).select(
        pl.col("valid_hour").alias("valid_time"), pl.col(TARGET).alias("truth")
    )
    if truth["valid_time"].n_unique() != truth.height:
        raise ValueError("truth must contain one QC label per valid hour")
    frame = frame.join(truth, on="valid_time", how="left").filter(
        pl.col("valid_time") + timedelta(hours=2) <= datetime.now(UTC)
    )
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    frame.write_parquet(output / "matched_rows.parquet")
    result = [
        score_horizon(frame, horizon, mode, issue_coverage, protocol)
        for horizon in HORIZONS
    ]
    write_new(
        output / "evaluation.json",
        {
            "protocol_sha256": digest(protocol_path),
            "truth_sha256": digest(truth_path),
            "evaluated_at": datetime.now(UTC),
            "issuance_coverage": issue_coverage,
            "results": result,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("action", choices=("freeze", "record", "finalize", "evaluate"))
    parser.add_argument(
        "--input",
        type=Path,
        required=False,
        help="capsule directory, or recorded-issue store for evaluate",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--mode", choices=("replay", "prospective"), default="replay")
    parser.add_argument(
        "--truth", type=Path, help="frozen QC hourly truth for evaluate"
    )
    parser.add_argument("--remote", help="public forecast Git remote for finalize")
    parser.add_argument("--branch", default="refs/heads/data")
    parser.add_argument(
        "--source", type=Path, help="exact source chosen by the publisher"
    )
    args = parser.parse_args()
    if args.action == "freeze":
        print(freeze(args.protocol, args.out))
    elif args.input is None:
        parser.error("this action requires --input")
    elif args.action == "record":
        print(record(args.input, args.out, args.protocol, mode=args.mode))
    elif args.action == "finalize":
        if not args.remote:
            parser.error("finalize requires --remote")
        print(
            finalize(
                args.input,
                args.out,
                args.protocol,
                remote=args.remote,
                branch=args.branch,
                source=args.source,
                mode=args.mode,
            )
        )
    elif args.truth is None:
        parser.error("evaluate requires --truth")
    else:
        evaluate(args.input, args.truth, args.protocol, args.out, mode=args.mode)


if __name__ == "__main__":
    main()
