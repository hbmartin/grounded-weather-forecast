"""Inactive local temperature shadow: record immutable issues and score later.

Consumes supplied files only. It never calls the production CLI or a provider,
opens SQLite, publishes, promotes, or installs a job. See docs/reference/cli.md.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any, cast

# isort: off
import shadow_environment  # noqa: F401 -- limits must precede numerical imports
# isort: on
# isort: split

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
from shadow_evidence import (
    ARTIFACT_VERSION,
    BINDING,
    INPUT_FILES,  # noqa: F401 -- shared capture contract exposed to offline callers
    CapsuleSnapshot,
    ValidatedPublication,
    assert_equivalent,
    capsule_for,
    locked_truth,
    mature,
    native_guard,
    protocol_document,
    protocol_rules,
    publication,
    regular,
    staged_directory,
    truth_frame,
)
from shadow_evidence import (
    freeze_truth as seal_truth,
)

from grounded_weather_forecast.blenders.baselines import BestProvider
from grounded_weather_forecast.contracts import (
    ForecastMatrix,
    SupervisedSlice,
    TruthSemantics,
    hourly_variable,
)
from grounded_weather_forecast.dataset.matrix import (
    to_forecast_matrix,
    to_supervised_slice,
)
from grounded_weather_forecast.evaluation import code_identity
from grounded_weather_forecast.shadow_artifacts import (
    TARGET,
    EvidenceError,
    capsule_directory,
    current_artifact,
    digest,
    finite_number,
    forecast_rows,
    hex_identity,
    issuance_name,
    json_object,
    object_value,
    required_string,
    runtime_identity,
    sha256,
    sync_files,
    timestamp,
    write_bytes_new,
)

METHODS = (
    "production",
    "candidate",
    "decay21",
    "best_provider",
    "persistence",
    "climatology",
)
HORIZONS = ("0-6h", "6-24h", "24-72h", "72-96h")


def write_new(path: Path, payload: Any) -> bytes:
    raw = json.dumps(payload, indent=2, allow_nan=False, default=str).encode()
    write_bytes_new(path, raw)
    return raw


def number(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def issue_record_target(output: Path, origin: datetime) -> Path:
    target = output / issuance_name(origin)
    if (
        target.is_symlink()
        or target.is_junction()
        or (target.exists() and not target.is_dir())
        or target.resolve().parent != output.resolve()
    ):
        raise EvidenceError(
            "shadow issue directory must remain within the record store"
        )
    return target


@cache
def package_identity() -> str:
    return code_identity()


def implementation_lock(
    protocol_path: Path, protocol_bytes: bytes | None = None
) -> dict[str, Any]:
    return {
        "protocol_sha256": sha256(protocol_bytes)
        if protocol_bytes is not None
        else digest(protocol_path),
        "runner_sha256": digest(Path(__file__)),
        "candidate_sha256": digest(Path(__file__).with_name("shadow_candidate_v1.py")),
        "code_identity": package_identity(),
        "evidence_sha256": digest(Path(__file__).with_name("shadow_evidence.py")),
        "environment_sha256": digest(Path(__file__).with_name("shadow_environment.py")),
        "runtime": runtime_identity(Path(__file__).parents[1] / "uv.lock"),
    }


@dataclass(frozen=True)
class Enrollment:
    protocol_bytes: bytes
    lock_bytes: bytes
    protocol: dict[str, Any]
    lock: dict[str, Any]


def enrollment(
    protocol_path: Path, *, protocol_bytes: bytes | None = None
) -> Enrollment:
    for path in (protocol_path, protocol_path.with_suffix(".lock.json")):
        try:
            regular(path)
        except (FileNotFoundError,) as error:  # noqa: B013 -- missing enrollment evidence
            raise EvidenceError(
                f"{path}: missing required enrollment artifact"
            ) from error
    raw = protocol_path.read_bytes() if protocol_bytes is None else protocol_bytes
    locked = protocol_path.with_suffix(".lock.json").read_bytes()
    protocol, lock = (
        protocol_document(raw),
        json_object(locked, "enrollment lock"),
    )
    if lock != implementation_lock(protocol_path, raw):
        raise EvidenceError(
            "frozen protocol or implementation changed, including numerical runtime"
        )
    return Enrollment(raw, locked, protocol, lock)


def freeze(protocol_path: Path, output: Path) -> Path:
    """Seal a reviewed, future enrollment without scheduling any work."""
    protocol_bytes = protocol_path.read_bytes()
    protocol = protocol_document(protocol_bytes)
    if not current_artifact(protocol.get("artifact_version")):
        raise EvidenceError("freeze requires a fresh v3 protocol")
    template = json_object(
        (
            Path(__file__).parents[1] / "examples/temperature-shadow/protocol.json"
        ).read_bytes(),
        "protocol template",
    )
    fixed = template.keys() - {"window_start", "window_end", "status"}
    if any(protocol.get(key) != template[key] for key in fixed):
        raise EvidenceError(
            "candidate and evaluation rules are fixed; review a new protocol version"
        )
    if protocol["candidate_sha256"] != digest(
        Path(__file__).with_name("shadow_candidate_v1.py")
    ):
        raise EvidenceError("candidate differs from the reviewed protocol digest")
    if timestamp(protocol["window_start"]) <= datetime.now(UTC):
        raise EvidenceError(
            "enrollment window has started; explicit review of a new window is required"
        )
    if timestamp(protocol["window_end"]) - timestamp(
        protocol["window_start"]
    ) != timedelta(days=42):
        raise EvidenceError("the predeclared initial window must be six weeks")
    lock_bytes = json.dumps(
        implementation_lock(protocol_path, protocol_bytes), indent=2, allow_nan=False
    ).encode()
    with staged_directory(output) as folder:
        write_bytes_new(folder / "protocol.json", protocol_bytes)
        write_bytes_new(folder / "protocol.lock.json", lock_bytes)
    return output / "protocol.json"


def freeze_truth(truth: Path, protocol_path: Path, output: Path, now: datetime) -> Path:
    enrolled = enrollment(protocol_path)
    if not current_artifact(enrolled.protocol.get("artifact_version")):
        raise EvidenceError("truth sealing requires a fresh v3 enrollment")
    return seal_truth(
        truth,
        protocol_path,
        output,
        now,
        protocol_bytes=enrolled.protocol_bytes,
        enrollment_bytes=enrolled.lock_bytes,
    )


def publication_state(published: ValidatedPublication) -> dict[str, Any]:
    supplied = published.receipt
    names = (
        "schema_version",
        "public_schema_version",
        "commit_binding",
        "commit",
        "forecast_base64",
        "forecast_sha256",
        "source_sha256",
        "issued_at",
        "published_at",
        "remote",
        "branch",
        "publisher_version",
    )
    receipt = {name: supplied[name] for name in names}
    return {
        "artifact_version": ARTIFACT_VERSION,
        "commit": receipt["commit"],
        "issued_at": receipt["issued_at"],
        "published_at": receipt["published_at"],
        "source_sha256": receipt["source_sha256"],
        "published_sha256": receipt["forecast_sha256"],
        "remote": receipt["remote"],
        "branch": receipt["branch"],
        "commit_binding": BINDING,
        "publisher_receipt": receipt,
    }


def prospective_clock(
    meta: dict[str, Any],
    document: dict[str, Any],
    protocol: dict[str, Any],
    now: datetime,
) -> None:
    origin = timestamp(document["issued_at"])
    if (
        meta.get("capture_kind") != "serving_snapshot"
        or not timestamp(protocol["window_start"])
        <= origin
        < timestamp(protocol["window_end"])
        or not origin <= now <= origin + timedelta(minutes=30)
        or not origin <= timestamp(meta["publication"]["published_at"]) <= now
        or not origin <= timestamp(meta.get("capsule_exported_at", "")) <= now
        or timestamp(meta["input_as_of"]) != origin
        or document.get("timezone") != "America/Los_Angeles"
    ):
        raise EvidenceError(
            "prospective requires a contemporaneous serving snapshot in the frozen window"
        )


def verified_seal(
    snapshot: CapsuleSnapshot,
    enrolled: Enrollment,
) -> dict[str, Any]:
    if not snapshot.prospective:
        raise EvidenceError("prospective seal requires a prospective capsule snapshot")
    meta = snapshot.metadata()
    if "evidence.lock.json" not in snapshot.files:
        raise EvidenceError(
            "missing complete evidence seal: evidence.lock.json; late recovery requires a timely seal"
        )
    seal = snapshot.document("evidence.lock.json", "evidence seal")
    if (
        not current_artifact(seal.get("artifact_version"))
        or seal.get("protocol_sha256") != sha256(enrolled.protocol_bytes)
        or seal.get("enrollment_sha256") != sha256(enrolled.lock_bytes)
        or seal.get("input_sha256") != snapshot.digests(sealed=False)
    ):
        raise EvidenceError("sealed evidence identity differs")
    document, production = production_rows(snapshot)
    sealed_at = timestamp(
        required_string(seal.get("evidence_sealed_at"), "evidence.evidence_sealed_at")
    )
    if sealed_at > datetime.now(UTC):
        raise EvidenceError("evidence seal clock is in the future")
    prospective_clock(meta, document, enrolled.protocol, sealed_at)
    checked_issue(
        snapshot,
        timestamp(document["issued_at"]),
        document,
        production,
        sealed_at,
    )
    return seal


def ensure_bytes(path: Path, payload: bytes, *, snapshot: CapsuleSnapshot) -> None:
    if path.exists() or path.is_symlink():
        regular(path)
        if snapshot.read(path.name) != payload:
            raise EvidenceError(
                f"partial publication conflicts with sealed state: {path.name}"
            )
    else:
        write_bytes_new(path, payload)


def seal_publication(
    snapshot: CapsuleSnapshot,
    value: object,
    enrolled: Enrollment,
) -> tuple[CapsuleSnapshot, PreparedInputs | None]:
    capsule, meta = snapshot.capsule, snapshot.metadata()
    public = publication(value)
    payload, published, receipt = (
        public.payload,
        public.document,
        publication_state(public),
    )
    candidate = {**meta, "publication": receipt}
    if (
        object_value(meta.get("provenance"), "capsule.provenance").get(
            "production_code_identity"
        )
        != enrolled.lock["code_identity"]
    ):
        raise EvidenceError("production implementation differs from frozen enrollment")
    if timestamp(published["issued_at"]) != timestamp(meta["input_as_of"]):
        raise EvidenceError("public commit belongs to a different issuance")
    document, production = production_rows(snapshot, public=public)
    if "evidence.lock.json" in snapshot.files:
        verified_seal(snapshot, enrolled)
        original = meta["publication"]
        if (receipt["remote"], receipt["branch"]) != (
            original["remote"],
            original["branch"],
        ):
            raise EvidenceError("held publication destination differs")
        if (
            not timestamp(original["published_at"])
            <= timestamp(receipt["published_at"])
            <= datetime.now(UTC)
        ):
            raise EvidenceError(
                "held publisher receipt clock precedes original or is in the future"
            )
        return snapshot, None
    now = datetime.now(UTC)
    prospective_clock(candidate, document, enrolled.protocol, now)
    prepared = validated_inputs(snapshot, document, production, now)
    # Serialize before any exclusive write: malformed evidence cannot poison retries.
    receipt_bytes = json.dumps(receipt, indent=2, allow_nan=False).encode()
    ensure_bytes(capsule / "published.json", payload, snapshot=snapshot)
    ensure_bytes(capsule / "publication.json", receipt_bytes, snapshot=snapshot)
    snapshot = snapshot.with_files(
        {"published.json": payload, "publication.json": receipt_bytes},
        known_hashes={"published.json": receipt["published_sha256"]},
    )
    meta = snapshot.metadata()
    hashes = snapshot.digests(sealed=False)
    sync_files(capsule, (capsule / name for name in hashes))
    now = datetime.now(UTC)
    prospective_clock(meta, document, enrolled.protocol, now)
    seal_bytes = write_new(
        capsule / "evidence.lock.json",
        {
            "artifact_version": ARTIFACT_VERSION,
            "evidence_sealed_at": now.isoformat(),
            "protocol_sha256": sha256(enrolled.protocol_bytes),
            "enrollment_sha256": sha256(enrolled.lock_bytes),
            "input_sha256": hashes,
        },
    )
    return snapshot.with_files({"evidence.lock.json": seal_bytes}), prepared


def completed_record(
    path: Path,
    saved: dict[str, Any],
    snapshot: CapsuleSnapshot,
    enrolled: Enrollment,
    *,
    record_bytes: bytes | None = None,
    seal_snapshot: dict[str, Any] | None = None,
) -> None:
    regular(path)
    version = saved.get("artifact_version")
    if not current_artifact(version):
        raise EvidenceError(
            f"{saved.get('mode', 'unknown')} record version {version!r} is incompatible with v3 finalization; preserve it and use its original runner"
        )
    manifest_path = path.with_name("record.lock.json")
    regular(manifest_path)
    manifest = json_object(manifest_path.read_bytes(), "record manifest")
    if snapshot.prospective != (saved.get("mode") == "prospective"):
        raise EvidenceError("record mode differs from capsule snapshot")
    seal_hash = None
    if saved["mode"] == "prospective":
        seal_data = (
            seal_snapshot
            if seal_snapshot is not None
            else verified_seal(snapshot, enrolled)
        )
        if timestamp(
            required_string(saved.get("recorded_at"), "record.recorded_at")
        ) < timestamp(seal_data["evidence_sealed_at"]):
            raise EvidenceError("record clock precedes complete evidence seal")
        seal_hash = snapshot.hashes["evidence.lock.json"]
    if (
        type(manifest.get("artifact_version")) is not int
        or manifest.get("artifact_version") != ARTIFACT_VERSION
        or manifest.get("record_sha256")
        != (sha256(record_bytes) if record_bytes is not None else digest(path))
        or manifest.get("evidence_lock_sha256") != seal_hash
        or saved.get("evidence_lock_sha256") != seal_hash
        or saved.get("input_sha256") != snapshot.digests()
        or saved.get("protocol_sha256") != sha256(enrolled.protocol_bytes)
        or timestamp(required_string(saved.get("recorded_at"), "record.recorded_at"))
        > datetime.now(UTC)
        or path.parent.name
        != issuance_name(
            timestamp(required_string(saved.get("issued_at"), "record.issued_at"))
        )
    ):
        raise EvidenceError("completed shadow record conflicts with sealed evidence")


def finalize(
    capsule: Path,
    output: Path,
    protocol_path: Path,
    *,
    publication_receipt: object = None,
    source: Path | None,
    mode: str,
) -> Path:
    """Seal original evidence before fitting; later held pushes retain that seal."""
    enrolled = enrollment(protocol_path)
    capsule_directory(capsule, prospective=mode == "prospective")
    with native_guard(capsule / ".sealing", "capsule finalization", output=output):
        snapshot = CapsuleSnapshot.load(capsule, prospective=mode == "prospective")
        meta = snapshot.metadata()
        prepared = None
        if source is not None:
            source_hash = (
                snapshot.hashes["production.json"]
                if source.resolve() == (capsule / "production.json").resolve()
                else digest(source)
            )
            if source_hash != snapshot.hashes["production.json"]:
                raise EvidenceError(
                    "publisher chose a different source; do not relabel held forecasts"
                )
        if publication_receipt is not None:
            snapshot, prepared = seal_publication(
                snapshot, publication_receipt, enrolled
            )
        elif "publication" not in meta:
            raise EvidenceError(
                "missing original successful-push receipt; cannot reconstruct from a branch"
            )
        document, _ = production_rows(snapshot)
        target = issue_record_target(output, timestamp(document["issued_at"]))
        existing = target / "record.json"
        if existing.exists() or existing.is_symlink():
            regular(existing)
            raw = existing.read_bytes()
            saved = json_object(raw, "record")
            if saved.get("mode") != mode or not saved.get("predictions"):
                raise EvidenceError(
                    "existing shadow record conflicts with this enrollment"
                )
            completed_record(existing, saved, snapshot, enrolled, record_bytes=raw)
            return target
        return publish_record(snapshot, output, enrolled, prepared=prepared)


def validate_provenance(meta: dict[str, Any], document: dict[str, Any]) -> None:
    provenance = object_value(meta.get("provenance", {}), "capsule.provenance")
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
        raise EvidenceError(
            "prospective requires explicit code/config/data and availability provenance"
        )
    if provenance["dataset_fingerprint"] != document["dataset_fingerprint"]:
        raise EvidenceError("input data provenance differs from production")


def production_rows(
    snapshot: CapsuleSnapshot,
    *,
    public: ValidatedPublication | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    meta = snapshot.metadata()
    if public is not None:
        meta["publication"] = publication_state(public)
    document = snapshot.document("production.json", "source forecast")
    rows = forecast_rows(document, public=False)
    if not snapshot.prospective:
        return document, rows
    if "publication" not in meta or (
        public is None and "published.json" not in snapshot.files
    ):
        raise EvidenceError(
            "prospective requires a committed payload and publication receipt"
        )
    validate_provenance(meta, document)
    receipt = object_value(meta["publication"], "sealed publication receipt")
    for name in ("issued_at", "published_at", "remote", "branch"):
        required_string(receipt.get(name), "sealed publication." + name)
    for name in ("source_sha256", "published_sha256"):
        hex_identity(receipt.get(name), "sealed publication." + name, 64)
    if (
        not current_artifact(receipt.get("artifact_version"))
        or receipt.get("commit_binding") != BINDING
    ):
        raise EvidenceError("prospective requires a v3 exact-push publication receipt")
    if public is None:
        public = publication(
            receipt.get("publisher_receipt"),
            payload_snapshot=(
                snapshot.read("published.json"),
                snapshot.hashes["published.json"],
            ),
        )
    published = public.document
    published_times = [row["valid_time"] for row in published["hourly"]]
    if (
        len(published_times) != len(rows)
        or len(set(published_times)) != len(published_times)
        or set(published_times) != set(rows)
    ):
        raise EvidenceError("published timestamp grid differs")
    hex_identity(receipt.get("commit"), "sealed publication.commit", 40)
    if (
        receipt["published_sha256"] != public.payload_sha256
        or receipt["source_sha256"] != snapshot.hashes["production.json"]
        or receipt["issued_at"] != document["issued_at"]
        or published["issued_at"] != document["issued_at"]
        or published["dataset_fingerprint"] != document["dataset_fingerprint"]
        or set(published["release_ids"]) != set(document["release_ids"])
        or published.get("units", {}).get("temp_f") != "F"
        or published.get("truth_semantics", {}).get("temp_f") != "mean"
        or timestamp(receipt["published_at"]) != timestamp(published["published_at"])
        or public.receipt["source_sha256"] != snapshot.hashes["production.json"]
        or any(
            public.receipt[key] != receipt[key]
            for key in ("commit", "remote", "branch")
        )
    ):
        raise EvidenceError("publication receipt/source/payload mismatch")
    apply_published_temperature(rows, published["hourly"])
    return document, rows


def apply_published_temperature(
    rows: dict[str, dict[str, Any]], published: list[dict[str, Any]]
) -> None:
    """Verify the exact temperature projection of the real public contract."""
    for row in published:
        original = rows[row["valid_time"]]
        for field in ("methods", "release_ids", "selection_reasons"):
            if row.get(field, {}).get("temp_f") != original.get(field, {}).get(
                "temp_c"
            ):
                raise EvidenceError(
                    "published temperature attribution differs from source"
                )
        original_value = original["values"].get("temp_c")
        value_f = row["values"].get("temp_f")
        if (original_value is None) != (value_f is None):
            raise EvidenceError("published temperature availability differs")
        if value_f is not None:
            value_c = (value_f - 32) / 1.8
            # The actual public publisher rounds Celsius * 9/5 +32 to 2 decimals.
            if value_f != round(original_value * 9.0 / 5.0 + 32.0, 2):
                raise EvidenceError("published temperature differs from source")
            # Evaluate the actual public, rounded value.
            original["values"]["temp_c"] = value_c


def predictions(
    train: pl.DataFrame, issue: pl.DataFrame, truth: pl.DataFrame, origin: datetime
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    return prepared_predictions(prepare_matrices(train, issue, truth, origin))


@dataclass(frozen=True, eq=False)
class PreparedInputs:
    train: pl.DataFrame
    issue: pl.DataFrame
    truth: pl.DataFrame
    origin: datetime
    supervised: SupervisedSlice
    forecast: ForecastMatrix


def prepare_matrices(
    train: pl.DataFrame, issue: pl.DataFrame, truth: pl.DataFrame, origin: datetime
) -> PreparedInputs:
    train = known_training(train, origin)
    spec = hourly_variable("temp_c")
    supervised = to_supervised_slice(
        train, spec, semantics=TruthSemantics.INTERVAL_MEAN
    )
    x = to_forecast_matrix(issue, spec, sources=supervised.x.sources)
    return PreparedInputs(train, issue, truth, origin, supervised, x)


def prepared_predictions(
    prepared: PreparedInputs,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    train, issue, truth, origin = (
        prepared.train,
        prepared.issue,
        prepared.truth,
        prepared.origin,
    )
    supervised, x = prepared.supervised, prepared.forecast
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


def known_training(train: pl.DataFrame, origin: datetime) -> pl.DataFrame:
    train = independent_training(
        train.filter(
            (pl.col("issue_time") <= origin)
            & (pl.col("valid_time") + timedelta(hours=2) <= origin)
            & pl.col(TARGET).is_finite()
        )
    )
    if train["valid_time"].n_unique() < 48:
        raise EvidenceError("need at least 48 unique known training hours")
    return train


def capsule_frame(
    snapshot: CapsuleSnapshot,
    name: str,
    required: set[str],
) -> pl.DataFrame:
    frame = snapshot.parquet(name)
    if not required <= set(frame.columns):
        raise EvidenceError(
            f"{name}: missing required columns {sorted(required - set(frame.columns))}"
        )
    for column in required & {"issue_time", "valid_time", "valid_hour"}:
        dtype = frame.schema[column]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None:
            raise EvidenceError(f"{name}.{column}: expected timezone-aware datetime")
        if (
            frame[column].null_count()
            or frame.filter(pl.col(column).dt.epoch("ns") % 1000 != 0).height
        ):
            raise EvidenceError(
                f"{name}.{column}: expected nonnull microsecond-precision clocks"
            )
        frame = frame.with_columns(
            pl.col(column).dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC"))
        )
    for column in frame.columns:
        if (
            column in {"lead_hours", TARGET, "obs__temp_c"} or column.startswith("fx__")
        ) and not frame.schema[column].is_numeric():
            raise EvidenceError(f"{name}.{column}: expected numeric input")
    return frame


def checked_issue(
    snapshot: CapsuleSnapshot,
    origin: datetime,
    document: dict[str, Any],
    production: dict[str, Any],
    now: datetime,
) -> pl.DataFrame:
    issue = capsule_frame(
        snapshot,
        "issue.parquet",
        {"issue_time", "valid_time", "lead_hours", "lead_bucket"},
    ).sort("valid_time")
    if issue["lead_hours"].null_count() or not issue["lead_hours"].is_finite().all():
        raise EvidenceError("issue.lead_hours: expected finite values")
    if issue["issue_time"].n_unique() != 1 or issue["issue_time"][0] != origin:
        raise EvidenceError("issue matrix must contain exactly this issuance")
    issue = issue.filter((pl.col("lead_hours") >= 0) & (pl.col("lead_hours") <= 96))
    if issue.is_empty() or issue["valid_time"].n_unique() != issue.height:
        raise EvidenceError("empty or duplicated forecast grid")
    if {v.isoformat() for v in issue["valid_time"]} != set(production):
        raise EvidenceError("input matrix must cover exactly the published hourly grid")
    observation_at = document.get("observation_at")
    if (
        observation_at is None
        or not origin - timedelta(minutes=30) <= timestamp(observation_at) <= origin
    ):
        issue = issue.with_columns(pl.lit(None, dtype=pl.Float64).alias("obs__temp_c"))
    for row in issue.select("valid_time", "lead_hours").iter_rows(named=True):
        expected = (row["valid_time"] - origin).total_seconds() / 3600
        if abs(expected - row["lead_hours"]) > 0.001:
            raise EvidenceError("lead hours must agree with issuance and valid time")
    if snapshot.prospective and now >= cast(
        datetime, issue["valid_time"].min()
    ) + timedelta(hours=1):
        raise EvidenceError("refusing forecasts recorded after outcomes")
    return issue


def validated_inputs(
    snapshot: CapsuleSnapshot,
    document: dict[str, Any],
    production: dict[str, Any],
    now: datetime,
) -> PreparedInputs:
    """Prepare all consumed schemas before sealing, without fitting any model."""
    origin = timestamp(document["issued_at"])
    issue = checked_issue(
        snapshot,
        origin,
        document,
        production,
        now,
    )
    train = capsule_frame(
        snapshot,
        "training.parquet",
        {
            "issue_time",
            "valid_time",
            "lead_hours",
            "lead_bucket",
            "source_kind",
            TARGET,
        },
    )
    truth = capsule_frame(snapshot, "truth_hourly.parquet", {"valid_hour", TARGET})
    if "provider_completions.parquet" in snapshot.files:
        snapshot.parquet("provider_completions.parquet")
    validate_matrix_schema(train, "training.parquet")
    validate_matrix_schema(issue, "issue.parquet", require_sources=False)
    kinds = train["source_kind"].unique().to_list()
    if len(kinds) != 1 or kinds[0] not in {"live", "synthetic"}:
        raise EvidenceError(
            "training.parquet.source_kind: expected one live or synthetic provenance"
        )
    return prepare_matrices(train, issue, truth, origin)


def validate_matrix_schema(
    frame: pl.DataFrame, label: str, *, require_sources: bool = True
) -> None:
    """Reject known input defects before calling transformations; bugs still propagate."""
    columns = [name for name in frame.columns if name.startswith(("fx__", "fxd__"))]
    if require_sources and not columns:
        raise EvidenceError(f"{label}: matrix has no forecast columns")
    for name in columns:
        parts = name.split("__")
        if len(parts) != 3 or parts[0] != "fx" or not parts[1] or parts[2] != "temp_c":
            raise EvidenceError(f"{label}.{name}: invalid temperature forecast column")


def build_record(
    snapshot: CapsuleSnapshot,
    enrolled: Enrollment,
    *,
    now: datetime,
    seal_snapshot: dict[str, Any] | None = None,
    prepared: PreparedInputs | None = None,
) -> dict[str, Any]:
    """Pure reconstruction from retained files and the original recording clock."""
    prospective = snapshot.prospective
    mode = "prospective" if prospective else "replay"
    meta = snapshot.metadata()
    protocol = enrolled.protocol
    if prospective and not current_artifact(protocol.get("artifact_version")):
        raise EvidenceError("prospective requires a fresh v3 enrollment")
    document, production = production_rows(snapshot)
    origin = timestamp(document["issued_at"])
    if timestamp(meta["input_as_of"]) != origin:
        raise EvidenceError("features must have the production issuance time")
    evidence_at, seal_hash = now, None
    if prospective:
        seal = (
            seal_snapshot
            if seal_snapshot is not None
            else verified_seal(snapshot, enrolled)
        )
        evidence_at, seal_hash = (
            timestamp(
                required_string(
                    seal.get("evidence_sealed_at"), "evidence.evidence_sealed_at"
                )
            ),
            snapshot.hashes["evidence.lock.json"],
        )
        if now < evidence_at:
            raise EvidenceError("record clock precedes complete evidence seal")
    prepared = (
        prepared
        if prepared is not None
        else validated_inputs(snapshot, document, production, evidence_at)
    )
    issue = prepared.issue
    values, fitted = prepared_predictions(prepared)
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
                        prepared.forecast.sources,
                        prepared.forecast.values[index],
                        strict=True,
                    )
                    if np.isfinite(v)
                ],
                **{name: number(v[index]) for name, v in values.items()},
            )
        )
    hashes = snapshot.digests()
    return {
        "artifact_version": ARTIFACT_VERSION,
        "evidence_sha256": enrolled.lock["evidence_sha256"],
        "evidence_lock_sha256": seal_hash,
        "evidence_sealed_at": evidence_at.isoformat() if prospective else None,
        "mode": mode,
        "recorded_at": now.isoformat(),
        "issued_at": origin.isoformat(),
        "protocol_sha256": sha256(enrolled.protocol_bytes),
        "code_identity": enrolled.lock["code_identity"],
        "runner_sha256": enrolled.lock["runner_sha256"],
        "candidate_sha256": enrolled.lock["candidate_sha256"],
        "capsule": meta,
        "input_sha256": hashes,
        "fitted": fitted,
        "predictions": rows,
    }


def record(
    capsule: Path,
    output: Path,
    protocol_path: Path,
    *,
    mode: str,
) -> Path:
    return publish_record(
        CapsuleSnapshot.load(capsule, prospective=mode == "prospective"),
        output,
        enrollment(protocol_path),
    )


def publish_record(
    snapshot: CapsuleSnapshot,
    output: Path,
    enrolled: Enrollment,
    *,
    prepared: PreparedInputs | None = None,
) -> Path:
    payload = build_record(
        snapshot,
        enrolled,
        now=datetime.now(UTC),
        prepared=prepared,
    )
    payload["recorded_at"] = datetime.now(UTC).isoformat()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = issue_record_target(output, timestamp(payload["issued_at"]))
    if target.exists() and not any(target.iterdir()):
        target.rmdir()  # Recover an exclusively named empty pre-publication directory.
    with staged_directory(target) as staged:
        raw = json.dumps(payload, indent=2, allow_nan=False, default=str).encode()
        write_bytes_new(staged / "record.json", raw)
        write_new(
            staged / "record.lock.json",
            {
                "artifact_version": ARTIFACT_VERSION,
                "record_sha256": sha256(raw),
                "evidence_lock_sha256": payload["evidence_lock_sha256"],
            },
        )
    return target


def validate_record(
    path: Path,
    saved: dict[str, Any],
    snapshot: CapsuleSnapshot,
    enrolled: Enrollment,
    *,
    record_bytes: bytes | None = None,
) -> None:
    seal_snapshot = (
        verified_seal(snapshot, enrolled)
        if saved.get("mode") == "prospective"
        else None
    )
    completed_record(
        path,
        saved,
        snapshot,
        enrolled,
        record_bytes=record_bytes,
        seal_snapshot=seal_snapshot,
    )
    recorded = timestamp(
        required_string(saved.get("recorded_at"), "record.recorded_at")
    )
    expected = build_record(
        snapshot,
        enrolled,
        now=recorded,
        seal_snapshot=seal_snapshot,
    )
    expected = json.loads(json.dumps(expected, default=str, allow_nan=False))
    assert_equivalent(saved, expected)


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
        for block in (
            protocol["uncertainty"]["primary_block_days"],
            protocol["uncertainty"]["sensitivity_block_days"],
        )
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


def load_records(store: Path) -> list[tuple[Path, dict[str, Any], bytes]]:
    """Reject child aliases before reading JSON outside the selected store."""
    records = []
    for directory in sorted(store.iterdir()):
        if directory.is_symlink() or directory.is_junction():
            raise EvidenceError("record directory must remain within the record store")
        if not directory.is_dir():
            continue
        path = directory / "record.json"
        if path.exists() or path.is_symlink():
            regular(path)
            raw = path.read_bytes()
            records.append((path, json_object(raw, "record"), raw))
    return records


def record_schema(record: dict[str, Any], label: str) -> None:
    """Check scorer inputs, including legacy replay, before any direct indexing."""
    for name in ("code_identity", "runner_sha256", "candidate_sha256"):
        required_string(record.get(name), f"{label}.{name}")
    origin = timestamp(required_string(record.get("issued_at"), f"{label}.issued_at"))
    rows = record.get("predictions")
    if not isinstance(rows, list) or not rows:
        raise EvidenceError(f"{label}.predictions: expected nonempty object array")
    for index, value in enumerate(rows):
        row_label = f"{label}.predictions[{index}]"
        row = object_value(value, row_label)
        issued = timestamp(
            required_string(row.get("issued_at"), row_label + ".issued_at")
        )
        timestamp(required_string(row.get("valid_time"), row_label + ".valid_time"))
        if issued != origin:
            raise EvidenceError(f"{row_label}.issued_at: differs from record issuance")
        if not finite_number(row.get("lead_hours")):
            raise EvidenceError(f"{row_label}.lead_hours: expected finite number")
        if row.get("horizon") not in HORIZONS:
            raise EvidenceError(f"{row_label}.horizon: unsupported horizon")
        for method in METHODS:
            if method not in row or (
                row[method] is not None and not finite_number(row[method])
            ):
                raise EvidenceError(
                    f"{row_label}.{method}: expected finite number or null"
                )


@dataclass(frozen=True)
class EvaluationInputs:
    protocol_bytes: bytes
    protocol: dict[str, Any]
    selected: list[tuple[Path, dict[str, Any], bytes]]
    truth_bytes: bytes
    lock_bytes: bytes | None


def evaluation_inputs(
    store: Path,
    truth_path: Path | None,
    protocol_path: Path,
    *,
    mode: str,
    capsules: Path | None = None,
    truth_lock: Path | None = None,
) -> EvaluationInputs:
    protocol_bytes = protocol_path.read_bytes()
    protocol = protocol_document(protocol_bytes)
    enrolled = (
        enrollment(protocol_path, protocol_bytes=protocol_bytes)
        if mode == "prospective"
        else None
    )
    selected = [
        (p, r, raw)
        for p, r, raw in load_records(store)
        if r.get("mode") == mode and r.get("protocol_sha256") == sha256(protocol_bytes)
    ]
    records = [r for _p, r, _raw in selected]
    if not records:
        raise EvidenceError(
            "no matching records; replay is excluded from prospective analysis"
        )
    protocol_rules(protocol)
    for path, record, _raw in selected:
        record_schema(record, str(path))
    lock_bytes = None
    if mode == "prospective":
        mature(protocol, datetime.now(UTC))
        if capsules is None or truth_lock is None or enrolled is None:
            raise EvidenceError(
                "prospective evaluation requires --capsules and --truth-lock"
            )
        truth_bytes, lock_bytes = locked_truth(
            truth_lock,
            protocol_path,
            truth_path,
            datetime.now(UTC),
            protocol_bytes=protocol_bytes,
            enrollment_bytes=enrolled.lock_bytes,
        )
        origins = [
            timestamp(required_string(r.get("issued_at"), "record.issued_at"))
            for r in records
        ]
        if len(set(origins)) != len(origins):
            raise EvidenceError("duplicated recorded issuance")
        for (path, saved, raw), origin in zip(selected, origins, strict=True):
            issue_record_target(store, origin)
            validate_record(
                path,
                saved,
                CapsuleSnapshot.load(capsule_for(capsules, origin), prospective=True),
                enrolled,
                record_bytes=raw,
            )
    else:
        if capsules is not None or truth_lock is not None:
            raise EvidenceError(
                "replay evaluation does not accept --capsules or --truth-lock"
            )
        if truth_path is None:
            raise EvidenceError("replay evaluation requires --truth")
        regular(truth_path)
        truth_bytes = truth_path.read_bytes()
    return EvaluationInputs(protocol_bytes, protocol, selected, truth_bytes, lock_bytes)


def evaluate(
    store: Path,
    truth_path: Path | None,
    protocol_path: Path,
    output: Path,
    *,
    mode: str,
    capsules: Path | None = None,
    truth_lock: Path | None = None,
) -> None:
    inputs = evaluation_inputs(
        store,
        truth_path,
        protocol_path,
        mode=mode,
        capsules=capsules,
        truth_lock=truth_lock,
    )
    protocol_bytes, protocol = inputs.protocol_bytes, inputs.protocol
    selected, truth_bytes, lock_bytes = (
        inputs.selected,
        inputs.truth_bytes,
        inputs.lock_bytes,
    )
    records = [record for _path, record, _raw in selected]
    implementations = {
        (r["code_identity"], r["runner_sha256"], r["candidate_sha256"]) for r in records
    }
    if len(implementations) != 1:
        raise EvidenceError("cannot pool changed candidate implementations")
    production_codes = (
        {
            object_value(
                object_value(r.get("capsule"), "record.capsule").get("provenance"),
                "record.capsule.provenance",
            ).get("production_code_identity")
            for r in records
        }
        if mode == "prospective"
        else set()
    )
    if mode == "prospective" and len(production_codes) != 1:
        raise EvidenceError(
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
    truth = truth_frame(truth_bytes, prospective=mode == "prospective").select(
        pl.col("valid_hour").alias("valid_time"), pl.col(TARGET).alias("truth")
    )
    if truth["valid_time"].n_unique() != truth.height:
        raise EvidenceError("truth must contain one QC label per valid hour")
    frame = frame.join(truth, on="valid_time", how="left").filter(
        pl.col("valid_time") + timedelta(hours=2) <= datetime.now(UTC)
    )
    with staged_directory(output) as staged:
        frame.write_parquet(staged / "matched_rows.parquet")
        result = [
            score_horizon(frame, horizon, mode, issue_coverage, protocol)
            for horizon in HORIZONS
        ]
        write_new(
            staged / "evaluation.json",
            {
                "artifact_version": ARTIFACT_VERSION,
                "protocol_sha256": sha256(protocol_bytes),
                "truth_sha256": sha256(truth_bytes),
                "truth_lock_sha256": sha256(lock_bytes)
                if lock_bytes is not None
                else None,
                "record_sha256": {
                    p.parent.name: sha256(raw) for p, _r, raw in selected
                },
                "evaluated_at": datetime.now(UTC),
                "issuance_coverage": issue_coverage,
                "results": result,
            },
        )


def preflight(protocol_path: Path) -> dict[str, str]:
    """Inspect the selected runtime without capture, fitting, or publication."""
    try:
        regular(protocol_path)
        raw = protocol_path.read_bytes()
    except (FileNotFoundError,) as error:  # noqa: B013 -- missing supplied enrollment
        raise EvidenceError(
            f"{protocol_path}: missing required enrollment artifact"
        ) from error
    protocol = protocol_document(raw)
    if (
        not timestamp(protocol["window_start"])
        <= datetime.now(UTC)
        < timestamp(protocol["window_end"])
    ):
        return {"status": "outside_window"}
    from grounded_weather_forecast.serve.predict import predict  # noqa: PLC0415

    if "shadow_capture" not in inspect.signature(predict).parameters:
        raise EvidenceError(
            "selected prediction runtime lacks shadow capture capability"
        )
    enrolled = enrollment(protocol_path, protocol_bytes=raw)
    if not current_artifact(enrolled.protocol.get("artifact_version")):
        raise EvidenceError("fresh v3 enrollment required")
    return {"status": "ready"}


def report_error(error: BaseException, label: str) -> None:
    print(f"{label}: {error}", file=sys.stderr)
    for note in getattr(error, "__notes__", ()):
        print(f"shadow cleanup detail: {note}", file=sys.stderr)


def preflight_cli(protocol: Path) -> int:
    result: dict[str, Any] = {
        "kind": "temperature_shadow_preflight",
        "schema_version": 1,
    }
    try:
        result.update(preflight(protocol))
    except (EvidenceError, OSError) as error:
        code = 2 if isinstance(error, EvidenceError) else 1
        result.update(
            status="refused" if code == 2 else "runtime_failure", detail=str(error)
        )
        report_error(error, "shadow refused" if code == 2 else "shadow runtime failure")
    else:
        code = 0
    print(json.dumps(result))
    return code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "action",
        choices=(
            "freeze",
            "freeze-truth",
            "record",
            "finalize",
            "evaluate",
            "preflight",
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=False,
        help="capsule directory, or recorded-issue store for evaluate",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--mode", choices=("replay", "prospective"), default="replay")
    parser.add_argument(
        "--truth", type=Path, help="frozen QC hourly truth for evaluate"
    )
    parser.add_argument(
        "--publication-receipt", type=Path, help="exact successful-push receipt JSON"
    )
    parser.add_argument(
        "--capsules", type=Path, help="retained capsule root for prospective evaluation"
    )
    parser.add_argument(
        "--truth-lock", type=Path, help="sealed mature QC truth directory"
    )
    parser.add_argument(
        "--source", type=Path, help="exact source chosen by the publisher"
    )
    args = parser.parse_args()
    if args.action == "preflight":
        raise SystemExit(preflight_cli(args.protocol))
    if args.out is None:
        parser.error("this action requires --out")
    if args.action == "freeze":
        print(freeze(args.protocol, args.out))
    elif args.action == "freeze-truth":
        if args.truth is None:
            parser.error("freeze-truth requires --truth")
        print(freeze_truth(args.truth, args.protocol, args.out, datetime.now(UTC)))
    elif args.input is None:
        parser.error("this action requires --input")
    elif args.action == "record":
        print(record(args.input, args.out, args.protocol, mode=args.mode))
    elif args.action == "finalize":
        print(
            finalize(
                args.input,
                args.out,
                args.protocol,
                publication_receipt=(
                    json_object(
                        args.publication_receipt.read_bytes(), "publisher receipt"
                    )
                    if args.publication_receipt
                    else None
                ),
                source=args.source,
                mode=args.mode,
            )
        )
    elif args.truth is None and args.mode == "replay":
        parser.error("evaluate requires --truth")
    else:
        evaluate(
            args.input,
            args.truth,
            args.protocol,
            args.out,
            mode=args.mode,
            capsules=args.capsules,
            truth_lock=args.truth_lock,
        )


if __name__ == "__main__":
    try:
        main()
    except (EvidenceError,) as error:  # noqa: B013 -- domain refusal boundary
        report_error(error, "shadow refused")
        raise SystemExit(2) from None
    except (OSError,) as error:  # noqa: B013 -- operational boundary
        report_error(error, "shadow runtime failure")
        raise SystemExit(1) from None
