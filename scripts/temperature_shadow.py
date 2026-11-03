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

from filelock import FileLock, Timeout

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
    assert_equivalent,
    capsule_for,
    locked_truth,
    mature,
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
from grounded_weather_forecast.contracts import TruthSemantics, hourly_variable
from grounded_weather_forecast.dataset.matrix import (
    to_forecast_matrix,
    to_supervised_slice,
)
from grounded_weather_forecast.evaluation import code_identity
from grounded_weather_forecast.shadow_artifacts import (
    TARGET,
    EvidenceError,
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
    regular(protocol_path)
    regular(protocol_path.with_suffix(".lock.json"))
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


def capsule_metadata(capsule: Path, *, prospective: bool = False) -> dict[str, Any]:
    return CapsuleSnapshot.load(capsule, prospective=prospective).metadata()


def publication_state(value: object) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    payload, published = publication(value)
    supplied = object_value(value, "publisher receipt")
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
    state = {
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
    return payload, published, state


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
    capsule: Path,
    meta: dict[str, Any],
    enrolled: Enrollment,
    *,
    snapshot: CapsuleSnapshot | None = None,
) -> tuple[dict[str, Any], bytes]:
    snapshot = (
        snapshot
        if snapshot is not None
        else CapsuleSnapshot.load(capsule, prospective=True)
    )
    if "evidence.lock.json" not in snapshot.files:
        raise EvidenceError(
            "missing complete evidence seal: evidence.lock.json; late recovery requires a timely seal"
        )
    payload = snapshot.read("evidence.lock.json")
    seal = snapshot.document("evidence.lock.json", "evidence seal")
    if (
        not current_artifact(seal.get("artifact_version"))
        or seal.get("protocol_sha256") != sha256(enrolled.protocol_bytes)
        or seal.get("enrollment_sha256") != sha256(enrolled.lock_bytes)
        or seal.get("input_sha256") != snapshot.digests(sealed=False)
    ):
        raise EvidenceError("sealed evidence identity differs")
    document, production = production_rows(
        capsule, meta, prospective=True, snapshot=snapshot
    )
    sealed_at = timestamp(
        required_string(seal.get("evidence_sealed_at"), "evidence.evidence_sealed_at")
    )
    if sealed_at > datetime.now(UTC):
        raise EvidenceError("evidence seal clock is in the future")
    prospective_clock(meta, document, enrolled.protocol, sealed_at)
    checked_issue(
        capsule,
        timestamp(document["issued_at"]),
        document,
        production,
        sealed_at,
        prospective=True,
        snapshot=snapshot,
    )
    return seal, payload


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
    capsule: Path,
    meta: dict[str, Any],
    value: object,
    enrolled: Enrollment,
    *,
    snapshot: CapsuleSnapshot,
) -> CapsuleSnapshot:
    payload, published, receipt = publication_state(value)
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
    document, production = production_rows(
        capsule,
        candidate,
        prospective=True,
        published_payload=payload,
        published_digest=receipt["published_sha256"],
        snapshot=snapshot,
    )
    if "evidence.lock.json" in snapshot.files:
        verified_seal(capsule, meta, enrolled, snapshot=snapshot)
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
        return snapshot
    now = datetime.now(UTC)
    prospective_clock(candidate, document, enrolled.protocol, now)
    validated_inputs(snapshot, document, production, now, prospective=True)
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
    return snapshot.with_files({"evidence.lock.json": seal_bytes})


def completed_record(
    path: Path,
    saved: dict[str, Any],
    capsule: Path,
    enrolled: Enrollment,
    *,
    record_bytes: bytes | None = None,
    seal_snapshot: tuple[dict[str, Any], bytes] | None = None,
    snapshot: CapsuleSnapshot | None = None,
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
    snapshot = (
        snapshot
        if snapshot is not None
        else CapsuleSnapshot.load(capsule, prospective=saved["mode"] == "prospective")
    )
    meta = snapshot.metadata()
    seal_hash = None
    if saved["mode"] == "prospective":
        seal_data, _seal_bytes = (
            seal_snapshot
            if seal_snapshot is not None
            else verified_seal(capsule, meta, enrolled, snapshot=snapshot)
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
    lease = capsule / ".sealing"
    if lease.exists() or lease.is_symlink():
        regular(lease)
    lock = FileLock(lease, timeout=0)
    try:
        lock.acquire()
    except (Timeout,) as error:  # noqa: B013 -- coordination boundary
        raise OSError(f"capsule finalization is already running: {capsule}") from error
    try:
        snapshot = CapsuleSnapshot.load(capsule, prospective=mode == "prospective")
        meta = snapshot.metadata()
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
            snapshot = seal_publication(
                capsule, meta, publication_receipt, enrolled, snapshot=snapshot
            )
            meta = snapshot.metadata()
        elif "publication" not in meta:
            raise EvidenceError(
                "missing original successful-push receipt; cannot reconstruct from a branch"
            )
        document, _ = production_rows(
            capsule, meta, prospective=mode == "prospective", snapshot=snapshot
        )
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
            completed_record(
                existing, saved, capsule, enrolled, record_bytes=raw, snapshot=snapshot
            )
            return target
        return record(
            capsule,
            output,
            protocol_path,
            mode=mode,
            enrolled=enrolled,
            snapshot=snapshot,
        )
    finally:
        lock.release()


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
    capsule: Path,
    meta: dict[str, Any],
    *,
    prospective: bool,
    published_payload: bytes | None = None,
    published_digest: str | None = None,
    snapshot: CapsuleSnapshot | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    snapshot = snapshot if snapshot is not None else CapsuleSnapshot.load(capsule)
    document = snapshot.document("production.json", "source forecast")
    rows = forecast_rows(document, public=False)
    if not prospective:
        return document, rows
    if "publication" not in meta or (
        published_payload is None and "published.json" not in snapshot.files
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
    object_value(receipt.get("publisher_receipt"), "sealed publisher receipt")
    if (
        not current_artifact(receipt.get("artifact_version"))
        or receipt.get("commit_binding") != BINDING
    ):
        raise EvidenceError("prospective requires a v3 exact-push publication receipt")
    payload = (
        snapshot.read("published.json")
        if published_payload is None
        else published_payload
    )
    published = json_object(payload, "published forecast")
    payload_digest = (
        snapshot.hashes["published.json"]
        if published_payload is None
        else published_digest or sha256(payload)
    )
    forecast_rows(published, public=True)
    published_times = [row["valid_time"] for row in published["hourly"]]
    if (
        len(published_times) != len(rows)
        or len(set(published_times)) != len(published_times)
        or set(published_times) != set(rows)
    ):
        raise EvidenceError("published timestamp grid differs")
    hex_identity(receipt.get("commit"), "sealed publication.commit", 40)
    if (
        receipt["published_sha256"] != payload_digest
        or receipt["source_sha256"] != snapshot.hashes["production.json"]
        or receipt["issued_at"] != document["issued_at"]
        or published["issued_at"] != document["issued_at"]
        or published["dataset_fingerprint"] != document["dataset_fingerprint"]
        or set(published["release_ids"]) != set(document["release_ids"])
        or published.get("units", {}).get("temp_f") != "F"
        or published.get("truth_semantics", {}).get("temp_f") != "mean"
        or timestamp(receipt["published_at"]) != timestamp(published["published_at"])
    ):
        raise EvidenceError("publication receipt/source/payload mismatch")
    validate_push_binding(
        receipt,
        snapshot.read("production.json"),
        payload,
        source_digest=snapshot.hashes["production.json"],
        published_digest=payload_digest,
    )
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
    train = known_training(train, origin)
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


def validate_push_binding(
    receipt: dict[str, Any],
    source: Path | bytes,
    published: Path | bytes,
    *,
    source_digest: str | None = None,
    published_digest: str | None = None,
) -> None:
    public_bytes = published.read_bytes() if isinstance(published, Path) else published
    payload, _ = publication(
        receipt["publisher_receipt"],
        payload_snapshot=(public_bytes, published_digest)
        if published_digest is not None
        else None,
    )
    if (
        payload != public_bytes
        or receipt["publisher_receipt"]["source_sha256"]
        != (
            source_digest
            if source_digest is not None
            else digest(source)
            if isinstance(source, Path)
            else sha256(source)
        )
        or receipt["publisher_receipt"]["commit"] != receipt["commit"]
        or receipt["publisher_receipt"]["remote"] != receipt.get("remote")
        or receipt["publisher_receipt"]["branch"] != receipt.get("branch")
    ):
        raise EvidenceError("publication receipt/source/payload mismatch")


def capsule_frame(
    capsule: Path,
    name: str,
    required: set[str],
    *,
    snapshot: CapsuleSnapshot | None = None,
) -> pl.DataFrame:
    snapshot = snapshot if snapshot is not None else CapsuleSnapshot.load(capsule)
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
    capsule: Path,
    origin: datetime,
    document: dict[str, Any],
    production: dict[str, Any],
    now: datetime,
    *,
    prospective: bool,
    snapshot: CapsuleSnapshot | None = None,
) -> pl.DataFrame:
    issue = capsule_frame(
        capsule,
        "issue.parquet",
        {"issue_time", "valid_time", "lead_hours", "lead_bucket"},
        snapshot=snapshot,
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
    if prospective and now >= cast(datetime, issue["valid_time"].min()) + timedelta(
        hours=1
    ):
        raise EvidenceError("refusing forecasts recorded after outcomes")
    return issue


def validated_inputs(
    snapshot: CapsuleSnapshot,
    document: dict[str, Any],
    production: dict[str, Any],
    now: datetime,
    *,
    prospective: bool,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Prepare all consumed schemas before sealing, without fitting any model."""
    capsule = snapshot.capsule
    origin = timestamp(document["issued_at"])
    issue = checked_issue(
        capsule,
        origin,
        document,
        production,
        now,
        prospective=prospective,
        snapshot=snapshot,
    )
    train = capsule_frame(
        capsule,
        "training.parquet",
        {"issue_time", "valid_time", "lead_hours", "lead_bucket", TARGET},
        snapshot=snapshot,
    )
    truth = capsule_frame(
        capsule, "truth_hourly.parquet", {"valid_hour", TARGET}, snapshot=snapshot
    )
    if "provider_completions.parquet" in snapshot.files:
        snapshot.parquet("provider_completions.parquet")
    try:
        spec = hourly_variable("temp_c")
        supervised = to_supervised_slice(
            known_training(train, origin), spec, semantics=TruthSemantics.INTERVAL_MEAN
        )
        to_forecast_matrix(issue, spec, sources=supervised.x.sources)
    except (ValueError, TypeError, pl.exceptions.PolarsError) as error:
        raise EvidenceError(f"training/issue matrix schema: {error}") from error
    return train, issue, truth


def build_record(
    capsule: Path,
    protocol_path: Path,
    *,
    mode: str,
    now: datetime,
    enrolled: Enrollment | None = None,
    seal_snapshot: tuple[dict[str, Any], bytes] | None = None,
    snapshot: CapsuleSnapshot | None = None,
) -> dict[str, Any]:
    """Pure reconstruction from retained files and the original recording clock."""
    prospective = mode == "prospective"
    snapshot = (
        snapshot
        if snapshot is not None
        else CapsuleSnapshot.load(capsule, prospective=prospective)
    )
    meta = snapshot.metadata()
    enrolled = enrolled if enrolled is not None else enrollment(protocol_path)
    protocol = enrolled.protocol
    if prospective and not current_artifact(protocol.get("artifact_version")):
        raise EvidenceError("prospective requires a fresh v3 enrollment")
    document, production = production_rows(
        capsule, meta, prospective=prospective, snapshot=snapshot
    )
    origin = timestamp(document["issued_at"])
    if timestamp(meta["input_as_of"]) != origin:
        raise EvidenceError("features must have the production issuance time")
    evidence_at, seal_hash = now, None
    if prospective:
        seal, _seal_bytes = (
            seal_snapshot
            if seal_snapshot is not None
            else verified_seal(capsule, meta, enrolled, snapshot=snapshot)
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
    train, issue, truth = validated_inputs(
        snapshot, document, production, evidence_at, prospective=prospective
    )
    values, fitted = predictions(train, issue, truth, origin)
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
    enrolled: Enrollment | None = None,
    snapshot: CapsuleSnapshot | None = None,
) -> Path:
    payload = build_record(
        capsule,
        protocol_path,
        mode=mode,
        now=datetime.now(UTC),
        enrolled=enrolled,
        snapshot=snapshot,
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
    capsule: Path,
    protocol: Path,
    *,
    enrolled: Enrollment | None = None,
    record_bytes: bytes | None = None,
    snapshot: CapsuleSnapshot | None = None,
) -> None:
    enrolled = enrolled if enrolled is not None else enrollment(protocol)
    snapshot = (
        snapshot
        if snapshot is not None
        else CapsuleSnapshot.load(
            capsule, prospective=saved.get("mode") == "prospective"
        )
    )
    seal_snapshot = (
        verified_seal(capsule, snapshot.metadata(), enrolled, snapshot=snapshot)
        if saved.get("mode") == "prospective"
        else None
    )
    completed_record(
        path,
        saved,
        capsule,
        enrolled,
        record_bytes=record_bytes,
        seal_snapshot=seal_snapshot,
        snapshot=snapshot,
    )
    recorded = timestamp(
        required_string(saved.get("recorded_at"), "record.recorded_at")
    )
    expected = build_record(
        capsule,
        protocol,
        mode=saved["mode"],
        now=recorded,
        enrolled=enrolled,
        seal_snapshot=seal_snapshot,
        snapshot=snapshot,
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
                capsule_for(capsules, origin),
                protocol_path,
                enrolled=enrolled,
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
    raw = protocol_path.read_bytes()
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
        print(json.dumps(preflight(args.protocol)))
        return
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
        print(f"shadow refused: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    except (OSError,) as error:  # noqa: B013 -- operational boundary
        print(f"shadow runtime failure: {error}", file=sys.stderr)
        raise SystemExit(1) from None
