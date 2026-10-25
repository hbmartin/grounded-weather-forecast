"""Full pruning verification audits, with bounded journal samples and safe expiry."""

import json
import os
import re
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import polars as pl

from grounded_weather_forecast.clocks import aware_utc
from grounded_weather_forecast.config import Config
from grounded_weather_forecast.retention_status import (
    VERIFICATION_SAMPLE_LIMIT,
    RetentionVerification,
)
from grounded_weather_forecast.runs import read_runs, runs_path
from grounded_weather_forecast.storage import temporary_sibling

AUDIT_RETENTION_DAYS = 90
_AUDIT_NAME = re.compile(r"[0-9a-f]{32}\.jsonl")


def audit_directory(config: Config) -> Path:
    return config.artifacts_dir / "history" / "pruning_verifications"


def _publish_audit(
    path: Path, verification: RetentionVerification, *, dry_run: bool
) -> None:
    temporary = temporary_sibling(path)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "pruning_verification",
                        "check_id": verification.check_id,
                        "recorded_at": verification.recorded_at.isoformat()
                        if verification.recorded_at
                        else None,
                        "dry_run": dry_run,
                    }
                )
                + "\n"
            )
            for index, detail in enumerate(verification.details):
                source = (
                    verification.paths[index]
                    if index < len(verification.paths)
                    else None
                )
                # ApprovalIssue diagnostics already include their typed category.
                explanation = detail.removeprefix(f"{source}: ") if source else detail
                stream.write(
                    json.dumps(
                        {
                            "kind": "issue",
                            "path": source,
                            "category": verification.categories[index]
                            if index < len(verification.categories)
                            else "verification_invalid",
                            "detail": explanation,
                        }
                    )
                    + "\n"
                )
            stream.write(
                json.dumps(
                    {"kind": "complete", "issue_count": len(verification.details)}
                )
                + "\n"
            )
        # Exclusive publication: even an ID collision must never rewrite an audit.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_verification(
    config: Config, verification: RetentionVerification, *, dry_run: bool
) -> RetentionVerification:
    identified = replace(
        verification, check_id=uuid4().hex, issue_count=len(verification.details)
    )
    audit_path = None
    error = None
    if identified.total:
        path = audit_directory(config) / f"{identified.check_id}.jsonl"
        try:
            _publish_audit(path, identified, dry_run=dry_run)
            audit_path = str(path.relative_to(config.artifacts_dir))
        except (OSError, ValueError, TypeError) as problem:
            error = f"audit unavailable: {type(problem).__name__}: {problem}"[:400]
    return replace(
        identified,
        details=identified.details[:VERIFICATION_SAMPLE_LIMIT],
        paths=identified.paths[:VERIFICATION_SAMPLE_LIMIT],
        audit_path=audit_path,
        audit_error=error,
        categories=identified.categories[:VERIFICATION_SAMPLE_LIMIT],
    )


def _referenced_audits(config: Config) -> set[str]:
    config.dataset.dir.stat()  # A vanished dataset cannot establish absent references.
    ledger = runs_path(config)
    try:
        ledger.stat()
    except (FileNotFoundError,):  # noqa: B013 -- only genuine absence means no references
        try:
            ledger.lstat()
        except (FileNotFoundError,):  # noqa: B013
            return set()
        raise ValueError("run ledger is a dangling reference") from None
    references = set()
    for value in read_runs(ledger)["failure_details_json"].drop_nulls():
        raw = json.loads(value)
        if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
            raise ValueError("run diagnostic references could not be verified")
        for item in raw:
            if item.get("audit_path") is None:
                continue
            reference = item["audit_path"]
            if not isinstance(reference, str):
                raise ValueError("invalid audit reference")
            expected = Path("history/pruning_verifications") / Path(reference).name
            if Path(reference) != expected or not _AUDIT_NAME.fullmatch(expected.name):
                raise ValueError("invalid audit reference")
            references.add(expected.name)
    return references


def _expired_audit(path: Path, now: datetime) -> bool:
    # Validate the whole stream before deleting a recognized completed audit.
    with path.open(encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if (
            not isinstance(header, dict)
            or header.get("schema_version") != 1
            or header.get("kind") != "pruning_verification"
            or header.get("check_id") != path.stem
        ):
            return False
        moment = aware_utc(header.get("recorded_at"))
        if moment is None or moment >= now - timedelta(days=AUDIT_RETENTION_DAYS):
            return False
        count = 0
        complete = False
        for line in stream:
            item = json.loads(line)
            if not isinstance(item, dict) or complete:
                return False
            if item.get("kind") == "issue":
                count += 1
            elif item.get("kind") == "complete" and item.get("issue_count") == count:
                complete = True
            else:
                return False
        return complete


def cleanup_audits(config: Config, *, now: datetime, dry_run: bool) -> tuple[str, ...]:
    """Only expired, unreferenced, verified audit files may be removed."""
    notes: list[str] = []
    try:
        if aware_utc(now) is None:
            raise ValueError("audit cutoff must be timezone-aware")
        references = _referenced_audits(config)
        directory = audit_directory(config)
        try:
            entries = tuple(directory.iterdir())
        except (FileNotFoundError,):  # noqa: B013
            return ()
        for path in entries:
            if (
                path.name in references
                or not _AUDIT_NAME.fullmatch(path.name)
                or path.is_symlink()
            ):
                continue
            try:
                if _expired_audit(path, now) and not dry_run:
                    path.unlink()
            except (
                OSError,
                ValueError,
                TypeError,
                OverflowError,
                StopIteration,
            ) as error:
                if len(notes) < VERIFICATION_SAMPLE_LIMIT:
                    notes.append(
                        f"audit cleanup kept {path.name}: {type(error).__name__}"[:400]
                    )
    except (
        OSError,
        ValueError,
        TypeError,
        OverflowError,
        pl.exceptions.PolarsError,
    ) as error:
        return (f"audit cleanup not completed: {type(error).__name__}: {error}"[:400],)
    return tuple(notes)
