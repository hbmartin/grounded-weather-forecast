"""Safe forecast publication and bounded detached recovery scheduling."""

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from grounded_weather_forecast.config import Config
from grounded_weather_forecast.evaluation import code_identity, config_fingerprint
from grounded_weather_forecast.serve.history import append_history
from grounded_weather_forecast.serve.schema import Forecast
from grounded_weather_forecast.storage import (
    atomic_write_text,
    locked_path,
    output_target,
)

_RECOVERY_COOLDOWN = timedelta(hours=6)
_RECOVERY_STATE = "auto-restore.json"
_RECOVERY_LOG = "auto-restore.log"
_PUBLISH_ATTEMPT = "latest_publish_attempt.json"
_MAX_HOLD_AGE = timedelta(hours=6)
_CLOCK_SKEW = timedelta(minutes=5)


@dataclass(frozen=True, slots=True)
class PublishOutcome:
    """What became externally visible after evaluating one candidate."""

    action: str
    history_rows: int
    served_document: Forecast
    history_error: str | None = None


def _ready_document(path: Path, candidate: Forecast, now: datetime) -> Forecast | None:
    if not path.exists():
        return None
    try:
        document = Forecast.from_json(path.read_text(encoding="utf-8"))
        issued = datetime.fromisoformat(document.issued_at)
    except (OSError, ValueError, TypeError, KeyError):
        return None
    if issued.tzinfo is None:
        issued = issued.replace(tzinfo=UTC)
    age = now - issued.astimezone(UTC)
    if not -_CLOCK_SKEW <= age <= _MAX_HOLD_AGE:
        return None
    if (
        document.status != "ready"
        or document.schema_version != candidate.schema_version
        or document.latitude != candidate.latitude
        or document.longitude != candidate.longitude
        or document.timezone != candidate.timezone
    ):
        return None
    return document


def publish_document(
    document: Forecast,
    destination: Path,
    history_path: Path,
    *,
    now: datetime | None = None,
) -> PublishOutcome:
    """Publish ready/degraded output while preserving an existing last-good file."""
    target = output_target(destination)
    moment = now or datetime.now(tz=UTC)
    with locked_path(target):
        held = _ready_document(target, document, moment)
        if document.status == "degraded" and held is not None:
            return PublishOutcome(
                action="held_last_good", history_rows=0, served_document=held
            )
        atomic_write_text(document.to_json(), target)
        try:
            added = append_history(document, history_path)
        except Exception as exc:  # the output is already visible
            action = (
                "published_ready"
                if document.status == "ready"
                else "published_degraded"
            )
            return PublishOutcome(
                action=action,
                history_rows=0,
                served_document=document,
                history_error=f"{type(exc).__name__}: {exc}",
            )
    action = "published_ready" if document.status == "ready" else "published_degraded"
    return PublishOutcome(action=action, history_rows=added, served_document=document)


def _recovery_signature(
    config: Config, reason: str | None, semantics: str, methods: str, window: str
) -> str:
    payload = {
        "reason": reason,
        "config": config_fingerprint(config),
        "code": code_identity(),
        "semantics": semantics,
        "methods": methods,
        "window": window,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _recent_attempt(state: object, signature: str, now: datetime) -> bool:
    if not isinstance(state, dict) or state.get("signature") != signature:
        return False
    mapping = cast("dict[str, object]", state)
    try:
        attempted_at = datetime.fromisoformat(str(mapping["attempted_at"]))
    except (KeyError, ValueError, TypeError):
        return False
    if attempted_at.tzinfo is None:
        attempted_at = attempted_at.replace(tzinfo=UTC)
    return now - attempted_at < _RECOVERY_COOLDOWN


def schedule_recovery(
    config: Config,
    config_path: Path,
    document: Forecast | None,
    *,
    now: datetime | None = None,
    reason: str | None = None,
    semantics: str = "auto",
    methods: str = "all",
    window: str = "expanding",
) -> bool:
    """Spawn one detached recovery per unchanged signature every six hours."""
    if document is not None and document.status != "degraded":
        return False
    cause = reason if document is None else document.status_reason
    if document is None and not cause:
        return False
    moment = now or datetime.now(tz=UTC)
    state_path = config.artifacts_dir / _RECOVERY_STATE
    signature = _recovery_signature(config, cause, semantics, methods, window)
    with locked_path(state_path):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = None
        if _recent_attempt(state, signature, moment):
            return False
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        log_path = config.artifacts_dir / _RECOVERY_LOG
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(  # noqa: S603
                [
                    sys.executable,
                    "-m",
                    "grounded_weather_forecast.cli",
                    "--config",
                    str(config_path),
                    "recover",
                    "--semantics",
                    semantics,
                    "--methods",
                    methods,
                    "--window",
                    window,
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        atomic_write_text(
            json.dumps(
                {
                    "signature": signature,
                    "attempted_at": moment.isoformat(),
                    "reason": cause,
                    "pid": process.pid,
                },
                indent=2,
            ),
            state_path,
        )
    return True


def record_publish_attempt(
    config: Config,
    candidate: Forecast,
    outcome: PublishOutcome,
    destination: Path,
    *,
    now: datetime | None = None,
) -> None:
    """Keep the latest candidate visible even when the last good output is held."""
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        json.dumps(
            {
                "attempted_at": (now or datetime.now(tz=UTC)).isoformat(),
                "candidate_status": candidate.status,
                "status_reason": candidate.status_reason,
                "action": outcome.action,
                "served_issued_at": outcome.served_document.issued_at,
                "served_status": outcome.served_document.status,
                "output_path": str(output_target(destination).absolute()),
            },
            indent=2,
        ),
        config.artifacts_dir / _PUBLISH_ATTEMPT,
    )
