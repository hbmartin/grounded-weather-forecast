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
from grounded_weather_forecast.storage import atomic_write_text, locked_path

_RECOVERY_COOLDOWN = timedelta(hours=6)
_RECOVERY_STATE = "auto-restore.json"
_RECOVERY_LOG = "auto-restore.log"


@dataclass(frozen=True, slots=True)
class PublishOutcome:
    """What became externally visible after evaluating one candidate."""

    action: str
    history_rows: int


def _ready_document(path: Path) -> Forecast | None:
    if not path.exists():
        return None
    try:
        document = Forecast.from_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, KeyError):
        return None
    return document if document.status == "ready" else None


def publish_document(
    document: Forecast,
    destination: Path,
    history_path: Path,
) -> PublishOutcome:
    """Publish ready/degraded output while preserving an existing last-good file."""
    with locked_path(destination):
        if document.status == "degraded" and _ready_document(destination) is not None:
            return PublishOutcome(action="held_last_good", history_rows=0)
        atomic_write_text(document.to_json(), destination)
        added = append_history(document, history_path)
    action = "published_ready" if document.status == "ready" else "published_degraded"
    return PublishOutcome(action=action, history_rows=added)


def _recovery_signature(config: Config, document: Forecast) -> str:
    payload = {
        "reason": document.status_reason,
        "dataset": document.dataset_fingerprint,
        "config": config_fingerprint(config),
        "code": code_identity(),
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
    document: Forecast,
    *,
    now: datetime | None = None,
) -> bool:
    """Spawn one detached recovery per unchanged signature every six hours."""
    if document.status != "degraded":
        return False
    moment = now or datetime.now(tz=UTC)
    state_path = config.artifacts_dir / _RECOVERY_STATE
    signature = _recovery_signature(config, document)
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
                    "reason": document.status_reason,
                    "pid": process.pid,
                },
                indent=2,
            ),
            state_path,
        )
    return True
