"""Identity and persistence for evaluation evidence and promoted releases."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from omni_forecast import __version__
from omni_forecast.config import Config
from omni_forecast.contracts import TruthSemantics


def dataset_fingerprint(config: Config) -> str:
    """Fingerprint of the materialized dataset currently in use."""
    manifest = config.dataset.dir / "manifest.json"
    if not manifest.exists():
        return "unknown"
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    return str(raw.get("fingerprint", "unknown"))


def config_fingerprint(config: Config) -> str:
    """Stable identity for every setting that influences an evaluation."""
    return hashlib.sha256(repr(config).encode()).hexdigest()[:16]


def _identity(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    """The complete context under which a scores frame was produced."""

    evaluation_id: str
    created_at: str
    dataset_fingerprint: str
    source_kind: str
    source_set: tuple[str, ...]
    product: str
    window: str
    semantics: dict[str, str]
    methods: tuple[str, ...]
    code_version: str
    config_fingerprint: str

    @classmethod
    def create(
        cls,
        config: Config,
        *,
        source_kind: str,
        source_set: tuple[str, ...],
        product: str,
        window: str,
        semantics: dict[str, TruthSemantics],
        methods: tuple[str, ...],
    ) -> "EvaluationRun":
        stable = {
            "dataset_fingerprint": dataset_fingerprint(config),
            "source_kind": source_kind,
            "source_set": list(source_set),
            "product": product,
            "window": window,
            "semantics": {key: value.value for key, value in sorted(semantics.items())},
            "methods": list(methods),
            "code_version": __version__,
            "config_fingerprint": config_fingerprint(config),
        }
        created_at = datetime.now(tz=UTC).isoformat()
        return cls(
            evaluation_id=_identity(stable | {"created_at": created_at}),
            created_at=created_at,
            dataset_fingerprint=str(stable["dataset_fingerprint"]),
            source_kind=source_kind,
            source_set=source_set,
            product=product,
            window=window,
            semantics={key: value.value for key, value in semantics.items()},
            methods=methods,
            code_version=__version__,
            config_fingerprint=config_fingerprint(config),
        )


@dataclass(frozen=True, slots=True)
class ModelRelease:
    """Promoted serving choices tied to compatible evaluation evidence."""

    release_id: str
    promoted_at: str
    dataset_fingerprint: str
    config_fingerprint: str
    evaluation_ids: tuple[str, ...]
    evaluation_contexts: tuple[dict[str, object], ...]
    training_cutoff: str | None
    selections: dict[str, dict[str, object]]

    @classmethod
    def create(
        cls,
        *,
        dataset: str,
        configuration: str,
        evaluation_ids: tuple[str, ...],
        evaluation_contexts: tuple[dict[str, object], ...],
        training_cutoff: datetime | None,
        selections: dict[str, dict[str, object]],
        promoted_at: datetime | None = None,
    ) -> "ModelRelease":
        promoted = promoted_at or datetime.now(tz=UTC)
        stable = {
            "dataset_fingerprint": dataset,
            "config_fingerprint": configuration,
            "evaluation_ids": list(evaluation_ids),
            "evaluation_contexts": evaluation_contexts,
            "training_cutoff": training_cutoff.isoformat() if training_cutoff else None,
            "selections": selections,
        }
        return cls(
            release_id=_identity(stable),
            promoted_at=promoted.isoformat(),
            dataset_fingerprint=dataset,
            config_fingerprint=configuration,
            evaluation_ids=evaluation_ids,
            evaluation_contexts=evaluation_contexts,
            training_cutoff=stable["training_cutoff"],
            selections=selections,
        )

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.release_id}.json"
        if path.exists():
            return path
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path
