"""Fitted-model artifact store.

Layout: ``{root}/{fingerprint}/{method_id}/{product}.{variable}/`` holding a
JSON ``state.json`` (human-inspectable; LightGBM boosters travel as model
strings inside it) and a ``manifest.json`` with provenance. A top-level
``latest.json`` maps ``product.variable`` to the artifact most recently saved
for it, so ``predict`` can refuse fingerprint mismatches explicitly.
"""

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from grounded_weather_forecast.storage import atomic_write_text, locked_path


class ArtifactError(ValueError):
    """A requested artifact is missing or inconsistent."""


_RESERVED_MANIFEST_KEYS = frozenset(
    {"method_id", "product", "variable", "dataset_fingerprint", "created_at"}
)


@dataclass(frozen=True, slots=True)
class ArtifactStore:
    root: Path

    def _slot(
        self, fingerprint: str, method_id: str, product: str, variable: str
    ) -> Path:
        return self.root / fingerprint / method_id / f"{product}.{variable}"

    def save(
        self,
        *,
        fingerprint: str,
        method_id: str,
        product: str,
        variable: str,
        state: dict[str, Any],
        meta: dict[str, Any] | None = None,
        reclaim_unreferenced: bool = False,
        lock_timeout: float = -1,
    ) -> Path:
        slot = self._slot(fingerprint, method_id, product, variable)
        if overlap := _RESERVED_MANIFEST_KEYS.intersection(meta or {}):
            msg = f"artifact metadata may not override reserved keys: {sorted(overlap)}"
            raise ArtifactError(msg)
        manifest = {
            "method_id": method_id,
            "product": product,
            "variable": variable,
            "dataset_fingerprint": fingerprint,
            "created_at": datetime.now(tz=UTC).isoformat(),
            **(meta or {}),
        }
        # One transaction: the slot must exist before the pointer names it, and
        # the pointer's read-modify-write cannot interleave with another serve.
        # `predict` runs every 10 minutes over several variables, so concurrent
        # saves into one store are the normal case, not an edge case.
        with locked_path(self._latest_path(), timeout=lock_timeout):
            # Validate the pointer map before touching a possibly existing slot.
            # A corrupt pointer must not let a failed save overwrite recoverable
            # state even when the caller reuses the same fingerprint.
            latest = self.read_latest()
            slot.mkdir(parents=True, exist_ok=True)
            atomic_write_text(json.dumps(state), slot / "state.json")
            atomic_write_text(json.dumps(manifest, indent=2), slot / "manifest.json")
            latest = self._update_latest(
                latest, fingerprint, method_id, product, variable
            )
            if reclaim_unreferenced:
                self._reclaim_unreferenced(latest)
        return slot

    def _reclaim_unreferenced(self, latest: Mapping[str, Any]) -> None:
        """Delete fingerprint trees no ``latest.json`` pointer names.

        Takes the pointer map the caller just wrote, and the caller holds the
        lock on it — both matter: a slot is written before the pointer that
        names it, so an unsynchronized pass would race a concurrent ``save``
        and delete the tree it is still writing into. Every state read reaches
        a slot through a pointer (``load_latest_state``,
        ``load_observability_states``), so a tree no pointer names is
        unreachable rather than merely old.
        """
        referenced = {
            entry["fingerprint"]
            for entry in latest.values()
            if isinstance(entry, Mapping) and isinstance(entry.get("fingerprint"), str)
        }
        if not referenced or not self.root.is_dir():
            return
        for child in self.root.iterdir():
            if child.is_dir() and child.name not in referenced:
                shutil.rmtree(child, ignore_errors=True)

    @staticmethod
    def _read_slot_json(path: Path, kind: str) -> dict[str, Any]:
        if not path.exists():
            msg = f"no artifact {kind} at {path.parent}"
            raise ArtifactError(msg)
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            msg = f"corrupt artifact {kind} at {path}"
            raise ArtifactError(msg) from exc
        if not isinstance(loaded, dict):
            msg = f"corrupt artifact {kind} at {path}"
            raise ArtifactError(msg)
        return loaded

    def load_state(
        self, *, fingerprint: str, method_id: str, product: str, variable: str
    ) -> dict[str, Any]:
        slot = self._slot(fingerprint, method_id, product, variable)
        return self._read_slot_json(slot / "state.json", "state")

    def load_manifest(
        self, *, fingerprint: str, method_id: str, product: str, variable: str
    ) -> dict[str, Any]:
        slot = self._slot(fingerprint, method_id, product, variable)
        return self._read_slot_json(slot / "manifest.json", "manifest")

    def load_latest_state(
        self, *, method_id: str, product: str, variable: str
    ) -> tuple[str, dict[str, Any]]:
        """Load the newest state for a slice, independent of dataset identity.

        Stateful online methods validate their own processed history before
        advancing, so a new dataset fingerprint does not by itself force a
        replay. The pointer identity is checked before it is trusted.
        """
        fingerprint, state, _manifest = self.load_latest_bundle(
            method_id=method_id, product=product, variable=variable
        )
        return fingerprint, state

    def load_latest_bundle(
        self,
        *,
        method_id: str,
        product: str,
        variable: str,
        lock_timeout: float = -1,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Load one current state and manifest while reclamation is excluded."""
        key = f"{product}.{variable}.{method_id}"
        with locked_path(self._latest_path(), timeout=lock_timeout):
            pointer = self.read_latest().get(key)
            if not isinstance(pointer, dict):
                msg = f"no latest artifact for {key}"
                raise ArtifactError(msg)
            expected = {
                "method_id": method_id,
                "product": product,
                "variable": variable,
            }
            if any(pointer.get(name) != value for name, value in expected.items()):
                msg = f"inconsistent latest artifact pointer for {key}"
                raise ArtifactError(msg)
            fingerprint = pointer.get("fingerprint")
            if not isinstance(fingerprint, str) or not fingerprint:
                msg = f"latest artifact pointer for {key} has no fingerprint"
                raise ArtifactError(msg)
            identity = {
                "fingerprint": fingerprint,
                "method_id": method_id,
                "product": product,
                "variable": variable,
            }
            return (
                fingerprint,
                self.load_state(**identity),
                self.load_manifest(**identity),
            )

    def _latest_path(self) -> Path:
        return self.root / "latest.json"

    def read_latest(self) -> dict[str, dict[str, str]]:
        """The pointer map, or an ArtifactError if it exists but cannot be read.

        A missing pointer is a cold start and reads as empty. A corrupt or
        unreadable one is not: returning ``{}`` there would let a reclamation
        pass conclude that nothing is referenced and delete every state tree.
        """
        path = self._latest_path()
        if not path.exists():
            return {}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            msg = f"corrupt artifact pointer at {path}"
            raise ArtifactError(msg) from exc
        except OSError as exc:
            msg = f"unreadable artifact pointer at {path}"
            raise ArtifactError(msg) from exc
        if not isinstance(loaded, dict):
            msg = f"corrupt artifact pointer at {path}"
            raise ArtifactError(msg)
        return loaded

    def _update_latest(
        self,
        latest: dict[str, dict[str, str]],
        fingerprint: str,
        method_id: str,
        product: str,
        variable: str,
    ) -> dict[str, dict[str, str]]:
        """Merge one pointer entry, returning the map written.

        Callers hold the lock on ``latest.json``.
        """
        latest[f"{product}.{variable}.{method_id}"] = {
            "fingerprint": fingerprint,
            "method_id": method_id,
            "product": product,
            "variable": variable,
        }
        atomic_write_text(
            json.dumps(latest, indent=2, sort_keys=True), self._latest_path()
        )
        return latest
