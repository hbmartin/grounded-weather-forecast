"""Fitted-model artifact store.

Layout: ``{root}/{fingerprint}/{method_id}/{product}.{variable}/`` holding a
JSON ``state.json`` (human-inspectable; LightGBM boosters travel as model
strings inside it) and a ``manifest.json`` with provenance. A top-level
``latest.json`` maps ``product.variable`` to the artifact most recently saved
for it, so ``predict`` can refuse fingerprint mismatches explicitly.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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
    ) -> Path:
        slot = self._slot(fingerprint, method_id, product, variable)
        if overlap := _RESERVED_MANIFEST_KEYS.intersection(meta or {}):
            msg = f"artifact metadata may not override reserved keys: {sorted(overlap)}"
            raise ArtifactError(msg)
        slot.mkdir(parents=True, exist_ok=True)
        (slot / "state.json").write_text(json.dumps(state), encoding="utf-8")
        manifest = {
            "method_id": method_id,
            "product": product,
            "variable": variable,
            "dataset_fingerprint": fingerprint,
            "created_at": datetime.now(tz=UTC).isoformat(),
            **(meta or {}),
        }
        (slot / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        self._update_latest(fingerprint, method_id, product, variable)
        return slot

    def load_state(
        self, *, fingerprint: str, method_id: str, product: str, variable: str
    ) -> dict[str, Any]:
        slot = self._slot(fingerprint, method_id, product, variable)
        state_path = slot / "state.json"
        if not state_path.exists():
            msg = f"no artifact at {slot}"
            raise ArtifactError(msg)
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            msg = f"corrupt artifact state at {state_path}"
            raise ArtifactError(msg)
        return loaded

    def load_latest_state(
        self, *, method_id: str, product: str, variable: str
    ) -> tuple[str, dict[str, Any]]:
        """Load the newest state for a slice, independent of dataset identity.

        Stateful online methods validate their own processed history before
        advancing, so a new dataset fingerprint does not by itself force a
        replay. The pointer identity is checked before it is trusted.
        """
        key = f"{product}.{variable}.{method_id}"
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
        return fingerprint, self.load_state(
            fingerprint=fingerprint,
            method_id=method_id,
            product=product,
            variable=variable,
        )

    def _latest_path(self) -> Path:
        return self.root / "latest.json"

    def read_latest(self) -> dict[str, dict[str, str]]:
        path = self._latest_path()
        if not path.exists():
            return {}
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}

    def _update_latest(
        self, fingerprint: str, method_id: str, product: str, variable: str
    ) -> None:
        latest = self.read_latest()
        latest[f"{product}.{variable}.{method_id}"] = {
            "fingerprint": fingerprint,
            "method_id": method_id,
            "product": product,
            "variable": variable,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self._latest_path().write_text(
            json.dumps(latest, indent=2, sort_keys=True), encoding="utf-8"
        )
