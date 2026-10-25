"""Pure verification issues and allowlisted public explanations.

Raw details are evidence, never a source of public prose. Codes are interpreted
at rendering time so unrecognized legacy/additive metadata cannot expose paths.
"""

import json
from dataclasses import dataclass
from pathlib import Path

_REASONS = {
    "permission_denied": "Access denied; check filesystem permissions.",
    "not_found": "Referenced artifact is missing.",
    "invalid_json": "Artifact contains invalid JSON.",
    "audit_lock_contended": "Audit directory is busy; publication was deferred.",
    "audit_sync_failed": "Audit synchronization failed; publication is unconfirmed.",
    "audit_link_failed": "Exclusive audit publication failed; check hard-link support and permissions.",
    "audit_cleanup_failed": "Audit staging or lock cleanup failed; publication is unconfirmed.",
    "audit_io_failed": "Audit I/O failed; inspect the stored diagnostic.",
    "audit_incomplete": "Audit publication did not complete.",
}
_CATEGORIES = {
    "directory_unreadable": "Directory listing could not be verified.",
    "non_object": "Artifact must contain a JSON object.",
    "unreadable": "Artifact could not be read or decoded.",
    "invalid_event": "Approval identity, sequence or clock is invalid.",
    "legacy_identity_missing": "Legacy filename identity is usable for retention only.",
    "release_identity": "Release identity does not match its filename.",
    "publication_clock": "Publication clock is missing or is not a valid timezone-aware timestamp.",
    "legacy_future_clock": "Legacy publication clock is beyond the verification cutoff.",
    "active_reference_invalid": "Active pointer fields or release references could not be verified.",
    "active_document_invalid": "Active served document could not be verified.",
    "archived_document_invalid": "Archived served document could not be verified.",
    "release_reference_invalid": "Release evidence references could not be verified.",
}


def normalize_detail(detail: str, path: str | None, category: str | None = None) -> str:
    """Remove exact recorded legacy prefixes once; never rewrite embedded prose."""
    value = detail.removeprefix(f"{path}: ") if path else detail
    return value.removeprefix(f"{category}: ") if category else value


def reason_for_error(error: object) -> str | None:
    """Use exception types and explicit causes, never parse exception messages."""
    seen: set[int] = set()
    while isinstance(error, BaseException) and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, PermissionError):
            return "permission_denied"
        if isinstance(error, FileNotFoundError):
            return "not_found"
        if isinstance(error, json.JSONDecodeError):
            return "invalid_json"
        error = error.__cause__
    return None


def public_detail(
    detail: str,  # noqa: ARG001 -- retained compatibility argument, never public prose
    path: str | None = None,  # noqa: ARG001 -- paths are rendered separately
    *,
    category: str | None = None,
    reason_code: str | None = None,
    audit: bool = False,
) -> str:
    """Render only fixed templates; detail/path remain compatibility arguments."""
    explanation = _CATEGORIES.get(category or "")
    reason = _REASONS.get(reason_code or "")
    if audit and reason:
        return f"Audit publication: {reason}"
    if explanation or reason:
        return " ".join(value for value in (explanation, reason) if value)
    return (
        "Audit publication could not be verified; see the stored diagnostic."
        if audit
        else "Protection could not be verified; see the stored diagnostic."
    )


@dataclass(frozen=True, slots=True)
class VerificationIssue:
    path: Path | None
    category: str
    detail: str
    reason_code: str | None = None

    def diagnostic(self) -> str:
        prefix = f"{self.path}: " if self.path is not None else ""
        return f"{prefix}{self.category}: {self.detail}"
