"""Pure verification issues and allowlisted public explanations.

Raw details are evidence, never a source of public prose. Codes are interpreted
at rendering time so unrecognized legacy/additive metadata cannot expose paths.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_REASONS = {
    "permission_denied": "Access denied; check filesystem permissions.",
    "not_found": "Referenced artifact is missing.",
    "invalid_json": "Artifact contains invalid JSON.",
    "audit_lock_contended": "Audit directory is busy; no audit was published for this check.",
    "audit_sync_failed": "Audit synchronization failed; publication is unconfirmed.",
    "audit_link_failed": "Exclusive audit publication failed; check hard-link support and permissions.",
    "audit_cleanup_failed": "Audit staging or lock cleanup failed; publication is unconfirmed.",
    "audit_io_failed": "Audit I/O failed; inspect the stored diagnostic.",
    "audit_incomplete": "Audit publication did not complete.",
    "audit_directory_busy": "Audit directory is busy.",
    "release_missing": "Referenced release file is missing.",
    "release_unapproved": "Referenced release has no eligible approval.",
    "release_evidence_invalid": "Referenced release contains invalid evaluation evidence.",
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
    "audit_cleanup_file_failed": "Audit cleanup kept this file because verification or deletion failed.",
    "audit_cleanup_aborted": "Audit cleanup was not completed; uncertain files were preserved.",
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


def public_explanation(
    *,
    category: str | None = None,
    reason_code: str | None = None,
    cause_code: str | None = None,
    audit: bool = False,
) -> str:
    """Render structured codes only; raw prose is never an input."""
    explanation = _CATEGORIES.get(category or "")
    reason = _REASONS.get(reason_code or "")
    cause = _REASONS.get(cause_code or "")
    if audit and reason:
        return " ".join(
            value for value in (f"Audit publication: {reason}", cause) if value
        )
    if explanation or reason:
        return " ".join(value for value in (explanation, reason) if value)
    return (
        "Audit publication could not be verified; see the stored diagnostic."
        if audit
        else "Protection could not be verified; see the stored diagnostic."
    )


def public_detail(
    detail: str,  # noqa: ARG001 -- compatibility only, never public prose
    path: str | None = None,  # noqa: ARG001 -- paths are rendered separately
    *,
    category: str | None = None,
    reason_code: str | None = None,
    audit: bool = False,
) -> str:
    """Compatibility adapter for callers carrying raw details separately."""
    return public_explanation(category=category, reason_code=reason_code, audit=audit)


def category_code(value: object) -> str:
    return value if isinstance(value, str) and value else "verification_invalid"


def optional_code(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


class IssueSource(Protocol):
    @property
    def path(self) -> Path | None: ...

    @property
    def category(self) -> str: ...

    @property
    def detail(self) -> str: ...

    @property
    def reason_code(self) -> str | None: ...


@dataclass(frozen=True, slots=True)
class VerificationIssue:
    path: Path | None
    category: str
    detail: str
    reason_code: str | None = None

    @classmethod
    def from_issue(cls, issue: IssueSource) -> "VerificationIssue":
        if isinstance(issue, cls):
            return issue
        return cls(issue.path, issue.category, issue.detail, issue.reason_code)

    @classmethod
    def from_legacy(
        cls,
        detail: str,
        *,
        path: object,
        category: object,
        reason_code: object,
        normalize_prefix: bool = True,
    ) -> "VerificationIssue":
        source = path if isinstance(path, str) and path else None
        kind = category_code(category)
        return cls(
            Path(source) if source else None,
            kind,
            normalize_detail(detail, source, optional_code(category))
            if normalize_prefix
            else detail,
            optional_code(reason_code),
        )

    def diagnostic(self) -> str:
        prefix = f"{self.path}: " if self.path is not None else ""
        return f"{prefix}{self.category}: {self.detail}"
