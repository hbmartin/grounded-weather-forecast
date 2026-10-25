"""Pure normalization and safe public explanations for verification diagnostics."""

import re

_PATH_TOKEN = re.compile(
    r"(?:^|[\s\"'(:=])(?:/(?![/\s])|[A-Za-z]:[\\/]|\\\\|~[/\\])"
    r"|(?:^|[\s\"'(=])//(?=\S)|\bfile://"
)


def normalize_detail(detail: str, path: str | None) -> str:
    """Remove only a recorded legacy path prefix; never rewrite prose."""
    return detail.removeprefix(f"{path}: ") if path else detail


def public_detail(detail: str, path: str | None = None) -> str:
    explanation = normalize_detail(detail, path)
    if _PATH_TOKEN.search(explanation):
        return "Protection detail contains filesystem paths; see the full verification diagnostic."
    return explanation
