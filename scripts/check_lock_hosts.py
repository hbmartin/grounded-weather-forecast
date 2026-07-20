"""Reject dependency lock sources outside the project's explicit allowlist."""

import re
from pathlib import Path
from urllib.parse import urlparse

APPROVED_HOSTS = frozenset({"files.pythonhosted.org", "pypi.org"})
# Any scheme, not just https: a `git+ssh://`, `http://` or `file://` source is
# exactly what an allowlist exists to catch, and matching only https URLs made
# the check blind to every one of them.
URL_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\"'\s]+")
# uv records non-registry dependencies as a source kind rather than a URL, so
# a `{ path = "../evil" }` or `{ workspace = true }` entry carries no scheme at
# all and would otherwise pass silently. Capture the value as well: allowlisting
# the `editable` kind alone would wave through `editable = "../evil"`.
LOCAL_SOURCE_PATTERN = re.compile(
    r"source\s*=\s*\{\s*(?P<kind>[a-z_]+)\s*=\s*(?P<value>\"[^\"]*\"|[^\s},]+)"
)
# The project's own editable install is `editable = "."`; an editable source
# pointing anywhere else escapes the workspace and is what we are looking for.
APPROVED_EDITABLE_PATH = "."


def unapproved_hosts(lockfile: Path) -> set[str]:
    """Hosts, schemes, and non-registry or off-project sources to reject."""
    text = lockfile.read_text(encoding="utf-8")
    findings: set[str] = set()
    for match in URL_PATTERN.finditer(text):
        parsed = urlparse(match.group())
        if parsed.scheme != "https":
            findings.add(f"{parsed.scheme or 'missing'} scheme")
        if parsed.hostname is None:
            findings.add(f"hostless {parsed.scheme or 'URL'} source")
        elif parsed.hostname not in APPROVED_HOSTS:
            findings.add(parsed.hostname)
    for source in LOCAL_SOURCE_PATTERN.finditer(text):
        match (source.group("kind"), source.group("value").strip('"')):
            case ("registry", _):
                pass  # the URL is validated above by URL_PATTERN
            case ("editable", path) if path != APPROVED_EDITABLE_PATH:
                findings.add(f"editable {path}")
            case ("editable", _):
                pass  # this project itself
            case (kind, _):
                findings.add(f"{kind} source")
    return findings


def main() -> int:
    unexpected = sorted(unapproved_hosts(Path("uv.lock")))
    if unexpected:
        print(f"uv.lock contains unapproved package sources: {', '.join(unexpected)}")
        return 1
    print("uv.lock package sources are approved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
