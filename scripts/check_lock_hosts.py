"""Reject dependency lock URLs outside the project's explicit allowlist."""

import re
from pathlib import Path
from urllib.parse import urlparse

APPROVED_HOSTS = frozenset({"files.pythonhosted.org", "pypi.org"})
URL_PATTERN = re.compile(r'https://[^"]+')


def unapproved_hosts(lockfile: Path) -> set[str]:
    """Return HTTPS hosts found in the lockfile but absent from the allowlist."""
    hosts = {
        urlparse(match.group()).hostname
        for match in URL_PATTERN.finditer(lockfile.read_text(encoding="utf-8"))
    }
    return {host for host in hosts if host and host not in APPROVED_HOSTS}


def main() -> int:
    unexpected = sorted(unapproved_hosts(Path("uv.lock")))
    if unexpected:
        print(f"uv.lock contains unapproved package hosts: {', '.join(unexpected)}")
        return 1
    print("uv.lock package hosts are approved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
