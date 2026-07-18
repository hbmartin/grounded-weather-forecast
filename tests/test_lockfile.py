import re
from pathlib import Path
from urllib.parse import urlparse


def test_lockfile_contains_only_approved_package_hosts():
    text = Path("uv.lock").read_text(encoding="utf-8")
    hosts = {
        urlparse(match.group()).hostname
        for match in re.finditer(r'https://[^"]+', text)
    }
    assert hosts <= {"files.pythonhosted.org", "pypi.org"}
