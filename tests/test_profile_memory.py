"""Exercise the unattended profiler against read-only forecast CLI commands."""

import subprocess
import sys
from pathlib import Path

import pytest

from grounded_weather_forecast import __version__

memray = pytest.importorskip("memray", reason="requires the profiling group")
PROFILE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/profile_memory.py"


def _profile(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - a fixed interpreter and local script
        [sys.executable, str(PROFILE_SCRIPT), "--no-native", *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )


def test_profile_records_allocations_and_generates_report(tmp_path):
    result = _profile(tmp_path, "--trace-python-allocators", "--", "--version")

    assert result.returncode == 0, result.stderr
    assert __version__ in result.stdout
    assert "Memory capture:" not in result.stdout
    assert "Memory report:" in result.stderr
    captures = list((tmp_path / "artifacts/memory").glob("*/capture.bin"))
    assert len(captures) == 1
    with memray.FileReader(str(captures[0])) as reader:
        assert (
            sum(
                record.size for record in reader.get_high_watermark_allocation_records()
            )
            > 0
        )
    assert "<html" in captures[0].with_name("peak.html").read_text(encoding="utf-8")


def test_profile_preserves_failed_command_exit_and_keeps_capture(tmp_path):
    result = _profile(tmp_path, "--no-report", "--", "unknown-command")

    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert list((tmp_path / "artifacts/memory").glob("*/capture.bin"))
    assert not list((tmp_path / "artifacts/memory").glob("*/peak.html"))


def test_repeated_runs_create_unique_directories(tmp_path):
    for _ in range(2):
        result = _profile(
            tmp_path, "--output-dir", "captures", "--no-report", "--", "--version"
        )
        assert result.returncode == 0, result.stderr

    assert len(list((tmp_path / "captures").glob("*/capture.bin"))) == 2


def test_missing_command_exits_before_creating_outputs(tmp_path):
    result = _profile(tmp_path)

    assert result.returncode == 2
    assert "provide forecast CLI arguments" in result.stderr
    assert not (tmp_path / "artifacts").exists()
