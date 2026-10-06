"""Capture a forecast CLI run with Memray for unattended analysis."""

import argparse
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from tempfile import mkdtemp


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/memory"),
        help="parent directory for unique captures (default: artifacts/memory)",
    )
    parser.add_argument(
        "--trace-python-allocators",
        action="store_true",
        help="also trace individual small Python allocations (higher overhead)",
    )
    parser.add_argument(
        "--no-native",
        action="store_true",
        help="omit native C/C++ stack frames",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="save the capture without generating HTML",
    )
    parser.add_argument(
        "cli_args", nargs=argparse.REMAINDER, help="forecast CLI arguments after --"
    )
    args = parser.parse_args(argv)
    cli_args = args.cli_args[1:] if args.cli_args[:1] == ["--"] else args.cli_args
    if not cli_args:
        parser.error("provide forecast CLI arguments after --, e.g. -- maintain")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ-")
    run_dir = Path(mkdtemp(prefix=timestamp, dir=args.output_dir)).resolve()
    capture = run_dir / "capture.bin"
    memray_command = [sys.executable, "-m", "memray"]
    command = [*memray_command, "run", "--quiet", "--output", str(capture)]
    if not args.no_native:
        command.append("--native")
    if args.trace_python_allocators:
        command.append("--trace-python-allocators")
    command.extend(["-m", "grounded_weather_forecast.cli", *cli_args])
    print(f"Memory capture: {capture}", file=sys.stderr, flush=True)
    result = subprocess.run(command, check=False)  # noqa: S603 - argv, never a shell
    exit_code = result.returncode if result.returncode >= 0 else 128 - result.returncode

    if not args.no_report and capture.is_file():
        report = run_dir / "peak.html"
        report_result = subprocess.run(  # noqa: S603 - argv, never a shell
            [
                *memray_command,
                "flamegraph",
                "--no-web",
                "--output",
                str(report),
                str(capture),
            ],
            stdout=sys.stderr,
            check=False,
        )
        if report_result.returncode:
            print("Memory report generation failed; capture retained.", file=sys.stderr)
            return exit_code or 1
        print(f"Memory report: {report}", file=sys.stderr, flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
