"""Workers must measure the requested tree and keep outputs at the chosen path."""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/benchmark_fit_diagnostics.py"


@pytest.fixture
def benchmark():
    spec = importlib.util.spec_from_file_location("fit_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source_tree(tmp_path):
    source = tmp_path / "snapshot/src"
    shutil.copytree(REPO / "src", source, ignore=shutil.ignore_patterns("__pycache__"))
    return source


@pytest.mark.parametrize("workload", ["aggregation", "successful_fit", "pending_fit"])
def test_worker_reports_verified_snapshot_modules(
    benchmark, source_tree, tmp_path, workload
):
    completed = subprocess.run(  # noqa: S603 -- local worker and fixture argv
        [
            sys.executable,
            str(SCRIPT),
            "--worker",
            "--source",
            str(source_tree),
            "--rows",
            "3",
            "--workload",
            workload,
        ],
        cwd=tmp_path,
        env=os.environ | benchmark.THREADS,
        text=True,
        capture_output=True,
        check=True,
    )
    record = json.loads(completed.stdout)
    assert Path(record["source_root"]) == source_tree.resolve()
    assert record["module_origins"]
    assert "grounded_weather_forecast.serve.predict" in record["module_origins"]
    assert all(
        Path(origin).is_relative_to(source_tree.resolve())
        for origin in record["module_origins"].values()
    )


@pytest.mark.parametrize("exists", [False, True])
def test_worker_rejects_missing_or_file_source(tmp_path, exists):
    source = tmp_path / "not-a-source"
    if exists:
        source.write_text("file")
    completed = subprocess.run(  # noqa: S603 -- local worker and fixture argv
        [
            sys.executable,
            str(SCRIPT),
            "--worker",
            "--source",
            str(source),
            "--rows",
            "3",
            "--workload",
            "aggregation",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert not completed.stdout
    assert (
        "must be a directory" if exists else "FileNotFoundError"
    ) in completed.stderr


def test_worker_rejects_cached_checkout_modules(benchmark, source_tree, monkeypatch):
    import grounded_weather_forecast.serve.predict  # noqa: F401, PLC0415 -- cache before worker

    monkeypatch.setattr(sys, "path", sys.path.copy())
    with pytest.raises(RuntimeError, match="wrong source tree"):
        benchmark.worker(
            argparse.Namespace(source=source_tree, rows=3, workload="aggregation")
        )


def test_worker_rejects_unverifiable_module_origin(benchmark, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "grounded_weather_forecast.origin_missing",
        ModuleType("origin_missing"),
    )
    with pytest.raises(RuntimeError, match="cannot verify module origin"):
        benchmark.module_origins((REPO / "src").resolve())


@pytest.mark.parametrize("kind", ["default", "relative", "absolute"])
def test_output_path_contract_from_another_cwd(benchmark, tmp_path, monkeypatch, kind):
    monkeypatch.chdir(tmp_path)
    supplied = (
        None
        if kind == "default"
        else Path("chosen/results.jsonl")
        if kind == "relative"
        else tmp_path / "chosen.jsonl"
    )
    destination = benchmark.output_path(REPO, supplied)
    if supplied is None:
        assert destination.parent == REPO / "reports/benchmarks"
        assert destination.is_absolute()
        stamp, identifier = destination.stem.removeprefix("fit-diagnostics-").split("-")
        assert datetime.strptime(stamp, "%Y%m%dT%H%M%S%fZ").replace(
            tzinfo=UTC
        ) <= datetime.now(UTC)
        assert len(identifier) == 32
        assert destination != benchmark.output_path(REPO, None)
    else:
        assert destination == supplied
        if not supplied.is_absolute():
            assert destination.resolve() == tmp_path / supplied


@pytest.mark.parametrize("artifact", [0, 1])
@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "broken_symlink"])
def test_existing_destination_rejected_before_workers(
    benchmark, tmp_path, monkeypatch, artifact, kind
):
    output = tmp_path / "samples.jsonl"
    collision = benchmark.artifact_paths(output)[artifact]
    if kind == "file":
        collision.write_bytes(b"previous evidence")
    elif kind == "directory":
        collision.mkdir()
    else:
        target = tmp_path / "target"
        if kind == "symlink":
            target.write_bytes(b"previous evidence")
        collision.symlink_to(target)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", str(output)])

    def no_workers(*_args, **_kwargs):
        pytest.fail("preflight collision must stop before git or workers")

    monkeypatch.setattr(benchmark.subprocess, "check_output", no_workers)
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2
    assert collision.exists() or collision.is_symlink()
    if kind in {"file", "symlink"}:
        assert collision.read_bytes() == b"previous evidence"


@pytest.mark.parametrize("output", [".", "/"])
def test_nameless_directory_output_is_an_existing_destination(
    benchmark, tmp_path, monkeypatch, output
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", output])
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2


@pytest.mark.parametrize("artifact", [0, 1])
def test_late_collision_preserves_existing_bytes_and_removes_only_owned_outputs(
    benchmark, tmp_path, artifact
):
    output = tmp_path / "samples.jsonl"
    benchmark.check_output_paths(output)
    collision = benchmark.artifact_paths(output)[artifact]
    collision.write_bytes(b"late evidence")
    with pytest.raises(FileExistsError):
        benchmark.publish_outputs(output, "new samples", "new summary")
    assert collision.read_bytes() == b"late evidence"
    assert not benchmark.artifact_paths(output)[1 - artifact].exists()


def test_failed_publication_preserves_replacement_owned_by_another_writer(
    benchmark, tmp_path, monkeypatch
):
    output = tmp_path / "samples.jsonl"
    summary = benchmark.artifact_paths(output)[1]
    real_open = Path.open

    def replace_sample(path, *args, **kwargs):
        if path == summary:
            output.unlink()
            output.write_bytes(b"other writer")
            raise PermissionError("summary permission denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", replace_sample)
    with pytest.raises(PermissionError, match="summary permission denied"):
        benchmark.publish_outputs(output, "new samples", "new summary")
    assert output.read_bytes() == b"other writer"
    assert not summary.exists()


def test_write_failure_removes_both_owned_outputs(benchmark, tmp_path, monkeypatch):
    output = tmp_path / "samples.jsonl"
    real_open = Path.open

    class BrokenWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def fileno(self):
            return self.stream.fileno()

        def write(self, content):
            self.stream.write(content[:2])
            raise OSError("write failed")

    def broken_open(path, *args, **kwargs):
        return BrokenWriter(real_open(path, *args, **kwargs))

    monkeypatch.setattr(Path, "open", broken_open)
    with pytest.raises(OSError, match="write failed"):
        benchmark.publish_outputs(output, "samples", "summary")
    assert all(not path.exists() for path in benchmark.artifact_paths(output))


def test_exclusive_publication_writes_both_outputs(benchmark, tmp_path):
    output = tmp_path / "new/samples.jsonl"
    benchmark.publish_outputs(output, "samples\n", "summary\n")
    assert [path.read_text() for path in benchmark.artifact_paths(output)] == [
        "samples\n",
        "summary\n",
    ]


@pytest.mark.parametrize("fail", [False, True])
def test_main_reports_artifact_paths_or_publication_failure(
    benchmark, tmp_path, monkeypatch, capsys, fail
):
    output = tmp_path / "samples.jsonl"
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--runs", "1", "--output", str(output)]
    )

    def worker_output(command, **kwargs):
        if command[:2] == ["git", "rev-parse"]:
            return "0" * 40 + "\n"
        if command[:2] == ["git", "archive"]:
            return b"fixture archive"
        if "--worker" in command:
            return json.dumps(
                {
                    "workload": command[command.index("--workload") + 1],
                    "rows": int(command[command.index("--rows") + 1]),
                    "median_ms": 1.0,
                    "payload": {"same": True},
                }
            )
        pytest.fail(f"unexpected subprocess: {command}")

    def extract_archive(command, *, input, check):
        assert command[:4] == ["tar", "-xf", "-", "-C"]
        assert input == b"fixture archive"
        assert check
        source = Path(command[4]) / "src"
        source.mkdir()
        (source / "fixture.py").write_text("# immutable fixture source\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(benchmark.subprocess, "check_output", worker_output)
    monkeypatch.setattr(benchmark.subprocess, "run", extract_archive)
    if fail:

        def collision(*_args):
            raise FileExistsError("late collision")

        monkeypatch.setattr(benchmark, "publish_outputs", collision)
        with pytest.raises(SystemExit) as error:
            benchmark.main()
        assert error.value.code == 1
        assert (
            "could not publish benchmark outputs: late collision"
            in capsys.readouterr().err
        )
    else:
        benchmark.main()
        printed = capsys.readouterr().out
        assert all(
            f"wrote {path}" in printed for path in benchmark.artifact_paths(output)
        )


def test_origin_checks_skip_blocked_imports(benchmark, monkeypatch):
    monkeypatch.setitem(sys.modules, "grounded_weather_forecast.blocked", None)
    origins = benchmark.module_origins((REPO / "src").resolve())
    assert "grounded_weather_forecast.blocked" not in origins


def test_vanished_origin_is_actionable_and_chained(benchmark, tmp_path, monkeypatch):
    module = ModuleType("vanished")
    module.__file__ = str(tmp_path / "vanished.py")
    monkeypatch.setitem(sys.modules, "grounded_weather_forecast.vanished", module)
    with pytest.raises(
        RuntimeError, match=r"cannot resolve module origin.*vanished"
    ) as error:
        benchmark.module_origins((REPO / "src").resolve())
    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_worker_rechecks_origins_after_warmup(source_tree, tmp_path, benchmark):
    probe = tmp_path / "probe.py"
    probe.write_text(f"""import argparse
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("benchmark", {str(SCRIPT)!r})
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
real = benchmark.module_origins
checks = []
def verify(source):
    checks.append(source)
    if len(checks) == 2:
        raise RuntimeError("foreign origin discovered after warmup")
    return real(source)
def timed():
    raise AssertionError("timing began before origin validation")
benchmark.module_origins = verify
benchmark.time.perf_counter = timed
benchmark.worker(argparse.Namespace(source=Path({str(source_tree)!r}), rows=3, workload="successful_fit"))
""")
    completed = subprocess.run(  # noqa: S603 -- local worker and fixture argv
        [sys.executable, str(probe)],
        env=os.environ | benchmark.THREADS,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "foreign origin discovered after warmup" in completed.stderr
    assert "timing began" not in completed.stderr
