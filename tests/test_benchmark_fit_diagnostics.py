"""Workers must measure the requested tree and keep outputs at the chosen path."""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/benchmark_fit_diagnostics.py"


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
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


@pytest.mark.parametrize("fail", [False, True])
def test_main_reports_artifact_paths_or_publication_failure(
    benchmark, tmp_path, monkeypatch, capsys, fail
):
    import benchmark_outputs  # noqa: PLC0415 -- script helper

    output = tmp_path / "samples.jsonl"
    repository = tmp_path / "repository"
    source = repository / "src"
    source.mkdir(parents=True)
    (source / "fixture.py").write_text("# source fixture\n")
    script = repository / "scripts/benchmark.py"
    script.parent.mkdir()
    script.write_text("# runner fixture\n")
    monkeypatch.setattr(benchmark, "__file__", str(script))
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--runs", "1", "--output", str(output)]
    )

    def worker_output(command, **_kwargs):
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

    def extract_archive(command, **kwargs):
        assert command[:4] == ["tar", "-xf", "-", "-C"]
        assert kwargs["input"] == b"fixture archive"
        assert kwargs["check"]
        source = Path(command[4]) / "src"
        source.mkdir()
        (source / "fixture.py").write_text("# immutable fixture source\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(benchmark.subprocess, "check_output", worker_output)
    monkeypatch.setattr(benchmark.subprocess, "run", extract_archive)
    if fail:
        real_publish = benchmark_outputs.publish_artifacts

        def collision(paths, contents):
            if paths["samples"] == output:
                raise FileExistsError("late collision")
            return real_publish(paths, contents)

        monkeypatch.setattr(benchmark_outputs, "publish_artifacts", collision)
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
            f"wrote {path}" in printed
            for path in benchmark_outputs.artifact_paths(output).values()
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
import sys
from pathlib import Path
sys.path.insert(0, {str(SCRIPT.parent)!r})
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
