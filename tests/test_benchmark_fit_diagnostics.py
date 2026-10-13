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
        assert destination == REPO / "docs/research/fit-diagnostics-2026-10-13.jsonl"
        assert destination.is_absolute()
    else:
        assert destination == supplied
        if not supplied.is_absolute():
            assert destination.resolve() == tmp_path / supplied
