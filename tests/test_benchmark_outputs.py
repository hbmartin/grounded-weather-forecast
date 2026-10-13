"""Benchmark publication refuses overwrites and retains recoverable results."""

import importlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def outputs(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("benchmark_outputs")


def load_runner(name, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    path = SCRIPTS / f"benchmark_{name}.py"
    spec = importlib.util.spec_from_file_location(f"benchmark_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["fit_diagnostics", "score_readers"])
def runner(request, monkeypatch):
    return load_runner(request.param, monkeypatch)


@pytest.fixture
def fitting_runner(monkeypatch):
    return load_runner("fit_diagnostics", monkeypatch)


@pytest.mark.parametrize("prefix", ["fit-diagnostics", "score-readers"])
@pytest.mark.parametrize("kind", ["default", "relative", "absolute"])
def test_unique_defaults_and_explicit_paths_from_another_cwd(
    outputs, tmp_path, monkeypatch, prefix, kind
):
    repository = tmp_path / "repository"
    monkeypatch.chdir(tmp_path)
    supplied = (
        None
        if kind == "default"
        else Path("chosen/results.jsonl")
        if kind == "relative"
        else tmp_path / "chosen.jsonl"
    )
    destination = outputs.output_path(repository, supplied, prefix=prefix)
    if supplied is None:
        assert destination.parent == repository / "reports/benchmarks"
        stamp, identifier = destination.stem.removeprefix(f"{prefix}-").split("-")
        assert datetime.strptime(stamp, "%Y%m%dT%H%M%S%fZ").replace(
            tzinfo=UTC
        ) <= datetime.now(UTC)
        assert len(identifier) == 32
        assert destination != outputs.output_path(repository, None, prefix=prefix)
    else:
        assert destination == supplied
        assert destination.resolve() == tmp_path / supplied


@pytest.mark.parametrize("artifact", ["samples", "summary"])
@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "broken_symlink"])
def test_all_registered_collisions_stop_before_git(
    runner, outputs, tmp_path, monkeypatch, artifact, kind
):
    destination = tmp_path / "samples.jsonl"
    collision = outputs.artifact_paths(destination, include_payloads=True)[artifact]
    if kind == "file":
        collision.write_bytes(b"previous evidence")
    elif kind == "directory":
        collision.mkdir()
    else:
        target = tmp_path / "target"
        if kind == "symlink":
            target.write_bytes(b"previous evidence")
        collision.symlink_to(target)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output", str(destination)])

    def no_git(*_args, **_kwargs):
        pytest.fail("preflight must stop before Git, fixtures, or workers")

    monkeypatch.setattr(runner.subprocess, "check_output", no_git)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert collision.exists() or collision.is_symlink()
    if kind in {"file", "symlink"}:
        assert collision.read_bytes() == b"previous evidence"


@pytest.mark.parametrize("destination", [".", "/"])
def test_directory_only_destinations_exit_two(
    runner, tmp_path, monkeypatch, destination
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output", destination])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


@pytest.mark.parametrize("cause", ["write_probe", "parent_is_file"])
def test_unwritable_parents_stop_before_git(
    runner, outputs, tmp_path, monkeypatch, cause
):
    parent = tmp_path / "parent"
    if cause == "parent_is_file":
        parent.write_bytes(b"keep parent")
    else:

        def denied(*_args, **_kwargs):
            raise PermissionError("output write denied")

        monkeypatch.setattr(outputs.tempfile, "TemporaryFile", denied)
    monkeypatch.setattr(
        sys, "argv", ["benchmark", "--output", str(parent / "out.jsonl")]
    )

    def no_git(*_args, **_kwargs):
        pytest.fail("unwritable parent must stop before Git")

    monkeypatch.setattr(runner.subprocess, "check_output", no_git)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert not (parent / "out.jsonl").exists()


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="POSIX permissions")
def test_preflight_actually_checks_parent_writability(outputs, tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir(mode=0o500)
    try:
        with pytest.raises(PermissionError):
            outputs.BenchmarkOutputs.prepare(tmp_path, "probe", parent / "out.jsonl")
    finally:
        parent.chmod(0o700)


def test_artifact_registry_drives_preflight_and_publication(outputs, tmp_path):
    paths = outputs.artifact_paths(
        tmp_path / "new/samples.jsonl", include_payloads=True
    )
    paths["extra"] = tmp_path / "new/extra.json"
    outputs.preflight_artifacts(paths)
    contents = {name: name for name in paths}
    outputs.publish_artifacts(paths, contents)
    assert {name: path.read_text() for name, path in paths.items()} == contents
    with pytest.raises(FileExistsError):
        outputs.preflight_artifacts(paths)


def test_duplicate_destinations_are_rejected(outputs, tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        outputs.preflight_artifacts(
            dict.fromkeys(("samples", "summary"), tmp_path / "a")
        )


@pytest.mark.parametrize("artifact", ["samples", "summary", "payloads"])
def test_late_collision_never_removes_destinations(outputs, tmp_path, artifact):
    paths = outputs.artifact_paths(tmp_path / "samples.jsonl", include_payloads=True)
    outputs.preflight_artifacts(paths)
    collision = paths[artifact]
    collision.write_bytes(b"late evidence")
    with pytest.raises(FileExistsError):
        outputs.publish_artifacts(paths, dict.fromkeys(paths, "new evidence"))
    assert collision.read_bytes() == b"late evidence"
    prior = list(paths)[: list(paths).index(artifact)]
    assert all(paths[name].read_bytes() == b"" for name in prior)


def test_failed_publication_retains_another_writers_replacement(
    outputs, tmp_path, monkeypatch
):
    paths = outputs.artifact_paths(tmp_path / "samples.jsonl")
    real_open = Path.open

    def replace_sample(path, *args, **kwargs):
        if path == paths["summary"]:
            paths["samples"].rename(tmp_path / "parked")
            paths["samples"].write_bytes(b"other writer")
            raise PermissionError("summary permission denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", replace_sample)
    with pytest.raises(PermissionError):
        outputs.publish_artifacts(paths, {"samples": "new", "summary": "new"})
    assert paths["samples"].read_bytes() == b"other writer"


def test_write_failure_retains_incomplete_files(outputs, tmp_path, monkeypatch):
    paths = outputs.artifact_paths(tmp_path / "samples.jsonl")
    real_open = Path.open

    class BrokenWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, content):
            self.stream.write(content[:2])
            raise OSError("write failed")

    monkeypatch.setattr(
        Path,
        "open",
        lambda path, *args, **kwargs: (
            BrokenWriter(real_open(path, *args, **kwargs))
            if args and args[0] == "x"
            else real_open(path, *args, **kwargs)
        ),
    )
    with pytest.raises(OSError, match="write failed"):
        outputs.publish_artifacts(paths, {"samples": "samples", "summary": "summary"})
    assert paths["samples"].read_text() == "sa"
    assert paths["summary"].read_text() == ""


def test_publication_failure_rescues_validated_results_and_returns_failure(
    outputs, tmp_path, capsys
):
    publisher = outputs.BenchmarkOutputs.prepare(
        tmp_path, "probe", tmp_path / "out.jsonl"
    )
    publisher.paths["summary"].write_bytes(b"late evidence")
    contents = outputs.result_contents([{"sample": 1}], summary=[{"valid": True}])
    with pytest.raises(SystemExit) as error:
        publisher.publish(contents)
    assert error.value.code == 1
    assert publisher.paths["summary"].read_bytes() == b"late evidence"
    assert publisher.paths["samples"].read_bytes() == b""
    recovery = next((tmp_path / "reports/benchmarks").glob("*recovery-*.jsonl"))
    assert recovery.read_text() == contents["samples"]
    assert recovery.with_suffix(".summary.json").read_text() == contents["summary"]
    assert "no files were removed" in capsys.readouterr().err


def test_recovery_failure_dumps_samples_after_one_attempt(
    outputs, tmp_path, monkeypatch, capsys
):
    publisher = outputs.BenchmarkOutputs.prepare(
        tmp_path, "probe", tmp_path / "out.jsonl"
    )
    calls = []

    def fail_publish(paths, _contents):
        calls.append(paths["samples"])
        raise OSError("disk full")

    monkeypatch.setattr(outputs, "publish_artifacts", fail_publish)
    with pytest.raises(SystemExit) as error:
        publisher.publish(outputs.result_contents([{"sample": 1}]))
    assert error.value.code == 1
    assert len(calls) == 2
    stderr = capsys.readouterr().err
    assert "could not publish" in stderr
    assert "could not rescue" in stderr
    assert json.loads(stderr.splitlines()[-1]) == {"sample": 1}


def test_measurement_failure_rescues_rows_but_no_summary(outputs, tmp_path):
    publisher = outputs.BenchmarkOutputs.prepare(
        tmp_path, "probe", tmp_path / "out.jsonl"
    )
    original = RuntimeError("worker stopped")

    def measured_failure():
        with publisher.measurements() as results:
            results.add_sample({"sample": 1})
            raise original

    with pytest.raises(RuntimeError) as error:
        measured_failure()
    assert error.value is original
    recovery = next((tmp_path / "reports/benchmarks").glob("*recovery-*.jsonl"))
    assert json.loads(recovery.read_text()) == {"sample": 1}
    assert not recovery.with_suffix(".summary.json").exists()
    assert not publisher.paths["samples"].exists()


def test_failure_before_samples_does_not_create_recovery(outputs, tmp_path):
    publisher = outputs.BenchmarkOutputs.prepare(
        tmp_path, "probe", tmp_path / "out.jsonl"
    )
    with pytest.raises(RuntimeError), publisher.measurements() as results:
        raise RuntimeError("Git failed")
    assert results.journal_path.read_text() == ""
    assert not list((tmp_path / "reports/benchmarks").glob("*recovery-*.jsonl"))


def test_success_prints_created_paths_and_omits_unused_payloads(
    outputs, tmp_path, capsys
):
    publisher = outputs.BenchmarkOutputs.prepare(
        tmp_path, "score-readers", tmp_path / "out.jsonl", include_payloads=True
    )
    publisher.publish(outputs.result_contents([{"sample": 1}], summary=[]))
    printed = capsys.readouterr().out
    assert f"wrote {publisher.paths['samples']}" in printed
    assert f"wrote {publisher.paths['summary']}" in printed
    assert not publisher.paths["payloads"].exists()


@pytest.fixture
def score_runner(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "score_benchmark", SCRIPTS / "benchmark_score_readers.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repository = tmp_path / "repository"
    script = repository / "scripts/benchmark.py"
    script.parent.mkdir(parents=True)
    script.write_text("# runner fixture\n")
    source = repository / "src"
    source.mkdir()
    (source / "fixture.py").write_text("# current source\n")
    monkeypatch.setattr(module, "__file__", str(script))

    def fixtures(directory, evaluations, cases):
        assert evaluations == 2
        assert cases == 80
        for scenario in module.SCENARIOS[:2]:
            target = directory / scenario
            target.mkdir()
            for index in range(evaluations):
                (target / f"scores_{index}.parquet").write_bytes(b"immutable scores")

    def reporting(directory):
        target = directory / "reporting"
        target.mkdir()
        (target / "scores.parquet").write_bytes(b"immutable reporting scores")

    monkeypatch.setattr(module, "prepare", fixtures)
    monkeypatch.setattr(module, "prepare_reporting", reporting)
    monkeypatch.setattr(module.platform, "platform", lambda: "fixture-platform")
    return module


def stub_score_processes(module, monkeypatch, mode):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        source = b"# baseline fixture\n"
        entry = tarfile.TarInfo("src/fixture.py")
        entry.size = len(source)
        archive.addfile(entry, io.BytesIO(source))
    workers = []
    failure = subprocess.CalledProcessError(1, ["worker"], output="worker failed")

    def process(command, **_kwargs):
        if command[:2] == ["git", "rev-parse"]:
            return "0" * 40 + "\n"
        if command[:2] == ["git", "archive"]:
            return buffer.getvalue()
        assert "--worker" in command
        assert all(
            _kwargs["env"][key] == value for key, value in module.THREADS.items()
        )
        workers.append(command)
        if mode == "worker_failure" and len(workers) == 2:
            raise failure
        if mode == "runner_change" and len(workers) == 2:
            Path(module.__file__).write_text("# changed runner\n")
        workload = command[command.index("--workload") + 1]
        policy = command[command.index("--policy") + 1]
        payload = {"same": policy if mode == "mismatch" else True}
        return json.dumps(
            {
                "workload": workload,
                "policy": policy,
                "seconds": 1.0,
                "peak_rss_mib": 1.0,
                "full_load_calls": 2,
                "payload_sha256": module.digest_payload(payload),
                "payload": payload,
            }
        )

    monkeypatch.setattr(module.subprocess, "check_output", process)
    return failure


@pytest.mark.parametrize("kind", ["default", "relative", "absolute"])
def test_score_runner_publishes_validated_samples_without_payload_diagnostics(
    score_runner, tmp_path, monkeypatch, capsys, kind
):
    stub_score_processes(score_runner, monkeypatch, "success")
    monkeypatch.chdir(tmp_path)
    argv = ["benchmark", "--runs", "3", "--evaluations", "2", "--cases", "80"]
    if kind != "default":
        argv.extend(
            [
                "--output",
                "out.jsonl" if kind == "relative" else str(tmp_path / "out.jsonl"),
            ]
        )
    monkeypatch.setattr(sys, "argv", argv)
    score_runner.main()
    output = (
        next(
            path
            for path in (tmp_path / "repository/reports/benchmarks").glob("*.jsonl")
            if "-journal-" not in path.name
        )
        if kind == "default"
        else tmp_path / "out.jsonl"
    )
    assert len(output.read_text().splitlines()) == 48
    assert len(json.loads(output.with_suffix(".summary.json").read_text())) == 16
    assert not output.with_suffix(".payloads.json").exists()
    assert (
        f"wrote {output if kind != 'relative' else Path('out.jsonl')}"
        in capsys.readouterr().out
    )


@pytest.mark.parametrize(
    "mode", ["worker_failure", "mismatch", "publication_failure", "runner_change"]
)
def test_score_runner_failure_retains_results_and_original_failure(
    score_runner, outputs, tmp_path, monkeypatch, capsys, mode
):
    original = stub_score_processes(score_runner, monkeypatch, mode)
    output = tmp_path / "out.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--runs",
            "3",
            "--evaluations",
            "2",
            "--cases",
            "80",
            "--output",
            str(output),
        ],
    )
    if mode == "publication_failure":
        real_publish = outputs.publish_artifacts

        def late_collision(paths, contents):
            if paths["samples"] == output:
                paths["summary"].write_bytes(b"other writer")
            return real_publish(paths, contents)

        monkeypatch.setattr(outputs, "publish_artifacts", late_collision)
        expected = SystemExit
    else:
        expected = (
            subprocess.CalledProcessError if mode == "worker_failure" else RuntimeError
        )
    with pytest.raises(expected) as error:
        score_runner.main()
    if mode == "worker_failure":
        assert error.value is original
    elif mode == "publication_failure":
        assert error.value.code == 1
        assert output.read_bytes() == b""
        assert output.with_suffix(".summary.json").read_bytes() == b"other writer"
    else:
        assert (
            "runner changed" if mode == "runner_change" else "payloads differ"
        ) in str(error.value)
    recovery = next(
        (tmp_path / "repository/reports/benchmarks").glob("*recovery-*.jsonl")
    )
    assert (
        len(recovery.read_text().splitlines())
        == {
            "worker_failure": 1,
            "mismatch": 12,
            "publication_failure": 48,
            "runner_change": 48,
        }[mode]
    )
    assert recovery.with_suffix(".summary.json").exists() == (
        mode == "publication_failure"
    )
    assert recovery.with_suffix(".payloads.json").exists() == (mode == "mismatch")
    assert f"rescued {recovery}" in capsys.readouterr().err


def test_fitting_validation_failure_rescues_before_propagating(
    fitting_runner, tmp_path, monkeypatch
):
    runner = fitting_runner
    repository = tmp_path / "repository"
    monkeypatch.setattr(runner, "__file__", str(repository / "scripts/benchmark.py"))
    monkeypatch.setattr(
        sys, "argv", ["benchmark", "--output", str(tmp_path / "out.jsonl")]
    )
    original = AssertionError("payloads differ")

    def failed_collection(_repository, _args, results):
        results.add_sample({"payload": {"point": 1.0}})
        raise original

    monkeypatch.setattr(runner, "collect", failed_collection)
    with pytest.raises(AssertionError) as error:
        runner.main()
    assert error.value is original
    recovery = next((repository / "reports/benchmarks").glob("*recovery-*.jsonl"))
    assert json.loads(recovery.read_text())["payload"]["point"] == 1.0
    assert not recovery.with_suffix(".summary.json").exists()


def test_score_metadata_collected_once_and_runner_revalidated(
    score_runner, tmp_path, monkeypatch
):
    stub_score_processes(score_runner, monkeypatch, "success")
    runner_reads = []
    platforms = []
    real_read = Path.read_bytes

    def read(path):
        if path == Path(score_runner.__file__):
            runner_reads.append(path)
        return real_read(path)

    def platform():
        platforms.append(True)
        return "fixture-platform"

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(score_runner.platform, "platform", platform)
    output = tmp_path / "out.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--runs",
            "3",
            "--evaluations",
            "2",
            "--cases",
            "80",
            "--output",
            str(output),
        ],
    )
    score_runner.main()
    assert len(runner_reads) == 2
    assert platforms == [True]
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len({record["runner_sha256"] for record in records}) == 1
    assert all(record["platform"] == "fixture-platform" for record in records)
    assert all(
        record["threads"]
        == {
            key: value
            for key, value in score_runner.THREADS.items()
            if key != "PYTHONHASHSEED"
        }
        for record in records
    )


def test_thread_configuration_shared_by_both_runners(
    outputs, fitting_runner, score_runner
):
    assert fitting_runner.THREADS is score_runner.THREADS is outputs.THREADS


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "broken_symlink"])
def test_unused_primary_payload_sidecar_does_not_block_score_runner(
    score_runner, tmp_path, monkeypatch, kind
):
    stub_score_processes(score_runner, monkeypatch, "success")
    destination = tmp_path / "samples.jsonl"
    sidecar = destination.with_suffix(".payloads.json")
    if kind == "file":
        sidecar.write_bytes(b"previous diagnostic")
    elif kind == "directory":
        sidecar.mkdir()
    else:
        target = tmp_path / "target"
        if kind == "symlink":
            target.write_bytes(b"previous diagnostic")
        sidecar.symlink_to(target)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--evaluations",
            "2",
            "--cases",
            "80",
            "--output",
            str(destination),
        ],
    )
    score_runner.main()
    assert len(destination.read_text().splitlines()) == 48
    assert destination.with_suffix(".summary.json").exists()
    assert sidecar.exists() or sidecar.is_symlink()
    if kind in {"file", "symlink"}:
        assert sidecar.read_bytes() == b"previous diagnostic"
