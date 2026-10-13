"""Completed benchmark samples survive abrupt termination and failed recovery."""

import importlib
import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def outputs(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("benchmark_outputs")


@pytest.fixture
def publisher(outputs, tmp_path):
    return outputs.BenchmarkOutputs.prepare(tmp_path, "probe", tmp_path / "out.jsonl")


@pytest.mark.parametrize("partial", [False, True])
def test_killed_process_retains_acknowledged_samples(tmp_path, partial):
    child = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from benchmark_outputs import BenchmarkOutputs
root = Path(sys.argv[2])
outputs = BenchmarkOutputs.prepare(root, 'probe', root / 'out.jsonl')
with outputs.measurements() as results:
    results.add_sample({'sample': 1})
    results.add_sample({'sample': 2})
    if sys.argv[3] == 'True':
        results.journal.write('{"sample":')
        results.journal.flush()
    print('checkpointed', flush=True)
    sys.stdin.read(1)
"""
    process = subprocess.Popen(  # noqa: S603 -- local child, no shell
        [sys.executable, "-c", child, str(SCRIPTS), str(tmp_path), str(partial)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    acknowledged = queue.Queue()

    def read_acknowledgement():
        journal = process.stdout.readline().strip()
        marker = process.stdout.readline().strip()
        acknowledged.put((journal, marker))

    reader = threading.Thread(target=read_acknowledgement, daemon=True)
    reader.start()
    try:
        journal, marker = acknowledged.get(timeout=10)
        assert marker == "checkpointed"
        process.kill()
        process.wait(timeout=10)
        path = Path(journal.removeprefix("journal "))
        lines = path.read_text().splitlines()
        assert [json.loads(line) for line in lines[:2]] == [
            {"sample": 1},
            {"sample": 2},
        ]
        assert lines[2:] == (['{"sample":'] if partial else [])
        assert not (tmp_path / "out.jsonl").exists()
        assert not list(path.parent.glob("*.summary.json"))
        assert not list(path.parent.glob("*recovery-*.jsonl"))
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()
        reader.join(timeout=10)


def test_journal_retained_after_success_and_written_before_measurements(
    publisher, capsys
):
    with publisher.measurements() as results:
        assert f"journal {results.journal_path}" in capsys.readouterr().out
        results.add_sample({"sample": 0})
        assert results.journal_path.read_text() == '{"sample": 0}\n'
    assert results.journal.closed
    assert results.journal_path.exists()
    assert not results.journal_path.with_suffix(".summary.json").exists()


@pytest.mark.parametrize("failure", ["serialization", "rescue", "report", "close"])
@pytest.mark.parametrize(
    "secondary_type", [RuntimeError, MemoryError, KeyboardInterrupt]
)
def test_secondary_failures_preserve_original_exception_and_traceback(
    outputs, publisher, monkeypatch, failure, secondary_type
):
    original = RuntimeError("original worker failure")
    attempts = []
    real_rescue = outputs.BenchmarkOutputs.rescue
    real_contents = outputs.BenchmarkResults.contents
    real_close = outputs.BenchmarkResults.close

    def secondary(*_args, **_kwargs):
        raise secondary_type("secondary failure")

    def rescue(self, contents):
        attempts.append(contents)
        if failure == "rescue":
            secondary()
        return real_rescue(self, contents)

    def original_failure():
        with publisher.measurements() as results:
            results.add_sample({"sample": 1})
            monkeypatch.setattr(outputs.BenchmarkOutputs, "rescue", rescue)
            if failure == "serialization":
                monkeypatch.setattr(outputs.BenchmarkResults, "contents", secondary)
            elif failure == "report":
                monkeypatch.setattr(outputs, "print", secondary, raising=False)
            elif failure == "close":
                monkeypatch.setattr(outputs.BenchmarkResults, "close", secondary)
            raise original

    with pytest.raises(RuntimeError) as error:
        original_failure()
    assert error.value is original
    assert any(entry.name == "original_failure" for entry in error.traceback)
    assert len(attempts) == (0 if failure == "serialization" else 1)
    journal = next(publisher.repository.glob("reports/benchmarks/*journal-*.jsonl"))
    assert json.loads(journal.read_text()) == {"sample": 1}
    monkeypatch.setattr(outputs.BenchmarkResults, "contents", real_contents)
    monkeypatch.setattr(outputs.BenchmarkResults, "close", real_close)
    # Close the owned handle after deliberately disabling closure in the handler.
    if failure == "close":
        # The traceback retains the yielded results until the exception is released.
        for entry in error.traceback:
            if entry.name == "original_failure":
                real_close(entry.frame.f_locals["results"])


@pytest.mark.parametrize("failure", ["write", "flush", "close", "serialization"])
def test_checkpoint_failures_are_fatal_and_recover_available_samples(
    outputs, publisher, monkeypatch, failure
):
    original = OSError("checkpoint failed")
    owned = None
    real_close = outputs.BenchmarkResults.close

    class BrokenJournal:
        def __init__(self, stream):
            self.stream = stream

        def write(self, text):
            if failure == "write":
                self.stream.write(text[:2])
                self.stream.flush()
                raise original
            return self.stream.write(text)

        def flush(self):
            if failure == "flush":
                raise original
            return self.stream.flush()

        def close(self):
            self.stream.close()
            if failure == "close":
                raise original

    def failed_checkpoint():
        nonlocal owned
        with publisher.measurements() as results:
            owned = results
            results.journal = BrokenJournal(results.journal)
            if failure == "serialization":

                def cannot_serialize(*_args, **_kwargs):
                    raise original

                monkeypatch.setattr(outputs.json, "dumps", cannot_serialize)
            results.add_sample({"sample": 1})

    with pytest.raises(OSError, match="checkpoint failed") as error:
        failed_checkpoint()
    assert error.value is original
    assert owned.journal_path.exists()
    if failure != "serialization":
        recovery = next(owned.journal_path.parent.glob("*recovery-*.jsonl"))
        assert recovery.read_text() == '{"sample": 1}\n'
        assert not recovery.with_suffix(".summary.json").exists()
    if failure != "close":
        real_close(owned)


@pytest.mark.parametrize("failure", ["collision", "late_collision", "open", "probe"])
def test_journal_preparation_never_overwrites(
    outputs, publisher, tmp_path, monkeypatch, failure
):
    path = tmp_path / "fixed-journal.jsonl"
    monkeypatch.setattr(outputs, "output_path", lambda *_args, **_kwargs: path)
    real_open = Path.open
    if failure == "collision":
        path.write_bytes(b"previous sample")
    if failure == "probe":

        def denied(*_args, **_kwargs):
            raise PermissionError("journal parent denied")

        monkeypatch.setattr(outputs.tempfile, "TemporaryFile", denied)
    elif failure in {"late_collision", "open"}:

        def failed_open(destination, *args, **kwargs):
            if destination == path and args and args[0] == "x":
                if failure == "late_collision":
                    with real_open(path, "w") as stream:
                        stream.write("late sample")
                else:
                    raise PermissionError("journal open denied")
            return real_open(destination, *args, **kwargs)

        monkeypatch.setattr(Path, "open", failed_open)
    with pytest.raises(outputs.JournalPreparationError), publisher.measurements():
        pytest.fail("must stop before measurements")
    if failure in {"collision", "late_collision"}:
        assert path.read_text() == (
            "previous sample" if failure == "collision" else "late sample"
        )


@pytest.mark.parametrize("runner", ["fit_diagnostics", "score_readers"])
def test_journal_preparation_failure_exits_two_before_git(
    outputs, tmp_path, monkeypatch, runner
):
    spec = importlib.util.spec_from_file_location(
        "runner", SCRIPTS / f"benchmark_{runner}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys, "argv", ["benchmark", "--output", str(tmp_path / "out.jsonl")]
    )
    monkeypatch.setattr(
        outputs.BenchmarkOutputs,
        "open_journal",
        lambda _self: (_ for _ in ()).throw(
            outputs.JournalPreparationError("journal denied")
        ),
    )
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda *_args, **_kwargs: pytest.fail("Git must not run"),
    )
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


@pytest.mark.parametrize("runner", ["fit_diagnostics", "score_readers"])
@pytest.mark.parametrize("mode", ["-P", "-I", "environment"])
def test_safe_path_launchers(tmp_path, runner, mode):
    arguments = [] if mode == "environment" else [mode]
    environment = os.environ | (
        {"PYTHONSAFEPATH": "1"} if mode == "environment" else {}
    )
    completed = subprocess.run(  # noqa: S603 -- local script, no shell
        [sys.executable, *arguments, str(SCRIPTS / f"benchmark_{runner}.py"), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert "--output" in completed.stdout


@pytest.mark.parametrize("mode", ["-P", "-I", "environment"])
def test_safe_path_score_workers(tmp_path, mode):
    spec = importlib.util.spec_from_file_location(
        "score_runner", SCRIPTS / "benchmark_score_readers.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    module.prepare(fixtures, 2, 4)
    arguments = [] if mode == "environment" else [mode]
    environment = os.environ | module.THREADS
    if mode == "environment":
        environment["PYTHONSAFEPATH"] = "1"
    completed = subprocess.run(  # noqa: S603 -- real local worker and immutable scores
        [
            sys.executable,
            *arguments,
            str(SCRIPTS / "benchmark_score_readers.py"),
            "--worker",
            "--source",
            str(SCRIPTS.parent / "src"),
            "--fixtures",
            str(fixtures),
            "--workload",
            "many_evaluations",
            "--policy",
            "bounded",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    record = json.loads(completed.stdout)
    assert record["workload"] == "many_evaluations"
    assert record["payload_sha256"] == module.digest_payload(record["payload"])


@pytest.mark.parametrize(
    "secondary_type", [RuntimeError, MemoryError, KeyboardInterrupt]
)
def test_publication_rescue_interrupt_preserves_failure_exit(
    outputs, publisher, monkeypatch, secondary_type
):
    original = OSError("publication failed")
    attempts = []

    def fail_publish(*_args):
        raise original

    def fail_rescue(*_args):
        attempts.append(True)
        raise secondary_type("secondary failure")

    with publisher.measurements() as results:
        results.add_sample({"sample": 1})
        contents = results.contents(summary=[])
    monkeypatch.setattr(outputs, "publish_artifacts", fail_publish)
    monkeypatch.setattr(outputs.BenchmarkOutputs, "rescue", fail_rescue)
    with pytest.raises(SystemExit) as error:
        publisher.publish(contents, journal_path=results.journal_path)
    assert error.value.code == 1
    assert error.value.__cause__ is original
    assert attempts == [True]
    assert results.journal_path.read_text() == contents["samples"]
