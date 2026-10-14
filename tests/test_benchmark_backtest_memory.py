"""Benchmark comparisons are invariant to Parquet encoding and row groups."""

import importlib
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("benchmark_backtest_memory")


def test_value_digest_checks_values_nulls_order_and_schema(tmp_path, benchmark):
    frame = pl.DataFrame(
        {
            "category": ["a", "b", None, "a", "c"] * 5,
            "values": [1.0, None, 2.0, 3.0, 4.0] * 5,
            "json": ["[1.0]", None, "", "[2.0]", "[3.0]"] * 5,
        }
    )
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    frame.write_parquet(first, row_group_size=3)
    frame.write_parquet(second, row_group_size=11)
    expected = benchmark.value_digest(first, ["category"])
    assert benchmark.value_digest(second, ["category"]) == expected
    for changed in (
        frame.reverse(),
        frame.with_columns(pl.col("values").fill_null(0.0)),
        frame.with_columns(pl.col("values").cast(pl.Float32)),
        frame.with_columns(pl.lit("different").alias("category")),
        frame.with_columns(pl.col("json").fill_null("")),
    ):
        changed.write_parquet(second, row_group_size=11)
        assert benchmark.value_digest(second, ["category"]) != expected


@pytest.mark.parametrize(
    ("ratio", "growth", "passed"),
    [(0.5, 128, True), (0.51, 128, False), (0.5, 129, False)],
)
def test_acceptance_boundaries(benchmark, ratio, growth, passed):
    summary = []
    for rows in benchmark.ROW_COUNTS:
        summary.append(
            {
                "rows": rows,
                "variant": "baseline",
                "payload_sha256": "same",
                "peak_physical_footprint_mib_median": 1000,
            }
        )
        summary.append(
            {
                "rows": rows,
                "variant": "updated",
                "payload_sha256": "same",
                "peak_physical_footprint_mib_median": 1000 * ratio
                - (growth if rows == benchmark.ROW_COUNTS[0] else 0),
            }
        )
    gates = benchmark.acceptance(summary)
    assert gates["memory_acceptance"] is passed
    assert gates["exact_output_agreement"] is True


def test_missing_physical_metrics_do_not_claim_acceptance(benchmark):
    summary = [
        {"rows": rows, "variant": variant, "payload_sha256": "same"}
        for rows in benchmark.ROW_COUNTS
        for variant in ("baseline", "updated")
    ]
    assert isinstance(benchmark.acceptance(summary)["memory_acceptance"], str)
