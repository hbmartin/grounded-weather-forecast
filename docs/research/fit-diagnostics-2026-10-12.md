# Fitting diagnostics benchmark — 2026-10-12

This compares the merged PR #54 baseline (`2ccd7e0`) with the implementation
that records authoritative fitting branches and resolves raw truth only for
failed promoted targets. Historical research samples were left unchanged.

## Method

Three fresh processes per variant/workload/size, each with one warm-up and nine
timed iterations. Both variants run from immutable source snapshots. Fixtures
contain 13 truth columns and two provider columns for one hourly temperature
target. Matrices and prediction inputs are already in memory; matrix loading,
provider databases, other variables, dressing, and forecast-document assembly
are excluded. These are isolated collection/fitting measurements, not full
predict/publish latency or memory measurements.

`POLARS_MAX_THREADS=2`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, and `PYTHONHASHSEED=0`, matching the existing score-reader
benchmark settings. Python 3.13.14; Polars 1.42.1.

The aggregation workload repeats the earlier all-column collection benchmark
(90% eligible rows). The updated path retains metadata only when no fit failed.
The fitting workloads include causal filtering, collection, a real equal-weight
blend attempt, and—under the updated code—resolution of failed-target causes.
Pending-truth rows are issued before the cutoff but become knowable afterward.
Both fitting fixtures assert that the selected lead bucket carries a promoted
release: successful fits retain it and pending fits record exactly one failure.

## Results

Each entry is the median of three process medians, in milliseconds.

| Workload | Raw rows | Baseline | Updated |
|---|---:|---:|---:|
| aggregation | 100,000 | 1.384 | 0.001 |
| aggregation | 1,000,000 | 11.587 | 0.001 |
| successful_fit | 100,000 | 3.664 | 2.080 |
| successful_fit | 1,000,000 | 33.409 | 20.656 |
| pending_fit | 100,000 | 1.718 | 1.296 |
| pending_fit | 1,000,000 | 7.537 | 2.389 |

All variants produced equivalent point values, method labels, release IDs, and
quantile maps. Fitting-diagnosis wording intentionally changes and is covered
by regressions rather than the output-equivalence comparison.

Successful fits request zero diagnostic truth scans. Failed promoted targets
resolve their required columns in one query per product, deduplicating repeated
targets. Diagnostic contexts contain no full-frame references. Raw matrices stay
local through product fitting, so this experiment does not claim unchanged peak
memory; regressions verify frame release and one matrix read.

## Reproduction and samples

```bash
uv run python scripts/benchmark_fit_diagnostics.py --baseline 2ccd7e0 --runs 3
```

- [Raw samples](fit-diagnostics-2026-10-12.jsonl)
- [Summary](fit-diagnostics-2026-10-12.summary.json)

Samples include the complete baseline SHA, immutable source SHA-256 values,
runner SHA-256, thread settings, versions, warm timing samples, and output values.

Baseline source SHA-256: `6eaf9060ff9c7463c9ad6b05bb20460fc44df09f4cb3a012f2b2403709984470`.
Updated source SHA-256: `094e4f61f0ca3ea87cc5d98415865f6b0641f9d977b5871b4c8849bc42b7780e`.
