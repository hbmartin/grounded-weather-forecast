# Verified fitting diagnostics benchmark — 2026-10-13

This repeats the fitting diagnostics comparison against merged PR #54
(`2ccd7e0`) with explicit import-provenance checks. The updated implementation
also retains every promoted unavailable-fit attempt within a product snapshot.
The earlier [2026-10-12 measurements](fit-diagnostics-2026-10-12.md) remain
unchanged.

## Method and provenance

Three fresh processes per variant, workload, and size, with one warm-up and nine
timed iterations per process. Both variants use immutable source snapshots.
Workers resolve the requested source directory and reject missing or foreign
origins for every imported `grounded_weather_forecast` module before variant
selection and again after warm-up. These checks run outside the timed sections.
All 36 samples verified 74 module paths against their requested snapshot;
the paths and resolved source roots are included in the raw samples.

Fixtures contain 13 truth columns and two provider columns for one hourly
temperature target. Matrices and prediction inputs are already in memory;
matrix loading, provider databases, other variables, dressing, and forecast
document assembly are excluded. These are isolated collection and fitting
measurements, not end-to-end predict/publish latency or memory measurements.

`POLARS_MAX_THREADS=2`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, and `PYTHONHASHSEED=0`, matching the original comparison.
Python 3.13.14; Polars 1.42.1.

The aggregation workload retains the original all-column collection fixture
with 90% eligible rows. The updated path captures row metadata without scanning
truth when no fit failed. The fitting workloads include causal filtering,
collection, a real equal-weight fit, and updated failed-target resolution.
Pending-truth rows were issued before the cutoff but become knowable afterward.
Both fitting fixtures verify a promoted selection: the successful fit retains
its release and the pending fit records exactly one scoped failure.

## Results

Each entry is the median of three process medians, in milliseconds.

| Workload | Raw rows | Baseline | Updated |
|---|---:|---:|---:|
| aggregation | 100,000 | 1.203 | 0.001 |
| aggregation | 1,000,000 | 10.653 | 0.001 |
| successful_fit | 100,000 | 3.100 | 1.999 |
| successful_fit | 1,000,000 | 30.278 | 17.860 |
| pending_fit | 100,000 | 1.215 | 1.022 |
| pending_fit | 1,000,000 | 6.911 | 2.284 |

All variants produced equivalent point values, method labels, release IDs, and
quantile maps. Scoped diagnosis wording intentionally differs and is covered by
regressions rather than this payload comparison. The numbers compare the
original baseline with the current source; they do not isolate the cost of
retaining multiple failures.

Successful fits request no diagnostic truth aggregations. Failed promoted
attempts resolve their required truth columns once per product. Contexts retain
small facts and failures rather than frames; raw matrices remain local through
fitting. This experiment makes no peak-memory claim. Regressions cover one
matrix read, frame lifetime, retained failures after later attempts, and ready
forecasts with surviving finite attribution.

## Reproduction and samples

```bash
uv run python scripts/benchmark_fit_diagnostics.py --baseline 2ccd7e0 --runs 3
```

The recorded run was launched from `/tmp` using absolute project and script
paths. With no `--output`, results still went to this repository's
`docs/research/fit-diagnostics-2026-10-13.jsonl`. Explicit relative output paths
remain relative to the caller's working directory.

- [Raw samples](fit-diagnostics-2026-10-13.jsonl)
- [Summary](fit-diagnostics-2026-10-13.summary.json)

Baseline commit: `2ccd7e0a3a0cf914d1dc56fc5088de93ab2c3779`.
Baseline source SHA-256: `6eaf9060ff9c7463c9ad6b05bb20460fc44df09f4cb3a012f2b2403709984470`.
Updated source SHA-256: `6de1f827c71626245884af52353e7750b2f8ddfa599e1eae2835b2211d11f06f`.
Runner SHA-256: `41c9279b53165b38f577aa413da0043c231113aae9b6a309367121a1d4e46ac5`.
Measured serving code identity: `0.4.0+d4ab422da01f`.

Deployment is separate. The changed code fingerprint requires fresh live
backtest and report evidence before rollout; these measurements neither refresh
production evidence nor rewrite historical artifacts.
