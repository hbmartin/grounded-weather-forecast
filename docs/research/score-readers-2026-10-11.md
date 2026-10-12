# Score fallback and report reuse (2026-10-11)

Report now reuses its raw leaderboard and corrected winners for dashboard
summaries under `mcs` and `legacy`. The 12-file `mcs` benchmark has **24 → 12 full
load calls**, equivalent tables/charts, and **28.1% lower median report runtime**.
`seq_mcs` retains independent dashboard collection after report persists all
wealth updates, so earlier files see the final store. Decoder settings remain
`parallel="none", low_memory=True`.

*Implemented in: `serve/score_scan.py`, `dashboard/score_summary.py`,
`dashboard/context.py`, `dashboard/writer.py`, `cli.py`, `reports/eprocess.py`*

## Source and reproduction

The baseline is `4019cc661653af8955d265fc29ecd53f0f805736`. The measured source is
saved as a [source patch](score-readers-2026-10-11.source.patch) against that
revision. Applying it to the baseline was verified to reconstruct the measured
source exactly. Source hashes cover filenames and bytes under `src`, excluding
Python caches and Finder metadata.

```bash
POLARS_MAX_THREADS=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --locked python scripts/benchmark_score_readers.py \
  --baseline 4019cc6 --runs 3 --evaluations 12 --cases 4000 \
  --output docs/research/score-readers-2026-10-11.jsonl
```

Every sample runs in a fresh process against immutable source and score fixtures.
Dataset, artifacts, history, and report outputs use isolated temporary paths.
Input/source hashes are checked after measurement. Dataset/config/code identities
are fixed, and recorder/dashboard clocks remain fixed to 2026-10-10 UTC, matching
the [previous experiment](score-readers-2026-10-10.md). No tests or other benchmark
workloads run concurrently. `PYTHONHASHSEED=0`; the four thread settings above are
applied to every worker.

The baseline uses the bounded policy; the updated source is measured under all
three existing policies: bounded (`parallel="none", low_memory=True`), auto low
memory (`parallel="auto", low_memory=True`), and auto (`parallel="auto",
low_memory=False`). Peak RSS is total process `resource.ru_maxrss`, including
imports and decoding. Wall time covers the entry point and comparison payload
extraction.

The renamed metric **`full_load_calls` counts `load_scores` invocations**. It
excludes projected Parquet scans and batches used by selection. A selection count
of zero still involves reading metadata and selected score rows. Historical
samples, including their `full_decodes` field, remain unchanged.

## Fixtures and equivalence

Fixtures and their hashes match the previous experiment. Both selection archives
have 12 files, four methods, one hourly slice, and an 8,196-byte repeated feature
JSON value. Many evaluations has **192,000 rows**, selecting the newest 16,000;
mostly incompatible has **176,320 rows**, selecting 320 from the only compatible
evaluation. Their compressed sizes are 639,492 and 595,454 bytes respectively.

Standalone dashboard and report share a **384,000-row**, 644,196-byte archive:
12 files, four methods, two hourly slices (`temp_c` and `pop`), and three quantile
levels (0.1, 0.5, 0.9). Temperature challenger errors alternate around reference
error to keep sequential wealth finite; PoP predictions are 0.5 with alternating
binary truths. This retains the existing fixture that avoids an unrelated
baseline `math.exp(log_e)` overflow.

All 48 samples agree on their comparison payload within each workload across
sources, policies, and repetitions. Full report comparisons include every
leaderboard Markdown table and dashboard chart. Table rows are sorted for
comparison because tied aggregate errors have undefined display order; headers
and values remain included. Raw rendered hashes are saved separately.

The benchmark uses `mcs`. Deterministic regressions separately verify equivalent
reused and independently collected summaries under both `mcs` and `legacy`,
including pooled daily gates, synthetic scores, and empty archives. They assert
one full load per file, release before the next decode, and rendering after
source files disappear. `seq_mcs` retains two loads per file; a two-file regression
confirms that its dashboard reflects later persisted wealth, preserves store
cursors, and leaves settled report tables and ledgers unchanged on further runs.
Complete fitting and real deployment archives are outside these measurements.

## Measurements

Python 3.13.14, Polars 1.42.1, macOS-27.0.1-arm64-arm-64bit-Mach-O.

Baseline source SHA-256: `12adf9728c3a0a2a1fa129ac7ccc469defba2b37c2f36bed42d0c73174fbaf9b`.
Updated source SHA-256: `80d5ef6a67e040a61909d3adc358bd7270baaf1e0c3b584ab02686f4077aca3b`.
Runner SHA-256: `4d2a5b0cd89b47ac186dd1652691964cc213304185689f098c90243147f10861`.
Combined fixture SHA-256: `484976b1b3420b3241e43065d60cafb0dc55c5da176db8e8c05d7c88aa8e24aa`.

Ranges are minimum–maximum across three fresh processes. MiB is $2^{20}$ bytes.

| Workload | Source / policy | Median RSS (MiB) | RSS range | Median time (s) | Time range | Full load calls |
|---|---|---:|---|---:|---|---:|
| many evaluations | Baseline / bounded | 260.80 | 260.69–260.83 | 0.1989 | 0.1987–0.2167 | 0 |
| many evaluations | Updated / bounded | 261.56 | 260.78–262.72 | 0.2018 | 0.1988–0.2057 | 0 |
| many evaluations | Updated / auto low memory | 260.33 | 258.88–260.67 | 0.1982 | 0.1982–0.2050 | 0 |
| many evaluations | Updated / auto | 260.11 | 259.22–261.03 | 0.2013 | 0.1986–0.2055 | 0 |
| mostly incompatible | Baseline / bounded | 151.45 | 151.23–151.67 | 0.0316 | 0.0313–0.0343 | 0 |
| mostly incompatible | Updated / bounded | 151.34 | 151.16–151.70 | 0.0316 | 0.0314–0.0319 | 0 |
| mostly incompatible | Updated / auto low memory | 151.59 | 151.36–153.17 | 0.0320 | 0.0316–0.0328 | 0 |
| mostly incompatible | Updated / auto | 151.16 | 151.14–151.41 | 0.0313 | 0.0312–0.0360 | 0 |
| dashboard | Baseline / bounded | 293.92 | 293.52–295.23 | 3.8834 | 3.8663–4.0352 | 12 |
| dashboard | Updated / bounded | 291.14 | 290.64–293.09 | 3.8880 | 3.8773–3.9890 | 12 |
| dashboard | Updated / auto low memory | 294.16 | 293.20–296.34 | 4.0042 | 3.8718–4.0835 | 12 |
| dashboard | Updated / auto | 293.94 | 292.19–294.55 | 3.8818 | 3.8602–4.0339 | 12 |
| report | Baseline / bounded | 305.72 | 304.64–307.30 | 6.7075 | 6.7002–6.7184 | 24 |
| report | Updated / bounded | 304.77 | 302.92–305.95 | 4.8193 | 4.8137–4.9282 | 12 |
| report | Updated / auto low memory | 304.77 | 304.48–305.45 | 4.8205 | 4.8100–4.9434 | 12 |
| report | Updated / auto | 304.41 | 302.27–305.89 | 4.9734 | 4.7732–6.0642 | 12 |

The bounded report's runtime ranges do not overlap, and its median improves by
28.1%. Report RSS ranges overlap, so this does not establish a general memory
improvement. Standalone dashboard still loads each file once and has nearly the
same median runtime. Selection runtime ranges overlap, with small RSS variation.
Neither automatic policy establishes the required clear runtime improvement
across workloads; bounded decoding remains in place.

[Raw JSONL samples](score-readers-2026-10-11.jsonl) and
[aggregated JSON](score-readers-2026-10-11.summary.json) contain exact source,
runner, fixture, thread, payload, and load-count evidence. The previous dated
artifacts remain historical; their different baseline makes absolute timings
unsuitable for measuring this change. Three samples describe the observed spread.

## Correctness, validation, and rollout

Score selection now invalidates every affected evaluation across all slices and
reconsiders the newest remaining complete evaluation within the same catalog.
That finite loop continues only while the invalid set grows. Unknown catalog
failures or no-progress materialization failures consume at most three fresh
catalog attempts, then fail closed. Three materialization-time prunes can
therefore fall back to an older complete evaluation without exhausting that
outer retry budget. See [model selection](../methods/model-selection.md).

PIT and reliability failures are isolated independently from core scoring and
from each other. Valid leaderboards, winners, and other diagnostics survive;
failed diagnostics use local amber warnings without making live scores unusable.
For example, ragged quantile grids can fail PIT while valid CRPS/coverage and
PoP reliability remain visible. See [dashboard diagnostics](../dashboard.md).

The full suite passes **1,400 tests**, with **13 skips** and **91.78% coverage**.
All `CONTRIBUTING.md` gates pass: lockfile/host checks, Ruff, both type checkers,
Semgrep tests/scans, dependency hygiene, packaging (10/10), complexity (maximum
27 against ceiling 27), and strict documentation build. Regressions include
fallback progress/exhaustion, whole-evaluation exclusion, independent diagnostic
failures, report reuse, final sequential wealth, and cached policy-neutral store
loading.

Deploying the changed source rotates the code fingerprint. Before rollout,
refresh compatible evidence with `backtest --source live` and `report`; existing
forecast artifacts retain their last-good behavior during that transition.
Deployment installation and live evidence refresh are outside this implementation.
