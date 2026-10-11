# Score-reader benchmark (2026-10-10)

The final implementation retains `parallel="none", low_memory=True`.
Automatic decoding must stay at or below the reviewed head's median peak RSS
in every workload and show a clear runtime improvement before adoption.

*Implemented in: `scripts/benchmark_score_readers.py`, `backtest/scores.py`,
`serve/score_scan.py`, `dashboard/score_summary.py`*

## Source and reproduction

The reviewed baseline is `6a4933ff81bda0fd89cb802f391a11f7a8d2fe0e`.
The final working-tree source is saved as a
[source patch](score-readers-2026-10-10.source.patch) against that revision.
Source hashes cover filenames and bytes under `src`, excluding Python caches
and Finder metadata. The raw samples also identify the exact benchmark runner.
Applying the source patch to the baseline reconstructs the measured source.

```bash
POLARS_MAX_THREADS=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run python scripts/benchmark_score_readers.py \
  --baseline 6a4933f --runs 3 --evaluations 12 --cases 4000 \
  --output docs/research/score-readers-2026-10-10.jsonl
```

Each measurement uses a fresh subprocess. The baseline uses its conservative
decoder policy; the final source is measured with all three pairs:

| Policy | `parallel` | `low_memory` |
|---|---|---|
| bounded | `none` | `True` |
| auto low memory | `auto` | `True` |
| auto | `auto` | `False` |

All workers use identical immutable inputs and isolated temporary dataset,
artifact, history, and output paths. Score files are read-only, and input/source
hashes are checked again after the measurements. Fixed dataset/config/code
identities prevent source differences from invalidating evidence. Recorder and
dashboard clocks are fixed to 2026-10-10 UTC; the benchmark still runs their
normal recording and rendering code. No live data or artifacts are used.

Thread settings are `POLARS_MAX_THREADS=2`, `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`; `PYTHONHASHSEED=0`.
Peak RSS is total process `resource.ru_maxrss`, including imports and decoding.
Wall time covers each entry point and extraction of its comparison payload.
No repository tests or other benchmark workloads run concurrently.

## Fixtures and equivalence

Both existing selection fixtures retain their original inputs: 22 score columns,
four methods, one hourly slice, and an 8,196-byte repeated feature-set JSON value.
The many-evaluation fixture contains 12 compatible evaluations with 4,000 cases
per method: **192,000 rows**, selecting the newest 16,000. The mostly-incompatible
fixture contains 11 obsolete configurations plus one compatible evaluation with
80 cases per method: **176,320 rows**, selecting 320.

Dashboard and full report share a separate **384,000-row** archive: 12 files,
four methods, and two hourly slices (`temp_c` and `pop`), with 4,000 cases per
method and three quantile levels (0.1, 0.5, 0.9). PoP predictions are 0.5 with
alternating binary truths. Temperature challenger errors alternate around the
reference error, keeping sequential wealth finite. Using the selection fixture's
constant challenger advantage for full report instead triggers the reviewed
head's existing `math.exp(log_e)` overflow; the reporting fixture avoids that
unrelated failure without replacing any report calculations.

All selection payload hashes agree within each workload across source variants,
decoder policies, and repetitions. Dashboard embedded data agrees exactly.
Full report comparisons cover every leaderboard Markdown table and every
dashboard chart. Markdown table rows are sorted for comparison because aggregate
MAE ties have undefined display order; all values and headers remain included.
Raw rendered hashes are saved separately. Complete forecast fitting, a populated
release ledger, live-history verification, and real deployment archives are
outside these score-reader measurements.

## Measurements

Measured with Python 3.13.14, Polars 1.42.1, and macOS-27.0.1-arm64-arm-64bit-Mach-O.

Baseline source SHA-256: `ad6f0940ed6c2cda08a5a3fbfbc1bcd13e64f59dcb93993b67056872247ec834`.
Final source SHA-256: `3598f1ffe12f751085cb518cb4b46a149fbf4b3ab63c245ac0623ae3af9f6a73`.
Runner SHA-256: `d53b82f115b52732176447537872b49f80e2319e1fbab5044cc6fbcb9805d9e0`.

Ranges are minimum–maximum across three fresh processes. MiB is $2^{20}$ bytes.

| Workload | Source / policy | Median RSS (MiB) | RSS range | Median time (s) | Time range |
|---|---|---:|---|---:|---|
| many evaluations | Reviewed / bounded | 260.41 | 260.08–262.19 | 0.2088 | 0.2055–0.2091 |
| many evaluations | Final / bounded | 260.03 | 259.94–261.88 | 0.2083 | 0.2016–0.2145 |
| many evaluations | Final / auto low memory | 260.36 | 260.03–260.48 | 0.2102 | 0.2015–0.2312 |
| many evaluations | Final / auto | 259.83 | 259.61–261.00 | 0.2112 | 0.2056–0.2116 |
| mostly incompatible | Reviewed / bounded | 151.19 | 150.92–151.97 | 0.0317 | 0.0313–0.0319 |
| mostly incompatible | Final / bounded | 151.62 | 151.12–153.03 | 0.0324 | 0.0318–0.0329 |
| mostly incompatible | Final / auto low memory | 151.53 | 150.92–151.59 | 0.0324 | 0.0319–0.0328 |
| mostly incompatible | Final / auto | 151.30 | 151.25–151.58 | 0.0334 | 0.0318–0.0384 |
| dashboard | Reviewed / bounded | 291.59 | 290.39–296.36 | 4.5693 | 4.4921–4.5883 |
| dashboard | Final / bounded | 295.14 | 294.44–295.14 | 3.9837 | 3.9668–3.9863 |
| dashboard | Final / auto low memory | 293.61 | 291.77–294.05 | 4.0096 | 3.9686–4.0164 |
| dashboard | Final / auto | 294.89 | 292.97–295.41 | 3.9851 | 3.9731–3.9922 |
| report | Reviewed / bounded | 307.27 | 304.64–311.73 | 7.4682 | 7.4467–7.5184 |
| report | Final / bounded | 306.47 | 304.39–309.08 | 7.0911 | 6.9228–7.0954 |
| report | Final / auto low memory | 305.19 | 293.84–306.34 | 6.9337 | 6.9189–7.0063 |
| report | Final / auto | 304.98 | 293.98–305.92 | 6.9381 | 6.9355–7.1181 |

Neither automatic policy meets the adoption rule: its median RSS exceeds the reviewed baseline on at least one selection workload, and selection runtime ranges overlap. Conservative settings remain in place.

The conservative dashboard has 60 → 12 full decodes across 12 files, with median runtime lower by **12.8%** on this fixture. The conservative report has 72 → 24 full decodes across 12 files, with median runtime lower by **5.0%** on this fixture.

Selection uses projected scans and batches rather than `load_scores`, so its full-loader decode count is zero; it still reads Parquet metadata and selected rows. Small selection RSS differences are reported as measured, without claiming a universal memory saving.


Raw samples, exact source/runner and fixture hashes, compressed sizes, thread
settings, decode counts, and payload checks are saved in
[JSONL](score-readers-2026-10-10.jsonl) and
[aggregated JSON](score-readers-2026-10-10.summary.json).
The [`9511279` measurements](selection-memory-2026-10-07.md) remain historical:
they use default threads and a different baseline, so their absolute RSS numbers
are not directly comparable to this experiment. Three samples describe the
observed spread; memory thresholds remain outside CI.

## Validation and rollout

The final suite passes **1,356 tests**, with **13 skips** and **91.52% coverage**.
Every `CONTRIBUTING.md` gate passes: lockfile/source-host checks, Ruff, both
type checkers, Semgrep tests/scans, dependency hygiene, packaging (10/10),
complexity (maximum 27 against the ceiling of 27), and the strict documentation
build. Regression coverage includes loader compatibility, maintenance contention
and cleanup, one-pass dashboard summaries, normalized semantics, split and
conflicting evaluations, prune/replacement races, and retry exhaustion.

Deployment installs the updated launchd wrappers and refreshes compatible live
backtest/report evidence because the source fingerprint changes. The existing
last-good forecast behavior remains in place during that transition. This work
changes repository files only; it does not install scripts or refresh live data.
