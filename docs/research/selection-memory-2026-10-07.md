# Selection memory benchmark (2026-10-07)

Complete `select_methods` runs use less peak process memory on both fixtures
after the score-scanning and evidence fixes. Median peak RSS fell **10.7%** on
many compatible evaluations and **12.0%** on mostly incompatible evaluations.
Selection runtime increased on the many-evaluation fixture; the mostly
incompatible fixture's runtimes were similar. This change meets the memory
target while adding metadata validation and scanning work.

## Scope and reproduction

*Implemented in: `scripts/benchmark_selection_memory.py`*

Baseline: PR #46's original head,
`a4ec76de15dc4206ff48cf3a46988451cf7228a5`. Updated source:
`9511279a0d3a131d70080d1630b0ae4147d67432`.

```bash
uv run python scripts/benchmark_selection_memory.py \
  --baseline a4ec76d --runs 3 --evaluations 12 --cases 4000
```

Three fresh subprocesses per variant and scenario use Python 3.13.14 and
Polars 1.42.1 on macOS 27.0 arm64. No repository tests ran during these final
measurements. Library thread counts use their defaults. Each worker imports
its designated source tree and runs selection, promotion gates, live-history
lookup, and release creation. Process peak RSS uses `resource.ru_maxrss` and
includes imports and score loading. Runtime measures `select_methods` itself.

Fixtures use 22 score columns, four methods, one hourly slice, and an 8 KiB
repeated feature-set JSON value. The many-evaluation fixture has 12 compatible
evaluations of 4,000 cases per method: **192,000 rows**, of which only the newest
16,000 need materialization. The mostly-incompatible fixture has 11 obsolete
configuration identities with 16,000 rows each, plus one compatible evaluation
of 80 cases per method: **176,320 rows** total, with 320 compatible rows.

Dataset/config/code identities are fixed so source changes do not invalidate
the comparison. Each worker uses an empty temporary release ledger and no live
history. The runner checks that all selection payload hashes agree within each
scenario. Deterministic/probabilistic metrics, pooled daily gates, retention,
and live replacement have separate regression coverage.

## Measurements

Ranges are the minimum and maximum of three runs. MiB means $2^{20}$ bytes.

| Fixture | Variant | Median peak RSS (MiB) | RSS range (MiB) | Median runtime (s) | Runtime range (s) |
|---|---|---:|---|---:|---|
| Many evaluations | Original PR #46 | 462.62 | 457.25–467.14 | 0.2277 | 0.2154–0.2344 |
| Many evaluations | Updated | 413.19 | 412.33–414.84 | 0.3881 | 0.3832–0.5077 |
| Mostly incompatible | Original PR #46 | 201.20 | 196.50–201.77 | 0.0596 | 0.0520–0.0602 |
| Mostly incompatible | Updated | 177.08 | 176.09–178.50 | 0.0578 | 0.0502–0.0645 |

Median runtime increased by 0.160 s (70.4%) on many evaluations. It fell by
0.002 s (3.1%) on mostly incompatible evaluations, whose runtime ranges overlap.
See [raw samples and payload hashes](selection-memory-2026-10-07.jsonl).

These measurements cover complete selection on the stated fixtures. Forecast
fitting, a populated release ledger, live-history verification, and complete
`predict` or `report` runs need workload-specific measurement. Three runs show
the observed spread; they do not establish a universal percentage saving.
Memory thresholds remain outside CI.

Polars column projection avoids numeric and string-view allocations while
shared string payloads may stay alive through retained frames. The exact gain
depends on column count, row count, and buffer lifetimes; see the
[StringView implementation reference](https://pola.rs/posts/polars-string-type/).

## Validation and rollout

The final suite passes **1,324 tests**, with 13 skips and **91.14% coverage**.
All gates in `CONTRIBUTING.md` pass, including both type checkers, Semgrep,
lockfile, dependencies, packaging, complexity ≤27, and the strict documentation
build. The final full-suite rerun limits OpenMP and OpenBLAS to one thread;
the preceding default-thread run also passed (before the last race regression).

The behavior changes are intentional: pins report their own statistics,
replacement references pass a quality guard, blocked live verdicts remain
visible, conflicting evaluations are excluded, and schema 6 adds actual
minutely provenance. Unaffected statistical calculations remain covered.
Deployment requires fresh compatible live backtest/report evidence because
source changes alter code fingerprints. Existing artifacts remain intact;
attributed minutely live evidence accumulates prospectively.
