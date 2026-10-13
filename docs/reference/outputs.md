# Outputs and on-disk layout

What each command writes, where, and which files are safe to delete.

Three roots, configured by `[dataset] dir`, `[artifacts] dir`, and
`[reports] dir`:

| Root | Default | Contains | Regenerable? |
|---|---|---|---|
| `data/` | `data` | parquet: truth, matrices, scores, history | mostly — see below |
| `artifacts/` | `artifacts` | JSON and parquet: releases, ledgers, studies | **partly not** |
| `reports/` | `reports` | markdown and HTML for humans | entirely |

The short version: **`reports/` is disposable, `data/` is rebuildable from the
two SQLite files, and `artifacts/` contains history that cannot be recovered.**

---

## `data/` — the dataset layer

Written by `build-dataset` unless noted.

| File | Contents |
|---|---|
| `truth_minute.parquet` | QC'd per-minute station observations |
| `truth_hourly.parquet` | hourly truth, **both** `_inst` and `_mean` semantics |
| `truth_daily.parquet` | local-day extremes and sums |
| `forecasts_long.parquet` | canonical long frame of hourly provider forecasts |
| `daily_long.parquet` | canonical long frame of daily provider forecasts |
| `minutely_long.parquet` | native provider minutely nowcasts |
| `hourly_matrix_live.parquet` | the supervised hourly matrix, **live** provenance |
| `hourly_matrix_synthetic.parquet` | the supervised hourly matrix, **backfilled** |
| `daily_matrix_live.parquet` / `_synthetic.parquet` | supervised daily matrices |
| `ensembles.parquet` | written by `ingest-ensembles`; ensemble spread features |
| `manifest.json` | per-file SHA-256 plus the **dataset fingerprint** |
| `predict_history.parquet` | written by `predict` and `publish`; every accepted served forecast |
| `runs.parquet` | parsed command invocations and maintenance/recovery child stages (`record_kind`, `parent_run_id`, `failure_kind`); older ledgers remain readable; pruned to 90 days / 50k rows |
| `scores/` | written by `backtest` |
| `served_forecasts/` | archived served documents |

!!! danger "`predict_history.parquet` is not regenerable"
    It is the record of what the system actually served, and it is the only input
    to [self-verification](../methods/verification.md#8-self-verification). Delete
    it and you permanently lose the ability to measure serving-path drift for
    that period. Back it up; do not point `[predict] history_path` at a temp
    directory.

New hourly/daily documents carry release IDs only for finite normalized values,
including zero. Null predictions contribute no release to document readiness or
the top-level release list. Cold-start blends remain unattributed, and rainfall
floors that fill their null fields retain the evidence diagnosis. Archived
documents replay verbatim; historical forecast files are not rewritten. Null
points also omit quantiles and their source; source labels require usable
emitted quantiles. With compatible evidence, unavailable fitting for a promoted
selection is diagnosed separately from rainfall attribution loss. Reasons name
the affected product/variable/semantics and distinguish pending truth on already-issued
rows from finite truth on later-issued rows, missing matrices, missing forecast
columns, and absent finite truth. Polling helps only as live issuance advances;
repeating a fixed historical `--now` never admits future evidence. A fitted
variable with unavailable prediction inputs is not mislabeled as an unavailable
fit, and a surviving attributed finite value keeps the document ready.

### The provenance wall is in the filename

`_live` and `_synthetic` matrices are separate files. This is deliberate: the
provenance is a filesystem property, not merely a runtime check, so mixing them
requires deleting a guard rather than forgetting one. Attempting to pool raises
`MixedProvenanceError`. See
[Methods: verification §7](../methods/verification.md#7-the-provenance-wall).

### `data/scores/`

`backtest.scores.load_scores(path)` preserves the stored types, including
String columns, so callers can concatenate scores vertically or join to
String keys. `load_scores(path, compact=True)` encodes repeated text as
categories in memory. Report, dashboard, and selection readers use compact
decoding; numeric scores, per-case quantiles, and the on-disk schema stay
unchanged. Both modes reject mixed provenance unless `allow_mixed=True`.
`scan_score_file(path)` defaults to compact categories; passing `compact=False`
preserves stored column types, matching the eager loader's default.

Readers use `parallel="none", low_memory=True` and streaming collection.
Serving partitions selected evidence in batches of `SCORE_BATCH_ROWS = 65_536`
rows. The dashboard collects each archived file once, sequentially, retaining
leaderboards, winners, calibration bins, provenance, and fold metadata, then
releases its full frame before loading the next file. Rendering uses these
summaries without reopening score files. With `mcs` or `legacy`, a complete
`report` shares these summaries with its dashboard and loads each file once.
With `seq_mcs`, the dashboard retains its independent collection after report
persists final wealth, for two full loads per file overall. PIT/reliability
failures produce local dashboard warnings and retain valid scoring results.
Report skips missing files as absence and forwards genuine score-read faults
so the supplied-summary dashboard retains unreadable-score and unreadable-artifact
warnings. It finishes available outputs and returns nonzero if a genuine fault
occurred or no file loaded successfully. This does not add score reads or change
sequential final-wealth readback.

```
scores_{product}_{kind}_{window}_{evaluation_id}.parquet
```

| Component | Values |
|---|---|
| `product` | `hourly`, `daily`, `minutely` |
| `kind` | `live`, `synthetic` |
| `window` | `expanding`, `rolling` |
| `evaluation_id` | identity of the backtest run |

The scores frame is deliberately **long and dumb**: one row per scored case per
method, and no winner declared. Every leaderboard is a downstream `group_by`,
which is what lets `report` recompute leaderboards under new statistical rules
without re-running a single fit.

These files are the bulk of the disk footprint. `prune-scores` manages them.
It does not repair corruption or remove uncataloged evidence. Follow the
[manual score recovery procedure](cli.md#recovering-from-a-damaged-score-file)
when a read fault persists.

---

## `artifacts/` — evidence and identity

| Path | Written by | Contents |
|---|---|---|
| `releases/*.json` | `backtest`, `predict`, `publish` | immutable promoted `ModelRelease` records — the serving boundary |
| `active_release.json` | `predict`, `publish` | atomic record of the local output: `release_ids` (possibly empty) and `output_path`; older `release_id` pointers remain readable |
| `eprocess/{product}_{kind}.json` | `report` | accumulated betting wealth per candidate/reference pair |
| `history/*.parquet` | `report` | ten append-only ledgers (below) |
| `alignment.json` | `alignment` | measured truth semantics per provider |
| `truth_qc.json` | `truth-qc` | neighbour verdicts, shield fit, and optional UTC ISO `generated_at` refresh timestamp (schema v3) |
| `drift.json` | `report` | provider drift alarms |
| `observability/` | `predict`, `publish` | per-serve snapshots |
| `auto-restore.json` | `publish` | last automatic-recovery signature and attempt time |
| `latest_publish_attempt.json` | `publish` | latest candidate status, reason, hold decision, and locally served document |
| `auto-restore.log` | `recover` | detached recovery stdout and stderr |

A handled truth-QC failure during CLI maintenance preserves the previous
`truth_qc.json` and still permits report/dashboard/catalog refresh. The failed
stage is recorded in `data/runs.parquet`; maintenance returns its first nonzero
stage result even when report finishes. A refreshed report may therefore use an
older QC verdict. The dashboard shows its refresh time and age, plus a failed
refresh warning. The panel becomes amber after a failed refresh,
when older than 48 hours, or when freshness is unknown; latched drift stays red.
Legacy `drift_verdict.as_of_date` is displayed as date-only evidence, becoming
stale only when its latest possible refresh time is beyond 48 hours. File mtime
is not evidence of refresh. Chips report independent conditions, so a quiet
verdict remains green while an unknown refresh time is amber. One central amber
alert uses the same assessment. Run records add optional `error_detail`, a
whitespace-normalized message capped at 400 characters; older ledgers null-fill
it, and `error` retains the exception class. No historical artifacts need
migration or rewriting.
Warnings display class and detail together when both exist. Maintenance's
parent run retains its options, while the linked QC stage records `{"days": N}`;
unrelated child stages do not inherit the parent's private failure kind.
Child stage arguments contain public options only. Failure kind and handled
detail are passed explicitly; an active exception supplies its class and message
in preference to a handled detail. Details are normalized when persisted.
Build/backtest failures and unexpected exceptions still stop
maintenance. Source changes alter the code fingerprint: generate fresh live
backtest and report evidence with the updated code before rollout.

### Benchmark scratch outputs

The fitting and score-reader benchmark scripts default to unique UTC
timestamp-and-UUID JSONLs under the repository's ignored `reports/benchmarks/`.
Validated results include a `.summary.json`; each written path is printed.
Explicit relative destinations use the caller's working directory.

Existing primary sample or summary destinations are refused
before measuring (exit `2`), as are unwritable parents discovered by a temporary
write/flush probe. Mismatch payload destinations are reserved only during
recovery; an existing primary payload sidecar does not block a run.
Exclusive creation protects late collisions too. Publication
failures exit `1`, retain possibly incomplete destination files, and attempt one
rescue to unique `*-recovery-...` scratch paths. If rescue fails, the available
sample JSONL is emitted to stderr. Nothing deletes destination files during
failure handling.

Before Git or workers, each runner exclusively creates and prints a unique
`*-journal-*.jsonl` in the repository's ignored scratch directory. Each completed
sample is appended and flushed outside measured worker timings. Journals survive
process termination after an acknowledged checkpoint; flush without `fsync`
does not promise power-loss durability. Interrupted writes may leave a partial
last line. Journals are retained after every run, hold raw possibly unvalidated
samples, and never include a summary. Journal preparation failures exit `2`;
checkpoint failures stop measurement and trigger recovery.

Worker/validation failures attempt one rescue of collected samples before
propagating their original exception and traceback. Secondary failures during
serialization, rescue, reporting, journal closure, or another interruption do
not displace the original failure. The printed journal path identifies retained
samples even when recovery cannot complete. Such results omit an unvalidated summary; score mismatches
also retain `.payloads.json` diagnostics. Recovery files may describe an
incomplete experiment and never imply success. Inspect retained destinations
before removing them manually, and choose a new output path for another run.
Historical research memos describe their original runners; current commands use
these scratch defaults without replacing the committed evidence. See
[CLI output semantics](cli.md#benchmark-output-safety-and-recovery).

### The history ledgers

`artifacts/history/` holds ten append-only parquet ledgers driving dashboard zone
H: `quality`, `churn`, `verdicts`, `served_quality`, `eprocess_wealth`,
`pipeline`, `provider_health`, `build_funnel`, `changes`, `evaluations`.

Append-only is the point — they record what was true *at the time*, so a
regression is visible as a trend rather than only as a current value.

!!! danger "`artifacts/` holds two things that cannot be rebuilt"
    **`eprocess/`** — a betting e-process is a sequential accumulation. A bet,
    once made, is never revised, and deleting the store discards all accumulated
    evidence; the gate restarts from zero wealth and re-earns significance from
    scratch. (It resets deliberately on a config or code-version change, because
    a different implementation is a different null hypothesis — but *only* then.)

    **`history/`** — append-only ledgers. Deleting them erases the past.

    `releases/`, `active_release.json`, and `alignment.json` can be regenerated
    by re-running their pipeline commands; the two above cannot.

---

## `reports/` — for humans

Everything here is safe to delete. The producer for each file is explicit
below; `report` does not regenerate the independently run alignment or truth-QC
studies.

| File | Contents |
|---|---|
| `dashboard.html` | the nine-zone offline operator console |
| `leaderboard_{product}_{source}_{window}_{hash}.md` | per-slice leaderboards |
| `drift.md` | provider drift alarms |
| `correlation_{variable}.md` | provider error correlation and $k_{\text{eff}}$ |
| `alignment.md` | truth-semantics study, written by `alignment` |
| `truth_qc.md` | neighbour cross-check and shield fit, written by `truth-qc` |
| `pipeline_health.md` | end-to-end freshness and funnel |
| `selection_churn.md` | how often the served method changes |

Six files have stable names and are overwritten. Leaderboards are
**hash-suffixed per evaluation run**, so they accumulate — dozens of files is
normal. `prune-scores` does **not** touch them (it deletes only scores
parquet files); delete old leaderboard markdown by hand if the directory
bothers you.

`dashboard.html` is fully self-contained: CSS, vendored Chart.js, and the data
payload are inlined, so it opens from `file://` with no server and can be emailed
as a single file. See [Operator dashboard](../dashboard.md).

---

## Retention: `prune-scores`

```bash
grounded-weather-forecast prune-scores --dry-run   # always look first
grounded-weather-forecast prune-scores
```

Kept:

- the **newest three scores files per group** (product × kind × window);
- any evaluation referenced by the **active release**, regardless of age;
- any evaluation **referenced by a release within the last 7 days**;
- **all catalog rows** — the ledgers survive; only the bulky per-case scores go.

If `active_release.json` is missing or malformed after an upgrade, the newest
readable release is conservatively treated as active. Thus an evaluation still
backing current serving is never deleted, even when that release is older than
the rollback horizon.

---

## Backup priority

If you back up only some of this:

1. **The two upstream SQLite files.** Everything in `data/` derives from them,
   and the forecast archive in particular *cannot be recreated* — you cannot ask
   a provider what it predicted last Tuesday.
2. **`artifacts/history/` and `artifacts/eprocess/`.** Append-only and sequential;
   not rebuildable.
3. **`data/predict_history.parquet`.** The record of what you served.
4. `config.toml`.

Everything else is a function of those.
