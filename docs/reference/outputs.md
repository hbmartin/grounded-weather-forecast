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
| `runs.parquet` | parsed command invocations and maintenance/recovery child stages (`record_kind`, `parent_run_id`, `failure_kind`, nullable `failure_details_json` component diagnostics); older ledgers remain readable; pruned to 90 days / 50k rows |
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
| `releases/*.json` | `backtest` | immutable `ModelRelease` content; managed content requires an approval event |
| `releases/approvals/*.json` | `backtest` | committed sequence, actual `approved_at`, and nullable historical `visible_at` floor |
| `active_release.json` | `predict`, `publish` | atomic record of the local output: `release_ids` (possibly empty) and `output_path`; older `release_id` pointers remain readable |
| `eprocess/{product}_{kind}.json` | `report` | accumulated betting wealth per candidate/reference pair |
| `history/*.parquet` | `report` | eleven append-only ledgers (below) |
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

`artifacts/history/` holds eleven append-only parquet ledgers driving dashboard zones
F and H: `quality`, `churn`, `verdicts`, `served_quality`, `eprocess_wealth`,
`pipeline`, `provider_health`, `build_funnel`, `changes`, `evaluations`, `live_gate`.

Append-only is the point — they record what was true *at the time*, so a
regression is visible as a trend rather than only as a current value.

!!! danger "Preserve irreplaceable evidence in `artifacts/`"
    **`eprocess/`** — a betting e-process is a sequential accumulation. A bet,
    once made, is never revised, and deleting the store discards all accumulated
    evidence; the gate restarts from zero wealth and re-earns significance from
    scratch. (It resets deliberately on a config or code-version change, because
    a different implementation is a different null hypothesis — but *only* then.)

    **`history/`** — append-only ledgers. Deleting them erases the past.

    `releases/` contains immutable content and append-only approval history;
    keep it for historical replay. `active_release.json` describes the locally
    served document. Its references also protect legacy score evidence.
    Approval events add nullable `visible_at` for conservative historical visibility;
    current decisions follow committed sequence. Selection markers
    `legacy_context_missing` preserve exact-dataset legacy compatibility per slice.
    `history/live_gate.parquet` records gate diagnostics separately from approval
    identity. Churn history adds explicit approval clocks and sequences; older
    rows are normalized in memory without rewriting their recorded values.

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

If `active_release.json` is genuinely absent, the newest current approval under
sequence ordering and recent served-document references are protected together.
A corrupt pointer suspends deletion. Explicit empty/degraded pointers do not
trigger the newest-approval fallback.

If any release JSON file is unreadable or is not an object, all published scores
are kept and the diagnostic names the affected paths. A non-active file can
still protect evidence, so pruning resumes only after those files can be
verified. This safeguard also applies to `--dry-run` and leaves the existing
temporary-recovery and exit-code behavior unchanged.


Approval verification uses typed internal path/category/detail diagnostics with the
existing `issues` path interface. Unreadable releases/events, invalid metadata,
denied listings, corrupt active pointers and unreadable served archives suspend
published-score deletion through `PruneResult.skipped`, including dry runs. These
conditions do not authorize excluded releases. Genuine pointer absence alone uses
the newest current approval fallback. Pre-#79 readers require a compatibility
backport before reading events with `visible_at`; no artifacts are migrated.

`PruneResult.verification` additively carries a typed clear/suspended result and
recording time. The CLI records it as informational `retention_verification`
components inside the existing `failure_details_json` run field, with exit 0 and
`retention_clear` or `retention_suspended`. Details are bounded to 400 characters
per component. Optional `path` preserves full path identity outside that bound;
optional `recorded_at` records completion of the check. Both are additive fields
in diagnostic JSON entries; older entries remain readable. Coordination failures
retain their original exit/failure metadata.
No new ledger is created. Suspended verification publishes an immutable diagnostic audit at `artifacts/history/pruning_verifications/<check-id>.jsonl`. This is operational telemetry, separate from scientific evidence and approvals. The dashboard's published-score protection
panel and pipeline report distinguish the last recorded check (time and age)
from the latest attempt. Completed verification survives later exceptions in its
one existing invocation/stage row. Failed/unverified attempts cannot erase a
suspension; legacy rows do not establish a clean check. The dashboard does not
rescan served archives, and a later clear check supersedes a prior suspension.
Pruning still freshly verifies all protection requirements on every invocation.

Each new check has a unique `check_id`. Additive diagnostic fields `audit_path`,
`issue_count`, `omitted_count`, `audit_error`, and nullable boolean `dry_run` accompany that identity. Nullable `category` and `reason_code` describe each issue; nullable `audit_reason_code` identifies the failed publication operation and nullable `audit_cause_code` records its known underlying cause. Missing legacy cause metadata remains null. Renderers accept only known fixed explanations and never publish raw exception prose. Unknown or missing codes use neutral diagnostic-reference wording. A check
records one summary and at most 20 issue samples, each with the existing
400-character explanatory bound; full stored sample paths remain separate.
Legacy rows remain readable and their rendering is also bounded. Each audit is
UTF-8 JSON Lines schema 1: a header (`kind=pruning_verification`, check ID,
recording time, dry-run flag), one `kind=issue` record per affected path with its
category/detail and nullable reason code, and a final `kind=complete` record carrying `issue_count`.
The staged file is flushed and synchronized before exclusive hard-link publication;
the directory is synchronized afterward. Immediately after its own exclusive link
succeeds, a check records nullable `audit_candidate_path`, relative to the artifacts
directory. This means complete but unconfirmed, not successful publication. Only
after synchronization and lock release succeed does `audit_path` replace that
candidate reference. Catchable post-link failures retain the candidate and failure
reason in the same check summary. Collisions cannot rewrite or adopt an audit.
Unsupported hard links or synchronization failures are explicit best-effort audit
failures. Clear checks need no audit. An unavailable audit is explicit, preserves
samples and suspension, and changes no command exit or cancellation behavior. The
report's additive `audit_state` is `recorded` for confirmed publication,
`unconfirmed` for a linked candidate, `unavailable` for a failed publication without
a candidate, `not_needed` for a clear check, or `not_recorded` for legacy unknown
publication. `audit_candidate_path` is displayed separately from `audit_path`.
These are operational diagnostics; the daily-report schema remains version 1.

Audits are kept for 90 days and while referenced by retained run rows, including
unconfirmed `audit_candidate_path` references. Valid legacy check IDs protect exact
matching complete audits without granting confirmation or rewriting old rows. Pruning
removes only recognized, complete, verifiably older unreferenced files; uncertain
run references, corrupt/future clocks and symlinks prevent their removal. Reference
ledgers are inspected only when expired committed-audit candidates exist. A shared
nonblocking audit lock protects publishers and cleanup. Exact staging names
`.<32 lowercase hex>.jsonl.<16 lowercase hex>.tmp` may be removed only when regular,
unchanged on revalidation and older than 90 days. Fresh, active, unrecognized or
unverifiable files remain. Cleanup
failures are informational. A file-local revalidation or deletion failure keeps that
file while cleanup continues for independent verified candidates. Uncertain run
references remain a global stop, including staging cleanup. Cleanup returns at most
20 notes, reserving space for an abort summary and an explicit omitted count.
Dry runs can record audits and telemetry but delete
neither audits nor scores. Dashboard rendering never opens the audit archive.

The report adds nullable `check_age_minutes` from its reporting cutoff and shows
`contended`/`unknown` attempt states. Contention alone is informational and clears
no warning; unresolved malformed evidence is amber. Equal start/end clocks use
the last persisted row. Multiple checks are grouped by effective progress identity with check-ID fallback
(legacy unidentified rows use recorded time), independently of their parent attempt. Only a later successful
explicitly clear non-dry-run check clears an outstanding pruning execution failure.
Dry runs can update verification independently. Check metadata precedes recorded
arguments when determining mode; unknown legacy mode does not establish recovery.
Reports add nullable `dry_run` and `latest_attempt_dry_run` fields. Identified bounded
progress precedes audit I/O and is replaced for that check after publication,
preserving one check through catchable interruptions without duplicate rows.
No journaling guarantee is made for SIGKILL or power loss. Public path labels and
safe explanations do not replace full internal path/detail evidence.

`RetentionVerification.sample_issues()` supplies the same normalized, at-most-20
issue projection to CLI, dashboard, reports and diagnostic serialization. Empty
paths remain null, empty/missing categories become `verification_invalid`, and
absent reasons remain null. Unknown codes keep their original internal values
and receive neutral public wording. Report explanation summaries are deduplicated
in first-seen order; affected paths and complete audit rows are not collapsed.
The legacy full-audit adapter streams every supplied issue, beyond the sample cap.
Verification stores canonical immutable typed issues. Constructor `details=` and
text-only replacements now mean literal raw text; round-tripping the raw tuple views
does not strip genuine evidence. Labelled legacy input must use the explicit
`RetentionVerification.from_legacy_details(...)` adapter. This replaces automatic
prefix stripping at the constructor boundary; persisted journal details stay raw
and historical artifacts are not rewritten. Accepted lists and compatible issue objects are copied into immutable canonical
tuples; already canonical tuples are reused. `details` exposes canonical raw text;
`formatted_details` adds path/category labels explicitly. Journal reads never
reattach labels to raw text or strip genuine evidence prefixes. Legacy text edits
retain positional metadata, supplied legacy metadata fields override their corresponding
structured projection, and explicit empty details clear issues. Missing metadata
for new positions uses the existing null/default rules. Metadata-only replacement
preserves issues; typed replacement uses the supplied complete issues without
borrowing previous evidence.
A nullable `operation` in publication summaries distinguishes phases such as
`staging_sync`, `directory_sync`, `link`, and `lock_release`; renderers combine
that phase with linked-candidate state. A failure before linking is unavailable,
not an unconfirmed publication. Unknown legacy phases are described neutrally.

`PruneResult.audit_cleanup` additively exposes a typed cleanup result: bounded
issues, abort state, omitted count, recording time, nullable `dry_run`, and
nullable `deleted_audit_count` / `deleted_staging_count`. New invocations record
known counts, incremented only after successful deletion; dry runs record zero.
Legacy missing values remain unknown. Nullable `eligible_audit_count` and
`eligible_staging_count` count only unreferenced, unchanged candidates at final
revalidation. Dry runs report these as “would delete” counts, separately from zero
actual deletions; aborted runs expose verified progress only. Nullable
`cleanup_completed` is derived from the shared typed `CleanupOutcome` classification
(`unknown`, `completed`, `incomplete`). Contradictory legacy flags are rejected;
recorded deletion counts do not establish overall completion. Unknown summaries
state the final-status uncertainty separately from completed deletions.
Default results emit no execution diagnostic; recorded unknown outcomes use
`audit_cleanup_unverified`, never `audit_cleanup_complete`. Cleanup records one
summary even when clean; a failure after deletion preserves completed counts. The immutable diagnostic
batch is captured once, recorded immediately, and reused in the command result.
`AuditCleanupRecord` atomically binds cleanup result, check ID, and diagnostic batch.
`PruneResult.audit_cleanup` and `audit_cleanup_diagnostics` remain readable
compatibility views, and constructor adapters remain available. Metadata-only
copies reuse the record; changed cleanup or verification identity rebinds the batch
once. Explicitly supplied stale batches are rejected. A standalone bound record
retains its check identity when verification is omitted; explicit verification can
change it. Later authoritative batches replace earlier cleanup diagnostics for the
same identified check at journaling, preserving independent checks and unrelated
components. Exceptions without returned outcomes retain captured progress. The CLI
never independently reconstructs the batch. Informational
`retention_audit_cleanup` run components retain original bounded details and
separate paths in the same invocation/stage row. Nullable `operation` records the
failed phase, independently of its reason and underlying cause. Captured sample
details are single-line and bounded to 400 characters; original cleanup failure
details remain available internally, outside public rendering. Their exit code is zero; they
change neither failed-run counters nor pruning exits. Canonical cleanup entries own each bounded issue and its optional full original
failure evidence, including nullable `error_type`. `issues` and `raw_failures` are
readable compatibility views; the latter contains only known original failures.
Legacy issue replacement cannot inherit a previous entry’s operation or cause;
explicit issues plus compact failure metadata require a unique order-preserving
match by validated issue content. Sparse associations remain sparse; ambiguous,
excess, or mismatched metadata fails instead of inventing attribution.
The nullable positional `aligned_failures` view and constructor input permit
lossless reconstruction even with duplicate issues. Aligned input requires equal
length and validates every non-null pair; it cannot accompany nonempty compact
metadata. Without explicit issues, nonempty `raw_failures` fully replaces
failure-derived entries, truncating old tails while preserving failure roles.
Empty failure metadata does not clear canonical entries;
explicit empty issues still clear them. Omitted typed context
may use verified caller context; explicitly unknown paths remain null. The existing
tuple-returning cleanup adapter emits safe public notes. `skipped` and `protection_skipped` retain
their compatibility interfaces. Compatibility notices use structured path/category
metadata; old unstructured strings receive informational wording and never imply
a failed protection check.

The internal/library failure record adds nullable `failure_role` with values
`abort_cause` and `file_local`. Known roles come from capture sites or explicit
legacy issue categories. Unknown raw failures remain unclassified and use
`audit_cleanup_failure` with fixed neutral public wording. No role is inferred
from an index, operation name or exception text. Canonical entries and compact/
aligned projections preserve this metadata; journal categories encode the role
without a new ledger column. Known abort-cause evidence requires `aborted=True`,
including during copies. A completed result must explicitly remove or correct
that evidence. Removing an abort-cause sample cannot relabel a remaining file
failure. Existing raw details and paths are retained unchanged.

Verification and cleanup summaries/samples add nullable `progress_id` inside
`failure_details_json`, without changing the run-ledger columns. It starts from
the first identified check and survives copies that change `check_id`.
Replacement uses `(stage, progress_id)` with legacy check-ID fallback, so final
journal merging and partial-progress capture apply the same rule. A copied result
supersedes its own verification and cleanup batches while independent checks
remain visible. Each new physical verification receives its own progress identity.
Identified batches preserve sample multiplicity; only repeated batch delivery is
replaced. A summary is emitted for a known progress identity even without a check
ID. Reading uses that effective identity before legacy clock grouping and recovers
consistent identity from older summary-less samples; malformed or conflicting
metadata cannot establish a clear check. Structured legacy samples retain their
observed occurrence count without a summary. A shared resolver supplies the same
progress/check fallback at binding, merge, read and serialization boundaries.
`progress_id` is omitted from JSON only when equal to the row's `check_id`;
different or progress-only sample identities remain explicit. The wire omission
does not remove internal ownership or combine independent equal-clock checks.
The explicit legacy detail adapter has typed constructor-compatible arguments,
including nullable metadata, without stripping typed or journal evidence. Empty
`audit_cleanup_diagnostics=()` means omitted; nonempty stale batches remain invalid.

### Damaged verification evidence

The retention report's boolean `verification_unknown` is independent of the last
valid check and outstanding execution failure. Invalid identity, clock or count
metadata quarantines its affected group; malformed entries and missing/non-string
stages create uncertainty even when no verification owner can be recovered.
Contradictory retention-stage metadata is also damaged evidence. Valid unrelated
components remain unrelated, and independently valid checks stay visible.
Malformed `retention_audit_cleanup` evidence stays separate from verification,
including when their identities match. Unattributable or contradictory stages
retain conservative uncertainty.
Cleanup evidence is assessed separately through additive report/status fields:
`cleanup_evidence_unknown` (boolean), `cleanup_recorded_at` (nullable aware ISO
clock), `cleanup_outcome` (`not_recorded`, `completed`, `incomplete`, or `unknown`),
and nullable safe `cleanup_warning`. Damaged cleanup evidence is amber while valid
verification and apply-run recovery remain intact. Its boundary is the damaged
run's validated ending clock; only a valid cleanup summary recorded strictly
later clears it. Dry runs and incomplete summaries may supply valid evidence.
The last valid summary remains visible; legacy absence and unknown clocks never
establish unrecorded success or fabricated timestamps. These are report fields,
not new run-ledger columns or failed-run counters.

Only a valid clear check recorded strictly after the damaged run's validated
ending clock resolves verification uncertainty. Equal-clock and suspended checks
do not. A clear dry run can resolve that uncertainty. Separately, a successful
apply invocation with an independently valid selected clear check clears an
execution failure even if another group in the same row is damaged. The panel
remains amber for the unresolved uncertainty until a strictly later clear check.
A dry run never clears an execution failure itself.

Both clear and suspended legacy verdict-only rows are verdicts, not issue samples.
Historical detail-only samples remain literal evidence with duplicate occurrences.
Only exact documented fallback verdict strings are candidates for sample-free
recognition; summary/count metadata resolves identical-text samples when possible.
Ambiguous or inconsistent counts remain uncertain. Frozen fixtures from the
`bd9f70c` and `35fc7e0` writers cover compatibility without rewriting artifacts.
Explicitly empty recorded category/path fields on suspended samples retain their
canonical unknown-category/null-path normalization.
A suspended verdict without samples or an explicit count keeps its suspension
and reports an unknown count. Additive nullable `issue_count_known` metadata
preserves that distinction through diagnostic round trips; false means the count
is unavailable. The report's nullable `issue_count`/`omitted_count` and dashboard em dash expose
that uncertainty. Existing integer library views retain zero recovered samples.
Older zero-valued omission placeholders remain readable.
Explicit counts, duplicate sample occurrences and omitted counts remain validated.
Ownerless legacy groups preserve their original recorded-clock strings: absent
clocks and equivalent instants written with different offsets remain distinct.
Parsed aware clocks still determine validity, chronological order and cutoffs.

Diagnostic merging and serialization execute inside the recorder's protected
boundary. Progress capture independently contains ordinary diagnostic errors.
Malformed fields retain bounded JSON-safe evidence, with `metadata_invalid`
marking invalid fields; an `invalid_diagnostic_metadata` fallback retains a core
run row when optional processing fails. These are informational damage markers,
not execution failures. They cannot replace the command's return or original
exception/cancellation, and do not add ledger columns. Ledger storage remains best
effort. Public warnings use fixed explanations rather than raw diagnostic prose.
Field validation follows cached declared annotations, accepts supported primitive
subclasses and enums, and normalizes them to JSON primitives while keeping bool
distinct from int. Each field is assessed once per serialization snapshot.
The schema follows the concrete diagnostic class and supports primitive/null
unions (`str`, `int`, `bool`, `None`) and primitive extension fields. Unsupported
annotations receive field-local damage without losing base stage/identity.
Normalization accepts actual primitive subclasses, never objects merely spoofing
their `__class__`. Strict identities and retention states raise `ValueError` for
invalid inputs; journal boundaries remain tolerant.
Authoritative command exits are validated before storage, so Polars never coerces
invalid exits into plausible outcomes. Invalid exits become 1 with
`invalid_command_outcome`; valid exits retain null classification and bounded
damage when optional classification is invalid. Invocation-local explicit
exception capture prevents ambient caller exceptions from contaminating records.
Ordinary outcome-hint failures cannot prevent the recorder attempt; malformed
optional classifications record null plus bounded damage evidence and confer no
continuation exemption. Cancellation retains null-exit/error journaling.

Authoritative summary boundaries replace repeated batch delivery while preserving
the relative order of every surviving item, including interleaved owners, and
all duplicate samples inside a surviving batch. Repeated untrusted deliveries
are deduplicated without hashing raw values. Bounded previews never establish
that distinct original evidence is equal. Historical artifacts are not rewritten.
Valid stage/identity ownership controls replacement independently of payload
validity, allowing final snapshots to supersede malformed progress safely.

## Daily report reader

Daily-report schema 1 retains its envelope and adds nullable realized-evidence
attribution: `evidence_match`, `evidence_release_id`, and
`evidence_dataset_fingerprint`, plus nullable `latest_scored_issue_at`. The latter
is present as null even on no-match/error realized slices and also extends verification output and the served-quality ledger; older rows load
with null clocks and no historical migration. Forecast release identities remain separate from
the evidence cohort. Station diagnostics expose the resolved fallback path; latest
run entries carry individual freshness. Served-quality daily deduplication now
includes dataset identity; existing rows and retention bounds remain intact.

Equal evidence/recording clocks prefer the latest scored issue when all tied
candidates have trustworthy clocks, then the forecast dataset; remaining distinct
cohorts produce `ambiguous_quality_evidence` rather than arbitrary scores.
Method-bearing unattributed slices remain unknown. Realized recorder code is
distinct from evaluation code, and compatible fallback requires strictly older
immutable release content. Malformed relevant run rows produce `invalid_run_rows`
without discarding independent commands; no completed outcomes at the cutoff use
`no_completed_runs_as_of`. These are documented schema-1 limitation codes, not a
schema migration.

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
