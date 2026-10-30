# Verification and the evaluation protocol

A forecasting system that cannot honestly measure itself is a random number
generator with good marketing. This page specifies the scores, the significance
machinery, and the protocol that makes both meaningful. The *decision* procedure
built on top — which method actually gets promoted — is
[model selection](model-selection.md).

---

## 1. Point scores

*Implemented in: `metrics/deterministic.py`*

$$\text{MAE} = \frac{1}{n}\sum_t |y_t - \hat{y}_t|
\qquad
\text{RMSE} = \sqrt{\frac{1}{n}\sum_t (y_t - \hat{y}_t)^2}
\qquad
\text{bias} = \frac{1}{n}\sum_t (\hat{y}_t - y_t)$$

MAE is the promotion metric. Bias gets its own column rather than being folded
into MAE, because a method can have unremarkable MAE while being systematically
and *correctably* wrong — that column is what caught the defect recorded in
[ADR 0004](../adr/0004-grounding-defaults-to-bias-only.md).

Note the estimand mismatch this creates: MAE is minimized by the conditional
median, RMSE by the conditional mean. Methods fitted by least squares and
promoted on MAE are optimizing one functional and judged on another. See
[notation §5](notation.md#5-estimands-and-losses).

---

## 2. Probabilistic scores

*Implemented in: `metrics/probabilistic.py` — the only module that touches `scoringrules`*

### Pinball loss

$$\rho_\tau(y, q) = \max\bigl(\tau(y - q),\ (\tau - 1)(y - q)\bigr)$$

The proper scoring rule for a single quantile; minimized in expectation by the
true $\tau$-quantile.

### CRPS

The continuous ranked probability score is the integral of the Brier score over
all thresholds, and equivalently $2\int_0^1 \rho_\tau\,d\tau$. The leaderboard
reports the **energy form for an ensemble**, evaluated on the sorted quantile
grid treated as members (`crps_ensemble`, via `scoringrules`).

A quantile-grid approximation also exists, and it is worth being precise about
because a common shorthand gets it wrong. `crps_from_quantiles` is **not** twice
the mean pinball loss — it is a weighted rectangle rule over probability levels:

$$\widehat{\text{CRPS}} = 2\sum_i w_i\,\overline{\rho_{\tau_i}},
\qquad
w_i = \frac{\tau_{i+1} - \tau_{i-1}}{2}$$

with the first and last weights closed against the boundaries $0$ and $1$:
$w_1 = (\tau_1 + \tau_2)/2 - 0$ and $w_m = 1 - (\tau_{m-1} + \tau_m)/2$. The two
coincide only for an equally-spaced grid; the board reports the energy form.
Levels must be strictly increasing within $(0,1)$ or the function raises.

### Brier score and reliability

$$\text{BS} = \frac{1}{n}\sum_t (p_t - o_t)^2$$

for PoP against binary occurrence. `reliability_bins` produces the calibration
table — forecast probability versus observed frequency over 10 bins — because a
probability scored only by MAE is a probability nobody is checking the
calibration of.

### PIT and coverage

$$\text{PIT}_t = \widehat{F}_t(y_t)$$

computed by interpolating truth into the quantile grid (clipped to $[0,1]$ at the
ends). A calibrated forecast has uniform PIT; the histogram's shape names the
failure — U-shaped means under-dispersed, hump-shaped over-dispersed, sloped
means biased. `empirical_coverage` reports the fraction of truths inside a
declared interval, checked against nominal.

### Dashboard diagnostics

*Implemented in: `dashboard/score_summary.py`, `dashboard/zones/evaluation.py`*

The dashboard retains a 10-bin PIT histogram on `[0, 1]` and PoP reliability
bins when at least 20 probability rows are available. PIT and reliability
derivation have independent failure guards: either failed diagnostic receives
an amber warning while valid leaderboard, winner, and other diagnostic results
remain available. The summary flags `pit_unusable` and `reliability_unusable`
describe these local failures; `unusable` is reserved for core scoring failures.
These views use the proper-score conventions cited in
[Bibliography](bibliography.md#verification-testing-and-selection).

---

## 3. Skill

Skill is always relative, and always relative to a **named** reference:

$$\text{skill} = 1 - \frac{\text{MAE}_{\text{method}}}{\text{MAE}_{\text{reference}}}$$

The default references are three:

| Reference | What beating it proves |
|---|---|
| `best_provider` | you beat the best single API, chosen per bucket by training MAE |
| `equal_weight` | the **raw**, ungrounded arithmetic mean — beating it proves *grounding* works |
| `damped_grounded_equal_weight` | grounded, climatology-damped equal weight — the strong baseline |

Three, not one, because each answers a different question, and "we beat the worst
provider" is not a claim worth making. Per-variable overrides live in
`[promotion.references]`; the leaderboard computes columns for the *union* of
defaults and overrides so a pinned `skill_vs_equal_weight` column never silently
changes meaning when someone edits config.

The floor every method must clear is the baseline set: `persistence` (the current
station reading held constant), `climatology` (ridge-regularized Fourier
regression of truth on month and hour), and `best_provider`.

---

## 4. Diebold–Mariano

*Implemented in: `metrics/dm.py::diebold_mariano`*

An MAE difference of 0.02 °C over 700 samples is noise. DM tests
$H_0$: equal expected loss, on the **paired** loss differentials
$d_t = \ell^A_t - \ell^B_t$.

$$\text{DM} = \frac{\bar{d}}{\sqrt{\widehat{\operatorname{Var}}(\bar{d})}} \cdot \text{HLN}(n, h)$$

**Bartlett-kernel HAC variance.** At multi-hour leads, consecutive forecast errors
are serially correlated and the naive variance is far too small:

$$\widehat{\operatorname{Var}}(\bar{d}) = \frac{1}{n}\left(\hat\gamma_0 +
2\sum_{j=1}^{\min(h-1,\,n-1)} \left(1 - \frac{j}{h}\right)\hat\gamma_j\right)$$

floored at 0 (a truncated HAC estimator can go negative).

**Harvey–Leybourne–Newbold correction**, with a Student-$t(n-1)$ reference,
because these slices have hundreds rather than millions of samples:

$$\text{HLN}(n, h) = \sqrt{\frac{n + 1 - 2h + h(h-1)/n}{n}}$$

**Guards.** `MIN_SAMPLES = 8`; $h \ge 1$ and $h < n$ required. A zero variance
with a nonzero mean yields $\pm\infty$ and $p = 0$ — the honest reading of "these
two methods differ identically on every case".

$n$ is printed next to every leaderboard row, so a thin daily slice announces its
own lack of power rather than hiding behind a p-value.

!!! warning "DM is a test, not a decision procedure"
    A leaderboard runs dozens of these every night, on overlapping data, and
    re-reads them daily. Taking the argmin and calling it the winner commits
    three errors at once — multiplicity, optional stopping, and the winner's
    curse. What DM feeds is [model selection](model-selection.md); it is no
    longer the arbiter itself.

---

## 5. Rolling-origin protocol

*Implemented in: `backtest/splits.py`, `backtest/engine.py`*

At each origin $O$:

$$\text{train} = \{t : \texttt{truth\_known\_at}_t \le O\},
\qquad
\text{test} = \{t : T_t \in (O,\ O + \text{step}]\}$$

The subtlety is entirely in the first line. It is **not** $T_t \le O$. A row
issued yesterday *about tomorrow* has an issue time in the past and a truth that
has not happened yet; training on it is leakage.

$$
\texttt{truth\_known\_at} = \begin{cases}
V + 5\text{ min}, & \text{minutely}\\
V + 2\text{ h}, & \text{hourly (the hour must end, plus ingest lag)}\\
\text{end of local day} + 1\text{ h}, & \text{daily}
\end{cases}
$$

Fold plans are computed in epoch microseconds. Both **expanding** and **rolling**
(180-day) windows are supported and reported side by side, because they answer
different questions: expanding asks *what do you know?*, rolling asks *what have
you learned lately?*

### Evaluation outcomes and snapshot clock

*Implemented in: `backtest/splits.py`, `backtest/outcomes.py`,
`backtest/engine.py`, `backtest/minutely.py`,
`dataset/evaluation_snapshot.py`, `cli.py`*

The planner's first origin is the first issue plus `initial_train_days`
(default 90); an issue must occur strictly after it. `step_days` (default 7)
spaces later origins, rather than requiring a full first test interval.
Producers distinguish `evaluated`, `pending`, `not_applicable`, and `failed`.
Any emitted scores complete a product. Otherwise valid short clocks are pending;
future test truth is pending only when slices with usable training have eligible
test truth clocks later than the observation cutoff. Unusable variables do not
veto usable slices; all-unusable training, due-but-missing truth, and mature
zero-score output remain failures. Truth delays above are unchanged.

Live and synthetic manifests add `truth_snapshot_at`, captured before the
consistent observation read; `created_at` remains publication time. Synthetic
truth and causal features use the same read. Virtual synthetic issue times need
not precede those clocks. Neither timestamp enters the content fingerprint.
Input hashes and required schemas are always validated. Missing/malformed maturity
clocks permit positive scoring and short-history pending, but cannot grant
`awaiting_truth`; publication time, file modification time, and backtest wall time
are never substitutes. CLI pending exit 3 retains the historical `no_folds` ledger
classification. Products are evaluated independently under unchanged gates.

### Approved decisions and compatible retention

*Implemented in: `serve/selection.py`, `serve/evidence.py`, `serve/approvals.py`,
`reports/evidence.py`, `reports/operations.py`, `serve/provenance.py`, `evaluation.py`, `cli.py`, `status.py`, `runs.py`*

Only completed products in a live backtest may receive new selections. Prediction
loads approvals without elections, promotion-time live gates, or release writes.
Compatible incumbents survive pending, failed, unrequested, and completed slices
without a qualifying replacement. Code, configuration, product sources, exact
feature set, and truth semantics must match; data-only fingerprint rotation does
not retire an approval. Each selection and context retain their original dataset
and evaluation identity and available cutoff. New releases identify their creation
dataset separately from retained evidence. Legacy records remain readable without
invented context. Historical reconstruction retains exact dataset and issue-time
checks. `PromotionResult.created` is decided under the release write lock so
retention and duplicate writes are never logged as new promotions.

A single locked approval snapshot supplies promotion decisions. Current managed
approvals follow committed sequence without wall-clock eligibility checks; legacy
implicit approvals require publication-time eligibility behind managed events.
UTC conversion overflow, missing zones and malformed clocks are excluded.
Historical queries require publication, approval and conservative visibility within
the cutoff. `visible_at` is the maximum of actual approval time, stored publication
time and all preceding managed visibility floors; older events derive this floor
without rewrites. Events retain actual commit time in `approved_at`. First approval
versus reapproval uses committed history, independently of historical visibility.

Only final selected-variable contexts are merged. Required context omissions and
conflicts are typed failures before any content/event publication. Known legacy
references survive missing context without invented cutoffs. Blocked live gates
retain selection identity and record measurements separately in the bounded
`live_gate` telemetry ledger (730 days, 100,000 rows); statistical constants and
reference qualifications remain unchanged. The diagnostic batch captures identity
once and records after unlocking, even when approval publication fails.
Active/recent-served legacy references
can protect scores without authorizing predictions. Churn uses separate publication
and approval clocks, sequence order and approval identity; old rows normalize in
memory without rewriting evidence.

Snapshot enumeration uses explicit directory scans. Typed path/category/detail
issues retain damaged release/event files, invalid identities/clocks and listing
failures. Visibility and authorization ordering are cached once per snapshot.
Historical adapters also check externally supplied content publication clocks.
Expected publication failures and stale decisions are typed promotion-only
outcomes, preserving evaluation failures and diagnostics. Unknown evaluation
placeholders never enter live-gate cohorts.

### Bounded score persistence

*Implemented in: `backtest/engine.py::iter_backtest_scores`,
`backtest/minutely.py::iter_minutely_scores`, `backtest/scores.py`,
`backtest/score_storage.py`, `storage.py`,
`reports/operations.py::prune_scores_files`, `cli.py`*

Published retention keeps the newest 3 scores per product/source/window group,
current active evidence and approvals from the last 7 days. A genuinely missing
active pointer protects the newest current approval together with ready documents
served in the last 6 hours; an explicit empty pointer preserves degraded semantics.
Corrupt pointers, unreadable archives, unverifiable release/event metadata and
listing failures suspend all published-score deletion in actual and dry runs.
Future managed clocks remain usable under current sequence ordering. Uncertain
legacy clocks suspend protection decisions rather than authorize serving or lose
scores. `PruneResult.skipped` identifies the exact affected paths; exits retain
existing semantics and temporary cleanup remains independent. Dashboard artifact
warnings expose verification failures. No historical artifact is repaired or
rewritten automatically.

The CLI retains the same fold/variable/method order and fits exactly the same
models. `SCORE_BATCH_ROWS = 65_536` slices predictions before stamping or quantile
JSON serialization. All string columns except `quantiles_json` use dynamic
categorical codes inside these batches. The writer buffers at most 65,536 rows
and writes one Parquet row group per full buffer, then the final partial buffer.

PyArrow writes with Zstandard level 3, dictionary encoding for categorical
columns and `store_schema=False`; omitting Arrow schema metadata preserves the
existing String columns when Polars reads the stored file. UTC timestamps,
nulls, Float64 predictions and quantile JSON retain their existing meanings.
One evaluation ID spans each product's batches. Schema or identity mismatch,
iteration/write/close failure, or handled interruption before commit aborts publication.
The producer and writer close before atomic replacement, with existing output
permissions preserved. Empty score streams retain the historical empty schema.

Full prediction and truth arrays must have exactly one value per key before
batching; quantiles must be two-dimensional with the same row count and a width
matching nonempty quantile levels. Levels must be finite, strictly increasing and
inside `(0, 1)`; nonempty levels without quantiles are rejected. Point-only
`None`/empty level metadata retains its historical JSON representation. Hourly
and daily keys and supervised arrays share the same finite-truth filter, as does
minutely proxy training; invalid truth is excluded before empty-fold checks. Each compact
batch is cast directly to the canonical schema, retaining the former timestamp
and numeric normalization. Naive datetimes are interpreted as UTC, matching the
materializing API; aware datetimes convert to UTC while preserving instants.
PyArrow loads only in the writer. Ordinary secondary
close failures annotate the original error; a new cancellation or process-exit
exception during cleanup propagates with its original exception context.
Both score writers use independent OS-backed leases for private
`.score-stage-v1.<destination>.<16 lowercase hex>.tmp` stages and their `.lease`
companions. Leases span encoding, producer closure and atomic replacement, while
`.score-stage-v1.lock` guards only allocation, replacement and removal. Writers
can encode concurrently, including to the same destination (last replacement
wins). The guard reenters on the same process/thread using the directory's
filesystem identity; recovery always uses fresh lease handles, so it cannot
reclaim a live stage even inside the writer thread.

`SCORE_COORDINATION_TIMEOUT_S = 60.0` bounds each publication guard acquisition.
Allocation and publication each have that budget (up to 120 seconds combined,
excluding model computation and other locks). Only acquisition contention becomes
`ScoreCoordinationTimeout`; CLI backtest, maintenance and recovery return 75,
release the pipeline lock and skip promotion. Cleanup acquisitions are nonblocking. Readers never take these locks. Backtest startup
and `prune-scores` discover candidates outside coordination and revalidate each
under its own short guard and fresh lease. The guard releases between candidates;
contention defers the remainder. Only inactive private stages are reclaimed,
including payload-only allocation and lease-only publication remnants. Legacy names, unrelated stores,
symlinks and malformed names remain untouched. Dry runs may create lock metadata
but never explicitly unlink payloads or companions. Only payloads contribute to
temporary counts/bytes, and completed-evaluation retention meanings are unchanged.
Ordinary housekeeping errors after publication produce diagnostics without
turning a committed write into failure. Deletion accounting survives release
errors; `ScoreCoordinationReleaseError` retains the guard path and original cause,
supports ordinary pickle round trips with notes and cause chaining, and stops
recovery by raising if guard release fails. Its optional `partial_result` contains
a `ScoreTempCleanup` with completed payload paths, decimal megabytes and prior
nonfatal `skipped` diagnostics; the exception also carries the partial-counts/bytes
note. The fatal release failure is not duplicated in `skipped`. Pickle and process
transport preserve this partial result, metadata and notes. Current CPython process
executors may substitute a remote traceback as `__cause__`; callers use `.cause`
for original exception metadata rather than depending on that representation.
Backtest startup catches the release error before invoking product writers and
prints partial progress plus one fatal diagnostic. Both startup
and uncommitted writer release failures are handled by CLI backtest, maintenance
and recovery with exit 1, journal failure kind `score_coordination_release`, the
diagnostic in `error_detail`, and no promotion. `prune-scores` retains its partial
temporary counts and diagnostic while continuing published-score retention.
It then returns exit 1, including for dry runs and unverifiable-release early
returns, and journals the same failure kind. `PruneResult.coordination_release_failure`
contains an immutable `ScoreCoordinationFailure`: lock path, cause type name,
message and notes. Neither this snapshot nor `ScoreTempCleanup` contains an
exception, traceback or lock reference.

Every failed directory-guard release, including during an existing failure or
control-flow exception, records a snapshot in a process-wide registry protected
by a mutex. The key is `(pid, st_dev, st_ino)`, separate from thread-local reentrant
ownership. Acquisition checks this registry before reentrant bypass and before
translating a timeout; known failures raise fresh typed exceptions with synthetic
causes describing the original failure. Aliases and other threads therefore fail
immediately rather than waiting `SCORE_COORDINATION_TIMEOUT_S = 60.0`. Markers
persist until process exit, even if finalization subsequently unlocks the guard.
An `os.register_at_fork` child callback resets the registry and mutex. Committed
publications retain warning-only behavior; a final-product release warning alone
does not fail backtest or suppress promotion. Later coordination still fails.

Diagnostics identify the affected
resource and cleanup phase. Unexpected programming errors in recovery work
propagate with partial payload counts and bytes in an exception note. One internal
progress accumulator records these immediately after each successful payload
action, before housekeeping. Expected standalone `OSError` lease-release and
companion-removal errors remain best effort; standalone programming errors in
either phase propagate. Ordinary secondary cleanup errors annotate a pending
failure, while new control-flow exceptions propagate with the
original context. Committed warnings retain all secondary notes and distinguish
release attempts, retry outcomes and the lease's final reported ownership state
(`still held`, `reported released` or `not held on entry`). Companion-removal
completion is recorded separately, without additional filesystem probes. Expected
payload disappearance is handled before secondary cleanup, so disappearance alone
is silent while later release/companion errors follow the standalone-error policy.

Directory acquisition I/O errors stop the pass with one directory-level diagnostic;
per-file I/O errors under coordination can continue. Recovery skips candidates
whose payload and companion are both absent. It rechecks directory existence
before each acquisition and stops on observed disappearance. Directory removal
after the existence check can race with parent creation by `score_write_lock`,
`locked_path` or FileLock; no strict noncreation guarantee or alternative recovery
lock backend is introduced.

Disposal retains its zero-wait guard attempt. Even handled failures or successful
publications can leave payloads or lease companions when that attempt loses
contention; a later recovery pass reclaims them. The per-thread reentrancy state
is explicitly typed and initialized separately for each thread.

The DataFrame-returning library APIs remain materializing compatibility paths.
The [backtest memory benchmark](../reference/cli.md#backtest-memory-benchmark)
measures score construction and writing separately from model fitting.
The encoding follows the [Polars categorical data guide](https://docs.pola.rs/user-guide/expressions/categorical-data-and-enums/)
and [PyArrow Parquet writer documentation](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html).

---

## 6. Leakage: assumed present until proven absent

Four defences, all executable:

1. **Fold-plan invariants** — property-based, via Hypothesis over randomized
   configurations:
   $\max(\texttt{truth\_known\_at}[\text{train}]) \le O < \min(T[\text{test}])$,
   $\text{train} \cap \text{test} = \emptyset$, rolling windows really are bounded.

2. **The poisoning sentinel.** For each fold, corrupt *every* truth value not yet
   knowable at that fold's origin (add $10^6$) and re-run the whole engine.
   Assert the fold's test predictions are **bit-identical**. If future truth
   reaches a model through *any* path — a feature, a fitted aggregate, a stateful
   blender — this test fails. It is the single most valuable test in the codebase,
   because it does not require anyone to have anticipated the leak.

3. **Fresh instances.** The registry stores *factories*, never instances; the
   engine constructs a new blender per fold. A stateful method (the online
   experts) cannot leak yesterday's weights into today's fit.

4. **Feature audit.** No column beginning `t__` may reach a blender —
   enforced in `ForecastMatrix.__post_init__`, so the illegal object cannot be
   constructed.

A fifth discipline sits in the dataset layer rather than the engine: the
**causal QC variant**. Issue-time `obs__` features are filtered by
`apply_causal_qc`, which applies bounds immediately, flags flatlines only once
their duration crosses the threshold, and **omits the isolated-spike rule
entirely** — because that rule needs a *later* sample to fire. See
[truth QC](truth-qc.md#1-station-quality-control).

---

## 7. The provenance wall

Backfilled ("synthetic") and live rows are **never pooled**. They live in
separate files, carry a `source_kind` tag, and any attempt to mix them raises
`MixedProvenanceError`. The provenance is a *filesystem* property — it is in the
matrix filename — not merely a runtime check.

This is not fastidiousness. The synthetic archive contains three open NWP models
at whole-day native offsets (canonical rain starts one hour earlier); the live
archive contains eleven commercial providers
at native cadence. A leaderboard built on the former says nothing about the
latter, and a number quietly averaging the two would be worse than no number.

Identity is tracked by three fingerprints — `dataset_fingerprint` (byte-level,
over stably-sorted parquet), `config_fingerprint`, and `code_identity` — and
new elections require matching fingerprints. Approved decisions may survive a
new dataset fingerprint only under the compatibility checks above; original
evidence retains its own fingerprint. See
[ADR 0005](../adr/0005-promoted-model-releases-are-the-serving-boundary.md).

---

### Approval events and retained evidence

*Implemented in: `serve/approvals.py`, `serve/selection.py`, `serve/evidence.py`,
`evaluation.py`*

Release content is immutable; changed approval decisions commit separate ordered
events under one ledger lock. A → B → A records reapproval, while A → A is unchanged.
Managed content without an event is not approved. Legacy releases use their aware
`promoted_at` clock with publication-time eligibility. Current managed decisions
follow sequence across clock corrections. Historical replay requires publication,
approval and the nondecreasing `visible_at` floor within its cutoff, retaining exact
dataset and issue-time constraints. Invalid, naive and overflowing clocks are
excluded in both modes. Reporting shares the same snapshot; operational pruning
protects current decisions. Nullable approval IDs in
the churn ledger distinguish repeated transitions without duplicating unchanged
reports; legacy ledger rows remain readable.

Combined releases validate shared evaluation identity and union variable semantics.
Nullable per-slice `training_cutoff` values determine the remaining context/release
summaries after replacement; a legacy aggregate never supplies an unknown slice's
cutoff. This is operational provenance policy, not a new statistical gate. Existing
live-gate constants apply to carried choices in completed products, while pending,
failed and unrequested products preserve their approvals.

Missing legacy context is marked per slice with `legacy_context_missing`. It is
inferred only through accepted exact-dataset compatibility and cannot borrow
another selected variable's context or authorize cross-dataset serving. Historical
`"unknown"` evaluation IDs normalize to missing provenance. Raw contexts always
validate conflicts; promotion consumes a typed, once-merged context set.
Evaluation failures take precedence over promotion-only failures while nullable
`failure_details_json` component diagnostics retain both; completed scores remain.

### Published-score protection verification

*Implemented in: `reports/operations.py`, `retention_status.py`, `retention_audit.py`, `clocks.py`,
`artifact_labels.py`, `verification_text.py`,
`cli.py::_cmd_prune_scores`, `cli.py::_record_run`, `status.py::diagnostic_progress`,
`status.py::record_verification_progress`, `status.py::record_progress`,
`status.py::merge_diagnostics`, `run_health.py`,
`status.py::effective_progress_id`, `status.py::assess_progress_identity`,
`status.py::failure_details_json`,
`dashboard/context.py`, `dashboard/zones/operations.py`.*

Pruning freshly verifies release/event metadata, the active pointer and required
served-document references before deleting published scores. Uncertainty suspends
all published deletions, including dry runs, with exact affected-path diagnostics;
temporary cleanup and existing exit meanings remain intact. Retention still keeps
the newest three files per group, active/recent-served references and approvals
from the preceding seven days. Legacy reference protection does not authorize
serving. Pointer-field diagnostics remain distinct from referenced-output errors.

Each completed check carries an additive typed `PruneResult.verification` result
and records informational `retention_verification` components in the existing
run ledger, using exit 0 and `retention_clear`/`retention_suspended`; details are
bounded to 400 characters per component; an additive optional `path` preserves
the complete affected path independently. Archive errors are collected per file,
not stopped at the first failure. An optional `recorded_at` carries the completed
check clock. Invocation/stage-scoped context-local progress preserves completed
checks before audit I/O and through catchable audit/deletion/output exceptions without duplicate journal rows
or changed exception semantics. Coordination failures keep their own failure
status. Reports/dashboard display the last recorded check and its age separately
from the latest eligible attempt, without rescanning the archive. Failed or
unverified attempts cannot erase a suspension or establish a clean check. Missing
legacy metadata is unverified; only an explicit later clear check establishes
verified protection. See the [operational references](bibliography.md#3-operational-analogues).
Direct loaded-artifact checks remain live. A valid legacy omission of `release_id`
is compatibility information for retention, not an unreadable-file alarm or a
serving approval. No ledger migration or historical rewrite is required.

Each new check has a unique ID. Run diagnostics retain one summary and at most
20 issue samples with 400-character bounded explanations and separate exact paths;
full path/category/detail records stream to an immutable JSON Lines audit beneath
`artifacts/history/pruning_verifications/`. Its schema-1 header identifies the
check/time/dry-run, and its footer confirms the total issue count. Exclusive
publication flushes and synchronizes the complete staged stream before linking,
then synchronizes the directory; incomplete or colliding publication cannot
overwrite an audit. Hard-link and synchronization support are required for a
confirmed audit. Audit failure is explicit and cannot lift
suspension. Dry runs still record diagnostics without deleting scores or audits.
Audits expire only after 90 days and after all retained run references disappear;
uncertain references or audit metadata prevent cleanup. This does not change
scientific-evidence retention. A dedicated nonblocking audit lock coordinates
publishers and cleanup; contention remains informational. Cleanup scans candidates
before reading run references. Exact known staging names are reclaimed only when
regular, unchanged and older than 90 days under the same lock. Other uncertain
metadata still prevents deletion. SIGKILL and power loss do not guarantee journaling.

A successful exclusive link records nullable `audit_candidate_path` immediately in
bounded progress, before directory synchronization. This complete but unconfirmed
reference remains separate from confirmed `audit_path` until synchronization and
lock release succeed. Catchable post-link failures preserve the candidate; collisions
never adopt an existing file. Both relative references remain protected by retained
run diagnostics. Valid legacy check IDs protect exact matching complete audits
without inventing publication confirmation. Per-file cleanup failures retain that
candidate and allow independent verified candidates to proceed. Global reference
uncertainty continues to block all audit/staging cleanup. The 20-note cleanup bound
reserves room for an overall abort summary and reports omitted file issues.

Public explanations use fixed category/reason templates, validated when reading
run diagnostics; raw exception prose is never public text. Unknown codes and
unstructured legacy details use a neutral diagnostic reference. Full original
path/category/detail and nullable reason code remain in audit evidence. Unknown
issue paths remain null. Exact legacy path/category prefixes are removed once only through explicit
legacy ingestion; embedded prose and raw constructor inputs are not rewritten. Audit-publication causes have separate templates.
This retains the existing `fsync` policy without a stronger power-loss guarantee.

`RetentionVerification` stores canonical immutable typed issues. Constructor
`details=` and text-only replacements preserve literal raw text, so reconstruction
from `details`, `paths`, `categories`, and `reason_codes` is lossless. Labelled
legacy strings use `from_legacy_details(...)`, which strips exact recorded prefixes
once. Known prefixed callers use this adapter; format is never inferred from prose.
Its typed arguments match the constructor's positional metadata and support null
sequences and typed issues. Only supplied labelled text authorizes prefix removal;
typed issues alone retain their raw evidence.
Metadata-only replacements retain canonical issues, and journal reads stay raw.
Accepted mutable sequences and compatible issue objects are frozen through the
shared converter; already canonical immutable tuples are reused. Equal objects
have identical issue projections; no hidden normalization mode is carried across
replacements. `details` exposes canonical raw evidence and `formatted_details`
explicitly composes labels. Legacy text edits retain positional metadata, explicit
legacy metadata fields override corresponding structured projections, and empty
details clear issues. These metadata associations do not authorize prefix stripping. New code replaces complete typed issues; journal reads never strip
genuine raw evidence prefixes. The bounded `sample_issues()` projection shares these
canonical values. Input adapters normalize empty
paths to null, empty/missing categories to `verification_invalid`, and absent
reason codes to null. All renderers and run diagnostic serialization share it;
the full legacy audit adapter streams every supplied issue through the same
normalizer. Unknown codes retain their internal evidence but render neutrally.
Report explanations deduplicate in first-seen order without collapsing paths,
samples or full audit rows. Required-path `ApprovalIssue` remains distinct from
nullable-path `VerificationIssue`, with a shared conversion preserving normalized
instances.

Audit classification is operation-specific. Only `filelock.Timeout` during lock
acquisition establishes contention; ordinary I/O timeouts remain I/O, link or sync
failures. `audit_reason_code` preserves the failed operation, while additive nullable
`audit_cause_code` records a known cause independently. Link EPERM/ENOENT do not
erase exclusive-publication context or establish filesystem capabilities. Busy
publication is skipped for that check without a retry. Missing referenced releases,
absent eligible approvals and malformed evidence have distinct typed reason codes;
unlocated failures remain pathless.

Cleanup yields a typed bounded result and safe tuple compatibility notes. Raw
exception prose never supplies public cleanup text, including global aborts.
Typed audit failures carry the operation, nullable path, reason and underlying
cause at the failure site. Cleanup uses this context rather than a mutable location
sentinel. Captured issue details use the shared single-line 400-character bound;
original raw failure details and nullable exception `error_type` remain internal.
Omitted typed context may use verified caller metadata; explicitly unknown paths
stay null. Canonical cleanup entries own issue and optional original failure
evidence together, so issue replacement cannot inherit unrelated operation/cause
metadata. The `issues` and `raw_failures` compatibility views are derived, not
independently indexed stores. With explicit issues, compact failures require a
unique order-preserving association by validated issue content. The earliest and
latest compatible index sequences must agree; otherwise attribution is ambiguous
and rejected. Unmatched issues retain no invented metadata. The nullable
`aligned_failures` view and input preserve every position, including duplicate
issues and absent evidence; length and each non-null pair are validated.
Nonempty compact and aligned inputs cannot be combined. Without explicit issues,
nonempty `raw_failures` replaces the failure-derived entries completely, truncates
old tails and retains recorded failure roles. Aligned input provides
explicit annotation of canonical entries. Excess or mismatched metadata is rejected.
Empty `raw_failures=()` does not erase canonical entries. Explicit `issues=()` still
clears issues; replacing an issue collection does not borrow old failure metadata.
Nullable `AuditCleanupFailure.failure_role` is `abort_cause`, `file_local`, or
unknown. Producers set it at capture; neither position, result outcome, operation
name nor raw prose establish a role. Explicit legacy categories are recorded role
evidence and are preserved through canonical entries and compact/aligned views.
Unclassified raw failures use the neutral `audit_cleanup_failure` category.
The explicit `issue(aborted=...)` adapter cannot override a known role. Known
abort-cause evidence requires `aborted=True`; inconsistent completed copies fail
until their evidence is explicitly corrected or replaced. Filtering out an abort
cause cannot transfer its role to another failure. No historical artifact is
rewritten; raw details, paths and operation/cause data remain unchanged.
`operation` also survives publication
summary round-trips: pre-link failures report no audit publication, while post-link
failures retain an unconfirmed candidate. Historical contention and link failures
also prove pre-link failure when no candidate exists; legacy sync failures with
unknown phases remain neutral.

Every cleanup emits one informational summary with nullable `dry_run`,
`deleted_audit_count`, and `deleted_staging_count`. New counts increase only after
successful unlink, including before a later lock-release failure; dry runs record
zero, and missing legacy values remain unknown. Nullable `eligible_audit_count`
and `eligible_staging_count` increase only for unreferenced candidates passing
final signature revalidation. A completed dry run reports “Would delete” counts;
aborts report verified eligibility so far. Eligibility is independent of unlink
success. The typed `CleanupOutcome` property classifies `unknown`, `completed`, or
`incomplete` once for all renderers. Contradictory legacy flags such as
`aborted=True, completed=True` are rejected. Nullable `cleanup_completed` is derived
from that outcome; known deletions never establish overall completion. Unknown
summaries distinguish final status from recorded deletions. Default results
emit no execution diagnostic, and recorded unknown outcomes use
`audit_cleanup_unverified` instead of claiming success. Cleanup abort wording uses
recorded progress without suggesting deletion before acquisition. The same
`AuditCleanupRecord` binds the immutable cleanup source, check ID, and its diagnostic
batch. The producer records that exact batch immediately and reuses it in the
command outcome, including through later output/deletion failures. Metadata-only
prune-result copies reuse the record; cleanup or check-ID changes rebind once.
Compatibility constructors and readable result views remain available; an explicitly
supplied incompatible nonempty diagnostic batch is rejected; an empty tuple retains
the legacy omitted-batch meaning. A standalone bound record keeps
its known identity unless verification explicitly changes it. Journal merging uses
the later authoritative verification or cleanup batch, replacing earlier samples
and summary together for that stage. A nullable `progress_id` starts from the
first identified check and survives copies even when `check_id` changes. Both
progress capture and final merging key batches by stage and this stable identity,
falling back to check ID for older diagnostics. Each new physical verification
gets its own identity; genuinely independent checks and unrelated diagnostics survive;
without a returned outcome, captured progress remains authoritative. Legacy batches
without either identity retain exact-deduplication behavior. This is additive JSON
metadata, not a new ledger column or historical rewrite. No CLI recomputation occurs.
Identified batches preserve identical sample occurrences and their order; repeated
delivery replaces a whole batch rather than deduplicating its evidence. Counts,
the 20-sample cap and omitted counts therefore remain consistent with full audits.
Progress-only verifications emit summaries and group by effective identity before
legacy clocks, preserving independent checks at equal timestamps. Older rows can
recover consistent identity from samples without a summary. Invalid or conflicting
identity metadata cannot establish a clear check. The same pure resolver is used
by binding, merge, reading and serialization. JSON omits only redundant progress
IDs equal to that row's check ID; renamed and progress-only samples retain their
owner. Summary-only identity storage is deliberately not used.
Strict domain construction retains identity validation. Journal boundaries use
non-raising identity assessment. Merging and serialization execute inside the
recorder's protected boundary; progress capture independently contains ordinary
telemetry failures. Malformed fields have no trusted ownership and serialize as
bounded invalid metadata with `metadata_invalid`; `invalid_diagnostic_metadata`
marks core-row fallback if optional processing fails. The original result,
exception/cancellation and one invocation/stage row are preserved when ledger
storage is available. Public warnings never render raw metadata prose. Safe exact
deduplication contains only built-in keys; opaque/cyclic values use object identity,
not bounded preview text. Different evidence is not collapsed by truncation.
Identity assessment is reused once per object during a merge. Indexed batch
replacement uses valid summary boundaries, including within concatenated inputs,
without moving surviving interleaved items; summary-less legacy samples remain
one batch per owner per input. Repeated sample
values never establish a delivery boundary.

Recorded groups are assessed independently. Invalid identity, clock or summary
metadata quarantines its affected group, including related known ownership hints,
without discarding unrelated valid checks. The additive report/status field
`verification_unknown` keeps damaged evidence amber while retaining the last valid
check. Its uncertainty boundary is the damaged run's validated ending clock. Only
a valid clear check strictly later than that boundary resolves it; a clear dry run
can resolve check uncertainty independently of outstanding execution failure.
Successful apply invocations with an independently valid selected clear check clear
execution failure even when unrelated damage occurs in that row. Equal-clock or
suspended checks cannot resolve damaged verification evidence. Malformed entries,
missing/non-string stages and contradictory retention-stage data are uncertain;
valid unrelated stages remain unrelated. Group assessment is repeatable and does
not mutate ingestion state. Valid equal-clock checks continue to use persisted order.

Legacy clear and suspended verdict-only rows are excluded from samples.
Explicitly empty recorded category/path fields on suspended samples retain their
canonical unknown-category/null-path normalization. A suspended
verdict with no samples or recorded total carries `issue_count_known=False`, showing
an unavailable count without fabricating an issue. Nullable diagnostic metadata
preserves this knowledge through summary and sample round trips; integer library
views retain zero recovered samples. Explicit totals and omitted counts remain
validated. Ownerless groups retain raw recorded-clock boundaries, including absent
versus explicit clocks and equivalent offset strings; parsed aware clocks control
validation and stable chronological sorting. Invalid summaries or totals smaller
than observed occurrences cannot prove clear protection. No failure counter or
run-ledger column changes.

Informational `retention_audit_cleanup` components preserve bounded details and
separate full paths in the existing journal row; they cannot change command exits or failure
counters. The 20-note cap still reserves abort and omission slots, and dry runs
retain deletion-phase revalidation for diagnostic parity. Structured compatibility
notices supply artifact labels and fixed informational text; legacy string notes
remain informational without being trusted as prose. Legacy strings display the
full aggregate count; structured notices display at most 20 individual entries
with an omitted count. Normal dashboard collection derives the string view from
structured notices and uses direct typed access. The old string constructor
remains a compatibility helper, not a rendering trust marker.

Checks group by ID, or by recorded clock for legacy diagnostics. Multiple checks
in one row select the latest completed clock, with persisted order breaking ties.
The report's nullable `check_age_minutes` and dashboard age share the explicit
reporting cutoff. Relevant malformed newer/unorderable runs remain unknown/amber;
provably superseded damage is informational. Exit 75 is contention, not a new
protection failure, and cannot clear a suspension/outstanding failure. Only a
later successful clear non-dry-run check clears an execution failure. Nullable
`dry_run` metadata takes precedence over recorded arguments; unknown legacy mode
cannot establish execution recovery. A dry run may still establish a clear
protection check. Progress replacement uses check identity and preserves unrelated
diagnostics and distinct checks. Path labels use
lexical relative paths with specific-root precedence and never edit prose or
resolve links. Unsafe legacy/raw explanations receive neutral public text while
original detail remains in diagnostic evidence. Audit availability and at most
20 compatibility notes are separate from the 20 issue samples; omitted notes are
counted. The [operational references](bibliography.md#3-operational-analogues)
cover the clock, stream, and atomic-publication mechanisms.

## 8. Self-verification

*Implemented in: `reports/verification.py`, `reports/evidence.py::served_quality_rows`*

The backtest can only measure the backtest. Serving has its own code path,
its own snapshot construction, and its own failure modes, so the system scores
**what it actually served** against truth as truth arrives.

`verify_history` joins served forecasts to subsequently-arrived truth and
computes `live_mae`, `live_rmse`, `live_bias` per
(product, variable, truth semantics, lead bucket, method, dataset fingerprint,
release id), with `_MIN_SCORED = 5`. Each qualifying cohort records the maximum `issued_at`
among its truth-matched scored rows as `latest_scored_issue_at`; unmatched newer
forecasts do not advance it. This metadata changes neither the cohort nor its
metrics or live gates. The served-quality ledger deduplicates daily per dataset
and release cohort, preserving first-write behavior and its 500,000-row cap.
`compare_to_backtest` places

$$\text{mae\_gap} = \text{live\_mae} - \text{backtest\_mae}$$

(n-weighted) beside it. A persistent positive gap is serving-path drift — a bug
no backtest can find, because the backtest is not the thing that is broken.
The gap also drives the [live demotion gate](model-selection.md#6-the-live-demotion-gate).
