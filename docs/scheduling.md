# Scheduling: the four crons that feed the system

The archive is the binding constraint on everything this project can learn
([Limitations §1](limitations.md)) — and three of the four inputs cannot be
backfilled after the fact. These launchd templates (in
[`docs/launchd/`](https://github.com/hbmartin/grounded-weather-forecast/tree/main/docs/launchd))
keep the pipeline fed unattended on macOS. On Linux, translate each to a
systemd timer or crontab line; the cadence rationale is identical.

## The jobs and how often to run them

| Job | Cadence | Why this cadence |
|---|---|---|
| `poll` — the upstream `omni-weather` collector | **hourly** | Every missed hour of provider vintages is unrecoverable. Hourly catches every provider's update cycle while staying inside free-tier quotas (the upstream quota tracker enforces per-provider caps). If quotas pinch, drop to per-run-cycle (6-hourly) for the slow-refresh providers via the upstream config, not by slowing this job. |
| `ingest-ensembles` | **every 6 h** (`StartInterval` 21600) | Ensemble models run on 00/06/12/18 UTC cycles and Open-Meteo retains only the latest run's members — a missed cycle's spread is gone forever. A fixed 6-hour interval catches every cycle regardless of publication lag; the as-of join tolerates any offset. |
| `publish` (the `predict` launchd job) | **every 10 min** (`StartInterval` 600) | Matches the 10-minute snapshot grid, so each accepted serve sees at most one new snapshot. Ready forecasts replace the published document and enter self-verification history; a degraded candidate holds the last ready document without polluting history. Widen to 15–30 min if the machine is battery-constrained; the cost is coarser verification history, not correctness. |
| `maintain` — `build-dataset` → `backtest --source live` → `truth-qc` → `report` | **daily, 02:15 local** | The retrain loop: refreshed truth and ensemble features, refreshed evidence, re-promoted winners in the release ledger, and the neighbor/shield sensor checks. Schedule it after the latest successful ensemble ingest: ensemble rows become model features only when `build-dataset` rematerializes the matrix. Daily is the right floor — truth accrues by the hour but promotion decisions move on days. As the archive and method count grow, backtest runtime grows too; if the nightly run gets slow, pass a curated `--methods` subset nightly and run the full sweep weekly. Follow with `prune-scores` periodically (or append it to the chain): superseded scores files accumulate at nightly cadence. |

The `backfill` commands are deliberately *not* scheduled: they are one-off
cold-start tools, and re-running them is idempotent but pointless on a cron.
If an ensemble ingest runs after maintenance, rebuild the dataset again before
backtesting or serving methods that consume ensemble features.

## Installing

1. Install the shared launcher on the internal disk, then copy the four plist
   templates into `~/Library/LaunchAgents/`. Fill their `__PLACEHOLDERS__`
   (including the absolute `__LAUNCHER__` and `__UV__` paths for `maintain`
   and `predict`, repository paths, log directory, coordinates, output path,
   forecast database, and Synoptic token). Keep each label matching its filename.

   ```bash
   mkdir -p ~/Library/LaunchAgents ~/.local/state/grounded-weather-forecast \
     "$HOME/Library/Application Support/grounded-weather-forecast"
   cp docs/launchd/launch-grounded \
     "$HOME/Library/Application Support/grounded-weather-forecast/launch-grounded"
   chmod 755 "$HOME/Library/Application Support/grounded-weather-forecast/launch-grounded"
   for job in poll ingest-ensembles predict maintain; do
     cp "docs/launchd/com.grounded-weather-forecast.${job}.plist" ~/Library/LaunchAgents/
     "$EDITOR" "$HOME/Library/LaunchAgents/com.grounded-weather-forecast.${job}.plist"
   done
   ```

2. Load the jobs (modern launchctl syntax). `maintain` and `predict` run on
   their schedules; loading them does not start an extra run.

   ```bash
   for job in poll ingest-ensembles predict maintain; do
     launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.grounded-weather-forecast.${job}.plist"
   done
   ```

3. Verify and watch:

   ```bash
   for job in poll ingest-ensembles predict maintain; do
     launchctl print "gui/$(id -u)/com.grounded-weather-forecast.${job}" | head
   done
   tail -f __LOG_DIR__/predict.log
   ```

   To unload one job, run
   `launchctl bootout "gui/$(id -u)/com.grounded-weather-forecast.predict"`;
   substitute any of the other three labels as needed.

## Standalone Python wrappers

The standalone Python wrappers in `docs/launchd/daily-maintenance.py` and
`hourly-predict-publish.py` run only through guarded `main()` entry points.
Importing them does not change the environment or working directory, register
exit logging, or execute jobs. Their entry points default `POLARS_MAX_THREADS`
to **2** and log the job's start/end and exit status.

Install `launchd_paths.py` alongside both standalone wrappers:

```bash
cp docs/launchd/{launchd_paths.py,daily-maintenance.py,hourly-predict-publish.py} \
  "$HOME/Library/Application Support/grounded-weather-forecast/"
```

The project directory comes from the job's working directory, selected by
`uv --directory`. Both wrappers validate the project identity and parse its TOML
once; CLI children perform scientific validation. Prefer the existing `UV`
executable, falling back to `PATH` only when unset. Invalid explicit executables
fail with exit 2, and launcher symlinks remain intact for macOS permissions.
Configuration and argument errors have START/END records, including stripped
Python docstrings and unavailable working directories.

The hourly wrapper predicts before discovering the sibling `ambientweather2sqlite`
project. Use `--station-project /path/to/collector` for another layout. The station
child uses the collector's working directory and environment, allowing its existing
`aw2sqlite.toml` or home configuration to select the live database. It does not use
the forecast evaluation `station.db_path`, which may be a static snapshot.
Station-only failures reuse staged documents; without staged documents the wrapper
refuses publication. Forecast runtime configuration errors still exit 2 before
prediction. No volume or personal database defaults are embedded in the wrappers.

The daily wrapper holds `maintenance-job.lock` across its child stages using
a context manager, releasing it on early returns and exceptions. Concurrent
maintenance or restore jobs skip as duplicates. A successful daily run writes
`last-maintenance`; a fresh stamp skips catch-up for **20 hours**. Restore runs
leave that stamp untouched, and a failed report skips score pruning.

CLI `maintain` also runs truth-QC before report. Build and ordinary backtest errors stop the
pipeline; coordination failures still stop immediately. Promotion-only evidence/publication errors
preserve completed scores and continue truth-QC/report while retaining failure
status only when evaluation itself succeeded. Product failures outrank promotion
errors and stop diagnostics; nullable `failure_details_json` retains both components
without duplicate product journal rows. Completed independent products may still
promote unless coordination or promotion validation prevents publication.
The standalone wrapper uses a private temporary child-outcome file, checked
against the process exit code, to apply the same continuation boundary. Missing
or invalid metadata never exempts a failed evaluation; logs remain diagnostic text.
Promotion evidence conflicts and missing required
contexts record `promotion_evidence_conflict` or `promotion_evidence_invalid`,
publish no approval changes, and preserve `last-maintenance`. An insufficient-history
backtest (exit 3) and a handled truth-QC
failure permit diagnostics under the same lock. Pending evaluation returns 0
when the other required stages pass; a real truth-QC failure remains nonzero.
The cross-check panel shows the retained verdict’s refresh time: amber after
a failed refresh, beyond 48 hours (two nightly cycles), or when freshness is
unknown. A latched drift verdict remains red; refreshing the dashboard does not
refresh the underlying QC verdict.
Maintenance returns the first nonzero stage result after reporting. Failed QC
refreshes preserve the previous artifact and remain in the run ledger; a fresh
dashboard can therefore contain an older QC verdict. Its central alerts share
the panel’s freshness assessment, and failed standalone/stage attempts expose
a bounded ledger detail. Individual chips retain their own status. Unexpected exceptions
still stop the pipeline. A report read fault preserves available outputs and
returns failure after truth-QC has been attempted. Corrupt scores are not repaired by pruning or
automatic quarantine; follow the [manual recovery procedure](reference/cli.md#recovering-from-a-damaged-score-file).

The hourly wrapper checks whether a degraded document says `implementation
changed` before starting detached restore maintenance. The restore child
acquires the job lock itself; the hourly eligibility check does not probe it.
Healthy hourly checks therefore cannot prevent a daily job from acquiring its
lock. Publication continues while restoration runs, retaining the existing
last-good behavior. Installing updated wrappers and refreshing compatible live
evidence are deployment steps: source changes, including diagnostic corrections,
alter the code fingerprint. Run a fresh live backtest and report with the updated
code before rollout. Rainfall guards that remove no release preserve the
implementation-change diagnosis required by the wrapper's restore check.

Rain interval-alignment changes also require rebuilding live and synthetic
matrices before regenerating their score evidence. Existing synthetic forecast
and ensemble archives retain their native interval-end timestamps and need no
rewrite; normalization occurs when they are consumed. Rebuild synthetic
matrices with a backfill invocation (which merges the retained archive), then
regenerate synthetic evidence where used. Run `build-dataset`, a fresh
`backtest --source live`, and `report` under the updated code before rollout.
Source implementation and PR delivery do not install the deployment.

Backtest exit 3 covers valid short history and test truth still future at a
hash-validated observation snapshot cutoff (`truth_snapshot_at`). The manifest
publication time (`created_at`) cannot substitute for that cutoff. Frozen snapshots retain their classification;
stale archives are a separate health alarm. Legacy files without metadata remain
scoreable but cannot receive the future-truth exemption. Malformed maturity
clocks follow the same rule; integrity failures still fail. Rebuild to establish
the clock. Completed products may promote independently while real errors determine
the overall exit. Entirely inapplicable requests exit 2.

Backtests own release changes; hourly prediction consumes approved selections.
Compatible data-only rebuilds preserve incumbents with their original evidence
identity. Pending/failed products and slices without a qualifying replacement do
not lose approvals. Code, config, product source/feature, or truth-semantic changes
still invalidate incompatible approvals. Live-gate method changes occur during
backtests, including checks of carried choices in completed products when no new
winner qualifies; prediction-time safeguards remain active. Separate approval
events order decisions without rewriting release content. Reapproving an older
release becomes current immediately; unchanged reruns add no event. New release
content is ignored until its approval event commits.

Current managed approvals follow committed sequence across forward and backward
clock corrections. Historical queries wait for the conservative `visible_at` floor
and the publication/approval clocks. These additive fields do not change content
identity, and legacy events derive floors without migration. Operational pruning
protects current approvals plus recent served references when the active pointer
is genuinely absent; explicitly referenced unreadable outputs suspend score deletion.

A failed optional restore launch never prevents the station/publication stages.
If publication succeeds, the hourly job still returns 1 to expose the failed
launch, including ordinary launch argument and subprocess exceptions. Cancellation
still propagates. The support directory is initialized before job-lock acquisition;
setup and policy validation return 2; later filesystem failures,
including stamp updates, return 1. Late synthetic imports fail that stage while
remaining diagnostics continue. Pending or failed runs do not advance the stamp.
A secondary failure checking score coordination records both errors as
`score_coordination_probe`, stops remaining score writers, and suppresses promotion.
Code-change diagnostics take precedence over dataset rotation so hourly restoration
is still triggered. Approvals for minutely alone call for a live backtest to establish
hourly/daily approvals rather than merely more polling.

Pending embedded stages are visible in the current report and deduplicated against
completed parents. Failed stages are counted provisionally before parent recording;
pending metadata cannot hide a real failure or exception.

The daily wrapper treats only backtest exit 3 as expected pending evaluation.
It continues diagnostics and returns 0 if all remaining required stages pass,
but preserves `last-maintenance` until every requested, applicable evaluation
completes. All other nonzero child exits remain failures. Logs identify full
stage commands and pruning is skipped when report fails. A later error cannot
inherit a pending exemption. CLI `recover` still exits 3 for pending evaluation
without claiming recovery completed.

The wrapper-only `[maintenance].synthetic_backtest` setting defaults to `auto`:
on UTC Sundays run when both configured hourly/daily synthetic matrices exist,
skip explicitly when both entries are absent in an accessible dataset directory,
and fail on broken links, missing storage, partial, unreadable, or corrupt inputs. Set `required` on hosts where backfill availability is mandatory; set
`off` to disable this scheduled evaluation. Invalid settings exit 2. Dataset paths
come from the active project's configuration; relative paths resolve from that
project. This scheduling setting does not change the scientific config fingerprint.
Routine scheduling settings are validated after the lock is acquired and before
the success-stamp check. Restore ignores synthetic settings, performs only live
evaluation, and leaves the success stamp untouched.
The hourly job's standalone `--restore` child returns 3 after diagnostics when
evaluation is pending, retaining the incomplete recovery status.

## Profiling a scheduled run

The optional Memray wrapper records allocations from the start of a command,
saves a unique capture for each invocation, and generates an HTML report after
it finishes. Install it in the source checkout with
`uv sync --locked --group profiling`. To invoke it unattended, use an absolute
`uv` path and project directory:

```bash
__UV__ --directory __REPO__ run --locked --group profiling \
  python scripts/profile_memory.py -- maintain
```

For the existing `maintain` or `predict` launchd templates, keep the shared
launcher, mode, repository, environment, and schedule. In `ProgramArguments`,
add `--group`, `profiling` to the `uv run` options and replace the
`grounded-weather-forecast` executable argument with these three arguments:

```xml
<string>python</string>
<string>scripts/profile_memory.py</string>
<string>--</string>
```

Keep the existing forecast command and its arguments after the separator (for
example, `maintain`, or `publish --out __OUT_JSON__`). Each value is a separate
plist argument. Apply this only to jobs you want to profile; profiling adds
runtime overhead and captures accumulate under `artifacts/memory/`.
The wrapper's stderr log prints the capture and report paths. See the
[CLI reference](reference/cli.md#memory-profiling-with-memray) for wrapper flags,
exit behavior, and later analysis commands.

## Notes

- The `maintain` and `predict` templates invoke a launcher copied to the
  internal disk. It checks that an external project volume is mounted and
  delegates project access to `uv --directory __REPO__ run --locked`, which
  establishes the working directory and rejects an outdated lockfile. Use
  absolute paths for the launcher and `uv`; no login-shell setup is needed.
  A mounted volume with a subsequent `Operation not permitted` error from `uv`
  points to macOS file-access permissions, not a missing volume.
- Diagnose startup from the log markers: no `START` means inspect the launchd
  timer and service state; `START` followed by a mount error means the volume
  was absent; `MOUNT_OK` followed by a `uv` error means the volume was present
  but project access or environment setup failed. Application output follows
  `INVOKE`; use `launchctl print` for the final exit status.
- A degraded `publish` candidate and a backtest finding no folds can occur
  during a cold start. A proven insufficient-history backtest exits 3. The daily
  wrapper reports evaluation pending and does not advance its success stamp;
  maintenance itself exits 0 when its other required stages pass.
  The candidate names its cause in `status_reason`; `publish`
  preserves a matching ready document issued within six hours byte-for-byte, or publishes
  and archives the degraded candidate on a cold start so an output still exists.
- launchd `StartCalendarInterval` fires in **local time**; the 6-hourly
  ensemble job uses `StartInterval` (elapsed seconds) precisely so daylight
  saving cannot skip a model cycle.
- `StartCalendarInterval` jobs missed while the machine is **off** are never
  replayed (asleep is fine — launchd coalesces on wake). A machine that
  shuts down overnight therefore silently skips that day's `maintain` run.
  Manually `kickstart` it after such a shutdown;
  the template does not run speculative maintenance at login.
- The station logger upstream (`aw2sqlite serve`) is a hard dependency of
  truth: run it as its own `KeepAlive` LaunchAgent, never from a Terminal
  session — a closed session or reboot otherwise stops truth silently while
  forecasts keep accruing (a week of unusable, label-less snapshots).
- The scheduled `publish` run appends only accepted ready forecasts (and a
  degraded cold-start forecast) to self-verification history. Rejected degraded
  candidates are neither published over a ready document nor appended. Manual
  `predict --no-history` remains the opt-out for experiments.
- `grounded-weather-forecast prune-scores` (preview with `--dry-run`) deletes
  superseded backtest scores files: the newest three per
  (product, source, window), anything referenced by the active release, and
  anything a release promoted in the last 7 days are kept. Files the evaluations
  catalog has never seen are never deleted.
  Every completed check journals clear/suspended protection and its recording
  time, including dry runs. The next report/dashboard reads that result rather
  than rescanning the served archive. With report before prune in the daily chain,
  the displayed verification can be from the preceding run; its timestamp makes
  that explicit. Pruning still freshly verifies before deleting, and protection
  suspension keeps exit 0. Check age and the latest attempt are distinct; completed
  verification survives later errors. Failed/unverified attempts do not erase a
  suspension, and legacy attempts do not establish a clean check. A later explicit
  clear successful check replaces a protection suspension. Only a successful clear
  non-dry-run prune clears an execution failure. Check metadata takes precedence
  over recorded arguments; unknown legacy execution mode cannot establish recovery.
  Lock contention (75) is a distinct informational attempt, not a new protection
  failure, and clears no older warning. Relevant malformed newer run evidence
  is unknown/amber rather than exposing an old success as current.
  Malformed verification groups preserve unrelated valid checks and set the report's
  `verification_unknown` field. The last valid check remains visible, with amber
  uncertainty until a clear check is recorded strictly after the damaged run ends.
  A clear dry run may resolve this check uncertainty; execution recovery still
  requires a successful clear apply run. A successful apply with an independently
  valid selected clear check clears an execution failure despite unrelated damaged
  evidence; unresolved verification uncertainty independently keeps the panel amber.
  Missing/non-string stages and unattributable malformed entries also create
  uncertainty. Equal-clock valid checks continue to use persisted order; raw legacy
  clock strings preserve old grouping boundaries.
  Optional diagnostic merging/serialization stays inside the recorder's protected
  boundary, and progress capture is independently safe. Invalid fields retain bounded
  evidence; a fallback preserves the core row without replacing command outcomes or
  original exceptions/cancellation, subject to existing best-effort ledger storage.
  Summary boundaries make concatenated batch delivery idempotent while preserving
  interleaved order and duplicate sample occurrences. Clear and suspended legacy
  verdict-only rows do not become issues. A suspension without samples/counts shows
  an unavailable count through nullable `issue_count_known` metadata.
  Each check records at most 20 issue samples with a full JSONL audit reference.
  Audits under `artifacts/history/pruning_verifications/` preserve every affected
  path, survive for 90 days and while referenced by retained run rows, and are
  cleaned only when verifiably old/unreferenced. Dry runs can record an audit
  but never delete one or an abandoned audit stage. Completed bounded progress
  is recorded before audit I/O and enriched under the same check ID afterward.
  Catchable interruptions preserve it; SIGKILL and power loss do not guarantee
  journaling. Audit publication requires hard links and file/directory synchronization.
  A dedicated nonblocking audit lock coordinates publication and cleanup; contention
  is informational. Known regular unchanged stages older than 90 days can expire;
  fresh, active, unrecognized and unverifiable files remain. Audit failures remain
  explicit without changing pruning exit semantics. Reports/dashboard display check
  age and dry-run mode using their own cutoff; audit availability and bounded
  compatibility notes remain visible independently of issue samples.
  A linked complete audit is recorded immediately as an unconfirmed candidate;
  confirmation requires synchronization and successful lock release. Retained runs
  protect both references. Catchable interruptions preserve the candidate without
  claiming crash durability. Public diagnostics use safe category/reason templates;
  raw paths and causes remain in diagnostic evidence. File-local cleanup failures
  do not block independent candidates; global reference uncertainty still does.
  Bounded cleanup notes always retain an abort summary and omitted count. Their
  public text uses fixed safe explanations; original details and nullable paths
  remain in informational run diagnostics without affecting failure counts or exits.
  Audit publication preserves its operation and known cause separately. Only an
  actual audit-lock acquisition timeout is contention; a busy directory means no
  publication for that check, with no scheduled retry. Dry runs retain the same
  revalidation and uncertainty diagnostics while deleting nothing. Compatibility
  notices, including legacy strings, remain informational rather than claiming
  protection failure. Legacy note counts include all notes; only structured
  notices receive per-item sampling and omission counts. Every cleanup records
  one informational summary with nullable mode and completed audit/staging deletion
  counts. Eligible audit/staging counts are tracked separately after reference and
  signature checks; dry runs preview these while actual deletion counts stay zero.
  Aborts report eligibility verified so far, and a later lock-release failure
  preserves earlier completed deletions without implying rollback. Unknown legacy
  execution remains unverified; a default result cannot establish completion.
  One validated outcome drives all cleanup rendering and emitted completion flags;
  counts do not prove that the entire cleanup completed. Raw diagnostic text is
  literal at construction; labelled legacy input uses an explicit adapter. Bound
  cleanup records keep their source and check identity through copies, rebinding
  only when either changes. A stable nullable `progress_id` survives copied check-ID
  changes; new physical checks receive separate identities. Journaling and progress
  capture replace earlier verification and cleanup batches per stage and logical
  check, preserving independent checks and captured progress through exceptions.
  Legacy IDs remain readable, and empty diagnostic tuples retain omitted-batch
  compatibility. Compact failure-only replacement removes old tails; sparse
  metadata annotation requires a unique association, with nullable aligned input
  for duplicate issues. Abort and file-local roles are recorded at capture;
  filtering never relabels the next failure, and known abort evidence must be
  corrected or replaced before a completed copy is accepted. Unknown legacy roles
  remain neutral. Identical samples survive journal merging within each identified
  batch; repeated delivery still replaces the batch. Progress-only checks retain
  identity through reload and remain independent at equal clocks. JSON omits only
  progress IDs equal to the row's check ID, preserving renamed sample ownership.
  No execution or recovery policy changes. Staging sync
  failures before linking report no publication; post-link failures preserve an
  unconfirmed candidate. Historical contention/link reasons prove no publication
  when no linked candidate exists; unknown sync phases remain neutral. Cleanup samples
  are single-line and bounded to 400 characters, with full raw evidence retained
  internally and never rendered publicly.
- `maintain` waits for `<dataset dir>/pipeline.lock` and holds it once across
  build → live backtest → truth-QC → report. Interactive mutators wait 60 s and
  exit `75` on contention. This prevents another command from slipping between
  scheduled steps. `predict` and `publish` copy and validate the live dataset
  under the short `dataset.lock`, then select and fit outside it;
  `ingest-ensembles` retains its independent store lock.
- Every degraded candidate is eligible for automatic recovery, not only a
  code-identity mismatch. `publish` derives a signature from the reason plus
  config/code/recovery-profile identities and starts detached `recover` at most once per
  unchanged signature every six hours; a changed signature is immediately
  eligible. `recover` deduplicates concurrent requests, waits behind maintenance,
  rechecks readiness, then runs build → live backtest → report only if still
  degraded. State and output live in `artifacts/auto-restore.json` and
  `artifacts/auto-restore.log`.
- Every `publish` attempt writes `artifacts/latest_publish_attempt.json`, so
  alerts can show a degraded candidate even when an earlier ready document is
  held. The local `active_release.json` tracks the document at `--out`; the
  personal Git publisher may subsequently hold or fail to push that document.
- Keep the Synoptic token out of the plist if you prefer: set
  `synoptic_token = "$SYNOPTIC_TOKEN"` in `config.toml` and provide the
  variable via `launchctl setenv SYNOPTIC_TOKEN ...` instead of the
  `EnvironmentVariables` block.

### Verification failures and approval rollback

Unverifiable retention inputs suspend published-score deletion with exact paths in
`skipped` and dashboard artifact warnings. This keeps the existing pruning exit
contract; inspect repeated warnings and repair or restore the named artifact. A
corrupt pointer is a verification fault, while genuine absence permits the current
approval fallback. Unreadable archives and event/directory errors are also faults.

`promotion_publication_failed` and `promotion_snapshot_stale` preserve completed
evaluations and continue maintenance diagnostics only when evaluation succeeded.
They return failure and never advance `last-maintenance` or complete recovery.
The child-only outcome environment leaves the wrapper environment unchanged.

Events written since #79 carry `visible_at`; pre-#79 readers reject that additive
key. Before rolling back, backport the compatible reader. Do not rewrite events,
publication clocks or visibility floors. Current serving follows sequence across
clock corrections; historical queries can conservatively remain behind an earlier
future-skewed floor. Identify the earliest elevated event to explain the cutoff,
without changing append-only evidence.
