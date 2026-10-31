# grounded-weather-forecast

Station-grounded blending of multi-provider weather forecasts: bias correction,
anchoring, and blending judged by a rolling-origin backtest leaderboard.

grounded-weather-forecast turns two SQLite files — a personal weather station's minute-level
observation log ([ambientweather2sqlite](https://github.com/hbmartin/ambientweather2sqlite))
and a multi-provider forecast archive
([omni-weather-forecast-apis](https://github.com/hbmartin/omni-weather-forecast-apis)) —
into three forecast products for one location:

- **next hour, by minute** — an anchored nowcast blending the live station reading into
  the hourly blend, plus native minutely precipitation where providers supply it
- **next day, by hour**
- **next 10 days, by day**

## How it works

Three composable stages. Nothing ships because it sounds good — a stage is used
for a given variable and lead time only if it wins that slice on the backtest
leaderboard.

1. **Grounding** — per-source correction toward the station, fitted per
   variable × lead bucket. Most providers repackage the same global models, so
   their *shared* bias is invisible to any weighting scheme; only correction
   removes it. A bias correction by default — the slope is opt-in, for reasons
   the data taught us (see [ADR 0004](https://hbmartin.github.io/grounded-weather-forecast/adr/0004-grounding-defaults-to-bias-only/)).
2. **Blending** — combining grounded sources: equal weight, trimmed mean
   (drops the extremes per row — robustness with zero parameters),
   inverse-MSE and inverse-MAE weighting, gradient-boosted stacking, and
   online expert aggregation with sleeping experts (ragged provider horizons
   need no special casing) and fixed share (so a provider that silently swaps
   its backend model loses weight in days). Grounding also comes in a
   MAE-consistent median-intercept variant.
3. **Anchoring** — short-lead correction toward the latest live observation,
   decaying exponentially with lead. Your thermometer is the one input no
   provider has.

Ground truth is QC'd (plausibility bounds, spike and flatline filters) and
aggregated from minute data. Provider forecasts get their own conservative
QC before grounding: absolute physical bounds plus a robust cross-source
outlier pass (temperature, humidity, pressure, dew point, and — since a
provider stored a 138 mm cloudburst for a bone-dry day — precipitation,
whose generous floors null a lone hallucination but never drizzle
disagreement or genuine storm consensus). Scoring uses MAE/RMSE/bias, CRPS, and
Brier/reliability for precipitation probability, with Diebold–Mariano
significance per variable × lead bucket, under strict rolling-origin splits.
Live and synthetic (backfilled) data are never pooled.

## Installation

grounded-weather-forecast requires Python 3.13 or newer. Install the command in an isolated
environment with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install grounded-weather-forecast
grounded-weather-forecast --version
```

## Usage

Download the [example configuration](https://github.com/hbmartin/grounded-weather-forecast/blob/main/config.example.toml),
edit the two database paths and your station's coordinates, then:

```bash
grounded-weather-forecast qc              # check your column mapping is right
grounded-weather-forecast build-dataset   # truth tables + supervised matrices
grounded-weather-forecast predict         # a forecast, as JSON on stdout
```

That works on day one. Everything else needs an archive with some history — see
[Getting started](https://hbmartin.github.io/grounded-weather-forecast/getting-started/).

### Commands at a glance

| Command | What it does | Flags |
|---|---|---|
| `qc` | summarize station truth quality control | — |
| `read-daily-report` | read schema-1 daily JSON with write-free SQLite snapshots, fail-closed WAL fallback, deterministic attributed quality using scored-issue recency (unresolved ties stay unknown), and independent component retries; missing realized recency is explicitly null; see [caller contract](https://hbmartin.github.io/grounded-weather-forecast/reference/cli/#read-daily-report) | `--start` `--as-of` `--hours` `--forecast` `--published-station` |
| `build-dataset` | materialize truth tables and supervised matrices as parquet | — |
| `backtest` | recover abandoned leased score stages, then run a rolling-origin backtest; allocation and publication each wait at most 60 seconds; a startup-recovery or uncommitted writer guard-release failure returns 1 and suppresses promotion | `--methods` `--hourly-variables` `--daily-variables` `--products` `--window` `--source` `--semantics` |
| `maintain` | run one locked live build → backtest → truth-QC → report transaction; finish report after a handled truth-QC failure, then return failure | `--methods` `--hourly-variables` `--daily-variables` `--products` `--window` `--semantics` `--truth-qc-days` |
| `report` | leaderboards, evidence ledgers, and dashboard; skip missing scores, show read faults and return nonzero after partial output; one load per file with `mcs`/`legacy` | — |
| `alignment` | study truth semantics per provider; write the alignment artifact | — |
| `backfill` | fetch archived forecasts into the synthetic supervised matrix | `--provider` `--models` `--start` `--end` `--chunk-days` |
| `truth-qc` | cross-check truth against lapse-adjusted neighbors and fit the radiation-shield error model | `--days` |
| `ingest-ensembles` | poll the Open-Meteo Ensemble API for per-model spread features | `--models` |
| `predict` | consume approved releases without electing methods; emit schema-6 JSON with rainfall guards and finite-value release attribution; `--method` bypasses automatic rain safeguards for experiments | `--out` `--method` `--no-history` `--semantics` `--now` |
| `publish` | safely publish an automatic forecast, holding a matching ready document for up to six hours | `--out` `--semantics` `--recovery-methods` `--recovery-window` |
| `recover` | recheck degradation, then rebuild live evidence and reports when recovery is still needed | `--semantics` `--methods` `--window` |
| `hourly-predict-publish.py` (standalone launchd wrapper) | predict first, then render live station data through collector configuration; retain staged station files on render failure | `--station-project` |
| `prune-scores` | delete superseded scores and recover inactive leased stages one at a time; `--dry-run` updates verification but cannot clear execution failures; retain completed checks through audit interruptions and later failures, distinguish contention/unknown attempts, and display check age; unverifiable protection suspends score deletion; full JSONL audits retain all issues while samples stop at 20; safe category/reason explanations, including cleanup aborts, keep raw paths in diagnostics; audit failures preserve both operation and cause; compatible legacy notes remain informational; audit publication requires hard links and synchronization, and linked but unconfirmed audits remain separately referenced and protected; cleanup previews verified eligible audit/staging counts separately from completed deletions and mode, leaves unknown outcomes unverified, preserves literal raw evidence and uniquely attributable sparse metadata with aligned compatibility input, restores failure-only replacement, keeps copied verification/cleanup batches under one stable journal identity, preserves duplicate samples and progress-only checks, isolates malformed check evidence with an explicit `verification_unknown` warning until a strictly later clear check, contains malformed fields inside the recorder while retaining a core row, preserves interleaved sample order, separates apply-run execution recovery from unrelated verification uncertainty, and labels sample-free legacy suspensions with an unknown count, records abort/file roles without positional inference, rejects completed copies retaining abort causes, preserves bounded abort summaries, and continues after file-local failures; publication diagnostics distinguish pre-link failure from linked-but-unconfirmed content; recovery guard-release errors retain partial counts, finish published-score retention, then return 1 | `--dry-run` |
| `uv run --group profiling python scripts/profile_memory.py -- <command>` | capture a command's memory allocations and write a peak-memory HTML report | `--output-dir` `--trace-python-allocators` `--no-native` `--no-report` |
| `uv run python scripts/benchmark_selection_memory.py` | compare full selection runs against a Git baseline in fresh processes | `--baseline` `--runs` `--evaluations` `--cases` |
| `uv run python scripts/benchmark_backtest_memory.py` | compare streamed score generation and writing against an immutable Git baseline | `--baseline` `--runs` `--rows` `--output` |
| `uv run python scripts/benchmark_fit_diagnostics.py` | compare diagnostic collection and real fitting with verified source imports; journal completed samples, refuse overwrites, and rescue failed runs | `--baseline` `--runs` `--output` |
| `uv run python scripts/benchmark_score_readers.py` | compare selection, dashboard, and report score readers; journal completed samples, use unique ignored outputs, and rescue failed runs | `--baseline` `--runs` `--evaluations` `--cases` `--output` |

Global: `--config PATH` (default `config.toml`), `--version`. Exit codes: `0` ok,
`1` command failure, `2` config error, `3` evaluation pending because history is
too short or test truth is still future at the observation snapshot cutoff, `75` lock contention (retry later).

Backtests own promotions and live-gate method changes. Prediction reads approved
choices across compatible rebuilds: code, config, product sources, features, and
truth semantics must match. Retained choices keep their original evaluation
dataset identity. Pending, failed, unrequested, and completed slices with no
qualifying replacement keep compatible incumbents. Completed products' carried
choices still pass the live gate; pending, failed, and unrequested products retain
their approvals. Immutable release content has separate ordered approval events:
A → B → A reapproves A, while A → A adds no event. Event-managed releases are not
visible before their approval commits. Committed sequence determines approval
order for current serving, even across clock corrections. Historical reads use
conservative `visible_at` floors: the maximum of the actual approval clock,
stored content publication clock, and preceding managed visibility times.
Older events derive these floors without rewrites. Malformed, timezone-less,
and overflowing clocks remain invalid. New release objects require an event by default; legacy JSON
without the marker retains implicit approval. First approval of existing content
is distinguished from reapproval. An unchanged release is not reported as newly promoted. Input hashes remain mandatory when a manifest is
present; missing or malformed maturity clocks do not prevent positive scoring.
The additive manifest field `truth_snapshot_at` records the observation-read
cutoff; `created_at` records publication. Legacy manifests remain scoreable but
cannot establish future-truth pending status.

Blocked live-gate measurements are recorded in `artifacts/history/live_gate.parquet`
and shown in the dashboard, without changing carried approval reasons or creating
new releases. Known legacy evaluation references remain protected even without
full contexts; explicit active/recent-served legacy references protect scores
without authorizing serving. A per-slice `legacy_context_missing` marker preserves
exact-dataset restrictions when other selected variables share the evaluation ID.
The historical `"unknown"` placeholder remains missing provenance. Churn keeps publication and approval clocks separate,
with additive nullable approval timestamps, sequences and a clock-schema version.
If the active pointer is genuinely absent, retention protects both the newest current
approval and recent served references. Corrupt pointers, unreadable archives/events,
denied listings and uncertain legacy clocks suspend published-score deletion with
specific diagnostics. Explicit empty pointers retain degraded-output semantics;
pruning exit codes remain unchanged. Repair the named source before expecting
retention to resume.

Historical pruning journals retain detail-only samples, including duplicates;
only the documented fallback verdict markers may represent sample-free checks.
Unknown aggregate totals and omissions display as unavailable. Valid final
snapshots replace damaged progress using stage and identity, while cleanup damage
stays separate from verification. Damaged cleanup evidence has its own amber
warning, cleared by a valid cleanup summary recorded strictly after the damaged
run ends; dry runs and incomplete cleanup outcomes can supply valid evidence.
Legacy absence means cleanup was not recorded. Verification and apply-run
execution recovery remain independent of this warning.

Command outcomes are normalized before policy, hint transport, and journaling.
An invalid exit (including a float, string, boolean, or integer outside the ledger
range) returns 1 with `invalid_command_outcome`, never a coerced success or pending
exemption. Invalid optional classifications retain valid exits and bounded damage
evidence. Explicit invocation exceptions keep callers' handled exceptions out of
the run journal. Optional hint failures cannot prevent the recording attempt;
original exceptions and cancellation still propagate.

Daily reads query rollback-mode SQLite bytes in memory and never pass source
filenames to SQLite. WAL or journal-dependent sources fail closed: station reads
use the published JSON fallback and provider health becomes explicitly unknown.
Quality keeps evaluation provenance across compatible dataset rebuilds. Realized
evidence prefers the exact release; otherwise a compatible older cohort is labelled
with nullable `evidence_match`, `evidence_release_id` and
`evidence_dataset_fingerprint`, while the forecast release identity remains intact.
No cohort scores are pooled. Changed inputs invalidate only dependent components.

Approval verification diagnostics are shared by pruning and dashboard warnings.
Expected approval publication failures record `promotion_publication_failed`; stale
elections record `promotion_snapshot_stale`. Both retain completed evaluation
diagnostics and permit maintenance diagnostics when evaluation succeeded. Events
remain append-only. Rolling back below the #79 approval reader requires a parser
compatibility backport: old readers silently discard events containing `visible_at`.

The hourly wrapper publishes even if optional restore launch fails, then returns
1 so monitoring sees that failure. Startup/configuration errors return 2; later
runtime errors return 1.
If the coordination check after a product failure also fails, the command records
`score_coordination_probe` with both errors, stops remaining score writers, and
suppresses promotion. Prediction reads observations once for truth and causal
features; code-change diagnosis takes precedence over data rotation so restoration
remains actionable. Minutely-only approvals do not satisfy hourly/daily readiness.

Score guard-release failure during startup recovery or an uncommitted write
returns 1 from backtest, maintain or recover, records `score_coordination_release`
and its diagnostic in the run journal, and suppresses promotion, including when
unlocking fails alongside another uncommitted publication error. Startup recovery
reports completed cleanup and stops before invoking writers. Recovery preserves
secondary cleanup diagnostics after payload disappearance, preserves pending
cancellation, and attaches partial counts/bytes to unexpected errors. Standalone
programming errors during lease release or companion removal propagate. A failed
guard release marks that directory for the rest of the process: later acquisitions,
including through aliases or other threads, fail immediately with exit 1 rather
than waiting for contention. A committed publication remains successful with a
warning; a final-product warning alone does not suppress promotion. Pruning retains
its partial progress, finishes published-score retention, then returns 1 and journals
`score_coordination_release`, including with `--dry-run`. Recovery stops on
directory coordination I/O failure or observed directory disappearance; removal
after the existence check can still race with parent creation by project helpers
or FileLock. Disposal remains best effort with a zero-wait guard attempt, even after publication, so
recoverable payloads or companions may remain. Legacy temporaries stay untouched.
Pipeline mutators serialize on `<dataset dir>/pipeline.lock`;
scheduled `maintain` waits for it and holds it across the whole transaction.
Build and ordinary backtest failures stop CLI maintenance. Promotion-evidence
failures preserve completed scores and allow truth-QC/report diagnostics, while
maintenance still returns failure. Coordination failures still stop immediately.
Evaluation failures outrank promotion-only errors; both are recorded in the nullable
`failure_details_json` run-ledger field. Completed independent products may still
promote when another product fails, unless coordination or evidence validation prevents it.
Conflicting selected provenance blocks every approval change and records
`promotion_evidence_conflict`; missing new evaluation context records
`promotion_evidence_invalid`. Neither failure advances the success stamp. Insufficient-history
backtests (exit 3) continue to diagnostics without a failure alarm. A handled truth-QC
failure still permits report, which may use the previous QC artifact; the failed
refresh stays in the run ledger and maintenance returns the first nonzero stage
result. A refreshed dashboard therefore does not imply successful maintenance.
The truth cross-check panel retains the available verdict and chart, showing amber
after a failed QC refresh, beyond 48 hours, or when refresh time is unknown; a
latched drift verdict remains red. Serving diagnoses name affected promoted
variables and distinguish causally excluded training from missing matrices or
invalid truth. Pending truth can become eligible as live issuance advances; future-issued
archive rows remain excluded at a fixed historical `--now`. A rebuild does not
make future evidence available at that issuance. QC chips show their own health;
the central alert list shares the panel’s 48-hour freshness and failure assessment.
Failed QC runs retain a bounded diagnostic detail in the ledger.
`predict` and `publish` copy and validate the live dataset under the short
`<dataset dir>/dataset.lock`, then select and fit outside it.

**Full semantics for every flag — defaults, choices, what each command reads and
writes — are in the
[CLI reference](https://hbmartin.github.io/grounded-weather-forecast/reference/cli/).**
Every configuration key is in the
[configuration reference](https://hbmartin.github.io/grounded-weather-forecast/reference/configuration/),
and the emitted document is specified in
[Forecast JSON](https://hbmartin.github.io/grounded-weather-forecast/reference/forecast-json/).

### Running it for real

Steady state is four crons — poll, ingest ensembles, `publish`, and nightly
`maintain`. Cadence rationale and
launchd templates are in
[Scheduling](https://hbmartin.github.io/grounded-weather-forecast/scheduling/).
The daily wrapper treats backtest exit 3 as pending and continues diagnostics;
it returns 0 if the remaining required work passes, but preserves its success
stamp until every requested, applicable evaluation completes. All other nonzero
stage exits fail maintenance. Sunday synthetic evaluation follows the wrapper-only
`[maintenance].synthetic_backtest` setting (`auto`, `required`, or `off`; default
`auto` skips when both backfill matrices are absent in an accessible dataset directory).
Completed products may promote independently; unsupported individual products
are skipped, while entirely inapplicable requests exit 2. Pending stage counts
are visible during report generation and deduplicated against their parent run.
The hourly wrapper prefers `UV` over `PATH` and resolves station prerequisites
after prediction. Its collector working directory selects `aw2sqlite.toml`; the
forecast evaluation database may remain a static snapshot.

## Status

Alpha, and honest about it: with a young forecast archive the backtest reports
that it has no folds rather than inventing a leaderboard, and `predict` refuses
to serve from stale provider data rather than guessing.

## Documentation

Full docs: **<https://hbmartin.github.io/grounded-weather-forecast/>**

**Using it**

- **[Getting started](https://hbmartin.github.io/grounded-weather-forecast/getting-started/)** — install, configure, first forecast
- **[Concepts](https://hbmartin.github.io/grounded-weather-forecast/concepts/)** — the ideas in plain language, no equations
- **[FAQ and troubleshooting](https://hbmartin.github.io/grounded-weather-forecast/faq/)** — every error message, and what to do about it
- **[Glossary](https://hbmartin.github.io/grounded-weather-forecast/glossary/)** — every term, one line each

**Operating it**

- **[Advanced usage](https://hbmartin.github.io/grounded-weather-forecast/advanced-usage/)** — backfilling, tuning, adding your own blending method
- **[CLI](https://hbmartin.github.io/grounded-weather-forecast/reference/cli/)** · **[Configuration](https://hbmartin.github.io/grounded-weather-forecast/reference/configuration/)** · **[Forecast JSON](https://hbmartin.github.io/grounded-weather-forecast/reference/forecast-json/)** · **[Outputs](https://hbmartin.github.io/grounded-weather-forecast/reference/outputs/)** — complete references
- **[Scheduling](https://hbmartin.github.io/grounded-weather-forecast/scheduling/)** — the four crons, with launchd templates
- **[Operator dashboard](https://hbmartin.github.io/grounded-weather-forecast/dashboard/)** — the nine-zone offline console `report` writes

**The mathematics**

- **[Theory and concepts](https://hbmartin.github.io/grounded-weather-forecast/theory/)** — why grounding beats weighting, what the
  forecast-combination puzzle costs you, and how the evaluation is kept honest
- **[Methods](https://hbmartin.github.io/grounded-weather-forecast/methods/)** — ten deep-dive pages specifying every estimator, score,
  and decision rule as implemented: grounding, combination, calibration,
  uncertainty quantification, verification, model selection, precipitation,
  truth QC, and a [bibliography](https://hbmartin.github.io/grounded-weather-forecast/methods/bibliography/) matched to modules
- **[Architecture](https://hbmartin.github.io/grounded-weather-forecast/architecture/)** — layers, contracts, storage, libraries, leakage defences
- **[Limitations](https://hbmartin.github.io/grounded-weather-forecast/limitations/)** — what this cannot do, and the three real
  bugs the evaluation harness caught. **Read before trusting any number.**
- **[API reference](https://hbmartin.github.io/grounded-weather-forecast/reference/api/)** — the importable public surface

**Project**

- [`CONTRIBUTING.md`](https://github.com/hbmartin/grounded-weather-forecast/blob/main/CONTRIBUTING.md) — dev setup and the gates CI enforces
- [`CHANGELOG.md`](https://github.com/hbmartin/grounded-weather-forecast/blob/main/CHANGELOG.md) — release history and migration instructions
  (0.3.0 changed scoring semantics; re-run backtest before comparing eras)
- [`CONTEXT.md`](https://github.com/hbmartin/grounded-weather-forecast/blob/main/CONTEXT.md) — the project's naming discipline
- **[Decisions](https://hbmartin.github.io/grounded-weather-forecast/adr/0004-grounding-defaults-to-bias-only/)** — architecture decision records

## Development

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
uv run ruff check src --fix && uv run ruff format src tests
uvx --from semgrep==1.170.0 semgrep scan --test --config semgrep/provider-qc.yml semgrep/tests/provider_qc_grouping.py
uvx --from semgrep==1.170.0 semgrep scan --metrics=off --error --config semgrep/provider-qc.yml src/grounded_weather_forecast/dataset/matrix.py
uvx --from semgrep==1.170.0 semgrep scan --test --config semgrep/artifact-pointer-paths.yml semgrep/tests/artifact_pointer_paths.py
uvx --from semgrep==1.170.0 semgrep scan --metrics=off --error --config semgrep/artifact-pointer-paths.yml src/grounded_weather_forecast/artifacts.py
uv run pyrefly check src && uv run ty check src
uv run deptry src
uv run pyroma --strict --min 10 .
uv run lizard -Eduplicate -C 27 -x "*/dashboard/assets/*" src
uv run pytest tests/ --cov=src --cov-report=term-missing
```

The packaging check uses [pyroma621](https://hbmartin.github.io/pyroma621/),
installed with the development dependencies as the `pyroma` command. CI
requires a score of 10/10 with `--strict`, including advisory checks.

See the [release guide](https://github.com/hbmartin/grounded-weather-forecast/blob/main/docs/releasing.md)
for the TestPyPI and PyPI trusted publishing setup and checklist.

## License

Apache-2.0
