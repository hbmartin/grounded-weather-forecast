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
save it as `config.toml`, and point it at your two SQLite files, coordinates,
and elevation:

```bash
curl -L https://raw.githubusercontent.com/hbmartin/grounded-weather-forecast/main/config.example.toml \
  -o config.toml
```

```bash
# 1. Inspect the station truth: per-channel bounds/spike/flatline flag counts
#    and hourly/daily coverage after QC.
grounded-weather-forecast qc

# 2. Optional: poll the Open-Meteo Ensemble API before building matrices.
#    Real ensemble spread becomes leakage-safe ens__* feature columns. Run
#    this once per model cycle; configure [ensembles].models.
grounded-weather-forecast ingest-ensembles              # --models <ids>

# 3. Materialize truth tables, canonical long frames, and the supervised
#    hourly/daily matrices as parquet + manifest.json under [dataset].dir.
#    Re-run this after every ensemble ingest before backtesting or serving.
grounded-weather-forecast build-dataset

# 4. Optional cold start. A forecast archive is only useful once it holds months
#    of stored *vintages*, so a new one can say nothing yet. Open-Meteo's
#    Previous Runs API backfills real archived forecasts (leads of exactly 1-7
#    days) for open NWP models, tagged `synthetic` and never pooled with live.
grounded-weather-forecast backfill --end 2026-07-12   # --models, --start, --chunk-days

#    A second backfill provider reads dynamical.org's free Zarr archives of
#    FULL forecast cycles (GEFS since 2020, AIFS-ENS since 2025-07) at native
#    3-6h steps — populating the sub-24h lead buckets Previous Runs cannot.
#    For an installed CLI, put the extra in that same tool environment:
uv tool install --force 'grounded-weather-forecast[backfill]'
grounded-weather-forecast backfill --provider dynamical --start 2026-06-01
#    From a checkout instead:
#    uv sync --extra backfill
#    uv run grounded-weather-forecast backfill --provider dynamical --start 2026-06-01

#    A third provider backfills archived ENSEMBLE mean/spread (Open-Meteo keeps
#    them from ~March 2026) into the ens__ feature store with true run
#    vintages, months before live polling accumulates the same history.
grounded-weather-forecast backfill --provider open_meteo_ensemble --start 2026-03-01

# 5. Study whether each hourly variable should use instantaneous or interval-mean
#    truth. Misalignment masquerades as provider bias; this measures it.
grounded-weather-forecast alignment

# 6. Rolling-origin backtest. Identified evaluation runs land in
#    [dataset].dir/scores without overwriting other windows/runs.
grounded-weather-forecast backtest --source live       # or --source synthetic
#   --methods all|<ids>  --products hourly,daily  --window expanding|rolling
#   --hourly-variables ...  --daily-variables ...  --semantics auto|inst|mean

# 6b. Optional: cross-check station truth against lapse-adjusted neighbor
#    stations and fit the radiation-shield error model. A drifting or
#    decorrelating sensor alarms here before it poisons truth. Keyless by
#    default: nearby NWS METAR stations are discovered through the
#    api.weather.gov points API; setting [truth_qc].synoptic_token routes
#    through the Synoptic timeseries API instead. Widen
#    [truth_qc].elevation_band_m when the only nearby METARs sit far above
#    or below the station — the lapse adjustment is what makes them usable.
grounded-weather-forecast truth-qc                      # --days 30

# 7. Leaderboards (per-slice skill with Diebold-Mariano, aggregate, winners,
#    absolute error, consumer %-within-3F), the provider error-correlation
#    matrix, and self-verification of forecasts this system actually served.
#    Also writes reports/dashboard.html — a fully offline, self-contained
#    operator console (nine zones: liveness, data trust, learning
#    readiness, evaluation, model internals, serving, explainability,
#    quality over time — trend charts over the artifacts/history ledgers:
#    recent-window MAE per variable, selection churn per release, served
#    MAE vs backtest promise, A/B verdict shares, and e-process wealth
#    against the promotion threshold — and operations: end-to-end
#    freshness, provider collector health, stage runtimes, the
#    collector-to-matrix build funnel, the scores-file footprint, and
#    config/code identity changes) with threshold alerts sourced from
#    the existing config knobs.
grounded-weather-forecast report

# 7b. Housekeeping: delete superseded backtest scores files. Keeps the
#    newest three per (product, source, window) group plus anything a
#    release promoted in the last 30 days still references; files the
#    evaluations catalog has never seen are skipped, never deleted, so
#    pruning cannot destroy unsummarized evidence.
grounded-weather-forecast prune-scores                 # --dry-run to preview

# 8. Emit the current blended forecast (minutely + hourly + daily) as JSON.
#    Schema version 4 carries ready/degraded status plus per-variable release
#    identity and truth semantics. It is appended atomically to a history so
#    each row is later scored against the same truth target used to select and
#    fit it — backtest skill is an estimate, this is the measurement.
grounded-weather-forecast predict                      # to stdout
grounded-weather-forecast predict --out forecast.json
#   --method auto|<id>   --now <iso>   --no-history   --semantics ...
#   Unarchived --now reconstructions degrade to equal_weight when the historical
#   release's implementation is unavailable; archived documents replay exactly.
```

Every command takes `--config <path>` (default `config.toml`). Once that
configuration loads successfully, the invocation appends one row to
`[dataset].dir/runs.parquet` — a rolling ledger (command, timing, exit
status, dataset/config fingerprints) that the dashboard renders as the pipeline
heartbeat, kept to the last 90 days and 50,000 rows so it stays bounded under a
scheduled cadence. Parser and configuration-loading failures cannot be recorded
because the ledger destination comes from that configuration. Each `predict` run
additionally snapshots the fitted models' internals (grounding coefficients,
expert weights, GBM importances, anchoring decay) into
`[artifacts].dir/observability/` for the dashboard's glass-box zone, reclaiming
snapshot trees superseded by a newer dataset fingerprint; snapshot failures
never affect serving.

Backtest evidence records the package version plus a digest of the installed
first-party Python sources. The live demotion gate pools recent served rows only
when configuration, method, implementation identity, provider source set, and
the exact serving feature schema and per-variable truth semantics all match.
`predict --semantics` selects matching evaluation evidence and records the
actual target on every hourly row; a changed ensemble, truth target, or
implementation cannot inherit an incompatible verdict. The flag binds only
variables with dual truth semantics — single-truth variables (`wind_gust_ms`,
`precip_mm`, `pop`) always score against instantaneous truth, so a `mean` run
cannot strand their evidence. Score files written before feature-schema
identity are ignored by selection; re-run `backtest` after upgrading to
restore promotions.

Served slices whose winning method emits no native quantiles are dressed with
empirical residual quantiles from that method's own live backtest errors
(asymmetric, per lead bucket, finite-sample corrected); the document marks
them in a per-variable `quantiles_source` map so dressed bands are never
mistaken for a method's own distribution. The leaderboard report adds a
"Blocked promotions" section naming every slice whose served MAE exceeds the
slice's board minimum by more than `[promotion].report_gap_threshold`
(default 0.15), along with which gate blocked the better method.

Methods can be registered with a variable scope (`register(...,
variables=frozenset({"pop"}))`), so specialist heads are simply never fitted
off-scope: `precip_sparse_shrink` shrinks the daily precipitation blend
toward harmonic climatology by a per-row source-count weight `n / (n + 2)` —
past day six only one or two providers publish dailies, and an unfitted trust
schedule is well-defined exactly where per-bucket evidence is too thin to fit
one (the A/B against `damped_grounded_equal_weight`'s fitted alpha),
`pop_platt` vs `pop_beta` recalibrate the provider PoP mean (the
wet-season A/B, arbitrated by the Brier column, identity-guarded on dry
archives), `csgd_emos` fits a censored shifted-gamma to precipitation (the
first quantile emitter whose censored mass IS the dry probability), and the
daily temperature product gets its own A/B — `daily_marginal_emos` (direct
Meng–Taylor marginal on daily values + hourly-path extremes) vs
`daily_path_extreme` (grounded ensemble of per-source path extremes). The
daily matrix carries `path__{source}__max/min` features (coverage-gated) to
power them. Far daily buckets whose per-bucket evidence sits under the
eligibility floor are gated on pooled D3-10 evidence instead — a promotion
through that path is labeled `pooled_D3-10` in the winners table while
scoring and selection stay per fine bucket.

Three structure challengers round out the method pool, pure leaderboard
candidates: `raft_grounded` (a freely fitted per-bucket response to the
issue-time residual, replacing the assumed exponential anchor decay — RAFT,
Schuhen et al. 2020), `seamless_regression` (one per-bucket ridge over
forward-filled source columns plus the issue-time observation — Dabernig &
Atencia's single-model architecture, the honest test of the
grounding→blending→anchoring decomposition), and `inverse_covariance`
(Ledoit–Wolf-shrunk GLS weights with capped deviations — the honest test of
diagonal-only weighting).

Promotion offers two mutually exclusive statistical gates, compared side by
side in every live report's "Promotion rule comparison" section:
`[promotion].rule = "mcs"` (bootstrap Model Confidence Set, re-run from
scratch nightly; replicates and block length tunable via
`[promotion].mcs_bootstrap` / `mcs_block_length`) and `"seq_mcs"` — an
anytime-valid betting e-process per (slice, candidate, reference) whose
wealth accumulates across nightly re-runs (state under
`artifacts/eprocess/`, reset whenever config or code identity changes) and
promotes when every reference's e-process exceeds `1/alpha`. The leaderboard
also carries `dm_q_vs_*` Benjamini–Hochberg FDR-adjusted q-values beside
every DM p-value, and — on live boards — the mutually exclusive e-value
correction beside it: `e_vs_*` (current e-process wealth read at report
time), `ebh_sig_vs_*` (e-BH discoveries: FDR <= alpha under arbitrary
dependence, valid under the nightly re-reading that DM p-values are not),
and `ebh_threshold_vs_*` (the wealth a pair must reach to be a discovery —
why a Ville-passing pair may still not clear the FDR bar). The verdicts
ledger accumulates both corrections' discovery counts. Where the default references are bias-dominated against
station truth, `[promotion.references]` overrides the gate's reference class
per variable (e.g. `pressure_sea_hpa = ["grounded_equal_weight",
"damped_grounded_equal_weight"]`); skill columns keep their default meaning
and extend with the configured references.

Every `report` also appends to ten append-only history ledgers under
`artifacts/history/`. Five track quality: `quality.parquet` (per-evaluation
leaderboard metrics plus a 14-day recent-window MAE, so genuine movement is
not diluted by months of expanding-window history), `churn.parquet`
(per-slice diffs between consecutive promoted releases, also rendered as
`reports/selection_churn.md`), `verdicts.parquet` (A/B summaries: quantile
recalibration win shares and promotion-gate agreement), `eprocess_wealth.parquet`
(per-pair wealth snapshots, era-keyed across resets), and
`served_quality.parquet` (daily realized served MAE vs its backtest
promise, recorded per product against that product's own board). The
quality ledger also records 14-day recent-window interval coverage and
CRPS for quantile-emitting methods — the pooled expanding-window coverage
is frozen by history, so only a recent window can show a calibration
repair working (dashboard panel h6 plots it against the 0.80 target).

Five more track operations — the pipeline's edges, where silent failures
live: `pipeline.parquet` (one end-to-end freshness row per day: age of the
newest station observation, collector run, served-history row, and
published forecast document, plus 24h run counts; hard-threshold alarm
strings are printed by the report as `PIPELINE ALARMS`), `provider_health.parquet`
(per-provider success rate, latency, point volumes, and maximum stored
lead; the report prints a contraction note when a provider's lead falls
below its own 14-day median — the plan-downgrade and quota-change
detector), `build_funnel.parquet` (rows and max lead per source at each
storage layer, collector → long → matrix, including the daily
native-vs-path split, so data lost between layers is a visible trend),
`changes.parquet` (every config-fingerprint and code-version transition
with the changed config keys — config.toml is gitignored, so this ledger
is its only history; secrets are redacted), and `evaluations.parquet` (a
catalog row per scores file — size, folds, per-fold scored counts, issue
span — that outlives the file, which is what makes `prune-scores` safe).
All five are summarized in `reports/pipeline_health.md` and trended in the
dashboard's operations zone.
Appends are idempotent (re-running `report` is a no-op), bounded
(~2 years by age plus row caps), carry full fingerprint provenance so
trends segment across resets, and can never fail the report. The report
prints a one-line quality delta comparing the two newest live evaluations.

Natively-emitted quantiles can additionally be recalibrated at serve time:
live leaderboard reports carry a "Quantile recalibration (offline holdout)"
section that fits two mutually exclusive post-hoc repairs — PIT level
remapping and per-level CQR margins — on the stored scores and compares
their holdout coverage against the raw bands, and
`[predict].quantile_recalibration` applies the winning transform to native
quantile rows in the served document — routed per product, because the A/B
found the winner differs by aggregation level (daily favors `cqr`, hourly
favors raw quantiles — the documented norm in the postprocessing
literature, not an anomaly). A bare string (`"none" | "pit" | "cqr"`,
default `"none"`) routes every product identically; a table sets each
product:

```toml
[predict.quantile_recalibration]
hourly = "none"
daily = "cqr"
minutely = "none"
```

Flip policy: switch a product only after its win share holds >= 0.60 for
>= 7 consecutive evaluations in the verdicts ledger, and do not flip back
within 14 days — data-driven routing is itself a selection made on the
holdout, and hysteresis is the guard. Recalibrated rows are labeled
`recalibrated_{mode}_*` in `quantiles_source`; dressed rows are never
transformed twice.

The nowcast is measured: `backtest --products ... minutely` (on by
default, live-only) scores the minutely PATH CONSTRUCTIONS — the un-anchored
interpolation and flat observation persistence as the reference class, an
anchor-decay tau grid (0.25/0.5/1/3 h, each its own method id so the
leaderboard is the grid search), the full shift, and two fitted per-bucket
responses (`minutely_ramp`, `minutely_fitted_slope`) — against per-minute
station truth on sub-hourly lead buckets (0-5m through 45-60m), with the
proxy hourly path fold-fitted under the stricter hourly truth clock.
Promotion runs through the standard gates against `MINUTELY_REFERENCES`, and
serving consults the promoted construction per (variable, minutely bucket);
`[predict].minutely_tau_hours` survives as the named no-evidence fallback,
and an already-anchored hourly path is never re-anchored. Served minutely
history carries the applied `minutely_*` method id, closing the
self-verification loop for the nowcast.

Promoted winner MAEs carry winner's-curse corrections beside every
argmin-selected row: `winner_bias` / `mae_debiased` (moving-block-bootstrap
bias of the min functional — how much taking a minimum over ~40 methods
flatters the report) and `mae_hybrid` / `mae_hybrid_upper`
(Andrews-Kitagawa-McCloskey conditional-on-winner median-unbiased estimate
and upper bound, hybrid scheme, Sigma from the same bootstrap), with
`near_tie_flag` marking slices where the two estimands diverge beyond the
winner's bootstrap SE. Report-layer only: gates, selections, and release
ids are unchanged; the verdicts ledger accumulates `winner_bias_mean`, and
the quality delta line carries the uncorrected-argmin caveat inline.

The IDR benchmark ships as a three-way A/B: `idr` (global fit, the
control), `idr_bucket` (per-lead-bucket fits with subagging — the
smoothing arm), and `idr_bucket_dcp` (per-bucket fits under split
distributional conformal prediction — the guarantee arm, finite-sample
marginal coverage however misspecified the isotonic fit is). The board and
the coverage ledger arbitrate.

Station-truth drift detection is attribution-grade: `truth-qc` runs an
SNHT change-point test (small-sample Monte-Carlo criticals; Pettitt as the
registered alternative via `[truth_qc].drift_statistic`) on a daytime-only,
inversion-screened, elevation-similar station-minus-neighbor daily series,
attributes any break with a miniature pairwise-PHA over the neighbor
network (coincident station-minus-neighbor breaks with quiet
neighbor-vs-neighbor pairs = station drift; broken neighbor pairs =
regional regime), and latches only after three consecutive daily
exceedances. When most providers throw residual drift alarms on one
variable simultaneously, the drift report collapses them into a single
`common_mode` headline quoting the neighbor verdict. A latched
station-drift verdict can quarantine the affected temperature labels from
new fits via `[truth_qc].gate_fitting` (default `false` — flag, never
delete; the ECMWF-blacklist pattern). Dashboard zone B carries the
cross-check panel.

## Status

Alpha, and honest about it: with a young forecast archive the backtest reports
that it has no folds rather than inventing a leaderboard, and `predict` refuses
to serve from stale provider data rather than guessing.

## Documentation

- **[Getting started](https://hbmartin.github.io/grounded-weather-forecast/getting-started/)** — install, configure, first forecast
- **[Advanced usage](https://hbmartin.github.io/grounded-weather-forecast/advanced-usage/)** — backfilling, tuning, reading the
  leaderboard, adding your own blending method
- **[Theory and concepts](https://hbmartin.github.io/grounded-weather-forecast/theory/)** — why grounding beats weighting, what the
  forecast-combination puzzle costs you, and how the evaluation is kept honest
- **[Architecture](https://hbmartin.github.io/grounded-weather-forecast/architecture/)** — layers, contracts, storage, libraries,
  leakage defences
- **[Limitations](https://hbmartin.github.io/grounded-weather-forecast/limitations/)** — what this cannot do, and the three real
  bugs the evaluation harness caught. **Read before trusting any number.**
- **[Scheduling](https://hbmartin.github.io/grounded-weather-forecast/scheduling/)** — launchd templates and cadence rationale for the
  polling, ensemble-ingest, predict, and nightly-retrain crons
- [`docs/changes-0.4.0.md`](https://github.com/hbmartin/grounded-weather-forecast/blob/main/docs/changes-0.4.0.md) — 0.4.0 dashboard + instrumentation changes
- [`docs/changes-0.3.0.md`](https://github.com/hbmartin/grounded-weather-forecast/blob/main/docs/changes-0.3.0.md) — 0.3.0 migration instructions and change
  rationale (scoring semantics changed; re-run backtest before comparing)
- [`CONTEXT.md`](https://github.com/hbmartin/grounded-weather-forecast/blob/main/CONTEXT.md) — project glossary (issue time, valid time, lead,
  grounding, anchoring, …)
- [`docs/adr/`](https://github.com/hbmartin/grounded-weather-forecast/tree/main/docs/adr) — architecture decision records

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
uv run lizard -Eduplicate -C 27 -x "*/dashboard/assets/*" src
uv run pytest tests/ --cov=src --cov-report=term-missing
```

See the [release guide](https://github.com/hbmartin/grounded-weather-forecast/blob/main/docs/releasing.md)
for the TestPyPI and PyPI trusted publishing setup and checklist.

## License

Apache-2.0
