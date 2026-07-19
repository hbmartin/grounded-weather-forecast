# Operator dashboard

`grounded-weather-forecast report` writes `reports/dashboard.html` on every
run — a **fully offline, self-contained** operator console. All CSS, the
vendored Chart.js library, and the data payload are inlined, so the file
renders from `file://` with no network access. There is no server and no
daemon: the dashboard is a read-only projection of the artifacts the
pipeline already writes to disk, regenerated each `report` run, exactly like
the markdown reports beside it.

The page answers four operator questions, split across seven zones:

| Question | Zones |
|---|---|
| Is fresh, trustworthy data flowing in? | **A** liveness · **B** data trust |
| Can the system even learn yet? | **C** learning readiness |
| Are the models valid, and which is winning and why? | **D** evaluation · **E** model internals |
| Is what we actually served any good, and why that number? | **F** serving · **G** explainability |

Every panel carries its own explanatory prose (what it shows, why an
operator cares, and where its thresholds come from) in a collapsible
*about this panel* block, so the reference below stays brief.

## Zone reference

**A — Liveness & freshness.** The latest served document's status
(`ready`/`degraded` with the exact `status_reason`), station observation lag
against the 30-minute serve staleness cap, and per-provider fetch ages
against `[forecasts].max_forecast_age_hours`. A provider that ages past the
cap silently drops out of snapshots — it is drawn grey, not omitted.

**B — Ingestion & data trust.** Station QC flag counts per channel
(out-of-bounds / spike / flatline; flagged samples are nulled, never
corrected), a daily truth-coverage calendar against
`[dataset].min_hour_coverage`, per-provider forecast null shares, and the
live/synthetic provenance wall asserted as a badge.

**C — Learning readiness.** Archive issue-time span versus the
`initial_train_days + step_days` a first rolling-origin fold needs (a
progress bar mirroring the backtest's own "no folds" arithmetic), snapshots
collected per day, synthetic-backfill coverage, and the truth-semantics
alignment study with defaulted (not-yet-data-backed) variables flagged.

**D — Evaluation.** Per-slice leaderboards with `n` first and bias cells
color-classed against the per-variable consumer tolerances, slice winners
after the promotion gate (`[promotion].rule` at its alpha), the baseline
floor sanity check, the provider error-correlation matrix with the derived
effective ensemble size *k_eff*, and the calibration views the CLI computes
but never surfaces: CRPS/pinball/interval-coverage columns, a PIT histogram
rebuilt from the persisted quantile grids, and the PoP reliability diagram.

**E — Model internals.** The glass box, read from the observability
snapshots each `predict` run writes to `[artifacts].dir/observability/`:
grounding coefficients per provider × lead bucket (IDENTITY fallbacks
greyed), online-expert weights (a trajectory line once two or more
snapshots span the window — the provider-backend-swap detector), GBM feature
importances (with a loud red state when lightgbm is missing and the method
silently absent from the registry), the fitted anchoring decay timescale and
its weight curve, and best_provider's per-bucket source rankings.

**F — Serving & self-verification.** Served-vs-realized MAE per slice with
the `mae_gap` against the backtest's promise (red past
`[promotion].live_gap_factor`), served slices per day stacked by selection
reason with the degraded share, and the release lineage table — dataset
fingerprint → evaluations → release → served documents — with stale
fingerprints flagged.

**G — Explainability.** Pick any point/variable of the latest served
document and see the method that produced it, its selection reason, release
ids, quantiles, and anchor observation. Hourly and daily provider inputs are
drawn from the newest provider snapshot visible at the served issue and matched
to the exact served point. Minutely rows are selectable, but the dataset
contract does not expose their raw provider-input matrix.
The same document is replayable byte-for-byte with `predict --now <issue>`.

## Alerts and their thresholds

The alert strip at the top is computed at generation time by
`reports/alerts.py`. Every threshold names an existing config knob or module
constant — the alerting invents no policy:

| Alert | Threshold source |
|---|---|
| ingestion stalled / anchor lost | `serve/predict.py::OBS_STALENESS` (30 min); `[forecasts].max_forecast_age_hours` |
| provider dropped / aged out | `[forecasts].max_forecast_age_hours`; `manifest.sources` |
| serving refused | `NoForecastDataError` via the runs ledger |
| serving degraded | `Forecast.status_reason` (`no_evidence_reason`) |
| truth thinning | `[dataset].min_hour_coverage` / `min_day_coverage` |
| stuck sensor | `QC_FLATLINE` bit; `[qc].flatline_minutes` |
| provider drifting | `artifacts/drift.json` (consensus/residual tiers) |
| grounding bias | `reports/leaderboard.py::CONSUMER_TOLERANCES` |
| baseline implausible | structural heuristic (labeled as such — no knob) |
| backend swap | leading-expert flip within the 3-day drift window |
| serving diverged | `[promotion].live_gap_factor`, `min_live_n` |
| artifacts stale | manifest vs release fingerprints |
| silent-empty states | manifest sources/snapshots/rows; `LOCATION_TOLERANCE` |

Families that cannot be evaluated yet return a single *not evaluable yet*
info chip instead of silence or a false alarm.

## What a young deployment looks like

On day one most of the dashboard is **supposed** to be grey or amber:
zero live folds is correct behaviour (zone C says how far away the first
fold is), no releases means zone F reports "no promotion has ever occurred",
and zone E stays in "not yet" states until a `predict` run persists its
first observability snapshots. The real red flags on a young deployment are
the silent-empty states — zero sources, zero snapshots, zero-row files, or
an archive/station location mismatch — because those look like health when
nothing is actually flowing.

## New on-disk signals

- `[dataset].dir/runs.parquet` — append-only ledger of every command whose
  configuration loads successfully: command, args, start/end, duration, exit
  code or exception name, dataset/config fingerprints, code version. Parser
  and configuration-loading failures cannot be recorded because that
  configuration supplies the ledger destination. Telemetry writes never fail
  a command (5-second lock timeout, errors swallowed).
- `[artifacts].dir/observability/` — per-(method, product, variable)
  latest-state snapshots (`ArtifactStore` layout) plus
  `history.parquet`, an ewa/boa-only weight trajectory pruned to
  `[backtest].rolling_window_days`. Write-only: serving output is identical
  whether snapshots land or not.

## Updating the vendored Chart.js

The chart library is pinned and committed at
`src/grounded_weather_forecast/dashboard/assets/chart.umd.min.js`
(Chart.js 4.4.9, MIT). To upgrade: download the new
`dist/chart.umd.js` from the official release, replace the file (keep the
license header), update the version asserted in
`tests/dashboard/test_assets_packaging.py`, and re-run the suite. Lizard is
configured to skip `*/dashboard/assets/*` in CI — keep that exclusion in
step with the path.
