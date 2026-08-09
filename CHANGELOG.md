# Changelog

Notable changes to grounded-weather-forecast. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **GBM containment, part two**: `gbm` gains a `blend_mean` feature (the
  equal-weight consensus) with a `+1` monotone constraint — the booster can
  correct the blend but not invert it — which required moving its objective
  from `regression_l1` to `huber` (LightGBM forbids monotone constraints
  under leaf-renewing objectives). New `gbm_quantile` variant trains one
  pinball booster per level of the 19-level grid and serves native
  quantiles, so the leaderboard scores its CRPS/pinball/coverage directly.
- **`pava_isotonic` now delegates to scipy's compiled `isotonic_regression`**
  (values are identical; it was 59% of backtest wall-clock at 26,571 calls
  per `idr_bucket` fit) and the idr predict path batches rows per bucket and
  grid position (~10× faster `idr_bucket`).
- **Incumbent retention (winner-curse guard)**: selection keeps the previous
  release's method when a new argmin winner is within one bootstrap SE of it
  on the current board — near-ties stop churning the served method and stop
  re-realizing the argmin's optimism. Retained selections carry the
  current-board evidence, a `retained` flag in the release payload, and the
  live demotion gate keeps the last word; config pins override entirely.
- **NBM benchmark row**: new `provider_nbm` single-source passthrough method
  (abstains where the provider has no data), so the operational baseline
  gets an explicit leaderboard row; `report` prints an n-weighted
  blend-vs-NBM benchmark line and records the scalars in the verdicts
  ledger.
- **Pipeline mutex**: `build-dataset`, `backtest`, `report`, `alignment`,
  `backfill`, `truth-qc`, and `prune-scores` serialize on
  `<dataset dir>/pipeline.lock`; contention exits `75` (EX_TEMPFAIL) after
  60 s. `predict` stays lock-free and instead retries its scores-directory
  scan once when a file vanishes mid-read, so a concurrent prune can neither
  crash serving nor feed it a half-pruned evidence set. Scores files are now
  written atomically.

- Forecast document **schema 5**: per-variable `truth_semantics`,
  `selection_reasons`, and a `quantiles_source` map; point-only winners are
  dressed with live residual quantiles at serve time, and blocked promotions
  are exposed in the release ledger.
- Sequential e-process promotion gate (`[promotion] rule = "seq_mcs"`) with
  configurable MCS bootstrap replicates/block length, BH-FDR q-values and
  e-BH columns beside every DM p-value, and per-variable reference overrides.
- Quantile-recalibration A/B layer with per-product routing.
- Minutely backtest product: anchored path constructions scored on
  sub-hourly buckets, with serving consulting the promoted construction per
  bucket (config tau as the no-evidence fallback).
- Winner's-curse-corrected reporting: bootstrap bias and AKM hybrid columns
  on every argmin-selected winner.
- Evidence-history ledgers under `artifacts/history/`, dashboard zones
  **H** (quality over time) and **I** (operations), and
  `reports/pipeline_health.md`.
- `prune-scores` command (preview with `--dry-run`): deletes superseded
  backtest scores files while protecting the newest three per slice,
  anything promoted in the last 7 days, and files the evaluations catalog
  has never seen.
- Keyless NWS METAR neighbor stations for `truth-qc`, SNHT/PHA sensor-drift
  attribution with common-mode collapse, and the `[truth_qc].gate_fitting`
  truth quarantine (default off).
- New registered blenders: `damped_grounded_equal_weight`,
  `analog_ensemble`, `cluster_equal_weight`, `raft_grounded`,
  `seamless_regression`, `inverse_covariance`, `csgd_emos`, `idr_bucket`,
  `idr_bucket_dcp`, `pop_platt`, `pop_beta`, `precip_sparse_shrink`,
  `daily_marginal_emos`, `daily_path_extreme`.
- `[forecasts].exclude` per-variable source exclusion, per-source lead
  caps, and ensemble-mean backfill.
- Precipitation cross-source QC floors and a baseline-relative truth alarm.

### Changed

- Online-experts loss switched from squared to absolute error; per-bucket
  fitting gained empirical-Bayes shrinkage; GBM gained a min-training-rows
  promotability floor.
- Re-run `backtest` after upgrading: score files written before
  feature-schema identity are ignored by selection.

### Fixed

- Conformal calibration repair (proper chronological split state).
- Backfilled ensemble vintages can no longer shadow the live-polled era.
- `release_id` schema inference crash in served-forecast history append.

## [0.4.0] - 2026-07-23

The operator dashboard: `report` now writes `reports/dashboard.html`, a
fully offline self-contained console (seven zones at release) with a
threshold-alert strip whose every limit is an existing config knob or
module constant.

### Added

- `[dataset].dir/runs.parquet`: rolling command ledger (pruned to 90 days /
  50,000 rows); telemetry never breaks a command.
- `[artifacts].dir/observability/`: write-only snapshots of fitted blender
  internals on every `predict`; serving output is bit-identical with or
  without them.
- `reports/alerts.py`, public `OBS_STALENESS` / `LOCATION_TOLERANCE`
  constants, optional `timeout` on `storage.locked_path`,
  `ArtifactStore.load_manifest`.
- `truth_daily.parquet` now emits `rain_coverage`.

### Changed

- Forecast document schema 3: hourly and daily points carry a
  `release_ids` map so each variable is attributable to the exact promotion
  that selected its method. Schema-1/2 documents remain readable.

### Fixed

- Ensemble features are resolved identically in training and serving
  (`predict` now applies the same `[ensembles]` model/variable filter as
  `build-dataset`).
- The minutely nowcast no longer steps between anchoring regimes; the
  regime of the row owning lead zero governs the whole range.

### Upgrade

No manual migration. `grounded-weather-forecast report` writes
`reports/dashboard.html`; on a young deployment most panels are
deliberately grey or amber — zero live folds and no promoted releases are
correct behaviour, and the page says so.

## [0.3.0] - 2026-07-18

All twelve milestones of the improvement program (full rationale:
`research/improvement-methods-2026-07.md`): metric-consistent scoring and
honest DM tests, the upstream NOAA NBM source plugin, Open-Meteo ensemble
spread as feature columns, solar-geometry and cyclical context features,
adaptive EWMA + harmonic grounding, the dynamical.org sub-24 h backfill,
the fitted-anchor rework, calibrated distributions (EMOS/IDR with
CRPS/PIT scoring), online conformal intervals, the Model-Confidence-Set
promotion gate with live-verification feedback, true online expert state
with two-tier drift detection, and the neighbor/physics truth-QC layer.
New method ids: `grounded_median_equal_weight`, `inverse_mae`,
`trimmed_mean`, `grounded_trimmed_mean`, `ewma_grounded_equal_weight`,
`ewma_inverse_mae`, `harmonic_grounded_equal_weight`.

### Migration

1. Upgrade and verify: `grounded-weather-forecast --version` → 0.3.0.
2. **Numbers from 0.2.0 are not comparable.** Bounds are now clamped
   inside scoring, each method is scored on its own cases (`n` differs per
   method), and DM p-values are honestly larger after per-valid-time
   collapse. Regenerate rather than mixing eras.
3. **Rebuild the dataset — and expect one serving-degradation window.**
   The hourly matrix gained feature columns, so the dataset fingerprint
   changes and `predict` falls back to degraded equal weight until fresh
   evidence exists against the new matrix:

   ```bash
   grounded-weather-forecast ingest-ensembles
   grounded-weather-forecast build-dataset
   grounded-weather-forecast backtest --source live
   grounded-weather-forecast report
   ```

4. Optional new config section `[ensembles]` + an `ingest-ensembles` cron
   (Open-Meteo retains only the latest run's members).
5. Upstream `omni-weather-forecast-apis` gained a keyless `nbm` plugin —
   both an input and the operational benchmark to beat.
6. Python API rename: `InverseMseWeights` → `InverseErrorWeights`
   (registry ids unchanged).

## [0.2.0] - 2026-07-18

Leaderboard fix and version bump (commit 779bc1e).

## [0.1.0] - 2026-07-13

Initial release: grounding/blending/anchoring pipeline over the two SQLite
inputs, rolling-origin backtest, leaderboard reports, JSON serving.
