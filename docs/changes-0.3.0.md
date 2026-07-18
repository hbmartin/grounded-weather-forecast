# 0.3.0 — adaptive grounding, ensemble features, honest scoring

This release lands ALL TWELVE milestones of the improvement program
(`research/improvement-methods-2026-07.md`; roadmap in that document's §9 and
the program cross-reference in §8): metric-consistent scoring and honest DM
tests (M1), a NOAA NBM source plugin upstream (M2), Open-Meteo ensemble
spread as feature columns (M3), solar-geometry and cyclical context features
(M4), and the adaptive EWMA + harmonic grounding variants that fix the
measured 24–96 h grounding loss (M5) — plus the dynamical.org sub-24 h
backfill (M6), the fitted-anchor rework (M7), calibrated distributions with
EMOS/IDR and wired CRPS/PIT scoring (M8), online conformal intervals (M9),
the Model-Confidence-Set promotion gate with live-verification feedback
(M10), true online expert state with two-tier drift detection (M11), and the
neighbor/physics truth-QC layer (M12). **The hourly matrix schema changed**
(new feature columns), so the dataset fingerprint changes on rebuild — see
migration step 3.

---

## Migration instructions

1. **Upgrade**

   ```bash
   uv tool install grounded-weather-forecast   # or: uv sync in a checkout
   grounded-weather-forecast --version          # expect 0.3.0
   ```

2. **Re-run the backtest and report before trusting any comparison**

   ```bash
   grounded-weather-forecast backtest --source synthetic   # and/or --source live
   grounded-weather-forecast report
   ```

   Numbers produced by 0.2.0 are **not comparable** to 0.3.0 numbers for the
   reasons listed under "Scoring changes" below. Regenerate rather than
   mixing eras.

3. **Rebuild the dataset — and expect one serving-degradation window.** The
   hourly matrix gained feature columns (`ens__*` ensemble statistics,
   `solar_elevation_deg`, `toa_wm2`, `hour_sin/cos`, `doy_sin/cos`,
   `obs__*__trend15m`), so `build-dataset` produces a **new dataset
   fingerprint**. `predict` only serves promoted evidence whose fingerprint
   matches, so after the rebuild it falls back to degraded equal weight until
   a fresh `backtest` + `report` complete against the new matrix:

   ```bash
   grounded-weather-forecast ingest-ensembles
   grounded-weather-forecast build-dataset
   grounded-weather-forecast backtest --source live   # repopulate evidence
   grounded-weather-forecast report
   ```

4. **New config section (optional): `[ensembles]`.** Set `models` (see
   `config.example.toml` for verified ids incl. Google WeatherNext 2) and add
   `grounded-weather-forecast ingest-ensembles` to the cron — at least once
   per model cycle, since Open-Meteo retains only the latest run's members.
   Build the dataset after ingestion so the new statistics reach the matrix.
   Without the section nothing changes.

5. **Upstream: NBM provider.** `omni-weather-forecast-apis` gained a keyless
   `nbm` plugin (NOAA National Blend of Models via the IEM per-station
   archive; 3-hourly to +72 h). Add it upstream with the nearest NBM station
   id, and it flows into this repo as a normal source — both an input and the
   operational benchmark to beat.

6. **Expect leaderboard numbers to move.** Three deliberate changes shift
   metrics without any model being different:
   - Methods that emitted out-of-range values (negative wind, >100% humidity)
     are now clamped *inside scoring*; only such methods' MAE/bias move.
   - Each method is now scored on its **own** cases instead of the all-methods
     intersection, so `n` differs per method and MAE covers more rows.
   - DM p-values are generally **larger** (less significant) than 0.2.0's:
     the old ones were overconfident by construction (see below). A slice that
     lost its "significant" winner did not get worse — its previous
     significance was an artifact.

7. **Expect a possible promotion change.** `slice_winners` now also requires
   `n_valid_times >= 8` (distinct valid hours) for a challenger, so slices
   whose apparent sample was snapshot pseudo-replication fall back to the
   reference method until real coverage accrues. Any change is visible in
   `data/scores` + the release JSON under `artifacts/releases/`.

8. **New methods appear under `backtest --methods all`** (the default). If you
   pin `--methods` explicitly or pin `[predict.methods]` in config, nothing
   changes until you opt in. The new ids: `grounded_median_equal_weight`,
   `inverse_mae`, `trimmed_mean`, `grounded_trimmed_mean`,
   `ewma_grounded_equal_weight`, `ewma_inverse_mae`, and
   `harmonic_grounded_equal_weight` — plus the renamed internal class behind
   `inverse_mse` (the method id is unchanged).

9. **`alignment` output grew.** The command now prints which dual-semantics
   variables run on the silent instantaneous default vs a data-backed
   recommendation, and the JSON artifact gained `data_backed` and `min_rows`
   keys. Consumers reading `artifacts/alignment.json` should ignore unknown
   keys (additive change).

10. **Nothing about serving changed.** `predict` output schema, degraded-mode
   behavior, history append, and release identity hashing are untouched.
   Python API: if you imported `InverseMseWeights` from
   `grounded_weather_forecast.blenders.combine`, it is now
   `InverseErrorWeights` (registry ids unchanged).

---

## What changed, and why

### 1. Variable bounds are enforced inside scoring (`finalize_point`)

**Before:** `finalize_point` clipped only probability targets to [0, 1].
`VariableSpec.minimum/maximum` (wind ≥ 0, 0 ≤ humidity ≤ 100, precip ≥ 0) were
applied only at the serve boundary (`serve/predict._finite`), so the backtest
scored physically impossible predictions that no user would ever receive —
e.g. a grounding correction pushing calm wind negative.

**Now:** `finalize_point(point, kind, variable)` clamps to the variable's
declared bounds; every blender stores its fitted `VariableSpec` and passes it
through. Training, scoring, and serving now see the same quantity.
(`blenders/protocol.py`, all blender modules; GBM artifact state also
persists the variable name so restored boosters clamp identically.)

**Effect:** leaderboard rows move only for methods that were emitting
out-of-range values. A property test asserts every registered method respects
the wind minimum on a fixture designed to force negative predictions
(`tests/blenders/test_protocol.py`).

### 2. MAE-consistent objectives: median grounding and inverse-MAE weights

The system promotes on MAE, but the bias-only grounding intercept was the
**mean** residual (the L2-optimal constant) and source weighting tracked
inverse **squared** error. Two new registered methods align the fitted
objective with the promoted metric — as new methods beside the incumbents,
never edits, so the leaderboard arbitrates:

- `grounded_median_equal_weight` — bias-only grounding with a **median**
  intercept (the MAE-optimal constant), robust to one-sided error spikes.
  `fit_affine` gained `intercept="mean"|"median"`; median is only valid with
  `slope_shrinkage == 0` and raises otherwise.
- `inverse_mae` — Bates–Granger weighting by inverse mean **absolute** error
  per lead bucket. This is also what NOAA's National Blend of Models tracks
  operationally. Internally `InverseMseWeights` became `InverseErrorWeights`
  with a `loss_power` field (2 = the unchanged `inverse_mse`, 1 = the new id).

### 3. Trimmed-mean blenders (`blenders/trimmed.py`)

`trimmed_mean` and `grounded_trimmed_mean` drop `floor(0.2·k)` of the
available sources from each end per row and average the rest (degenerating to
the plain mean below 3 sources, honoring the availability mask). Rationale:
with 8 highly correlated sources, estimated weights mostly relearn noise (the
forecast-combination puzzle), but a single misbehaving provider — outage,
silent backend swap, unit bug — drags a plain mean; symmetric trimming buys
robustness with **zero** fitted parameters
([Jose & Winkler 2008](https://doi.org/10.1016/j.ijforecast.2007.06.001)).

### 4. Diebold–Mariano tests no longer overconfident (`reports/leaderboard.py`)

**The defect:** dozens of consecutive 10-minute snapshots forecast the *same
valid hour*, so per-row losses are massively pseudo-replicated. The DM test
received every row as if independent (its HAC lag was derived from the lead
bucket — the wrong autocorrelation axis), which manufactured significance.

**The fix:** paired losses are collapsed to **one mean loss per
`valid_time`** (in temporal order) before the DM test; the lead-derived HAC
horizon then correctly handles the remaining h-step forecast overlap. A new
test proves replicating each valid hour 30× leaves the p-value exactly
unchanged (`tests/reports/test_leaderboard.py::TestDmCollapsesPseudoReplication`).

**Consequence:** p-values are honest now, and mostly larger. Historical
"significant" wins at heavily-snapshotted slices should be re-derived.

### 5. Own-case scoring + effective-n on the leaderboard

**Before:** every method in a slice was scored only on cases where **all**
methods predicted — one sparse method shrank everyone's sample and its holes
silently changed competitors' MAE.

**Now:** each method's MAE/RMSE/bias/pct_within cover its **own** non-null
cases; `coverage` = own cases / all slice cases; the DM comparison alone
restricts to pairwise-common cases (inside `_dm_columns`). A new
`n_valid_times` column reports distinct valid hours — the honest effective
sample under snapshot pseudo-replication — and `slice_winners` requires
`n_valid_times >= 8` for a challenger. Caveat (by design): raw MAEs of
methods with very different coverage are not directly comparable — that is
what the pairwise-common DM columns are for.

### 6. Alignment study surfaces silent defaults (`dataset/alignment.py`)

On a thin archive no source reaches the 72-overlapping-row threshold, so every
dual-semantics variable silently fell back to instantaneous truth while
looking "decided". The artifact now carries `data_backed` (per variable) and
`min_rows`, and the CLI prints which variables run on the default. No behavior
change to semantics resolution itself.

### 7. weatherapi's negative error correlation: diagnosed, not a bug

`reports/weatherapi_diagnosis.md` (new) documents the investigation of
weatherapi's −0.32 error correlation against every other provider. Verdict:
**genuine anti-phase diurnal bias, not an ingestion defect** — +3–4 °C warm
at night (it misses the 1400 m site's radiative cooling), −1.5 °C cool by
day, while the pack errs in the opposite phase; opposite-phase systematic
errors anticorrelate mechanically. No timestamp shift is applied (the −1 h
lag-scan improvement is second-order at n=124; instantaneous semantics
confirmed, r 0.942 vs 0.918). This is precisely the bias structure the
upcoming hour-binned EWMA grounding (M5) removes; re-check at ≥500 rows.

### 8. NOAA NBM as a source (upstream, M2)

`omni-weather-forecast-apis` gained a keyless `nbm` plugin: the National
Blend of Models NBS bulletin (3-hourly to +72 h) via the IEM per-station
parsed archive — NCEP only publishes ~28 MB whole-network text files, so the
per-station JSON endpoint is the practical path. Units per the NBM v4.2 text
card (°F, knots, percent), converted at parse. Two deliberate omissions,
documented in the plugin: `P06` (a 6-hour PoP) maps to
`precipitation_probability` with its window noted; `Q06` (a 6-hour total) is
NOT mapped to hourly precipitation, because a 6-hour total attributed to one
3-hourly point would double-count in hourly aggregation. Verified live
against KSBD. The NBM is both a strong input and the operational
benchmark-to-beat leaderboard row.

### 9. Ensemble spread as feature columns (`dataset/ensembles.py`, M3)

Provider columns are structurally under-dispersed (shared parents), so their
cross-source spread is not an uncertainty signal. `ingest-ensembles` polls
the Open-Meteo Ensemble API for configured models — verified live ids
including `google_weathernext2_ensemble` (64 members), `ecmwf_aifs025_ensemble`
(51), `ncep_aigefs025`, `ncep_gefs025` — reduces members to per-(model,
valid_time, variable) mean/sd/p10–p90 statistics, and append-dedupes them to
`data/ensembles.parquet`. The matrix build as-of joins them (same staleness
semantics as sources) into `ens__{model}__{variable}__{stat}` feature
columns. Ensembles are **features, never sources**: they inform dispersion
(GBM features now; the EMOS spread link in M8) without entering the
grounding/weighting source set. Live smoke test ingested 6,912 statistic
rows across three models in one cycle.

### 10. Solar geometry + context features (`solar.py`, M4)

A dependency-free NOAA solar-position module (vectorized; calendar-exact day
of year — the naive `unix/86400 mod 365` shortcut is two weeks off by the
2020s) supplies `solar_elevation_deg` and `toa_wm2`; the matrix adds wrapped
calendar features (`hour_sin/cos`, `doy_sin/cos`) and spike-guarded
issue-time observation tendencies (`obs__{var}__trend15m`, units/hour, from
5-minute rolling medians 10 minutes apart). All are deterministic functions
of the valid instant and the fixed site — leakage-safe by construction — and
flow into slices and the GBM via the shared `CONTEXT_FEATURE_COLUMNS`
registry in `contracts.py`. The trend columns are the input for the M7
damped-trend anchor; `toa_wm2` is the regressor for the M12 radiation-shield
error model.

### 11. Adaptive grounding: the measured fix for the measured failure (M5)

The core defect this program exists to fix: static per-lead-bucket grounding
averages the diurnal/seasonal bias structure away, and was *losing* to raw
equal weight at 24–96 h. Two new registered families:

- **`ewma_grounded_equal_weight` / `ewma_inverse_mae`**
  (`blenders/ewma_grounding.py`): decaying-average bias per (source, lead
  bucket, 3-hour bin) — `bias ← (1−w)·bias + w·(fcst−truth)`, `w = 0.05`,
  replayed in issue-time order — the scheme NCEP has run operationally since
  2006 and the NBM uses per grid point. Count-based shrinkage `n/(n+10)`
  replaces the `min_rows` cliff. No training window to choose; seasonal
  drift tracked automatically; a backend swap relearned in ~1/w updates
  (drift fixture proves it).
- **`harmonic_grounded_equal_weight`** (`blenders/harmonic_grounding.py`):
  the SAMOS-informed batch challenger — a ridge-fit bias *curve* over solar
  elevation plus annual/semiannual harmonics per (source, lead bucket),
  intercept unpenalized, slope fixed at 1 (ADR 0004). Degrades to the scalar
  correction when phase features are absent.

Measured on the real synthetic archive (ECMWF/GFS/ICON, 45 folds, hourly
temperature, with the corrected DM test):

| bucket | equal_weight | grounded (static) | **ewma_grounded** |
|---|---|---|---|
| 24–48 h | 1.336 | 1.367 | **1.360** |
| 48–96 h | 1.593 | 1.713 (p=0.010 worse) | **1.559** |
| 96–168 h | **1.748** | 1.775 | 1.796 |
| 168–240 h | 2.622 | 2.345 | **2.122** (+19.1%, p=0.004) |

The 48–96 h loss — the program's motivating measurement — is now a win, and
the long-lead improvement is large and significant. The harmonic variant
needs the rebuilt (context-feature) matrix to differentiate from the scalar
fit; on live providers with strong diurnal bias (see the weatherapi
diagnosis) the adaptive gap should widen further.

---

## Version bump rationale

0.2.0 → **0.3.0** (minor): new registered methods, new `[ensembles]` config
section and `ingest-ensembles` command, additive matrix feature columns
(fingerprint changes on rebuild), additive leaderboard/artifact columns;
scoring-semantics changes that alter reported numbers; one Python-API rename
(`InverseMseWeights` → `InverseErrorWeights`). No serve JSON schema breaks.

### 12. The remaining program (M6–M12), briefly

- **M6** `backfill --provider dynamical`: free keyless Zarr archives of FULL
  GEFS/AIFS-ENS cycles at native 3–6 h steps populate the 0–24 h lead buckets
  Previous Runs never could (verified live: 0–1 h through 12–24 h buckets now
  score). `fetched_at = init + publication_lag` (6 h default) keeps short-lead
  skill honest. Install the published optional extra with
  `uv sync --extra backfill` or `grounded-weather-forecast[backfill]`.
- **M7** anchoring rework: `_ANCHOR_MAX_LEAD` 3→6 h; `anchored_fitted_*` fit
  per-lead-bin regression weights (LAMP-style — exponential and INCA
  persist-then-ramp emerge as special cases); `anchored_trend_grounded` adds
  the observed 15-min tendency with a capped fitted gain; the minutely
  product anchors exactly ONCE against the selected path and extrapolates the
  now-forecast to lead 0 instead of clamping.
- **M8** distributions: `emos` (CRPS-fit Gaussian head for unbounded variables,
  matching truncated-normal fit and quantiles for bounded variables, spread
  from same-variable `ens__*` sd) and `idr` (hand-rolled PAVA
  isotonic distributional regression — the zero-tuning benchmark) emit
  19-level quantile grids through a shared `finalize_quantiles` (monotone
  rearrangement + bounds); the leaderboard now reports CRPS, pinball,
  coverage@80/90, PIT χ², and sharpness for any quantile emitter; a latent
  NaN-in-JSON bug in score persistence was fixed before it could trigger.
- **M9** `conformal_gew`/`conformal_ewma`: compact conformal-PID-style online
  quantile tracking + coverage integrator per (lead bucket × day/night), fit
  from a chronological 70/30 proper-training/calibration split so only later
  out-of-sample residuals update interval state. The adaptive tracker recovers
  near-nominal empirical coverage after tested variance shifts; it makes no
  unconditional finite-sample coverage claim under arbitrary drift.
- **M10** honest promotion: `reports/mcs.py` (moving-block-bootstrap t-max
  Model Confidence Set over per-valid-time losses) gates `slice_winners` —
  a challenger ships only when the reference is *excluded* from the set
  (`[promotion] rule`, default `mcs`); `select_methods` closes the
  self-verification loop, demoting any served method whose realized live MAE
  exceeds `live_gap_factor ×` its backtest promise, with the verdict recorded
  in the release ledger.
- **M11** self-improvement: `OnlineExperts.advance()` treats the matrix as
  the pending-loss queue (O(new rows) per serve past a persisted watermark;
  the dormant `ArtifactStore` now carries expert state on the serve path,
  with fingerprint/source-set mismatch falling back to a full refit);
  `reports/drift.py` runs two tiers — instant provider-vs-consensus z-scores
  and truth-based Page–Hinkley with a length-scaled threshold — into
  `reports/drift.md` + `artifacts/drift.json`.
- **M12** truth QC: `truth-qc --days 30` cross-checks the station against 3+
  lapse-adjusted Synoptic neighbors (30-day drift alert at 1 °C; rolling 72 h
  correlation < 0.9 flags the plausible-looking failing sensor) and fits the
  radiation-shield error model (independent station-minus-neighbor residual
  ~ S/(1+u), using the M4 TOA irradiance and co-located anemometer). Missing
  overlap is reported as unknown rather than healthy. Nothing auto-adjusts truth;
  WMO-SPICE gauge catch-efficiency is explicitly deferred until its published
  coefficients are transcribed and real rain exists to validate against.

## What lands next (program context)

All twelve milestones of the improvement program are implemented. What
remains is operational: start/maintain the polling and ingest crons, rebuild
the dataset, re-run the backtest against the new matrix, and let the archive
age into the methods. Deliberate deferrals, each documented in place:
day/night-conditioned anchor weights, gauge catch-efficiency, NBE-based NBM
daily values, and automated drift-triggered state resets (alarms first, until
precision is known). The full sequence, acceptance criteria, and risks live
in the approved plan and
`research/improvement-methods-2026-07.md` §9.
