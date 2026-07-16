# Improvement plan

This plan sequences the next round of modeling and evaluation work. Each item
states the motivation (with the current code it touches), a concrete design that
fits the existing contracts, how to A/B it on the backtest harness, and the
acceptance criteria. It is deliberately additive: every new method is registered
beside the current ones and only ships if the leaderboard — the sole arbiter
([Theory §6](theory.md)) — says it wins.

Two cross-cutting principles hold throughout:

1. **Never fuse stages.** Grounding, weighting, anchoring, and (new) cyclical
   correction stay separable so each one's contribution is measurable. A new
   correction is a new registered method, not an edit to an existing one.
2. **Condition on the cycle, not just the lead.** Almost every gap below has the
   same root cause: the pipeline stratifies only by lead bucket, so the diurnal
   and seasonal structure of bias, skill, and dispersion is averaged away. Any
   recency/adaptivity added must be *phase-aware*, or it pulls a forecast three
   months out toward the wrong seasonal phase.

Priorities: **P0** = do next; **P1** = high value, after P0; **P2/P3** = later.

---

## 1. Cyclicality — a harmonic / solar-phase grounding bias curve (P0)

### Why
Grounding is "the big win" ([Theory §4.1](theory.md)), yet `AffineGrounding.fit`
(`blenders/grounding.py`) fits **one intercept per (source, variable, lead
bucket)** via `PerBucketFitter` over `lead_hours` only. Because snapshots span all
clock hours, a fixed lead bucket is a near-uniform mix of all 24 valid hours, so
the diurnal component of a source's bias — large and systematic at a 1400 m
mountain site (midday radiation-shield warm bias, nocturnal cold-air-pooling cold
bias) — averages to ~zero and is left in the residual. The seasonal component is
lost the same way. `inverse_mse`, the online experts, and `best_provider` inherit
the same blindness. `HarmonicClimatology._harmonic_design`
(`blenders/baselines.py`) already has the machinery (Fourier month + hour, ridge
with an unpenalized intercept) but is used only as a baseline, and it is purely
**additive** — no hour×month interaction, so its diurnal amplitude is constant
year-round.

### Design
Add a new grounding variant and register it beside the current ones:

- `grounded_harmonic` / `harmonic_inverse_mse`: correct each source as
  `y ≈ x + b(φ)` where `b(φ)` is a **ridge-fit Fourier curve in a cyclical phase
  φ**, fitted per (source, variable, lead bucket). Keep slope `= 1`
  (bias-only, per [ADR 0004](adr/0004-grounding-defaults-to-bias-only.md)); we are
  making the *intercept* phase-dependent, not buying back the slope.
- **Phase = solar geometry, not clock hour.** Compute solar elevation / zenith and
  a clear-sky index from `config.station.{latitude,longitude,elevation_m}` +
  `valid_time` (see item 6). A diurnal curve in solar elevation is approximately
  season-invariant, which absorbs most of the diurnal×season interaction with far
  fewer parameters than an explicit hour×month tensor. Add a low-order annual term
  (sin/cos of day-of-year, plus a semiannual harmonic for the monsoon/precip
  season) for the residual seasonal drift.
- Reuse the `HarmonicClimatology` ridge so a short archive degrades gracefully to
  the scalar bias, and fall back to the current scalar intercept below a
  per-bucket row threshold (mirror `_MIN_FIT_ROWS`).

### A/B and acceptance
Register the methods; run `backtest` on the synthetic-backfill and (once live) the
live matrix. **Accept** when `grounded_harmonic` beats `grounded_equal_weight` on
MAE **and** the `bias` column shrinks in the diurnally-extreme hours (00–08 local),
with a significant Diebold–Mariano result. Add a synthetic fixture with a *known*
diurnal bias to `conftest.py` and assert the harmonic variant recovers it while the
scalar variant cannot.

---

## 2. Blending — objective alignment and correlation-aware combination

### 2a. Match the loss to the metric (P0)
`fit_affine` sets the bias-only intercept to the **mean** residual (OLS), but the
system promotes on **MAE**, whose optimal constant offset is the **median**
residual; `InverseMseWeights` weights by inverse *squared* error and `OnlineExperts`
minimises *squared* loss (`combine.py`, `experts.py`). Switch the bias-only
intercept to the median (or a Huberised mean) when `slope_shrinkage == 0`, and the
inverse-MSE / expert loss to L1. Cheap, metric-consistent, and more robust to the
noisy outlier provider. **Accept** when temperature MAE is unchanged-or-better and
`bias` is unchanged-or-better across buckets.

### 2b. Fix bounds-in-scoring (P0, correctness)
`finalize_point` (`blenders/protocol.py`) clips only probability targets to [0,1];
`VariableSpec.minimum/maximum` (wind ≥ 0, 0 ≤ humidity ≤ 100, precip ≥ 0) are
applied **only** at the serve boundary (`serve/predict._finite`), so the backtest
scores physically-impossible predictions (e.g. negative wind) that never reach a
user. Apply the same variable clamp inside `finalize_point` so training and scoring
see the served quantity. **Accept**: no scored prediction violates its variable
bounds; leaderboard numbers move only for methods that were emitting out-of-range
values.

### 2c. Empirical-Bayes shrinkage of thin buckets (P1)
`PerBucketFitter` has a hard cliff (full local fit at `≥ min_rows`, else global).
Replace with shrinkage toward the global state, weight ∝ local row count, so the
data-starved long-lead buckets (which dominate the live archive) stabilise
smoothly. **Accept**: lower variance of fitted coefficients across folds at
168–240 h and 240 h+, no MAE regression at well-populated buckets.

### 2d. Correlation-aware combiner (P1)
Measured error correlation gives `k_eff ≈ 1.8` of 8 live providers
([Limitations §5](limitations.md)); diagonal `inverse_mse` double-counts the
redundant sources. Add an `inverse_covariance` blender: grounded-residual
covariance with **Ledoit–Wolf shrinkage** toward its diagonal, GLS-style weights,
renormalised over availability. This is the honest test of the deliberate
diagonal-only choice; keep `inverse_mse` as the partner. **Accept**: wins where
providers are most redundant (mid leads), no worse than diagonal elsewhere.

### 2e. Feature-richer GBM (P1)
`gbm.build_features` uses `valid_hour_local`/`valid_month` as **raw integers**
(no wraparound), and no solar/trend features. Add sin/cos(hour), sin/cos(day-of-year),
solar elevation, the issue-time observation **trend** (item 3), and expose the
anchor residual `r0`. **Accept**: GBM CRPS/MAE improves and it stops needing extra
splits to approximate late-night/December.

---

## 3. Anchoring & nowcasting — the station's differentiator

### Why
The minutely product (`serve/predict.minutely_product`) uses a **fixed, never-fitted**
`minutely_tau_hours` constant, anchors against the possibly-already-anchored
selected hourly path (an undefined double-correction), and `np.interp` clamps the
"now-forecast" to the first hourly value (~0.5–1 h out, not lead 0). `Anchored`
(`blenders/anchoring.py`) uses one scalar τ per variable and anchors the **level**
residual only, discarding the clean 1-min derivative the station provides. The
short-lead regime is validated only on a synthetic; the backfill has no <24 h leads.

### Design
- **P0** — Anchor the minutely product against the **raw base** hourly path and
  **fit** its decay (reuse the training-slice τ search); put `minutely_tau` into the
  searched grid. Fixes the fixed-constant, double-anchor, and interp-clamp defects
  together.
- **P1** — `anchored_trend_*`: add a robust local-linear slope from the last
  ~10–15 min of 1-min obs (spike-guarded) so the anchor corrects tendency, not just
  level. Register as a new method.
- **P1** — Regime-conditioned τ (at least day/night) or a scalar AR(1)/Kalman
  residual filter with fitted process/obs noise and lead-adaptive gain — the
  "correct step of a Kalman filter" the theory says anchoring approximates, plus the
  missing adaptivity.
- **P2** — Station-anchored minutely **gust** and **precip onset/cessation**
  nowcasts (gust is absent; precip is provider-native-only), inheriting the
  `eventrain` reset-QC caveat (now noise-tolerant, see the truth layer).

### A/B and acceptance
**Blocking prerequisite:** close the short-lead evaluation gap — add a
minutely/short-lead backtest scored against aggregated per-minute truth with a
minute-granular `truth_known_at`, and populate 0–1 h / 1–3 h leads (live
high-cadence polling or fine-offset re-issue). Until then, keep `"no anchoring"` and
`"no trend"` in every grid so a method can only *not* help. **Accept**: a decayed/
trend anchor cuts MAE at 0–3 h and converges to the base blend by ~12 h on real
truth.

---

## 4. Probabilistic forecasting & calibration (P0)

### Why
CRPS, pinball, reliability, coverage, and PIT are implemented and unit-tested in
`metrics/probabilistic.py` but **never wired into the leaderboard** (only
`brier(pop)` is, `reports/leaderboard.py`), and no wave-1 method emits quantiles —
so no distribution can currently be measured. `crps_from_quantiles` is
`2·mean(pinball)`, a rectangle rule that is only unbiased for uniformly-spaced
levels and ignores tail mass. There is no quantile-crossing guard.

### Design
- **P0** — Wire probabilistic scoring into the leaderboard reading the existing
  `quantiles_json`: CRPS (via `crps_ensemble` over the sorted grid), pinball@levels,
  reliability with **equal-count** bins + a Brier reliability/resolution
  decomposition, coverage@{50,80,90}, and a PIT-histogram χ². Report **sharpness**
  (mean interval width) beside reliability, always stratified by lead × hour-of-day
  × season. Add a shared monotone-rearrangement (`np.sort`) next to `finalize_point`
  so emitted quantiles never cross.
- **P0** — Ship the first distribution: **split-conformal / Adaptive Conformal
  Inference (ACI)** on top of the current winning point method — residual quantiles
  per (lead bucket × hour-of-day), with an online ACI step that holds target
  coverage as the diurnal/seasonal cycle drifts. Near-free, distribution-free, and
  self-calibrating on a thin non-stationary archive.
- **P1** — **EMOS/NGR** for temperature (`N(a+Σbᵢxᵢ, exp(c+d·spread))`, CRPS-fit,
  variance linked to inter-provider spread **and** hour-of-day) and a **quantile-GBM**
  head reusing `gbm.build_features` (promote `source_spread`; add sin/cos hour/doy so
  dispersion is diurnal/seasonal-aware).
- **P2** — Two-stage **zero-inflated precipitation**: occurrence `P(precip>0)`
  (Brier + reliability/resolution) then a gamma/lognormal amount conditional on
  occurrence; cohere `pop` with `precip_mm` (today `pop=0` can co-occur with
  `precip>0`). Add a circular (von Mises / angular CRPS) metric before ever blending
  wind direction.

### Caveat
Providers share GFS/ECMWF parents, so any spread computed across provider columns is
**structurally under-dispersed**; a distributional method must inflate spread or
model the shared-parent correlation, not treat providers as independent members.

### Acceptance
CRPS/PIT/coverage appear on the leaderboard with synthetic coverage/drift fixtures;
the conformal baseline achieves near-nominal stratified coverage; distributional
numbers are marked illustrative until the live archive is deep enough
([Limitations §5](limitations.md)).

---

## 5. Self-improvement over time (P0/P1)

### Why
The "online" experts are a **batch replay from a uniform prior** each predict
(`serve/predict._fit_methods` calls `factory().fit(...)` per request; the
`ArtifactStore`/`to_state`/`from_state` machinery is never used on the serve path);
there is **no scheduler / auto-re-backtest**; and the self-verification loop is
**open** — `reports/verification.py` computes `mae_gap` (served-vs-realised) and it
is only written to a report, never fed back. Promotion (`serve/selection.py`,
`reports/leaderboard.py`) does one DM test against one reference with no
multiple-comparison control, selects the challenger by argmin MAE (winner's curse),
and decides on the same scores used to rank.

### Design
- **P0** — Close the loop: feed `verify_history`'s `live_mae`/`mae_gap` into
  `slice_winners` as a promotion **gate** and precision-weighted tiebreaker,
  defaulting to backtest-only under thin live evidence.
- **P0** — A scheduled `retrain` job (re-backtest → `select_methods` → report) with
  an append-only **promotion ledger**, leveraging the idempotent `ModelRelease` hash
  (`evaluation.py`). This is what makes "improves as the archive grows" actually
  happen unattended.
- **P1** — Selection-bias controls: Benjamini–Hochberg FDR across all slice DM
  tests, a **held-out promotion fold**, and hierarchical shrinkage of thin-slice MAE.
- **P1** — De-seasonalised per-provider **change-point detection** (CUSUM /
  Page-Hinkley / BOCPD) that triggers window-shortening / down-weight / an operator
  alert — turning a provider backend swap (the event the experts exist for,
  [Limitations §10.3](limitations.md)) into an explicit signal.
- **P1** — Phase-conditioned, recency-weighted, Bayesian-updated grounding
  (bias-only), so grounding improves with archive size instead of being a static
  all-history OLS. Recency **must** be season-aware (see the cross-cutting
  principle).
- **P2** — Warm-started incremental state (activate `ArtifactStore`) so experts /
  grounding update across serves; archive-driven hyperparameter tuning (`share`, τ
  grid, `slope_shrinkage` — `share = 0.005` was set on a synthetic fixture);
  leave-one-provider-out marginal-skill / value-of-information to prune redundant or
  out-of-domain providers (e.g. the marine `stormglass`).
- **P3** — Station-sensor fault detection on the cross-method live-`bias` invariant
  (a failing radiation shield "can look plausible for months"): suppress anchoring/
  promotion for a variable when every method's bias shifts together.

### Acceptance
A drift fixture where a provider swaps backends causes an alert and a re-fit within
days; auto-promotion churn is auditable in the ledger; the held-out fold reduces
false winners on the 12-methods × 10-buckets grid.

---

## 6. Statistics & evaluation integrity (P1)

- **DM autocorrelation.** `metrics/dm.py` is correct (Bartlett HAC, HLN, Student-t),
  but its `horizon_steps` is derived from the bucket lead, while the **dominant**
  loss autocorrelation is that dozens of consecutive 10-min issue snapshots forecast
  the **same valid hour**. Set the HAC lag from the temporal-overlap structure
  (block by `valid_time`, or collapse to one loss per valid_time) so p-values are not
  overconfident.
- **Leaderboard intersection.** `reports/leaderboard.py` restricts each slice to
  cases where **every** method produced a non-null prediction. Score each method on
  its own cases; use pairwise-common cases only inside the DM comparison.
- **Effective-n.** Report distinct provider vintages beside `n`, because the as-of
  join pseudo-replicates slow-refresh sources across snapshots and inflates the raw
  count.
- **Alignment study.** `dataset/alignment.py` needs `_MIN_ROWS = 72` overlapping
  rows before it expresses a semantics preference, so on a thin archive every
  dual-semantics variable silently defaults to instantaneous. Report which variables
  run on the default vs a data-backed recommendation; hard-code the known-unambiguous
  cases; adapt the threshold for a thin archive.

---

## 7. Prerequisite: solar-geometry features (enables 1, 2e, 4)

Add a small, dependency-free module that computes solar zenith/elevation,
azimuth, and a clear-sky index from `latitude`, `longitude`, `elevation_m`, and
`valid_time`, exposed as leakage-safe matrix feature columns (they are
deterministic functions of the valid instant and the fixed site, never of truth —
confirm they pass the `t__`-prefix feature audit and the poisoning sentinel).
`elevation_m` is now a required config field, so the site is fully specified.

---

## Sequencing summary

| Order | Item | Effort | Unlocks |
|---|---|---|---|
| 1 | Solar-geometry features (§7) | medium | §1, §2e, §4 |
| 2 | Median/L1 objective + bounds-in-scoring (§2a, §2b) | low | metric consistency |
| 3 | Harmonic/solar grounding (§1) | medium | the cyclicality win |
| 4 | Wire probabilistic scoring + conformal (§4 P0) | medium | calibrated products |
| 5 | Minutely-τ fit + short-lead eval (§3 P0) | medium | trustworthy nowcast |
| 6 | Close verification loop + scheduled retrain (§5 P0) | low | self-improvement |
| 7 | Everything P1 | — | after the above are measured |

Every item lands as a registered method or an additional report column, so none of
this changes what is served until the leaderboard promotes it.
