# Evaluation: measured state and improvement opportunities

_2026-07-26. Follow-up to `improvement-methods-2026-07.md` (2026-07-18): that
document's roadmap is now largely implemented, so this one starts from what the
**live data measures** — one week of served forecasts, today's leaderboards, the
self-verification table — and maps the remaining gaps. Data below is from the
2026-07-26 evaluation runs (`511f3375fca62fad` hourly, `90b0624cc3cb88cf` daily)
and the Jul-19 dataset build (`4051fe7992605cc7`)._

---

## 1. The binding constraint is operational, not methodological

The modeling machinery from the July roadmap is nearly all in place (§3). What
is starving it is evidence flow:

1. **The dataset is a week stale.** `build-dataset` last ran 2026-07-19
   (manifest `created_at`, forecasts_long max `fetched_at` = Jul 19 13:24). The
   weekly launchd job did not fire on Sunday Jul 26 (machine asleep at 06:30).
   Tonight's backtest/report ran against the Jul-19 matrices, so the live
   backtest still spans **3 days of issue times (Jul 16–19), 2 folds** — while
   a full extra week of collected snapshots sits unprocessed in SQLite.
2. **Ensemble ingestion has never run.** `dataset/ensembles.py` and the
   `ens__*` feature path are implemented, but `config.toml` has **no
   `[ensembles]` section** and `ingest-ensembles` appears nowhere in the runs
   ledger. Zero `ens__` columns exist in the hourly matrix. EMOS is therefore
   fitting its spread term against provider-column spread — the structurally
   under-dispersed signal the roadmap explicitly warned against.
3. **The synthetic leaderboard predates the new methods.** The synthetic score
   files are from Jul 16 and cover only 12 of the 26 registered methods
   (none of: ewma/harmonic/median grounding, trimmed, conformal, emos, idr,
   anchored_fitted/trend). The 14 new methods have never seen the multi-month
   archive; they have only ever been judged on 2 live folds. The synthetic
   issue range is also short (2026-03-08 → 05-14) relative to what Previous
   Runs (~2 years) and dynamical.org can provide.
4. **Promotion is (correctly) refusing to act on 2 folds.** At temp 0–1h,
   `persistence` measures MAE 1.129 vs served `equal_weight` 1.699 (DM
   p = 0.012, skill +0.35) — and `boa`/`ewa`/`idr`/anchored all sit between —
   but the MCS gate keeps the incumbent because the confidence set cannot
   separate 26 methods on 60 valid hours. That is the designed behaviour; the
   fix is folds, not a looser gate.

**Actions (highest leverage per unit effort in the whole document):**

- Run `build-dataset` → `backtest --source live` → `report` now, and re-run the
  **synthetic** backtest so the 14 new methods get months of folds tonight
  rather than in October.
- Add `[ensembles]` to `config.toml` (copy from `config.example.toml`) and
  schedule `ingest-ensembles` per model cycle; `build-dataset` after each.
- Fix the scheduler miss mode: the weekly Sunday-06:30 `StartCalendarInterval`
  silently skips when the machine sleeps through it. Either switch to the daily
  02:15 `maintain` template in `docs/launchd/`, or anchor the cadence to
  wake-safe scheduling (`StartInterval`, or RunAtLoad with a freshness check in
  the script). A missed run currently costs a week of folds *and* leaves any
  code change serving degraded until the next fire.
- Extend the synthetic backfill window (Previous Runs reaches back ~2 years for
  open models; dynamical.org fills sub-24h leads).

---

## 2. What the live week actually measured

### 2.1 Wins the leaderboard already shows

- **Online experts work.** `boa`/`ewa` lead the trained methods at 0–12h
  (e.g. 1.287 at 6–12h vs 1.536 equal_weight) and their warm-started
  delayed-feedback state survives serves.
- **Anchoring works but under-uses the station at lead → 0.** Every anchored
  variant beats its base at 0–1h (1.46–1.48 vs 1.59–1.72), yet raw
  `persistence` (1.129) beats them all. The anchor gain should approach 1 as
  lead → 0 (persist-then-ramp); it demonstrably does not.
- **Distributional heads win point MAE on wind.** `emos` takes wind_speed at
  every bucket and `idr`/`emos` split wind_gust — but see §2.3 on why wind MAE
  is a misleading victory.
- **`affine_equal_weight` (slope grounding) is the best method at 24–48h**
  (1.482 vs 1.743 equal_weight) — first live evidence that the opt-in slope
  earns its keep at longer leads, worth watching as folds accumulate.
- **Grounding still does not beat raw equal weight on temp** (1.718 vs 1.699 at
  0–1h; 1.760 vs 1.743 at 24–48h): with a July-only live archive the intercepts
  have nothing seasonal to remove yet, and `harmonic_grounded` ≈ no-op (thin
  buckets fall back to scalar). The EWMA variant is consistently between the
  two. This question will only be answered by cross-season folds.
- **GBM is a hazard on thin data.** Temp MAE 2.5–3.7 (worse than
  `best_provider`); daily Tmax 9.0 vs climatology 8.8. It nevertheless wins
  pressure mid-leads and some degenerate pop slices, so it is being promoted
  where its wins may be flukes. Consider a min-training-rows gate on its
  promotability (or exclude it from promotion until the archive matures).

### 2.2 Served-vs-realized (self-verification) caught real failures

- **Winner's curse, live:** a release promoted `affine_equal_weight` for
  dew_point 3–6h/6–12h; it served with a **constant −3.2 to −3.5 °C bias**
  (live MAE 3.16/3.46 vs backtest 2.05/2.13). The 1.5× live demotion gate is
  the right backstop; empirical-Bayes shrinkage of thin-bucket fits
  (plan §2c, still missing — `PerBucketFitter` cliff) is the prevention.
- **Earlier releases served pressure with 25–29 hPa MAE** (pure offset; see
  §2.3). Current release serves grounded methods (~1–2 hPa). The self-check
  works; the underlying station offset remains.
- Wind gust served `equal_weight` with +3.8 to +6.0 m/s bias before
  idr/emos were promoted.

### 2.3 Data-quality findings that cap what any method can do

1. **Station pressure is ~28–29 hPa off.** Truth-vs-provider "sea-level
   pressure" shows a constant ≈28.5 hPa offset — an Ambient console
   relative-pressure calibration error, not weather. Grounded methods absorb
   it, but every raw-blend serve and the `pressure` product's absolute values
   are wrong in the meteorological frame. Fix at the console (set relative
   pressure from a nearby METAR/altimeter), or rename the variable
   station-frame.
2. **The anemometer is effectively sheltered.** Station wind truth: mean
   0.36 m/s, median 0.29, max 2.13 over the whole archive, while providers
   forecast 2.3–4.5 m/s (10 m exposure). EMOS's 0.16 MAE ≈ "predict calm
   always" — sharp but nearly information-free, and correlation with provider
   wind is 0.2–0.3 for 8 of 10 sources. Decide what the wind product means:
   station-frame (fine, but say so), or fix siting (hardware), or lean on gust
   (truth mean 1.97, max 7.7 — more signal, and the alignment r for gust is
   healthier).
3. **`weatherapi` dew_point ingestion is broken**: correlation with truth is
   **negative** (−0.30 inst / −0.38 mean, n=1795) while its temp is fine. That
   is a units/field bug upstream (the −0.32 anomaly the July research flagged
   now has a concrete locus). Until fixed, it poisons every dew_point blend it
   enters.
4. **Coverage holes:** meteosource returns no humidity/dew/pressure (n=0);
   stormglass no dew/pressure; weatherbit is dead (lapsed key, 38 rows —
   remove it or renew it; `weather_unlocked` likewise lapsed upstream).
5. **A real July monsoon bust is in the archive:** truth recorded 8.6 mm in an
   hour with wet-hour pop = 1.0 while no provider exceeded pop 0.3. Consumer
   APIs will keep missing convective onset here; this is the concrete case for
   ensemble-PoP + station-conditioned precip nowcasting (§4) before the wet
   season.
6. **The drift detector saturates on zero-inflated variables.** Page–Hinkley
   statistics of 1e5–4e5 for precip/pop (vs ~30–45 for temp/wind alarms) mean
   the scale heuristic (`lam ≈ 4√n`, floor 25) is meaningless for
   near-constant series — every precip/pop alarm is noise. Use a robust scale
   (MAD with a floor) or exempt degenerate series; otherwise real alarms will
   be ignored.
7. **Provider redundancy confirmed at r ≈ 0.97** (nws / pirate_weather /
   visual_crossing temp errors). The diverse tail is met_norway (0.43–0.48 vs
   that cluster), stormglass (0.10–0.54), and the weatherapi/weatherbit oddity
   (0.12). Supports the roadmap's cluster-and-keep-one subset idea; with the
   dead keys removed the effective set is ~5.

### 2.4 Product-level gaps the week exposed

- **Daily is unpromoted and degenerate past D2.** All daily variables serve
  `equal_weight` ("no backtest evidence for this slice"); at D2 nearly every
  method collapses to the same fallback (identical 3.879 MAE), and D3+ has no
  resolved truth yet. Daily Tmax MAE ~3.4–3.9 °C at D1 is poor — partly thin
  data, partly that Tmax/Tmin deserve their own supervised treatment (features
  = blended hourly path + provider daily values) rather than inheriting the
  hourly machinery. Watch as folds accumulate; revisit §4.
- **PoP serves `climatology` (≈0)** for most buckets — optimal in a dry July,
  useless in December. The Brier/reliability machinery is wired; it has
  nothing to discriminate until wet-season data (or wet-season synthetic
  backfill) exists.
- **Minutely product still has a hand-set τ** on its fallback path
  (`minutely_tau_hours = 3.0`) and no minutely/short-lead backtest product
  exists, so the nowcast remains unevaluated (the roadmap's "blocking
  prerequisite" is still open).

---

## 3. Code status vs the July roadmap

Implemented (with real quality): EWMA hour-binned grounding (w=0.05, 8 bins,
count shrinkage), harmonic/solar grounding (ridge, solar-elevation + annual +
semiannual design), median-intercept grounding, trimmed means, L1 inverse
weights, bounds-in-scoring, solar features (`solar_elevation_deg`, `toa_wm2`,
hour/doy sin-cos), EMOS with truncated-normal family + ens-spread preference,
IDR, conformal-PID (day/night × lead cells, proper calibration split),
CRPS/pinball/coverage/PIT/sharpness on the leaderboard, quantile non-crossing +
cross-variable coherence, warm-started delayed-feedback experts (cursor +
digest), two-tier drift report (consensus z + Page–Hinkley), MCS promotion with
per-valid-time loss collapse, live demotion gate (1.5× at n≥24), fitted
per-lead anchor weights (`AnchoredEmpirical`), 15-min obs trend feature,
dynamical.org backfill (GEFS, AIFS-ENS).

Still missing or half-done (each was in a plan):

| Gap | Where | Why it matters now |
| --- | --- | --- |
| Experts minimise **squared** loss | `experts.py:376` | Promotion is on MAE; boa/ewa are current mid-lead leaders — cheap consistency win |
| CRPS still rectangle-rule | `leaderboard.py:130` (`crps_ensemble` exists unused) | Distributional promotions ride on a biased estimator |
| `PerBucketFitter` hard cliff | `protocol.py:96` | The dew_point −3.5 °C serve is the measured cost |
| Anchored weight curve flat-fills thin short-lead bins and never ramps toward 1 at lead→0 | `anchoring.py` `_fit_bins`/`_weights_at` | Persistence still beats anchored at 0–1h (correction *reach* already extends past 6 h; only the anchor-source gate is 6 h, which is honest) |
| Minutely τ hand-set; no minutely backtest | `predict.py:968`, engine | Nowcast unevaluated |
| No NBM ingestion | — | The benchmark-to-beat and a strong input, still absent |
| No inverse-covariance blender, no provider clustering/k_eff | — | Honest test of the diagonal choice; cheap given the measured 0.97 correlations |
| No BH-FDR / winner's-curse-corrected reporting; MCS knobs hard-coded | `mcs.py:22,67` | Promotion honesty at scale of 26 methods × slices |
| No zero-inflated precip stage; emos/idr excluded from precip instead | `registry.py:53-58` | The wet season will arrive with no calibrated PoP/amount path |
| Drift alarms feed nothing automatic | `drift.py` | Backend-swap response is still manual |
| Alignment `_MIN_ROWS=72` unchanged | `alignment.py:25` | Semantics defaults still silent on thin variables |
| Harmonic fit/predict feature mismatch silently no-ops | `harmonic_grounding.py:113` | Thin-bucket vs schema-bug indistinguishable |

---

## 4. Research additions (beyond the 2026-07-18 synthesis)

_Findings from the fresh paper/web sweeps are appended in §4.1–§4.2; the July
synthesis remains the canonical method bibliography._

### 4.1 Methods literature (new findings, mapped to measured gaps)

**For the 0–1h gap (persistence 1.13 vs anchored 1.46, §2.1):**

- **RAFT** — Schuhen, Thorarinsdottir & Lenkoski 2020, QJRMS
  ([10.1002/qj.3718](https://doi.org/10.1002/qj.3718)); companion
  [10.5194/npg-27-35-2020](https://doi.org/10.5194/npg-27-35-2020). Each time an
  observation verifies, re-correct the *remaining* leads of the issued
  trajectory via per-lead-pair error-correlation regressions (order: EMOS →
  RAFT → ECC). A principled, few-parameter upgrade over exponential-decay
  anchoring, trainable on weeks of data — the most direct answer to the
  measured short-lead gap.
- **Seamless multimodel postprocessing** — Dabernig & Atencia 2024
  ([arXiv:2410.11916](https://arxiv.org/abs/2410.11916)). Two tricks that map
  1:1 onto this project: carry a source's last lead forward as "model
  persistence" when its horizon ends (ragged horizons without sleeping-expert
  machinery), and add the latest station observation as a regression predictor
  so the blend closes seamlessly onto live obs. Thin-data linear regression.

**For self-updating fits (the EWMA/experts axis):**

- **ondil** — Hirsch, Berrisch & Ziel
  ([arXiv:2407.08750](https://arxiv.org/abs/2407.08750), Python pkg
  [`ondil`](https://github.com/simon-hirsch/ondil)). Incremental LASSO-regularized
  distributional regression (GAMLSS-type) with forgetting factors — one engine
  that could replace the static EMOS refit *and* the EWMA intercepts with
  per-sample `update()` calls, CPU-cheap and interpretable.
- **Training-window design** — Lang et al. 2020
  ([10.5194/npg-27-23-2020](https://doi.org/10.5194/npg-27-23-2020)):
  seasonally *weighted* all-history training beats short sliding windows at
  mountain stations, even across NWP model changes. Directly informs how to
  weight 2 years of synthetic vs weeks of live data in per-source fits (a
  weighting the live/synthetic separation currently forbids — worth a
  deliberate, flagged exception at fit time, never at scoring time).
- **Weighted conformal under drift** — Barber, Candès, Ramdas & Tibshirani
  2023 ([arXiv:2202.13415](https://arxiv.org/abs/2202.13415)): fixed-weight
  discounting of older/synthetic calibration points with a quantified coverage
  penalty — the theory that would let the conformal layer use synthetic
  residuals safely.

**For the wet season (PoP/precip, §2.4):**

- **Online Platt scaling + calibeating** — Gupta & Ramdas 2023, ICML
  ([arXiv:2305.00070](https://arxiv.org/abs/2305.00070)): two-parameter online
  logistic recalibration, provably robust under drift, no tuning — the best-fit
  PoP-from-tipping-bucket recalibrator found.
- **Beta calibration** (Kull et al. 2017, AISTATS) and **Venn–Abers** (Vovk &
  Petej 2014, [arXiv:1211.0025](https://arxiv.org/abs/1211.0025); generalized
  [arXiv:2502.05676](https://arxiv.org/abs/2502.05676)): drop-in upgrades when
  logistic misbehaves near 0/1; Venn–Abers gives finite-sample validity with
  ~50–100 wet samples.

**For the daily product (§2.4):**

- **Joint Tmin/Tmax** — Meng & Taylor 2022, IJF
  ([10.1016/j.ijforecast.2021.05.007](https://doi.org/10.1016/j.ijforecast.2021.05.007)):
  calibrate Tmax/Tmin marginals (EMOS-style), restore their dependence with
  ensemble copula coupling. Closed-form, low-parameter.
- **ECC / member-by-member** — Schefzik 2017, QJRMS
  ([10.1002/qj.2984](https://doi.org/10.1002/qj.2984)): calibrate hourly
  marginals, reorder by raw multi-source trajectory ranks, take max/min of the
  reconstructed paths — "extreme-of-path" Tmax/Tmin with zero extra fitted
  parameters. The direct-vs-path comparison appears genuinely unpublished at
  single-station scale — a novel leaderboard experiment.

**Validation that the current architecture is right (context for §1):**

- Kocsis & Baran 2026 ([arXiv:2606.02508](https://arxiv.org/abs/2606.02508)):
  per-station EMOS/QR on operational AIFS-ENS and IFS-ENS with only ~5 months
  of archive works, and parametric EMOS beats QR at short leads — this
  project's exact data regime.
- Trotta et al. 2025, BAMS ([arXiv:2504.12672](https://arxiv.org/abs/2504.12672)):
  standard postprocessing chains transfer unchanged to AI models, and blending
  AI + conventional adds skill even where the AI model alone is worse — treat
  AI-backed feeds as ordinary members.
- Asch et al. 2026 ([arXiv:2606.19642](https://arxiv.org/abs/2606.19642)):
  online conformal restores extreme-tail coverage that "well-calibrated" AI
  ensembles lose — supports the conformal-PID layer once ens features flow.

**Later, at archive maturity:**

- **Bayesian predictive synthesis** — McAlinn & West 2019, J. Econometrics
  ([10.1016/j.jeconom.2018.11.010](https://doi.org/10.1016/j.jeconom.2018.11.010)):
  dynamic latent-factor synthesis that learns time-varying bias *and*
  inter-source dependence — the principled endgame for 10 correlated sources
  on one series (BMA/QRA/CRPS-learning are special cases).
- **D-vine copula quantile regression** — Jobst, Möller & Groß 2023, QJRMS
  ([10.1002/qj.4521](https://doi.org/10.1002/qj.4521)): collinearity-robust
  nonlinear quantile regression at a single station — the dependence-aware
  alternative to QRA.

### 4.2 Data sources / operational (verified 2026-07-26)

**Breakage — affects existing code:**

- **Synoptic's Open Access program is now restricted to accredited .edu
  research users.** The `truth-qc` neighbor cross-check
  (`dataset/neighbors.py`) has likely lost its data source unless credentials
  are grandfathered. Alternatives: WeatherFlow **Tempest** public stations
  (developer API free, REST/WebSocket) and `api.weather.gov` METAR
  observations.
- **dynamical.org deprecated plain Zarr paths on 2026-07-23** (Icechunk 2.0
  migration mandatory, Python 3.12+ `dynamical-catalog` library).
  `backfill --provider dynamical` should be smoke-tested and migrated. Upside:
  the catalog grew — ECMWF **IFS ENS** (Nov 2025), **AIFS Single** (Apr 2026),
  ICON-EU, **MRMS** (Apr 2026), ASOS obs, low-latency HRRR virtual dataset
  (Jul 2026).
- **Open-Meteo retains individual ensemble members only ~3 days.** Self-polling
  remains the only member archive — but the new **Ensemble Mean API**
  (mean + spread, most models archived since **March 2026**) can *backfill* the
  `ens__*` mean/sd features months before live ingestion starts. Also new: the
  **Single Runs API** (May 2026) serves full archived runs (IFS HRES back to
  2024-03) — a second backfill channel at native leads for models already in
  the provider set.
- WeatherNext Gen (1.0) datasets were deprecated 2026-07-15; WeatherNext 2 is
  the supported line (still on Open-Meteo Ensemble API, 64 members).

**New diversity worth adding (free tiers):**

- **Google Weather API** (Maps Platform, GA): hourly to 240 h + daily 10 d,
  **10,000 calls/month free** — a genuinely different (DeepMind-stack) feed;
  Google is currently absent from the 10-source archive.
- **Vaisala Xweather**: free tier 15,000 accesses/month — an independent blend.
- **NBM v5.0** (operational 2026-05-05, hourly guidance now to 48 h): text
  bulletins (NBH/NBS/NBE) from NOMADS remain the easiest free point access;
  GribStream preserved pre-cutover v4 history (the de facto NBM archive, free
  tier / $9.90-mo). Still the benchmark this blend must beat.
- **RRFS v1 + REFS ensemble become operational 2026-08-31** (HRRR continues in
  parallel): a 3-km CAM *ensemble* for short-lead mountain precip/wind spread,
  on AWS `noaa-rrfs` via Herbie or GribStream — plan a schema slot before the
  wet season.
- MRMS QPE: unchanged on AWS, plus dynamical.org's Icechunk mirror — the
  radar-truth complement for the precip nowcast, with the known beam-blockage
  caveat at 1400 m.

---

## 5. Sequenced recommendation

| # | Action | Type | Effort | Expected effect |
| --- | --- | --- | --- | --- |
| 1 | build-dataset + live & **synthetic** re-backtest + report; fix scheduler miss mode | ops | S | Folds for 26 methods; promotions unblock; the single biggest MAE delta available this week |
| 2 | `[ensembles]` in config + ingest cron | ops | S | Honest spread for EMOS/conformal; convective-PoP signal |
| 3 | Fix weatherapi dew_point ingestion; drop dead providers; recalibrate console pressure | data | S | Removes a poisoned source and a 28 hPa product error |
| 4 | Robust drift scale for zero-inflated series | code | S | Alarms become meaningful before the wet season |
| 5 | Experts → L1 loss; CRPS → `crps_ensemble` | code | S | Metric consistency for the current leaders |
| 6 | Anchor: extend eligibility past 6 h, persist-then-ramp toward persistence at lead→0, fit minutely τ | code | M | Captures the measured 0.57 °C gap at 0–1h |
| 7 | Shrinkage in `PerBucketFitter` | code | M | Prevents the next −3.5 °C thin-bucket serve |
| 8 | Zero-inflated precip (occurrence × amount) + ensemble PoP before the wet season | code | M | The monsoon-bust class of error |
| 9 | Daily Tmax/Tmin as supervised targets over the hourly path | code | M | The weakest product per °C of MAE |
| 10 | NBM ingest as source + benchmark | data | M | The strongest single input available |
| 11 | Smoke-test/migrate dynamical backfill (Icechunk); replace Synoptic in truth-qc (Tempest/METAR) | ops | S–M | Two silent breakages from upstream changes |
| 12 | Backfill `ens__*` features via Ensemble Mean API (archived since 2026-03) | data | S | Months of spread features without waiting on live ingestion |
| 13 | Add Google Weather API (10k/mo free) and Xweather (15k/mo free) to the provider rotation | data | S | Real model diversity; replaces the two dead keys |

Items 1–3 require no new modeling and dominate everything below them.
