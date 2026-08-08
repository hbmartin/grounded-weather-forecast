# Methods and systems to improve grounded-weather-forecast

_Research synthesis, 2026-07-18. Four parallel literature/web sweeps (statistical
post-processing & combination; online learning & conformal calibration;
nowcasting & PWS quality control; data sources & ML weather models) mapped onto
this codebase's measured failure modes. Complements — and in places supersedes —
`docs/improvement-plan.md`; a cross-reference table is at the end._

---

## 0. Diagnosis: why it currently underperforms

From the code, the leaderboards, and `docs/limitations.md`:

1. **Grounding — the core thesis — is losing.** `grounded_equal_weight` trails
   raw `equal_weight` at 24–96 h (MAE 1.367 vs 1.336 at 24–48 h; 1.713 vs 1.593
   at 48–96 h, with DM p = 0.001). Cause: `AffineGrounding` fits one static
   intercept per (source, variable, lead bucket) over all history, so diurnal
   and seasonal bias structure averages to ~zero and the seasonally
   unrepresentative window injects error instead of removing it.
2. **The live leaderboard is empty.** The archive is too young to form folds, so
   nothing can be promoted and `predict` serves degraded equal weight.
3. **Providers are redundant** (error correlations 0.5–0.9; effective ensemble
   size ~1.8 of 8), so weighting schemes have almost nothing to win, and
   provider-column spread is a dishonest uncertainty signal.
4. **Anchoring is unfitted and unevaluated.** τ comes from a grid search that
   the synthetic backfill (24 h-multiple leads only) can never exercise below
   24 h; the minutely product double-anchors and uses a never-fitted constant.
5. **No calibrated probabilistic output.** CRPS/PIT/reliability are implemented
   but unwired; no method emits quantiles.
6. **The self-improvement loop is open.** "Online" experts are batch-replayed
   from a uniform prior at every serve; self-verification is written to a report
   and never fed back; promotion does argmin-MAE with one DM test (winner's
   curse across 12 methods × 10 buckets).
7. **Truth is trusted on faith.** Radiation-shield heating, gauge undercatch,
   and slow sensor drift are acknowledged but undetected (no external
   cross-check, no physical error model).

The literature has direct, proven answers to every one of these. The headline
finding: **the operational state of the art (NOAA's National Blend of Models)
is architecturally almost exactly this project — adaptive decaying-average bias
correction per input, then mild inverse-MAE weighting — which both validates
the design and hands it a recipe for the parts that are failing.**

---

## 1. Fix grounding: adaptive, diurnally-aware bias correction (highest priority)

Three independent research threads converged on the same answer.

### 1a. Hour-of-day-keyed decaying-average (EWMA) bias correction — the operational default

Maintain `bias[source, var, lead_bucket, hour_bin] ← (1−w)·bias + w·(fcst − obs)`
with w ≈ 0.02–0.1. This is literally what NCEP has run operationally since 2006
(NAEFS; [Cui, Toth, Zhu & Hou 2012, Wea. Forecasting](https://doi.org/10.1175/waf-d-11-00011.1))
and what the NBM uses per grid point/projection/element
([Veenhuis slides](https://www.weather.gov/media/mdl/Veenhuis_Presentation2015.pdf),
[NBM algorithm description](https://vlab.noaa.gov/documents/6609493/7858320/Description_of_Field-Selected_Algorithms_for_National_Blend_of_Models.pdf)).

Why it beats the current static fit *and* the planned harmonic fit:

- **No training window to choose.** [Lang et al. 2020, NPG](https://npg.copernicus.org/articles/27/23-2020) —
  evaluated at mountain stations — shows sliding short windows are the *worst*
  scheme for seasonally varying error; adaptive/EWMA and smooth-seasonal
  coefficients are the fixes. This is exactly the §4.1 grounding failure.
- **Survives archive gaps** (state goes stale, re-converges in ~1/w days) and
  adapts to season drift automatically — no phase model needed.
- 4–8 hour-of-day bins per lead bucket restore the diurnal structure the
  current per-bucket intercept averages away, at the cost of a handful of
  online scalars per source.

Converges in ~2–8 weeks of data; degrades gracefully to zero correction.
Register as e.g. `ewma_grounded_equal_weight` beside the current methods.

### 1b. The published single-PWS template: moving-average bias + lead-time Kalman filter

[Alerskans & Kaas 2021, Met. Apps](https://doi.org/10.1002/met.2006) compared six
adaptive schemes on **100 private weather stations** — the closest published
analog to this project. Winner: a long-memory moving-average bias term (≈ §1a)
**plus a lead-time Kalman filter** that propagates today's observed short-lead
error to longer leads, explicitly designed to need "no long record of
observations." Their MA+KF decomposition maps 1:1 onto grounding + anchoring.
Supporting lineage: [Homleid 1995](https://journals.ametsoc.org/doi/abs/10.1175/1520-0434%281995%29010%3C0689%3ADCOSTS%3E2.0.CO%3B2)
(eight time-of-day bias states — the diurnal KF), [Crochet 2004](https://doi.org/10.1017/s1350482704001252)
(adaptive per-station estimation of the KF noise variances, i.e. the gain and
decay are *fitted*, not hand-set — operational at met services for decades).

### 1c. SAMOS: standardized-anomaly grounding — the batch/mountain-terrain alternative

Transform obs and each source to standardized anomalies w.r.t. smooth
harmonic-in-(hour, day-of-year) site climatologies, then fit one small
regression valid across all seasons/hours
([Dabernig, Mayr, Messner & Zeileis 2017, QJRMS](https://doi.org/10.1002/qj.2975);
developed for Alpine terrain; modern extension
[MIXSAMOS-GB 2024](https://arxiv.org/abs/2412.09583)). The bias model no longer
needs to *learn* the diurnal cycle — it is removed before fitting, so all rows
pool into one seasonally valid fit. Key trick for the thin station record:
build the climatology from the *provider* archive or ERA5 rather than station
obs (the harmonic climatology has only ~10 dof). Météo-France's operational
QRF similarly works in forecast-anomaly space
([Taillardat & Mestre 2020](https://doi.org/10.5194/npg-27-329-2020)).
Cheap parametric middle road: add 4–6 sin/cos(hour, doy) terms to the intercept
(tsEMOS shows this significantly beats static EMOS —
[Jobst, Möller & Groß 2024](https://arxiv.org/abs/2402.00555)).

**Recommendation:** implement §1a first (days of work, no new deps, matches the
NBM), register the improvement plan's harmonic/solar variant as the batch
challenger informed by §1c, and let the leaderboard arbitrate. Fold §1b's
lead-time KF into the anchoring redesign (§3).

---

## 2. Fix data starvation: diverse sources, honest ensembles, deeper backfill

The scarcest resources are (i) genuinely decorrelated sources and (ii) honest
spread. Both became cheap in 2025–2026.

### 2a. ML-model ensembles via the Open-Meteo Ensemble API — zero new auth

The [Ensemble API](https://open-meteo.com/en/docs/ensemble-api) (free
non-commercial, same client already in use) now returns **individual members**
for, among others:

| Model | Members | Why it matters |
|---|---|---|
| **Google WeatherNext 2** (`google_weathernext…`, [announcement](https://openmeteo.substack.com/p/google-weathernext-2-is-available)) | 64 | FGN-based; >10% avg CRPS improvement over ECMWF ENS on >99% of targets ([arXiv:2506.10772](https://arxiv.org/pdf/2506.10772)); genuinely decorrelated from the physics repackagers. Also now powers the already-polled Google Maps Weather API. |
| **ECMWF AIFS-ENS** (`ecmwf_aifs025_ensemble`) | 51 | Operational Jul 2025, v2 May 2026; 5–15% error reduction vs IFS ([GMD 2026](https://gmd.copernicus.org/articles/19/4703/2026/)); CC-BY open data since Oct 2025. |
| **NOAA AIGEFS** (`ncep_aigefs025`) | 31 | GraphCast-derived, operational at NCEP Dec 2025 ([NOAA](https://www.noaa.gov/news-release/noaa-deploys-new-generation-of-ai-driven-global-weather-models)). |
| ECMWF IFS-ENS, GEFS, ICON-EPS, GEM, UKMO, BOM | 31–51 each | Physics ensembles for spread and diversity. |

One call per cycle for four families ≈ 200 members from independent modeling
systems: an honest predictive distribution to calibrate against the station,
replacing the structurally under-dispersed provider-column spread. Caveat:
Open-Meteo retains only the latest run for individual members — poll and
archive them (mean/spread or quantiles per variable × lead is enough).

### 2b. NBM as ingredient and benchmark

NBM v5.0 (operational May 2026) is a bias-corrected mega-blend at 2.5 km with
**percentile elements** — exactly what this project builds, with vastly more
inputs. Trivial access: NBM text bulletins (NBH/NBS/NBE/NBP) for the nearest of
9,000+ stations, plain text over HTTPS from
[NOMADS](https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod/); GRIB via
[Herbie](https://herbie.readthedocs.io/) or AWS byte-range. If the blend cannot
beat station-grounded NBM, that is the leaderboard's most informative row.

### 2c. Backfill that fills the sub-24 h gap

The Previous Runs backfill only populates ≥24 h buckets — the reason anchoring
is unevaluated. Two fixes:

- **[dynamical.org](https://dynamical.org/)**: free, keyless, analysis-ready
  Zarr/Icechunk archives on AWS — GEFS (35-day, 31 members) + GEFS analysis,
  HRRR back to 2014, **AIFS-ENS archived since Jul 2025**
  ([catalog](https://dynamical.org/catalog/ecmwf-aifs-ens-forecast/)), and MRMS.
  Full forecast cycles at native (6-hourly) steps → real 0–24 h leads for
  backtesting, and ensemble history to pre-train spread models before the live
  archive matures. Point extraction is a few lines of xarray.
- **[GribStream](https://gribstream.com/models/nbm)** (free tier / ~$10+ mo):
  as-of historical forecast queries for NBM/GFS/HRRR — NBM backfill, since
  NOAA's public NBM archives were discontinued.
- **Open-Meteo Historical Forecast + Single Runs APIs**
  ([docs](https://open-meteo.com/en/docs/historical-forecast-api)): archives of
  the exact models already polled (most from Jan 2024; full single runs
  archived from Apr 2026) — backfill every provider column that is an
  Open-Meteo model in one weekend job.

### 2d. Neighbor-station truth via Synoptic

[Synoptic Data](https://synopticdata.com/pricing/) free tier (5k requests +
5M service units/month; open-access public networks free; MesoWest classic was
retired 2025): RAWS/SNOTEL/DOT neighbors near the site that no consumer API
carries, via [SynopticPy](https://github.com/blaylockbk/SynopticPy). Feeds the
truth-QC layer (§5) and an elevation-aware truth panel. `api.weather.gov`
station observations remain the free METAR fallback.

### 2e. Minutely precipitation with radar information

Cheapest first step: Open-Meteo `minutely_15` — in North America this is real
**HRRR 15-minutely sub-hourly output** (radar-assimilating), not interpolation
([blog](https://openmeteo.substack.com/p/sub-hourly-15-minutely-weather-forecasts)).
The full option later: MRMS PrecipRate (1 km / 2-min, on AWS and dynamical.org)
+ [pysteps](https://pysteps.github.io/) optical-flow extrapolation of a small
patch around the site — with a beam-blockage check first at 1400 m (a 2025
W&F paper tunes STEPS for MRMS:
[WAF-D-24-0240.1](https://journals.ametsoc.org/view/journals/wefo/40/10/WAF-D-24-0240.1.xml)).

Also worth a look: `weatherapi`'s *negative* error correlation with every other
provider (−0.32 vs met_norway) — either it is the one genuinely diverse source
in the current set or its ingestion (timestamp semantics?) is broken. Both
possibilities are high-value to resolve.

---

## 3. Make anchoring real: fitted decay, trend state, single pass

The theory result that organizes everything: **if the obs-minus-blend residual
is stationary AR(1) with lag-1 autocorrelation ρ, the optimal anchor is exactly
exponential decay with rate −ln ρ** — so the current shape is right, but the
constant must be *fitted as the residual ACF*, and the documented deviations
from AR(1) are precisely the improvements:

1. **Fit τ from the residual ACF now.** The ACF of r(t) = obs − blend at lags
   1 min–6 h estimates ρ(τ) per variable (day/night separately) with *no*
   short-lead forecast archive needed. Near-zero code risk; kills the
   never-fitted constant.
2. **Persist, then ramp — not decay-from-zero.** INCA (validated in the eastern
   Alps — [Haiden et al. 2011, WAF](https://www.researchgate.net/publication/238035120))
   carries the observed level forward with the NWP *tendency* for ~2 h, then
   ramps the residual out by ~6 h. NOAA's LAMP fits per-lead regression weights
   obs↔MOS and finds observation information persists to ~12–20 h for
   temperature ([LAMP background](https://www.weather.gov/mdl/lamp_glmp_background))
   — an argument against forcing the anchor to zero by 3 h
   (`_ANCHOR_MAX_LEAD`).
3. **Anchor level + damped trend.** Replace the level-only residual with a
   damped-Holt (level, trend, damping φ) state on the 1-min residual series —
   uses the observed derivative the station uniquely provides, while damping
   prevents runaway extrapolation ([fpp3 ch. 8](https://otexts.com/fpp3/holt.html);
   fit via statsmodels ETS in weeks of data).
4. **One anchoring pass.** Anchor once at the minutely level against the
   *un-anchored* hourly path and aggregate minutely→hourly (the IMPROVER
   lesson: each processing step applied once, then blend —
   [BAMS 2023](https://journals.ametsoc.org/view/journals/bams/104/3/BAMS-D-21-0273.1.xml)).
   Fixes the double-anchor and the interp-clamp defects structurally.
5. **Probabilistic anchoring, later.** [Nipen, West & Stull 2011, WAF](https://journals.ametsoc.org/waf/article/26/4/564/39403/Updating-Short-Term-Probabilistic-Weather)
   is the probabilistic generalization: model the verifying CDF value as a
   fitted first-order Markov process — updated forecasts sharpen for the first
   hours and relax back to the base forecast.
6. **Start logging live issued paths immediately** (already partly done via
   `predict_history`) and, after ~60–90 days, fit inverse-MSE lead weights
   w(τ) = MSE_NWP/(MSE_now + MSE_NWP) per variable and day/night — the
   LAMP/Kober-style empirical crossover (typically ~3–6 h for temperature,
   ~1–3 h for precip).

Precipitation onset/cessation nowcast (new capability): logistic recalibration
of provider minutely PoP against tipping-bucket wet/dry outcomes (~50–100 wet
hours suffice for two parameters), plus a gauge-state-conditioned override —
if the bucket is tipping and a provider says dry within 10 min, trust the
bucket with a ~20–30 min decaying override.

---

## 4. Ship calibrated uncertainty

Sequenced by data appetite:

1. **EMOS/NGR first** ([Gneiting et al. 2005](https://doi.org/10.1175/MWR2904.1)):
   μ = a + b·blend, σ² = c + d·spread, CRPS-fit on a seasonally weighted 40–60
   day window — the original design point was thin data. Four parameters per
   variable × lead bucket; feeds the already-implemented CRPS/PIT metrics.
   With real ensemble spread from §2a, d becomes meaningful (with
   provider-column spread expect d ≈ 0 — the shared-parent caveat).
2. **IDR / EasyUQ as the zero-tuning benchmark**
   ([Henzi, Ziegel & Gneiting 2021, JRSS-B](https://doi.org/10.1111/rssb.12450);
   [Walz et al. 2024, SIAM Review](https://doi.org/10.1137/22m1541915);
   `isodisreg` on PyPI): calibrated conditional CDF from the single blend
   covariate, no hyperparameters — the universal baseline any fancier
   distributional method must beat.
3. **Online conformal wrapper for guaranteed coverage under drift.** The
   recommended stack from the 2021–2024 literature: **SAOCP**
   ([Bhatnagar et al., ICML 2023](https://arxiv.org/abs/2302.07869)) or
   **conformal PID control**
   ([Angelopoulos, Candès, Tibshirani, NeurIPS 2023](https://arxiv.org/abs/2307.16895)
   — benchmarked *on temperature forecasting*, code at
   [conformal-time-series](https://github.com/aangelopoulos/conformal-time-series))
   per variable × lead bucket, ~10 floats of state each; per-hour-of-day
   coverage verified via conditional-guarantees conformal
   ([Gibbs, Cherian & Candès, JRSS-B 2025](https://arxiv.org/abs/2305.12616))
   rather than 24 Mondrian bins that shatter a thin sample. Batch baseline via
   [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) / [crepes](https://github.com/henrikbostrom/crepes).
   Note the improvement plan's plain-ACI choice is superseded: ACI's single
   learning rate oscillates or lags; DtACI/SAOCP remove the knob.
4. **QRA for quantiles straight from point providers**
   ([Nowotarski & Weron 2015](https://doi.org/10.1007/s00180-014-0523-0);
   lasso-regularized for collinearity —
   [Uniejewski & Weron 2021](https://doi.org/10.1016/j.eneco.2021.105121)):
   pinball regression on provider points; GEFCom-winning pedigree; LightGBM
   quantile mode with a monotone constraint on the blend feature is the
   nonlinear variant. Always sort quantiles (monotone rearrangement).
5. **CRPS Learning as the unifying upgrade**
   ([Berrisch & Ziel 2023, J. Econometrics](https://arxiv.org/abs/2102.00968)):
   quantile-wise BOA with smoothing across levels — turns the *existing*
   BOA/fixed-share/sleeping machinery into the probabilistic blender itself,
   with forgetting built in. A 2026 successor, **AdaWeather**
   ([arXiv:2606.02663](https://arxiv.org/abs/2606.02663)), gets log-regret
   against the best static *mixture* on temperature — direct evidence the
   GBM-vs-experts dichotomy should eventually be unified.
6. **Defer** neural distributional methods (DRN, BQN, transformers — every
   success pools hundreds of stations or years), D-vine copulas, and BMA
   (weights degenerate under near-duplicate members). Reconsider analog
   ensembles ([Delle Monache et al. 2013](https://doi.org/10.1175/MWR-D-12-00281.1))
   at the ~12-month archive mark. Operational warning from Météo-France:
   minimizing CRPS alone does not guarantee reliability — keep reliability
   tests beside CRPS in promotion
   ([Zamo, Bel & Mestre 2020](https://arxiv.org/abs/2005.03540)).

---

## 5. Trust the truth: a station-QC layer with physics and neighbors

Everything upstream optimizes toward the station; if the station drifts, the
whole system calibrates to a broken sensor. Three cheap, proven defenses:

1. **Radiation-shield error model (detection + correction).** Daytime error of
   passively ventilated shields ≈ f(solar S, wind u), classically increasing in
   S and decreasing in u (1–3 °C at high-S/low-u for consumer shields;
   [BAST 2022 regression method](https://link.springer.com/article/10.1007/s42865-022-00046-z)).
   The station has the co-located anemometer (and pyranometer): regress daytime
   residual on S/(1+u); a significant positive slope that grows over months is
   the failing-shield signature; the fitted curve doubles as a correction and
   as inflated observation-error variance so sunny-calm observations anchor
   more weakly.
2. **Neighbor cross-check.** MET Norway's [titanlib](https://github.com/metno/titanlib)
   buddy check / Spatial Consistency Test (designed for sparse mountain
   networks; [Båserud et al. 2020](https://asr.copernicus.org/articles/17/153/2020/))
   against 3–10 lapse-adjusted Synoptic/METAR neighbors (§2d), as a daily cron:
   30-day median of station-minus-consensus, alert on drift > ~1 °C. The
   single-station-applicable core of CrowdQC+
   ([Fenner et al. 2021](https://www.frontiersin.org/articles/10.3389/fenvs.2021.720747/full)):
   a rolling ≥72 h correlation with a reference series (neighbor or the blend
   itself) below ~0.9 flags a failing sensor even when values stay plausible —
   the direct answer to "a failing radiation shield can look plausible for
   months."
3. **Gauge undercatch and rain QC.** WMO-SPICE transfer functions (catch
   efficiency as a function of wind and temperature —
   [Kochendorfer et al. 2018, HESS](https://hess.copernicus.org/articles/22/1437/2018/))
   apply directly because the anemometer is co-located; PWSQC's false-zero /
   high-influx filters ([de Vos et al. 2019, GRL](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019GL083731))
   with provider/radar QPE as the "neighbor."

---

## 6. Self-improvement: true online state, drift alarms, honest promotion

1. **Replace batch replay with delayed-feedback online updates.**
   [Flaspohler et al., *Online Learning with Optimism and Delay*, ICML 2021](https://arxiv.org/abs/2106.06885)
   was developed *for weather forecasting with late-arriving truth*: DORM+ /
   AdaHedgeD are tuning-free with optimal regret under delay, and "delay as
   optimism" (use the last resolved loss as a hint) cancels much of the delay
   penalty. Code: [poold](https://github.com/geflaspohler/poold). Concretely:
   per (variable × lead bucket), persist expert weights + a queue of
   outstanding (issue-time, pending-losses) records; when truth resolves, pop
   and apply the update. Removes the per-serve refit and activates the dormant
   `ArtifactStore` machinery.
2. **Two-tier drift detection for provider backend swaps.** The delayed-truth
   trap: residual-based detection lags by the lead time. So (i) fast alarm on
   *provider-vs-consensus* deviation at issue time (a swap is visible against
   the other 7–12 providers within hours), triggering down-weight; (ii) slow
   confirmation on truth-based residuals (BOCPD —
   [Adams & MacKay 2007](https://arxiv.org/abs/0710.3742) — or Page-Hinkley /
   ADWIN via [river](https://riverml.xyz) / [Frouros](https://www.sciencedirect.com/science/article/pii/S2352711024001043)),
   triggering a grounding-state reset. ADWIN's adaptive window length is
   directly "how much history is still valid for grounding."
3. **Promotion: sequential Model Confidence Sets instead of argmin-MAE + one DM
   test.** MCS ([Hansen, Lunde & Nason 2011, Econometrica](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=522382);
   Python: [model-confidence-set](https://github.com/JLDC/model-confidence-set),
   `arch.bootstrap.MCS`) returns the *set* statistically indistinguishable from
   best — thin data ⇒ larger set ⇒ keep the incumbent. The e-process
   **sequential MCS** ([JRSS-B 2025](https://arxiv.org/pdf/2404.18678)) is valid
   at every look, so promotion can re-run after every backtest refresh with no
   alpha-spending bookkeeping. Stopgap this week: Harvey–Leybourne–Newbold
   small-sample DM correction + Benjamini–Hochberg across the slice grid.
   Report winner's-curse-corrected scores for whatever is promoted
   ([Andrews, Kitagawa & McCloskey 2024, QJE](https://academic.oup.com/qje/article/139/1/305/7276491)).
4. **Soft promotion as the default posture.** Since full-information losses for
   *all* methods arrive with every resolved forecast, a second-level
   fixed-share aggregation *over the 12 methods* gives drift-adaptive "soft
   promotion" with regret guarantees; reserve hard promotion (interpretability,
   cost) for the sequential-MCS gate.
5. **DM autocorrelation fix** (from the improvement plan, confirmed): HAC lag
   from the valid-time overlap structure, or collapse to one loss per
   valid_time.

---

## 7. Blending: what *not* to build

The combination literature blesses the current outcome and warns against the
obvious next steps:

- **Equal-ish weights are near-optimal at this correlation and sample size.**
  Weight-estimation error, not method choice, is what loses
  ([Claeskens et al. 2016](https://doi.org/10.1016/j.ijforecast.2015.12.005));
  with ρ ≈ 0.9 the optimal-weight variance explodes, and
  [Smith & Wallis 2009](https://doi.org/10.1111/j.1468-0084.2008.00541.x)
  explicitly endorse ignoring error covariances on short samples. The NBM
  itself uses only decaying-average bias + inverse-**MAE** EWMA weights — no
  covariance terms. The planned Ledoit–Wolf `inverse_covariance` blender is
  still worth registering as the honest test, but expectations should be low;
  cap its weight deviations from 1/K.
- **Do build the 20%-trimmed mean.** Zero parameters, small consistent gains,
  and real protection when a provider misbehaves
  ([Jose & Winkler 2008](https://doi.org/10.1016/j.ijforecast.2007.06.001)) —
  the best effort/benefit ratio in the whole combination axis, and a better
  degraded-mode default than plain equal weight.
- **Cluster providers by error correlation and keep one per cluster** (subset
  selection beats fancy weighting under near-duplicates —
  [Radchenko, Vasnev & Wang](https://doi.org/10.2139/ssrn.3647603)); this also
  raises the effective ensemble size behind any spread estimate.
- **Skip BMA** (weights degenerate on near-duplicate members). Skip full-cov
  GLS until years of archive exist.

---

## 8. Cross-reference with docs/improvement-plan.md

| Plan item | Research verdict |
|---|---|
| §1 harmonic/solar grounding (P0) | **Confirmed but reframed**: the adaptive EWMA hour-bin correction (§1a here) is the operationally proven fix and needs no window/phase model; SAMOS (§1c) is the principled batch version of the harmonic idea (fit climatology from provider/ERA5 archive, not station obs). Register both; expect EWMA to win on a young archive. |
| §2a median/L1 objective | Confirmed (MAE-consistent; also NBM uses MAE-tracked weights). |
| §2b bounds-in-scoring | Confirmed, unchanged. |
| §2c empirical-Bayes shrinkage of thin buckets | Confirmed; EWMA (§1a) largely obsoletes the cliff for grounding specifically. |
| §2d Ledoit–Wolf inverse-covariance | Keep as honest test; literature predicts it loses at this sample size (§7). |
| §3 anchoring P0s | Confirmed and extended: fit τ from residual ACF (no new data needed), persist-then-ramp (INCA), damped-trend state (Holt), LAMP-style fitted per-lead weights; drop the 3 h anchor cap for temperature. |
| §4 conformal = split + ACI | **Superseded**: use SAOCP or conformal-PID (temperature-benchmarked) with conditional-guarantee stratification; plain ACI's learning rate is a known failure knob. |
| §4 EMOS P1 | Promote to P0-adjacent once ensemble ingestion (§2a) exists — real spread makes the variance link meaningful. |
| §5 close verification loop, scheduled retrain | Confirmed; add delayed-feedback online updates (poold-style) and the two-tier drift detector. |
| §5 BH-FDR, held-out fold | **Upgraded**: sequential Model Confidence Sets are the modern, anytime-valid replacement; BH+HLN as stopgap. |
| §6 DM autocorrelation, leaderboard intersection | Confirmed, unchanged. |
| §7 solar-geometry features | Confirmed; also feeds the radiation-shield error model (§5.1 here). |
| (absent) station-QC physics + neighbors | **New**: §5 here — protects everything else. |
| (absent) ensembles / ML models / NBM / backfill | **New**: §2 here — the highest-leverage system change. |

---

## 9. Sequenced roadmap

Ordered by (impact ÷ effort), respecting data availability:

| # | Action | Effort | Needs | Expected effect |
|---|---|---|---|---|
| 1 | Ingest Open-Meteo Ensemble API (WeatherNext 2, AIFS-ENS, AIGEFS, GEFS) + NBM bulletins into the archive | M | none | Diversity + honest spread + the benchmark to beat; every later probabilistic method depends on it |
| 2 | EWMA hour-binned grounding (§1a) registered beside current methods | S–M | 2–8 wk archive | Directly targets the measured grounding loss at 24–96 h |
| 3 | Fit anchor τ from residual ACF; single anchoring pass; drop 3 h cap; persist-then-ramp | S | existing data | Trustworthy nowcast mechanics before evaluation exists |
| 4 | dynamical.org backfill (GEFS/AIFS-ENS full cycles) → sub-24 h backtest folds | M | none | Unblocks short-lead + ensemble evaluation now |
| 5 | Trimmed-mean blender + weatherapi correlation investigation | S | none | Robustness for free; possible diversity gold or bug fix |
| 6 | Truth-QC layer: shield S/(1+u) model, Synoptic+titanlib neighbor cron, SPICE catch efficiency | M | 30–90 d obs | Protects all fitting from sensor drift |
| 7 | EMOS + IDR benchmark + SAOCP/PID conformal wrapper; wire CRPS/PIT/coverage into the leaderboard | M | 1–2 mo + #1 | First calibrated distributions with guaranteed coverage |
| 8 | Delayed-feedback online experts (poold pattern) + two-tier drift detection | M | live archive | True self-improvement between serves; days-not-months swap response |
| 9 | Sequential-MCS promotion gate + HLN/BH stopgap + winner's-curse-corrected reporting | M | scores | Honest promotion; kills leaderboard churn |
| 10 | Damped-trend anchoring, precip onset calibration, LAMP-style fitted lead weights | M | 2–3 mo live | The station's differentiator, measured |
| 11 | CRPS learning (quantile-wise BOA); analog ensemble; SAMOS climatologies | L | 9–12 mo | The unified probabilistic blender, once the archive can feed it |

The polling cron remains the binding constraint (`docs/limitations.md` §1) —
items 1 and 4 are the two actions that make the wait productive rather than
idle.
