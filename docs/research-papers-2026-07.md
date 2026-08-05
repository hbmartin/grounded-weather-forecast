# Papers research, July 2026

_Literature sweep of 2026-07-26/27, targeted at what the first live week
measured (see `research/evaluation-2026-07-26.md`). The 2026-07-18 synthesis
(`research/improvement-methods-2026-07.md`) remains the canonical bibliography
for the methods already implemented; everything below is **new** relative to
it. Sourcing note: the paper-search MCP arXiv/Semantic Scholar endpoints
returned empty results during this sweep, so findings were located via
OpenAlex, Crossref, and web search, with every arXiv/DOI reference verified
against its abstract._

Each entry states: what the paper does, why it fits this project's constraints
(one station, thin live archive, CPU, interpretability), where it would land
in this codebase, and what would count as a win on the leaderboard.

---

## 1. Short-lead anchoring — closing the measured 0–1 h gap

The first live week measured raw `persistence` at 1.129 °C MAE for temp 0–1 h
against 1.46–1.48 for every anchored method and 1.699 for served
`equal_weight`. The anchor family under-uses the observation as lead → 0.
Two papers give the principled fix.

### 1.1 RAFT — Rapid Adjustment of Forecast Trajectories

- Schuhen, Thorarinsdottir & Lenkoski (2020), *Rapid adjustment and
  post-processing of temperature forecast trajectories*, QJRMS 146,
  [doi:10.1002/qj.3718](https://doi.org/10.1002/qj.3718)
  ([arXiv:1910.05101](https://arxiv.org/abs/1910.05101)).
- Companion: Schuhen (2020), *Order of operation for multi-stage
  post-processing of forecast trajectories*, NPG 27,
  [doi:10.5194/npg-27-35-2020](https://doi.org/10.5194/npg-27-35-2020).

**Method.** When the observation for hour *t* verifies, update the *remaining*
hours *t+1 … t+H* of the already-issued trajectory using the estimated
correlation of forecast errors between lead times: for each pair (observed
lead, future lead), a simple linear regression of the future error on the
just-observed error. The companion paper shows the optimal composition is
univariate calibration first, then RAFT, then dependence restoration (ECC),
and that a RAFT-adjusted *stale* forecast can beat the *fresh* NWP run over
its first few hours.

**Fit here.** This is exactly the `Anchored*` family's job, done with the
error-*correlation structure across leads* instead of a single exponential-
decay assumption. `AnchoredEmpirical` (`blenders/anchoring.py`) already fits
per-lead-bin regression weights on the issue-time residual — RAFT generalizes
it from "regress on the residual at lead ≈ 0" to "regress on the most recent
*verified* error", which keeps updating between issue times. Parameters: one
slope per (observed-lead, target-lead) pair, poolable across hours with
shrinkage; weeks of data suffice.

**Implementation sketch.** New method `raft_grounded` beside the anchored
family: fit lead-pair error regressions per variable on the training slice
(day/night split when rows allow); at predict, use the latest resolved truth
row as the conditioning error. The minutely product then interpolates the
RAFT-corrected hourly path — no separate minutely τ constant at all.

**Win condition.** MAE at 0–1 h within noise of persistence, and better than
persistence at 1–6 h, on the live leaderboard.

### 1.2 Seamless multimodel postprocessing with the latest observation

- Dabernig & Atencia (2024), *Seamless multi-model postprocessing for air
  temperature forecasts in complex topography*,
  [arXiv:2410.11916](https://arxiv.org/abs/2410.11916) (GeoSphere Austria).

**Method.** Ordinary multimodel linear regression with two structural tricks:
(a) when a model's horizon ends, its last available lead is carried forward as
an extra "model persistence" predictor, so every source contributes at every
lead and there is no skill cliff where a short-horizon model drops out; (b)
the latest station observation enters as one more predictor, whose fitted
coefficient decays naturally with lead — the forecast "closes seamlessly"
onto live measurements without a separate anchoring stage.

**Fit here.** This is nearly a blueprint of the whole pipeline in one small
regression: 10 sources with ragged horizons (the sleeping-experts problem) and
a station-obs anchor, in complex topography, per variable × lead. It is the
strongest *challenger architecture* to the current grounding→blending→
anchoring decomposition, at a few coefficients per lead bucket.

**Implementation sketch.** Register `seamless_regression`: design matrix =
grounded source columns, with expired horizons forward-filled and flagged, plus
`obs__{var}` at issue time; ridge-regularized per lead bucket, availability-
renormalized. The three-stage decomposition stays; the leaderboard arbitrates.

**Win condition.** Beats `boa`/`ewa` at mid leads or the anchored family at
short leads without losing the long-lead buckets.

---

## 2. Self-updating fits

### 2.1 ondil — online distributional regression

- Hirsch, Berrisch & Ziel (2024, revised 2026), *Online Distributional
  Regression*, [arXiv:2407.08750](https://arxiv.org/abs/2407.08750). Python:
  [`ondil`](https://github.com/simon-hirsch/ondil) (CPU, incremental).

**Method.** GAMLSS-type distributional regression (location, scale, shape as
linear functions of covariates) estimated *incrementally*: each new sample
triggers an O(p²) coefficient update with LASSO regularization and forgetting
factors — no batch refits, drift handled by exponential downweighting.

**Fit here.** One engine that could subsume three current components: the
EWMA intercepts (location), the EMOS refit (scale), and the 45-day recency
window inside `emos.py` (forgetting factor). Per (variable × lead bucket),
state is a small coefficient vector updated when truth resolves — the same
delayed-feedback cursor machinery the experts already use
(`experts.py:188-226`).

**Caution.** It replaces interpretable named stages with one model; register
it as a method (`ondil_blend`), never as a rewrite of grounding/EMOS, so the
leaderboard can reject it.

### 2.2 Training-window design for a mixed synthetic/live archive

- Lang, Lerch, Mayr, Simon, Stauffer & Zeileis (2020), *Remember the past:
  a comparison of time-adaptive training schemes for non-homogeneous
  regression*, NPG 27,
  [doi:10.5194/npg-27-23-2020](https://doi.org/10.5194/npg-27-23-2020).

**Method.** At mountain stations, compares sliding windows vs seasonally
weighted all-history vs smooth seasonal-coefficient schemes for per-station
EMOS. Weighted all-history wins; short sliding windows are the worst option
for seasonally varying error — *even across NWP model upgrades*, because
variance reduction beats bias avoidance.

**Fit here.** Two years of synthetic backfill (now extended to 2024) vs weeks
of live data is exactly this trade. The live/synthetic never-pooled rule is
right for *scoring*; for *fitting* it discards most usable information. A
deliberate, flagged exception — fit on synthetic ∪ live with per-row weights
(season-distance × source-kind discount), score on live only — is the
literature-backed design. Needs an ADR because it touches a load-bearing
invariant.

### 2.3 Weighted conformal under drift

- Barber, Candès, Ramdas & Tibshirani (2023), *Conformal prediction beyond
  exchangeability*, Ann. Statist. 51(2),
  [arXiv:2202.13415](https://arxiv.org/abs/2202.13415).

**Method.** Conformal with fixed data weights retains coverage up to an
explicit penalty proportional to the total-variation distance the weighting
introduces; discounting stale or differently-distributed calibration points
costs a small, *quantified* slice of coverage.

**Fit here.** The theory that lets `conformal.py` calibrate on synthetic
residuals (down-weighted) while the live calibration set is thin, instead of
refusing to use them. Composable with the existing conformal-PID cells.

---

## 3. Wet-season probability calibration

Truth recorded an 8.6 mm/h monsoon cell with wet-hour PoP = 1.0 while no
provider exceeded 0.3 — and PoP currently serves `climatology` (≈ 0). Before
the first winter storms, the PoP path needs a calibrator that learns from a
single tipping bucket.

### 3.1 Online Platt scaling with calibeating

- Gupta & Ramdas (2023), *Online Platt scaling with calibeating*, ICML 2023,
  PMLR 202, [arXiv:2305.00070](https://arxiv.org/abs/2305.00070).

Two-parameter logistic recalibration fitted by online Newton step, hedged so
calibration holds even under adversarial/drifting sequences, no tuning. The
default choice for recalibrating blended provider PoP against wet/dry hours:
state is two floats per lead bucket, updated as truth resolves.

### 3.2 Beta calibration

- Kull, Silva Filho & Flach (2017), *Beta calibration: a well-founded and
  easily implemented improvement on logistic calibration for binary
  classifiers*, AISTATS 2017, PMLR 54.

Three parameters on log-odds of the raw PoP; fixes logistic's characteristic
mis-shape when the input score is itself already a probability (bounded,
skewed near 0) — precisely what vendor PoP is. Batch, trivial to fit; the
natural A/B partner to §3.1.

### 3.3 Venn–Abers predictors

- Vovk & Petej (2014), *Venn–Abers predictors*, UAI 2014,
  [arXiv:1211.0025](https://arxiv.org/abs/1211.0025); generalization: van der
  Laan & Alaa (2025), [arXiv:2502.05676](https://arxiv.org/abs/2502.05676).

Two isotonic fits yield distribution-free validity with ~50–100 samples — the
guarantee-carrying benchmark once one wet season of outcomes exists (the IDR
of probability calibration: no parameters, provable validity).

---

## 4. The daily product

Daily Tmax serves unpromoted `equal_weight` at 3.4–3.9 °C MAE — the weakest
product per degree. Two complementary designs:

### 4.1 Joint probabilistic Tmin/Tmax

- Meng & Taylor (2022), *Comparing probabilistic forecasts of the daily
  minimum and maximum temperature*, Int. J. Forecasting 38(1),
  [doi:10.1016/j.ijforecast.2021.05.007](https://doi.org/10.1016/j.ijforecast.2021.05.007).

Calibrate the Tmax and Tmin marginals directly (EMOS-style, provider daily
values + blended-path summary as predictors), then restore the Tmin–Tmax
dependence with a copula (ensemble copula coupling suffices). Low-parameter,
closed-form, evaluated against direct alternatives — the design template for
`temp_max_c`/`temp_min_c` as first-class supervised targets.

### 4.2 Extreme-of-path via ECC / member-by-member

- Schefzik (2017), *Ensemble calibration with preserved correlations:
  unifying and comparing ensemble copula coupling and member-by-member
  postprocessing*, QJRMS 143,
  [doi:10.1002/qj.2984](https://doi.org/10.1002/qj.2984); MBM: Van
  Schaeybroeck & Vannitsem (2015), QJRMS 141,
  [doi:10.1002/qj.2397](https://doi.org/10.1002/qj.2397).

Calibrate *hourly* marginals, reorder samples by the raw multi-source
trajectory ranks (ECC), then take max/min over each reconstructed 24 h path:
a daily extreme forecast that inherits hourly calibration with **zero** extra
fitted parameters. No published head-to-head of this "extreme-of-path" route
vs direct daily regression at single-station scale was found — running both
on this leaderboard is a genuinely novel comparison.

---

## 5. Validation of the current architecture (context, not new work)

- **Kocsis & Baran (2026)**, *AI and physics-based weather forecasting: a
  comparative study*, [arXiv:2606.02508](https://arxiv.org/abs/2606.02508):
  per-station EMOS/QR on operational AIFS-ENS + IFS-ENS with only ~5 months of
  archive; parametric EMOS beats quantile regression at short leads.
  Postprocessing in exactly this project's data regime works.
- **Trotta et al. (2025)**, *Statistical post-processing yields accurate
  probabilistic forecasts from AI weather models*, BAMS,
  [arXiv:2504.12672](https://arxiv.org/abs/2504.12672): the Bureau of
  Meteorology ran deterministic AIFS through its operational IMPROVER chain
  unchanged; gains match NWP, and blending AI + conventional adds skill even
  where the AI model alone is worse. License to treat AI-backed feeds
  (WeatherNext via Open-Meteo, Google Weather API, AIGEFS) as ordinary
  members of the grounded blend.
- **Asch, Rossellini, Hassanzadeh & Willett (2026)**, *Online conformal
  prediction of AI ensemble forecasts*,
  [arXiv:2606.19642](https://arxiv.org/abs/2606.19642): GenCast/NeuralGCM/
  AIFS-ENS lose coverage on extremes; an online conformal wrapper restores it
  without hurting CRPS — direct support for keeping `conformal_*` wrapped
  around whatever wins, especially once `ens__*` features flow.

---

## 6. At archive maturity (9–12 months)

### 6.1 Bayesian predictive synthesis

- McAlinn & West (2019), *Dynamic Bayesian predictive synthesis in time
  series forecasting*, J. Econometrics 210,
  [doi:10.1016/j.jeconom.2018.11.010](https://doi.org/10.1016/j.jeconom.2018.11.010);
  multivariate: McAlinn, Aastveit, Nakajima & West (2020), JASA,
  [doi:10.1080/01621459.2019.1660171](https://doi.org/10.1080/01621459.2019.1660171).

Dynamic latent-factor synthesis of multiple predictive *densities* on one
series: learns each source's time-varying bias, miscalibration, and — the part
nothing currently registered does — inter-source *dependence*. BMA, QRA, and
linear pools are special cases. A small Kalman-filter state space, CPU-cheap.
The principled endgame for "10 sources, ρ ≈ 0.9, one series", once there is
enough archive to identify the latent states.

### 6.2 D-vine copula quantile regression

- Jobst, Möller & Groß (2023), *D-vine-copula-based postprocessing of
  wind-speed ensemble forecasts*, QJRMS 149,
  [doi:10.1002/qj.4521](https://doi.org/10.1002/qj.4521); GAM extension
  [arXiv:2309.05603](https://arxiv.org/abs/2309.05603) (R implementations).

Quantile regression built from D-vine copulas that *selects* among many
correlated predictors and captures nonlinear dependence — outperformed
gradient-boosted EMOS at stations with nonlinear error structure. The
dependence-aware alternative to QRA for the near-duplicate provider set; wind
(the worst-correlated variable here) is its published home turf. Python
implementation cost is the main barrier.

---

## 7. Cross-reference: measured failure → paper → artifact

| Measured failure (live week) | Paper | Would land as |
| --- | --- | --- |
| Persistence beats all anchored methods at 0–1 h | RAFT (§1.1) | `raft_grounded` method |
| Ragged horizons need sleeping experts; obs anchoring is a separate stage | Dabernig & Atencia (§1.2) | `seamless_regression` method |
| EMOS refits batch per serve; EWMA and EMOS keep separate recency logic | ondil (§2.1) | `ondil_blend` method |
| 2 y synthetic unusable for live fits under never-pool rule | Lang et al. (§2.2) | weighted-fit ADR + fit-time source-kind weights |
| Conformal calibration sets too thin per cell | Barber et al. (§2.3) | weighted conformal in `conformal.py` |
| PoP serves climatology; monsoon bust missed | Gupta & Ramdas, Kull, Vovk (§3) | `pop_platt` / `pop_beta` / Venn–Abers calibrators |
| Daily Tmax 3.4–3.9 °C, unpromoted | Meng & Taylor, Schefzik (§4) | daily marginal EMOS + ECC extreme-of-path |
| Provider redundancy ρ up to 0.97 | McAlinn & West, Jobst (§6) | archive-maturity blenders |
