# Calibration and distributional heads

A point forecast answers "what will the temperature be?". A *calibrated
distributional* forecast answers "what will it be, and how sure are you?" — and
the second question is the one a proper scoring rule can actually adjudicate.

This page covers the methods that emit or repair full predictive distributions.
[Uncertainty quantification](uncertainty.md) covers the conformal machinery that
attaches finite-sample coverage guarantees to them; [verification](verification.md)
covers how they are scored.

Ten registered methods emit quantiles: `emos`, `csgd_emos`, `idr`, `idr_bucket`,
`idr_bucket_dcp`, `conformal_gew`, `conformal_ewma`, `analog_ensemble`,
`daily_marginal_emos`, `daily_path_extreme`.

---

## 1. EMOS / nonhomogeneous Gaussian regression

*Implemented in: `blenders/emos.py::Emos` — `emos`*

Gneiting, Raftery, Westveld & Goldman (2005). The predictive distribution is
Gaussian with **both** parameters regressed on the ensemble:

$$\mu = a + b\,\hat{y}_{\text{base}},
\qquad
\sigma = \max\!\bigl(\exp(c + d\log s),\ 10^{-3}\bigr)$$

where $s$ is the spread predictor. The log-link on $\sigma$ is what makes this
*nonhomogeneous*: the forecast is allowed to be less certain on days the ensemble
disagrees, which a fixed-variance model cannot express.

### The spread predictor, and an honest caveat

`_spread` uses the mean of real ensemble standard deviations
(`ens__{model}__{var}__sd`) where the Open-Meteo Ensemble API has been ingested,
and otherwise falls back to the cross-provider `nanstd`.

The fallback is **structurally under-dispersed** and the code says so. Providers
share parents: eight APIs repackaging three NWP models disagree far less than
three independent models would, so cross-provider spread understates true
uncertainty. The fitted $d$ therefore tends to be small and the model leans on
its intercept $c$ — which is a *working* calibration, just not one where the
spread term is carrying the information it appears to.

### Fitting

=== "Unbounded variables"

    Minimize the weighted **closed-form Gaussian CRPS**:

    $$\text{CRPS}(y; \mu, \sigma) = \sigma\left[z\bigl(2\Phi(z) - 1\bigr) + 2\varphi(z) - \frac{1}{\sqrt{\pi}}\right],
    \qquad z = \frac{y - \mu}{\sigma}$$

    A proper scoring rule minimized directly — no likelihood assumption beyond
    the Gaussian family itself.

=== "Bounded variables"

    Maximize the weighted **truncated-normal log-likelihood**
    (`scipy.stats.truncnorm`). Humidity cannot exceed 100%, and a Gaussian that
    puts mass at 105% is scored as if that were possible.

    With an explicit **Gaussian-CRPS fallback**: a single observation outside the
    declared support makes the truncated likelihood $-\infty$ everywhere, and no
    optimizer recovers from that. `to_state()` therefore reports `fit_family` and
    `serving_family` *separately*, so a fallback is visible in the artifact rather
    than inferred from a suspicious coefficient.

Optimizer: Nelder–Mead, `maxiter=400`, `xatol=1e-4`, `fatol=1e-6`, with a
pre-check that the initial loss is finite — an all-infinite simplex can never
converge, and failing fast beats 400 wasted iterations.

### Temporal decay

Training rows are weighted by recency:

$$w_t = \frac{\exp(-\Delta_t / 45)}{\sum_u \exp(-\Delta_u / 45)},
\qquad \Delta_t = \text{age in days}$$

`_RECENCY_SCALE_DAYS = 45.0`, an exponential half-life of
$45\ln 2 \approx 31.2$ days. Recent weather is more informative about tomorrow
than last spring's is, and a seasonal regime change should not be averaged
against.

Constants: `_MIN_FIT_ROWS = 60`, `_MIN_SIGMA = 1e-3`, 19 quantile levels
$(0.05, 0.10, \dots, 0.95)$.
`fit_status ∈ {unfitted, insufficient_rows, converged, gaussian_fallback}`.

---

## 2. CSGD — censored shifted gamma

*Implemented in: `blenders/csgd.py::CsgdEmos` — `csgd_emos`, scoped to `precip_mm` / `precip_sum_mm`*

Scheuerer & Hamill (2015). Precipitation is not Gaussian and not continuous: it
has an atom at zero (it usually does not rain) and a long right tail (when it
does, sometimes a lot). A censored shifted gamma handles both in one object —
see [precipitation](precipitation.md) for why this matters more than it sounds.

$$\mu = a + b\,\hat{y}_{\text{base}},
\qquad
\sigma = \exp(c + d\log s),
\qquad
\delta < 0$$

$$\text{shape } \kappa = \left(\frac{\mu - \delta}{\sigma}\right)^{2},
\qquad
\text{scale } \theta = \frac{\sigma^2}{\mu - \delta}$$

with $(\mu - \delta)$ floored at $10^{-3}$. The shift $\delta$ is constrained
negative — a non-negative shift is hard-rejected with $+\infty$ — because the
whole construction depends on the distribution's support starting *below* zero so
that censoring at zero produces a genuine probability atom.

Negative log-likelihood:

$$-\log L = -\sum_t w_t \begin{cases}
\log f_\Gamma(y_t - \delta;\ \kappa, \theta), & y_t > 10^{-9} \quad \text{(wet)}\\[4pt]
\log F_\Gamma(-\delta;\ \kappa, \theta), & y_t \le 10^{-9} \quad \text{(dry)}
\end{cases}$$

The second line is the point of the method: **the censored mass *is* the
probability of a dry hour**. A single fitted object gives both PoP and the
rainfall amount distribution, consistently, rather than bolting a classifier onto
a regressor.

Quantiles: $q_\tau = \max\bigl(\delta + F_\Gamma^{-1}(\tau;\ \kappa, \theta),\ 0\bigr)$.

Constants: `_MIN_FIT_ROWS = 60`, `_MIN_WET_ROWS = 10` (below which `fit_status =
"insufficient_wet_rows"` and the head abstains to the base point),
`_MIN_SHAPE_DISTANCE = 1e-3`, `_WET_EPS = 1e-9`, Nelder–Mead `maxiter = 1500`
for the five parameters.

The wet-row floor is not a formality on this archive. A dry Southern California
summer produces windows with two or three wet hours, and a five-parameter
distribution fitted to three wet observations is a random number generator.

---

## 3. Isotonic distributional regression

*Implemented in: `blenders/idr.py` — `idr`, `idr_bucket`, `idr_bucket_dcp`*

Henzi, Ziegel & Gneiting (2021); the EasyUQ construction. IDR is the
**nonparametric** answer to EMOS: instead of assuming a family and fitting its
parameters, assume only that the predictive distribution is *stochastically
increasing* in the covariate — a larger forecast implies a stochastically larger
outcome — and let the data supply everything else.

That single shape assumption is enough to make the estimator well-defined and
tuning-free.

### The estimator

`pava_isotonic(values, weights)` is a weighted pool-adjacent-violators L2
isotonic regression, implemented from scratch.

`fit_idr_state(x, y)`: for each threshold $z$ on a grid of 49 outcome quantiles
(`_THRESHOLD_LEVELS = np.linspace(0.02, 0.98, 49)`), fit the exceedance indicator
$\mathbb{1}\{y > z\}$ isotonically in $x$ (equal covariates pooled first via
`np.add.reduceat`), and take

$$\hat{F}(z \mid x) = 1 - \text{PAVA}\bigl(\mathbb{1}\{y > z\}\bigr)$$

then enforce monotonicity *across* thresholds with
`np.maximum.accumulate(axis=1)`. Isotonicity in $x$ is guaranteed by PAVA;
isotonicity in $z$ — i.e. that $\hat{F}(\cdot \mid x)$ is a valid CDF — is not,
and has to be imposed. `_MIN_FIT_ROWS = 100`.

### The three variants

| `method_id` | What it adds |
|---|---|
| `idr` | global fit — the control arm |
| `idr_bucket` | per-lead-bucket fit with **subagging** |
| `idr_bucket_dcp` | subagged, plus split distributional conformal calibration |

**Subagging** (`subagged_idr_state`): 50 halves drawn without replacement, CDFs
averaged on the full sample's covariate grid and thresholds. Linear aggregation
preserves isotonicity, so the average is still a valid IDR fit; the RNG is seeded
by $n$ for reproducibility. This smooths the step CDF that a single PAVA fit
produces on a few hundred rows.

The bucket→global fallback is a **cliff, not a blend** — unlike everywhere else
in the codebase. Two step CDFs defined on different covariate grids have no
principled linear interpolation, so `PerBucketFitter` is used in its
all-or-nothing mode.

`idr_bucket_dcp` adds a marginal coverage guarantee; it is described under
[uncertainty §2](uncertainty.md#2-distributional-conformal-prediction) because
the guarantee, not the fit, is its contribution.

---

## 4. Daily marginal EMOS

*Implemented in: `blenders/daily_heads.py::DailyMarginalEmos` — `daily_marginal_emos`*

Meng & Taylor (2022): calibrate the daily extreme *directly* rather than deriving
it from a calibrated hourly path.

$$\mu = a + b\,\hat{y}_{\text{base}} + e\cdot\text{path}$$

where $\text{path}$ is the cross-source mean of per-source hourly-path extremes
(`path__{src}__max/min`), base-filled where absent. Five parameters, CRPS-fitted,
`maxiter=1500`. Scoped to `temp_max_c` / `temp_min_c`.

The `path` term is what makes this more than a relabelled EMOS: it lets the model
learn that providers' *stated* daily highs are systematically low relative to
what their own hourly paths imply for this thermometer.

---

## 5. Probability calibration for PoP

*Implemented in: `blenders/pop_calibration.py::PopCalibrator` — `pop_platt`, `pop_beta`*

Two mutually exclusive maps, both registered, both scoped to `pop`, and both
deliberately **point-only** — quantiles of a Bernoulli parameter are not a
meaningful object.

$$
\begin{aligned}
\texttt{pop\_platt:}\quad & \hat{p} = \operatorname{sigmoid}\bigl(a + b\operatorname{logit} p\bigr)\\
\texttt{pop\_beta:}\quad & \hat{p} = \operatorname{sigmoid}\bigl(a + b\log p + c\,(-\log(1-p))\bigr)
\end{aligned}
$$

Platt scaling is a two-parameter logistic recalibration; beta calibration
(Kull, Silva Filho & Flach 2017) adds a third parameter and can express
asymmetric distortions — the common case where a forecaster is well-calibrated at
low probabilities and overconfident at high ones, which Platt's single slope
cannot represent.

Both families contain the identity: Platt at $(0, 1)$, beta at $(0, 1, 1)$, since
$\operatorname{sigmoid}(\log p - \log(1-p)) = p$. So a well-calibrated input is a
fixed point rather than something the fit has to rediscover.

Fitted per lead bucket with the shared EB shrinkage blend.

### The optimizer, and why it is not plain Newton

`_fit_logistic` is **damped Newton on ridge log-loss with backtracking line
search** (≤ 20 halvings). The code records the reason: undamped Newton oscillates
to $\pm 10^5$ at realistic bucket sizes. IRLS weights are floored at $10^{-6}$
and a ridge $\rho = 10^{-4}$ is added to *both* gradient and Hessian.

Guards worth knowing, because a dry archive triggers all of them:

| Guard | Value | Purpose |
|---|---|---|
| `_EPS` | $10^{-4}$ | probability clip before the logit |
| logit clip | $[-30, 30]$ | |
| `_MAX_COEF` | 8.0 | separation guard |
| `_MIN_EACH_CLASS` | 5 | **a one-class window returns the identity map** |
| `_NEWTON_STEPS` | 25 | |
| `_WET_THRESHOLD` | 0.5 | |

`_MIN_EACH_CLASS` is the critical one. On a window where it never rained, perfect
separation is available and the unconstrained MLE runs to infinity — the "optimal"
calibration of a dry month is "always predict 0". Returning the identity map
instead is the difference between a calibrator and a machine for forgetting that
rain exists.

---

## 6. Post-hoc quantile recalibration

*Implemented in: `reports/recalibration.py` (fit and A/B), `serve/recalibration.py` (apply)*

Two mutually exclusive repairs, fitted and scored **from stored scores alone** —
no backtest re-run — so the A/B is cheap enough to run every night.

**PIT remapping** (`fit_pit_levels` / `apply_pit_levels`, Kuleshov et al. 2018).
Take the empirical distribution of PIT values, and read the native quantile grid
at the remapped level:

$$\tau' = \widehat{Q}_{\text{PIT}}(\tau),
\qquad
q'_\tau = \text{interp}\bigl(\tau';\ \text{native grid}\bigr)$$

then re-sort. Because it only *reads* the existing grid, it **cannot escape the
outermost native levels** — it repairs distributional shape but not severe
under-dispersion.

**CQR margins** (`fit_cqr_margins` / `apply_cqr_margins`, a variant of Romano,
Patterson & Candès 2019). Per-level additive margin from the split-conformal
rank:

$$m_\tau = \widehat{Q}_\tau\bigl(y - q_\tau\bigr),
\qquad
q'_\tau = q_\tau + m_\tau$$

then re-sort. This *can* widen past the native grid, which is exactly what
under-dispersion needs.

!!! note "This is a variant, not the published score"
    Published CQR uses the symmetric conformity score
    $\max(q_{\text{lo}} - y,\ y - q_{\text{hi}})$ for a two-sided interval. The
    per-level additive margin here is a related but distinct construction; it
    does not inherit CQR's exact coverage theorem. Treated as an empirical repair
    judged by the A/B, not as a guarantee.

### Leakage discipline

Both are fitted under a strict chronological split on unique `valid_time`,
`_FIT_FRACTION = 0.7`, `MIN_FIT_ROWS = 50`, `_MIN_EVAL_ROWS = 20`. The
variable-global fallback pool still strictly precedes the bucket's evaluation
era. Only the **newest evaluation** is used, because expanding folds re-score
overlapping cases and pooling them would count the same case repeatedly.

At serve time `QuantileRecalibrator` applies the configured mode, routed **per
product** — this deployment runs `hourly="none"`, `daily="cqr"`,
`minutely="none"`. Dressed rows are skipped: they are already split-conformal, and
correcting twice double-counts the residual archive. Provenance guards restrict
the fit to `source_kind="live"` rows matching the current `dataset_fingerprint`,
`config_fingerprint`, and `code_version`; the first IO or schema failure disables
the transform for the whole run rather than applying it to some rows.

`reports/evidence.py::recalibration_verdicts` scores the three-way A/B: per cell
the winner minimizes $|\text{coverage}_{80} - 0.8|$ with pinball loss as
tiebreak, reported as `recalib_win_share_{raw,pit,cqr}`.

---

## 7. Quantile dressing

*Implemented in: `serve/dressing.py::corrected_error_quantiles`*

Most registered methods are point forecasters. Dressing gives them a
distribution at serve time from the empirical distribution of their own residuals
$e = y_{\text{true}} - \hat{y}$, using finite-sample split-conformal ranks:

$$\text{rank}(\tau) = \begin{cases}
\min\bigl(n,\ \lceil (n+1)\tau \rceil\bigr), & \tau \ge 0.5\\[4pt]
\max\bigl(1,\ \lfloor (n+1)\tau \rfloor\bigr), & \tau < 0.5
\end{cases}$$

The two branches make the interval **deliberately asymmetric**. A symmetric band
centred on the point forecast would hide a systematic bias inside itself; an
asymmetric one built from signed residuals puts the bias where a reader can see
it. `DRESSING_LEVELS = (0.05, 0.1, 0.25, 0.75, 0.9, 0.95)`, `MIN_POOL_ROWS = 24`,
with a bucket pool → variable-global pool → undressed fallback ladder and the
same provenance guards as recalibration.
