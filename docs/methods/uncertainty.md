# Uncertainty quantification

[Calibration](calibration.md) covers methods that *emit* a predictive
distribution. This page covers the machinery that attaches **finite-sample
coverage guarantees** to intervals, measures whether spread tracks skill, and
keeps a multivariate forecast internally consistent.

The distinction matters. An EMOS fit is calibrated if the Gaussian assumption
holds and the parameters are right. A conformal interval is calibrated if the
data are exchangeable — regardless of whether the underlying model is any good.
The second is a much weaker requirement, which is the whole appeal.

---

## 1. Adaptive conformal prediction (conformal PID)

*Implemented in: `blenders/conformal.py::Conformal` — `conformal_gew`, `conformal_ewma`*

Split conformal prediction gives exact marginal coverage under exchangeability.
Weather is emphatically **not** exchangeable — it has trend, seasonality, and
regime changes — so the guarantee is the wrong shape for this problem. Adaptive
conformal (Gibbs & Candès; Angelopoulos, Candès & Tibshirani 2023) replaces the
static quantile with an online controller that drives *realized* coverage toward
nominal, and gets a long-run coverage property that survives distribution shift.

State is kept per cell = (lead bucket × day/night). For target coverage $c$:

$$
\textbf{P term:}\quad
\rho_c \leftarrow \max\!\bigl(0,\ \rho_c + \gamma\,(c - \mathbb{1}\{S \le r_{\text{eff}}\})\bigr),
\qquad \gamma = 0.1 \cdot \max(\text{recent scores})
$$

$$
\textbf{I term:}\quad
E_c \mathrel{+}= c - \mathbb{1}\{\text{covered}\}
$$

$$
r_{\text{eff}}(c) = \max\!\left(0,\ \rho_c + \text{scale} \cdot
\tan\!\left(\operatorname{clip}\!\left(\frac{E_c \ln t}{t\,C_{\text{sat}}},\ \pm 1.2\right)\right)\right)
$$

Every miss pushes the radius up by $\gamma(1 - c)$ and every hit pulls it down by
$\gamma c$, so at equilibrium the miss rate is $1 - c$. That is the proportional
term, and alone it is enough for the asymptotic guarantee — but it wanders,
because it has no memory of *accumulated* debt.

The integral term supplies that memory, and its form is the interesting part. A
plain integrator winds up without bound during a persistent-miss run and then
overshoots for just as long. Here the accumulated error is passed through a
**saturating tangent** with authority decaying like $\ln t / t$:

| Constant | Value | Effect |
|---|---|---|
| `_STEP` $\gamma$ | 0.1 × recent score scale | proportional gain |
| `_CSAT` | 0.5 | integrator saturation |
| `_ARG_CLIP` | 1.2 | $\tan(1.2) \approx 2.57$ — the integrator can never add more than ≈2.57× the score scale |
| `_SCALE_WINDOW` | 50 | rolling window for the score scale |
| `_MIN_UPDATES` | 20 | warm start threshold |

So the controller can respond hard to a genuine regime change and still cannot
run away.

**Warm start.** Below $t = 20$ the radius is the conservative split-conformal
order statistic $S_{(\lceil (n+1)c \rceil)}$ — i.e. it begins with the classical
guarantee and only then hands over to the controller.

**Splitting.** Chronological by unique issue time, `_PROPER_FRACTION = 0.7`. Proper
(fitting) rows additionally require `resolution ≤ cutoff`: the base model is fitted
only on rows whose truth was *known* by the cutoff, not merely issued before it.
This is the same `truth_known_at` discipline as the backtest folds — a
calibration set contaminated by future truth produces a guarantee about nothing.

**Degradation.** `_MIN_PROPER_ROWS = 60`, `_MIN_CALIBRATION_ROWS = 20`; below
either, the method degrades to `strategy="point_only"` with a recorded reason.
Every calibration score also feeds a `__global__` cell per day/night flag, giving
the fallback ladder

$$(\text{bucket}, \text{phase}) \to (\text{bucket}, \neg\text{phase}) \to
(\text{global}, \text{phase}) \to (\text{global}, \neg\text{phase})$$

Coverage targets $(0.5, 0.8, 0.9)$ produce levels
$(0.05, 0.1, 0.25, 0.75, 0.9, 0.95)$.

!!! note "Intervals here are symmetric by construction"
    The score is $|y - \hat{y}|$, so the interval is $\hat{y} \pm r_{\text{eff}}$.
    Asymmetry is left to the EMOS, IDR, and
    [dressing](calibration.md#7-quantile-dressing) paths, which model the signed
    residual.

---

## 2. Distributional conformal prediction

*Implemented in: `blenders/idr.py::dcp_adjusted_levels`, `IdrBucketDcp` — `idr_bucket_dcp`*

Chernozhukov, Wüthrich & Zhu (2021). This is **the only method in the registry
carrying a marginal coverage guarantee that holds regardless of whether the
underlying model is correctly specified.**

The construction is short. Take the IDR fit's PIT values on a held-out
calibration split; the adjusted level is the corresponding order statistic:

$$\tau' = \text{PIT}_{\left(\lceil (n+1)\tau \rceil\right)},
\qquad \tau' \in [10^{-6},\ 1 - 10^{-6}]$$

Then read the IDR CDF at $\tau'$ instead of $\tau$. If the IDR fit is perfect the
PITs are uniform and $\tau' \approx \tau$; if it is miscalibrated, the order
statistic absorbs exactly the amount of miscalibration present, and marginal
coverage holds by the standard exchangeability argument on the calibration
sample.

The calibration split is chronological and taken from the **tail**:
`_CALIBRATION_FRACTION = 0.25`, `_MIN_CALIBRATION_ROWS = 25`. Taking it from the
tail rather than at random is the concession to non-exchangeability — the
calibration sample is the most recent data, so the guarantee is anchored to the
regime being served.

---

## 3. Spread, and spread–skill

*Implemented in: `blenders/emos.py::_spread`, `dataset/ensembles.py`*

A well-dispersed probabilistic forecast has a **spread–skill relationship**: on
days the ensemble disagrees, the error should be larger. Ensemble spread is used
here as a predictor of $\sigma$ in EMOS, CSGD, and as a `gbm` feature.

Real ensemble statistics come from the Open-Meteo Ensemble API, reduced to
$(\text{mean},\ \text{sd}\ (\text{ddof}=1),\ p_{10}, p_{25}, p_{50}, p_{75}, p_{90},\ n_{\text{members}})$
per (model, valid time, variable) and joined as-of into `ens__*` columns.

Two design decisions there are load-bearing:

- **Ensemble statistics are features, not sources.** Adding 51 members as 51
  sources would make $k_{\text{eff}}$ look enormous while adding almost no
  independent information. See
  [combination §2](combination.md#2-the-diversification-ceiling).
- **Backfilled vintages carry `mean` and `sd` only; percentiles stay null rather
  than being faked from a normal assumption.** A synthesized $p_{10}$ would be
  indistinguishable downstream from a measured one, and would quietly assert
  Gaussianity for a variable like precipitation where it is badly wrong.

Where no ensemble has been ingested, the fallback is cross-provider `nanstd`,
which is structurally under-dispersed — see
[calibration §1](calibration.md#the-spread-predictor-and-an-honest-caveat).

Spread–skill is **measured indirectly** rather than reported as a single
correlation: `coverage80` / `coverage90` against nominal, `sharpness`, and
`pit_chi2_p`. `reports/evidence.py::_recent_quantile_metrics` adds
`recent_coverage80/90` and `recent_crps` over a 14-day window
(`RECENT_WINDOW_DAYS = 14`, `_MIN_RECENT_QUANTILE_ROWS = 20`) — necessary because
pooled expanding-evaluation coverage is frozen by history and moves only tenths
of a point per day, so a calibration failure that started last week is invisible
in the pooled number.

---

## 4. Cross-variable quantile coherence

*Implemented in: `serve/predict.py::_cohere_pair`, `_enforce_mapped_pair`, `_keep_points_inside_quantiles`*

A forecast document that says the daily minimum will be 14 °C and the daily
maximum 12 °C is wrong in a way no scoring rule will catch, because each variable
is scored marginally. Coherence must be enforced structurally.

The constraints are the physical ones: $\text{temp\_min} \le \text{temp\_max}$,
$\text{dew\_point} \le \text{temp}$.

They are enforced **on the whole quantile curve, not just the point**. The
procedure: map both curves onto the union of their levels, project so the
ordering holds at every level, write back, then run a neighbour-projection pass
to restore between-knot coherence when the two source grids differ.
`np.maximum.accumulate` keeps each repaired curve monotone afterwards.

Projecting only the medians would leave a document whose 5th-percentile dew point
exceeds its 5th-percentile temperature — coherent where anyone would look,
incoherent where it would eventually matter.
