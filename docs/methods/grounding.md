# Grounding

**Grounding** is the per-source correction of a provider's forecast toward the
station, fitted per source × variable × lead bucket. It is the first stage of the
pipeline and, on this data, the one that matters most.

This page gives the estimators as implemented. For *why* grounding beats
weighting — the correlated-error argument and the +1.4 °C episode that produced
[ADR 0004](../adr/0004-grounding-defaults-to-bias-only.md) — see
[Theory §4.1](../theory.md#41-grounding-the-big-win).

Symbols follow [notation](notation.md).

---

## 1. Affine grounding

*Implemented in: `blenders/grounding.py::fit_affine`, `AffineGrounding`*

For each (source, variable, lead bucket) cell, fit $y \approx a + b\,x$ by

$$
b_{\text{ols}} = \frac{S_{xy}}{S_{xx}}
= \frac{\sum_t (x_t - \bar{x})(y_t - \bar{y})}{\sum_t (x_t - \bar{x})^2},
\qquad
b = 1 + \lambda\,(b_{\text{ols}} - 1),
\qquad
a = \bar{y} - b\,\bar{x}
$$

where $\lambda \in [0,1]$ is `slope_shrinkage`. The two named settings:

| Name | $\lambda$ | Result |
|---|---|---|
| `BIAS_ONLY` | $0.0$ | $b = 1$, $a = \bar{y} - \bar{x}$ — the mean training error. **The default.** |
| `FREE_SLOPE` | $1.0$ | $b = b_{\text{ols}}$ — textbook MOS. |

### Why $\lambda = 0$ is the default

Regression dilution. The regressor $x$ is a forecast, and therefore carries error
that $y$ does not. Writing $x = y^\ast + \varepsilon$ with $\varepsilon$
independent of the signal,

$$b_{\text{ols}} = \frac{\operatorname{cov}(x,y)}{\operatorname{var}(x)}
= \frac{\operatorname{var}(y^\ast)}{\operatorname{var}(y^\ast) + \operatorname{var}(\varepsilon)} \cdot \frac{\operatorname{cov}(y^\ast, y)}{\operatorname{var}(y^\ast)} < 1$$

essentially always. Geometrically OLS shrinks predictions toward
$\bar{y}_{\text{train}}$, which lowers in-sample MSE — it is Stein-like, and it is
why textbook MOS uses a free slope.

The problem is **equivariance**. Under a level shift $y \mapsto y + c$,
$x \mapsto x + c$ (a regime change: a warmer month, a different airmass), the
bias-only correction $a = \bar{y} - \bar{x}$ is unchanged. The free-slope
correction is not: its prediction becomes

$$\hat{y} = \bar{y} + b_{\text{ols}}(x - \bar{x})$$

which is a *function of the training-period mean*. Out of regime, "shrink toward
the training mean" is a mean-dependent tilt, and it re-injects exactly the bias
grounding exists to remove. Measured on 13 months of backfill: fitted slopes
0.76–0.89, a +1.2 to +1.4 °C warm bias carried into every test fold, and a loss
to doing nothing at all at 48–168 h leads.

Both variants stay registered (`grounded_equal_weight` at $\lambda = 0$,
`affine_equal_weight` at $\lambda = 1$) so the archive — not this document —
decides when a longer, seasonally representative history has earned the slope back.

### The MAE-consistent variant

Setting `intercept="median"` (legal only with $\lambda = 0$) gives

$$a = \operatorname{median}_t\,(y_t - x_t), \qquad b = 1$$

which is the offset minimizing $\sum_t |y_t - (x_t + a)|$ — consistent with the
MAE the leaderboard promotes on, where the mean intercept is not. Registered as
`grounded_median_equal_weight`.

### Degeneracy guards

| Guard | Value | Behaviour |
|---|---|---|
| `_MIN_FIT_ROWS` | 24 | fewer rows → no local fit |
| `_MIN_VARIANCE` | $10^{-9}$ | $S_{xx}$ below this → return `IDENTITY` $(0, 1)$ |
| `_MAX_ABS_SLOPE` | 5.0 | $|b_{\text{ols}}| > 5$ → return `IDENTITY` |

A near-constant regressor makes $b_{\text{ols}}$ arbitrary; an unguarded fit on
such a cell can emit a slope of several hundred and destroy the forecast. Both
guards fail to the identity map, which is the honest answer: *we learned nothing
here, so we change nothing.*

---

## 2. Per-bucket empirical-Bayes shrinkage

*Implemented in: `blenders/protocol.py::PerBucketFitter`, `FittedBuckets`*

"Fitted per lead bucket" would be a disaster taken literally: the `240h+` bucket
may hold 30 rows while `24-48h` holds thousands. The original design used a hard
`min_rows` cliff — local fit above, global fit below — which makes the correction
discontinuous in $n_b$ and throws away partial information.

The current scheme is empirical-Bayes shrinkage toward the global fit. With
$S_g$ the state fitted on all rows and $S_b$ the state fitted on bucket $b$'s
$n_b$ rows:

$$S_b^{\text{final}} = w\,S_b + (1 - w)\,S_g,
\qquad
w = \frac{n_b}{n_b + n_0}$$

with $n_0 = $ `prior_rows`, defaulting to $\texttt{min\_rows}/4$. With
`min_rows = 24` this gives $n_0 = 6$, so a bucket holding exactly `min_rows`
observations receives weight $w = 24/30 = 0.8$ on its own fit.

For affine grounding the blend is linear in both coefficients
(`_blend_affine`), which is valid because the correction $x \mapsto a + bx$ is
itself linear in $(a, b)$: a convex combination of coefficient vectors is the
convex combination of the corrections.

!!! note "Where the cliff is kept deliberately"
    `blend` is optional. When it is `None`, `PerBucketFitter` falls back to
    all-or-nothing: local fit if $n_b \ge \texttt{min\_rows}$, else global. This
    is used where linear interpolation of the *state* is not meaningful —
    `HarmonicGrounding` (a ridge coefficient vector fitted on a different design)
    and `BestProvider` (an argmin, where "half of source A and half of source B"
    is not a source).

---

## 3. EWMA grounding — adaptive bias

*Implemented in: `blenders/ewma_grounding.py::EwmaBiasGrounding`*

A batch affine fit assumes the bias is stationary over the training window. It is
not: a provider that silently swaps its backend model changes bias overnight.
This is NCEP's decaying-average bias correction, in operational use since 2006,
applied per (source, lead bucket, hour-of-day bin) and replayed in issue-time
order:

$$\beta \leftarrow (1 - w)\,\beta + w\,(x_t - y_t), \qquad w = 0.05$$

$$x^{\text{corrected}}_t = x_t - \frac{n}{n + n_0}\,\beta, \qquad n_0 = 10$$

The $n/(n + n_0)$ factor is a warm-up: a bias estimated from three observations is
applied at 23% strength, not in full. Hour-of-day bins are 3 hours wide
(`_HOURS_PER_BIN = 3`, so 8 bins), capturing the diurnal structure of a
radiation-shield or siting bias that a lead-only fit averages away. Daily
matrices have no `valid_hour_local` and collapse to a single bin.

Registered as `ewma_grounded_equal_weight` and `ewma_inverse_mae`.

---

## 4. Harmonic grounding

*Implemented in: `blenders/harmonic_grounding.py::HarmonicGrounding`*

SAMOS-informed. Instead of a scalar intercept, the bias becomes a smooth
*function of solar and seasonal phase* — the slope stays fixed at 1 and the
correction is a ridge regression of the residual $y - x$ on a harmonic design:

$$
D = \bigl[\,1,\ \sin(\text{el}),\ \sin^2(\text{el}),\
\sin(\text{doy}),\ \cos(\text{doy}),\
2\sin(\text{doy})\cos(\text{doy}),\ \cos^2(\text{doy}) - \sin^2(\text{doy})\,\bigr]
$$

where `el` is solar elevation (from the dependency-free NOAA solar position code
in `solar.py`) and the last two columns are the semiannual harmonics
$\sin(2\,\text{doy})$ and $\cos(2\,\text{doy})$ written out via double-angle
identities. Where solar elevation is unavailable the design falls back to
$[1, \sin(\text{hour}), \cos(\text{hour})]$ plus the same seasonal terms.

`_ridge_fit` solves the augmented system with an **unpenalized intercept**
(penalizing it would shrink the mean bias toward zero, which is the one thing
grounding must not do), $\rho = 1.0$, `_MIN_FIT_ROWS = 48`.

There is deliberately **no** shrinkage blend here: the thin-bucket fallback is a
zero coefficient vector — i.e. no correction — rather than a genuine curve
borrowed from another bucket. Registered as `harmonic_grounded_equal_weight`.

---

## 5. Which grounding to use

Nothing here is chosen by argument. All five variants are registered, all are
scored on the same rolling-origin folds, and the promotion machinery in
[model selection](model-selection.md) decides per variable × lead bucket. What
this page provides is the ability to read that verdict — to know that
`affine_equal_weight` losing to `grounded_equal_weight` at long lead is
regression dilution and not noise, and that `harmonic_grounded_equal_weight`
winning in summer afternoons is a diurnal shield bias the scalar intercept could
not see.
