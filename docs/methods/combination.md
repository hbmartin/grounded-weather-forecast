# Combination and blending

Once sources are grounded, they must be combined. This page gives every
registered combiner as implemented. Symbols follow [notation](notation.md); the
narrative case for the design is [Theory §4.2](../theory.md#42-blending-the-modest-hard-fought-win).

---

## 1. The availability algebra

*Implemented in: `blenders/protocol.py::renormalize_weights`, `masked_average`*

Every weighting scheme in this system produces a weight vector over *all* sources
and then renormalizes over the ones actually present in each row:

$$w_{ti} = \frac{w_i\,a_{ti}}{\sum_j w_j\,a_{tj}},
\qquad
\hat{y}_t = \sum_i w_{ti}\,x_{ti}$$

with the row set to `NaN` when $\sum_j w_j a_{tj} = 0$. This is not
error-handling bolted on; it is why ragged provider horizons need no special
casing anywhere else in the codebase. A source outside its horizon has
$a_{ti} = 0$ and simply drops out of that row's normalization.

Two shared finalizers close every path:

- **`finalize_point`** — clips `TargetKind.PROBABILITY` outputs to $[0,1]$, then
  clamps to the variable's declared `minimum`/`maximum`.
- **`finalize_quantiles`** — applies **monotone rearrangement** (row-wise
  `np.sort`) before the same clamps. Rearrangement never worsens a proper scoring
  rule, so it is free insurance against a fitted head emitting crossed quantiles.
  *Every* quantile emitter routes through it.

---

## 2. The diversification ceiling

Before any weighting scheme, the question of how much averaging can possibly buy.
For $k$ sources with equicorrelated errors of correlation $\rho$ and common
variance $\sigma^2$:

$$\operatorname{Var}\!\left(\frac{1}{k}\sum_i e_i\right)
= \frac{\sigma^2}{k}\bigl(1 + (k-1)\rho\bigr)
= \sigma^2\left(\frac{1}{k} + \left(1 - \frac{1}{k}\right)\rho\right)$$

Defining $k_{\text{eff}}$ as the number of *independent* sources giving the same
variance reduction:

$$\frac{\sigma^2}{k_{\text{eff}}} = \frac{\sigma^2}{k}\bigl(1 + (k-1)\rho\bigr)
\quad\Longrightarrow\quad
k_{\text{eff}} = \frac{k}{1 + (k-1)\rho}$$

As $k \to \infty$ this tends to $1/\rho$ — a hard ceiling independent of how many
providers you add. At $\rho = 0.51$ the ceiling is under 2. Most consumer weather
APIs repackage the same handful of global NWP models, so $\rho$ is structurally
large, and **this is the quantitative reason grounding matters more than
weighting**: no convex combination can remove a bias every source shares.

The error correlation matrix is reported by `reports/correlation.py` and rendered
in dashboard zone D.

---

## 3. Linear weighting schemes

| `method_id` | Class | Weights | Key constants |
|---|---|---|---|
| `equal_weight`, `grounded_equal_weight`, `affine_equal_weight`, `grounded_median_equal_weight` | `combine.GroundedEqualWeight` | $w_i = 1/k$ over available sources | — |
| `inverse_mse`, `inverse_mae` | `combine.InverseErrorWeights` | Bates–Granger, below | `_MIN_LOSS = 1e-6`, `_MIN_ROWS_PER_SOURCE = 12` |
| `trimmed_mean`, `grounded_trimmed_mean` | `trimmed.TrimmedMean` | symmetric trimmed mean | `TRIM_FRACTION = 0.2` |
| `inverse_covariance` | `invcov.InverseCovariance` | GLS, below | `_MIN_COMPLETE_ROWS = 30`, cap $2/k$ |
| `cluster_equal_weight` | `cluster.ClusterEqualWeight` | de-duplicate, then equal weight | threshold 0.9, `_MIN_OVERLAP = 24` |
| `damped_grounded_equal_weight` | `damped.DampedBlend` | shrink toward climatology | `_ALPHA_GRID` = 21 points |
| `precip_sparse_shrink` | `sparse_shrink.SparseShrink` | count-based shrinkage | $k_{\text{pseudo}} = 2.0$ |

### Bates–Granger inverse-error weights

$$w_i \propto \frac{1}{\frac{1}{n_i}\sum_t |e_{ti}|^p},
\qquad p = 2\ (\texttt{inverse\_mse}),\quad p = 1\ (\texttt{inverse\_mae})$$

fitted per lead bucket, then renormalized per row over availability. Correlation-
ignoring by design (Timmermann's argument: error covariances are too hard to
estimate to be worth estimating). A source with fewer than
`_MIN_ROWS_PER_SOURCE = 12` training rows is not dropped — it is assigned the
*worst observed loss*, so it participates weakly rather than being silently
excluded from a slice it might genuinely serve.

### Inverse covariance — the counterfactual

`inverse_covariance` exists to *test* Timmermann rather than assume him. It is
the minimum-variance (GLS) portfolio,

$$w \propto \Sigma^{-1}\mathbf{1}$$

with $\Sigma$ the Ledoit–Wolf-shrunk error covariance, implemented explicitly:

$$
S = \tfrac{1}{n}X_c^\top X_c,
\qquad
T = \mu I \ \text{ with } \ \mu = \tfrac{\operatorname{tr} S}{k},
$$
$$
d^2 = \|S - T\|_F^2,
\qquad
\bar{b}^2 = \frac{1}{n^2}\sum_t \bigl\|x_t x_t^\top - S\bigr\|_F^2,
\qquad
\delta = \min\!\left(1, \frac{\bar{b}^2}{d^2}\right),
$$
$$
\hat{\Sigma} = \delta\,T + (1 - \delta)\,S
$$

Raw GLS weights can be large and negative — an unbiased-but-wild portfolio. They
are clipped at 0 and projected onto the **capped simplex** with cap $2/k$.
`_capped_simplex` fixes over-cap entries and redistributes the remainder among
the free entries, iterating at most $k$ passes; a plain clip-then-renormalize can
push an already-capped weight back over the ceiling.

Requires ≥ 2 observed sources and `_MIN_COMPLETE_ROWS = 30` complete rows, else
abstains with `NaN`.

### Trimmed mean

Drop $\lfloor 0.2\,k_{\text{avail}} \rfloor$ from each end of the row's sorted
available values and average the rest. Robustness with **zero estimated
parameters** — it cannot overfit, which makes it a genuinely informative
competitor rather than a strawman.

### Cluster equal weight

Greedy correlation clustering on training errors: sources whose error correlation
exceeds 0.9 are near-duplicates, so keep the lowest-MAE member of each group and
equal-weight the survivors. This attacks $k_{\text{eff}}$ directly rather than
accepting it. `_MIN_OVERLAP = 24` shared rows are required to estimate a
correlation; an *unestimable* $\rho$ never counts as duplication — the failure
mode must be "keep both", not "silently drop a source we could not compare".

### Damped blend

$$\hat{y} = \alpha\,\text{base} + (1 - \alpha)\,\text{climatology}$$

with $\alpha$ chosen per bucket by **direct MAE grid search** over
`np.linspace(1.0, 0.0, 21)`, then EB-shrunk like any other bucket state. The grid
is ordered *descending* so that argmin ties resolve to $\alpha = 1$ — i.e. ties
go to "do not damp". At long lead the optimizer discovers on its own that a
forecast should decay toward climatology; this is Monhart et al.'s
lead-dependent correction, learned rather than imposed. It is also one of the
three default leaderboard references.

---

## 4. Gradient-boosted stacking

*Implemented in: `blenders/gbm.py::GbmStacker` — `gbm`*

One LightGBM model per variable, mapping

$$\bigl[\,x_{t,1..k},\ \ell,\ \sin/\cos(\text{hour}),\ \sin/\cos(\text{doy}),\
\texttt{age}\_\_\ast,\ \texttt{obs}\_\_\ast,\ \texttt{ewagg}\_\_\ast,\
\texttt{ens}\_\_\ast,\ \operatorname{nanstd}(x_t),\ n_{\text{avail}}\,\bigr]
\;\longrightarrow\; y$$

| Parameter | Value | Why |
|---|---|---|
| `objective` | `regression_l1` | matches the MAE promotion metric |
| `_NUM_ROUNDS` | 300 | |
| `learning_rate` | 0.05 | |
| `num_leaves` | 31 | |
| `min_child_samples` | 20 | |
| `feature_fraction` / `bagging_fraction` | 0.9 / 0.9, `bagging_freq=1` | |
| `seed` | 20260713, `deterministic=True`, `force_row_wise=True` | byte-reproducible refits |
| `min_fit_rows` | 500 | below this it **abstains with `NaN`** |

Trees absorb missing sources natively through learned default branches — no
imputation, no availability special-casing — and they express *interactions* (a
provider bad only on winter mornings) that no per-bucket affine model can. It is
the ceiling of the method set, and the `min_fit_rows = 500` abstention is what
keeps it from being the ceiling of the *overfitting* too.

---

## 5. Online expert aggregation

*Implemented in: `blenders/experts.py` — `ewa`, `boa`*

Philosophically disjoint from the regression family: no distributional
assumptions, no refits, sequential weight updates with regret guarantees.
Experts are the grounded sources; one state per lead bucket.

Each round, losses are **range-normalized** over the awake set — the precondition
for both regret bounds:

$$\tilde{\ell}_i = \frac{\ell_i}{\max_{j \in \text{awake}} \ell_j} \in [0,1],
\qquad
\ell_{\text{mix}} = \sum_{i \in \text{awake}} \tilde{w}_i\,\tilde{\ell}_i$$

$$
\begin{aligned}
\textbf{EWA:}\quad & \text{factor}_i = \exp\!\bigl(-\eta\,(\tilde{\ell}_i - \ell_{\text{mix}})\bigr),
&& \eta = \min\!\bigl(0.5,\ \sqrt{8\ln k / T}\bigr)\\[4pt]
\textbf{BOA:}\quad & r_i = \ell_{\text{mix}} - \tilde{\ell}_i,\quad V_i \mathrel{+}= r_i^2, \\
& \text{factor}_i = \exp\!\bigl(\eta_i r_i - (\eta_i r_i)^2\bigr),
&& \eta_i = \min\!\bigl(0.5,\ \sqrt{\ln k / V_i}\bigr)
\end{aligned}
$$

$$\textbf{fixed share:}\quad
w \leftarrow (1 - s)\,w^{\text{updated}} + s\,\frac{\text{mass}}{k_{\text{awake}}},
\qquad s = 0.005$$

then a global renormalization. Exponents are clipped to $[-50, 50]$;
$\eta_{\max} = 0.5$.

**The loss is absolute error.** This changed in state schema v3, to match the MAE
the leaderboard promotes on. The version is gated hard — a v2 state on disk
*raises* rather than being reinterpreted — because mixing two loss definitions
inside one weight trajectory produces a number that means nothing.

### Sleeping experts

A source outside its horizon is **absent from the round**: neither updated nor
penalized. Blum's standard reduction assigns a sleeping expert the awake
mixture's loss, giving update factor exactly 1. This is why $k$ in the learning
rates is the count of *awake* experts — a merely-absent source must not perturb
$\eta$. A round with fewer than `_MIN_AWAKE = 2` awake experts is skipped
entirely rather than handing the lone survivor a free win.

### Why fixed share is load-bearing

Vanilla EWA/BOA use a learning rate decaying like $1/\sqrt{t}$. An expert that
dominates early accumulates a lead later evidence cannot overturn. Implemented
without fixed share, this system's aggregators put weight **0.9999 on the wrong
expert** on a synthetic stream where the good and bad experts swap halfway
through — precisely the regime (a provider silently swapping its backend model)
that justifies having online experts at all.

Fixed share floors every weight, capping the achievable weight *ratio*, so a
recovering source can climb back. At $s = 0.005$ drift adaptation turned out to
be free: the aggregators track the regime change **and** match the best single
expert on stationary data.

### Replay and state

Replay cursors are keyed by **target-resolution time** — `truth_known_at`, then
`valid_time`, then `forecast_date`, then $T + \ell$ — with SHA-256 prefix digests
to detect archive corrections. A grounding refit invalidates the state and forces
a full replay, because the experts' losses were computed against a correction
that no longer exists.

---

## 6. Other combiners

**Seamless regression** (`seamless_regression`, Dabernig & Atencia 2024). Per-bucket
ridge over $[\text{forward-filled sources},\ \texttt{obs}\_\_\ast,\ 1]$:

$$\beta = (X^\top X + \rho I)^{-1} X^\top y, \qquad \rho = 10.0,\ \texttt{\_MIN\_FIT\_ROWS} = 48$$

Missing sources take the row mean of the available ones. This collapses
grounding, weighting, and anchoring into one coefficient vector — it is the
explicit test of whether the three-stage decomposition earns its keep, and it is
on the leaderboard so that question gets an answer rather than an opinion.

**Analog ensemble** (`analog_ensemble`, Delle Monache et al. 2011/2013). Normalized
Euclidean distance over the sources finite in *both* rows:

$$d(x, x_t) = \sqrt{\frac{1}{|\mathcal{S}|}\sum_{s \in \mathcal{S}}
\left(\frac{x_s - x_{t,s}}{\sigma_s}\right)^{2}}$$

with $\sigma_s$ from the training archive (`_MIN_SIGMA = 1e-6`, ≥2 finite rows
required). `n_analogs = 25`; the pool is the lead bucket if it holds at least
$25 \times 2 = 50$ rows, else the full archive. Point forecast = analog median;
quantiles = analog empirical quantiles — which makes it one of the few methods
whose uncertainty is *nonparametric and conditional*. `_MIN_FIT_ROWS = 100`.

**RAFT** (`raft_grounded`, Schuhen et al. 2020) and **cluster** are covered above and
in the registry.

---

## 7. Anchoring

*Implemented in: `blenders/anchoring.py`*

Anchoring is a **wrapper**: it takes any base blend and adds back the issue-time
residual, decayed in lead.

$$r_0 = \text{obs}(t_0) - \text{blend}(\ell \approx 0),
\qquad
\hat{y}(\ell) = \text{blend}(\ell) + \omega(\ell)\,r_0$$

$$\omega(\ell) = \begin{cases}
e^{-\ell/\tau}, & e^{-\ell/\tau} \ge 0.05\\
0, & \text{otherwise}
\end{cases}$$

The floor (`_WEIGHT_FLOOR = 0.05`) makes a decayed correction *stop* rather than
trail off asymptotically.

$\tau$ is selected by a one-dimensional search over

$$\tau \in \{\texttt{None}\} \cup \{0.5,\ 1,\ 2,\ 3,\ 6,\ 12,\ 24\}\ \text{h}$$

`None` — no anchoring at all — is a first-class candidate, so a residual carrying
no signal degrades the wrapper exactly to its base rather than injecting noise.

Two properties a reader should carry away:

!!! warning "The search minimizes MSE; promotion is on MAE"
    `_search_tau` minimizes $\frac{1}{n}\sum(\hat{y} - y)^2$. The leaderboard
    promotes on MAE. The selected $\tau$ is therefore not the MAE-optimal one.
    This is an acknowledged inconsistency, recorded rather than hidden — see
    [notation §5](notation.md#5-estimands-and-losses).

!!! warning "No short-lead row, no anchor"
    `_ANCHOR_MAX_LEAD = 6.0` hours. A snapshot whose earliest forecast row is
    beyond 6 h yields no residual, and the anchored method collapses to its base.
    This is why anchored variants are numerically identical to their bases on the
    synthetic backfill, whose leads begin at 24 h.

Anchoring is the useful core of a Kalman filter's "correct toward the
observation" step, extracted without the state-space machinery, the covariance
tuning, or the loss of ability to A/B test the pieces.

---

## 8. Minutely path constructions

*Implemented in: `backtest/minutely.py::minutely_methods`, `serve/predict.py::_minutely_plans`*

The minutely product is **not one formula**. Nine candidate path constructions
are registered and scored on minutely lead buckets like any other method:

| `method_id` | Construction |
|---|---|
| `minutely_interp` | interpolate the hourly path to lead $m/60$; no anchor |
| `minutely_persistence` | hold the current observation constant |
| `minutely_anchor_tau_0.25h` … `_3h` | exponential anchoring at $\tau \in \{0.25, 0.5, 1, 3\}$ h |
| `minutely_anchor_full` | undecayed residual across the whole minutely horizon |
| `minutely_ramp` | fitted response, clipped, constrained monotone non-increasing |
| `minutely_fitted_slope` | fitted response, clipped, unconstrained |

Serving consults the promoted construction **per (variable, minutely bucket)** —
so `0-5m` may serve persistence while `45-60m` serves interpolation, which is
exactly the crossover the physics implies. The config's `minutely_tau_hours` is
the **no-evidence fallback only**: until a minutely leaderboard exists the
selection map is empty, `_minutely_plans` returns `None`, and the fallback path
runs untouched.

Precipitation is the exception. It is the one variable for which some providers
publish a genuine minute-resolution nowcast, and those native minutely points are
blended directly rather than constructed from the hourly path.

---

## 9. Daily heads

*Implemented in: `blenders/daily_heads.py`*

Daily max/min are **nonlinear** aggregates, which is why hierarchical
reconciliation is refused ([ADR 0002](../adr/0002-anchored-nowcast-instead-of-temporal-disaggregation.md)
and [Theory §5](../theory.md#daily-a-hybrid-supervised-target-not-reconciliation)).
They are treated as their own supervised targets:

**`daily_path_extreme`** — take each source's hourly-path extreme as an ensemble
member, bias-correct per bucket ($\text{bias} = \operatorname{mean}(y - \text{member})$,
EB-shrunk), then `nanmedian` for the point and `nanquantile` for the spread.
`_MIN_MEMBERS = 3`, `_MIN_FIT_ROWS = 60`. The members' rank structure is
Schefzik's ensemble copula coupling collapsed to the scalar extreme.

**`daily_marginal_emos`** — see [calibration](calibration.md#4-daily-marginal-emos).
