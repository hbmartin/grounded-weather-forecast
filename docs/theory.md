# Theory and concepts

This page explains *what* grounded-weather-forecast computes and *why* those choices are the
right ones for this problem. It assumes you are comfortable with regression and
basic time-series ideas, but not that you know meteorological post-processing.

For the engineering that implements all of this, see
[Architecture](architecture.md). For where it currently falls short, see
[Limitations](limitations.md).

---

## 1. The problem

We have two data sources for one point on the Earth's surface:

- **A station.** An AmbientWeather unit in a backyard at Crestline, CA
  (34.28 N, 117.17 W, ~1,400 m), sampling roughly every 60 seconds. This is
  *truth* — but only truth *about the past*.
- **Providers.** Roughly a dozen forecast APIs, each of which repeatedly
  publishes its opinion about the *future* at that location, on an hourly grid
  out to 1–15 days, plus a handful of daily summaries and (for one or two) a
  minute-by-minute precipitation nowcast.

We want three products:

| Product | Horizon | Resolution |
|---|---|---|
| Nowcast | next 60 minutes | per minute |
| Hourly | next 48 hours | per hour |
| Daily | next 10 days | per local day |

The naive approach — average the providers — leaves most of the available
information on the table, and, as we will see, is *also* surprisingly hard to
beat. Understanding why both of those statements are true is the whole game.

---

## 2. Forecast objects: the alignment problem comes first

A forecast is not a number. It is a number attached to four things:

```
(source, issue time, valid time, temporal operator)
```

- **Source** — which provider *and which model* produced it. Open-Meteo alone
  serves ECMWF, GFS and ICON; treating "open_meteo" as one source throws away
  the distinction that matters most.
- **Issue time** (`fetched_at`) — when the forecast was retrieved. This is the
  *information boundary*: a forecast issued at 06:00 knows nothing that happened
  at 07:00.
- **Valid time** — the moment or interval the forecast is *about*.
- **Lead** — `valid_time − issue_time`. This is the single most important
  covariate in the entire system. Every parameter we fit is fitted *per lead
  bucket*, because a provider that is excellent at 3 hours may be mediocre at 7
  days, and vice versa.

We **always recompute lead from timestamps** and never trust a stored
`horizon_hours` column. (In the sample archive, that column, `fetched_at_unix`,
and `run_cycle` are all NULL — a stored derived quantity is a liability.)

### The temporal operator, and why it is not a detail

Does a provider's "14:00 temperature" mean *the temperature at 14:00*, or *the
mean temperature over 14:00–15:00*? Providers do not say. This matters more than
it sounds: on a clear day the temperature ramps ~2 °C/hour, so choosing the
wrong convention manufactures a systematic error of ~1 °C **that looks exactly
like provider bias**. Correcting a bias that is really our own misalignment is
worse than useless — it will fight the real bias.

So we refuse to guess. The dataset layer materializes *both* truths for every
state variable:

```
t__temp_c__inst   instantaneous: mean of clean samples within +/-5 min of the hour
                  (falling back to +/-10 min), i.e. "the temperature at 14:00"
t__temp_c__mean   interval mean over [14:00, 15:00), requiring >=80% minute coverage
```

and the `alignment` command measures, per provider and variable, which one that
provider's forecasts actually correlate with. Variables whose operator is
unambiguous get one definition only:

| Variable | Operator | Why |
|---|---|---|
| gust | max over the hour | a gust *is* an extreme, not a level |
| precipitation | sum over `[H, H+1)` | an accumulation |
| PoP | binary occurrence of precip ≥ 0.254 mm | 0.01 in, the standard "measurable" threshold |
| daily hi/lo | max/min over the *local* calendar day | what a human means by "today's high" |

This is recorded as [ADR 0003](adr/0003-empirical-truth-semantics-calibration.md).

### Snapshots (as-of semantics)

A **snapshot** at issue time `T` is: for each source, its most recent forecast
with `fetched_at <= T`, and not older than a staleness cap (default 12 h).

This is deliberately *not* a uniform grid. Providers are polled at different
cadences (hourly for a diverse core, 6-hourly for the rest). If we forced a
uniform hourly grid, a 6-hourly provider's single forecast would be duplicated
into six training rows, six-fold overweighting it. Snapshots anchor instead to
`forecast_runs.completed_at`, deduplicated onto a 10-minute grid, so every row
of the training matrix corresponds to a real moment at which the system could
actually have made a forecast.

The identical code path builds the training matrix and the live serving
snapshot. That is not tidiness for its own sake: it means the thing we backtest
is *literally* the thing we serve.

---

## 3. Truth: quality control and the aggregation ladder

Truth is a personal weather station, which means it is *wrong sometimes*. Three
classic failure modes, three filters (results recorded as a per-minute bitmask,
never a correction):

| Flag | Bit | Detects |
|---|---|---|
| `OUT_OF_BOUNDS` | 1 | Physically implausible values (a −60 °C reading in July). |
| `SPIKE` | 2 | An isolated excursion that exceeds a per-minute rate limit against **both** neighbours **with opposite signs** — the signature of a radiation-shield transient, not of real weather. A fast but monotone ramp is *not* a spike. The rate limit scales with the actual gap between samples, so a 10-minute gap tolerates 10 minutes' worth of change. |
| `FLATLINE` | 4 | A run of bit-identical values lasting longer than a per-channel threshold: a stuck sensor. |

There is a **second, causal variant** of this filter, and the distinction is
leakage-relevant rather than cosmetic. The `obs__` features — the live station
reading a blender is allowed to see at issue time — are QC'd by
`apply_causal_qc`, which applies bounds immediately, flags a flatline only from
the instant its duration crosses the threshold, and **deliberately omits the
spike rule entirely**, because the two-sided isolated-spike test needs a *later*
sample to fire. A feature that consulted the following minute would be a
forecast that knows the future.

**A flagged sample becomes `NULL`. It is never imputed, corrected, or
interpolated.** If we cannot see truth, we say so, and the row is excluded from
both training and scoring. Optimizing against imputed truth optimizes against
our own imputation.

Precipitation deserves special mention. The station reports `eventrain`, a
*monotone counter that resets*. Naively, "a decrease means the counter reset, so
credit the new value as fresh rain" — and that is exactly the rule that turns
one count of sensor jitter (10.0 → 9.8 → 10.0) into phantom rainfall, twice.

What the code actually does is a **reset-epoch running maximum**. A decrease
counts as a reset only when it falls below `precip_reset_fraction` (default 0.5)
of the prior value; that opens a new epoch. Within an epoch, rain is credited
only when the counter exceeds the highest value seen *since the last reset*, so a
dip-and-rebound contributes nothing:

$$
\Delta_t =
\begin{cases}
c_t, & c_t < f\,c_{t-1} \quad \text{(reset: new epoch)}\\[2pt]
c_t - M_{t-1}, & c_t > M_{t-1} \quad \text{(new epoch high)}\\[2pt]
0, & \text{otherwise}
\end{cases}
\qquad M_t = \max_{s \,\in\, \text{epoch}(t),\ s \le t} c_s
$$

with deltas spanning gaps > 10 minutes discarded as unattributable to any one
hour. Getting this wrong would silently invent or destroy rainfall.

→ [Methods: precipitation](methods/precipitation.md#2-truth-side-accumulation)

Derived quantities (units are normalized to metric on the way in):

- **Dew point** — Magnus approximation, `a = 17.625`, `b = 243.04`:
  ```
  gamma = ln(RH/100) + a*T/(b+T)
  Td    = b*gamma / (a - gamma)
  ```
- **Sea-level pressure** — the station's `RelPress` is *not* sea-level reduced
  (it reads ~25 inHg at 1,400 m, essentially equal to `AbsPress`). We therefore
  reduce it ourselves with the international barometric formula,
  ```
  SLP = p_station * (1 - L*h / (T + L*h + 273.15)) ^ (-5.257),   L = 0.0065 K/m
  ```
  so that it is comparable with what providers publish as `pressure_sea`.

---

## 4. The central decomposition

The system's core claim is that improving a multi-provider forecast decomposes
into **three separable, composable stages**, and that they matter in a specific
and *non-obvious order*:

```
            grounding   >>   anchoring   >   weighting   >   fancy weighting
        (fixes shared         (short-lead        (which          (learned
         + local bias)         only)          provider)         weights)
```

Keeping them separate is what makes the contribution of each one *measurable*.
Every blending method consumes grounded sources; anchoring wraps any blend. If
they were fused into one model, we could not tell which part was doing the work.

### 4.1 Grounding — the big win

Providers forecast a *grid cell*, several kilometres across, at a nominal
elevation. The station is a specific thermometer in a specific yard on a
mountainside. The difference — the microclimate offset — is systematic,
persistent, and invisible to every provider.

Worse, the providers are not independent. Most consumer APIs repackage the same
handful of global NWP models. Their errors are therefore **correlated**, and
correlated error is exactly the error that averaging cannot remove:

> For $k$ sources with equicorrelated errors of correlation $\rho$, the
> equal-weight mean has error variance
>
> $$\operatorname{Var}\!\left(\tfrac{1}{k}\textstyle\sum_i e_i\right)
>  = \left(\tfrac{1}{k} + \left(1 - \tfrac{1}{k}\right)\rho\right)\sigma^2$$
>
> times a single source's, and the *effective* number of independent sources is
> $k_{\text{eff}} = k / \bigl(1 + (k-1)\rho\bigr)$.

Measured on this station's own data, on an exploratory leaderboard run in July
2026. These are **historical illustrations, not packaged or production
evidence** — no module recomputes them, and
[Limitations §5](limitations.md#5-what-one-historical-exploratory-leaderboard-did-and-did-not-say)
says what that vintage does and does not license you to conclude:

| | sources $k$ | mean error correlation $\rho$ | $k_{\text{eff}}$ | variance of equal-weight mean |
|---|---|---|---|---|
| Live providers | 8 | 0.51 | **1.8** | 0.57× (not 0.12×) |
| Distinct NWP models (ECMWF/GFS/ICON) | 3 | 0.66 | **1.3** | 0.77× (not 0.33×) |

Eight providers behave like fewer than two independent ones. **No weighting
scheme can remove a bias that every source shares.** Only correction toward the
station can. That is why grounding comes first and matters most.

The correction is affine, per source × variable × lead bucket:

$$y \approx a + b\,x$$

but the slope $b$ is shrinkable, and **the default is $b = 1$** — a pure bias
correction. This looks timid. It is not; it is the most important thing the data
taught us, and it deserves its own subsection.

"Per lead bucket" is no longer literally true, and the qualification matters: a
bucket's fit is **empirically-Bayes shrunk toward the global fit**, with weight
$w = n_b / (n_b + \text{prior\_rows})$ on the local estimate. A thin bucket
borrows strength from the pooled fit instead of falling off a `min_rows` cliff.
The median-intercept variant, the EWMA and harmonic grounding families, and the
identity guards that stop a degenerate regression from emitting a wild slope all
live in the deep-dive.

→ [Methods: grounding](methods/grounding.md)

#### Why the slope is shrunk to 1 by default

If you regress truth on a *noisy* forecast, ordinary least squares gives

$$b_{\text{ols}} = \frac{\operatorname{cov}(x, y)}{\operatorname{var}(x)}
              = \rho_{xy}\,\frac{\sigma_y}{\sigma_x}$$

and because the forecast carries error that truth does not,
$\operatorname{var}(x) > \operatorname{var}(y)\,\rho^2$, so
**$b_{\text{ols}} < 1$ essentially always**. This is regression dilution. Geometrically,
OLS shrinks predictions *toward the mean of the training sample*.

Inside the training distribution, that shrinkage genuinely lowers MSE — it is
Stein-like, and it is why textbook MOS uses a free slope. But it makes the
correction *a function of the training-period mean*. The moment the evaluation
period sits in a different regime, "shrink toward the training mean" becomes a
mean-dependent tilt, and it **re-introduces exactly the bias grounding exists to
remove**.

This is not hypothetical. On 13 months of backfilled forecasts against this
station, where truth happens to exist only in summer 2025 and spring 2026:

- fitted slopes came out at **0.76–0.89**;
- the free-slope correction carried a **+1.2 to +1.4 °C warm bias** into every
  test fold;
- and it *lost to doing nothing at all* — an uncorrected equal-weight blend beat
  it at 48–168 h leads.

A bias-only correction (`b = 1`, intercept = mean training error) is
**equivariant to level shifts**: change the regime and the correction is
unchanged. It removed the bias (−0.4 to +0.7 °C) and beat the free-slope variant
in *every* lead bucket.

So the default is bias-only, the slope is opt-in via `slope_shrinkage`
($b = 1 + \lambda\,(b_{\text{ols}} - 1)$), and **both variants stay on the leaderboard** so the
archive — not this document — decides when a longer, seasonally representative
history has earned the slope back. Recorded as
[ADR 0004](adr/0004-grounding-defaults-to-bias-only.md).

### 4.2 Blending — the modest, hard-fought win

Once sources are grounded, they must be combined. Four canonical families anchor
the design space and are described below; the registry now holds **40 methods**
across roughly a dozen families, including distributional heads and conformal
wrappers that postdate this section's framing. The complete set, with the exact
estimator behind each `method_id`, is in
[Methods: combination](methods/combination.md) and
[Methods: calibration](methods/calibration.md).

**Equal weight.** The arithmetic mean over *available* sources. This is not a
strawman; it is the benchmark. The *forecast combination puzzle* (Stock &
Watson) is the repeated empirical finding that simple averaging beats
"optimally" estimated weights, because estimating a weight vector adds variance
faster than it removes bias. **It shows up in this project's own data**: raw
equal-weight wins the 96–168 h temperature slice outright, and is within 0.03 °C
of the best method overall.

**Inverse-MSE weights (Bates–Granger).** Weight each source by the inverse of its
grounded training MSE, per lead bucket, renormalized over whichever sources are
actually available in a given row:

$$w_i \propto \frac{1}{\text{MSE}_i}
\qquad
w_i \leftarrow \frac{w_i\,a_i}{\sum_j w_j\,a_j}$$

where $a_i \in \{0,1\}$ marks whether source $i$ is available in that row.
Correlation-ignoring by design. Timmermann's point is that error *covariances* are
too hard to estimate to be worth estimating; the diagonal is enough. This project
does not take that on faith — `inverse_covariance` (Ledoit–Wolf shrinkage of the
full error covariance) is registered precisely so the leaderboard can *measure*
what the diagonal costs.

**Gradient-boosted stacking (LightGBM).** A single model per variable mapping

```
[grounded source values, lead, hour-of-day, month, source ages,
 issue-time observation, ensemble spread, count of available sources]  ->  truth
```

L1 objective (we score MAE), 300 rounds, fixed seed. Trees absorb missing sources
natively via learned default branches — no imputation, no availability
special-casing — and they can express *interactions* (a provider that is only bad
in the morning, in winter) that no per-bucket affine model can. It is the ceiling
of the method set, and on this data it wins the 24–96 h temperature slices with
statistically significant margins.

**Online expert aggregation (EWA and BOA).** Philosophically disjoint from the
regression family: no distributional assumptions, no refits, sequential weight
updates with regret guarantees. Two mechanisms carry the weight:

- *Sleeping experts.* A provider outside its horizon is simply **absent from the
  round**: it is neither updated nor penalized. The standard reduction assigns a
  sleeping expert the awake mixture's loss, giving it an update factor of exactly
  1. Ragged provider horizons (24 h for one, 360 h for another) therefore need no
  special-casing at all — they fall out of the algorithm.
- *Fixed share.* After each multiplicative update a small fraction (0.5%) of the
  awake mass is redistributed uniformly.

  Fixed share is **not decoration**. Vanilla EWA/BOA use a learning rate that
  decays like `1/√t`; an expert that dominates early accumulates a lead that later
  evidence cannot overturn. Implemented without it, this system's aggregators put
  weight **0.9999 on the wrong expert** on a synthetic stream where the good and
  bad experts swap halfway through — precisely the regime (a provider silently
  swapping its backend model) that justifies having online experts at all. Fixed
  share floors every weight, capping the weight ratio, so a recovering source can
  climb back. With `share = 0.005`, drift adaptation became *free*: the
  aggregators track the regime change **and** match the best single expert on
  stationary data.

  $$
  \begin{aligned}
  \text{EWA:}\quad & w_i \leftarrow w_i\,\exp\!\bigl(-\eta\,(\ell_i - \ell_{\text{mix}})\bigr),
    && \eta = \min\!\bigl(0.5,\ \sqrt{8\ln k / T}\bigr)\\
  \text{BOA:}\quad & w_i \leftarrow w_i\,\exp\!\bigl(\eta_i r_i - (\eta_i r_i)^2\bigr),
    && r_i = \ell_{\text{mix}} - \ell_i,\ \ \eta_i = \min\!\bigl(0.5,\ \sqrt{\ln k / V_i}\bigr)\\
  \text{then:}\quad & w \leftarrow (1-s)\,w + s\,\frac{\text{mass}}{k_{\text{awake}}},
    && s = 0.005 \quad \text{[fixed share]}
  \end{aligned}
  $$

  The loss $\ell_i$ is **absolute** error — the aggregators optimize the same
  quantity the leaderboard promotes on. (It was squared error until state schema
  v3; the version is gated hard, so a v2 state on disk raises rather than being
  silently reinterpreted.) Losses are range-normalized per round
  ($\ell \in [0,1]$), which is the precondition for both regret bounds. $k$ is
  the count of *awake* experts — a source that is merely absent cannot perturb
  the learning rate — and a round with fewer than two awake experts is skipped
  entirely rather than handing the lone survivor a free win.

  → [Methods: combination](methods/combination.md#5-online-expert-aggregation)

**Baselines** (the floor every method must clear): persistence (the current
station reading, held constant), harmonic climatology (a ridge-regularized
Fourier regression of truth on month and hour — the ridge matters, see
[Limitations](limitations.md)), and best-single-provider (the source with lowest
training MAE in that lead bucket, with fallback when it is unavailable).

### 4.3 Anchoring — the free win

The station is a live thermometer in the forecast's own grid cell. **No provider
has it.** At short leads, the blend's current error is highly persistent: if the
blend says 18 °C and the yard says 20 °C *right now*, the blend is probably still
2 °C low in ten minutes.

So: take the residual at issue time and add it back, decayed exponentially in
lead.

$$r_0 = \text{obs}(t_0) - \text{blend}(\ell \approx 0)
\qquad
\widehat{y}(\ell) = \text{blend}(\ell) + e^{-\ell/\tau}\,r_0$$

with the weight zeroed once $e^{-\ell/\tau} < 0.05$, so a decayed correction stops
contributing rather than trailing off asymptotically.

$\tau$ is fitted per variable by a one-dimensional grid search over
$\{0.5, 1, 2, 3, 6, 12, 24\}$ h — **including the option of no anchoring at all**,
so if the residual carries no signal, the grid says so and anchoring degrades
exactly to its base blend. On an AR(1)-style synthetic where a per-snapshot offset
decays with lead, anchoring cuts MAE by >25% at 0–3 h leads and converges to the
base by 12 h, which is the behaviour we want and the shape the physics implies.

Two caveats a careful reader should have:

- **The grid search minimizes MSE, while promotion is decided on MAE.** These are
  different objectives, and the selected $\tau$ is therefore not the MAE-optimal
  one. It is an acknowledged inconsistency, not an oversight — recorded here so
  the leaderboard's verdict on `anchored_*` methods is read with it in mind.
- **The residual needs a short-lead row to exist at all.** `_ANCHOR_MAX_LEAD` is
  6 hours: a snapshot whose earliest forecast row is beyond 6 h yields no anchor,
  and the anchored method collapses to its base. This is why anchored variants are
  indistinguishable from their bases on the synthetic backfill, whose leads start
  at 24 h.

This is the useful core of a Kalman filter's "correct toward the observation"
step, extracted without the state-space machinery, the covariance tuning, or the
loss of ability to A/B test the pieces.

---

## 5. The three products

### Hourly

Directly the blended path over the snapshot's future rows, out to 48 h.

### Minutely — an anchored nowcast, *not* a disaggregation

The temptation is to reach for temporal disaggregation (Denton, Chow-Lin) to
"downscale" hourly forecasts to minutes. We deliberately do not, and this is
[ADR 0002](adr/0002-anchored-nowcast-instead-of-temporal-disaggregation.md).

The reason is that **a coarse forecast contains no fine-scale shape
information**. Disaggregating an hourly temperature into minutes cannot conjure
information that was never there; it can only impose an assumed shape. What
*does* carry minute-scale information is the live observation — and that is
exactly what anchoring uses.

The minutely product is **not one fixed formula**. Nine candidate path
constructions are registered and scored on minutely lead buckets like any other
method — straight interpolation of the hourly path, pure observation
persistence, exponential anchoring at each of $\tau \in \{0.25, 0.5, 1, 3\}$ h,
full anchoring, a monotone-decaying fitted ramp, and a fitted slope — and serving
consults the **promoted construction per (variable, minutely bucket)**. The
config's `minutely_tau_hours` is the *no-evidence fallback only*: until a
minutely leaderboard exists, the selection map is empty and the fallback path
runs untouched.

Precipitation is the exception, because it is the one variable for which some
providers publish a genuine minute-resolution nowcast; those native minutely
points are blended directly.

→ [Methods: combination](methods/combination.md#8-minutely-path-constructions)

Per-minute truth's job in this system is **aggregating up** to score hourly and
daily forecasts — not being predicted per-minute by statistical fiat.

### Daily — a hybrid supervised target, *not* reconciliation

The other temptation is hierarchical reconciliation (MinT, thief): force
minute/hour/day forecasts to be mutually coherent. We deliberately do not.

The reason is structural: **reconciliation constrains *linear* aggregates**, and
our daily fields are dominated by *nonlinear* ones — daily max, daily min, max
gust, max PoP. `max` is not a sum; it has no summing matrix; it sits outside
MinT's scope by construction. (Nonlinearly-constrained reconciliation exists as a
2025 preprint, carries no error-reduction guarantee, and max/min — non-
differentiable kinks — are its worst case.)

So daily hi/lo and daily PoP are treated as **their own supervised targets**:

```
features = [ every provider's native daily value,
             equal-weight aggregates of the blended hourly path (ewagg__*),
             lead in days ]
label    = the realized max/min of the QC'd station minutes over the LOCAL day
```

which sidesteps the nonlinear-reconciliation problem entirely and, incidentally,
lets the model learn that (say) providers' daily highs are systematically low
relative to *this* thermometer. The genuinely linear field (precipitation sum) is
derivable from the blended hourly path and is coherent by construction.

"Day" always means the **America/Los_Angeles calendar day**, with DST handled
properly: coverage thresholds are computed against the day's *actual* length
(1380, 1440 or 1500 minutes).

---

## 6. Evaluation: the only thing allowed to declare winners

A forecasting system that cannot honestly measure itself is a random number
generator with good marketing. The backtest engine is deliberately the *only*
component permitted to say a method is better.

### Rolling-origin splits, keyed by issue time

At each origin `O`:

```
train = rows whose truth was KNOWABLE by O
test  = rows ISSUED in (O, O + step]
```

The subtlety is in the first line. It is **not** `issue_time <= O`. A row issued
yesterday about *tomorrow* has an issue time in the past but a truth that has not
happened yet — training on it is leakage. So we compute, for every row, a
`truth_known_at`:

```
minutely: valid_time + 5 min
hourly:   valid_time + 2h                      (the hour must end, plus ingest lag)
daily:    end of the local day + 1h
```

and train only on rows satisfying `truth_known_at <= O`. Both **expanding** and
**rolling** (180-day) windows are supported and reported side by side, because
they answer different questions: expanding asks "what do you know?", rolling asks
"what have you learned lately?".

### Metrics

Point forecasts get **MAE**, **RMSE** and **bias** (the mean error — the column
that caught the grounding bug). Bias deserves its own column precisely because a
method can have unremarkable MAE while being systematically, correctably wrong.

Probability of precipitation gets the **Brier score** and reliability bins,
because a probability scored by MAE is a probability nobody is checking the
calibration of.

Distributional forecasts get **CRPS**, **PIT** histograms and empirical coverage.
The leaderboard's CRPS is the **energy form for an ensemble** (via
`scoringrules`), evaluated on the sorted quantile grid treated as members. A
quantile-grid approximation also exists and is *not* simply twice the mean
pinball loss — it is a weighted rectangle rule over probability levels, with
weights from the midpoints between adjacent levels and the $[0,1]$ endpoints:

$$\widehat{\text{CRPS}} = 2\sum_i w_i\,\overline{\rho_{\tau_i}},
\qquad
w_i = \frac{\tau_{i+1} - \tau_{i-1}}{2}$$

(with the first and last weights closed against 0 and 1). The two agree only for
an equally-spaced level grid; the board reports the former.

Carrying optional quantiles in the blender protocol from day one was a bet on a
future that arrived: **ten registered methods now emit quantiles** (`emos`,
`csgd_emos`, `idr`, `idr_bucket`, `idr_bucket_dcp`, `conformal_gew`,
`conformal_ewma`, `analog_ensemble`, `daily_marginal_emos`, `daily_path_extreme`),
and point-only winners are given quantiles at serve time by residual dressing —
all without a breaking change or a re-backtest.

**Skill** is always relative, and always relative to a *named* reference:

$$\text{skill} = 1 - \frac{\text{MAE}_{\text{method}}}{\text{MAE}_{\text{reference}}}$$

reported against **three** references by default — `best_provider`,
`equal_weight` (the *raw*, ungrounded mean), and `damped_grounded_equal_weight` —
because "we beat the worst provider" is not a claim worth making, and because a
method that only beats the ungrounded mean has demonstrated that grounding works,
not that the method does. Per-variable overrides live in `[promotion.references]`;
the board computes columns for the union so a pinned `skill_vs_equal_weight`
column never silently changes meaning.

### Diebold–Mariano: is that difference real?

An MAE difference of 0.02 °C over 700 samples is noise. So every skill number is
accompanied by a Diebold–Mariano test on the *paired* loss differentials, with:

- a **Bartlett-kernel HAC** variance, because at multi-hour leads consecutive
  forecast errors are serially correlated and the naive variance is far too
  small;
- the **Harvey–Leybourne–Newbold** small-sample correction and a Student-t
  reference, because these slices have hundreds, not millions, of samples;
- and **`n` printed next to every row**, so a thin daily slice announces its own
  lack of power rather than hiding behind a p-value.

### But DM is no longer what decides promotion

A single Diebold–Mariano test was the original arbiter, and it is now only the
*fallback* rule (`[promotion] rule = "legacy"`). The problem with it is
structural: a leaderboard runs dozens of pairwise comparisons every night, on
overlapping data, and re-reads them daily. Taking the argmin of that and calling
it the winner is three distinct statistical errors — multiplicity, optional
stopping, and the winner's curse — stacked on top of each other.

So the promotion decision is made by a *set*, not an argmin, under one of three
rules:

- **`mcs`** — the Model Confidence Set (Hansen–Lunde–Nason). A candidate is
  promoted only if it is *inside* the set and every reference is *outside* it.
- **`seq_mcs`** — an anytime-valid betting e-process per candidate/reference pair;
  the gate fires when accumulated wealth passes $1/\alpha$ by Ville's inequality.
  This is the rule this deployment actually runs.
- **`legacy`** — the original single-DM rule.

Around them sit BH and e-BH false-discovery control over the comparison family,
a winner's-curse correction on the reported MAE of whichever method won, and a
live demotion gate that pulls a promoted release when its realized error exceeds
its backtest error by more than a configured factor.

→ [Methods: model selection and promotion](methods/model-selection.md)

### Leakage: assumed present until proven absent

Four defences, all executable:

1. **Fold-plan invariants** (property-based, via Hypothesis over randomized
   configurations): `max(truth_known_at[train]) <= origin < min(issue_time[test])`,
   train ∩ test = ∅, rolling windows really are bounded.
2. **The poisoning sentinel.** For each fold, corrupt *every* truth value that was
   not yet knowable at that fold's origin (add 10⁶) and re-run the whole engine.
   Assert the fold's test predictions are **bit-identical**. If any future truth
   reaches a model through *any* path — a feature, a fitted aggregate, a stateful
   blender — this test fails. This is the single most valuable test in the
   codebase.
3. **Fresh instances.** The registry stores *factories*, never instances; the
   engine constructs a new blender per fold. A stateful method (the online
   experts) cannot leak yesterday's weights into today's fit.
4. **Feature audit.** No column beginning `t__` may ever reach a blender.

### The provenance wall

Backfilled ("synthetic") and live rows are **never pooled**. They live in
separate files, carry a `source_kind` tag, and any attempt to mix them raises
`MixedProvenanceError`. This is not fastidiousness: a leaderboard built on three
open NWP models says *nothing* about Google or Tomorrow.io, and a number that
quietly averages the two would be worse than no number.

---

## 7. Selected reading

Orientation only. The **canonical reference list lives in
[Methods: bibliography](methods/bibliography.md)**, where every citation is
matched to the module that implements it, and where the papers this project
deliberately *declined* to implement are listed with the recorded reason.

- **Forecast combination.** Bates & Granger (1969); Stock & Watson (2004) on the
  combination puzzle; Timmermann (2006) on why covariance-ignoring weights win —
  a claim this project treats as a hypothesis rather than a premise, which is why
  Ledoit–Wolf `inverse_covariance` is registered as its counterfactual.
- **Statistical post-processing.** Vannitsem et al. (2021, *BAMS*) — the survey
  that frames post-processing as correcting systematic bias and dispersion;
  Gneiting et al. (2005) on EMOS and Scheuerer & Hamill (2015) on CSGD, both now
  implemented; Henzi et al. (2021) on isotonic distributional regression. Our
  grounding is the MOS bias-correction stage of this literature.
- **Online learning.** Vovk / Littlestone–Warmuth (EWA); Wintenberger (2017) on
  BOA; Herbster & Warmuth (1998) on fixed share; Blum's sleeping-experts
  reduction; and the Météo-France expert-aggregation papers
  ([arXiv:2506.15217](https://arxiv.org/pdf/2506.15217)), the closest published
  analogue to this system's expert layer.
- **Evaluation and selection.** Gneiting & Raftery (2007) on proper scoring rules;
  Diebold & Mariano (1995) with the Harvey–Leybourne–Newbold correction;
  Tashman (2000) on rolling-origin evaluation; Hansen, Lunde & Nason (2011) on the
  Model Confidence Set; Waudby-Smith & Ramdas on betting martingales;
  Benjamini & Hochberg (1995) and Wang & Ramdas (2022) on FDR;
  Andrews, Kitagawa & McCloskey (2024) on the winner's curse.
- **What we chose *not* to use.** Athanasopoulos et al. (2017) and Wickramasuriya
  et al. (2019) on temporal hierarchies and MinT — excellent work, structurally
  inapplicable to max/min targets; Denton/Chow-Lin temporal disaggregation —
  solves a problem we do not have.

The institutional analogue of this whole architecture is NOAA's **National Blend
of Models**: bias-correct, weight by lead time, anchor to observations at short
lead. The consumer analogue is ForecastAdvisor/ForecastWatch — which this system
strictly dominates for one location, because it scores per hour, out to 10 days,
with proper scoring rules, against *your actual backyard*.
