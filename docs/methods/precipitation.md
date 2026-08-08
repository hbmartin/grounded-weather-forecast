# Precipitation

Precipitation breaks almost every assumption the rest of the system rests on. It
is not Gaussian, not continuous, not symmetric, and — on this archive — mostly
absent. Nearly every precipitation-specific guard in the codebase exists because
a general-purpose method did something indefensible on a dry day.

This page collects that machinery. Symbols follow [notation](notation.md).

---

## 1. Why it is a different problem

A temperature forecast is a location estimate on a roughly Gaussian variable.
Hourly precipitation is a **mixed discrete–continuous** random variable:

$$Y = \begin{cases}
0 & \text{with probability } 1 - \pi \quad (\text{it does not rain})\\
Z > 0 & \text{with probability } \pi,\quad Z \text{ heavily right-skewed}
\end{cases}$$

Consequences that propagate everywhere:

- **The mean is not a useful summary.** A forecast of 0.3 mm may mean "certainly
  a light drizzle" or "10% chance of 3 mm". These are different forecasts and
  MAE cannot tell them apart.
- **Averaging is not neutral.** Equal-weighting eight providers, two of whom
  forecast a storm, produces a value no provider believes and that will verify
  against neither outcome.
- **Robust statistics degenerate.** On a dry day the cross-source median is 0 and
  the MAD is 0, so any deviation is infinitely many MADs. Every outlier rule
  needs an absolute floor or it flags all rain as an error.
- **Fits starve.** A five-parameter distribution needs wet cases, and a Southern
  California summer supplies almost none.

---

## 2. Truth-side accumulation

*Implemented in: `dataset/truth.py::_precip_deltas`*

The station reports `eventrain`: a counter that climbs monotonically within a
rain event and resets toward zero when the event ends. Converting it to hourly
rainfall looks trivial and is not.

The naive rule — "a decrease means a reset, so credit the new value as fresh
rain" — turns one count of sensor jitter into phantom rainfall. A sequence
$10.0 \to 9.8 \to 10.0$ produces $9.8 + 10.0 = 19.8$ mm of rain that never fell.

The implemented rule is a **reset-epoch running maximum**. A decrease opens a new
epoch only when it falls below `precip_reset_fraction` (default 0.5) of the prior
value; within an epoch, rain is credited only above the highest value seen since
the last reset:

$$
\Delta_t =
\begin{cases}
c_t, & c_t < f\,c_{t-1} \quad \text{(reset: new epoch)}\\[4pt]
c_t - M_{t-1}, & c_t > M_{t-1} \quad \text{(new epoch high)}\\[4pt]
0, & \text{otherwise}
\end{cases}
\qquad
M_t = \max_{\substack{s \le t \\ s \in \text{epoch}(t)}} c_s
$$

The dip-and-rebound now contributes exactly zero. Deltas spanning gaps beyond
`_GAP_ATTRIBUTION_LIMIT_MINUTES = 10.0` are dropped as unattributable to any one
hour — a 40-minute logging gap containing 3 mm cannot be assigned.

!!! warning "The 0.5 fraction is calibrated against jitter, not against a storm"
    The reference archive contains almost no heavy rain (max event total 1.31 in),
    so the reset fraction has never been tested against a real event where the
    counter legitimately drops by more than half. `rainofhourly` is retained as a
    cross-validator. See [Limitations §8](../limitations.md#8-known-assumptions-worth-challenging).

---

## 3. PoP

Probability of precipitation is defined against the standard "measurable"
threshold:

$$\text{PoP}_V = \mathbb{1}\{\text{precip accumulation over } [V, V{+}1\text{h}) \ge 0.254\text{ mm}\}$$

0.254 mm is 0.01 inch. It is the conventional choice, and it is a *choice* — the
threshold is `[dataset] pop_threshold_mm` and moving it changes what every PoP
number in the system asserts.

PoP is scored by **Brier score and reliability bins**, never by MAE. A
probability judged by MAE is a probability whose calibration nobody is checking:
a forecaster who always says 0.1 on a 10%-rain climate has excellent MAE and has
told you nothing.

Two calibration maps are registered — `pop_platt` and `pop_beta` — with the
critical `_MIN_EACH_CLASS = 5` guard that returns the **identity map** when a
window contains only one class. See
[calibration §5](calibration.md#5-probability-calibration-for-pop).

---

## 4. Cross-source QC guards

*Implemented in: `dataset/provider_qc.py`, defaults in `config.py`*

Provider rows are otherwise trusted verbatim, so a single bad value flows
straight into the grounding fit and the blend. Two conservative filters run
before grounding. Truth is never consulted, so this introduces no leakage.

### Absolute bounds

Gross unit and garbage errors:

| Variable | Bounds |
|---|---|
| `precip_mm` | $[0, 500]$ |
| `precip_sum_mm` | $[0, 2000]$ |
| `pop` | $[0, 1]$ |

### Robust cross-source outlier pass

A value is nulled when it disagrees with the other providers at the same valid
time by **both** more than `mad_k` scaled MADs **and** more than an absolute
floor:

$$\bigl|x_i - \operatorname{med}(x)\bigr| > k \cdot 1.4826 \cdot \operatorname{MAD}(x)
\quad\textbf{and}\quad
\bigl|x_i - \operatorname{med}(x)\bigr| > \texttt{min\_deviation}$$

with `mad_k = 5.0`, `min_sources = 4`. A nulled value becomes `NaN` in the matrix
and drops out of every blender's availability mask, so no blender needs to change.

**The conjunction is the whole design.** On a dry day
$\operatorname{med} = \operatorname{MAD} = 0$, so $k \cdot \text{MAD} = 0$ and the
MAD test alone would flag every nonzero forecast as an outlier. The floor carries
the entire test:

| Variable | `min_deviation` |
|---|---|
| `temp_c`, `temp_max_c`, `temp_min_c`, `dew_point_c` | 8.0 |
| `humidity_pct` | 40.0 |
| `pressure_sea_hpa` | 20.0 |
| `precip_mm` | 10.0 |
| `precip_sum_mm` | 25.0 |

The precipitation floors are deliberately generous. At 10 mm/hour they null a
lone hallucination — one provider stored 138 mm for a bone-dry 2026-08-06 — while
leaving ordinary drizzle disagreement untouched. And in a genuine storm multiple
providers forecast heavy rain, the median rises, and a large value stops being an
outlier at all. The filter is designed to be **wrong in the direction of keeping
data**: genuine provider diversity is what the blend relies on.

!!! note "Gusts and PoP are excluded from the cross-source pass"
    `wind_gust_ms` and `pop` are in the bounds table but not in
    `cross_source_variables`. Gusts are genuine extremes — a provider forecasting
    a much higher gust than its peers is often *right*, and the whole point of a
    gust field is to capture the tail. PoP is bounded in $[0,1]$ where a robust
    outlier test has little to add.

---

## 5. Precipitation-aware methods

### CSGD

`csgd_emos` is the one method whose distributional assumption matches the
variable: a censored shifted gamma whose censored mass **is** the probability of
a dry hour, giving PoP and the amount distribution from one consistent fit.
`_MIN_WET_ROWS = 10`, below which it abstains with
`fit_status = "insufficient_wet_rows"`. Full derivation in
[calibration §2](calibration.md#2-csgd-censored-shifted-gamma).

### Sparse shrinkage

*Implemented in: `blenders/sparse_shrink.py::SparseShrink` — `precip_sparse_shrink`,
scoped to `precip_sum_mm`*

$$\hat{y} = w\cdot\overline{x} + (1 - w)\cdot\text{climatology},
\qquad
w = \frac{n}{n + k},\qquad k = 2.0$$

where $n$ is the count of providers available **in that row**.

The failure it addresses is specific: past day six only one or two providers
still publish daily precipitation, so equal weight degenerates to a single
provider's monsoon guess served nearly undiluted. `damped_grounded_equal_weight`
already regresses toward climatology, but its $\alpha$ is fitted by MAE search —
and the far buckets where the problem lives are exactly where that fit is
noisiest.

So this method imposes the trust schedule **structurally**: with one provider
available, $w = 1/3$; with four, $w = 2/3$. **No skill parameter is estimated**,
which makes the method well-defined on slices with almost no evidence — the exact
regime it exists for. The A/B against `damped_grounded_equal_weight` is arbitrated
by the leaderboard, not by this argument.

### Method scoping

*Implemented in: `blenders/registry.py::supports_product`, and per-method `variables`*

Precipitation heads are registered with an explicit variable scope —
`csgd_emos` to `precip_mm` / `precip_sum_mm`, `precip_sparse_shrink` to
`precip_sum_mm`, `pop_platt` / `pop_beta` to `pop`. A method whose assumptions
only hold for one variable is not offered the others, so the leaderboard never
has to discover by measurement that a gamma likelihood is a bad model for
temperature.

---

## 6. Coherence

The precipitation *sum* over a local day is a genuinely **linear** aggregate of
the hourly path, so it is derivable from the blended hourly forecast and coherent
by construction — unlike daily max/min, which are nonlinear and get their own
supervised targets ([combination §9](combination.md#9-daily-heads)).

This is the one place where the hierarchical-reconciliation literature would
apply, and the one place it is not needed.
