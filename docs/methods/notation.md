# Notation and objects

This page fixes the symbols, indexing, and estimands used throughout the Methods
section. It is the shortest page here and the one worth reading first: almost
every disagreement about a forecasting method turns out to be a disagreement
about what is being indexed.

For the narrative account of *why* the system is built this way, see
[Theory and concepts](../theory.md). For plain-language definitions of the terms,
see the [Glossary](../glossary.md).

---

## 1. The forecast object

A forecast value is never a bare number. It is a number attached to four things:

$$(s,\; T,\; V,\; \mathcal{O})$$

| Symbol | Name | Meaning |
|---|---|---|
| $s$ | **source** | provider *and model*. `open_meteo_ecmwf_ifs025` and `open_meteo_gfs_seamless` are different sources, not one "Open-Meteo". |
| $T$ | **issue time** | when the forecast was retrieved (`fetched_at`). The information boundary: a forecast issued at $T$ is a function of data available at $T$ and nothing later. |
| $V$ | **valid time** | the instant or interval the forecast is about. |
| $\mathcal{O}$ | **temporal operator** | what the number asserts about $V$ — an instantaneous level, an interval mean, an interval max, or an accumulation. |

Lead is derived, never stored:

$$\ell = V - T$$

expressed in **hours** everywhere downstream, including for daily products (the
daily bucket edges are multiplied by 24 rather than the units being switched).
Lead is the dominant covariate in this system: essentially every parameter is
fitted per lead bucket, because a source that is excellent at 3 hours may be
mediocre at 7 days.

!!! warning "Lead is always recomputed"
    Any stored `horizon_hours` column is ignored. In the reference archive that
    column, `fetched_at_unix`, and `run_cycle` are all NULL. A stored derived
    quantity is a liability.

---

## 2. Indexing

Throughout:

| Symbol | Ranges over |
|---|---|
| $i, j$ | sources, $i = 1,\dots,k$ |
| $t$ | rows of a supervised matrix — one $(s\text{-set}, T, V)$ triple |
| $b$ | lead buckets |
| $\tau$ | quantile levels in $(0,1)$ |
| $y_t$ | truth at row $t$ |
| $x_{ti}$ | source $i$'s value at row $t$ |
| $a_{ti} \in \{0,1\}$ | availability — whether $x_{ti}$ is finite |
| $\hat{y}_t$ | the blended point forecast |
| $e_{ti} = y_t - x_{ti}$ | source $i$'s error |

Sources are **ragged by construction**. One provider publishes to 24 h, another
to 360 h; a 6-hourly provider is stale at some snapshots. So $a_{ti}$ is not a
nuisance to be imputed away — it appears explicitly in every weighting formula,
and methods are judged partly on how gracefully they handle it.

### Lead buckets

`leads.py` defines three grids. Buckets are **left-closed, right-open**.

=== "Hourly"

    | Label | $\ell$ (hours) |
    |---|---|
    | `0-1h` | $[0, 1)$ |
    | `1-3h` | $[1, 3)$ |
    | `3-6h` | $[3, 6)$ |
    | `6-12h` | $[6, 12)$ |
    | `12-24h` | $[12, 24)$ |
    | `24-48h` | $[24, 48)$ |
    | `48-96h` | $[48, 96)$ |
    | `96-168h` | $[96, 168)$ |
    | `168-240h` | $[168, 240)$ |
    | `240h+` | $[240, \infty)$ |

=== "Daily"

    | Label | $\ell$ (days) |
    |---|---|
    | `D1` | $[0, 2)$ |
    | `D2` | $[2, 3)$ |
    | `D3-4` | $[3, 5)$ |
    | `D5-7` | $[5, 8)$ |
    | `D8-10` | $[8, 11)$ |

    `DAILY_BUCKETS_HOURS` is the same grid scaled by 24, because everything
    downstream assumes hours.

=== "Minutely"

    | Label | $\ell$ |
    |---|---|
    | `0-5m` | $[0, 5/60)$ |
    | `5-15m` | $[5/60, 1/4)$ |
    | `15-30m` | $[1/4, 1/2)$ |
    | `30-45m` | $[1/2, 3/4)$ |
    | `45-60m` | $[3/4, 61/60)$ |

    The upper edge is $61/60$ h, not $1$: minute 60 sits at lead exactly $1.0$ and
    must stay in the minutely product rather than falling into the hourly `1-3h`
    bucket.

---

## 3. Column naming as a type system

*Implemented in: `contracts.py`, `COLUMN_SEPARATOR = "__"`*

A matrix column's prefix declares what it is and, critically, whether a blender
may see it:

| Prefix | Contents | Visible to a blender? |
|---|---|---|
| `fx__{source}__{variable}` | hourly forecast value | yes |
| `fxd__{source}__{variable}` | daily forecast value | yes |
| `age__{source}` | staleness of that source at issue time | yes |
| `obs__{variable}` | causally-QC'd station reading at issue time | yes |
| `ens__{model}__{variable}__{stat}` | ensemble mean / sd / percentiles | yes (as a *feature*, never as a source) |
| `ewagg__{variable}` | equal-weight aggregate of the blended hourly path | yes (daily matrices) |
| `path__{source}__{max,min}` | per-source hourly-path extreme | yes (daily heads) |
| `t__{variable}[__{semantics}]` | **truth** | **never** |

`ForecastMatrix.__post_init__` raises `ContractViolationError` if any feature
column begins with `t__`. This is leakage defence #4, and it is a structural
invariant rather than a review convention — the object cannot be constructed in
the illegal state.

The `ens__*` columns deserve one note. Real ensemble members are ingested as
*features*, not as additional sources, so the effective-ensemble-size accounting
of the blend ($k_{\text{eff}}$, see [combination](combination.md#2-the-diversification-ceiling))
stays honest. Adding 51 ECMWF members as 51 "sources" would make the blend look
enormously diversified while adding almost no independent information.

---

## 4. Truth and its operators

*Implemented in: `dataset/truth.py`*

For state variables the temporal operator is genuinely ambiguous, so **both**
truths are materialized and the choice is measured, not assumed
([ADR 0003](../adr/0003-empirical-truth-semantics-calibration.md)):

$$
t^{\text{inst}}_V = \operatorname{mean}\{y_u : |u - V| \le 5\text{ min}\},
\qquad
t^{\text{mean}}_V = \operatorname{mean}\{y_u : u \in [V, V{+}1\text{h})\}
$$

with the instantaneous window widening to $\pm 10$ min on failure, and the
interval mean requiring $\ge 80\%$ minute coverage. Variables whose operator is
unambiguous get one definition:

| Variable | Operator |
|---|---|
| gust | $\max$ over the hour |
| precipitation | $\sum$ over $[V, V{+}1\text{h})$ |
| PoP | $\mathbb{1}\{\text{precip} \ge 0.254\text{ mm}\}$ |
| daily hi/lo | $\max/\min$ over the **local** calendar day |

"Local day" means the station timezone's calendar day, with DST handled by
computing coverage against the day's *actual* length (1380, 1440, or 1500
minutes) rather than assuming 1440.

---

## 5. Estimands and losses

The system's promotion metric is **MAE**, and this choice propagates further than
it might appear.

$$\text{MAE} = \frac{1}{n}\sum_t |y_t - \hat{y}_t|
\qquad
\text{RMSE} = \sqrt{\frac{1}{n}\sum_t (y_t - \hat{y}_t)^2}
\qquad
\text{bias} = \frac{1}{n}\sum_t (\hat{y}_t - y_t)$$

MAE is minimized by the conditional **median**; RMSE by the conditional **mean**.
So a method fitted by least squares and promoted on MAE is optimizing one
functional and being judged on another. Where the codebase does this it is called
out on the relevant page — `grounded_median_equal_weight` exists precisely to
offer the MAE-consistent alternative to the mean-intercept fit, and the
[anchoring τ search](combination.md#7-anchoring) is flagged as an acknowledged
inconsistency because it minimizes MSE.

`bias` is reported as its own column, not folded into MAE, because a method can
have unremarkable MAE while being systematically and *correctably* wrong. That
column is what caught the grounding defect recorded in
[ADR 0004](../adr/0004-grounding-defaults-to-bias-only.md).

Probabilistic estimands (CRPS, pinball, Brier, PIT, coverage) and the skill and
significance machinery are defined in [verification](verification.md).

---

## 6. Abstention is a first-class outcome

A recurring pattern, worth stating once here rather than repeating on ten pages:
**a method that cannot fit says so, rather than fitting badly.**

Nearly every fitted head reports a `fit_status`:

| Status | Meaning |
|---|---|
| `unfitted` | never given training data |
| `insufficient_rows` | below the method's `_MIN_FIT_ROWS` |
| `insufficient_wet_rows` | precipitation head with too few wet cases |
| `gaussian_fallback` | the intended likelihood was degenerate; a fallback objective was used |
| `converged` / `fit` | the optimizer succeeded |

An abstaining method returns `NaN`, or degrades to a named base, and the
leaderboard scores it on the cases it actually produced. It does not silently
emit a fit from 12 rows. This is why thin slices show missing methods rather than
noisy ones, and why `n` is printed beside every leaderboard row.
