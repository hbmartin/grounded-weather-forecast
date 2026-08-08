# Truth, quality control, and drift detection

Everything in this system is scored against one backyard thermometer. If that
thermometer is wrong, every leaderboard number is wrong in a way no amount of
statistical care downstream can detect — the errors would be *consistent*.

So truth gets its own verification stack: per-sample quality control, an
aggregation ladder with coverage gates, an empirical study of what providers'
numbers even mean, a spatial cross-check against neighbours, formal change-point
tests, and a physical error model for the specific failure mode consumer stations
actually have.

The governing rule throughout: **truth is never silently adjusted.** Every
mechanism here either nulls a value or raises an alarm. None of them correct one.

---

## 1. Station quality control

*Implemented in: `dataset/qc.py::apply_qc`, `apply_causal_qc`*

Three classic failure modes, three filters, recorded as a per-minute bitmask on
each channel (`{channel}_qc`):

| Flag | Bit | Detects |
|---|---|---|
| `OUT_OF_BOUNDS` | 1 | Physically implausible values — a −60 °C reading in July. |
| `SPIKE` | 2 | An isolated excursion exceeding a per-minute rate limit against **both** neighbours **with opposite signs**. |
| `FLATLINE` | 4 | A run of bit-identical values longer than a per-channel threshold: a stuck sensor. |

The spike rule's two-sidedness is what distinguishes a radiation-shield transient
from real weather: a fast but *monotone* ramp trips only one side and is not a
spike. The rate limit scales with the actual gap between samples, so a 10-minute
gap tolerates 10 minutes' worth of change rather than one minute's.

**A flagged sample becomes `NULL`.** It is never imputed, corrected, or
interpolated, and the row is excluded from both training and scoring. Optimizing
against imputed truth optimizes against your own imputation.

### The causal variant

`apply_causal_qc` produces the `obs__` features a blender sees at issue time. It
applies bounds immediately, flags a flatline only from the instant its duration
crosses the threshold, and **omits the isolated-spike rule entirely** — because
that rule requires a *following* observation.

This is a leakage defence, not a simplification. An `obs__` feature filtered by
the two-sided spike rule would be a feature that consulted the future.

---

## 2. The aggregation ladder

*Implemented in: `dataset/truth.py`*

Minute samples aggregate up to hourly and daily truth. Per-minute truth's job in
this system is **aggregating up** — it is not a per-minute prediction target
([ADR 0002](../adr/0002-anchored-nowcast-instead-of-temporal-disaggregation.md)).

### Dual hourly semantics

For state variables the temporal operator is ambiguous, so both are materialized
and the choice is measured rather than assumed:

$$
t^{\text{inst}}_V = \operatorname{mean}\{y_u : |u - V| \le 5\text{ min}\},
\qquad
t^{\text{mean}}_V = \operatorname{mean}\{y_u : u \in [V, V{+}1\text{h})\}
$$

with the instantaneous window widening to $\pm 10$ min on failure, and the
interval mean requiring $\ge 80\%$ minute coverage (`min_hour_coverage`).

The stakes: on a clear day temperature ramps ~2 °C/hour, so choosing the wrong
convention manufactures a systematic error of ~1 °C **that looks exactly like
provider bias**. Correcting a bias that is really your own misalignment is worse
than useless — it will fight the real bias.

### Unambiguous operators

| Variable | Operator |
|---|---|
| gust | $\max$ over the hour |
| precipitation | $\sum$ over $[V, V{+}1\text{h})$ — see [precipitation §2](precipitation.md#2-truth-side-accumulation) |
| PoP | $\mathbb{1}\{\text{precip} \ge 0.254\text{ mm}\}$ |
| daily hi/lo | $\max/\min$ over the **local** calendar day |

"Local day" is the station timezone's calendar day, with DST handled by computing
coverage against the day's *actual* length — 1380, 1440, or 1500 minutes —
rather than assuming 1440. `min_day_coverage` defaults to 0.8.

### Derived quantities

Units are normalized to metric on the way in (`units.py`).

**Dew point**, Magnus approximation with Alduchov–Eskridge (1996) constants
$a = 17.625$, $b = 243.04$:

$$\gamma = \ln\frac{\text{RH}}{100} + \frac{aT}{b + T},
\qquad
T_d = \frac{b\gamma}{a - \gamma}$$

**Sea-level pressure.** The station's `RelPress` is *not* sea-level reduced — it
reads ≈ `AbsPress` at 1,400 m — so it is reduced explicitly with the
international barometric formula:

$$p_{\text{SL}} = p_{\text{station}}\left(1 - \frac{Lh}{T + Lh + 273.15}\right)^{-5.257},
\qquad L = 0.0065\ \text{K/m}$$

so that it is comparable with what providers publish as `pressure_sea`. If your
station *does* reduce properly, this is wrong for you and the column mapping
should change.

---

## 3. Truth-semantics calibration

*Implemented in: `dataset/alignment.py`*

Rather than guessing which operator each provider uses, the `alignment` command
**measures** it: per provider and variable, which of `_inst` and `_mean` that
provider's forecasts actually track. `_MIN_ROWS = 72`; the recommendation is an
$n$-weighted majority across providers, written to `artifacts/alignment.json`
and consumed by `--semantics auto`.

Recorded as [ADR 0003](../adr/0003-empirical-truth-semantics-calibration.md).

---

## 4. Neighbour cross-check

*Implemented in: `dataset/neighbors.py`*

The station cannot validate itself. The homogenization literature's framing is
**relative testing**: judge a candidate only against its spatial context, never
against forecasts — because the forecasts are the thing under test.

Neighbours come from NWS METAR (keyless) and optionally Synoptic Data (token via
`"$VARNAME"` in config). Selection is by `radius_km` (default 25) and
`elevation_band_m` (default 300), and neighbour temperatures are **lapse-adjusted**
to the station's elevation before comparison:

$$T^{\text{adj}}_{\text{neighbour}} = T_{\text{neighbour}} - \Gamma\,(h_{\text{station}} - h_{\text{neighbour}})/1000,
\qquad \Gamma = 6.5\ \text{K/km}$$

`lapse_k_per_km` is configurable because 6.5 K/km is the *standard* environmental
lapse rate, not this mountainside's — a strong inversion reverses its sign, which
is exactly the situation a Crestline station sits in on winter mornings.

---

## 5. Change-point statistics

*Implemented in: `dataset/drift_stats.py`*

Applied to station-minus-neighbour **difference series**.

**SNHT** (Alexandersson 1986) is the primary statistic — most sensitive to breaks
near the series *end*, which is exactly the monitoring case. For a standardized
series $z$ and candidate break at position $a$:

$$T_a = a\,\bar{z}_1^2 + (n - a)\,\bar{z}_2^2,
\qquad
T = \max_a T_a$$

Critical values are **Monte Carlo against this implementation** (20k replicates),
not taken from a table:

| $n$ | 7 | 10 | 14 | 21 | 30 | 45 | 60 |
|---|---|---|---|---|---|---|---|
| 95% critical | 5.5 | 6.29 | 6.93 | 7.51 | 8.05 | 8.46 | 8.72 |

The $n = 30$ value of 8.05 matches the published Khaliq & Ouarda (2007) scale.
The multi-year $T > 100$ convention does not apply at monitoring-window lengths,
and using it would make the test never fire.

**Pettitt** (1979) is the rank-based robust alternative — a Mann–Whitney
statistic maximized over split points, insensitive to the outliers that a
misbehaving sensor produces by definition.

**Craddock cusum** (1979) is the operator's corroborating visual.

**Attribution** (`attribute_break`) follows Menne & Williams' pairwise logic in
miniature: a break belongs to the series common to *all* breaking pairs. A
station drift shows coincident breaks in most station-minus-neighbour series
while neighbour-minus-neighbour pairs stay quiet; a genuine regional weather
regime breaks the neighbour network too. `_MIN_ATTRIBUTION_SERIES = 3`.

!!! note "Prewhitening is deliberately refused"
    Daily difference series are autocorrelated, which inflates end-of-series test
    statistics. The standard remedy is prewhitening. This code instead guards with
    **persistence** — consecutive evaluations above threshold — on the reasoning
    that *drifts persist and regimes revert*. Prewhitening would also remove part
    of the slow trend that a failing sensor produces, which is the signal.

**The latch.** `cli.py::_drift_verdict_block` requires 3 consecutive days of
verdict before quarantining. One noisy day is not a broken station.

Quarantine (`apply_truth_quarantine`) nulls temperature labels under a latched
drift verdict, and only when `[truth_qc] gate_fitting = true`. It is off by
default: a wrong quarantine silently deletes truth.

---

## 6. The radiation-shield error model

*Implemented in: `dataset/truth_qc.py::solar_load`, `fit_shield_error`*

Consumer weather stations have one characteristic, well-understood error: a
passively ventilated radiation shield reads warm in sun and calm air. The error
increases with solar load and decreases with ventilation — the Nakamura–Mahrt
form. 1–3 °C of spurious warmth at high sun and low wind is typical.

The station has a co-located anemometer, and top-of-atmosphere irradiance comes
from the dependency-free NOAA solar code in `solar.py`, so the signature is
directly fittable:

$$\text{residual} \sim \beta_0 + \beta_1 \cdot \frac{S}{1 + u}$$

on daytime rows (`_DAYTIME_TOA_WM2 = 50.0`, `_MIN_DAYTIME_ROWS = 100`), reporting
slope, intercept, standard error, and $n$.

Reading it:

- a **significant positive slope** means sunny-calm readings run warm relative to
  the residual baseline — a shield conversation;
- a **slope growing across refits** is the failing-shield trajectory;
- the fitted curve doubles as a correction *candidate* and as an
  observation-error inflation signal for anchoring.

**Neither is auto-applied.** Truth is never silently adjusted.

!!! note "Gauge catch efficiency is deliberately deferred"
    WMO-SPICE transfer functions (Kochendorfer et al. 2018) would correct
    under-catch of precipitation in wind. The coefficients must be transcribed
    with care, and this archive has essentially no precipitation to validate
    against. Deferred rather than guessed.

---

## 7. Pipeline-level alarms

*Implemented in: `reports/drift.py`, `reports/operations.py`, `reports/alerts.py`*

Beyond the sensor, the *pipeline* can drift.

- **Provider drift** (`reports/drift.py`) — Page–Hinkley change detection on
  per-provider error series, catching a provider that silently swapped its
  backend model. This is the failure mode
  [fixed share](combination.md#why-fixed-share-is-load-bearing) exists to absorb.
- **Baseline-relative truth alarm** (`reports/operations.py`) — fires when the
  *baselines* degrade together. If persistence and climatology both get worse at
  once, the likely explanation is not that weather became unpredictable but that
  truth changed underneath them.
- **Structural alarms** (`reports/alerts.py`) — lead contraction, coverage
  collapse, build-funnel drops, staleness against the observation-lag cap,
  and identity changes. Rendered in dashboard zones A and I.

The unifying idea: a metric that only ever compares methods to each other cannot
detect a problem that moves them all. Every alarm here is anchored to something
outside the method pool.
