# Limitations, and what the harness caught

A forecasting system that only reports its wins is a marketing artifact. This
page is the other half: what this system cannot currently do, what it got wrong
before the evaluation harness caught it, and what would change its conclusions.

Read this page before you trust any number produced by this repository.

---

## 1. The binding constraint: archive age

**Every method here needs months of stored forecast *vintages*. Ground truth
alone is not enough.**

You cannot reconstruct what a provider said last Tuesday about last Wednesday
unless you *recorded it* last Tuesday. A forecast archive is the only input to
this system that cannot be backfilled, bought, or recovered — and its clock
starts the day you begin polling.

A newly started live archive may contain only a handful of snapshots. That is not
enough to backtest anything, and the system says so rather than inventing a
leaderboard. A typical early-run message is:

```
hourly: 0 score rows -> data/scores/scores_hourly_live_expanding.parquet
  hourly: no rolling-origin folds. The archive spans 0.0 days of issue times
  (2026-03-22 16:19:40 .. 2026-03-22 16:28:15) but a fold needs
  initial_train_days + step_days = 97. Keep polling, or backtest --source
  synthetic against an Open-Meteo backfill.
```

**The single highest-value action available to this project is to start the
polling cron today.** Every week of delay is a week of training data that can
never be recovered. See [Advanced usage](advanced-usage.md#the-polling-cron) for
the recommended cadence.

## 2. The station has holes

The `qc` command will show you this, and you should look. The sample station
database contains 76,507 samples spanning 2025-06-14 → 2026-07-13, but:

- **Scoreable hours cluster in two islands**: June–July 2025 and March–May 2026.
  There is a ~7-month outage in between.
- Only ~1,700 of ~9,400 calendar hours (~18%) have an instantaneous truth value.
- Pressure is missing for ~32% of samples; other channels ~4%.

This is not a bug in omni-forecast — it is what the station recorded — but it has
teeth. It means the **expanding** training window is seasonally unrepresentative
(all-summer training, spring testing), which is precisely the condition that
broke grounding (§4.1). It also means the **rolling** window is often *empty*
(a 180-day window ending in March 2026 covers only the outage), so the
rolling/expanding comparison the design promises is currently not informative on
this data.

---

## 3. What the synthetic backfill can and cannot tell you

The Open-Meteo Previous Runs backfill is the cold-start escape hatch. It can
produce a substantial local exploratory dataset, but no backfilled parquet or
score artifact is shipped with this repository. Any concrete leaderboard result
must therefore be regenerated against the operator's own station record and
configuration; numbers shown below are historical illustrations, not packaged or
production evidence.

But its limits are structural, not incidental:

| Limitation | Consequence |
|---|---|
| **Open-data NWP models only** (ECMWF, GFS, ICON via Open-Meteo). | The leaderboard says **nothing** about Google, Tomorrow.io, AccuWeather-style blends, NWS's human-edited grids, or MET Norway. These are exactly the sources most likely to be *diverse*, and they can only be acquired live. |
| **Leads are exact 24-hour multiples** (offsets `previous_day1..7`). | Only the buckets ≥ 24 h are populated. The 0–24 h buckets — where the product actually lives, and where anchoring earns its keep — are **entirely unevaluated** on this data. |
| **Day-0 is deliberately excluded.** | The unsuffixed Previous Runs field is the *latest* run for a past hour, whose effective lead is near zero. Including it would have filled the short-lead buckets with what is essentially an analysis and made anchoring look miraculous. Excluding it is honest, and it is why the short-lead gap above cannot be papered over. |
| **Synthetic and live are never pooled** (`MixedProvenanceError`). | Correct, but it means you currently have *no* leaderboard for the providers you actually serve from. |

The visible symptom in the leaderboard: `anchored_grounded_equal_weight` and
`grounded_equal_weight` have **identical MAE to three decimals**. That is not a
bug — with no lead under 3 hours, anchoring finds no anchor row, the τ grid search
selects "no anchoring", and the wrapper degrades exactly to its base. The
mechanism works; the data simply cannot exercise it.

---

## 4. Three things the harness caught

These are the reason the evaluation infrastructure exists, and they are more
valuable than any of the code they corrected.

### 4.1 Grounding was making forecasts *worse*

The core thesis of the whole project is *grounding ≫ everything else*. The first
real leaderboard said the opposite: `grounded_equal_weight` was **losing to an
uncorrected equal-weight blend** at 48–168 h leads, and its `bias` column read
**+1.2 to +1.4 °C** — the correction was *injecting* bias.

The cause, traced by inspecting the fitted coefficients fold by fold:

- Least squares regressing truth on a *noisy* forecast yields a slope below 1
  (regression dilution). Fitted slopes came out at **0.76–0.89**.
- A slope below 1 shrinks predictions **toward the training-period mean**.
- The training window was all-summer (mean 21.4 °C); the test folds were spring
  (mean 13–20 °C).
- So the "correction" was a warm tilt, and it was worst exactly where the test
  period was coldest (one fold: test mean 13.2 °C → bias **+2.10 °C**).

The fix was to default the slope to 1 — a **bias-only** correction, which is
equivariant to level shifts and *cannot* do this. It beat the free-slope variant
in every lead bucket:

| bucket | bias-only (now default) | free slope (previous default) |
|---|---|---|
| 24–48 h | MAE **1.367**, bias +0.23 | MAE 1.471, bias **+0.56** |
| 48–96 h | MAE **1.713**, bias +0.67 | MAE 2.104, bias **+1.45** |
| 96–168 h | MAE **1.775**, bias +0.26 | MAE 2.143, bias **+1.30** |
| 168–240 h | MAE **2.345**, bias −0.44 | MAE 2.368, bias **+1.21** |

Both variants remain registered (`grounded_equal_weight`, `affine_equal_weight`)
so a longer, seasonally representative archive can earn the slope back. Recorded
as [ADR 0004](adr/0004-grounding-defaults-to-bias-only.md).

**The generalizable lesson:** a correction fitted on an unrepresentative window
is not a neutral no-op — it is an *active* source of error, and the `bias` column
is what exposes it. A leaderboard that reported only MAE would have shown
grounding as merely "not helping", and the real mechanism would have gone
unnoticed.

### 4.2 The online experts could not adapt to drift

Online expert aggregation is in the method set for exactly one reason: it is
supposed to notice when a provider silently swaps its backend model, and
reallocate weight within days, with no refit.

The first implementation could not do this **at all**. On a synthetic stream where
the good and bad experts swap halfway through, both EWA and BOA finished with
weight **0.9999 on the wrong expert**.

The cause is textbook and easy to miss: both algorithms use a learning rate that
decays like `1/√t`. An expert that dominates the first half accumulates a
log-weight lead that the second half's smaller learning rates cannot overturn.
Vanilla EWA and BOA bound regret against the best *fixed* expert — not the best
*sequence* of experts. They are, by construction, unable to follow a regime
change.

The fix is **fixed share** (Herbster–Warmuth): after each update, redistribute a
small fraction of the awake mass uniformly, flooring every weight and capping the
weight ratio, so a recovering expert can climb back. Sweeping the share rate:

| share | stationary MAE ÷ best expert | drift: weight on the correct expert |
|---|---|---|
| 0.020 | 1.24–1.30 ✗ | 0.75 |
| 0.010 | 1.06–1.08 | 0.85 |
| **0.005** | **1.00–1.01** ✓ | **0.92** ✓ |
| 0.002 | 0.99 | 0.96 |

At `share = 0.005` drift adaptation is essentially **free**: the aggregators track
the regime change *and* match the best single expert on stationary data. My
initial guess of 0.02 was simply mistuned, and the sweep — not intuition — found
it.

**The generalizable lesson:** a method's headline property (here, "adaptive to
drift") is a claim, and claims need tests. Without a drift fixture, this would
have shipped as a method that was strictly worse than inverse-MSE while carrying
a docstring boasting about drift adaptation.

### 4.3 The climatology baseline was so bad it flattered everything

The climatology baseline initially posted MAE of **7.2–8.9 °C**. Since climatology
is a *floor* — the thing every real method must obviously beat — a broken floor
makes every method look good for free.

The cause: fitting an *annual* harmonic (`sin(2π·month/12)`, `cos(2π·month/12)`)
on a training window shorter than a year. Over a 3-month arc those two regressors
are nearly collinear, so unpenalized least squares fits huge, mutually cancelling
coefficients — which explode the instant they are extrapolated into a season the
window never saw.

A ridge penalty (intercept unpenalized) shrinks the seasonal terms toward zero, so
a short archive degrades gracefully to "mean plus diurnal cycle". MAE dropped to
**4.87–5.31 °C**: still bad, as climatology should be, but now a *meaningful*
floor.

---

## 5. What one historical exploratory leaderboard did and did not say

In one prior local run using 13 months of backfilled ECMWF/GFS/ICON forecasts
against one station (hourly temperature, expanding window, 45 folds), the report
looked as follows. These values are illustrative and must not be interpreted as
the repository's current evidence or as a release eligible for live serving:

| lead bucket | winner | MAE (°C) | skill vs best single provider | DM *p* |
|---|---|---|---|---|
| 24–48 h | `gbm` | 1.288 | +16.0% | 0.023 |
| 48–96 h | `gbm` | 1.572 | +16.0% | 0.031 |
| 96–168 h | `equal_weight` | 1.748 | +16.8% | <0.001 |
| 168–240 h | `inverse_mse` | 2.296 | +13.8% | 0.053 |

Read carefully, this says:

- **Blending genuinely beats the best single provider**, by 13–17%, with
  statistical significance at most leads. That is the headline, and it survives a
  Diebold–Mariano test with HAC variance.
- **The forecast-combination puzzle is real and visible.** Raw equal-weight wins
  the 96–168 h slice outright, and is within 0.03 °C of the best method on the
  n-weighted aggregate (1.770 vs 1.761). All the sophistication buys very little
  over an arithmetic mean of grounded sources — exactly as five decades of the
  combination literature predict.
- **The diversification ceiling is low.** Measured error correlations give an
  *effective* ensemble size of **1.3 independent sources out of 3** NWP models
  (ρ ≈ 0.66), and **1.8 out of 8** live providers (ρ ≈ 0.51). Equal-weighting 8
  correlated providers reduces error variance to 0.57× a single provider's — not
  the 0.12× that independence would give.

It does **not** say:

- Anything about the providers you actually serve from (§3).
- Anything about leads under 24 hours (§3) — including the anchoring stage, whose
  advantage is verified only on synthetic fixtures.
- Anything about wind, precipitation or PoP at production quality — those slices
  exist but are thin, and precipitation in particular is zero-inflated in a way
  that MAE handles badly (Brier and reliability are implemented and reported, but
  a proper occurrence/amount split is future work).

---

## 6. Statistical caveats

- **Diebold–Mariano power is limited in thin slices.** The daily buckets in
  particular have tens to hundreds of paired samples. `n` is printed beside every
  p-value precisely so a thin slice announces its own weakness. Treat `p` in
  D8–10 as advisory.
- **A per-slice winner selected on the same scores used to rank is a form of
  selection bias.** With 12 methods × 10 buckets, some winners are noise. Promotion
  now requires a common-case comparison, at least 80% coverage, and significant
  improvement over the best reference when choosing a challenger. This mitigates,
  but does not eliminate, repeated-selection bias.
  The self-verification loop — scoring what we actually served against what
  actually happened — is the intended cure, and it needs live history to work.
- **`bias` is computed on the same rows as MAE**; a method with low MAE and high
  bias is *correctable*, and that is a signal worth acting on rather than a
  contradiction.

---

## 7. Modeling gaps (deliberate, ordered)

Point-first was a deliberate decision; the following are known, sequenced, and
non-blocking:

| Gap | Status |
|---|---|
| **Calibrated distributions** (EMOS, conformal/ACI) | The `BlendResult` protocol carries quantiles and the metrics module implements CRPS/PIT/coverage *already*, so these can be added without a breaking change or a re-backtest. No wave-1 method emits them. |
| **Precipitation occurrence vs amount** | Currently one variable each for PoP and amount. The right structure is a two-stage occurrence (Brier/reliability) + amount (conditional on occurrence) model. |
| **Wind direction** | Circular; needs its own metric and blending rule. Deferred. |
| **Condition enum, UV, solar** | Deferred. The condition slug should be *derived* from blended precip/cloud at the end, never blended as an enum. |
| **Cloud, visibility** | Excluded: the station cannot verify them, so they cannot be scored, so they cannot be blended honestly. |
| **METAR cross-check of station truth** | Deferred. A PWS with a failing radiation shield can look plausible for months; an occasional sanity check against the nearest airport observation would catch it. |

## 8. Known assumptions worth challenging

- **Sea-level pressure** is *derived* from station pressure + elevation +
  temperature, because the station's `RelPress` is not actually sea-level reduced
  (it reads ≈ `AbsPress` at 1,400 m). If your station *does* reduce properly, this
  is wrong for you and the mapping should change.
- **The precipitation counter reset rule** (negative delta ⇒ reset ⇒ the new value
  *is* the accumulation) is untested against heavy rain, because the sample
  database contains almost none (max event total: 1.31 in). `rainofhourly` is
  retained as a cross-validator. Revisit when a real storm is in the archive.
- **PoP threshold** is 0.254 mm (0.01 in) — the standard "measurable" threshold,
  but a choice.
- **The 12-hour staleness cap** materially determines which sources are
  "available" at each snapshot for 6-hourly providers. It is a config knob and its
  effect on coverage should be checked against the correlation report.
- **`pyrefly` and `ty` disagree** occasionally; both are run because their
  disagreements have been useful. Neither is authoritative.

---

## 9. Operational limits

- **No daemon, no scheduler, no fetch-fresh.** `predict` is a batch command that
  reads whatever the upstream cron has written. Freshness is the cron's
  responsibility, and `predict` refuses to serve from stale data rather than
  guessing.
- **Single location.** Location is config, not a key in the model store. Adding a
  second station means a second config and a second dataset directory — no code
  change, but no shared learning either.
- **Dataset build is single-writer.** Concurrent `build-dataset` runs against the
  same dataset directory will race. Forecast-history appends themselves are locked
  and atomically replaced.

---

## 10. What would change the conclusions

In rough order of impact:

1. **Six months of live archive.** Everything above is provisional until the
   commercial providers and the 0–24 h leads are in it. This is the whole game.
2. **A seasonally complete year of station truth.** It would let the free-slope
   grounding variant be re-tested fairly — the slope may well earn its keep once
   the training window spans all four seasons, and `affine_equal_weight` is still
   on the leaderboard waiting to find out.
3. **A live archive that spans a provider's model change.** That is the event the
   online experts exist for, and it has not yet been observed in real data.
4. **Real precipitation.** Every precipitation conclusion here rests on a station
   record that has barely rained.
