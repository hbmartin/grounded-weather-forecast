# Glossary

Every term the documentation uses, one line each, with a link to the page that
develops it.

The project's naming discipline is deliberate: several terms have an
"**avoid**" note because an alternative word is ambiguous in this domain, and
consistency here is what keeps the code and the docs describing the same thing.

---

## A

**Abstention**
:   A method declining to produce a forecast — returning `NaN` or degrading to a
    named base — because it lacks the data to fit. Recorded as `fit_status`, not
    hidden. → [Notation §6](methods/notation.md#6-abstention-is-a-first-class-outcome)

**Anchoring**
:   Short-lead correction of a blend toward the latest observed residual, decaying
    exponentially with lead. *Avoid: nowcasting — the minutely product is a
    nowcast; the technique is anchoring.* → [Combination §7](methods/combination.md#7-anchoring)

**Anytime-valid**
:   A test whose error guarantee holds no matter how often or when you look at it.
    Necessary because this leaderboard is regenerated nightly. →
    [Model selection §2](methods/model-selection.md#2-anytime-valid-testing-with-e-processes)

**Availability mask**
:   The per-row pattern of which sources have a usable forecast. Every blender
    renormalizes its weights over it. → [Combination §1](methods/combination.md#1-the-availability-algebra)

## B

**Bias**
:   Mean signed error — being wrong in a consistent direction. Reported as its own
    column because, unlike error generally, it is correctable. →
    [Concepts §4](concepts.md#4-bias-vs-error)

**Blender**
:   Any method implementing `fit(train) / predict(matrix) -> point (+quantiles)`,
    baselines included. *Avoid: model — overloaded with provider models.*

**Blending**
:   Combining grounded sources into one forecast via weights or a learned stacker.
    *Avoid: ensembling, averaging as method names.* → [Combination](methods/combination.md)

**BOA**
:   Bernstein Online Aggregation — an online expert algorithm with a
    second-order update. → [Combination §5](methods/combination.md#5-online-expert-aggregation)

**Brier score**
:   Mean squared error of a probability forecast against the binary outcome. The
    proper score for PoP. → [Verification §2](methods/verification.md#2-probabilistic-scores)

## C

**Calibration**
:   A probabilistic forecast whose stated probabilities match observed
    frequencies. *In this project the word is reserved for probabilistic
    calibration* — correcting a point forecast toward the station is called
    grounding. → [Calibration](methods/calibration.md)

**Conformal prediction**
:   Interval construction with finite-sample coverage guarantees requiring only
    exchangeability, not a correct model. → [Uncertainty](methods/uncertainty.md)

**Coverage**
:   The fraction of truths falling inside a stated interval; compared against its
    nominal level.

**CRPS**
:   Continuous Ranked Probability Score — the proper scoring rule for a full
    predictive distribution; reduces to absolute error for a point forecast. →
    [Verification §2](methods/verification.md#2-probabilistic-scores)

**CSGD**
:   Censored Shifted Gamma Distribution — the precipitation distribution whose
    censored mass at zero *is* the probability of a dry hour. →
    [Calibration §2](methods/calibration.md#2-csgd-censored-shifted-gamma)

## D

**Degraded forecast**
:   A forecast emitted without compatible promoted evidence. Uses fit-free
    equal weight, sets `status = "degraded"`, and records why. A valid product
    with weaker guarantees, not an implicit trained model. →
    [FAQ](faq.md#why-does-my-forecast-say-degraded)

**Diebold–Mariano (DM)**
:   Test of equal expected forecast loss on paired loss differentials. A test, not
    a decision procedure. → [Verification §4](methods/verification.md#4-dieboldmariano)

**Dressing**
:   Attaching quantiles to a point forecaster from the empirical distribution of
    its own residuals. → [Calibration §7](methods/calibration.md#7-quantile-dressing)

## E

**e-BH**
:   The e-value analogue of Benjamini–Hochberg; controls false discovery rate
    under *arbitrary* dependence. → [Model selection §3](methods/model-selection.md#3-false-discovery-rate)

**e-process / e-value**
:   A nonnegative process with expectation ≤ 1 under the null; accumulated wealth
    is evidence against it, and Ville's inequality makes reading it at any time
    valid. → [Model selection §2](methods/model-selection.md#2-anytime-valid-testing-with-e-processes)

**EMOS**
:   Ensemble Model Output Statistics — a Gaussian predictive distribution whose
    mean *and* spread are regressed on the ensemble. →
    [Calibration §1](methods/calibration.md#1-emos-nonhomogeneous-gaussian-regression)

**Evaluation run**
:   One immutable production of score rows, identified by dataset fingerprint,
    source set and kind, product, window, truth semantics, method set, code
    version, and config fingerprint.

**EWA**
:   Exponentially Weighted Average forecaster — the classic multiplicative-weights
    online expert algorithm. → [Combination §5](methods/combination.md#5-online-expert-aggregation)

## F

**Fixed share**
:   Redistributing a small fraction of expert weight uniformly each round, so a
    recovering source can climb back. Not decoration — without it the aggregators
    lock onto the wrong expert. → [Combination §5](methods/combination.md#why-fixed-share-is-load-bearing)

**Fingerprint**
:   A content hash identifying a dataset, configuration, or code version. Serving
    refuses evidence whose fingerprints do not match the live system.

**`fit_status`**
:   A fitted head's self-report: `unfitted`, `insufficient_rows`,
    `insufficient_wet_rows`, `gaussian_fallback`, `converged`, `fit`.

## G

**Grounding**
:   Per-source correction toward the station, fitted per variable × lead bucket. A
    bias correction by default; the slope is opt-in. *Avoid: calibration
    (reserved), MOS (in code).* → [Grounding](methods/grounding.md)

## I

**IDR**
:   Isotonic Distributional Regression — a nonparametric predictive distribution
    assuming only that outcomes are stochastically increasing in the forecast. →
    [Calibration §3](methods/calibration.md#3-isotonic-distributional-regression)

**Issue time**
:   *In docs:* when a forecast was made — the information boundary. *In this
    project's vocabulary,* distinguish **source retrieved at** (the provider's
    `fetched_at`), **source available at** (the collector run's `completed_at`),
    and **forecast issued at** (when this system emits a product).

## K

**$k_{\text{eff}}$**
:   Effective number of independent sources, $k/(1 + (k-1)\rho)$. Eight providers
    measured 1.8 here. → [Combination §2](methods/combination.md#2-the-diversification-ceiling)

## L

**Leakage**
:   Any path by which information unavailable at issue time reaches a model.
    Assumed present until proven absent. → [Verification §6](methods/verification.md#6-leakage-assumed-present-until-proven-absent)

**Lead**
:   Distance from forecast-issued time to valid time. Always recomputed from
    timestamps, never read from an upstream `horizon_hours` column. *Avoid:
    horizon (in code).*

**Lead bucket**
:   One of the fixed left-closed lead intervals used to stratify fitting and
    evaluation. → [Notation §2](methods/notation.md#lead-buckets)

## M

**MAE**
:   Mean Absolute Error — the promotion metric. Minimized by the conditional
    median.

**MCS**
:   Model Confidence Set — the set of methods that cannot be ruled out as best, at
    a stated level. → [Model selection §1](methods/model-selection.md#1-model-confidence-set)

**Model release**
:   A promoted mapping from product × variable × lead bucket to a method, tied to
    compatible live evidence and a training cutoff. Serving consumes releases, not
    leaderboards. → [ADR 0005](adr/0005-promoted-model-releases-are-the-serving-boundary.md)

## O

**Observation**
:   One raw station sample (~1/minute), imperial units, unvalidated. Becomes
    *truth* only after QC and normalization.

## P

**PIT**
:   Probability Integral Transform — the forecast CDF evaluated at the observed
    value. Uniform if calibrated; its histogram names the failure mode. →
    [Verification §2](methods/verification.md#2-probabilistic-scores)

**Pinball loss**
:   The proper scoring rule for a single quantile level.

**PoP**
:   Probability of Precipitation — here, the probability that hourly accumulation
    reaches 0.254 mm. → [Concepts §5](concepts.md#5-what-probability-of-precipitation-claims)

**Product**
:   An emitted forecast bundle with its own temporal contract: minutely (next 60
    minutes), hourly (next 48 hours), daily (next 10 local days).

**Provenance wall**
:   The rule that live and backfilled (synthetic) rows are never pooled, enforced
    at the filesystem level. → [Verification §7](methods/verification.md#7-the-provenance-wall)

**Provider**
:   An upstream forecast API, by slug (e.g. `open_meteo`).

## Q

**QC**
:   Quality control. On the station: bounds, spike, and flatline filters producing
    a per-minute bitmask. A flagged sample becomes null, never corrected. →
    [Truth and QC](methods/truth-qc.md)

## R

**Rolling origin**
:   The backtest protocol: repeatedly train on what was knowable at an origin and
    test on what was issued just after it. →
    [Verification §5](methods/verification.md#5-rolling-origin-protocol)

## S

**Self-verification**
:   Scoring the system's own emitted products against later truth, alongside
    providers and backtest expectations. →
    [Verification §8](methods/verification.md#8-self-verification)

**Skill**
:   Relative accuracy against a *named* reference: $1 - \text{MAE}_{\text{method}}/\text{MAE}_{\text{ref}}$.
    → [Verification §3](methods/verification.md#3-skill)

**SNHT**
:   Standard Normal Homogeneity Test — the change-point statistic used for station
    drift, most sensitive near the series end. →
    [Truth and QC §5](methods/truth-qc.md#5-change-point-statistics)

**Snapshot**
:   The as-of view of all sources at one issue time: each source's latest forecast
    at or before that moment, within the staleness cap. *Avoid: vintage.*

**Sleeping expert**
:   A source outside its horizon: absent from the round, neither updated nor
    penalized. Why ragged provider horizons need no special casing.

**Source**
:   One provider+model forecast stream — the unit a blender weighs. *Avoid: expert
    (except in the online-experts method), member.*

**Source kind**
:   Whether rows came from the live archive (`live`) or a backfill (`synthetic`).
    Never pooled silently.

**Spread**
:   Ensemble standard deviation, used as a predictor of forecast uncertainty. →
    [Uncertainty §3](methods/uncertainty.md#3-spread-and-spreadskill)

## T

**Truth**
:   The QC'd, metric-normalized, time-aggregated observation series used for
    training and scoring. Either present and trusted, or null — never imputed.

**`truth_known_at`**
:   When a row's truth became knowable. Training uses this, *not* issue time —
    the distinction is the core leakage defence. →
    [Verification §5](methods/verification.md#5-rolling-origin-protocol)

**Truth semantics**
:   Which aggregation an hourly value means: instantaneous (`inst`, ±5-min
    centered mean) or interval mean (`mean`). Measured per provider, not assumed.
    → [ADR 0003](adr/0003-empirical-truth-semantics-calibration.md)

## V

**Valid time**
:   The moment or interval a forecast is about. *Avoid: target time, forecast
    time.*

**Ville's inequality**
:   The result making e-processes anytime-valid: a nonnegative martingale exceeds
    $1/\alpha$ with probability at most $\alpha$.

## W

**Winner's curse**
:   The selected method's reported score is optimistically biased, because
    selection favours methods that got lucky. Corrected explicitly. →
    [Model selection §4](methods/model-selection.md#4-the-winners-curse)
