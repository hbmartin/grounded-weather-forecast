# Verification and the evaluation protocol

A forecasting system that cannot honestly measure itself is a random number
generator with good marketing. This page specifies the scores, the significance
machinery, and the protocol that makes both meaningful. The *decision* procedure
built on top — which method actually gets promoted — is
[model selection](model-selection.md).

---

## 1. Point scores

*Implemented in: `metrics/deterministic.py`*

$$\text{MAE} = \frac{1}{n}\sum_t |y_t - \hat{y}_t|
\qquad
\text{RMSE} = \sqrt{\frac{1}{n}\sum_t (y_t - \hat{y}_t)^2}
\qquad
\text{bias} = \frac{1}{n}\sum_t (\hat{y}_t - y_t)$$

MAE is the promotion metric. Bias gets its own column rather than being folded
into MAE, because a method can have unremarkable MAE while being systematically
and *correctably* wrong — that column is what caught the defect recorded in
[ADR 0004](../adr/0004-grounding-defaults-to-bias-only.md).

Note the estimand mismatch this creates: MAE is minimized by the conditional
median, RMSE by the conditional mean. Methods fitted by least squares and
promoted on MAE are optimizing one functional and judged on another. See
[notation §5](notation.md#5-estimands-and-losses).

---

## 2. Probabilistic scores

*Implemented in: `metrics/probabilistic.py` — the only module that touches `scoringrules`*

### Pinball loss

$$\rho_\tau(y, q) = \max\bigl(\tau(y - q),\ (\tau - 1)(y - q)\bigr)$$

The proper scoring rule for a single quantile; minimized in expectation by the
true $\tau$-quantile.

### CRPS

The continuous ranked probability score is the integral of the Brier score over
all thresholds, and equivalently $2\int_0^1 \rho_\tau\,d\tau$. The leaderboard
reports the **energy form for an ensemble**, evaluated on the sorted quantile
grid treated as members (`crps_ensemble`, via `scoringrules`).

A quantile-grid approximation also exists, and it is worth being precise about
because a common shorthand gets it wrong. `crps_from_quantiles` is **not** twice
the mean pinball loss — it is a weighted rectangle rule over probability levels:

$$\widehat{\text{CRPS}} = 2\sum_i w_i\,\overline{\rho_{\tau_i}},
\qquad
w_i = \frac{\tau_{i+1} - \tau_{i-1}}{2}$$

with the first and last weights closed against the boundaries $0$ and $1$:
$w_1 = (\tau_1 + \tau_2)/2 - 0$ and $w_m = 1 - (\tau_{m-1} + \tau_m)/2$. The two
coincide only for an equally-spaced grid; the board reports the energy form.
Levels must be strictly increasing within $(0,1)$ or the function raises.

### Brier score and reliability

$$\text{BS} = \frac{1}{n}\sum_t (p_t - o_t)^2$$

for PoP against binary occurrence. `reliability_bins` produces the calibration
table — forecast probability versus observed frequency over 10 bins — because a
probability scored only by MAE is a probability nobody is checking the
calibration of.

### PIT and coverage

$$\text{PIT}_t = \widehat{F}_t(y_t)$$

computed by interpolating truth into the quantile grid (clipped to $[0,1]$ at the
ends). A calibrated forecast has uniform PIT; the histogram's shape names the
failure — U-shaped means under-dispersed, hump-shaped over-dispersed, sloped
means biased. `empirical_coverage` reports the fraction of truths inside a
declared interval, checked against nominal.

---

## 3. Skill

Skill is always relative, and always relative to a **named** reference:

$$\text{skill} = 1 - \frac{\text{MAE}_{\text{method}}}{\text{MAE}_{\text{reference}}}$$

The default references are three:

| Reference | What beating it proves |
|---|---|
| `best_provider` | you beat the best single API, chosen per bucket by training MAE |
| `equal_weight` | the **raw**, ungrounded arithmetic mean — beating it proves *grounding* works |
| `damped_grounded_equal_weight` | grounded, climatology-damped equal weight — the strong baseline |

Three, not one, because each answers a different question, and "we beat the worst
provider" is not a claim worth making. Per-variable overrides live in
`[promotion.references]`; the leaderboard computes columns for the *union* of
defaults and overrides so a pinned `skill_vs_equal_weight` column never silently
changes meaning when someone edits config.

The floor every method must clear is the baseline set: `persistence` (the current
station reading held constant), `climatology` (ridge-regularized Fourier
regression of truth on month and hour), and `best_provider`.

---

## 4. Diebold–Mariano

*Implemented in: `metrics/dm.py::diebold_mariano`*

An MAE difference of 0.02 °C over 700 samples is noise. DM tests
$H_0$: equal expected loss, on the **paired** loss differentials
$d_t = \ell^A_t - \ell^B_t$.

$$\text{DM} = \frac{\bar{d}}{\sqrt{\widehat{\operatorname{Var}}(\bar{d})}} \cdot \text{HLN}(n, h)$$

**Bartlett-kernel HAC variance.** At multi-hour leads, consecutive forecast errors
are serially correlated and the naive variance is far too small:

$$\widehat{\operatorname{Var}}(\bar{d}) = \frac{1}{n}\left(\hat\gamma_0 +
2\sum_{j=1}^{\min(h-1,\,n-1)} \left(1 - \frac{j}{h}\right)\hat\gamma_j\right)$$

floored at 0 (a truncated HAC estimator can go negative).

**Harvey–Leybourne–Newbold correction**, with a Student-$t(n-1)$ reference,
because these slices have hundreds rather than millions of samples:

$$\text{HLN}(n, h) = \sqrt{\frac{n + 1 - 2h + h(h-1)/n}{n}}$$

**Guards.** `MIN_SAMPLES = 8`; $h \ge 1$ and $h < n$ required. A zero variance
with a nonzero mean yields $\pm\infty$ and $p = 0$ — the honest reading of "these
two methods differ identically on every case".

$n$ is printed next to every leaderboard row, so a thin daily slice announces its
own lack of power rather than hiding behind a p-value.

!!! warning "DM is a test, not a decision procedure"
    A leaderboard runs dozens of these every night, on overlapping data, and
    re-reads them daily. Taking the argmin and calling it the winner commits
    three errors at once — multiplicity, optional stopping, and the winner's
    curse. What DM feeds is [model selection](model-selection.md); it is no
    longer the arbiter itself.

---

## 5. Rolling-origin protocol

*Implemented in: `backtest/splits.py`, `backtest/engine.py`*

At each origin $O$:

$$\text{train} = \{t : \texttt{truth\_known\_at}_t \le O\},
\qquad
\text{test} = \{t : T_t \in (O,\ O + \text{step}]\}$$

The subtlety is entirely in the first line. It is **not** $T_t \le O$. A row
issued yesterday *about tomorrow* has an issue time in the past and a truth that
has not happened yet; training on it is leakage.

$$
\texttt{truth\_known\_at} = \begin{cases}
V + 5\text{ min}, & \text{minutely}\\
V + 2\text{ h}, & \text{hourly (the hour must end, plus ingest lag)}\\
\text{end of local day} + 1\text{ h}, & \text{daily}
\end{cases}
$$

Fold plans are computed in epoch microseconds. Both **expanding** and **rolling**
(180-day) windows are supported and reported side by side, because they answer
different questions: expanding asks *what do you know?*, rolling asks *what have
you learned lately?*

---

## 6. Leakage: assumed present until proven absent

Four defences, all executable:

1. **Fold-plan invariants** — property-based, via Hypothesis over randomized
   configurations:
   $\max(\texttt{truth\_known\_at}[\text{train}]) \le O < \min(T[\text{test}])$,
   $\text{train} \cap \text{test} = \emptyset$, rolling windows really are bounded.

2. **The poisoning sentinel.** For each fold, corrupt *every* truth value not yet
   knowable at that fold's origin (add $10^6$) and re-run the whole engine.
   Assert the fold's test predictions are **bit-identical**. If future truth
   reaches a model through *any* path — a feature, a fitted aggregate, a stateful
   blender — this test fails. It is the single most valuable test in the codebase,
   because it does not require anyone to have anticipated the leak.

3. **Fresh instances.** The registry stores *factories*, never instances; the
   engine constructs a new blender per fold. A stateful method (the online
   experts) cannot leak yesterday's weights into today's fit.

4. **Feature audit.** No column beginning `t__` may reach a blender —
   enforced in `ForecastMatrix.__post_init__`, so the illegal object cannot be
   constructed.

A fifth discipline sits in the dataset layer rather than the engine: the
**causal QC variant**. Issue-time `obs__` features are filtered by
`apply_causal_qc`, which applies bounds immediately, flags flatlines only once
their duration crosses the threshold, and **omits the isolated-spike rule
entirely** — because that rule needs a *later* sample to fire. See
[truth QC](truth-qc.md#1-station-quality-control).

---

## 7. The provenance wall

Backfilled ("synthetic") and live rows are **never pooled**. They live in
separate files, carry a `source_kind` tag, and any attempt to mix them raises
`MixedProvenanceError`. The provenance is a *filesystem* property — it is in the
matrix filename — not merely a runtime check.

This is not fastidiousness. The synthetic archive contains three open NWP models
at 24-hour-multiple leads; the live archive contains eleven commercial providers
at native cadence. A leaderboard built on the former says nothing about the
latter, and a number quietly averaging the two would be worse than no number.

Identity is tracked by three fingerprints — `dataset_fingerprint` (byte-level,
over stably-sorted parquet), `config_fingerprint`, and `code_identity` — and
serving refuses evidence that does not match the current ones. See
[ADR 0005](../adr/0005-promoted-model-releases-are-the-serving-boundary.md).

---

## 8. Self-verification

*Implemented in: `reports/verification.py`*

The backtest can only measure the backtest. Serving has its own code path,
its own snapshot construction, and its own failure modes, so the system scores
**what it actually served** against truth as truth arrives.

`verify_history` joins served forecasts to subsequently-arrived truth and
computes `live_mae`, `live_rmse`, `live_bias` per
(product, variable, truth semantics, lead bucket, method, dataset fingerprint,
release id), with `_MIN_SCORED = 5`. `compare_to_backtest` places

$$\text{mae\_gap} = \text{live\_mae} - \text{backtest\_mae}$$

(n-weighted) beside it. A persistent positive gap is serving-path drift — a bug
no backtest can find, because the backtest is not the thing that is broken.
The gap also drives the [live demotion gate](model-selection.md#6-the-live-demotion-gate).
