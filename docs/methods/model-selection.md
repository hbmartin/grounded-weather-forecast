# Model selection and promotion

This is the part of the system with the least prior documentation and the most
statistical content. It answers one question: **given a leaderboard, which method
should actually be served?**

Taking the argmin is wrong three separate ways.

1. **Multiplicity.** Forty methods against three references across ten lead
   buckets and eight variables is thousands of comparisons. At $\alpha = 0.05$,
   hundreds are "significant" by construction.
2. **Optional stopping.** The leaderboard is regenerated nightly and read every
   morning. A fixed-$\alpha$ test read repeatedly has type-I error approaching 1.
3. **The winner's curse.** The MAE of the *selected* method is a biased estimate
   of its true MAE, because selection preferentially picks methods that got
   lucky on this sample.

Each has a machine here. The output of all of it is a `ModelRelease` — the
serving boundary of [ADR 0005](../adr/0005-promoted-model-releases-are-the-serving-boundary.md).

---

## 1. Model Confidence Set

*Implemented in: `reports/mcs.py::model_confidence_set`*

Hansen, Lunde & Nason (2011). The MCS answers the right question: not "which
method is best?" but "**which methods can I not rule out as best?**" — returning a
set with the property that it contains the true best model with probability
$\ge 1 - \alpha$.

The $T_{\max}$ variant, implemented from scratch:

$$
\text{rel}_{ij} = L_{ij} - \overline{L_{i\cdot}},
\qquad
t_j = \frac{\overline{\text{rel}_{\cdot j}}}{\widehat{\text{sd}}^\ast_j},
\qquad
T_{\max} = \max_j t_j
$$

$$p = \mathbb{P}^\ast\bigl(T^\ast_{\max} \ge T_{\max}\bigr),
\qquad
p_{\text{MCS}} \leftarrow \max\bigl(p_{\text{MCS}},\ p\bigr)$$

Eliminate $\arg\max_j t_j$ while $p_{\text{MCS}} < \alpha$; the survivors are the
confidence set. $\widehat{\text{sd}}^\ast_j$ comes from bootstrap deviations, and
$p_{\text{MCS}}$ is forced **non-decreasing** along the elimination sequence,
which is what makes the sequence of tests jointly valid rather than a chain of
independent ones.

| Constant | Value |
|---|---|
| `_DEFAULT_BOOTSTRAP` | 500 |
| `_MIN_TIMES` | 8 |
| `_SEED` | 20260718 |
| `block_length` | $\lceil n^{1/3} \rfloor$ by default |

**Moving-block bootstrap.** Loss differentials are serially correlated, so an iid
bootstrap understates variance. `_block_indices` draws circular blocks (indices
wrap modulo $n_{\text{times}}$) of length $\approx n^{1/3}$ — Künsch/Politis–Romano.

**`_MIN_TIMES = 8`: below eight distinct valid times, nothing is eliminated.**
Thinness is never evidence. A three-observation slice cannot rule anything out,
and a procedure that lets it would demote standing winners on noise.

**The common-case filter.** `collapsed_loss_frame` keeps only
$(\text{issue\_time}, \text{valid\_time})$ cases that *every* method scored, then
averages absolute loss per valid time per method. Comparing methods on different
case sets is not a comparison.

!!! warning "Scope the matrix to the decision"
    The MCS is computed over the candidate **plus its references only**, not the
    whole method pool. This was a fix, and the failure it repaired is instructive:
    intersecting cases over the full pool let a *newly registered, abstaining*
    method shrink the common-case sample, which mechanically widened the
    max-statistic null and demoted four standing winners on 2026-08-04. Adding a
    method that forecasts nothing must not change the verdict on methods that do.

---

## 2. Anytime-valid testing with e-processes

*Implemented in: `reports/eprocess.py::EProcessStore.update_pair`*

The MCS is a fixed-sample procedure, and this leaderboard is read every day. The
principled fix is a **test martingale**: a nonnegative process $e_t$ with
$\mathbb{E}[e_t] \le 1$ under $H_0$, so that by **Ville's inequality**

$$\mathbb{P}\bigl(\exists t:\ e_t \ge 1/\alpha\bigr) \le \alpha$$

You may look whenever you like, stop whenever you like, and the guarantee holds.
That is exactly the property a nightly cron needs.

The wealth process is a betting scheme with ONS (online Newton step) bet sizing,
in the Waudby-Smith & Ramdas lineage:

$$
z_t = \operatorname{clip}\!\left(\frac{\ell^{\text{ref}}_t - \ell^{\text{cand}}_t}{\text{scale}_{t-1}},\ -1,\ 1\right)
$$

$$
\log e \mathrel{+}= \log(1 + \lambda_t z_t),
\qquad
g_t = \frac{z_t}{1 + \lambda_t z_t},
\qquad
A_t \mathrel{+}= g_t^2,
\qquad
\lambda_{t+1} = \operatorname{clip}\!\left(\lambda_t + \frac{c\,g_t}{A_t},\ 0,\ \tfrac{1}{2}\right)
$$

| Constant | Value |
|---|---|
| `_ONS_RATE` $c$ | $2/(2 - \ln 3) \approx 2.2226$ |
| `_LAMBDA_INIT` | 0.1 |
| `_LAMBDA_MAX` | 0.5 |
| `ons_a` init | 1.0 |

The **scale is predictable** — a running max of $|{\cdot}|$ from strictly prior
rounds — which is what keeps the normalized differential from peeking at the
current observation and breaking the martingale property. $\lambda$ is confined
to $[0, 1/2]$ so wealth can never go non-positive.

The gate fires when $\log e \ge \ln(1/\alpha)$ against **every** reference.

Two operational properties make this safe to run nightly:

- **Idempotence.** Each entry carries a `cursor_valid_time`; re-running the
  report does not re-bet on cases already wagered. A bet, once made, is never
  revised — which is precisely what anytime validity requires.
- **Reset on identity change.** The store resets on a `config_fingerprint` or
  code-version change, because a different implementation is a different null
  hypothesis. It is deliberately **not** keyed to the dataset fingerprint: new
  data arriving is the normal case, and resetting on it would destroy the
  accumulated evidence the method exists to accumulate.

---

## 3. False discovery rate

*Implemented in: `reports/leaderboard.py::bh_adjusted`, `ebh_adjusted`*

**Benjamini–Hochberg**, per reference family:

$$q_{(i)} = \min_{j \ge i}\left(\frac{p_{(j)}\,m}{j}\right)$$

clipped to $[0,1]$, with NaN passthrough. Emits `dm_q_vs_*`. Report-layer only —
BH never changes what is stored, only what is called a discovery.

**e-BH** (Wang & Ramdas 2022) is the e-value analogue, and it is the one that
composes with the e-processes above. Sort finite e-values descending and reject
the top

$$k^\ast = \max\left\{k:\ \frac{k\,e_{(k)}}{K} \ge \frac{1}{\alpha}\right\}$$

This controls FDR $\le \alpha$ under **arbitrary dependence** — which matters
enormously here, because leaderboard comparisons share cases, share references,
and are about as far from independent as tests get. BH's guarantee requires PRDS;
e-BH's requires nothing.

The board emits `e_vs_*`, `ebh_sig_vs_*`, and

$$\texttt{ebh\_threshold\_vs\_*} = \frac{K}{\alpha\,\max(k^\ast, 1)}$$

which is the wealth a pair must reach to count as a discovery — this is what
explains the otherwise confusing case of a pair that passed its Ville threshold
and still is not an FDR discovery. Untracked pairs stay null: they are not
hypotheses, and are never max-pooled into the family.

`reports/evidence.py::discovery_verdicts` records the running
`pbh_discoveries` / `ebh_discoveries` A/B with **separate denominators**, because
the two families genuinely differ in size.

---

## 4. The winner's curse

*Implemented in: `reports/winner_curse.py`*

The selected method's reported MAE is optimistically biased. Two mutually
exclusive estimands are computed, both report-layer, and both applied only where
the served row really was chosen by argmin (`gate ∈ {None, "eligibility"}`).

### Bootstrap bias

Efron–Tibshirani plug-in bias of the `min` functional, reusing the MCS
moving-block bootstrap:

$$\text{winner\_bias} = \mathbb{E}^\ast\!\left[\min_k \overline{L^\ast_k}\right] - \min_k \overline{L_k} \;\le\; 0$$

$$\texttt{mae\_debiased} = \texttt{mae} - \text{winner\_bias}$$

Unconditional: it corrects for the fact that *some* minimum was taken, not for
which method won.

### AKM hybrid

Andrews, Kitagawa & McCloskey (2024). Conditional on **which** method won —
the sharper and more delicate estimand.

Model $X = -\text{column\_means} \sim \mathcal{N}(\mu, \Sigma)$ with $\Sigma$ the
covariance of bootstrap replicate means. Conditioning on "$w$ won" is
conditioning on a polyhedron. `_truncation_bounds` derives it in closed form:
substituting $X_j = Z_j + (\Sigma_{jw}/\Sigma_{ww})X_w$ gives, for each competitor
$j$, a one-sided bound

$$\frac{\Sigma_{ww}\,z_j}{\Sigma_{ww} - \Sigma_{jw}}$$

whose *direction* flips with $\operatorname{sign}(\Sigma_{ww} - \Sigma_{jw})$ —
a competitor more variable than the winner truncates from the other side.

`_solve_mu` then bisects (80 steps) for the $\mu$ solving

$$F_{\text{TN}}\bigl(x;\ \mu,\ [\text{lo}', \text{hi}']\bigr) = \tau$$

with the truncation additionally intersected with $[\mu - \text{cap}, \mu + \text{cap}]$,
$\text{cap} = c_\beta\,\text{sd}_w$ the **projection half-width**
(`_projection_halfwidth`, 2000 SVD-method multivariate-normal draws,
$\beta = \alpha/10$). This hybridization is what keeps the estimator from
degenerating when the conditioning event is nearly deterministic — pure
conditional inference has unbounded length there.

Outputs: `mae_hybrid` (median-unbiased, $\tau = 0.5$) and `mae_hybrid_upper` at
level $(\alpha - \beta)/(1 - \beta)$.

### Guards

`near_tie_flag` fires when the two estimates disagree by more than the winner's
bootstrap SE, or when the common-case argmin disagrees with the served own-case
winner — in which case only the unconditional bias applies. (Fang & Santos: at
exact ties the bootstrap is only first-order valid.)

Deltas are applied as **labeled shifts** to the own-case MAE, never as a
recomputed number from the smaller common-case sample. Otherwise the correction
would silently change the sample the headline number refers to.

---

## 5. Promotion gates

*Implemented in: `reports/leaderboard.py::slice_winners`*

**Eligibility** — all three required, before any rule runs:

$$\text{coverage} \ge 0.8, \qquad n \ge 8, \qquad n_{\text{valid\_times}} \ge 8$$

Then one of three rules (`[promotion] rule`):

| Rule | Condition to promote |
|---|---|
| `legacy` | $\text{skill} > 0$ **and** $p_{\text{DM}} < 0.05$ against every reference |
| `mcs` | candidate **in** the MCS and every reference **out** of it |
| `seq_mcs` | e-process wealth $\ge \ln(1/\alpha)$ against every reference |

`seq_mcs` is the rule this deployment runs. `alpha = 0.1`.

Gate failures fall back to `_reference_fallback` — the best-MAE reference, with
`equal_weight` breaking exact ties. **The system serves a known baseline rather
than an unproven winner**, which is the entire point of having a gate.

`_DAILY_GATE_POOL` pools `D3-4` / `D5-7` / `D8-10` **for the gate only** —
scoring and selection stay per fine bucket — labeled `gate = "pooled_D3-10"`.
Long daily leads individually have too few valid times to clear $n \ge 8$, and
pooling for the *decision* while keeping the fine buckets for the *estimate* is
the honest compromise.

`blocked_promotions(threshold=0.15)` surfaces slices serving measurably above
their board minimum — i.e. where the gate is costing real accuracy — so the
operator can see the price of the discipline rather than only its benefits.

`reports/eprocess.py::promotion_comparison` computes **both** rules on every
report so the operator can watch them disagree, and
`evidence.gate_verdicts` records the running `gate_agree_rate`.

---

## 6. The live demotion gate

*Implemented in: `serve/selection.py::apply_live_gate`, `_live_verdict`*

Promotion is not permanent. A release whose realized error exceeds its backtest
error is demoted:

$$\text{demote if } \quad \text{live\_mae} > \texttt{live\_gap\_factor} \times \text{backtest\_mae}
\quad \text{with } n \ge \texttt{min\_live\_n}$$

`live_gap_factor = 1.5`, `min_live_n = 24`, evidence window
`_LIVE_EVIDENCE_WINDOW = 14 days`.

`_pooled_live_skill` pools eligible release cohorts with an $n$-weighted mean
(exact for MAE). This is necessary rather than convenient: a release lives about
a day, while a 24–48 h forecast's truth arrives a day or two later, so no single
release ever accumulates 24 scored cases on its own. Without pooling the gate
would never fire.

---

## 7. What all of this produces

A `ModelRelease` — an immutable record pinning method, product, variable, lead
bucket, the evaluation that justified it, and the dataset/config/code
fingerprints under which it was earned. Serving reads releases, not leaderboards;
a release whose fingerprints no longer match the live system is not used, and
`predict` reports `status="degraded"` with `equal_weight` as the fallback rather
than serving evidence it cannot vouch for.
