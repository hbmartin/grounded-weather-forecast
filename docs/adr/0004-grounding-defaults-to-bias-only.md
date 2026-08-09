# ADR 0004: Grounding defaults to a bias-only correction, not a free-slope affine fit

## Status

Accepted. The most-cited decision in this project.

## Context

Grounding corrects each source toward the station with an affine map
$y \approx a + b\,x$, fitted per source × variable × lead bucket. The textbook
choice — and what model output statistics (MOS) has done for decades — is to fit
both parameters by least squares.

We did that first. **It made forecasts worse.**

`grounded_equal_weight` was **losing to an uncorrected equal-weight blend** at
48–168 h leads, and its `bias` column read **+1.2 to +1.4 °C**. The correction was
*injecting* bias.

The mechanism, traced by inspecting fitted coefficients fold by fold, is
**regression dilution**. Regressing truth on a *noisy* predictor gives

$$b_{\text{ols}} = \frac{\operatorname{cov}(x,y)}{\operatorname{var}(x)} < 1$$

essentially always, because the forecast carries error the truth does not. Fitted
slopes came out at **0.76–0.89**.

A slope below 1 shrinks predictions **toward the training-period mean**. Inside
the training distribution that genuinely lowers MSE — it is Stein-like, and it is
why the textbook recommends it. But it makes the correction *a function of the
training-period mean*, and the moment the evaluation period sits in a different
regime, "shrink toward the training mean" becomes a mean-dependent tilt.

The station's data made this vivid rather than subtle: truth exists only in
summer 2025 and spring 2026. The training window was all-summer (mean 21.4 °C);
the test folds were spring (mean 13–20 °C). So the "correction" was a warm tilt,
worst exactly where the test period was coldest — one fold with test mean 13.2 °C
carried a bias of **+2.10 °C**.

## Decision

**Default the slope to $b = 1$** — a pure bias correction, $a = \bar{y} - \bar{x}$,
the mean training error.

The slope is opt-in via `slope_shrinkage`, $b = 1 + \lambda(b_{\text{ols}} - 1)$,
with $\lambda = 0$ the default (`BIAS_ONLY`) and $\lambda = 1$ the free fit
(`FREE_SLOPE`).

**Both variants stay registered** — `grounded_equal_weight` and
`affine_equal_weight` — so the leaderboard, not this document, decides when a
longer and seasonally representative archive has earned the slope back.

### Why this works

A bias-only correction is **equivariant to level shifts**. Under
$y \mapsto y + c$, $x \mapsto x + c$, the fitted intercept $\bar{y} - \bar{x}$ is
unchanged. Change the regime and the correction is the same. The free-slope
correction has no such property, which is precisely the failure above.

### Measured result

Bias-only beat the free slope **in every lead bucket**:

| bucket | bias-only (now default) | free slope (previous default) |
|---|---|---|
| 24–48 h | MAE **1.367**, bias +0.23 | MAE 1.471, bias **+0.56** |
| 48–96 h | MAE **1.713**, bias +0.67 | MAE 2.104, bias **+1.45** |
| 96–168 h | MAE **1.775**, bias +0.26 | MAE 2.143, bias **+1.30** |
| 168–240 h | MAE **2.345**, bias −0.44 | MAE 2.368, bias **+1.21** |

Measured on 13 months of Open-Meteo Previous Runs backfill against the Crestline
station, July 2026. As with every number in this documentation, this is a
historical illustration from one station — see
[Limitations §4.1](../limitations.md#41-grounding-was-making-forecasts-worse).

## Consequences

**Good.**

- The correction cannot re-inject a mean-dependent tilt, which is the specific
  failure that motivated this.
- It is one parameter instead of two, so it is estimable on far thinner slices.
- The behaviour is explicable to an operator in one sentence.

**Costs.**

- Where a provider genuinely has a scale error — systematically compressing the
  diurnal range, say — bias-only cannot correct it. `affine_equal_weight` and
  `harmonic_grounded_equal_weight` remain available for the leaderboard to
  promote if that case is real.
- On a seasonally representative archive, the free slope may well win. The
  default is right for the data we have, not for all data.

**The generalizable lesson**, which is why this ADR is cited so often: *a
correction fitted on an unrepresentative window is not a neutral no-op — it is an
active source of error.* And the only reason it was caught is that `bias` is
reported as its own column. A leaderboard showing only MAE would have shown
grounding as merely "not helping", and the mechanism would have gone unnoticed.

## Related

- [Methods: grounding](../methods/grounding.md) — the estimator, the shrinkage,
  the guards, and the alternative grounding families.
- [Limitations §4.1](../limitations.md#41-grounding-was-making-forecasts-worse)
