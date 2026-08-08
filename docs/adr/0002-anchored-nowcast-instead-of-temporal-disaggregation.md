# ADR 0002: Anchored nowcast instead of temporal disaggregation or reconciliation

## Status

Accepted.

## Context

The system emits three products at three resolutions — minutely, hourly, daily —
and two mature literatures address exactly that situation.

**Temporal disaggregation** (Denton 1971; Chow & Lin 1971) takes a coarse series
and produces a finer one consistent with it. The obvious application: downscale
the hourly temperature forecast to minutes.

**Hierarchical reconciliation** (MinT, Wickramasuriya et al. 2019; temporal
hierarchies, Athanasopoulos et al. 2017) forces forecasts at different
aggregation levels to be mutually coherent, with a proven error-reduction
guarantee under its assumptions.

Both are well-founded and neither fits.

**Against disaggregation: a coarse forecast contains no fine-scale shape
information.** Disaggregating an hourly temperature into 60 minute values cannot
conjure structure that was never in the input; it can only impose an assumed
shape — typically a smooth interpolation. The result *looks* like a minutely
forecast and carries no minute-scale information.

But minute-scale information does exist. It is in the **live station reading**.
If the blend says 18 °C and the yard says 20 °C right now, the blend is probably
still about 2 °C low in ten minutes. That is real, exploitable, minute-resolution
signal — and it comes from observation, not from redistributing a coarse forecast.

**Against reconciliation: our daily targets are nonlinear.** MinT constrains
*linear* aggregates; it needs a summing matrix $S$ with $y = S b$. The daily
fields here are dominated by daily max temperature, daily min temperature, max
gust, and max PoP. `max` is not a sum. It has no summing matrix and sits outside
MinT's scope by construction. (Nonlinearly-constrained reconciliation exists as a
2025 preprint, carries no error-reduction guarantee, and max/min — non-
differentiable kinks — are its worst case.)

The genuinely linear daily field, precipitation sum, is already derived from the
blended hourly path and is therefore coherent by construction. Reconciliation
would have nothing to fix.

## Decision

**The minutely product is an anchored nowcast, not a disaggregation.** The
issue-time residual between observation and blend is decayed into the interpolated
hourly path, and providers' native minutely precipitation is blended directly
where it exists.

Serving does not use a single fixed formula: nine candidate path constructions
are registered and scored on minutely lead buckets, and the promoted construction
per (variable, minutely bucket) is used. The configured `minutely_tau_hours` is
the no-evidence fallback only.

**Daily extremes are their own supervised targets, not aggregates.** Features are
every provider's native daily value, equal-weight aggregates of the blended
hourly path (`ewagg__*`), and lead in days; the label is the realized max/min of
the QC'd station minutes over the local day.

**Per-minute truth's job is aggregating up** — providing the hourly and daily
labels — not being predicted per-minute by statistical fiat.

## Consequences

**Good.**

- The minutely product carries information that is actually there, and its
  accuracy at 0–5 minutes reflects a real observation rather than an assumption.
- Treating daily extremes as supervised targets sidesteps the nonlinearity
  entirely, and incidentally lets the model learn that providers' stated daily
  highs run systematically low relative to *this* thermometer.
- Because the constructions compete on a leaderboard, the crossover between
  persistence and interpolation is measured rather than asserted.

**Costs.**

- Daily max need not equal the max of the hourly block. This is intentional but
  will surprise a consumer who expects coherence, so it is documented in
  [Forecast JSON](../reference/forecast-json.md#daily).
- Anchoring needs a forecast row within 6 hours (`_ANCHOR_MAX_LEAD`). Where none
  exists — notably on the synthetic backfill — anchored methods degrade to their
  base and the minutely product loses its distinguishing input.

**Reversibility.** If precipitation-sum coherence ever proves valuable, temporal
MinT can be added downstream on the linear fields without unwinding any of this.

## Related

- [Methods: combination §8](../methods/combination.md#8-minutely-path-constructions)
- [Bibliography §2](../methods/bibliography.md#2-declined) — the declined
  references, with reasons.
