# ADR 0003: Hourly truth semantics are calibrated empirically per provider

## Status

Accepted.

## Context

A provider publishes "14:00: 22 °C". Two readings are possible:

- **instantaneous** — the temperature *at* 14:00;
- **interval mean** — the mean temperature *across* 14:00–15:00.

**Providers do not document which they mean.** Some are inconsistent between
variables. Some have changed convention between API versions.

The size of the problem is easy to underestimate. On a clear day the temperature
ramps roughly 2 °C per hour, so the two conventions differ by about 1 °C at
mid-ramp — and that difference is *systematic*, sign-stable, and diurnally
structured.

Which makes it **indistinguishable from provider bias by inspection.**

That is the real danger. Grounding exists to measure and remove provider bias. If
a chunk of the apparent bias is actually our own bucket misalignment, grounding
will dutifully "correct" it — fitting an offset that fights a real bias with a
fake one, and doing so in a way that varies with time of day and season. The
correction would then be wrong in a manner that looks like it is working.

Hardcoding one convention is a coin flip on every provider, and getting it wrong
is worse than not correcting at all.

## Decision

**Materialize both truths and measure which one each provider tracks.**

The dataset layer produces, for every state variable:

```
t__{var}__inst   instantaneous: mean of clean samples within ±5 min of the hour
                 (widening to ±10 min on failure)
t__{var}__mean   interval mean over [H, H+1), requiring ≥80% minute coverage
```

The `alignment` command then measures, per provider × variable, which definition
that provider's forecasts actually correlate with (`_MIN_ROWS = 72`), and writes
an $n$-weighted majority recommendation to `artifacts/alignment.json`.
`--semantics auto` consumes it; `inst` is the fallback when no study exists.

**Variables with an unambiguous operator get exactly one definition** — gusts are
a max over the hour, precipitation is a sum, PoP is a threshold on that sum,
daily hi/lo are extremes over the local day. Offering a choice where there is
none would invite a meaningless measurement.

**Training never varies the target definition between providers.** The study
compares every provider, but one canonical meaning is selected per *emitted*
variable. A label that changed meaning per source would make the scores
incomparable.

## Consequences

**Good.**

- Grounding corrects real bias rather than our own alignment error.
- A provider that changes convention shows up as a change in the study, not as an
  unexplained bias shift.
- The convention is stated in every emitted document
  (`truth_semantics` per variable), so downstream consumers can compare like with
  like.

**Costs.**

- Two truth columns per state variable, roughly doubling that part of the truth
  tables.
- An extra command and artifact in the pipeline, which must be run before
  `--semantics auto` means anything.
- The `_mean` column requires ≥80% minute coverage and is therefore null more
  often than `_inst` on a station with gaps — so the choice interacts with data
  availability, not only with correctness.

**A caveat worth stating.** The study measures correlation, which identifies the
convention a provider *tracks*, not the one it *intends*. For a provider whose
forecast is poor enough, the two are indistinguishable. `_MIN_ROWS = 72` and the
$n$-weighted majority mitigate this; they do not eliminate it.

## Related

- [Methods: truth and QC §3](../methods/truth-qc.md#3-truth-semantics-calibration)
- [Forecast JSON](../reference/forecast-json.md#reading-truth_semantics) — how a
  consumer reads the convention.
