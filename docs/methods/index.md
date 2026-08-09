# Methods

Rigorous specifications of every estimator, score, and decision procedure in the
system, **as implemented** — with the constants, the guards, and the places where
the code knowingly departs from the textbook.

[Theory and concepts](../theory.md) is the narrative companion: it owns *why* and
*which*. These pages own *what exactly*. No formula appears in both.

If you want the plain-language version first, read
[Concepts](../concepts.md); if a term is unfamiliar, the
[Glossary](../glossary.md) has it.

<div class="grid cards" markdown>

- **[Notation](notation.md)** — symbols, indexing, the forecast object, column
  naming as a type system, and why abstention is a first-class outcome. **Start here.**
- **[Grounding](grounding.md)** — affine correction toward the station, regression
  dilution, slope shrinkage, empirical-Bayes pooling across lead buckets.
- **[Combination](combination.md)** — the availability algebra, the
  $k_{\text{eff}}$ ceiling, every weighting scheme, GBM stacking, online expert
  aggregation, anchoring, and the minutely path constructions.
- **[Calibration](calibration.md)** — EMOS, CSGD, isotonic distributional
  regression, PoP calibration, PIT/CQR recalibration, quantile dressing.
- **[Uncertainty](uncertainty.md)** — adaptive conformal prediction, distributional
  conformal coverage, spread–skill, cross-variable coherence.
- **[Verification](verification.md)** — proper scoring rules, Diebold–Mariano with
  HAC and HLN, the rolling-origin protocol, four leakage defences, the provenance
  wall.
- **[Model selection](model-selection.md)** — Model Confidence Set, betting
  e-processes, BH and e-BH, the winner's curse, and the promotion gates.
- **[Precipitation](precipitation.md)** — mixed discrete–continuous handling,
  reset-aware accumulation, cross-source QC guards, sparse shrinkage.
- **[Truth and QC](truth-qc.md)** — station quality control, the aggregation
  ladder, neighbour cross-checks, SNHT/Pettitt change points, the radiation-shield
  error model.
- **[Bibliography](bibliography.md)** — every reference, matched to its module,
  plus the papers deliberately *declined* and why.

</div>

---

## Three properties worth knowing before reading any of it

**Nothing here is chosen by argument.** Every method on these pages is registered,
scored on the same rolling-origin folds, and promoted or not by the machinery in
[model selection](model-selection.md). Where two approaches compete, both are
usually registered and the leaderboard arbitrates. This documentation explains
what the verdict *means*; it does not substitute for it.

**Abstention is designed for.** A method that cannot fit returns `NaN` or degrades
to a named base, and records why. Thin slices show missing methods rather than
noisy ones.

**The evidence is dated and local.** Every measured number in this documentation
comes from one station in Crestline, CA, over a specific window. Read
[Limitations](../limitations.md) before generalizing any of it.
