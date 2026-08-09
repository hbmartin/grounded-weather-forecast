# Research notes

Dated working memos, kept as written. They record what was measured and what was
being considered **at the time**, and they are deliberately not revised as the
system changes.

!!! warning "These are snapshots, not maintained pages"
    A claim in a memo was true when its author wrote it. Where a memo and the rest
    of the documentation disagree, **the documentation is current and the memo is
    history**. For the state of the system today, see
    [Methods](../methods/index.md) and [Limitations](../limitations.md).

    Several of these memos propose work that has since shipped, and several
    propose work that was subsequently declined — see
    [Bibliography §2](../methods/bibliography.md#2-declined).

| Memo | Date | What it is |
|---|---|---|
| [Evaluation review](evaluation-2026-07-26.md) | 2026-07-26 | Measured-state review: what the first live week actually showed, code status against the July roadmap, a methods-literature survey mapped to measured gaps, and a sequenced recommendation. Its central finding — *the binding constraint is operational, not methodological* — still holds. |
| [Improvement methods](improvement-methods-2026-07.md) | 2026-07 | Literature-driven improvement plan: adaptive diurnal bias correction (EWMA hour-of-day, Kalman, SAMOS) and fixing data starvation (Ensemble API, NBM, sub-24h backfill, neighbour truth, radar minutely precipitation). Much of the first half has shipped — see [Grounding](../methods/grounding.md). |
| [Week 2 provider diagnostic](week2-provider-diagnostic-2026-08-04.md) | 2026-08-04 | Data-driven diagnostic: horizon census, `temp_c` and `humidity_pct` error by lead bin, findings and the actions taken. |
| [Cross-cutting summary](summary.md) | — | Synthesis across the three: trade-offs between method families, pushback specific to this project, and what to decide up front. |
