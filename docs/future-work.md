# Future work

_2026-07-27, pruned as items ship. Work identified by the first-live-week
evaluation (`research/evaluation-2026-07-26.md`) that is **not** part of the
immediate fix list executed that week (dataset/scheduler repair, ensemble
ingestion, backfill extension, provider key rotation, experts-L1,
`crps_ensemble`, anchor persist-then-ramp, `PerBucketFitter` shrinkage,
drift-scale fix). Each item notes its trigger — the measurement or event
that should start it. Method papers are cited inline;
`research/improvement-methods-2026-07.md` remains the canonical
bibliography for what is already implemented._

---

## 1. Before the wet season (target: October)

The archive is summer-only and bone-dry; every precipitation pathway is
untested against real events — including the provider-QC cross-source floors
for precipitation (2026-08: 25 mm daily / 10 mm hourly): sanity-check the
first real storm's ledger rows to confirm consensus passed and nothing
genuine was nulled. The July monsoon bust (truth 8.6 mm/h, wet-hour
PoP = 1.0, no provider above 0.3) is the preview.

1. **Serve-time pop/amount coherence.** The occurrence/amount split is now
   modeled in the backtest — `csgd_emos` fits a censored shifted-gamma whose
   censored mass is the dry probability, and `precip_sparse_shrink` handles
   the sparse daily case — but the served document can still emit `pop = 0`
   alongside a positive `precip_mm` when the two variables promote different
   methods. Add a serve-time coherence pass so occurrence and amount never
   contradict. _Trigger: before first sustained wet spell._
2. **Venn–Abers as the PoP validity benchmark.** Online Platt scaling and
   batch beta calibration shipped as `pop_platt` / `pop_beta`; the remaining
   piece is Venn–Abers once ~50–100 wet hours exist (Vovk & Petej 2014,
   arXiv:1211.0025; generalization: van der Laan & Alaa 2025,
   arXiv:2502.05676). _Trigger: with item 1._
3. **REFS/RRFS ingestion.** The 3 km convection-allowing *ensemble* becomes
   operational 2026-08-31 (HRRR continues in parallel). A REFS
   PoP/amount-spread feature is the single most relevant new signal for
   convective and orographic precip at this site. Access: AWS `noaa-rrfs`
   via Herbie, or GribStream. _Trigger: REFS operational + item 1 landed._
4. **MRMS gauge-grounding.** dynamical.org now mirrors MRMS as
   Icechunk/Zarr; a beam-blockage check for the San Bernardino rim, then
   station-gauge calibration of MRMS QPE, gives the precip nowcast a radar
   prior. _Trigger: after item 1; winter._
5. **Gauge undercatch.** WMO-SPICE transfer functions (wind- and
   temperature-dependent catch efficiency) using the co-located anemometer;
   currently explicitly deferred in `truth_qc.py`. _Trigger: first snow —
   undercatch is worst for solid precip._

## 2. Product redesigns

6. **Native minutely precip/pop scoring.** The minutely backtest scores
   anchored path constructions on sub-hourly buckets, but precip and pop are
   excluded: minute-granular intensity truth needs reset-aware rain-counter
   handling, and only one candidate method exists today. _Trigger:
   wet-season minutely evidence becomes decision-relevant._
7. **Wind product decision.** Station wind truth (sheltered siting) has mean
   0.36 m/s vs provider 2.3–4.5; the current EMOS "win" is near-constant
   calm. Either declare the product station-frame (document it), pivot the
   product to gust (truth mean 2.0, healthier provider correlation), or
   re-site the anemometer (hardware). Blending effort spent on wind speed
   before this decision is wasted. _Trigger: operator decision._

## 3. New blender registrations (leaderboard-arbitrated)

8. **`ondil_blend`** — incremental distributional regression with forgetting
   (location+scale in one online model; Hirsch, Berrisch & Ziel,
   arXiv:2407.08750; Python `ondil`). _Trigger: when EMOS shows window
   sensitivity, or the per-serve refit cost matters._
9. **Fit-time synthetic weighting.** Seasonally weighted all-history fitting
   (Lang et al. 2020, doi:10.5194/npg-27-23-2020) across synthetic ∪ live
   with source-kind discounts — scoring stays live-only. Requires an ADR: it
   relaxes a load-bearing invariant deliberately and visibly. _Trigger: when
   grounding still trails equal-weight after ~8 live weeks (the
   seasonal-representativeness hypothesis will be testable by then)._
10. **Weighted conformal calibration** (Barber, Candès, Ramdas & Tibshirani
    2023, arXiv:2202.13415) so thin conformal cells can borrow down-weighted
    synthetic residuals. _Trigger: coverage columns show cells running on
    the global fallback._

## 4. Data-layer work

11. **NBM as leaderboard benchmark row.** The NBM plugin (station KL35) is
    now collecting; once it has archive depth, surface `nbm` explicitly in
    the aggregate report as the operational-baseline row — "does the blend
    beat station-grounded NBM" is the project's most informative single
    number. _Trigger: ~4 weeks of NBM rows._
12. **Station-pressure calibration follow-through.** After the console's
    relative-pressure offset (~+28.5 hPa) is corrected, truth history
    becomes discontinuous: either re-derive sea-level pressure from
    `AbsPress` for the full history, or version the truth series and let
    grounding re-fit across the step. _Trigger: immediately after the
    console change._

## 5. Promotion & evaluation statistics

13. **Drift-alarm automation.** Alarms currently feed nothing. Wire the fast
    consensus tier to a grounding-state down-weight and the slow
    (Page–Hinkley) tier to a state reset + operator alert, with the
    magnitudes gated by the MCS incumbent logic. _Trigger: after the
    drift-scale fix has produced one quiet month (alarm precision must be
    trusted before it acts)._
14. **Alignment-study thin-archive handling.** `_MIN_ROWS = 72` still makes
    dual-semantics variables default silently to instantaneous; hard-code the
    known-unambiguous cases and report default-vs-data-backed per variable.
    _Trigger: next dataset-layer work._

## 6. Deferred / conditional

15. **SAMOS anomaly-space grounding** — climatologies fitted from the
    provider archive (not station obs), then one pooled regression across all
    seasons/hours. The batch alternative if EWMA + harmonic grounding still
    trail after a full season. _Trigger: cross-season live evidence exists._
16. **Bayesian predictive synthesis; D-vine copula QR** — dependence-aware
    combination at archive maturity (McAlinn & West 2019,
    doi:10.1016/j.jeconom.2018.11.010; Jobst, Möller & Groß 2023,
    doi:10.1002/qj.4521). _Trigger: ~9–12 months live._
17. **GBM containment, part two.** The min-training-rows promotability floor
    shipped (GBM measured worse than climatology on daily temp yet won
    isolated pressure/pop slices on 2 folds); before GBM is trusted
    anywhere, give it a quantile head and a monotone constraint on the blend
    feature. _Trigger: before the next promotion cycle._
18. **Performance hygiene.** `ewma_grounding.py` and `idr.py` per-row Python
    loops (O(rows × sources) per fold); vectorize when backtest wall-clock
    becomes annoying. EMOS Nelder–Mead → gradient method with multi-start.
    _Trigger: backtest > ~10 min._
19. **Provider-tier upgrades.** WeatherKit (Apple, $99/yr dev account,
    500k calls/mo) and Met Office DataHub as further diverse feeds; both have
    collector plugins already. _Trigger: operator appetite._
20. **Deeper build-funnel instrumentation.** The shipped funnel ledger
    (2026-08) records layer endpoints (collector → long → matrix); the
    intermediate steps — post-QC, post-exclusion, post-cap, post-coverage-gate
    row counts — need counters inside the matrix builders. Worth adding the
    first time a funnel discrepancy needs attribution beyond "lost between
    long and matrix". Push-style alerting (the freshness alarms currently
    surface only in report output and the dashboard) is a similarly cheap
    follow-on. _Trigger: first unexplained funnel loss, or a missed alarm._
