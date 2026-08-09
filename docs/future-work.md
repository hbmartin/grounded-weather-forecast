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
3. **REFS/RRFS ingestion.** The 3 km convection-allowing *ensemble* was
   rescheduled (SCN 26-47/26-48, revised 2026-07-27): real-time parallel
   NOMADS feed on or about 2026-08-11, operational 2026-10-06 12 UTC
   (retires NAM, SREF, HREF, HiresW, and NAM MOS). A REFS
   PoP/amount-spread feature is the single most relevant new signal for
   convective and orographic precip at this site. Access: AWS `noaa-rrfs`
   via Herbie, or GribStream. Start collecting from the parallel feed
   early — ~8 weeks of archive depth by the operational date means the
   feature has evidence before the wet season. _Trigger: parallel feed
   live + item 1 landed._
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

11. **NBM as leaderboard benchmark row.** _Shipped 2026-08-08_: `provider_nbm`
    single-source passthrough (registered method, not a promotion reference —
    NBM's missing humidity/pressure/daily coverage would trip the
    fail-closed reference gate) plus a blend-vs-NBM benchmark line in
    `report`. Remaining follow-on: consider per-variable
    `[promotion.references]` entries once its evaluations have depth._
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
17. **GBM containment, part two.** _Shipped 2026-08-08_: `blend_mean`
    consensus feature with a `+1` monotone constraint (objective moved
    `regression_l1` → `huber`; LightGBM forbids monotone constraints under
    leaf-renewing objectives) and the `gbm_quantile` native-quantile
    variant (19 pinball boosters, unconstrained — same LightGBM
    restriction applies to the quantile objective; containment there is the
    fit-rows floor and board arbitration)._
18. **Performance hygiene.** 2026-08-08 profiling (largest live fold,
    34,956 rows × 11 sources, all 34 hourly methods) rewrote this item's
    premise: the `ewma_grounding.py` loops were 1.5% of a fold and the EMOS
    Nelder–Mead 3.4% — measured non-issues, left alone — while the
    pure-Python PAVA in `idr.py` was 59% (26,571 calls per `idr_bucket`
    fit). Shipped same day: `pava_isotonic` now delegates to scipy's
    compiled `isotonic_regression`, and the idr predict path batches per
    bucket/grid-position (~10× on `idr_bucket`). Remaining known costs:
    `gbm` fit (17%, library-bound) and `analog_ensemble` predict (7.8%).
    The report step's own wall-clock (50–67 min in the 06:30 chain) has not
    yet been profiled — attribute before optimizing; suspects are the
    per-scores-file MCS/winner-curse bootstraps. _Trigger: report > ~30 min
    after the scores directory is pruned to steady state._
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
    follow-on — and its trigger has now fired: on 2026-08-08 serving sat
    degraded for ~4 hours (bridged silently by the publisher's
    hold-last-good guard) and the 07:45 hourly launchd fire was missed with
    nothing raising either event. Minimum viable: notify on served
    `status != ready` and on a missed hourly-publish heartbeat.
    _Trigger: first unexplained funnel loss, or a missed alarm (fired
    2026-08-08 for the alerting half)._

## 7. Added 2026-08-08 (first-month live evidence)

_Surfaced by the 2026-08-08 release-`20adc887` evidence review; numbered
continuing the sequence above._

21. **Temp far-lead bucket-inversion diagnostic + cross-bucket borrowing.**
    In release 20adc887 the 168-240h best MAE (idr_bucket, 3.84) is *worse*
    than the 240h+ best (damped, 2.81) — the week-2 provider-bias episode
    still dominates that bucket's window. Cheap first step: a
    selection-report diagnostic flagging non-monotone best-MAE across
    adjacent lead buckets. Second step: let a neighboring bucket's winner
    compete cross-bucket, in the spirit of seamless multimodel
    postprocessing across horizon boundaries (Dabernig & Atencia 2024,
    arXiv:2410.11916). Longer term, lead-time-as-covariate fits (tsEMOS:
    Jobst, Möller & Groß 2024, arXiv:2402.00555) replace hard bucket edges
    entirely. _Trigger: the inversion persists after the week-2 bias
    episode ages out of the evidence window._
22. **`rh_from_t_td` thermodynamic-coherence blender.** Humidity 168-240h
    is the worst-in-class cell (seamless_regression, MAE 13.95) while
    far-lead temp is 2.8-3.8 and dew point 4.7-5.1. Register a candidate
    that derives RH from the served temp and dew-point selections via
    Magnus inversion — propagated error should land well under 13.95, and
    it makes served RH physically consistent with served T/Td for free.
    Leaderboard-arbitrated like any registration. _Trigger: humidity
    far-lead cells still worst-in-class at the next evidence review._
23. **Dry-window low-power flag.** pop/precip MAEs of ~0.000 over the
    all-dry summer window carry no discriminative power, but the selection
    ledger records them like any other win. Stamp selections whose
    evaluation window has near-zero truth variance with a
    "non-discriminative window" marker (selection_reasons and/or evaluation
    contexts) so the wet season does not inherit false confidence from
    dry-season evidence. _Trigger: with item 1 (serve-time coherence),
    before the wet season._
