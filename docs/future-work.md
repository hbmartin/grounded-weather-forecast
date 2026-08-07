# Future work

_2026-07-27. Work identified by the first-live-week evaluation
(`research/evaluation-2026-07-26.md`) that is **not** part of the immediate
fix list executed that week (dataset/scheduler repair, ensemble ingestion,
backfill extension, provider key rotation, experts-L1, `crps_ensemble`,
anchor persist-then-ramp, `PerBucketFitter` shrinkage, drift-scale fix).
Each item notes its trigger — the measurement or event that should start it.
Method papers referenced here are detailed in
[research-papers-2026-07.md](research-papers-2026-07.md)._

---

## 1. Before the wet season (target: October)

The archive is summer-only and bone-dry; every precipitation pathway is
untested against real events. The July monsoon bust (truth 8.6 mm/h, wet-hour
PoP = 1.0, no provider above 0.3) is the preview.

1. **Zero-inflated precipitation stage.** Occurrence model
   (`P(precip > 0)`) scored by Brier + reliability, then a gamma/log-normal
   amount conditional on occurrence; cohere `pop` with `precip_mm` so
   `pop = 0` cannot co-occur with positive amounts. Today `registry.py`
   simply excludes the distributional heads from precip rather than owning
   it. _Trigger: before first sustained wet spell._
2. **PoP recalibration from the tipping bucket.** Online Platt scaling
   (calibeating) as the running method; beta calibration as the batch A/B;
   Venn–Abers as the validity benchmark once ~50–100 wet hours exist.
   _Trigger: with item 1._
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

6. **Daily Tmax/Tmin as first-class supervised targets.** Features: provider
   daily values + summaries of the blended hourly path; labels: realized
   station max/min. Then the two-design A/B from the papers doc: direct
   marginal EMOS (Meng & Taylor) vs extreme-of-path via ECC reordering
   (Schefzik) — an apparently novel comparison at single-station scale.
   _Trigger: ≥ 60 resolved daily folds (roughly late September)._
7. **Minutely/short-lead backtest product.** `Product.MINUTELY` is never
   backtested; the nowcast's differentiator is unmeasured. Score the anchored
   minutely path against per-minute truth with minute-granular
   `truth_known_at`, and put `minutely_tau_hours` (still a hand-set 3.0 on
   the fallback path) into a searched grid — or delete it in favour of the
   RAFT design below. _Trigger: next serving-layer work._
   _Shipped 2026-08-07 (`backtest/minutely.py`): path constructions scored
   on sub-hourly buckets, tau grid leaderboard-arbitrated, serving consults
   the promoted construction per bucket with config tau as the no-evidence
   fallback. Deferred: native precip/pop minutely scoring (needs reset-aware
   rain-counter intensity truth; only one candidate method exists anyway)._
8. **Wind product decision.** Station wind truth (sheltered siting) has mean
   0.36 m/s vs provider 2.3–4.5; the current EMOS "win" is near-constant
   calm. Either declare the product station-frame (document it), pivot the
   product to gust (truth mean 2.0, healthier provider correlation), or
   re-site the anemometer (hardware). Blending effort spent on wind speed
   before this decision is wasted. _Trigger: operator decision._

## 3. New blender registrations (leaderboard-arbitrated)

9. **`raft_grounded`** — lead-pair error-correlation trajectory adjustment
   (RAFT); the principled fix for the measured 0–1 h anchoring gap; also
   subsumes the minutely τ constant. _Trigger: after anchor persist-then-ramp
   ships and is measured; RAFT is the next rung._
10. **`seamless_regression`** — Dabernig & Atencia multimodel regression with
    model-persistence columns for expired horizons and the latest observation
    as a predictor. The strongest single-model challenger to the three-stage
    decomposition. _Trigger: ≥ 8 weeks of live folds so its coefficients are
    identified._
11. **`ondil_blend`** — incremental distributional regression with forgetting
    (location+scale in one online model). _Trigger: when EMOS shows window
    sensitivity, or the per-serve refit cost matters._
12. **`inverse_covariance`** — Ledoit–Wolf-shrunk GLS weights; the honest
    test of the diagonal-only choice. Expectation from the literature: it
    loses at this sample size; cap deviations from 1/K. _Trigger: ≥ 6 months
    live archive._
13. **Provider clustering / subset selection.** Compute k_eff from the error-
    correlation matrix in `reports/correlation.py` (nws/pirate_weather/
    visual_crossing sit at ρ ≈ 0.97); keep-one-per-cluster as a registered
    blend variant. _Trigger: cheap; next reports-layer work._
14. **Fit-time synthetic weighting.** Seasonally weighted all-history fitting
    (Lang et al.) across synthetic ∪ live with source-kind discounts —
    scoring stays live-only. Requires an ADR: it relaxes a load-bearing
    invariant deliberately and visibly. _Trigger: when grounding still trails
    equal-weight after ~8 live weeks (the seasonal-representativeness
    hypothesis will be testable by then)._
15. **Weighted conformal calibration** (Barber et al.) so thin conformal
    cells can borrow down-weighted synthetic residuals. _Trigger: coverage
    columns show cells running on the global fallback._

## 4. Data-layer work

16. **Per-variable source exclusion.** `weatherapi`'s dew point
    anti-correlates with truth (−0.42) while its temp is fine; the field
    mapping is correct — the provider's signal is genuinely bad here. There
    is no way to exclude one variable of one source; `[forecasts].sources`
    is all-or-nothing. Add `[forecasts].exclude = ["weatherapi:dew_point_c"]`
    honoured at long-frame build. _Trigger: soon; it poisons every dew blend
    it enters._
17. **Ensemble Mean API backfill.** Open-Meteo archives ensemble mean/spread
    (not members) back to ~March 2026; a one-shot ingestion would extend
    `ens__*` features months before live polling accumulates. New fetch path
    beside `ingest-ensembles`. _Trigger: when EMOS's spread link (d
    coefficient) is evaluated — more history makes that test meaningful._
18. **NBM as leaderboard benchmark row.** The NBM plugin (station KL35) is
    now collecting; once it has archive depth, surface `nbm` explicitly in
    the aggregate report as the operational-baseline row — "does the blend
    beat station-grounded NBM" is the project's most informative single
    number. _Trigger: ~4 weeks of NBM rows._
19. **Truth-QC neighbor source replacement.** Synoptic's free tier is now
    .edu-restricted; `truth-qc` cross-checks need a new source: NWS
    `api.weather.gov` observations (KSBD/KRIV/L35 METARs, keyless) and/or
    nearby public WeatherFlow Tempest stations. Same lapse-adjustment and
    consensus logic; different fetcher. _Trigger: soon — sensor-drift
    detection is currently blind._
20. **Station-pressure calibration follow-through.** After the console's
    relative-pressure offset (~+28.5 hPa) is corrected, truth history
    becomes discontinuous: either re-derive sea-level pressure from
    `AbsPress` for the full history, or version the truth series and let
    grounding re-fit across the step. _Trigger: immediately after the
    console change._

## 5. Promotion & evaluation statistics

21. **BH-FDR across slice DM tests** (stopgap) and **winner's-curse-corrected
    reporting** for promoted scores (Andrews, Kitagawa & McCloskey 2024).
    _Trigger: when the live grid has enough slices for multiplicity to bite
    (~10+ promotable slices)._
    _Winner's-curse reporting shipped 2026-08-07 (`reports/winner_curse.py`): bootstrap bias + AKM hybrid columns on every argmin-selected winner; gates unchanged._

22. **E-process sequential MCS** — anytime-valid promotion so re-running
    after every backtest refresh needs no alpha bookkeeping; replaces the
    fixed-alpha block-bootstrap MCS as folds accumulate. Also expose the MCS
    bootstrap replicates/block length (`mcs.py`, currently hard-coded)
    through `PromotionConfig`. _Trigger: scheduled retrain cadence
    established._
22a. **Consensus-tier robust scale.** The 2026-07-27 fix made the *residual*
    Page–Hinkley tier scale-aware and degenerate-safe; the fast
    consensus-deviation tier still divides by a near-zero baseline for
    zero-inflated variables (observed: statistic 8333 on a "+0.00 mm" shift).
    Apply the same MAD-floor + skip-note treatment to the consensus tier.
    _Done 2026-08-04: `CONSENSUS_SKIPPED_TIER` ships the floored robust scale._
23. **Drift-alarm automation.** Alarms currently feed nothing. Wire the fast
    consensus tier to a grounding-state down-weight and the slow
    (Page–Hinkley) tier to a state reset + operator alert, with the
    magnitudes gated by the MCS incumbent logic. _Trigger: after the
    drift-scale fix has produced one quiet month (alarm precision must be
    trusted before it acts)._
24. **Sensor-fault gating.** When every method's live bias shifts together
    on one variable, suppress anchoring/promotion for that variable (the
    cross-method invariant from the improvement plan §5 P3). _Trigger: with
    item 19 (needs a trustworthy external reference)._
    _Shipped 2026-08-07: SNHT/Pettitt drift verdict with mini-PHA
    attribution and inversion screening (`dataset/drift_stats.py`,
    `drift_verdict`), the drift report's common-mode collapse, and the
    `[truth_qc].gate_fitting` truth quarantine (default off until alarm
    precision earns trust)._

25. **Alignment-study thin-archive handling.** `_MIN_ROWS = 72` still makes
    dual-semantics variables default silently to instantaneous; hard-code the
    known-unambiguous cases and report default-vs-data-backed per variable.
    _Trigger: next dataset-layer work._

## 6. Deferred / conditional

26. **SAMOS anomaly-space grounding** — climatologies fitted from the
    provider archive (not station obs), then one pooled regression across all
    seasons/hours. The batch alternative if EWMA + harmonic grounding still
    trail after a full season. _Trigger: cross-season live evidence exists._
27. **Bayesian predictive synthesis; D-vine copula QR** — dependence-aware
    combination at archive maturity. _Trigger: ~9–12 months live._
28. **GBM containment.** GBM measured worse than climatology on daily and
    2.5–3.7 °C on hourly temp yet wins isolated pressure/pop slices on 2
    folds. Add a min-training-rows promotability floor per method, and give
    GBM a quantile head + monotone constraint on the blend feature before it
    is trusted anywhere. _Trigger: before the next promotion cycle._
29. **Performance hygiene.** `ewma_grounding.py` and `idr.py` per-row Python
    loops (O(rows × sources) per fold); vectorize when backtest wall-clock
    becomes annoying. EMOS Nelder–Mead → gradient method with multi-start.
    _Trigger: backtest > ~10 min._
30. **Provider-tier upgrades.** WeatherKit (Apple, $99/yr dev account,
    500k calls/mo) and Met Office DataHub as further diverse feeds; both have
    collector plugins already. _Trigger: operator appetite._
