## The good news: your repos already solved the hard boring parts

The three reports all spend pages on forecast canonicalization, and you've essentially built it. `omni-weather-forecast-apis` stores `fetched_at`, a 6-hour `run_cycle` bucket, and `horizon_hours` on every hourly point, plus the `stacking_features` view — that's the supervised matrix keyed by (run, provider, model, valid_time, lead) that every method in all three reports trains on. Units are already normalized, and the README already declares ensemble/verification as a separate downstream package. `ambientweather2sqlite` gives 60-second truth with gap detection. Two genuinely missing pieces:

1. **Archive age is your binding constraint, not method choice.** Every method needs months of stored forecast vintages; ground truth alone isn't enough. If you haven't already, the single highest-value action is to start an hourly (or at least per-run-cycle) cron of `omni-weather --sqlite` at your station's coordinates today, with all 13 providers plus several explicit Open-Meteo models (`ecmwf_ifs025`, `gfs_seamless`, `icon_seamless`, ...). Every week of delay is a week of unrecoverable training data.
2. **The cold-start escape hatch:** Open-Meteo's [Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) serves archived forecasts at fixed lead offsets (`_previous_day0`…`_previous_day7`, most models from ~Jan 2024, GFS 2m temperature back to 2021), explicitly designed for training bias-correction with horizon as a controlled variable. So you can build and backtest the whole pipeline *now* against 2+ years of synthetic archive — but only for open-data NWP models. Your commercial providers (Google, Tomorrow.io, AccuWeather-style blends) can't be backfilled, which is another reason to start the live archive immediately.

## A framing that cuts across all three reports

"Grounded forecasts" actually bundles three separable improvements, with very different expected payoffs:

**Grounding (per-provider bias correction to your station) — the big win.** A MOS-style affine correction per variable × lead bucket fixes your microclimate offset and each provider's systematic bias. This works even with one provider and modest data. Critically, most of your 13 providers repackage the same handful of global models (GFS/ECMWF/HRRR/NBM), so their *errors are highly correlated* — a_forecast's caveat, and it means shared bias is large and reweighting among them can't remove it. Only correction can.

**Blending (combining providers) — the modest, hard-fought win.** The forecast-combination puzzle is real: equal-weight after bias correction will be embarrassingly hard to beat, and with correlated providers the diversification ceiling is low. The genuinely diverse sources in your set are roughly: raw NWP families (via Open-Meteo per-model), NWS (human-edited grids), Google (ML-heavy stack), MET Norway (ECMWF+postprocessing), Tomorrow.io. Expect the measured error-correlation matrix — which your harness should print on day one — to justify treating this as ~5 effective providers, not 13.

**Anchoring (short-lead correction toward current observations) — the free win nobody's report emphasized.** Your unique asset is a live thermometer in your yard that no provider has. At leads 0–6h, decaying the blend toward the current observed residual (obs-now minus blend-now, decayed with lead) is cheap and typically dominates everything else at short horizons. This is the useful core of g_forecast's Kalman-filter proposal, extractable without building the Kalman machinery.

## Tradeoffs between the method families

| Family                                                    | Can it fix shared bias?   | Data to work    | Dropout/drift                             | Debuggability     | Python reality                                               |
| --------------------------------------------------------- | ------------------------- | --------------- | ----------------------------------------- | ----------------- | ------------------------------------------------------------ |
| Equal/trimmed mean                                        | No                        | None            | Renormalize / immune                      | Trivial           | Trivial                                                      |
| Inverse-MSE weights per (var × lead)                      | No                        | Weeks           | Renormalize / rolling window              | High              | Trivial                                                      |
| Affine bias correction + weights (MOS)                    | **Yes**                   | Weeks–months    | Renormalize / rolling refit               | High              | Trivial (sklearn/scipy)                                      |
| EMOS-style distributional regression                      | **Yes** + calibrated PDFs | Months          | Renormalize / rolling refit               | Medium            | No package; Gaussian EMOS is ~50 lines of scipy + [scoringrules](https://pypi.org/project/scoringrules/) |
| GBM stacker (LightGBM + lead/hour/season/spread features) | **Yes**, nonlinearly      | Many months     | Native missing / retrain                  | Low–medium (SHAP) | Excellent                                                    |
| Online experts (EWA/BOA, sleeping experts)                | No (weights only)         | Days to warm up | **Native** (sleeping) / **best-in-class** | Medium            | [opera-python](https://github.com/Dralliag/opera-python) exists but young; hand-rolling EWA/BOA is ~150 lines |
| Kalman mixed-frequency state-space                        | Yes (as bias states)      | Months + tuning | Native / native                           | **Low**           | statsmodels MLEModel; most engineering effort                |

The tensions worth internalizing:

**Bias correction vs. weighting.** Given high provider correlation, expect the ordering of wins to be: grounding ≫ anchoring (short leads) > weighting > fancy weighting. Structure the pipeline so correction and combination are separate composable stages — then every combination method runs on "grounded providers" as input, and you can measure each stage's contribution honestly.

**One big model vs. many small models.** Stratified small regressions (per variable × lead bucket) are transparent and per-slice debuggable, but slices get data-thin (daily hi/lo gives you 365 samples/provider/year — one winter of daily extremes is thin). A GBM pools strength across slices with lead/hour/season as features and handles missing providers natively, but it's opaque, needs leakage-careful rolling-origin training, and will overfit a young archive. This is the classic a_forecast "few models richly featured" point — the GBM is the ceiling, the stratified regressions are the floor that tells you what the GBM should be learning.

**Offline refits vs. online updating.** Rolling weekly refits are reproducible and easy to backtest; they cover most drift. Online expert aggregation earns its keep specifically for: provider silently swapping their backend model (happens regularly in consumer APIs), providers entering/leaving (sleeping experts — your 3-day vs 16-day horizon raggedness is exactly this), and regret guarantees. Its cost: stateful, order-dependent, harder to reproduce in backtests. The Météo-France temperature papers (cited in your a_forecast doc) are the direct template. Good candidate for one of your parallel implementations because it's philosophically disjoint from the regression family — it wins in different regimes.

**Kalman state-space: elegant, but a poor first move.** It unifies mixed frequency, ragged edges, and nowcasting in one machine — and that's the problem: it's a monolith you can't A/B in stages, with covariance tuning as a permanent tax. Decomposed equivalents (bias correction + weighting + anchoring) get ~90% of it in inspectable pieces. Keep it as a later contender, not a foundation.

## Where I'd push back on the reports, for this project specifically

**Skip temporal disaggregation entirely (g_forecast's Denton/Chow-Lin section).** Nothing in your product needs statistically disaggregated per-minute temperature. Minutely data exists only as precipitation from a few providers (0–60 min); per-minute truth's job is *aggregating up* to score hourly/daily forecasts. Spline interpolation for display is a UI concern, not statistics.

**Defer MinT/thief reconciliation, maybe forever.** Reconciliation constrains *linear* aggregates. Your daily fields are mostly nonlinear (max/min temp, max gust, max PoP) — out of MinT's scope by construction — and the linear ones (precip sum) are better handled by *deriving* daily products from the blended hourly path, which is coherent by construction. Nixtla's [hierarchicalforecast](https://github.com/Nixtla/hierarchicalforecast) does temporal MinT in Python if precip-sum coherence ever proves valuable, so nothing is foreclosed. Daily hi/lo should be its own supervised target: features = blended hourly path + providers' daily values; label = realized station max/min. That sidesteps the nonlinear-reconciliation open-research-problem a_forecast flagged.

**Treat variables as different problems, sequenced by difficulty.** Temperature/dew point/humidity/pressure: dense, Gaussian-ish — every method works; use them to validate the harness. Wind: skewed, plus a physical wrinkle — your anemometer height/siting differs from the 10m forecast standard, so raw "provider error" is partly siting; grounding absorbs it automatically, which is the whole point. Precipitation: zero-inflated, tipping-bucket noise, and AW rain counters reset (hourly accumulation must be derived from monotone counters carefully) — split into occurrence (blend PoP as a probability, scored with Brier/reliability against binary station outcomes) and amount. Condition enum: don't blend enums; derive from blended precip/cloud at the end.

**Add a truth-QC layer the reports underweight.** PWS sensors have radiation-shield temp spikes, gauge undercatch, and outages (your gap detection already helps). A plausibility+spike filter, and an occasional sanity cross-check against the nearest METAR, keeps you from optimizing toward sensor faults.

## Architecture: built for parallel approaches

Since you expect to implement several approaches in parallel, the thing to freeze first is the *contract*, not any method:

```
grounded-weather-forecast/
  src/omni-forecase/
    dataset/     # ATTACH both sqlite DBs; unit conversion (station→metric);
                 # truth QC; truth aggregation; per-provider timestamp-semantics
                 # alignment; materialize supervised matrix (parquet/duckdb)
    blenders/    # one module per approach, all implementing one protocol:
                 #   fit(train) / predict(X, availability) -> point (+ quantiles)
                 # baselines.py, bias_affine.py, inverse_mse.py, emos.py,
                 # gbm_stack.py, online_experts.py, ...
    backtest/    # rolling-origin engine, split by fetch time; leakage tests
    metrics/     # MAE/RMSE/bias, CRPS/PIT/coverage, Brier/reliability,
                 # skill vs {persistence, climatology, best-provider, equal-weight}
    reports/     # leaderboard per variable × horizon bucket; error-correlation matrix
    serve/       # emit the actual grounded forecast (best method per var×lead)
```

Two design details that make the parallelism real: every blender receives an **availability mask** and must renormalize (so 3-day and 16-day providers coexist without special-casing), and the backtest engine is the *only* thing allowed to declare winners (per variable × horizon, with Diebold-Mariano checks, since a method that wins at 2h routinely loses at 7d). Once the dataset contract + protocol + metrics land as PR #1, each approach is an independent ~1-file PR — that's also the shape that lets you throw parallel Claude sessions at one method each without merge conflicts.

Two small optional upstreams in the existing repos: a `stacking_features`-style view for daily/minutely points (horizon is computable from `fetched_at`, just not materialized), and possibly a bundled "archive mode" response hook. Neither blocks the new project.

For prior art context: no OSS project I could find does "multi-API blend calibrated to a personal station" — the niche is genuinely open. The commercial analog is [ForecastAdvisor/ForecastWatch](https://www.forecastadvisor.com/docs/about/) (since 2004: 1–3 day high/low/PoP, percent-within-3°F, rolling 12 months, per city). You'd strictly dominate it for your own location: per-hour, out to 10+ days, proper scores, your actual backyard. The institutional analog is NOAA's National Blend of Models — same architecture you're converging on (bias-correct, weight by lead, anchor to obs at short lead).

## What I'd decide up front (your call, everything else can wait)

1. **Polling budget:** hourly fan-out across all providers will strain free tiers (your quota tracker will enforce this). A sane split: hourly for a diverse core (open_meteo multi-model, NWS, MET Norway, Google), 6-hourly (per run_cycle) for the rest. Worth choosing deliberately since it defines your training data forever.
2. **Point-first or probabilistic-first:** my recommendation is point blends + CRPS-ready evaluation from day one (via `scoringrules`), adding EMOS/quantiles/conformal ([MAPIE](https://mapie.readthedocs.io/en/v1.2.1/examples_regression/3-scientific-articles/plot_zaffran2022_comparison.html) has ACI/EnbPI for the drift-aware variants) as a second wave — but if calibrated intervals are a hard product requirement, EMOS moves up and the GBM moves down.
3. **What the product actually is:** accuracy leaderboard, a served JSON/dashboard forecast, home-automation feed, or all three — this mostly affects `serve/` and whether coherence between hourly and daily outputs matters cosmetically.
4. **Where it lives:** I'd make it a third repo (this session only has the two existing ones in scope, so you'd create/add it — or say the word and I'll scaffold it inside one of the existing repos' designated branches to start).

If you want my straight ranking for the first four parallel implementations after baselines: **affine grounding + equal weight** (the floor that will be hard to beat), **GBM stacker** (the ceiling), **online sleeping-experts** (disjoint failure modes, best drift story), and **Gaussian EMOS on the temperature family** (cheapest path to calibrated distributions). That set spans the design space; everything else is a variation you can add once the leaderboard exists.

Sources: [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api), [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api), [ForecastAdvisor methodology](https://www.forecastadvisor.com/docs/about/), [scoringrules](https://pypi.org/project/scoringrules/) ([repo](https://github.com/frazane/scoringrules)), [properscoring](https://github.com/properscoring/properscoring), [MAPIE time-series conformal](https://mapie.readthedocs.io/en/v0.8.1/examples_regression/4-tutorials/plot_ts-tutorial.html), [Nixtla hierarchicalforecast](https://github.com/Nixtla/hierarchicalforecast), [opera-python](https://github.com/Dralliag/opera-python), [opera (R)](https://rdrr.io/cran/opera/man/opera-package.html), [Météo-France expert aggregation for temperature](https://arxiv.org/pdf/2506.15217).
