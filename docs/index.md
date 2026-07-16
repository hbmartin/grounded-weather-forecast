# grounded-weather-forecast

Station-grounded blending of multi-provider weather forecasts.

grounded-weather-forecast turns two SQLite files — a personal weather station's minute-level
observation log ([ambientweather2sqlite](https://github.com/hbmartin/ambientweather2sqlite))
and a multi-provider forecast archive
([omni-weather-forecast-apis](https://github.com/hbmartin/omni-weather-forecast-apis)) —
into three forecast products for one location:

- **next hour, by minute** — an anchored nowcast
- **next day, by hour**
- **next 10 days, by day**

Nothing ships because it sounds good. A method is used for a given variable and
lead time only if it wins that slice on a rolling-origin backtest leaderboard,
with a Diebold–Mariano test to say whether the margin is real.

## The three stages

1. **Grounding** — per-source correction toward *your* thermometer, per variable ×
   lead bucket. Most providers repackage the same global models, so their shared
   bias is invisible to any weighting scheme; only correction removes it.
2. **Blending** — equal weight, inverse-MSE, gradient-boosted stacking, and online
   expert aggregation with sleeping experts and fixed share.
3. **Anchoring** — short-lead correction toward the latest live observation. Your
   station is the one input no provider has.

## Start here

<div class="grid cards" markdown>

- **[Getting started](getting-started.md)** — install, configure, and get your
  first forecast.
- **[Advanced usage](advanced-usage.md)** — backfilling a synthetic archive,
  tuning, reading the leaderboard, adding your own method.
- **[Theory and concepts](theory.md)** — the mathematics: why grounding beats
  weighting, what the forecast-combination puzzle costs you, and how the
  evaluation is kept honest.
- **[Architecture](architecture.md)** — layers, contracts, storage, libraries, and
  the leakage defences.
- **[Limitations](limitations.md)** — what this cannot do, and the three real bugs
  the evaluation harness caught. **Read this before trusting any number.**

</div>

## A taste of what the harness found

Measured on 13 months of real archived forecasts against this station:

- Blending beats the **best single provider** by 13–17% (MAE), significant at most
  leads.
- But eight providers behave like **1.8 independent ones** (mean error correlation
  0.51) — the diversification ceiling is low, and a plain arithmetic mean of
  grounded sources is within 0.03 °C of the best method overall. The
  forecast-combination puzzle is real.
- And grounding, done the textbook way with a free regression slope, was actively
  **injecting a +1.4 °C bias** — a bug the `bias` column caught and a bug that a
  leaderboard reporting only MAE would have missed. See
  [ADR 0004](adr/0004-grounding-defaults-to-bias-only.md).
