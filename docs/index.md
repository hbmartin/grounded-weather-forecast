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
and clears a promotion gate designed to survive being read every morning.

## The three stages

1. **Grounding** — per-source correction toward *your* thermometer, per variable ×
   lead bucket. Most providers repackage the same global models, so their shared
   bias is invisible to any weighting scheme; only correction removes it.
2. **Blending** — equal weight, inverse-MSE, gradient-boosted stacking, and online
   expert aggregation with sleeping experts and fixed share.
3. **Anchoring** — short-lead correction toward the latest live observation. Your
   station is the one input no provider has.

---

## Pick a door

<div class="grid cards" markdown>

- ### :material-play-circle: I want to use it

    ---

    Install it, point it at your two databases, get a forecast.

    **[Getting started](getting-started.md)** — install, configure, first run.
    **[Concepts](concepts.md)** — the ideas, in plain language, no equations.
    **[Glossary](glossary.md)** — every term, one line each.
    **[FAQ](faq.md)** — every error message, and what to do about it.

- ### :material-console: I want to operate it

    ---

    Run it on a schedule, read what it tells you, tune it.

    **[Advanced usage](advanced-usage.md)** — backfilling, tuning, adding a method.
    **[CLI reference](reference/cli.md)** — every command and flag.
    **[Configuration](reference/configuration.md)** — every setting.
    **[Scheduling](scheduling.md)** · **[Dashboard](dashboard.md)** ·
    **[Outputs](reference/outputs.md)** · **[Forecast JSON](reference/forecast-json.md)**

- ### :material-function-variant: I want to check the math

    ---

    Every estimator, score, and decision rule as implemented — with constants.

    **[Theory and concepts](theory.md)** — why grounding beats weighting, what the
    combination puzzle costs, how evaluation is kept honest.
    **[Methods](methods/index.md)** — ten deep-dive pages: grounding, combination,
    calibration, uncertainty, verification, model selection, precipitation, truth QC.
    **[Bibliography](methods/bibliography.md)** — every reference, matched to its
    module, plus what was deliberately declined.

</div>

!!! warning "Read this before trusting any number"
    [Limitations](limitations.md) — what this cannot do, what the archive age
    prevents, and the three real bugs the evaluation harness caught. Every measured
    figure in this documentation comes from one station over a specific window.

---

## A taste of what the harness found

Measured on 13 months of real archived forecasts against this station, on an
exploratory run in July 2026. These are historical illustrations, not packaged or
production evidence — see [Limitations §5](limitations.md#5-what-one-historical-exploratory-leaderboard-did-and-did-not-say).

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
