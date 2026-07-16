# omni-forecast

Station-grounded blending of multi-provider weather forecasts: bias correction,
anchoring, and blending judged by a rolling-origin backtest leaderboard.

omni-forecast turns two SQLite files — a personal weather station's minute-level
observation log ([ambientweather2sqlite](https://github.com/hbmartin/ambientweather2sqlite))
and a multi-provider forecast archive
([omni-weather-forecast-apis](https://github.com/hbmartin/omni-weather-forecast-apis)) —
into three forecast products for one location:

- **next hour, by minute** — an anchored nowcast blending the live station reading into
  the hourly blend, plus native minutely precipitation where providers supply it
- **next day, by hour**
- **next 10 days, by day**

## How it works

Three composable stages. Nothing ships because it sounds good — a stage is used
for a given variable and lead time only if it wins that slice on the backtest
leaderboard.

1. **Grounding** — per-source correction toward the station, fitted per
   variable × lead bucket. Most providers repackage the same global models, so
   their *shared* bias is invisible to any weighting scheme; only correction
   removes it. A bias correction by default — the slope is opt-in, for reasons
   the data taught us (see [ADR 0004](https://hbmartin.github.io/omni-forecast/adr/0004-grounding-defaults-to-bias-only/)).
2. **Blending** — combining grounded sources: equal weight, inverse-MSE,
   gradient-boosted stacking, and online expert aggregation with sleeping
   experts (ragged provider horizons need no special casing) and fixed share
   (so a provider that silently swaps its backend model loses weight in days).
3. **Anchoring** — short-lead correction toward the latest live observation,
   decaying exponentially with lead. Your thermometer is the one input no
   provider has.

Ground truth is QC'd (plausibility bounds, spike and flatline filters) and
aggregated from minute data. Scoring uses MAE/RMSE/bias, CRPS, and
Brier/reliability for precipitation probability, with Diebold–Mariano
significance per variable × lead bucket, under strict rolling-origin splits.
Live and synthetic (backfilled) data are never pooled.

## Installation

omni-forecast requires Python 3.13 or newer. Install the command in an isolated
environment with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install omni-forecast
omni-forecast --version
```

## Usage

Download the [example configuration](https://github.com/hbmartin/omni-forecast/blob/main/config.example.toml),
save it as `config.toml`, and point it at your two SQLite files, coordinates,
and elevation:

```bash
curl -L https://raw.githubusercontent.com/hbmartin/omni-forecast/main/config.example.toml \
  -o config.toml
```

```bash
# 1. Inspect the station truth: per-channel bounds/spike/flatline flag counts
#    and hourly/daily coverage after QC.
omni-forecast qc

# 2. Materialize truth tables, canonical long frames, and the supervised
#    hourly/daily matrices as parquet + manifest.json under [dataset].dir.
omni-forecast build-dataset

# 3. Optional cold start. A forecast archive is only useful once it holds months
#    of stored *vintages*, so a new one can say nothing yet. Open-Meteo's
#    Previous Runs API backfills real archived forecasts (leads of exactly 1-7
#    days) for open NWP models, tagged `synthetic` and never pooled with live.
omni-forecast backfill --end 2026-07-12   # --models, --chunk-days

# 4. Study whether each hourly variable should use instantaneous or interval-mean
#    truth. Misalignment masquerades as provider bias; this measures it.
omni-forecast alignment

# 5. Rolling-origin backtest. Identified evaluation runs land in
#    [dataset].dir/scores without overwriting other windows/runs.
omni-forecast backtest --source live       # or --source synthetic
#   --methods all|<ids>  --products hourly,daily  --window expanding|rolling
#   --hourly-variables ...  --daily-variables ...  --semantics auto|inst|mean

# 6. Leaderboards (per-slice skill with Diebold-Mariano, aggregate, winners,
#    absolute error, consumer %-within-3F), the provider error-correlation
#    matrix, and self-verification of forecasts this system actually served.
omni-forecast report

# 7. Emit the current blended forecast (minutely + hourly + daily) as JSON.
#    Every emitted forecast carries ready/degraded status and release identity,
#    and is appended atomically to a history so it can later be scored
#    against the truth that arrives — backtest skill is an estimate, this is the
#    measurement.
omni-forecast predict                      # to stdout
omni-forecast predict --out forecast.json
#   --method auto|<id>   --now <iso>   --no-history   --semantics ...
```

Every command takes `--config <path>` (default `config.toml`).

## Status

Alpha, and honest about it: with a young forecast archive the backtest reports
that it has no folds rather than inventing a leaderboard, and `predict` refuses
to serve from stale provider data rather than guessing.

## Documentation

- **[Getting started](https://hbmartin.github.io/omni-forecast/getting-started/)** — install, configure, first forecast
- **[Advanced usage](https://hbmartin.github.io/omni-forecast/advanced-usage/)** — backfilling, tuning, reading the
  leaderboard, adding your own blending method
- **[Theory and concepts](https://hbmartin.github.io/omni-forecast/theory/)** — why grounding beats weighting, what the
  forecast-combination puzzle costs you, and how the evaluation is kept honest
- **[Architecture](https://hbmartin.github.io/omni-forecast/architecture/)** — layers, contracts, storage, libraries,
  leakage defences
- **[Limitations](https://hbmartin.github.io/omni-forecast/limitations/)** — what this cannot do, and the three real
  bugs the evaluation harness caught. **Read before trusting any number.**
- [`CONTEXT.md`](https://github.com/hbmartin/omni-forecast/blob/main/CONTEXT.md) — project glossary (issue time, valid time, lead,
  grounding, anchoring, …)
- [`docs/adr/`](https://github.com/hbmartin/omni-forecast/tree/main/docs/adr) — architecture decision records

## Development

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
uv run ruff check src --fix && uv run ruff format src tests
uvx --from semgrep==1.170.0 semgrep scan --test --config semgrep/provider-qc.yml semgrep/tests/provider_qc_grouping.py
uvx --from semgrep==1.170.0 semgrep scan --metrics=off --error --config semgrep/provider-qc.yml src/omni_forecast/dataset/matrix.py
uv run pyrefly check src && uv run ty check src
uv run lizard -Eduplicate -C 27 src
uv run pytest tests/ --cov=src --cov-report=term-missing
```

See the [release guide](https://github.com/hbmartin/omni-forecast/blob/main/docs/releasing.md)
for the TestPyPI and PyPI trusted publishing setup and checklist.

## License

Apache-2.0
