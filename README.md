# grounded-weather-forecast

Station-grounded blending of multi-provider weather forecasts: bias correction,
anchoring, and blending judged by a rolling-origin backtest leaderboard.

grounded-weather-forecast turns two SQLite files — a personal weather station's minute-level
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
   the data taught us (see [ADR 0004](https://hbmartin.github.io/grounded-weather-forecast/adr/0004-grounding-defaults-to-bias-only/)).
2. **Blending** — combining grounded sources: equal weight, trimmed mean
   (drops the extremes per row — robustness with zero parameters),
   inverse-MSE and inverse-MAE weighting, gradient-boosted stacking, and
   online expert aggregation with sleeping experts (ragged provider horizons
   need no special casing) and fixed share (so a provider that silently swaps
   its backend model loses weight in days). Grounding also comes in a
   MAE-consistent median-intercept variant.
3. **Anchoring** — short-lead correction toward the latest live observation,
   decaying exponentially with lead. Your thermometer is the one input no
   provider has.

Ground truth is QC'd (plausibility bounds, spike and flatline filters) and
aggregated from minute data. Provider forecasts get their own conservative
QC before grounding: absolute physical bounds plus a robust cross-source
outlier pass (temperature, humidity, pressure, dew point, and — since a
provider stored a 138 mm cloudburst for a bone-dry day — precipitation,
whose generous floors null a lone hallucination but never drizzle
disagreement or genuine storm consensus). Scoring uses MAE/RMSE/bias, CRPS, and
Brier/reliability for precipitation probability, with Diebold–Mariano
significance per variable × lead bucket, under strict rolling-origin splits.
Live and synthetic (backfilled) data are never pooled.

## Installation

grounded-weather-forecast requires Python 3.13 or newer. Install the command in an isolated
environment with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install grounded-weather-forecast
grounded-weather-forecast --version
```

## Usage

Download the [example configuration](https://github.com/hbmartin/grounded-weather-forecast/blob/main/config.example.toml),
edit the two database paths and your station's coordinates, then:

```bash
grounded-weather-forecast qc              # check your column mapping is right
grounded-weather-forecast build-dataset   # truth tables + supervised matrices
grounded-weather-forecast predict         # a forecast, as JSON on stdout
```

That works on day one. Everything else needs an archive with some history — see
[Getting started](https://hbmartin.github.io/grounded-weather-forecast/getting-started/).

### Commands at a glance

| Command | What it does | Flags |
|---|---|---|
| `qc` | summarize station truth quality control | — |
| `build-dataset` | materialize truth tables and supervised matrices as parquet | — |
| `backtest` | rolling-origin backtest over the supervised matrices | `--methods` `--hourly-variables` `--daily-variables` `--products` `--window` `--source` `--semantics` |
| `report` | leaderboards, correlation reports, evidence ledgers, and `reports/dashboard.html` | — |
| `alignment` | study truth semantics per provider; write the alignment artifact | — |
| `backfill` | fetch archived forecasts into the synthetic supervised matrix | `--provider` `--models` `--start` `--end` `--chunk-days` |
| `truth-qc` | cross-check truth against lapse-adjusted neighbors; fit the radiation-shield model | `--days` |
| `ingest-ensembles` | poll the Open-Meteo Ensemble API for per-model spread features | `--models` |
| `predict` | emit the current blended forecast as JSON | `--out` `--method` `--no-history` `--semantics` `--now` |
| `prune-scores` | delete superseded scores files | `--dry-run` |

Global: `--config PATH` (default `config.toml`), `--version`. Exit codes: `0` ok,
`1` command failure, `2` config error, `75` another pipeline command holds the
lock (retry later). All commands except `predict` and `ingest-ensembles`
serialize on `<dataset dir>/pipeline.lock`.

**Full semantics for every flag — defaults, choices, what each command reads and
writes — are in the
[CLI reference](https://hbmartin.github.io/grounded-weather-forecast/reference/cli/).**
Every configuration key is in the
[configuration reference](https://hbmartin.github.io/grounded-weather-forecast/reference/configuration/),
and the emitted document is specified in
[Forecast JSON](https://hbmartin.github.io/grounded-weather-forecast/reference/forecast-json/).

### Running it for real

Steady state is four crons — poll, ingest ensembles, predict, and a nightly
`build-dataset && backtest && report && truth-qc` chain. Cadence rationale and
launchd templates are in
[Scheduling](https://hbmartin.github.io/grounded-weather-forecast/scheduling/).

## Status

Alpha, and honest about it: with a young forecast archive the backtest reports
that it has no folds rather than inventing a leaderboard, and `predict` refuses
to serve from stale provider data rather than guessing.

## Documentation

Full docs: **<https://hbmartin.github.io/grounded-weather-forecast/>**

**Using it**

- **[Getting started](https://hbmartin.github.io/grounded-weather-forecast/getting-started/)** — install, configure, first forecast
- **[Concepts](https://hbmartin.github.io/grounded-weather-forecast/concepts/)** — the ideas in plain language, no equations
- **[FAQ and troubleshooting](https://hbmartin.github.io/grounded-weather-forecast/faq/)** — every error message, and what to do about it
- **[Glossary](https://hbmartin.github.io/grounded-weather-forecast/glossary/)** — every term, one line each

**Operating it**

- **[Advanced usage](https://hbmartin.github.io/grounded-weather-forecast/advanced-usage/)** — backfilling, tuning, adding your own blending method
- **[CLI](https://hbmartin.github.io/grounded-weather-forecast/reference/cli/)** · **[Configuration](https://hbmartin.github.io/grounded-weather-forecast/reference/configuration/)** · **[Forecast JSON](https://hbmartin.github.io/grounded-weather-forecast/reference/forecast-json/)** · **[Outputs](https://hbmartin.github.io/grounded-weather-forecast/reference/outputs/)** — complete references
- **[Scheduling](https://hbmartin.github.io/grounded-weather-forecast/scheduling/)** — the four crons, with launchd templates
- **[Operator dashboard](https://hbmartin.github.io/grounded-weather-forecast/dashboard/)** — the nine-zone offline console `report` writes

**The mathematics**

- **[Theory and concepts](https://hbmartin.github.io/grounded-weather-forecast/theory/)** — why grounding beats weighting, what the
  forecast-combination puzzle costs you, and how the evaluation is kept honest
- **[Methods](https://hbmartin.github.io/grounded-weather-forecast/methods/)** — ten deep-dive pages specifying every estimator, score,
  and decision rule as implemented: grounding, combination, calibration,
  uncertainty quantification, verification, model selection, precipitation,
  truth QC, and a [bibliography](https://hbmartin.github.io/grounded-weather-forecast/methods/bibliography/) matched to modules
- **[Architecture](https://hbmartin.github.io/grounded-weather-forecast/architecture/)** — layers, contracts, storage, libraries, leakage defences
- **[Limitations](https://hbmartin.github.io/grounded-weather-forecast/limitations/)** — what this cannot do, and the three real
  bugs the evaluation harness caught. **Read before trusting any number.**
- **[API reference](https://hbmartin.github.io/grounded-weather-forecast/reference/api/)** — the importable public surface

**Project**

- [`CONTRIBUTING.md`](https://github.com/hbmartin/grounded-weather-forecast/blob/main/CONTRIBUTING.md) — dev setup and the gates CI enforces
- [`CHANGELOG.md`](https://github.com/hbmartin/grounded-weather-forecast/blob/main/CHANGELOG.md) — release history and migration instructions
  (0.3.0 changed scoring semantics; re-run backtest before comparing eras)
- [`CONTEXT.md`](https://github.com/hbmartin/grounded-weather-forecast/blob/main/CONTEXT.md) — the project's naming discipline
- **[Decisions](https://hbmartin.github.io/grounded-weather-forecast/adr/0004-grounding-defaults-to-bias-only/)** — architecture decision records

## Development

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
uv run ruff check src --fix && uv run ruff format src tests
uvx --from semgrep==1.170.0 semgrep scan --test --config semgrep/provider-qc.yml semgrep/tests/provider_qc_grouping.py
uvx --from semgrep==1.170.0 semgrep scan --metrics=off --error --config semgrep/provider-qc.yml src/grounded_weather_forecast/dataset/matrix.py
uvx --from semgrep==1.170.0 semgrep scan --test --config semgrep/artifact-pointer-paths.yml semgrep/tests/artifact_pointer_paths.py
uvx --from semgrep==1.170.0 semgrep scan --metrics=off --error --config semgrep/artifact-pointer-paths.yml src/grounded_weather_forecast/artifacts.py
uv run pyrefly check src && uv run ty check src
uv run lizard -Eduplicate -C 27 -x "*/dashboard/assets/*" src
uv run pytest tests/ --cov=src --cov-report=term-missing
```

See the [release guide](https://github.com/hbmartin/grounded-weather-forecast/blob/main/docs/releasing.md)
for the TestPyPI and PyPI trusted publishing setup and checklist.

## License

Apache-2.0
