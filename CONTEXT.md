# grounded-weather-forecast

Blends multiple providers' weather forecasts into station-grounded minutely/hourly/daily
forecasts for a single personal weather station, with a backtest leaderboard as the sole
arbiter of which method ships.

## Language

### Time

**Source retrieved at**:
The provider-specific `fetched_at` instant when the upstream collector received a
forecast. This controls source age and as-of eligibility.
_Avoid_: issue time

**Source available at**:
The run's `completed_at` instant when a collector run became available as a coherent
archive snapshot. This anchors dataset snapshots.
_Avoid_: fetched at

**Forecast issued at**:
The instant grounded-weather-forecast emits a product. Training evidence, observations, provider
rows, and promoted releases must all have been available by this instant.
_Avoid_: fetch time, run time

**Valid time**:
The moment or interval a forecast is about.
_Avoid_: target time, forecast time

**Lead**:
Distance from forecast-issued time to validity in the product's lead unit: minutes for
minutely, elapsed hours for hourly, and local-calendar days for daily. Provider hourly
leads are always recomputed from timestamps, never read from the untrusted upstream
`horizon_hours` column.
_Avoid_: horizon (in code)

**Lead bucket**:
One of the fixed left-closed lead intervals used to stratify fitting and evaluation
(hourly: 0–1, 1–3, 3–6, 6–12, 12–24, 24–48, 48–96, 96–168, 168–240, 240+ h; daily: D1, D2, D3–4, D5–7, D8–10).

**Snapshot**:
The as-of view of all sources at one forecast-issued time: for each source, its latest
forecast retrieved at or before that moment and not older than the staleness cap. All
products are read in one SQLite transaction and are scoped to the configured location.
_Avoid_: vintage

### Data

**Observation**:
One raw station sample row (~1/minute), imperial units, unvalidated.

**Truth**:
The QC'd, metric-normalized, time-aggregated observation series used for training and
scoring. Truth is either present and trusted or null — never imputed or corrected.

**Truth semantics**:
The canonical aggregation used to score an hourly state variable: instantaneous
(±5-min centered mean at the valid hour) or interval mean over the hour. The alignment
study compares every provider but selects one canonical meaning per emitted variable;
training never changes the target definition from provider to provider.

**Provider**:
An upstream forecast API, identified by its slug (e.g. `open_meteo`).

**Source**:
One provider+model forecast stream; the unit a blender weighs. Slug: provider, or
`provider_model` when a provider exposes multiple models.
_Avoid_: expert (except in the online-experts method), member

**Source kind**:
Whether supervised rows came from the live archive (`live`) or the Previous Runs
backfill (`synthetic`). Never pooled silently.

**Availability mask**:
The per-row boolean pattern of which sources have a usable forecast; every blender must
renormalize over available sources.

**Evaluation run**:
One immutable production of score rows, identified by dataset fingerprint, source set
and kind, product, window, per-variable truth semantics, method set, code version, and
configuration fingerprint. Re-running the same specification creates a distinct run.

**Model release**:
A promoted mapping from product × variable × lead bucket to a method, tied to compatible
live evaluation-run evidence and a training cutoff. Serving consumes the release identity;
synthetic evidence can inform reports but cannot be promoted for live serving.

**Degraded forecast**:
A forecast emitted without compatible promoted evidence. It uses the fit-free
availability-renormalized equal-weight method, says `status = "degraded"`, and records
the reason. It is a valid product with weaker guarantees, not an implicit trained model.

### Methods

**Grounding**:
Per-source correction toward the station, fitted per variable × lead bucket. A
bias (intercept) correction by default; the slope is opt-in (see ADR 0004).
_Avoid_: calibration (reserved for probabilistic calibration), MOS (in code)

**Blending**:
Combining grounded sources into one forecast via weights or a learned stacker.
_Avoid_: ensembling, averaging (as method names)

**Anchoring**:
Short-lead correction of a blend toward the latest observed residual, decaying
exponentially with lead.
_Avoid_: nowcasting (the anchored minutely product is a nowcast; the technique is anchoring)

**Blender**:
Any method implementing the shared protocol `fit(train) / predict(matrix) -> point (+quantiles)`,
baselines included.
_Avoid_: model (overloaded with provider models)

**Product**:
An emitted forecast bundle with its own temporal contract (`ProductSpec`): minutely
(next 60 minutes), hourly (next 48 elapsed hours), or daily (next 10 local-calendar days).

**Self-verification**:
Scoring the system's own emitted products against later truth, alongside providers and
backtest expectations.
