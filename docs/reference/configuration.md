# Configuration reference

Every key in `config.toml`, its type, default, validation rule, and effect.

Configuration is loaded by `config.py` into **frozen dataclasses** with explicit
validation — an unknown value fails at load with a `ConfigError` naming the key,
not at 3 a.m. inside a blender. Start from `config.example.toml`.

`--config PATH` selects the file; the default is `config.toml` in the working
directory.

!!! tip "Only `[station]` and `[forecasts]` are mandatory"
    Every other section has working defaults. A first config is about twelve
    lines — see [Getting started](../getting-started.md).

---

## Environment variables

There is exactly **one** environment-variable mechanism in the entire codebase: a
value of the form `"$VARNAME"` in `[truth_qc] synoptic_token` is read from the
environment at use time.

```toml
[truth_qc]
synoptic_token = "$SYNOPTIC_TOKEN"
```

No other setting consults the environment. Everything else is in the file, which
is what makes the `config_fingerprint` a complete description of how a run was
configured.

---

## `[station]` — where truth comes from

| Key | Type | Default | Meaning |
|---|---|---|---|
| `db_path` | path | *required* | the `ambientweather2sqlite` SQLite file |
| `timezone` | IANA name | *required* | defines the **local day** for daily products and DST handling |
| `latitude` | float | *required* | decimal degrees, north positive |
| `longitude` | float | *required* | decimal degrees, east positive |
| `elevation_m` | float | *required* | metres; used for pressure reduction and neighbour lapse adjustment |
| `immutable` | bool | `false` | open the DB with `?immutable=1` |

!!! warning "`immutable` must be `false` for a live database"
    `immutable = true` is only safe for a static snapshot file. A live WAL
    database opened immutable will return stale or inconsistent reads. The
    default is the safe one.

### `[station.columns]` — DB column → canonical channel

Maps your station's column names onto the channels the pipeline understands.
Defaults cover a standard AmbientWeather unit:

```toml
[station.columns]
outTemp = "temp"
outHumi = "humidity"
avgwind = "wind_speed"
gustspeed = "wind_gust"
eventrain = "rain_counter"
AbsPress = "pressure_station"
```

Canonical channels: `temp`, `humidity`, `wind_speed`, `wind_gust`,
`rain_counter`, `pressure_station`.

### `[station.units]` — channel → source unit

```toml
[station.units]
temp = "degF"
humidity = "pct"
wind_speed = "mph"
wind_gust = "mph"
rain_counter = "inch"
pressure_station = "inHg"
```

Accepted units: `degF`, `degC`, `pct`, `mph`, `ms`, `inch`, `mm`, `inHg`, `hpa`.
Everything is normalized to metric on ingest.

!!! note "`rain_counter` must be the *event* counter"
    The pipeline expects a monotone counter that resets between events, and
    computes reset-aware increments from it. A field that already reports hourly
    accumulation is a different quantity — see
    [Methods: precipitation §2](../methods/precipitation.md#2-truth-side-accumulation).

Run `grounded-weather-forecast qc` after editing either table. A wrong mapping
shows up immediately as a channel that is 100% null.

---

## `[forecasts]` — where the providers' opinions come from

| Key | Type | Default | Meaning |
|---|---|---|---|
| `db_path` | path | *required* | the `omni-weather-forecast-apis` archive |
| `immutable` | bool | `false` | as above — keep `false` while the collector writes |
| `sources` | list of str | `[]` | allowlist of source slugs; empty means every source discovered |
| `max_forecast_age_hours` | float > 0 | `12.0` | staleness cap: a source older than this is unavailable at that snapshot |
| `exclude` | list of `"source:variable"` | `[]` | null one variable of one source without excluding the source |

`max_forecast_age_hours` materially determines which sources count as available
for 6-hourly providers, and therefore what the blend is actually averaging.
Raising it admits staler forecasts; lowering it thins coverage. Check its effect
against the correlation report rather than guessing.

### `[forecasts.max_lead_hours]`

Per-source lead cap in hours. Rows beyond the cap are dropped at matrix build,
trimming a provider's degraded horizon tail.

```toml
[forecasts.max_lead_hours]
stormglass_sg = 120
```

---

## `[dataset]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `dir` | path | `data` | where parquet outputs go |
| `min_hour_coverage` | fraction in $[0,1]$ | `0.8` | minute coverage required for an interval-mean hourly truth |
| `min_day_coverage` | fraction in $[0,1]$ | `0.8` | minute coverage required for a daily truth, against the day's *actual* length |
| `pop_threshold_mm` | float > 0 | `0.254` | the "measurable precipitation" threshold defining PoP — 0.01 inch |
| `precip_reset_fraction` | fraction | `0.5` | a counter drop below this fraction of the prior value is a genuine reset; a smaller dip is noise |

`pop_threshold_mm` changes what every PoP number in the system asserts. It is a
convention, not a constant of nature.

---

## `[qc]` — station quality control

Metric units throughout. Defaults exist for every standard channel; override
per channel.

```toml
[qc.bounds]
temp = [-40.0, 55.0]

[qc.max_step]
temp = 5.0

[qc.flatline_minutes]
temp = 180
```

| Sub-table | Type | Meaning |
|---|---|---|
| `bounds` | `[low, high]` | outside → `OUT_OF_BOUNDS` |
| `max_step` | float > 0 | per-minute rate limit for the two-sided spike rule; scales with the actual sample gap |
| `flatline_minutes` | int > 0 | run of bit-identical values that counts as a stuck sensor |

Flagged samples become `NULL`, never corrected. See
[Methods: truth and QC §1](../methods/truth-qc.md#1-station-quality-control).

---

## `[provider_qc]` — plausibility QC on forecast values

Applied *before* grounding, so a single bad provider value cannot poison the fit.
Truth is never consulted here, so it introduces no leakage.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | bool | `true` | master switch |
| `mad_k` | float > 0 | `5.0` | scaled-MAD multiple for the cross-source outlier test |
| `min_sources` | int > 0 | `4` | providers required before the cross-source test runs at all |
| `bounds` | table of `[low, high]` | per-variable defaults | absolute physical bounds |
| `cross_source_variables` | list of str | temp/dew/humidity/pressure/precip | which variables get the outlier test |
| `min_deviation` | table of float | per-variable defaults | absolute deviation floor |

A value is nulled only when it exceeds **both** `mad_k` scaled MADs **and** the
`min_deviation` floor.

!!! danger "Never add a variable to `cross_source_variables` without a `min_deviation`"
    On a dry day the cross-source median and MAD are both zero, so
    `mad_k × MAD` is zero and the MAD test alone flags *any* lone nonzero value.
    Without a floor, a variable added here will null all light precipitation.
    Always set both together. See
    [Methods: precipitation §4](../methods/precipitation.md#4-cross-source-qc-guards).

Gusts and PoP are deliberately excluded from the cross-source pass: a gust
disagreeing with its peers is often correct, and that is the point of a gust
field.

---

## `[backfill.open_meteo]` / `[backfill.dynamical]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `models` | list of str | per provider | models to fetch |
| `start_date` | date | — | first date to backfill |
| `publication_lag_hours` | float > 0 | `6.0` | **`dynamical` only** — `fetched_at = init + lag` |
| `max_lead_hours` | float > 0 | `48.0` | **`dynamical` only** — cap on backfilled lead |

`publication_lag_hours` is a leakage control, not a performance knob. A model
cycle initialized at 00Z is not *published* until hours later; pretending it was
visible at init would inflate short-lead skill with information nobody had.

---

## `[ensembles]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `models` | list of str | — | Open-Meteo ensemble models to poll; empty disables ingestion |
| `variables` | list of str | `temp_c`, `dew_point_c`, `wind_speed_ms`, `wind_gust_ms`, `pressure_sea_hpa`, `precip_mm` | which variables to reduce |

Open-Meteo keeps only the latest run's members, so run `ingest-ensembles` at
least once per model cycle or the members are gone. `build-dataset` must run
after each ingest before the `ens__*` features are visible to backtests and
serving.

---

## `[backtest]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `initial_train_days` | int > 0 | `90` | archive span required before the first fold origin |
| `step_days` | int > 0 | `7` | spacing between rolling origins |
| `rolling_window_days` | int > 0 | `180` | training window length when `--window rolling` |

`initial_train_days + step_days` is the minimum archive age before `backtest`
produces anything at all. With defaults that is 97 days — the "no rolling-origin
folds" message means you are not there yet.

---

## `[predict]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `selection` | str | `skill_per_slice` | how methods are chosen per slice |
| `history_path` | path | `data/predict_history.parquet` | where served forecasts are appended for self-verification |
| `minutely_tau_hours` | float > 0 | `3.0` | anchor decay for the minutely **fallback** path |
| `quantile_recalibration` | str or per-product table | `none` | one of `none`, `pit`, `cqr` |

`history_path` is what self-verification reads. Point it somewhere durable —
losing it loses the only measurement of serving-path drift.

`minutely_tau_hours` applies **only when no minutely evidence exists**. Once a
minutely leaderboard has promoted constructions, serving uses those per
(variable, bucket) and this value is inert. See
[Methods: combination §8](../methods/combination.md#8-minutely-path-constructions).

`quantile_recalibration` may be a single string or a per-product table:

```toml
[predict.quantile_recalibration]
hourly = "none"
daily = "cqr"
minutely = "none"
```

### `[predict.methods]` — per-slice pins

```toml
[predict.methods]
"hourly.temp_c" = "gbm"
```

A pin overrides the leaderboard. Useful for reproducing an old serving decision
or isolating a method in production; it bypasses the promotion gate, so it is a
deliberate override of the system's own evidence rather than a tuning knob.

---

## `[promotion]` — the statistical gate

| Key | Type | Default | Meaning |
|---|---|---|---|
| `rule` | `mcs` \| `legacy` \| `seq_mcs` | `mcs` | which promotion gate to apply |
| `alpha` | float > 0 | `0.1` | level for the MCS, the e-process threshold $\ln(1/\alpha)$, and FDR control |
| `live_gap_factor` | float > 0 | `1.5` | demote when realized MAE exceeds this multiple of backtest MAE |
| `min_live_n` | int > 0 | `24` | scored live cases required before demotion can fire |
| `report_gap_threshold` | float > 0 | `0.15` | relative gap above which a slice is listed under "Blocked promotions" |
| `mcs_bootstrap` | int > 0 | `500` | moving-block bootstrap replicates |
| `mcs_block_length` | int ≥ 0 | `0` | block length; `0` means $\approx n^{1/3}$ |

### `[promotion.references]`

Per-variable overrides of the default reference set
(`best_provider`, `equal_weight`, `damped_grounded_equal_weight`). The leaderboard
computes columns for the *union* of defaults and overrides, so a pinned
`skill_vs_equal_weight` column never silently changes meaning.

Full treatment: [Methods: model selection](../methods/model-selection.md).

---

## `[truth_qc]` — neighbour cross-checks

| Key | Type | Default | Meaning |
|---|---|---|---|
| `synoptic_token` | str | — | Synoptic Data token; `"$VAR"` reads the environment |
| `radius_km` | float > 0 | `25.0` | neighbour search radius |
| `elevation_band_m` | float > 0 | `300.0` | neighbour elevation band around the station |
| `lapse_k_per_km` | float > 0 | `6.5` | lapse rate used to adjust neighbour temperatures to station elevation |
| `drift_statistic` | `snht` \| `pettitt` | `snht` | change-point test |
| `gate_fitting` | bool | `false` | null temperature labels under a latched drift verdict |

NWS METAR neighbours need no token; Synoptic is the optional richer source.

`lapse_k_per_km = 6.5` is the *standard* environmental lapse rate, not your
mountainside's. A strong inversion reverses its sign — if your station sits in a
cold-air pool, this default will systematically misjudge neighbour agreement.

!!! warning "`gate_fitting` is off by default, deliberately"
    Turning it on lets a drift verdict delete temperature truth. A wrong
    quarantine silently destroys the labels everything is scored against. Turn it
    on only after you have watched the verdict behave for a while.

---

## `[reports]` and `[artifacts]`

| Key | Type | Default |
|---|---|---|
| `[reports] dir` | path | `reports` |
| `[artifacts] dir` | path | `artifacts` |

See [Outputs](outputs.md) for what lands in each.

---

## Keeping `config.toml` out of git

`config.toml` holds your station's real coordinates and local paths.
`config.example.toml` is the committed template; copy it, edit the copy, and keep
the copy untracked.
