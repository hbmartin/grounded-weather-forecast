# Database design and structure

> Reference snapshot from
> [`omni-weather-forecast-apis` commit `ee9e166`](https://github.com/hbmartin/omni-weather-forecast-apis/blob/ee9e166897726fdb0916a99433fa7f512c26a188/docs/database.md).
> It documents this project's upstream SQLite contract; the upstream page is
> authoritative if the two diverge.

The CLI can persist each aggregated forecast response to a SQLite database.
The database is designed as an append-oriented record of forecast runs: request
metadata is stored once, provider outcomes are stored below the run, and each
provider/model forecast owns its normalized minutely, hourly, daily, and alert
rows.

This page describes the schema created by
`omni_weather_forecast_apis.sqlite_store` and the quota table created by
`omni_weather_forecast_apis.quota.SqliteQuotaTracker`.

## Scope and design goals

The SQLite layer has four main goals:

1. Preserve the provenance of every normalized value from CLI run to provider
   and model.
2. Keep provider failures beside successful results so partial failures remain
   queryable.
3. Store normalized time-series data in relational columns that work directly
   with SQLite, pandas, DuckDB, and similar tools.
4. Provide a convenient hourly feature view for ensemble forecasting and
   verification workloads.

The database is not the HTTP response cache. The conditional-request cache is
process-local and in memory. The database also does not store observations or
computed consensus forecasts; those are intended to live in downstream
systems.

## Creation and write lifecycle

Set `--sqlite PATH` on the CLI, or set `sqlite = "PATH"` in the TOML
configuration:

```bash
uv run omni-weather \
  --config ./config.toml \
  --lat 40.7128 \
  --lon -74.0060 \
  --sqlite ./forecasts.sqlite
```

For each CLI invocation, persistence happens in this order:

1. `SqliteQuotaTracker` creates `provider_quota_usage` when quota tracking is
   first used. Each actual provider request attempt, including a retry, reserves
   quota before the HTTP request is sent.
2. The completed `ForecastResponse` initializes or upgrades the forecast
   schema.
3. One `forecast_runs` row is inserted.
4. Every provider outcome is inserted into `provider_results`. Successful
   outcomes also create model rows and normalized forecast data.
5. The forecast transaction is committed and its `run_id` is returned.
6. Provider lifecycle events are written to `provider_logs` in a separate
   transaction associated with that `run_id`.

Because forecast data and logs use separate transactions, a successfully saved
run can exist without its logs if the later log write fails. A failed forecast
write is rolled back when its connection closes.

The public library client does not automatically persist forecast responses.
Library users can call `save_forecast_response()` directly, use a response
hook, or provide their own storage adapter.

## Relationship model

```text
forecast_runs (one CLI invocation)
├── provider_results (one outcome per returned provider result)
│   └── source_forecasts (one normalized forecast per provider/model)
│       ├── minutely_points
│       ├── hourly_points
│       ├── daily_points
│       └── alerts
└── provider_logs (zero or more lifecycle events)

provider_quota_usage (independent provider/day counters)
```

The principal relationship path is:

```text
forecast_runs.id
  -> provider_results.run_id
  -> source_forecasts.provider_result_id
  -> *_points.source_forecast_id / alerts.source_forecast_id
```

`provider_quota_usage` is intentionally independent. A quota counter can be
incremented before a forecast run is saved, and it must survive deletion of
historical forecast data.

## Schema summary

| Object | Cardinality and purpose |
|--------|-------------------------|
| `forecast_runs` | One row per persisted aggregate response |
| `provider_results` | One success or error outcome per result in a run |
| `source_forecasts` | One model-specific normalized forecast inside a successful provider result |
| `minutely_points` | Zero or more normalized precipitation points per source forecast |
| `hourly_points` | Zero or more normalized weather points per source forecast |
| `daily_points` | Zero or more normalized daily summaries per source forecast |
| `alerts` | Zero or more normalized weather alerts per source forecast |
| `provider_logs` | Zero or more provider lifecycle events, optionally associated with a run |
| `provider_quota_usage` | One durable counter per provider and UTC day |
| `stacking_features` | Read-only joined view of successful hourly forecasts |
| `db_repairs` | Audit log written by `scripts/repair_db.py`; absent unless a repair ran |

Alongside the database, the CLI writes a **raw payload archive**: a `raw/`
directory of gzipped JSONL files (one per invocation) recording every network
response, linked from `forecast_runs.raw_archive_path`. See
[Data corrections](https://github.com/hbmartin/omni-weather-forecast-apis/blob/ee9e166897726fdb0916a99433fa7f512c26a188/docs/data-corrections.md)
for the format and rationale.

SQLite uses dynamic typing, but each declaration below establishes the
expected type affinity. Unless a column is marked `NOT NULL`, missing provider
data is stored as SQL `NULL` rather than a sentinel value.

## Table reference

### `forecast_runs`

This is the root table for a persisted aggregate response.

| Column | Type and constraints | Meaning |
|--------|----------------------|---------|
| `id` | `INTEGER PRIMARY KEY AUTOINCREMENT` | Stable run identifier returned by `save_forecast_response()` |
| `latitude` | `REAL NOT NULL` | Requested latitude in decimal degrees |
| `longitude` | `REAL NOT NULL` | Requested longitude in decimal degrees |
| `granularity` | `TEXT NOT NULL` | JSON array of requested granularity slugs, such as `["hourly", "daily"]` |
| `language` | `TEXT NOT NULL` | Resolved language preference |
| `completed_at` | `TEXT NOT NULL` | UTC ISO 8601 completion timestamp |
| `total_latency_ms` | `REAL NOT NULL` | End-to-end aggregate latency in milliseconds |
| `total_results` | `INTEGER NOT NULL` | Total number of provider results |
| `succeeded` | `INTEGER NOT NULL` | Number of successful provider results |
| `failed` | `INTEGER NOT NULL` | Number of failed provider results |
| `raw_archive_path` | `TEXT` | Path to the gzipped JSONL raw payload archive for this run; `NULL` when archiving was disabled or no network traffic occurred |
| `app_version` | `TEXT` | Package version that wrote the run; `NULL` for rows written before versions were stamped (pre correctness-sweep data — see [Data corrections](https://github.com/hbmartin/omni-weather-forecast-apis/blob/ee9e166897726fdb0916a99433fa7f512c26a188/docs/data-corrections.md)) |

The summary counts are stored rather than recomputed so the database preserves
the exact response summary emitted by the client.

### `provider_results`

This table records both sides of the `ProviderSuccess | ProviderError` union.
Success-only and error-only columns are nullable.

| Column | Type and constraints | Meaning |
|--------|----------------------|---------|
| `id` | `INTEGER PRIMARY KEY AUTOINCREMENT` | Provider outcome identifier |
| `run_id` | `INTEGER NOT NULL`, foreign key | Parent `forecast_runs.id`; deleted with its run |
| `provider` | `TEXT NOT NULL` | Stable provider slug, such as `open_meteo` |
| `status` | `TEXT NOT NULL` | `success` or `error` |
| `fetched_at` | `TEXT` | Successful fetch timestamp as UTC ISO 8601 text |
| `fetched_at_unix` | `INTEGER` | Successful fetch timestamp as Unix seconds |
| `run_cycle` | `TEXT` | Synthetic six-hour UTC cycle bucket for a successful fetch |
| `latency_ms` | `REAL NOT NULL` | Provider latency, including elapsed time until failure for errors |
| `error_code` | `TEXT` | Normalized error slug for failed results |
| `error_message` | `TEXT` | Human-readable failure detail |
| `http_status` | `INTEGER` | HTTP status when the failure has one |
| `raw_json` | `TEXT` | Optional provider success or error payload serialized as JSON |

`run_cycle` is calculated by flooring `fetched_at` to the previous `00:00`,
`06:00`, `12:00`, or `18:00` UTC boundary. It is a convenient grouping key,
not a provider-declared model initialization time. Providers do not all expose
their authoritative model cycle.

Successful raw payloads are normally `NULL`. They are populated when the
request enables `include_raw` or the CLI uses `--include-raw`. Error payloads
are stored when the normalized error contains raw detail. Values are encoded
with sorted JSON keys and fall back to string conversion for otherwise
non-serializable objects.

### `source_forecasts`

A provider response may contain more than one model. This table gives each
model its own parent row before inserting time-series data.

| Column | Type and constraints | Meaning |
|--------|----------------------|---------|
| `id` | `INTEGER PRIMARY KEY AUTOINCREMENT` | Source forecast identifier |
| `provider_result_id` | `INTEGER NOT NULL`, foreign key | Parent `provider_results.id`; deleted with its provider result |
| `provider` | `TEXT NOT NULL` | Source attribution slug |
| `model` | `TEXT NOT NULL` | Provider model or source name |

The provider slug is repeated here deliberately: analytics can identify a
model directly from `source_forecasts` without treating the model name as
globally unique.

### `minutely_points`

| Column | Type and constraints | Meaning and normalized unit |
|--------|----------------------|-----------------------------|
| `source_forecast_id` | `INTEGER NOT NULL`, foreign key | Parent `source_forecasts.id`; deleted with its source |
| `timestamp` | `TEXT NOT NULL` | UTC ISO 8601 valid time |
| `timestamp_unix` | `INTEGER NOT NULL` | Valid time as Unix seconds |
| `precipitation_intensity` | `REAL` | Precipitation rate in mm/h |
| `precipitation_probability` | `REAL` | Probability from `0` to `1` |

### `hourly_points`

Each row represents one provider/model forecast at one valid time. The schema
keeps both normalized values and selected provider-native condition fields.

| Column | Type and constraints | Meaning and normalized unit |
|--------|----------------------|-----------------------------|
| `source_forecast_id` | `INTEGER NOT NULL`, foreign key | Parent `source_forecasts.id`; deleted with its source |
| `timestamp` | `TEXT NOT NULL` | UTC ISO 8601 valid time |
| `timestamp_unix` | `INTEGER NOT NULL` | Valid time as Unix seconds |
| `horizon_hours` | `REAL` | Hours from provider fetch time to valid time |
| `temperature` | `REAL` | Air temperature at 2 m, °C |
| `apparent_temperature` | `REAL` | Feels-like temperature, °C |
| `dew_point` | `REAL` | Dew point, °C |
| `humidity` | `REAL` | Relative humidity, percent from `0` to `100` |
| `wind_speed` | `REAL` | Wind speed at 10 m, m/s |
| `wind_gust` | `REAL` | Wind gust speed, m/s |
| `wind_direction` | `REAL` | Meteorological direction in degrees |
| `pressure_sea` | `REAL` | Sea-level pressure, hPa |
| `pressure_surface` | `REAL` | Surface pressure, hPa |
| `precipitation` | `REAL` | Liquid-equivalent precipitation, mm |
| `precipitation_probability` | `REAL` | Probability from `0` to `1` |
| `rain` | `REAL` | Rain amount, mm |
| `snow` | `REAL` | Liquid-equivalent snowfall, mm — only for providers that report liquid equivalent (e.g. OpenWeather, Open-Meteo via `snowfall_water_equivalent`) |
| `snowfall_depth` | `REAL` | New snowfall depth, mm — only for providers that report depth (e.g. Open-Meteo `snowfall`, Pirate Weather `snowAccumulation`) |
| `snow_depth` | `REAL` | Snow depth on the ground, mm |
| `cloud_cover` | `REAL` | Total cloud cover, percent |
| `cloud_cover_low` | `REAL` | Low cloud cover, percent |
| `cloud_cover_mid` | `REAL` | Mid cloud cover, percent |
| `cloud_cover_high` | `REAL` | High cloud cover, percent |
| `visibility` | `REAL` | Visibility, km |
| `uv_index` | `REAL` | UV index |
| `solar_radiation_ghi` | `REAL` | Global horizontal irradiance, W/m² |
| `solar_radiation_dni` | `REAL` | Direct normal irradiance, W/m² |
| `solar_radiation_dhi` | `REAL` | Diffuse horizontal irradiance, W/m² |
| `condition` | `TEXT` | Normalized `WeatherCondition` slug |
| `condition_original` | `TEXT` | Provider-native condition text |
| `condition_code_original` | `TEXT` | Provider-native condition or icon code, stringified before storage |
| `is_day` | `INTEGER` | Boolean daylight flag stored as `1` or `0` |

`horizon_hours` is derived as:

```text
(timestamp_unix - fetched_at_unix) / 3600.0
```

It can be fractional, zero, or negative. It measures lead time from the local
fetch timestamp, not necessarily from the upstream model's initialization
time.

### `daily_points`

| Column | Type and constraints | Meaning and normalized unit |
|--------|----------------------|-----------------------------|
| `source_forecast_id` | `INTEGER NOT NULL`, foreign key | Parent `source_forecasts.id`; deleted with its source |
| `forecast_date` | `TEXT NOT NULL` | Forecast date in ISO `YYYY-MM-DD` form |
| `temperature_max`, `temperature_min` | `REAL` | Daily extrema, °C |
| `apparent_temperature_max`, `apparent_temperature_min` | `REAL` | Apparent-temperature extrema, °C |
| `wind_speed_max`, `wind_gust_max` | `REAL` | Daily maximum speeds, m/s |
| `wind_direction_dominant` | `REAL` | Dominant meteorological direction in degrees |
| `precipitation_sum` | `REAL` | Total liquid-equivalent precipitation, mm |
| `precipitation_probability_max` | `REAL` | Maximum probability from `0` to `1` |
| `rain_sum`, `snowfall_sum` | `REAL` | Daily rain and liquid-equivalent snowfall totals, mm |
| `snowfall_depth_sum` | `REAL` | Daily new snowfall depth total, mm |
| `cloud_cover_mean` | `REAL` | Mean cloud cover, percent |
| `uv_index_max` | `REAL` | Maximum UV index |
| `visibility_min` | `REAL` | Minimum visibility, km |
| `humidity_mean` | `REAL` | Mean relative humidity, percent |
| `pressure_sea_mean` | `REAL` | Mean sea-level pressure, hPa |
| `condition` | `TEXT` | Representative normalized condition slug |
| `summary` | `TEXT` | Provider summary text |
| `sunrise`, `sunset` | `TEXT` | Optional UTC ISO 8601 timestamps |
| `moonrise`, `moonset` | `TEXT` | Optional UTC ISO 8601 timestamps |
| `moon_phase` | `REAL` | Moon phase from `0` to `1` |
| `daylight_duration` | `REAL` | Daylight duration in seconds |
| `solar_radiation_sum` | `REAL` | Shortwave radiation total, MJ/m² |

### `alerts`

| Column | Type and constraints | Meaning |
|--------|----------------------|---------|
| `source_forecast_id` | `INTEGER NOT NULL`, foreign key | Parent `source_forecasts.id`; deleted with its source |
| `sender_name` | `TEXT NOT NULL` | Alert issuer |
| `event` | `TEXT NOT NULL` | Event or alert title |
| `start` | `TEXT NOT NULL` | UTC ISO 8601 start timestamp |
| `end` | `TEXT` | Optional UTC ISO 8601 end timestamp |
| `description` | `TEXT NOT NULL` | Alert body |
| `severity` | `TEXT` | Normalized severity slug |
| `url` | `TEXT` | Optional source URL |

### `provider_logs`

The client emits structured lifecycle events with phases `start`, `retry`,
`success`, and `error`.

| Column | Type and constraints | Meaning |
|--------|----------------------|---------|
| `id` | `INTEGER PRIMARY KEY AUTOINCREMENT` | Log event identifier |
| `run_id` | `INTEGER`, foreign key | Optional parent `forecast_runs.id`; set to `NULL` if the run is deleted |
| `provider` | `TEXT NOT NULL` | Provider slug |
| `phase` | `TEXT NOT NULL` | Lifecycle phase |
| `message` | `TEXT NOT NULL` | Human-readable event message |
| `latency_ms` | `REAL NOT NULL DEFAULT 0` | Elapsed time associated with the event |
| `error_code` | `TEXT` | Normalized error slug when relevant |
| `http_status` | `INTEGER` | HTTP status when relevant |
| `extra_json` | `TEXT` | Optional structured metadata such as attempt, delay, or limit |
| `logged_at` | `TEXT NOT NULL` | Event timestamp as ISO 8601 text |

`run_id` is nullable so callers can persist events independently. The CLI
normally assigns it after the forecast transaction returns the new run ID.

### `provider_quota_usage`

This table is initialized by `SqliteQuotaTracker`, not by the forecast schema
initializer. It therefore appears only when the SQLite quota tracker is used.

| Column | Type and constraints | Meaning |
|--------|----------------------|---------|
| `provider` | `TEXT NOT NULL`, composite primary key | Provider slug |
| `day` | `TEXT NOT NULL`, composite primary key | UTC day in ISO `YYYY-MM-DD` form |
| `request_count` | `INTEGER NOT NULL DEFAULT 0` | Reserved provider request attempts for that day |

Quota consumption uses `BEGIN IMMEDIATE` around the read-check-increment
sequence. That obtains a write reservation before reading, so two local
writers cannot both consume the last available request. Connections use a
10-second SQLite busy timeout.

Quota rows are not linked to `forecast_runs`: retries count separately, a
request may fail after consuming quota, and quota enforcement must remain
durable even when old forecast history is deleted.

## Keys, uniqueness, and deletion behavior

`forecast_runs`, `provider_results`, `source_forecasts`, and `provider_logs`
have generated integer primary keys. `provider_quota_usage` has the composite
primary key `(provider, day)`.

The point and alert tables do not define generated primary keys or uniqueness
constraints. Their identity is the parent source plus their row contents. The
writer appends each normalized list exactly once for a new source forecast;
the schema does not deduplicate repeated timestamps supplied by a provider.

The forecast writer enables `PRAGMA foreign_keys = ON`. With foreign-key
enforcement enabled, deleting a run has these effects:

| Deleted row | Effect |
|-------------|--------|
| `forecast_runs` | Cascades through provider results, source forecasts, all point rows, and alerts |
| `provider_results` | Cascades through source forecasts, points, and alerts |
| `source_forecasts` | Cascades through its point rows and alerts |
| `forecast_runs` with logs | Preserves log rows but sets `provider_logs.run_id` to `NULL` |
| Any forecast history | Does not change `provider_quota_usage` |

SQLite foreign-key enforcement is connection-local. External maintenance
scripts must execute `PRAGMA foreign_keys = ON` before relying on these
cascades.

## Time representation

The normalized Pydantic schema converts point and response datetimes to UTC
before persistence. SQLite stores time in three forms chosen for the expected
query:

- ISO 8601 `TEXT` preserves readable, timezone-bearing timestamps.
- Unix-second `INTEGER` columns support range scans and arithmetic for
  minutely/hourly valid times and provider fetch times.
- Daily forecast dates and quota days use ISO `YYYY-MM-DD` text.

The duplicated ISO and Unix representations are intentional. Use Unix columns
for comparisons and window queries; use ISO columns for display and export.

## Indexes

The schema initializer creates these indexes with `IF NOT EXISTS`:

| Index | Columns | Intended access path |
|-------|---------|----------------------|
| `idx_provider_results_run_provider` | `provider_results(run_id, provider)` | Outcomes for a provider within one run |
| `idx_provider_results_run_cycle` | `provider_results(run_cycle)` | Results grouped by synthetic cycle |
| `idx_source_forecasts_provider_result` | `source_forecasts(provider_result_id)` | Models belonging to a result |
| `idx_hourly_points_source` | `hourly_points(source_forecast_id)` | Hourly series for one source |
| `idx_hourly_points_horizon` | `hourly_points(horizon_hours)` | Lead-time slices |
| `idx_hourly_points_timestamp` | `hourly_points(timestamp_unix)` | Valid-time range scans |
| `idx_hourly_points_horizon_timestamp` | `hourly_points(horizon_hours, timestamp_unix)` | Lead-time and valid-time filters |
| `idx_minutely_points_source_timestamp` | `minutely_points(source_forecast_id, timestamp_unix)` | Ordered minutely series for one source |
| `idx_daily_points_source_date` | `daily_points(source_forecast_id, forecast_date)` | Ordered daily series for one source |

Primary keys create their own SQLite indexes. There are currently no secondary
indexes on alerts, provider logs, or quota counts.

## `stacking_features` view

`stacking_features` is a read-only convenience view over successful hourly
data. It joins:

```text
hourly_points
  -> source_forecasts
  -> provider_results
  -> forecast_runs
```

The view exposes provenance and features in a single row:

- `run_id`
- `valid_time`, `valid_time_unix`, `horizon_hours`
- `run_cycle`, `fetched_at`, `fetched_at_unix`
- `provider`, `model`, `latitude`, `longitude`
- all normalized hourly numeric fields
- normalized `condition` and `is_day`

It filters to `provider_results.status = 'success'`. Error rows and
provider-native `condition_original` / `condition_code_original` fields are
not included.

Successive runs can contain forecasts for the same provider, model, location,
and valid time. Always include `run_id` when selecting one invocation, and use
an exact `valid_time_unix` when aligning providers. Filtering only on a rounded
`horizon_hours` can mix adjacent forecast times.

Example: select aligned features from the latest run at the first valid time
at least 24 hours ahead:

```sql
WITH latest_run AS (
    SELECT MAX(id) AS run_id
    FROM forecast_runs
), target_time AS (
    SELECT MIN(valid_time_unix) AS valid_time_unix
    FROM stacking_features
    WHERE run_id = (SELECT run_id FROM latest_run)
      AND horizon_hours >= 24
)
SELECT
    valid_time,
    provider,
    model,
    horizon_hours,
    temperature,
    wind_speed,
    precipitation_probability
FROM stacking_features
WHERE run_id = (SELECT run_id FROM latest_run)
  AND valid_time_unix = (SELECT valid_time_unix FROM target_time)
ORDER BY provider, model;
```

The initializer drops and recreates this view whenever it initializes the
schema. This keeps existing database files synchronized with the current view
definition.

## Query examples

### Summarize recent runs

```sql
SELECT
    id,
    completed_at,
    latitude,
    longitude,
    succeeded,
    failed,
    total_latency_ms
FROM forecast_runs
ORDER BY id DESC
LIMIT 20;
```

### Inspect provider failures

```sql
SELECT
    fr.id AS run_id,
    fr.completed_at,
    pr.provider,
    pr.error_code,
    pr.http_status,
    pr.error_message
FROM provider_results AS pr
JOIN forecast_runs AS fr ON fr.id = pr.run_id
WHERE pr.status = 'error'
ORDER BY fr.id DESC, pr.provider;
```

### Read one model's hourly series

```sql
SELECT
    hp.timestamp,
    hp.horizon_hours,
    hp.temperature,
    hp.wind_speed,
    hp.condition
FROM hourly_points AS hp
JOIN source_forecasts AS sf ON sf.id = hp.source_forecast_id
JOIN provider_results AS pr ON pr.id = sf.provider_result_id
WHERE pr.run_id = :run_id
  AND sf.provider = :provider
  AND sf.model = :model
ORDER BY hp.timestamp_unix;
```

### Inspect retries and error events

```sql
SELECT
    provider,
    phase,
    message,
    error_code,
    http_status,
    extra_json,
    logged_at
FROM provider_logs
WHERE run_id = :run_id
  AND phase IN ('retry', 'error')
ORDER BY id;
```

### Check daily quota usage

```sql
SELECT provider, day, request_count
FROM provider_quota_usage
WHERE day >= date('now', '-7 days')
ORDER BY day DESC, provider;
```

## Schema initialization and evolution

The project does not currently use a schema-version table or an external
migration framework. Initialization is idempotent and runs before forecast or
log writes:

1. Base tables are created with `CREATE TABLE IF NOT EXISTS`.
2. `PRAGMA table_info` detects older schemas.
3. Missing additive columns are introduced with `ALTER TABLE ... ADD COLUMN`.
4. Secondary indexes are created with `CREATE INDEX IF NOT EXISTS`.
5. `stacking_features` is dropped and recreated from the current definition.

The built-in upgrade path currently recognizes these additions:

| Table | Additive columns |
|-------|------------------|
| `provider_results` | `fetched_at_unix`, `run_cycle` |
| `hourly_points` | `horizon_hours` |
| `provider_logs` | `extra_json` |

Future destructive or semantic changes require an explicit migration plan;
`CREATE TABLE IF NOT EXISTS` does not reconcile changed constraints or column
types in an already-existing table. Back up long-lived databases before
upgrading across releases that change the persistence schema.

## Operational guidance

- Treat the database as append-oriented history. One CLI invocation creates a
  new run even when its request matches an earlier run.
- Keep `run_id`, provider, model, location, and valid time in downstream keys.
  A valid timestamp is not unique across runs or sources.
- Use `--include-raw` only when provider-native payloads are needed. Raw JSON
  can substantially increase database size and may contain data you do not
  want to retain.
- Enable foreign keys on every external SQLite connection that performs
  deletes or relationship-sensitive writes.
- Copy or back up the database before applying retention deletes or upgrading
  across schema-changing releases.
- For sustained multi-process ingestion or warehouse-scale history, use a
  response hook to stream the normalized response to a database designed for
  that workload. The bundled SQLite store favors a local CLI and analysis
  workflow.
