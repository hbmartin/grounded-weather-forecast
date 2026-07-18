# Getting started

This page takes you from nothing to a blended forecast. It assumes no prior
knowledge of the project and no meteorology.

---

## What you need

**1. A station observation database.** A SQLite file written by
[ambientweather2sqlite](https://github.com/hbmartin/ambientweather2sqlite),
containing an `observations` table with one row per sample (roughly one per
minute). This is your **ground truth** — the thing every forecast is scored
against.

**2. A provider forecast archive.** A SQLite file written by
[omni-weather-forecast-apis](https://github.com/hbmartin/omni-weather-forecast-apis),
containing the forecasts that a set of weather APIs published for your location,
*recorded over time*.

!!! warning "The archive is the hard part"
    The archive has to have been **recorded as it happened**. You cannot go back
    and ask a provider what it predicted last Tuesday. If you have not started
    polling yet, do that first — see [Start the cron](#5-start-the-cron-do-this-today) —
    because until the archive has some age, this system has very little to work
    with. It will tell you so honestly rather than pretending.

**3. Python 3.13+ and [uv](https://docs.astral.sh/uv/).**

---

## Install

```bash
uv tool install grounded-weather-forecast
```

`uv` creates an isolated environment and exposes the command on your path. Check
it worked:

```bash
grounded-weather-forecast --version
grounded-weather-forecast --help
```

To contribute to the project instead, clone the
[source repository](https://github.com/hbmartin/grounded-weather-forecast) and run
`uv sync --dev`.

---

## Configure

Download the example config and edit it:

```bash
curl -L https://raw.githubusercontent.com/hbmartin/grounded-weather-forecast/main/config.example.toml \
  -o config.toml
```

`config.toml` is git-ignored, so your local paths stay local. Here is what each
section does.

### `[station]` — where truth comes from

```toml
[station]
db_path = "aw2sqlite.db"          # your observation database
timezone = "America/Los_Angeles"  # YOUR local timezone. "Today's high" means
                                  # the max over the LOCAL calendar day.
latitude  = 34.2768
longitude = -117.1692
elevation_m = 1400.0              # REQUIRED: reduces pressure to sea level and
                                  # anchors the solar-geometry features
immutable = true                  # true = a static snapshot file
                                  # false = a live database another process writes
```

!!! danger "Set `immutable = false` for a live database"
    `immutable = true` tells SQLite "nobody is writing this file, skip the
    locking". That is a lie if your collector daemon is running, and SQLite will
    happily read garbage. Use `true` only for a copy.

### `[station.columns]` and `[station.units]` — what your sensors are called

Different stations name things differently. This maps *your* column names onto
the canonical channels the system understands:

```toml
[station.columns]
outTemp   = "temp"              # station column  ->  canonical channel
outHumi   = "humidity"
avgwind   = "wind_speed"
gustspeed = "wind_gust"
eventrain = "rain_counter"
AbsPress  = "pressure_station"

[station.units]
temp             = "degF"       # canonical channel  ->  its unit in YOUR data
wind_speed       = "mph"
rain_counter     = "inch"
pressure_station = "inHg"
```

Everything is converted to metric internally (°C, m/s, mm, hPa). The defaults
above cover a standard AmbientWeather unit; if yours reports Celsius, write
`temp = "degC"` and it will pass through unchanged.

### `[forecasts]` — where the providers' opinions come from

```toml
[forecasts]
db_path = "crestline_forecasts.sqlite"
immutable = false              # false while the collector writes this WAL DB
sources = []                    # empty = use every provider in the archive
max_forecast_age_hours = 12.0   # a forecast older than this is not used
```

### The rest

```toml
[dataset]
dir = "data"                    # where parquet outputs go (git-ignored)
precip_reset_fraction = 0.5     # a rain-counter drop below this fraction of the
                                # prior value is a real reset; a smaller dip is noise

[reports]
dir = "reports"                 # where markdown leaderboards go

[artifacts]
dir = "artifacts"               # alignment studies and promoted releases
```

Defaults are sensible for `[qc]`, `[provider_qc]`, `[backtest]` and `[predict]` —
you can ignore them until you read [Advanced usage](advanced-usage.md).

---

## Your first run

### 1. Check your truth

```bash
grounded-weather-forecast qc
```

This is the most important command to run first, and the one people skip. It
tells you what your station actually recorded, and what the quality-control
filters think of it:

```
observations: 76507 samples, 2025-06-14 18:39:53 .. 2026-07-13 19:21:03
┌──────────────────┬─────────┬─────────┬───────────────┬───────┬──────────┬───────┐
│ channel          ┆ samples ┆ missing ┆ out_of_bounds ┆ spike ┆ flatline ┆ clean │
╞══════════════════╪═════════╪═════════╪═══════════════╪═══════╪══════════╪═══════╡
│ temp             ┆ 76507   ┆ 2774    ┆ 0             ┆ 0     ┆ 0        ┆ 73733 │
│ pressure_station ┆ 76507   ┆ 24541   ┆ 0             ┆ 0     ┆ 0        ┆ 51966 │
...
```

**How to read this:**

- `missing` — the sensor reported nothing. Some of this is normal.
- `out_of_bounds` — physically implausible values (a −60 °C reading in July).
- `spike` — an isolated jump-and-return, the signature of a sensor glitch rather
  than real weather.
- `flatline` — the value never changed for hours. A stuck sensor.
- `clean` — usable as truth.

Flagged samples are **excluded**, never corrected. If `clean` is very low for a
channel, forecasts for that variable will be poorly trained and the system will
tell you so rather than quietly making things up.

It also prints how many *hours* and *days* survived aggregation. If those numbers
are much smaller than you expect, you have gaps — worth knowing before you draw
conclusions from anything downstream.

### 2. Build the dataset

If `[ensembles].models` is configured, ingest the latest ensemble cycles first:

```bash
grounded-weather-forecast ingest-ensembles
```

Ensemble statistics are materialized into the matrix, so every later ingest
must be followed by another `build-dataset` before backtesting or serving.

```bash
grounded-weather-forecast build-dataset
```

This reads both databases and writes parquet files into `data/`: the QC'd truth
at minute, hour and day resolution, and the **supervised matrix** — one row per
(moment a forecast could have been made × hour it was about), with every
provider's opinion as a column and the truth that eventually happened as the
label.

```
dataset fingerprint: c5cdd0ee7777973f
sources: met_norway, meteosource, nws, open_meteo, pirate_weather, visual_crossing, weatherapi, weatherbit
snapshots: 2
  truth_minute: 76507 rows
  hourly_matrix: 700 rows
  ...
```

`snapshots: 2` is the number of distinct moments your archive can reconstruct. If
that number is small, your archive is young — see below.

### 3. Get a forecast

```bash
grounded-weather-forecast predict
```

This prints a JSON document with all three products. Trimmed:

```json
{
  "schema_version": 2,
  "issued_at": "2026-03-22T17:00:00+00:00",
  "status": "ready",
  "release_ids": ["df227d411b814f78"],
  "observation_at": "2026-03-22T16:55:49+00:00",
  "sources": ["met_norway", "nws", "open_meteo", "..."],
  "minutely": [
    {"valid_time": "...T17:01:00+00:00", "minutes_ahead": 1,
     "temp_c": 19.522, "precip_intensity_mmh": 0.0, "pop": 0.0}
  ],
  "hourly": [
    {"valid_time": "...T18:00:00+00:00", "lead_hours": 1.0, "lead_bucket": "1-3h",
     "values":  {"temp_c": 20.83, "wind_speed_ms": 1.2, "pop": 0.0},
     "methods": {"temp_c": "grounded_equal_weight", "pop": "equal_weight"}}
  ],
  "daily": [
    {"date_local": "2026-03-23", "lead_days": 1,
     "values":  {"temp_max_c": 23.8, "temp_min_c": 13.1, "pop": 0.0},
     "methods": {"temp_max_c": "gbm"}}
  ]
}
```

**Things worth noticing:**

- `observation_at` — the station reading the nowcast is *anchored* to. The
  minutely forecast starts from what your yard says right now and relaxes toward
  the providers' consensus over the hour. That live reading is the one input no
  provider has.
- `status` and `release_ids` — say whether compatible live evidence justified the
  forecast and identify the promoted decision. A young archive emits an explicit
  `degraded` equal-weight forecast rather than pretending grounding was fitted.
- `methods` — **every single value tells you which method produced it.** When you
  wonder why tomorrow's high is 23.8 °C, the answer is in the document.
- `lead_bucket` — how far ahead this is, grouped. Skill is measured per bucket
  because a method that wins at 2 hours often loses at 7 days.

Write it to a file instead of stdout:

```bash
grounded-weather-forecast predict --out forecast.json
```

---

## "It didn't work" — the two messages you will probably see

### `cannot predict: no provider forecast within 12.0h of ...`

Your archive's most recent forecast is more than 12 hours old, so the system
refuses to serve a stale forecast rather than pretending it is current. Either
your polling cron is not running, or you are testing with an old archive.

To load the exact archived document previously served at an instant—or, if none
was served then, reconstruct one using only data/evidence available by that
instant (handy for testing):

```bash
grounded-weather-forecast predict --now 2026-03-22T17:00:00
```

### `no rolling-origin folds. The archive spans 0.0 days ...`

You ran `backtest` and it found nothing to test. This is not a failure — it is the
system being honest. Backtesting means "train on the past, test on the future,
repeatedly", and that needs an archive with some *history*. With a 97-day
requirement (90 days training + a 7-day test step, both configurable) and a
one-day-old archive, there is nothing to do.

**Two ways forward**, and you should do both:

1. **Start the cron** so the archive begins accumulating (below).
2. **Backfill a synthetic archive** so you can measure something *today* — see
   [Advanced usage](advanced-usage.md#cold-start-backfilling-a-synthetic-archive).

---

## 4. Look at the leaderboard

Once you have an archive (real or backfilled) with some history:

```bash
grounded-weather-forecast backtest --source live      # or --source synthetic
grounded-weather-forecast report
```

`report` writes markdown into `reports/` and prints the winners:

```
winners (scores_hourly_synthetic)
┌─────────┬───────────────┬─────────────┬───────────────┬──────┬──────────┐
│ product ┆ variable      ┆ lead_bucket ┆ method_id     ┆ n    ┆ mae      │
╞═════════╪═══════════════╪═════════════╪═══════════════╪══════╪══════════╡
│ hourly  ┆ temp_c        ┆ 24-48h      ┆ gbm           ┆ 752  ┆ 1.288    │
│ hourly  ┆ temp_c        ┆ 96-168h     ┆ equal_weight  ┆ 2256 ┆ 1.748    │
└─────────┴───────────────┴─────────────┴───────────────┴──────┴──────────┘
```

`predict` then uses only winners from a live evaluation run compatible with the
current dataset. **You do not have to choose a method** — promotion chooses, and if
it has no evidence for a slice, the document is marked `degraded`, uses fit-free
equal weight, and says why.

The markdown report has much more: skill against the best single provider,
Diebold–Mariano significance tests (is that difference *real*, or noise?), a
provider error-correlation matrix, and — once you have served some forecasts — a
self-verification section scoring what this system actually predicted against
what actually happened.

---

## 5. Start the cron (do this today)

This is the highest-value thing you can do, and it is not part of grounded-weather-forecast —
it belongs to the upstream `omni-weather-forecast-apis` project. Something like:

```cron
# hourly for a diverse core of providers
0 * * * *  cd /path && uv run omni-weather --config config.toml --sqlite forecasts.sqlite

# every 6 hours for the rest (respects free-tier quotas)
0 */6 * * * cd /path && uv run omni-weather --config config-extended.toml --sqlite forecasts.sqlite
```

Poll a **diverse** core hourly — Open-Meteo with several *explicit* models
(`ecmwf_ifs025`, `gfs_seamless`, `icon_seamless`), NWS, MET Norway, and one or two
commercial providers — and the rest every 6 hours. Diversity matters more than
count: this project's own measurements show eight providers behaving like fewer
than two *independent* ones, because most of them repackage the same underlying
weather models.

Every week you delay is a week of training data that cannot be recovered.

---

## Where to go next

- **[Advanced usage](advanced-usage.md)** — backfilling, tuning, adding your own
  blending method, reading the leaderboard properly.
- **[Theory](theory.md)** — why the system does what it does.
- **[Limitations](limitations.md)** — what it cannot do, and three real bugs the
  evaluation harness caught. Read this before trusting any number.
- **`CONTEXT.md`** — the project glossary (issue time, valid time, lead, grounding,
  anchoring, …). If a word in the output confuses you, it is defined there.
