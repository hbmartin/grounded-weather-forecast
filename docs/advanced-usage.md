# Advanced usage

For readers who have run the basic pipeline and now want to backfill, tune,
extend, or interrogate it. Assumes [Getting started](getting-started.md) and a
skim of [Theory](theory.md).

---

## Cold start: backfilling a synthetic archive

A forecast archive cannot be reconstructed after the fact — with one exception.
Open-Meteo's **Previous Runs API** serves archived forecasts at fixed day
offsets: `temperature_2m_previous_day3` is what the model predicted for *this*
valid hour, three days earlier. That is a real forecast archive with lead as a
controlled variable, and it can be fetched today.

```toml
[backfill.open_meteo]
models = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"]
start_date = 2025-06-15         # no earlier than your station truth begins
```

```bash
omni-forecast backfill --end 2026-07-12
# backfilled 198072 forecast points
# sources: open_meteo_ecmwf_ifs025, open_meteo_gfs_seamless, open_meteo_icon_seamless
# synthetic matrix: 66024 rows -> data/hourly_matrix_synthetic.parquet
```

Then backtest against it exactly as you would against live data:

```bash
omni-forecast backtest --source synthetic --products hourly,daily
omni-forecast report
```

### What backfilled data is good for — and what it is not

| Good for | Not valid for |
|---|---|
| Validating the entire pipeline end-to-end **today**. | Any conclusion about commercial providers (Google, Tomorrow.io, …) — they cannot be backfilled. |
| Ranking methods at 24–240 h leads with real skill numbers. | **Leads under 24 h.** Previous Runs offsets are whole days, so those buckets are empty. |
| Measuring provider error correlation and effective ensemble size. | Evaluating **anchoring**, which needs a short-lead row to compute the residual. |
| Fitting grounding coefficients for open NWP models. | Choosing a serving method for your *live* provider set. |

The day-0 (unsuffixed) Previous Runs field is **deliberately not requested**: for
a past hour it returns the *latest* run, whose effective lead is near zero.
Including it would fill the short-lead buckets with what is essentially an
analysis and make anchoring look miraculous. Its absence is why the short-lead gap
above cannot be papered over.

Synthetic and live rows are stored in separate files, tagged `source_kind`, and
any attempt to pool them raises `MixedProvenanceError`. This is deliberate: a
leaderboard built on three NWP models says nothing about eight consumer APIs.

Flags: `--models` (override the config list), `--chunk-days` (default 90 —
requests are chunked to keep URLs and responses sane), `--end` (default:
yesterday).

---

## Driving the backtest

```bash
omni-forecast backtest \
    --source live|synthetic \
    --products hourly,daily \
    --methods all \
    --hourly-variables temp_c,wind_speed_ms,pop \
    --window expanding \
    --semantics auto
```

### `--window expanding` vs `rolling`

Two different questions:

- **expanding** — "what does the system know, given everything it has ever seen?"
  Trains from the archive start to each fold origin.
- **rolling** — "what has it learned *lately*?" Trains on a fixed trailing window
  (`[backtest].rolling_window_days`, default 180).

Run both. If rolling wins, your providers are drifting or your training window is
seasonally unrepresentative — and that is exactly the condition under which a
free-slope grounding correction becomes dangerous
([ADR 0004](adr/0004-grounding-defaults-to-bias-only.md)).

!!! note "Rolling needs a dense archive"
    With a gappy station record, a 180-day rolling window can contain *zero*
    scoreable rows and the fold is skipped. On the sample data this makes rolling
    uninformative — see [Limitations](limitations.md#2-the-station-has-holes).

### `--semantics auto|inst|mean`

Whether a provider's "14:00 temperature" means *at* 14:00 or the *mean over*
14:00–15:00. Providers do not document it, and guessing wrong manufactures ~1 °C
of fake bias.

```bash
omni-forecast alignment
# recommended: {'temp_c': 'inst', 'humidity_pct': 'mean', 'dew_point_c': 'inst', ...}
# wrote artifacts/alignment.json
```

`alignment` correlates every source's forecasts against *both* truth definitions
and writes one canonical recommendation per variable. `--semantics auto` (the
default) applies each variable's own recommendation; `inst`/`mean` force one. The
study needs ≥72 overlapping rows per source×variable to make a call, and reports
`null` rather than guessing when it cannot.

### `--methods`

Twelve are registered:

```
persistence  climatology  best_provider  equal_weight        <- baselines (the floor)
grounded_equal_weight  affine_equal_weight  inverse_mse      <- grounding + weighting
anchored_grounded_equal_weight  anchored_inverse_mse         <- + short-lead anchoring
gbm                                                          <- the nonlinear ceiling
ewa  boa                                                     <- online experts
```

`--methods all` runs everything (recommended — the baselines are the point). Or
name a subset: `--methods gbm,equal_weight,best_provider`.

---

## Reading the leaderboard properly

`reports/leaderboard_hourly_*.md` has four sections. The one that matters is the
per-slice board:

| lead_bucket | method_id | n | mae | bias | skill_vs_best_provider | dm_p_vs_best_provider |
|---|---|---|---|---|---|---|
| 24-48h | gbm | 752 | 1.288 | **+0.119** | **+0.160** | **0.023** |
| 24-48h | equal_weight | 752 | 1.336 | +0.012 | +0.129 | 0.048 |
| 24-48h | affine_equal_weight | 752 | 1.471 | **+0.559** | +0.041 | 0.639 |
| 24-48h | best_provider | 752 | 1.534 | −0.983 | — | — |

**Read the columns in this order:**

1. **`n`** — how many paired samples. A slice with n = 40 is a rumour, not a
   result. It is printed first for a reason.
2. **`skill_vs_best_provider`** — `1 − MAE/MAE_ref`. Positive means you beat the
   best single provider. Also reported against grounded equal-weight, because "we
   beat the *worst* provider" is not a claim worth making.
3. **`dm_p_vs_...`** — the Diebold–Mariano p-value: **is this difference real?**
   It uses a Bartlett HAC variance (consecutive multi-hour forecast errors are
   serially correlated, and the naive variance is far too small) and the
   Harvey–Leybourne–Newbold small-sample correction. `p = 0.639` on
   `affine_equal_weight` above means: that 4% "skill" is noise.
4. **`bias`** — the mean error. **Do not skip this column.** A method with
   unremarkable MAE and a large bias is *systematically and correctably* wrong —
   and it is exactly this column that exposed grounding injecting a +1.4 °C warm
   bias ([Limitations §4.1](limitations.md#41-grounding-was-making-forecasts-worse)).

Also in the report: `pct_within` (a consumer-legible "within 3 °F" rate), `brier`
for PoP, an **error-correlation matrix** across providers (which tells you how
many *effective* independent sources you really have), and — once you have served
forecasts — a **self-verification** section.

---

## Serving

```bash
omni-forecast predict --out forecast.json
```

| flag | effect |
|---|---|
| `--method <id>` | Force one method for every slice, ignoring the leaderboard. Useful for A/B'ing. |
| `--now <iso>` | Return the exact archived document if that instant was served; otherwise reconstruct causally using only evidence available then. |
| `--no-history` | Do not append to the self-verification history. |
| `--semantics` | As for `backtest`. |
| `--out -` | stdout (default). |

Method selection resolves in this order:

```
[predict.methods] config pin   ->   per-slice backtest winner   ->   named fallback
```

Only live scores matching the current dataset, source set, and requested issue time
are promotable. The cold-start fallback is fit-free `equal_weight`; the document is
marked `status = "degraded"` and records the reason. **It never presents an
unfitted grounded method as trained.**

Pin a method when you have a reason to override the leaderboard:

```toml
[predict.methods]
"hourly.temp_c" = "gbm"
"daily.temp_max_c" = "grounded_equal_weight"
```

### The self-verification loop

Every served forecast is archived as an exact JSON document and appended under a
file lock to `data/predict_history.parquet`. Later, when product-specific minute,
hourly, or daily truth arrives, `report` scores it:

```
## Self-verification (served vs realized)

| product | variable | method_id | n | live_mae | live_rmse | live_bias | backtest_mae | mae_gap |
| hourly  | temp_c   | grounded_equal_weight | 45 | 1.206 | 1.477 | -0.335 | 1.775 | -0.569 |
```

**Why this matters:** backtest skill is an *estimate* of live skill. `mae_gap` is
the difference between what the backtest promised and what actually happened. A
large positive gap means the serving path has quietly diverged from the
backtested one — a class of bug that a backtest, by construction, can never catch
on its own.

This section is attached **only** to the live leaderboard, because comparing
live-provider forecasts against a synthetic-source backtest would be
apples-to-oranges.

---

## Tuning

All of these are code constants with deliberate defaults. Change them knowing
what they trade.

### Grounding slope (`blenders/grounding.py`)

```python
AffineGrounding(slope_shrinkage=0.0)   # 0.0 = bias-only  (default)
                                       # 1.0 = free least-squares slope
                                       # b = 1 + λ·(b_ols − 1)
```

The default is bias-only for a measured reason
([ADR 0004](adr/0004-grounding-defaults-to-bias-only.md)). Both variants are on
the leaderboard (`grounded_equal_weight` vs `affine_equal_weight`) — **let the
data tell you** when your archive is seasonally representative enough to earn the
slope back. Watch `affine_equal_weight`'s `bias` column: while it is large and
positive, the slope is not earning anything.

### Fixed share (`blenders/experts.py`)

```python
_SHARE = 0.005     # the drift-vs-average-case knob
```

The loser's steady-state weight is roughly `share / (2 · learning rate)`. Raising
it adapts faster to a provider swapping its model, at the cost of permanently
carrying weight on a worse expert. The sweep that produced 0.005 is in
[Limitations §4.2](limitations.md#42-the-online-experts-could-not-adapt-to-drift);
0.02 cost 24–30% average-case MAE for no benefit.

### Anchoring decay (`blenders/anchoring.py`)

```python
TAU_GRID_HOURS = (0.5, 1.0, 2.0, 3.0, 6.0, 12.0, 24.0)
_ANCHOR_MAX_LEAD = 3.0     # a snapshot needs a row under 3h to have an anchor
_WEIGHT_FLOOR = 0.05       # below this the correction is zeroed
```

`τ` is fitted per variable by grid search, and **"no anchoring" is on the grid** —
so if the residual carries no signal, anchoring degrades exactly to its base
blend rather than adding noise.

### GBM (`blenders/gbm.py`)

`objective: regression_l1` (we score MAE, so we train MAE), 300 rounds, lr 0.05,
31 leaves, fixed seed, `deterministic: true`. Features: grounded source values,
lead, hour-of-day, month, source ages, issue-time observation, ensemble spread,
count of available sources. **Do not add a truth-derived feature** — the
`ForecastMatrix` contract will raise, and the poisoning sentinel will fail.

### Lead buckets (`leads.py`)

```
0-1h  1-3h  3-6h  6-12h  12-24h  24-48h  48-96h  96-168h  168-240h  240h+
D1  D2  D3-4  D5-7  D8-10
```

Quasi-logarithmic (error grows that way), with edges on the provider raggedness
cliffs (24/48/168/240 h — where providers' horizons actually end). `leads.py` is
the single source of truth: change it there and every fit, every score and every
report follows. Fewer buckets = more data per bucket but less lead resolution.

### QC thresholds (`[qc]` in config)

```toml
[qc.bounds]
temp = [-40.0, 55.0]          # metric units, per canonical channel
[qc.max_step]
temp = 4.0                    # per minute; scales with the actual sample gap
[qc.flatline_minutes]
temp = 180                    # a run of identical values this long is a stuck sensor
```

---

## Adding a blending method

A blender is anything satisfying the protocol. That is the whole interface:

```python
# src/omni_forecast/blenders/mine.py
from dataclasses import dataclass
from typing import Self
import numpy as np

from omni_forecast.blenders.grounding import AffineGrounding
from omni_forecast.blenders.protocol import finalize_point, masked_average
from omni_forecast.blenders.registry import register
from omni_forecast.contracts import (
    BlendResult, ForecastMatrix, SupervisedSlice, TargetKind,
)


@dataclass
class MedianBlend:
    method_id: str = "median"
    _kind: TargetKind = TargetKind.CONTINUOUS

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._grounding = AffineGrounding().fit(train)   # compose the stages
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        corrected = self._grounding.transform(x)
        point = np.nanmedian(corrected, axis=1)          # NaN = unavailable source
        return BlendResult(point=finalize_point(point, self._kind))


register("median", MedianBlend)
```

Then add the import to `blenders/__init__.py`. That's it — it now appears in
`--methods all`, gets backtested, can win slices, and can be selected by
`predict`.

**Four rules the protocol enforces:**

1. **Renormalize over available sources.** `x.availability` is a `(n, k)` bool
   array; `masked_average` and `renormalize_weights` in `blenders/protocol.py` do
   this correctly. A method that assumes all sources are present will produce
   garbage the first time a provider is late.
2. **Never touch truth in `predict`.** `x.features` cannot contain truth columns
   (the contract raises), and the poisoning sentinel will catch you if you find
   another route.
3. **`NaN` means "no opinion"**, not zero. The engine stores it as null and scores
   promotion candidates on one common-case mask, with coverage shown.
4. **Be constructible fresh.** The registry stores a *factory*; the engine builds a
   new instance per fold. Do not cache state on the class.

**Tests you get for free:** the protocol-compliance suite is parametrized over the
whole registry, so your method is automatically checked for output shape and for
invariance to a never-available source column.

**Tests you should write:** a *directional* one, on synthetic data where you know
the answer. `tests/conftest.py::synthetic_hourly_matrix` makes a sinusoidal
climate with known provider biases, known noise levels, and optional ragged
horizons:

```python
def test_median_resists_an_outlier_provider():
    matrix = synthetic_hourly_matrix(days=40, biases={"alpha": 20.0})  # alpha is broken
    train = to_supervised_slice(matrix, hourly_variable("temp_c"))
    median = get_factory("median")().fit(train).predict(train.x)
    mean   = get_factory("equal_weight")().fit(train).predict(train.x)
    assert mae(median.point, train.y) < mae(mean.point, train.y)
```

That is what the fixture is *for*: assertions about behaviour, not about shapes.

---

## Programmatic use

The CLI is a thin wrapper. Everything is importable:

```python
from pathlib import Path
import polars as pl

from omni_forecast.config import load_config
from omni_forecast.contracts import hourly_variable
from omni_forecast.dataset.matrix import to_supervised_slice, matrix_path
from omni_forecast.backtest.engine import BacktestRequest, run_backtest
from omni_forecast.reports.leaderboard import leaderboard, slice_winners
from omni_forecast.serve.predict import predict
from omni_forecast.serve.selection import select_methods

config = load_config(Path("config.toml"))

# a backtest
matrix = pl.read_parquet(matrix_path(config.dataset.dir, "hourly", "synthetic"))
scores = run_backtest(
    matrix,
    BacktestRequest(
        variables=(hourly_variable("temp_c"),),
        methods=("gbm", "equal_weight", "boa"),
        window="rolling",
    ),
    config,
)
print(slice_winners(leaderboard(scores)))

# a forecast
selections = select_methods(config, config.dataset.dir / "scores")
document = predict(config, selections)
print(document.to_json())
```

The scores frame is deliberately long and dumb — `method_id, variable, product,
source_kind, window, fold_origin, issue_time, valid_time, lead_hours, lead_bucket,
y_pred, y_true`. Every report is a `group_by` over it, so a *new* analysis costs a
polars expression, not a 45-fold re-run.

---

## The polling cron

Not part of this repo (it belongs to `omni-weather-forecast-apis`), but it
determines everything this system can ever learn.

**Recommended split:**

- **Hourly**, a diverse core: Open-Meteo with several *explicit* models
  (`ecmwf_ifs025`, `gfs_seamless`, `icon_seamless` — not just `best_match`), NWS,
  MET Norway, plus one or two commercial providers.
- **Every 6 hours**, everything else — this respects free-tier quotas, and the
  quota tracker will enforce them anyway.

**Diversity beats count.** This project measured an error correlation of ρ ≈ 0.51
across eight live providers, giving an *effective* ensemble of **1.8 independent
sources**. Adding a ninth provider that repackages GFS buys you almost nothing;
adding one with a genuinely different modelling stack buys you a lot.

The harness never assumes a uniform cadence — leads are always recomputed from
`fetched_at`, and the as-of snapshot logic means a 6-hourly provider is used at
its true age rather than being duplicated across hourly rows.

---

## Artifacts and reproducibility

The dataset build is byte-reproducible and records a **fingerprint** over the file
hashes:

```
dataset fingerprint: c5cdd0ee7777973f
```

Evaluation Runs and promoted Model Releases are stored with dataset, source-set,
semantics, window, code, and configuration identity. Several blenders also expose
serializable state through the artifact-store API, but serving currently refits from
the causally compatible matrix for a newly requested issue time.

Every emitted document is archived. `predict --now <its issued_at>` returns that
exact archived document; an issue time never served is reconstructed causally.

---

## Where the bodies are buried

Before you trust a number this system produces, read
**[Limitations](limitations.md)**. In particular:

- A young live archive can have too few snapshots for any live leaderboard. Synthetic
  NWP results never justify serving choices for commercial providers.
- Nothing under 24 h lead has been evaluated on real data, including the anchoring
  stage.
- The evaluation harness has already caught three real bugs — grounding *injecting*
  bias, the online experts being unable to adapt to drift, and a climatology
  baseline so broken it flattered everything measured against it. It will catch
  more. That is the point of it.
