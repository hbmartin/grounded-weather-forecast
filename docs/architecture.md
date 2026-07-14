# Architecture

This page describes how omni-forecast is built: the layers, the contracts
between them, the storage formats, the libraries, and the tooling gates. It is
written for someone who is about to change the code.

For *why* the algorithms are what they are, see [Theory](theory.md).

---

## 1. Shape of the system

```
        aw2sqlite.db                     crestline_forecasts.sqlite
     (station observations)                (provider forecast archive)
              |                                        |
              |                                        |            Open-Meteo
              v                                        v          Previous Runs API
     +-----------------+                     +------------------+        |
     | dataset/station |                     | dataset/providers|<-------+
     | dataset/qc      |                     | dataset/snapshots|  dataset/backfill
     | dataset/truth   |                     +------------------+
     +-----------------+                              |
              |                                       |
              +------------------+--------------------+
                                 v
                        +------------------+
                        |  dataset/matrix  |   the supervised matrix:
                        +------------------+   one row per (snapshot, valid time)
                                 |
                +----------------+-----------------+
                v                                  v
       +----------------+                  +----------------+
       |   backtest/    |   scores.parquet |    serve/      |
       |  splits+engine |----------------->|   selection    |
       +----------------+        ^         |   predict      |
                |                |         +----------------+
                v                |                  |
       +----------------+        |                  v
       |    reports/    |--------+          forecast JSON  ---> predict_history
       |  leaderboard   |  self-verification <-------------------------+
       +----------------+
                ^
                |
       +----------------+     +----------------+
       |    blenders/   |     |    metrics/    |
       |  (the registry)|     | MAE/CRPS/DM... |
       +----------------+     +----------------+
```

Two rules keep this from becoming a ball of mud:

1. **`contracts.py` and `leads.py` are the only deep-import targets.** Everything
   else talks through them.
2. **Blenders import `contracts` only — never `dataset`.** A blending method must
   be expressible purely in terms of "here is a matrix of numbers and an
   availability mask". If a method needs to reach into the dataset layer, that is
   a signal the contract is wrong, not that the rule should be bent.

---

## 2. The contract

Everything hinges on four frozen dataclasses and one protocol, all in
`contracts.py`.

```python
@dataclass(frozen=True)
class ForecastMatrix:
    sources:      tuple[str, ...]   # column identity and order
    values:       FloatArray        # (n, k) float64, NaN where unavailable
    availability: BoolArray         # (n, k) — explicitly ~isnan(values)
    lead_hours:   FloatArray        # (n,)
    features:     pl.DataFrame      # aligned context; NEVER truth columns
    product:      Product           # controls lead buckets/capabilities

@dataclass(frozen=True)
class SupervisedSlice:
    x: ForecastMatrix
    y: FloatArray                   # truth; contains no NaN by construction
    variable: VariableSpec
    source_kind: SourceKind         # LIVE | SYNTHETIC — one kind, enforced

@dataclass(frozen=True)
class BlendResult:
    point:            FloatArray
    quantiles:        FloatArray | None = None    # (n, q) — reserved for wave 2
    quantile_levels:  tuple[float, ...] = ()

class Blender(Protocol):
    method_id: str
    def fit(self, train: SupervisedSlice) -> Self: ...
    def predict(self, x: ForecastMatrix) -> BlendResult: ...
```

Design decisions worth knowing:

- **`availability` is materialized, not derived on the fly.** It is
  `~np.isnan(values)`, but every blender reads the same array rather than each
  re-deriving it, so "available" cannot mean two different things in two places.
  Every blender **must** renormalize over available sources — this is what makes
  a 24-hour provider and a 360-hour provider coexist without special-casing.
- **The `features` frame may not contain truth.** `ForecastMatrix.__post_init__`
  raises if any column starts with `t__`. This is a runtime guard against the
  most likely form of leakage.
- **Quantiles travel end to end** through scores, forecast JSON, and served history.
  Wave-1 methods are point forecasters, but EMOS/conformal can be added without a
  storage or API break.
- **`BlendResult.point` may contain `NaN`**, which means "this method has nothing
  to say about this row" (e.g. persistence with no recent observation). The engine
  stores that as `null`; promotion compares all methods on one common-case mask and
  reports coverage, so declining difficult rows cannot manufacture a win.

### Column naming

One convention, one place (`contracts.py` owns the builders and the parser):

| Pattern | Meaning |
|---|---|
| `fx__{source}__{var}` | hourly forecast from a source |
| `fxd__{source}__{var}` | daily forecast from a source |
| `age__{source}` | hours between that source's fetch and the snapshot |
| `obs__{var}` | station observation **at issue time** (past data — leakage-safe; the anchoring input) |
| `t__{var}__inst` / `t__{var}__mean` | truth, dual semantics |
| `t__{var}` | truth, single semantics (gust, precip, PoP) |
| `ewagg__{var}` | equal-weight aggregate of the hourly path over a local day |

`__` is the structural separator. Column builders percent-escape separator and
percent characters inside source/variable segments; `parse_fx_col()` is the exact
inverse and refuses anything malformed.

---

## 3. The dataset layer

### Readers

- `dataset/station.py` — opens the observation DB **read-only** by URI
  (`?immutable=1` for a static snapshot; `mode=ro` for a live WAL database — never
  `immutable` on a file another process is writing). Handles *mixed timestamp
  precision*: the sample DB has 41k rows with microseconds and 35k without, so the
  parser tries both formats and coalesces. Missing tables, missing columns and
  NULL timestamps all degrade gracefully to empty/null rather than crashing — the
  upstream databases are explicitly documented as damaged.
- `dataset/providers.py` — reads hourly, daily, minutely, and completion rows in one
  SQLite transaction, filters every run to the configured coordinates, and joins
  `hourly_points → source_forecasts →
  provider_results`, filters to `status = 'success'`, and **recomputes lead from
  the ISO `fetched_at` text**. It never reads `horizon_hours`, `fetched_at_unix`
  or `run_cycle`, all of which are NULL in the sample archive. A source slug is
  `provider` normally, `provider_model` when a provider exposes several models
  (so `open_meteo_ecmwf_ifs025` is distinct from `open_meteo_gfs_seamless`).
- `dataset/backfill.py` — the Open-Meteo Previous Runs adapter. The HTTP fetcher
  is **injected**, so tests never touch the network.

### Snapshots

`dataset/snapshots.py` implements the as-of join: `snapshot_times()` dedupes
`forecast_runs.completed_at` onto a 10-minute grid, and `as_of_selection()` uses
a `join_asof(..., strategy="backward", tolerance=max_age)` per source. The same
function serves a *single* snapshot at "now" when predicting — training and
serving share one code path by construction.

### Matrices

`dataset/matrix.py` pivots long → wide, joins truth, and adds calendar features,
source ages, and issue-time observations. Two subtleties:

- **Deterministic column order.** Polars' `pivot` and `unique` do not guarantee
  order, so the frames are stably sorted before pivoting. Without this, rebuilding
  the same dataset produced different parquet bytes and the fingerprint changed —
  which would make artifact staleness detection meaningless.
- **`to_forecast_matrix(..., sources=...)`.** Serving pins the *training* source
  list, so a provider that is merely missing right now becomes an unavailable
  column rather than shifting every other blender's column indices.

---

## 4. Storage

Everything under `[dataset].dir` (git-ignored), all parquet:

| File | Grain |
|---|---|
| `truth_minute.parquet` | one row per station sample; values null when QC-flagged, plus `{var}_qc` bitmask |
| `truth_hourly.parquet` | one row per hour; **both** truth semantics for state vars, plus coverage counters |
| `truth_daily.parquet` | one row per local day; extremes and sums with DST-aware coverage |
| `forecasts_long.parquet` | one row per (source, fetched_at, valid_time) |
| `daily_long.parquet`, `minutely_long.parquet` | as above, daily / minutely grain |
| `hourly_matrix_live.parquet` | **one row per (snapshot, valid hour)** — the supervised matrix |
| `daily_matrix_live.parquet` | one row per (snapshot, target local date) |
| `hourly_matrix_synthetic.parquet` | the same shape, backfilled provenance |
| `manifest.json` | row counts, per-file SHA-256, and the **dataset fingerprint** |
| `scores/scores_{product}_{kind}_{window}_{evaluation}.parquet` | one identified evaluation run, one row per (method, variable, test case) |
| `predict_history.parquet` | every emitted value, atomically appended with release/method/quantile attribution |
| `served_forecasts/*.json` | exact versioned documents for historical replay |

**Matrices are keyed by provenance in the filename.** `..._live` and
`..._synthetic` can never collide on disk, which makes the provenance wall a
property of the filesystem and not merely of a runtime check.

The **scores frame** is the pivot of the whole design. It is deliberately *long
and dumb*:

```
evaluation_id | dataset_fingerprint | source_set_json | semantics | code_version
method_id | variable | product | source_kind | window | fold_origin
          | issue_time | valid_time | lead_hours | lead_bucket | y_pred | y_true
          | quantile_levels_json | quantiles_json
```

The engine writes it and stops. Every leaderboard, skill score, DM test and
winner-selection is a `group_by` over this frame, computed downstream. That is
what makes it possible to add a new report without re-running a 45-fold backtest,
and what keeps the engine from ever being tempted to declare a winner.

---

## 5. Blenders and the registry

`blenders/registry.py` maps `method_id → factory`. It stores **factories, never
instances** — the backtest engine calls the factory once per fold. This is not a
style preference; it is a leakage defence, and there is a test that asserts the
engine really does construct distinct objects.

Modules self-register on import; `blenders/__init__.py` imports them all.

| module | methods |
|---|---|
| `baselines.py` | `persistence`, `climatology`, `best_provider`, `equal_weight` |
| `grounding.py` | (not a method — the `AffineGrounding` stage others compose) |
| `combine.py` | `grounded_equal_weight`, `affine_equal_weight`, `inverse_mse` |
| `anchoring.py` | `anchored_grounded_equal_weight`, `anchored_inverse_mse` |
| `gbm.py` | `gbm` |
| `experts.py` | `ewa`, `boa` |

Two shared abstractions live in `blenders/protocol.py`:

- `renormalize_weights(w, availability)` / `masked_average(...)` — the availability
  contract, implemented once.
- `PerBucketFitter[S]` / `FittedBuckets[S]` — fit one state object per lead bucket,
  with a global-fit fallback for thin buckets. Grounding, inverse-MSE and
  best-provider all use it; the generic parameter `S` is the per-bucket state
  (coefficients, a weight vector, a source ranking).

**`Anchored` is a wrapper, not a special case.** It takes a base blender factory
and implements the same protocol, so it appears on the leaderboard as its own
method and the engine needs no anchoring-specific code.

**LightGBM is imported lazily** via `importlib`, and `gbm` registers only if the
import succeeds — so the package stays importable on platforms where its wheels
lag (a real concern on new CPython releases). `deptry` is told about the dynamic
import explicitly.

---

## 6. Backtest engine

```python
for fold in fold_plans(issue_time, truth_known_at, config, window):
    for variable in request.variables:
        train = to_supervised_slice(matrix[fold.train_rows], variable, ...)
        test  = to_supervised_slice(matrix[fold.test_rows],  variable, ...)
        for method_id in request.methods:
            blender = get_factory(method_id)()      # fresh instance, every fold
            scores.append(blender.fit(train).predict(test.x))
```

`splits.py` computes fold plans in **epoch microseconds** (integers) rather than
datetimes, which sidesteps a pile of timezone-aware/naive arithmetic hazards and
makes the invariants trivially checkable. Fold plans are plain data — inspectable
and testable with no model in the loop.

---

## 7. Serving

```
build_snapshot(config, now)          # the same as-of code the dataset build uses
  -> fit each selected method on the LIVE matrix's scoreable history
  -> predict the snapshot's future rows
  -> hourly product
  -> minutely product   (interpolate + decayed observation residual)
  -> daily product      (own supervised targets, ewagg__* features ride along)
  -> Forecast document (schema_version, provenance, per-value method attribution)
  -> append to predict_history.parquet
```

`serve/selection.py` reads only live Evaluation Runs compatible with the current
dataset and requested issue time, then promotes the common-case, significance-aware
per-slice winners into a Model Release. Config pins override. A slice with no
compatible evidence uses the fit-free `equal_weight` fallback, marks the document
`degraded`, and records the reason.

Every emitted value carries the `method_id` that produced it. When someone asks
"why is the 6 a.m. temperature 12 °C", the answer is in the document.

---

## 8. Library choices

| Chosen | Why |
|---|---|
| **polars** | Expression API, real nullability (distinct from NaN — which matters enormously when "no truth" and "not a number" are different things), fast group-by/as-of joins, strict schemas. |
| **numpy** | The model boundary. Every blender sees plain float64 arrays. |
| **scipy** | Exactly one function: Student-t survival, for the Diebold–Mariano p-value. |
| **scoringrules** | CRPS. Wrapped entirely inside `metrics/probabilistic.py`, so swapping it out is a one-file change. |
| **lightgbm** | The stacker. Native missing-value handling is the feature that matters. |
| **stdlib** `sqlite3`, `tomllib`, `argparse`, `zoneinfo` | No dependency justifies replacing any of these. |

Deliberately **not** used, and why:

- **pandas** — polars' nullability semantics and expression engine are a better
  fit, and we would gain nothing.
- **duckdb** — the joins are not hard enough to justify a second engine for a
  small-data problem.
- **statsmodels / a Kalman filter** — see [Theory](theory.md#4-the-central-decomposition):
  the state-space formulation is a monolith that cannot be A/B tested in stages,
  and carries permanent covariance-tuning cost. The decomposition (grounding +
  weighting + anchoring) gets most of it in inspectable pieces.
- **hierarchicalforecast / MinT** — structurally inapplicable to max/min targets
  ([ADR 0002](adr/0002-anchored-nowcast-instead-of-temporal-disaggregation.md)).
- **opera (online aggregation)** — EWA/BOA with fixed share is ~150 lines and we
  needed to control the sleeping-expert and fixed-share details precisely.

The upstream projects are **not** imported. The two SQLite files and their
documented schemas are the entire interface
([ADR 0001](adr/0001-sqlite-files-are-the-only-upstream-contract.md)).

---

## 9. Correctness gates

Every change must survive:

```bash
uv run ruff check src --fix        # lint.select = ALL
uv run ruff format src tests
uv run pyrefly check src           # two independent type checkers,
uv run ty check src                #   because they disagree usefully
uv run lizard -Eduplicate -C 27 src  # complexity ceiling + duplication
uv run deptry src                  # declared deps == imported deps
uv run pytest tests/ --cov=src     # >= 88% coverage (currently ~95%)
```

Type hints are first-class: `pyrefly` and `ty` both run, and they catch different
things (`ty` in particular refuses to narrow a match-guard capture, which forced
a genuinely clearer `BlendResult` validation).

### Testing strategy

- **Fixture databases are synthesized, never sampled.** `*.db`/`*.sqlite` are
  git-ignored, so CI has no sample data. `tests/conftest.py` builds
  aw2sqlite-shaped and omni-weather-shaped SQLite files programmatically —
  including the *damage* (NULL `horizon_hours`, NULL `fetched_at_unix`, mixed
  timestamp precision), so the graceful-degradation paths are actually exercised.
- **A synthetic weather generator** (`synthetic_hourly_matrix`) produces a
  sinusoidal climate with *known* provider biases, *known* noise levels, and
  optional ragged horizons. This is what makes **directional** assertions possible:
  "grounding must beat raw on biased sources", "anchoring must win at 0–3 h and
  converge by 12 h", "the experts must concentrate on the better expert".
- **Protocol compliance is parametrized over the entire registry**, so a new
  blender is automatically tested for shape correctness and for invariance to an
  all-NaN (never-available) source column.
- **The leakage gauntlet** — see [Theory](theory.md#leakage-assumed-present-until-proven-absent).
  The poisoning sentinel is the one to keep alive at all costs.

### Determinism

The dataset build is byte-reproducible (stable sorts before every pivot), and the
manifest records a **fingerprint** over the file hashes. Fitted artifacts are
stored under that fingerprint, so `predict` can refuse to serve a model fitted
against a dataset that no longer exists. LightGBM is seeded and run with
`deterministic: true`.

---

## 10. Scale

Sized for one station with years of data, on a laptop:

- 3 years of minute truth ≈ 1.6 M rows.
- The hourly matrix is one row per (snapshot, valid hour): with ~26 snapshots/day
  and 150–300 valid hours each, ≈ 5–8 M rows × ~110 columns over 3 years — a 1–2 GB
  parquet, read with column projection.
- The real 13-month backfill in this repo produced 198k forecast points → a 66k-row
  synthetic matrix; a 12-method × 3-variable × 45-fold backtest over it runs in
  well under two minutes.

No premature optimization has been done, and none appears to be needed. The one
thing to watch is that the online experts are a Python row loop (`O(n·k)`) —
fine at this scale, the first thing to vectorize if it ever isn't.
