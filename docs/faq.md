# FAQ and troubleshooting

Common questions, every error message you are likely to hit, and what to do about
each.

If a term here is unfamiliar, the [Glossary](glossary.md) has it.

---

## Getting started

### How long before this is actually useful?

That depends entirely on **archive age**, and it is the binding constraint on
everything.

| Milestone | Requires |
|---|---|
| `qc` and `build-dataset` work | a station DB and any forecast archive |
| `predict` works | one provider forecast within the last 12 hours |
| `backtest` produces folds | `initial_train_days + step_days` — **97 days by default** |
| Promotions start clearing the gate | more, and the exact amount depends on your slice sizes |

Until then `predict` will run and emit a `degraded` forecast — an ungrounded
equal-weight blend. That is a real, usable forecast; it is just not the
evidence-backed one.

**The archive is the part you cannot rush.** You cannot ask a provider what it
predicted last Tuesday. Start the polling cron today, even if you plan to do
nothing else for three months. See [Scheduling](scheduling.md).

### Can I run this without waiting three months?

Partly. `backfill` fetches archived forecasts from open-data sources to build a
*synthetic* archive you can backtest against immediately:

```bash
grounded-weather-forecast backfill --provider open_meteo --start 2024-01-01
grounded-weather-forecast backtest --source synthetic
```

Two honest caveats. Open-Meteo's Previous Runs API gives leads at exact 24-hour
multiples, so the 0–24 h range — where the product actually lives and where
anchoring earns its keep — stays **entirely unevaluated**. And it covers three
open NWP models, so the resulting leaderboard says nothing about the commercial
providers you will actually serve from.

Synthetic and live results are never pooled. Read
[Limitations §3](limitations.md#3-what-the-synthetic-backfill-can-and-cannot-tell-you)
before drawing conclusions.

Using `--provider dynamical` gets you native sub-24h leads, at the cost of an
optional dependency (`uv sync --extra backfill`).

### Can I use a different station, or no station at all?

**A different station: yes.** Map your columns and units in
`[station.columns]` and `[station.units]` — see the
[configuration reference](reference/configuration.md#stationcolumns-db-column-canonical-channel).
The pipeline needs a monotone *event* rain counter for precipitation; a field
already reporting hourly accumulation is a different quantity and will need code
changes.

**No station: no.** The station is not an optional extra — it is the input no
provider has, and grounding and anchoring both depend on it. Without truth there
is also nothing to score against, so the leaderboard has nothing to arbitrate.

### Do I need API keys?

Not for the core system. The upstream collector needs whatever keys its providers
require, but this project reads SQLite files.

Optional: `truth_qc` uses NWS METAR neighbours with no key, and can additionally
use Synoptic Data with a free-tier token supplied as `"$SYNOPTIC_TOKEN"`.

---

## Error messages

### `cannot predict: no provider forecast within 12.0h of ...`

Your archive's most recent forecast is older than `max_forecast_age_hours`, so
the system refuses to serve a stale forecast rather than pretending it is current.

Either the polling cron is not running, or you are testing against an old
archive. To reproduce a document as of a past instant:

```bash
grounded-weather-forecast predict --now 2026-03-22T17:00:00
```

### `no rolling-origin folds. The archive spans 0.0 days ...`

`backtest` found nothing to test. Not a failure — the system being honest.
Backtesting means "train on the past, test on the future, repeatedly", which
needs history. With a 97-day requirement and a one-day-old archive there is
nothing to do.

Start the cron, and optionally backfill so you can measure something today.

### `MixedProvenanceError`

Something tried to combine live and synthetic rows. This is a guard, not a bug —
see [Methods: verification §7](methods/verification.md#7-the-provenance-wall).
Pass a single `--source`.

### `ContractViolationError: ... starts with 't__'`

A truth column reached a feature matrix. This is leakage defence #4 firing. If
you are adding a blender or a feature, the column naming rules are in
[Methods: notation §3](methods/notation.md#3-column-naming-as-a-type-system).

### A `ConfigError` naming a key

Configuration is validated at load with explicit messages. The named key has the
wrong type, is out of range, or is not one of the allowed choices — check it
against the [configuration reference](reference/configuration.md).

### `unknown method 'x'; available: ...`

A `--methods` value or a `[predict.methods]` pin names an unregistered method.
The error lists every valid id.

### A channel is 100% null after `qc`

Your `[station.columns]` mapping is wrong — the named DB column does not exist or
holds something else. This is exactly what `qc` is for; fix it before running
anything else, because every later command will fail more confusingly.

---

## Interpreting output

### Why does my forecast say `degraded`?

`status: "degraded"` means no promoted release matched the current dataset,
config, and code fingerprints, so serving fell back to `equal_weight` rather than
using evidence it cannot vouch for. `status_reason` says which. Common ones:

| Reason | Fix |
|---|---|
| no backtest evidence for this slice | not enough archive yet — wait, or backfill |
| implementation changed since the last backtest | re-run `backtest --source live` then `report` |
| dataset fingerprint changed | re-run `backtest` then `report` after `build-dataset` |
| configuration changed | as above |

The pattern: **anything that changes the system's identity invalidates the
evidence that justified a promotion.** Re-running `backtest` then `report`
re-earns it. See
[ADR 0005](adr/0005-promoted-model-releases-are-the-serving-boundary.md).

Degraded status prints to stderr, so `--out -` still gives clean JSON on stdout.

### Why is `n` so small on my leaderboard?

`n` is the number of scored cases in that slice, and it is printed precisely so a
thin slice announces its own lack of power rather than hiding behind a p-value.

Small `n` usually means: a short archive, a long lead bucket (a 240h+ bucket
needs 10 days of archive per case), a variable few providers publish, or a method
that **abstained** on most rows because it could not fit. The last is by design —
see [Methods: notation §6](methods/notation.md#6-abstention-is-a-first-class-outcome).

Eligibility requires `n ≥ 8` and `n_valid_times ≥ 8` before any promotion gate
even runs.

### Why did my served method change overnight?

Promotions are re-decided every `report`. Some churn is legitimate — new evidence
arrived — and some is noise. The system tracks this: `reports/selection_churn.md`
reports how often the served method changes, and a high churn rate is itself a
signal that slices are too thin to decide.

The promotion gate exists to suppress exactly this. If churn is high, consider
whether `[promotion] rule` should be the stricter `seq_mcs`.

### Why is a method missing from a slice entirely?

Either it is scoped out (precipitation heads are not offered temperature), or it
**abstained** — below its `_MIN_FIT_ROWS`, or lacking wet cases, or without a
short-lead anchor row. Abstention is a first-class outcome here: a method that
cannot fit says so rather than fitting badly on twelve rows.

### The leaderboard says equal weight is winning. Is something broken?

Probably not. A plain average of grounded sources being within a hair of the best
method is the **forecast combination puzzle**, a well-documented empirical result:
estimating weights adds variance faster than it removes bias. On this data raw
equal weight wins the 96–168 h temperature slice outright.

See [Concepts §3](concepts.md#3-why-averaging-more-forecasts-stops-helping).

### `anchored_*` methods have identical scores to their base

Expected on synthetic data. Anchoring needs a forecast row with lead under 6
hours to compute its residual, and the Previous Runs backfill starts at 24 h. With
no anchor row the τ search selects "no anchoring" and the wrapper degrades
exactly to its base. The mechanism works; that data cannot exercise it.

---

## Operations

### Where does each command write?

Summarized in [Outputs](reference/outputs.md); per-command in the
[CLI reference](reference/cli.md).

### What are the exit codes?

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | command-level failure — missing inputs, no scores, a backfill error |
| `2` | configuration error, or unknown command |

`truth-qc` deliberately returns `0` when it finds no evaluable checks: a cold
start is not a fault and should not page anyone.

### What is safe to delete?

`reports/` entirely — regenerated by `report`. `data/scores/` via `prune-scores`
(use `--dry-run` first).

**Not safe:** `data/predict_history.parquet` (the record of what you served),
`artifacts/history/` (append-only ledgers), and `artifacts/eprocess/` (sequential
accumulated evidence — deleting it restarts every gate from zero wealth). And
above all, the two upstream SQLite files, which cannot be recreated.

See [Outputs](reference/outputs.md#backup-priority).

### `report` is slow / disk is filling up

`report` iterates every scores file, and expanding-window backtests accumulate
them. Run `prune-scores`; retention keeps the newest per group plus anything a
release from the last 7 days depends on.

### How do I reproduce a forecast the system served last week?

```bash
grounded-weather-forecast predict --now 2026-03-22T17:00:00 --no-history
```

`--now` reissues from an archived snapshot using only data available by that
instant. Add `--no-history` so the experiment does not pollute the
self-verification record.

### How do I force one method for testing?

```bash
grounded-weather-forecast predict --method gbm --no-history
```

Or pin per slice in `[predict.methods]`. Both bypass the promotion gate — a
deliberate override of the system's own evidence, not a tuning knob. Always pair
ad-hoc runs with `--no-history`.

---

## Extending it

### How do I add my own blending method?

Implement the `Blender` protocol and register a **factory** (never an instance —
the engine builds a fresh blender per fold as a leakage defence). Worked example
in [Advanced usage](advanced-usage.md#adding-a-blending-method); the protocol is
in [Methods: notation](methods/notation.md) and the
[API reference](reference/api/index.md).

Then let the leaderboard decide. Nothing ships because it sounds good.

### Can I use this as a library?

Yes — see [Advanced usage](advanced-usage.md#programmatic-use) and the
[API reference](reference/api/index.md). `contracts`, `config`,
`blenders.protocol`, `blenders.registry`, `metrics`, `leads`, and `serve.schema`
are the intended surface.
