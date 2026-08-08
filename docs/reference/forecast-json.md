# Forecast JSON (schema v5)

The document `predict` emits. This is the system's integration surface — if you
are consuming forecasts from another program, this page is the contract.

*Implemented in: `serve/schema.py`, `SCHEMA_VERSION = 5`*

```bash
grounded-weather-forecast predict --out forecast.json
```

Loading it back into typed objects:

```python
from grounded_weather_forecast.serve.schema import Forecast

forecast = Forecast.from_json(open("forecast.json").read())
print(forecast.status, len(forecast.hourly))
```

---

## Design principle: every number carries its provenance

Most forecast APIs give you a number. This document gives you a number **plus how
it was produced, which evidence justified it, what it is measuring, and how
confident the system is that it should be serving that method at all**.

That is why each point carries parallel dictionaries keyed by variable name
rather than a flat list of values. A consumer that only wants numbers reads
`values` and ignores the rest; a consumer that needs to defend a decision has
everything it needs.

---

## Top level

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | `5`. Check it. |
| `issued_at` | ISO 8601 | the information boundary — nothing after this instant informed the document |
| `latitude`, `longitude` | float | station location |
| `timezone` | IANA name | defines `date_local` in the daily block |
| `dataset_fingerprint` | str | identity of the matrices this forecast was fitted on |
| `sources` | list of str | the provider/model slugs available at this snapshot |
| `observation_at` | ISO 8601 or `null` | timestamp of the station reading used for anchoring; `null` means no usable observation |
| `status` | `"ready"` \| `"degraded"` | see below |
| `status_reason` | str or `null` | why, when degraded |
| `release_ids` | list of str | every promoted release that contributed |
| `minutely` | list | next 60 minutes |
| `hourly` | list | next 48 hours |
| `daily` | list | next 10 local days |

### `status`

`"ready"` means every served method was chosen by a promoted release matching the
current dataset, config, and code fingerprints.

`"degraded"` means it was not, and the system fell back to `equal_weight`. The
forecast is still usable — an ungrounded equal-weight blend is a reasonable
forecast — but it is **not** the evidence-backed one. Typical reasons:

```json
"status": "degraded",
"status_reason": "implementation changed since the last backtest; re-run `backtest --source live` then `report`"
```

Degraded status is also printed to **stderr**, so `--out -` still yields clean
JSON on stdout. See the [FAQ](../faq.md#why-does-my-forecast-say-degraded).

!!! note "Degraded is a feature"
    The alternative — serving a promoted method whose justifying evidence no
    longer applies — is how a system quietly starts lying. See
    [ADR 0005](../adr/0005-promoted-model-releases-are-the-serving-boundary.md).

---

## `hourly[]`

| Field | Type | Meaning |
|---|---|---|
| `valid_time` | ISO 8601 | the hour this point is about |
| `lead_hours` | float | `valid_time − issued_at`, in hours |
| `lead_bucket` | str or `null` | the bucket that governed method selection |
| `values` | `{variable: float\|null}` | the forecast |
| `methods` | `{variable: method_id}` | which blender produced each value |
| `quantiles` | `{variable: {level: value}}` | predictive quantiles, when available |
| `quantiles_source` | `{variable: str}` | where the quantiles came from |
| `selection_reasons` | `{variable: str}` | why that method was chosen |
| `release_ids` | `{variable: release_id}` | which promotion justified it |
| `truth_semantics` | `{variable: "inst"\|"mean"}` | what the number is *measuring* |

A real point:

```json
{
  "valid_time": "2026-08-08T18:00:00+00:00",
  "lead_hours": 3.25,
  "lead_bucket": "3-6h",
  "values": {
    "temp_c": 29.807,
    "humidity_pct": 26.104,
    "dew_point_c": 8.847,
    "wind_speed_ms": 3.268,
    "wind_gust_ms": 5.844,
    "pressure_sea_hpa": 1013.585,
    "precip_mm": 0.0,
    "pop": 0.03
  },
  "methods":           { "temp_c": "equal_weight", "...": "..." },
  "quantiles":         {},
  "quantiles_source":  {},
  "selection_reasons": { "temp_c": "no backtest evidence for this slice", "...": "..." },
  "release_ids":       {},
  "truth_semantics":   { "temp_c": "mean", "wind_gust_ms": "inst", "...": "..." }
}
```

### Reading `truth_semantics`

This is the field most consumers overlook and the one most likely to cause a
subtle integration bug.

- `"inst"` — the value is the **instantaneous** level at `valid_time`.
- `"mean"` — the value is the **interval mean** over `[valid_time, +1h)`.

In the example above `temp_c` is an interval mean while `wind_gust_ms` is
instantaneous, because a gust *is* an extreme and a mean gust is not a
meaningful object. If you compare these numbers against your own observations,
compare like with like — mismatching the convention manufactures roughly 1 °C of
apparent error on a clear day. See
[Methods: truth and QC §2](../methods/truth-qc.md#2-the-aggregation-ladder).

### Reading `quantiles` and `quantiles_source`

`quantiles` is empty when the serving method is a point forecaster and no
dressing pool was available. When present, keys are string levels:

```json
"quantiles": {
  "temp_c": { "0.05": 26.1, "0.25": 28.4, "0.75": 31.2, "0.95": 33.8 }
}
```

`quantiles_source` records provenance. An **absent key means the quantiles came
from the method itself** (EMOS, IDR, conformal, analog); a present value names
the residual-dressing or recalibration path that produced them. This matters
because a method-native distribution and a dressed one carry different
guarantees — see [Methods: calibration](../methods/calibration.md).

Quantiles are always monotone in level (rearranged) and coherent across related
variables (`temp_min ≤ temp_max`, `dew_point ≤ temp`) at *every* level, not just
at the median.

### Reading `selection_reasons`

Human-readable strings explaining the choice. `"no backtest evidence for this
slice"` is the cold-start case; others name the gate that promoted or blocked a
method. Paired with `release_ids`, a scored row can be attributed to the exact
promotion that served it.

---

## `daily[]`

Same shape as `hourly[]`, keyed by local day:

| Field | Type | Meaning |
|---|---|---|
| `date_local` | `YYYY-MM-DD` | the **local** calendar day, per `timezone` |
| `lead_days` | int | days ahead |
| `values`, `methods`, `quantiles`, `quantiles_source`, `selection_reasons`, `release_ids`, `truth_semantics` | as above | |

Typical variables: `temp_max_c`, `temp_min_c`, `pop`, `precip_sum_mm`.

Daily extremes are their own supervised targets, not aggregates of the hourly
path — so a daily max need not equal the max of the hourly block, and that is
intentional ([ADR 0002](../adr/0002-anchored-nowcast-instead-of-temporal-disaggregation.md)).
The precipitation *sum* is linear and is coherent with the hourly path by
construction.

---

## `minutely[]`

A flatter shape, because the minutely product serves a fixed variable set.

| Field | Type | Meaning |
|---|---|---|
| `valid_time` | ISO 8601 | the minute |
| `minutes_ahead` | int | 0–60 |
| `temp_c`, `humidity_pct`, `dew_point_c`, `wind_speed_ms`, `precip_intensity_mmh`, `pop` | float or `null` | the forecast |
| `methods` | `{variable: method_id}` | the promoted path construction, per variable |
| `quantiles` | `{variable: {level: value}}` | when available |

`precip_intensity_mmh` is an **intensity** (mm/hour), not an accumulation — the
only place the units differ from the hourly block.

`methods` here names a minutely path construction (`minutely_interp`,
`minutely_persistence`, `minutely_anchor_tau_1h`, …) rather than a blender. See
[Methods: combination §8](../methods/combination.md#8-minutely-path-constructions).

---

## Compatibility

`Forecast.from_json` accepts documents that omit optional blocks (`minutely`,
`hourly`, `daily`, `timezone`, `status`, `status_reason`, `release_ids`), which is
what lets schema-1 documents still load. Required in every version:
`schema_version`, `issued_at`, `latitude`, `longitude`, `dataset_fingerprint`,
`sources`.

Serialization uses `allow_nan=False`: a missing value is JSON `null`, never
`NaN`. The document is therefore valid JSON for every strict parser.

**Check `schema_version` before parsing.** A version bump means fields changed.
