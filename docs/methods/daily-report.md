# Daily reporting read contract

Implemented in: `daily_read.py`; the recorder bypass is in `cli.py::_read_daily`.
The shared read-only identity helper is `config.py::config_fingerprint`;
`evaluation.py` retains its existing import API. `health_policy.py` owns shared
freshness constants without importing persistence or filelock (which performs
an import-time temporary-file probe).

This is an operational selection policy, not a new forecasting or scoring
method. See [CLI and schema](../reference/cli.md#read-daily-report) and the
[bibliography](bibliography.md#3-operational-analogues).

The window is exactly 12 elapsed hours, normalized to UTC before arithmetic;
there are 12 expected unique, hour-aligned timestamps in `[start, end)`.
The start and as-of must include offsets and as-of must be at or after start.
Execution after the window ends retains the scheduled window and stale/missing
data limitations instead of shifting the period.
Local labels use the configured IANA timezone. Missing timestamps and duplicate
timestamps produce absent values, never interpolation or arbitrary selection.
All eight canonical hourly variables declare their metric units; PoP is a
dimensionless probability on 0–1. Missing keys, explicit null and non-finite or
non-numeric/out-of-range values have distinct arrays. No conversion or forecast aggregation
is performed.

Freshness limits are 120 minutes for station/forecast observation, 180 minutes
for forecast issuance and collector completion, and 2,880 minutes for evidence,
pipeline history and truth QC. Ages are measured from source timestamps rather
than file mtimes; missing/future timestamps are unknown. File metadata is used
only to detect concurrent changes. An unknown source schema or required-column
change produces an unknown component, preserving the rest of the read.

Provider history joins `provider_results.run_id = forecast_runs.id` and filters
parent `completed_at` inclusively from `as_of - 24h` to `as_of`, using SQLite
date normalization. Failed rows need not have `fetched_at`. The output contains
provider slug, counts, outcome and completion timestamps only. Historical
pipeline alarms are always historical, even when their artifact was recently
rewritten. Run outcomes use latest completion by command within 24 hours;
`failure_kind=no_folds` without an error and an exit in `PENDING_EXIT_CODES`
is pending, not failed.

Quality cohorts are limited to the 12 emitted hourly rows. The current manifest
fingerprint must match the forecast. Each slice must name one of the forecast's
releases; its immutable release must match that ID, dataset and current config
fingerprint, and its selection must match variable, bucket, method and truth
semantics. Backtest evidence additionally matches that selection's evaluation ID
and `source_kind=live`. Realized evidence additionally matches the release ID
and evaluation date. The latest matching evaluation/as-of date wins, then latest
recording within that cohort; future evaluations/recordings are excluded.
The output includes MAE, positive sample count, dates and freshness. A null or
non-finite score or nonpositive count is insufficient; finite evidence with
positive samples is available, without implying statistical confidence or
quality improvement. No score is pooled across products/releases/datasets.
Unknown release IDs after a rebuild cannot inherit older promoted evidence.

Legacy truth-QC schema 3 commonly has only a verdict date, not `generated_at`;
that date is retained but precise freshness/current alerts remain unknown.
Schema-2 drift lacks dataset attribution and is historical only. Neither alarm
set is promoted to a current pipeline failure. Sources are independently
maintained, so two stable reads cannot promise a global atomic snapshot; a
changed file triggers one complete retry, followed by unknown components if
changes continue. Live SQLite uses ordinary read-only WAL-aware connections,
never immutable mode, with a one-second busy timeout and a refusal to create a
missing WAL shared-memory sidecar.
