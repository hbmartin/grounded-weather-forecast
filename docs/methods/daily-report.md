# Daily reporting read contract

*Implemented in: `daily_read.py`, `daily_read_contract.py`, `serve/provenance.py`, `support_paths.py`;
`health_policy.py::truth_qc_freshness`; the recorder bypass and schema-1 error
boundary are in `cli.py::_read_daily`.*

This is operational evidence-selection policy, not a forecasting or scoring
method. See the [CLI and schema](../reference/cli.md#read-daily-report) and
[bibliography](bibliography.md#3-operational-analogues).

The window is exactly 12 elapsed hours, normalized to UTC before arithmetic,
with hour-aligned timestamps in `[start, end)`. The start and as-of include UTC
offsets; as-of must be at or after start. Range arithmetic and timezone labels
are validated before reads. Delayed execution retains the scheduled window.
Missing/duplicate hours and invalid values remain explicit, without interpolation,
arbitrary duplicate selection, conversion or forecast aggregation. Placeholder
hours without a method never become quality cohorts. A method-bearing slice
without a release remains visible as unknown and prevents available quality status.

## Independent, write-free reads

Every tracked byte read records its file identity, size and modification time.
JSON bytes/objects and Parquet frames are cached within a snapshot; SQLite bodies
are not retained there. Component footprints include reads served from cache.
Each component gets its initial read and at most one retry. A component first
affected during another's retry still gets its own retry. A changed forecast
invalidates backtest and realized-quality dependents. Continued changes beyond
each component's budget make only the affected components unknown,
preserving stable station/provider/other evidence. This does not claim a global
transaction across independently maintained sources.

SQLite never receives a source filename. The reader rejects existing WAL, shared
memory or journal dependencies and WAL-mode headers. It validates a 100-byte
header and its file signature before allocating the full database body. It reads stable rollback-mode
bytes, then queries a deserialized in-memory connection with query-only and
memory-temporary mode. It does not set immutable mode, ignore WAL, create sidecars
or write private disk snapshots. Unsupported deserialization also fails closed.
Connections close after each component query. Station failures use the published JSON fallback; provider health is unknown
when safe database reads are unavailable. Python audit tests and source-byte
checks cover both missing and existing sidecars.

The fallback path resolves once at invocation from `$HOME`, or system home lookup
when unset. An explicit `--published-station` bypasses lookup; an empty or invalid
home is a configuration error. `published_station_path` identifies the attempted
fallback. Reader imports do not resolve home, import writers or create locks.

## Health and freshness

`FRESHNESS_THRESHOLDS` owns separate values: 120 minutes for observations,
180 minutes for forecast/collector completion, and 2,880 minutes each for general
evidence and truth QC. Equal values remain separate policies. Source clocks,
never filesystem mtimes, establish freshness. Missing/future timestamps are
unknown. Overflowing timestamps invalidate their component; invalid CLI ranges
or serialization return schema-1 errors with exit 2.

Provider results join their parent run and use normalized completion times in
`[as_of - 24h, as_of]`. Configured latitude/longitude use the ingestion tolerance
`LOCATION_TOLERANCE = 1e-4`; absent location columns are a limitation. No raw
payload or credential-bearing error is emitted. Latest completed command outcomes
at/before as-of retain per-command freshness even outside 24 hours; an old run can
therefore be stale. Pending classifications preserve the shared exit protocol.
Malformed rows are isolated to relevant commands and add `invalid_run_rows` with
partial/unknown availability. A valid row supersedes a malformed row only when
its ending timestamp is provably newer; unorderable uncertainty leaves that
command unknown. Other commands remain visible. `no_completed_runs_as_of` means
there are no usable completed relevant outcomes at the cutoff, not merely none
in the last 24 hours.

Truth-QC uses the same complete pure assessment as the dashboard: UTC refresh
instants, legacy date intervals through the following midnight, the 48-hour age
limit and latest completed failed refreshes. Date-only results retain evaluation
date and a null exact refresh instant, with conservative age from the interval's
end. Optional null alert flags do not veto other usable diagnostics. Malformed
optional verdict metadata retains valid freshness/alerts with a limitation.
Failed latest refreshes leave retained data visible with partial status and no
current alerts. Schema-2 drift remains historical because it lacks attribution.

## Quality provenance and attributed fallback

The current manifest must match the forecast dataset. Each actual slice names a
forecast release, method, variable, bucket and truth semantics. Pure provenance
decoding shares serving's direct fields and evaluation-context fallbacks. A
retained selection keeps its original evaluation dataset/code/configuration;
these may differ from release creation and current serving datasets. Backtest
rows require that exact evaluation identity and live provenance. Unknown IDs and
legacy missing contexts are never invented or borrowed from another variable.
Absent legacy IDs may recover unambiguous recorded code/semantics; empty,
non-string and `"unknown"` IDs never become wildcard context lookups. Direct
recorded fields remain usable, and context-free legacy approvals remain restricted
to their recorded dataset.

Realized rows require matching slice and configuration and prefer the exact
forecast release. Their dataset identifies the actual served cohort, which can
rotate while a compatible approval remains in use. Their `code_version` identifies
the recorder; it is not compared to the selection's evaluation code. Only when no exact row exists
may an older release supply evidence: both recorded contexts must match live
provenance, provider set, exact feature sequence and truth semantics, as well as
method/code/configuration. Its publication, recording and evidence date must be
within the cutoff. Its immutable publication time must also be strictly earlier
than the forecast release's publication; missing, naive, malformed, equal or later
clocks cannot establish older content. Approval/reapproval ordering is unchanged.
Missing compatibility metadata prevents fallback.

The latest matching evaluation/as-of date wins, followed by recording time.
Realized cohorts add nullable `latest_scored_issue_at`, the maximum issue time
among the forecasts actually scored after truth matching and the existing
minimum-sample qualification. When all tied rows have valid aware clocks no later
than their recording time, prefer the latest scored issue; otherwise skip this
tie-break for the entire group. Then prefer the forecast dataset and coalesce
identical rows. Legacy clocks stay null; there is no historical backfill. If
distinct equally eligible rows remain, the slice is unknown with
`ambiguous_quality_evidence`. One
existing cohort supplies its MAE and sample count; rows are never pooled across
releases/datasets. Exact evidence retains precedence when stale, insufficient or
ambiguous; those conditions do not enable fallback.
The forecast `release_id` stays unchanged; nullable additive fields
`evidence_match`, `evidence_release_id`, `evidence_dataset_fingerprint` and
`latest_scored_issue_at` identify
realized evidence. Fallback adds `older_compatible_release_evidence`. A finite MAE
with positive samples is available evidence, not a prediction-quality verdict.

Realized filtering, time ordering and latest-cohort reduction run in Polars before full-row deduplication and slice decoding. Backtest rows deduplicate earlier to bound duplicate-heavy window work; neither order changes cohort semantics. Compatible fallback validates all candidates in
the newest qualifying time group, then stops before older groups. Each quality ledger is read once and each distinct release is parsed
once per stable attempt. Fallback compatibility is cached per release/slice while time groups are traversed incrementally. Unknown components preserve the schema-1 envelope and
leave unavailable payload fields absent, except that every realized slice explicitly carries nullable `latest_scored_issue_at`, including no-match/error results.

### Relevant run evidence

*Implemented in: `clocks.py`, `run_health.py`, `health_policy.py`, `daily_read.py::read_runs`.*

Eligible records have the requested command and a null, invocation or stage kind.
Shared aware-clock validation requires both a timezone and a non-null UTC offset, rejecting missing zones, overflow and reversed run
intervals; future completions are outside the cutoff. Outcome fields are checked
only after eligibility. A provably older malformed record produces informational
`superseded_invalid_run_rows`; newer, equally timed and unorderable damage keeps
only its command unknown with `invalid_run_rows`. An otherwise valid unfinished
row is skipped. Valid outcomes sort by ending time, starting time, and persisted row position; the last row wins exact-clock ties. The same typed assessment preserves unresolved/superseded damage for truth-QC and pruning. Truth-QC retains available verdict data with an unknown-refresh warning when current attempt evidence cannot be verified. Retention remains 90 days and 50,000 rows; null-start records are
bounded by the existing row cap. A valid truth-QC refresh supersedes damaged attempt
evidence only when its earliest refresh bound is strictly later than the damaged
row's valid ending time. Equal/newer or unorderable damage remains unknown. The
shared freshness calculation assesses runs once, and the dashboard reuses it.
See the operational references in the
[bibliography](bibliography.md#3-operational-analogues).
