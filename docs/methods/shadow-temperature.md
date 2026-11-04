# Inactive temperature shadow candidate

Implemented in: `scripts/shadow_candidate_v1.py` and
`scripts/temperature_shadow.py`, `scripts/shadow_evidence.py`,
`scripts/shadow_environment.py`, `src/grounded_weather_forecast/shadow_artifacts.py` and the standalone
`docs/launchd/shadow_supervisor.py`. These standalone scripts register no new
production method. They preserve the preselected October 26 experiment's
`simplex_decay21` candidate and `decay21` reference. The definitions below are
self-contained local experimental choices, not a reproduction of an operational
NOAA implementation or a production skill claim.

For each source, estimate forecast-minus-truth bias using weights
`2 ** (-age_days / 21)`. Train only on labels whose valid hour plus two hours
is at or before issuance, with archive issue time at or before issuance.
Retain the latest issue per valid hour and lead group. Groups are [0,6),
[6,24), [24,72), [72,168), [168,∞) hours. Divide each global residual's
weight by that valid hour's lead-group multiplicity. Shrink each group bias
toward the global bias by `n / (n + 7)`, with `n` the count of UTC dates with
finite source values and positive weight. This is a count, not a decayed
effective sample size. Subtract the correction before combining providers.

Fit one nonnegative sum-to-one vector on corrected training rows. Minimize
weighted squared error divided by training target variance (standard-deviation
floor 0.1), plus `2 * sum((w - equal) ** 2)`. SLSQP uses bounds [0.0001,1],
100 iterations and tolerance 1e-8. Use equal weights with fewer than 20 usable
rows or a failed solve. Final weights are 50% fitted and 50% equal. Renormalize
over finite providers at both fit and prediction. The reference uses the same
corrections with equal weights throughout. Neither method silently alters truth.

Evaluation covers only the served 0–96 hours. It averages repeated issuance
losses per valid hour, then hours per UTC date. Circular calendar blocks of
3 and 7 days preserve missing dates. The proposal uses 10,000 resamples, seed
20261031, and two-sided Bonferroni limits for twelve predeclared comparisons
(four horizons times production, decayed equal weight and best provider).
Quality requires 28 matched dates, 300 matched valid hours, four weeks with
data, 80% common-row coverage and 80% of expected hourly issuances. Candidate
review requires at least 0.1°C and 5% daily-MAE improvement over production,
adjusted interval upper bound below zero against production and at most
0.1°C against the two main references under both block lengths. All six
methods must be finite on common cases. Raw pairwise availability and
attrition must also be inspected. Decisions occur once after six weeks plus
the last 96-hour outcomes and two-hour truth lag. No automatic promotion.

## V3 retained-evidence policy

Complete input/publication evidence is validated and synchronized before
`evidence.lock.json` is published. `evidence_sealed_at` must lie within issuance
+30 minutes and before the first outcome; only a complete timely seal authorizes
late deterministic reconstruction. Actual `recorded_at` is retained separately.
Publication receipts are prospective-only: replay finalization refuses them
before guard creation or writes, and the sealing helper independently enforces
the mode. Output-root/issuance checks and acquisition of the actual private
publication staging directory precede sealing and fitting. Unwritable output
stores fail before a new capsule seal; successful records reuse that transaction.
An unsuccessful attempt can leave private empty output parents and harmless
native guard sidecars, but no accepted record. New output
parents are private; macOS `/tmp` and `/var` prefixes are canonicalized without
relaxing selected-alias or prospective capsule-permission checks.
A completed `record.lock.json` binds record bytes to the evidence seal. Held
pushes reuse the original issuance/receipt, and evaluation reconstructs each
prediction with `rtol=1e-10`, `atol=1e-10`. Declared evidence files are hashed;
incidental regular files are excluded while aliases remain invalid.

The `.sealing` and `.NAME.publishing` files carry OS-managed locks, not persistent
lease ownership. Process exit releases ownership; a live concurrent owner causes
an operational refusal. Capsule state is read after taking the sealing lock.
All consumed matrix schemas are checked before publication evidence is written;
fitting happens after the complete timely seal. Enrollment publishes its computed
protocol/lock pair atomically. A per-invocation capsule snapshot reads and hashes
each declared existing file once, reuses parsed inputs without changing source
values, and is released before evaluating the next capsule.

Native guards require FileLock 3.31.0+, mode `0600`, nonblocking acquisition and
`fallback_to_soft=False`. Unsupported native locking returns operational exit 1;
no marker-existence fallback is permitted. Missing capsule paths are checked
before acquisition without creating directories. Prospective capsule roots and
issuance directories must exclude POSIX group/world access; existing evidence is
not chmodded. Primary failures and cancellation survive secondary unlock errors,
which remain visible diagnostics; successful publication followed by failed
unlock reports the preserved output with exit 1.

`PreparedInputs` reuses filtered training, the supervised slice and forecast
matrix once per computed record, including attribution. Candidate-internal
transforms and all numerical operations/constants remain unchanged. Explicit
schema checks precede transformation; unexpected transformation/contract errors
are failures rather than evidence refusals. Snapshot identity uses object
identity, its mode is explicit, and publication bytes carry their validated digest.
Training and issue matrices must each contain one valid matching live/synthetic
provenance; this preserves the source-kind boundary before sealing and fitting.
`VerifiedEvidence` retains the snapshot, source/publication projection, seal,
validated sealing timestamp, checked issue and optional prepared inputs for the
current owner. Record helpers enforce matching snapshot object identity as an
internal contract. First-seal decoding and correspondence validation each run once;
recovery/evaluation verify independent snapshots, and held pushes verify both
receipts without refitting completed records. Capture/readers share issue,
training, truth, completion, publication and evidence-lock filenames through
`shadow_artifacts.py`. Operator proposal/replay protocol inputs may follow a
symlink; prospective enrollment files remain nonaliased. Missing required files
produce named exit-2 refusals; permission/I/O failures remain exit 1.

The named `preflight --protocol FILE` checks the window first. Inside the window
it checks capture capability and complete v3 enrollment in the selected runtime;
outside it returns `outside_window` without those checks. The wrapper allows 30 seconds for preflight
and 300 seconds for finalization; capability/preflight failures preserve ordinary
publication and report shadow failure. There is no persistent validation cache.
The installed standard-library supervisor monitors a wrapper-only pipe and owns
the POSIX worker group. Internal protocol **2** acknowledges bootstrap ownership
before admitting the command. Exit observation uses `waitid(WNOWAIT)` to reserve
the group identity until cleanup and the terminal acknowledgment (or wrapper EOF).
The wrapper continuously drains the separate status pipe, bounded to **64 KiB**;
oversized diagnostics retain error/path context with explicit truncation markers.
Worker completion takes precedence over simultaneous cancellation. Repeated
SIGTERM is ignored while the shadow CLI unwinds; repeated wrapper cancellation
is deferred until bounded cleanup completes. Emergency rescue stops and verifies
the still-owned supervisor before signaling an unreleased worker identity.
Wrapper death, including SIGKILL, closes the liveness pipe. The **30/300-second**
execution deadlines retain **2 seconds** of TERM grace and **3 seconds** to force
termination/reap, with **1 second** of wrapper wait margin. Exceptional rescue
has a separate **3-second** bound. A surviving supervisor is required: unexpected
supervisor death reports inability to certify cleanup rather than signaling a
historical PID. Forced termination can leave recoverable staging files.
The monitoring interval remains **0.05 seconds**. Non-POSIX behavior is unchanged.

The [CLI reference](../reference/cli.md#standalone-temperature-shadow) owns the
versioned preflight envelope and exit-code contract. Only tagged refusals authorize
the wrapper's evidence/capability classification; launcher and usage exits retain
their own diagnostics. Older runtimes remain compatible for ordinary
publishing/maintenance only.

Freeze Python patch, numerical-library/backend/platform identity, two-thread
controls and dependency-lock digest. Truth matures at window end +98 hours and
normalizes to UTC microseconds after exact-hour validation. Evaluation hashes
and scores the same truth/protocol/lock/record byte snapshots. These are private
operator consistency checks, not authentication against rewriting every artifact.
The six-method cohort and all candidate/uncertainty constants above are unchanged;
descriptive persistence and climatology still restrict case availability.
V1/v2 prospective artifacts are preserved without migration; v3 needs fresh
future enrollment. The original November 2–December 14 proposal is unchanged
and its elapsed start blocks enrollment pending explicit future-window approval.

Platform identity covers OS family and architecture, not kernel release.
Unrelated kernel updates do not invalidate the frozen numerical environment.
