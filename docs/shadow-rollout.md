# Review before production rollout or shadow activation

The implementation is inactive by default. Merging a PR does not install a
wrapper, change a pinned runtime, start a job, or enroll an evaluation. No
weather provider, billing, account or security setting is changed by this feature.

Review the maintenance freshness fix independently. Its success stamp measures
the start of a successful run; each host-local daily slot (default 02:15)
requires a successful start since that slot. Custom schedules use `--scheduled-at`. Preserve the previous stamp on failure, and retain restore semantics.
Installing only that wrapper need not change the source implementation digest.
Both wrappers retain the older outcome-normalization API when the selected
interpreter lacks `CommandOutcome.from_subprocess`; ordinary operation does not
require installing the new source just to obtain that helper.
This compatibility guarantee applies only to ordinary publishing and maintenance.
Shadow needs compatible reviewed source/wrappers and a matching enrollment; the
#102 runner does not implement named preflight. Its rejection preserves ordinary
publication and reports shadow failure instead of inferring a capability defect
from exit 2 alone.

For source changes, choose and record an immutable reviewed release commit.
Record current runtime import path, implementation/config/dataset identities,
active approvals, wrapper hashes and last public commit before changing anything.
Keep the prior source checkout and installed wrappers for rollback. A source
change invalidates implementation-specific evidence: regenerate compatible
backtests and approval through the normal workflow. Do not force a method,
disable a gate, re-label old evidence or copy an old release into a new identity.

Use the operator's actual configured paths in the reviewed commands below.
These are preparation instructions, not an instruction to execute them now:

```sh
# Read-only preflight in the selected release checkout.
git rev-parse HEAD
uv run --no-sync --offline python -c 'from grounded_weather_forecast.evaluation import code_identity; print(code_identity())'

# After explicit rollout approval, using the existing configured runtime and
# maintenance entrypoint. A normal maintain run may collect ensemble/neighbor
# data; review those network actions and the job lock before starting catch-up.
uv run --no-sync python docs/launchd/daily-maintenance.py
```

Do not install dependencies on a production host without the operator's
permission. Deploy source and wrapper paths deliberately; an environment that
still pins an older `PYTHONPATH` will continue importing that older code.
Install publisher 2.3+ with `run_with_receipt()` before enabling v3 shadow flags
([publisher follow-up #48](https://github.com/hbmartin/open-sun/pull/48)).
Prepare reviewed `shadow_supervisor.py` beside the installed wrappers and
`launchd_paths.py`; it uses only the existing Python standard library. Its
private liveness pipe bounds worker lifetime after wrapper death, including
SIGKILL, assuming the supervisor survives. Keep this helper with the matching
wrapper set for rollback. Missing helpers preserve ordinary publication but
make requested shadow work fail. No schedule or activation changes follow from
copying or merging these sources.
Preserve old enrollments; the new runner requires a fresh v3 enrollment.
Verify one complete maintenance cycle, truthful gate explanations, a ready
forecast and successful public publication before enrolling the shadow.
Check station freshness and hourly-mean coverage. An available instantaneous
sample does not substitute for a coverage-qualified hourly mean. Resolve any
receiver reachability problem without inventing truth or guessing a new host.

The supplied proposal's start has elapsed and it cannot be enrolled. It begins November 2, 2026 at 08:00 UTC and ends December 14
at 08:00 UTC. Preserve these dates unless the operator explicitly approves a new
predeclared window before outcomes. An elapsed start is rejected, not silently
rolled forward. Amendments require a new enrollment directory and review trail.

After approval of both the implementation and the exact window, create a private
enrollment using a reviewed copy of `examples/temperature-shadow/protocol.json`:

```sh
uv run --no-sync python scripts/temperature_shadow.py freeze \
  --protocol REVIEWED_PROTOCOL --out PRIVATE_ENROLLMENT

# Add these paired arguments to the reviewed hourly wrapper invocation only
# after activation approval; do not create a second production publisher.
uv run --no-sync python docs/launchd/hourly-predict-publish.py \
  --shadow-root PRIVATE_SHADOW --shadow-protocol PRIVATE_ENROLLMENT/protocol.json
```

The wrapper exports the existing serving snapshot once. After the normal
publisher succeeds, it seals the exact commit and forecast bytes returned by
that successful explicit-commit push. It never reconstructs a receipt from a
later branch tip. Unchanged publication requires verified original evidence. It does not call a weather provider to
reconstruct inputs or refit a production selection. A held forecast retains
its original capsule and issuance. If that original capsule is unavailable,
the issue cannot enter the prospective comparison. No duplicate hourly task is
needed; changes to service arguments or scheduling require separate approval.

Verify one immutable record before allowing continued collection: source and
public payload hashes, full public commit, matching issue/grid/release/data,
production code/config identity, raw provider completion clocks, and honest
station receipt assumptions. Genuine archive run-completion clocks may be after
the rounded issuance instant; retain them, and do not call a read/export time a
receipt or commit timestamp. No independent station commit ledger is invented.

Evaluate once after the last 96-hour outcomes and two-hour truth lag:

```sh
uv run --no-sync python scripts/temperature_shadow.py freeze-truth \
  --truth REVIEWED_QC_HOURLY_TRUTH --protocol PRIVATE_ENROLLMENT/protocol.json \
  --out SEALED_TRUTH
uv run --no-sync python scripts/temperature_shadow.py evaluate \
  --input PRIVATE_SHADOW/records --capsules PRIVATE_SHADOW/capsules \
  --truth-lock SEALED_TRUTH --out NEW_EVALUATION \
  --protocol PRIVATE_ENROLLMENT/protocol.json --mode prospective
```

The protocol requires sufficient independent-day coverage and paired 3-/7-day
uncertainty at each actual 0–96h horizon. Inconclusive evidence retains production.
Replay never counts as prospective evidence. Review candidate forecasts against
actual published production, simple decayed correction, available provider,
persistence and climatology. Wind, gust, rain and daily extremes are outside
this enrollment. There is no automatic promotion.

Rollback removes the optional shadow arguments, restores the recorded source
and wrapper paths and verifies publication against compatible retained evidence.
Preserve immutable shadow records and their provenance; do not rewrite or pool
them into a new enrollment. The rollout decision must specify maintenance
catch-up timing, public hold behavior, the snapshot/receipt hook, and exact
window. A successful merge alone authorizes none of those production actions.

Before capture, the wrapper runs the selected runtime's named `preflight`
command with a 30-second timeout and logs its duration. Outside-window skips
remain successful even when an older checkout lacks shadow capability. A
preflight failure preserves ordinary prediction/publication and reports the
shadow failure. Install the reviewed source and compatible wrappers together.
Timeouts/cancellation terminate the POSIX uv/Python process group, including for
300-second finalization, so killing only the launcher cannot leave a runner alive.

V3 freezes Python, numerical-library versions, backend/platform, effective
thread controls (two threads) and dependency-lock digest. Preserve that runtime
for evaluation. Receipts are durably spooled before finalization, then complete
validated evidence is sealed within issuance +30 minutes and before outcomes.
Only that seal permits delayed reconstruction; a surviving transport file alone
does not. Held forecast pushes keep original evidence and completed record hashes.
Before rollout, verify the v3 seal, record manifest, runtime and declared input
hashes; incidental regular files do not change them. Review all six methods'
availability because descriptive persistence/climatology still restrict matching.
The `.sealing` and `.NAME.publishing` guards release ownership on process exit;
do not delete their sidecars to recover a stopped process. Verify an existing
complete record before reuse, retain conflicting/corrupt outputs, and investigate
any reported post-publication synchronization failure. Enrollments publish their
protocol and implementation lock as one complete directory. Validation and
reconstruction reuse each capsule's captured byte snapshot.
No old prospective evidence is migrated or relabeled. Legacy replay finalization
requires its original compatible runner; raw-truth replay evaluation remains
available without prospective-only flags.
