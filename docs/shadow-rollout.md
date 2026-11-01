# Review before production rollout or shadow activation

The implementation is inactive by default. Merging a PR does not install a
wrapper, change a pinned runtime, start a job, or enroll an evaluation. No
weather provider, billing, account or security setting is changed by this feature.

Review the maintenance freshness fix independently. Its success stamp measures
the start of a successful run; the next refresh is eligible after 20 hours
from that start. Preserve the previous stamp on failure, and retain restore semantics.
Installing only that wrapper need not change the source implementation digest.

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
Verify one complete maintenance cycle, truthful gate explanations, a ready
forecast and successful public publication before enrolling the shadow.
Check station freshness and hourly-mean coverage. An available instantaneous
sample does not substitute for a coverage-qualified hourly mean. Resolve any
receiver reachability problem without inventing truth or guessing a new host.

The supplied proposal begins November 2, 2026 at 08:00 UTC and ends December 14
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
publisher succeeds, it fetches the exact public Git commit into a temporary
object store and seals the receipt. It does not call a weather provider to
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
uv run --no-sync python scripts/temperature_shadow.py evaluate \
  --input PRIVATE_SHADOW/records --truth FROZEN_QC_HOURLY_TRUTH \
  --out NEW_EVALUATION --protocol PRIVATE_ENROLLMENT/protocol.json --mode prospective
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
