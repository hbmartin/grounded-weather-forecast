# ADR 0001: SQLite files are the only upstream contract

## Status

Accepted.

## Context

grounded-weather-forecast depends on two upstream projects:

- [ambientweather2sqlite](https://github.com/hbmartin/ambientweather2sqlite), which
  logs the station's observations;
- [omni-weather-forecast-apis](https://github.com/hbmartin/omni-weather-forecast-apis),
  which polls a dozen forecast APIs and archives what they said.

Both are Python packages by the same author. The obvious integration is to import
them: share Pydantic models, call their readers, perhaps have `predict` fetch
fresh provider data on demand rather than waiting for a cron.

That obvious design has three problems specific to this system.

**Backtests must run against old data.** The entire premise is evaluating what
providers said months ago. A shared model class that gains a required field
breaks every archive written before the change. The archive is append-only and
spans years; the code that reads it changes weekly.

**A fetch-fresh path would break the evaluation.** If `predict` could fetch on
demand, the thing being served would no longer be the thing being backtested —
the backtest reconstructs snapshots from the archive, and a live fetch would have
no archived counterpart. The claim "the thing we backtest is literally the thing
we serve" would become false.

**Upstream release churn is not this project's problem.** The collector's job is
to write rows. What it uses internally to do that is irrelevant here.

## Decision

**The two SQLite files are the entire interface.** Specifically:

- Neither upstream Python package is imported. There is no dependency on them.
- There is no fetch-fresh path in `predict`. Fresh data arrives only via the
  upstream cron.
- Both databases are opened **read-only** (`mode=ro`, optionally
  `?immutable=1` for static snapshots). This project never writes upstream.
- Schema drift is tolerated defensively: columns are read by name, absences are
  handled, and derived columns are recomputed rather than trusted.

The schemas we depend on are documented here rather than inferred:
[aw2sqlite](../upstream/aw2sqlite-database.md) and
[omni-weather-forecast-apis](../upstream/omni-weather-forecast-apis-database.md).

## Consequences

**Good.**

- The harness runs against any archive vintage. A years-old file works.
- Upstream can be rewritten in another language without affecting this project.
- The integration surface is small enough to document completely, and it is.
- Read-only access means a bug here can never corrupt the irreplaceable archive.

**Costs, accepted knowingly.**

- No shared models: canonical types are redefined in `contracts.py`.
- Schema changes are discovered at read time, not import time.
- Defensive reading is more code than trusting a typed model would be.

**A consequence that turned out to be a benefit.** Because nothing is trusted,
`fetched_at_unix`, `horizon_hours`, and `run_cycle` — all NULL throughout the
reference archive — were never depended on. Lead is always recomputed from
timestamps. A design that had imported a typed model would likely have trusted
`horizon_hours` and silently produced nothing.

## Related

- [ADR 0005](0005-promoted-model-releases-are-the-serving-boundary.md) — the
  downstream boundary, where evidence rather than data is the contract.
- [Architecture](../architecture.md) — the layering this enables.
