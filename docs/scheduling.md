# Scheduling: the four crons that feed the system

The archive is the binding constraint on everything this project can learn
([Limitations §1](limitations.md)) — and three of the four inputs cannot be
backfilled after the fact. These launchd templates (in
[`docs/launchd/`](https://github.com/hbmartin/grounded-weather-forecast/tree/main/docs/launchd))
keep the pipeline fed unattended on macOS. On Linux, translate each to a
systemd timer or crontab line; the cadence rationale is identical.

## The jobs and how often to run them

| Job | Cadence | Why this cadence |
|---|---|---|
| `poll` — the upstream `omni-weather` collector | **hourly** | Every missed hour of provider vintages is unrecoverable. Hourly catches every provider's update cycle while staying inside free-tier quotas (the upstream quota tracker enforces per-provider caps). If quotas pinch, drop to per-run-cycle (6-hourly) for the slow-refresh providers via the upstream config, not by slowing this job. |
| `ingest-ensembles` | **every 6 h** (`StartInterval` 21600) | Ensemble models run on 00/06/12/18 UTC cycles and Open-Meteo retains only the latest run's members — a missed cycle's spread is gone forever. A fixed 6-hour interval catches every cycle regardless of publication lag; the as-of join tolerates any offset. |
| `publish` (the `predict` launchd job) | **every 10 min** (`StartInterval` 600) | Matches the 10-minute snapshot grid, so each accepted serve sees at most one new snapshot. Ready forecasts replace the published document and enter self-verification history; a degraded candidate holds the last ready document without polluting history. Widen to 15–30 min if the machine is battery-constrained; the cost is coarser verification history, not correctness. |
| `maintain` — `build-dataset` → `backtest --source live` → `report` → `truth-qc` | **daily, 02:15 local** | The retrain loop: refreshed truth and ensemble features, refreshed evidence, re-promoted winners in the release ledger, and the neighbor/shield sensor checks. Schedule it after the latest successful ensemble ingest: ensemble rows become model features only when `build-dataset` rematerializes the matrix. Daily is the right floor — truth accrues by the hour but promotion decisions move on days. As the archive and method count grow, backtest runtime grows too; if the nightly run gets slow, pass a curated `--methods` subset nightly and run the full sweep weekly. Follow with `prune-scores` periodically (or append it to the chain): superseded scores files accumulate at nightly cadence. |

The `backfill` commands are deliberately *not* scheduled: they are one-off
cold-start tools, and re-running them is idempotent but pointless on a cron.
If an ensemble ingest runs after maintenance, rebuild the dataset again before
backtesting or serving methods that consume ensemble features.

## Installing

1. Install the shared launcher on the internal disk, then copy the four plist
   templates into `~/Library/LaunchAgents/`. Fill their `__PLACEHOLDERS__`
   (including the absolute `__LAUNCHER__` and `__UV__` paths for `maintain`
   and `predict`, repository paths, log directory, coordinates, output path,
   forecast database, and Synoptic token). Keep each label matching its filename.

   ```bash
   mkdir -p ~/Library/LaunchAgents ~/.local/state/grounded-weather-forecast \
     "$HOME/Library/Application Support/grounded-weather-forecast"
   cp docs/launchd/launch-grounded \
     "$HOME/Library/Application Support/grounded-weather-forecast/launch-grounded"
   chmod 755 "$HOME/Library/Application Support/grounded-weather-forecast/launch-grounded"
   for job in poll ingest-ensembles predict maintain; do
     cp "docs/launchd/com.grounded-weather-forecast.${job}.plist" ~/Library/LaunchAgents/
     "$EDITOR" "$HOME/Library/LaunchAgents/com.grounded-weather-forecast.${job}.plist"
   done
   ```

2. Load the jobs (modern launchctl syntax). `maintain` and `predict` run on
   their schedules; loading them does not start an extra run.

   ```bash
   for job in poll ingest-ensembles predict maintain; do
     launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.grounded-weather-forecast.${job}.plist"
   done
   ```

3. Verify and watch:

   ```bash
   for job in poll ingest-ensembles predict maintain; do
     launchctl print "gui/$(id -u)/com.grounded-weather-forecast.${job}" | head
   done
   tail -f __LOG_DIR__/predict.log
   ```

   To unload one job, run
   `launchctl bootout "gui/$(id -u)/com.grounded-weather-forecast.predict"`;
   substitute any of the other three labels as needed.

## Notes

- The `maintain` and `predict` templates invoke a launcher copied to the
  internal disk. It checks that an external project volume is mounted and
  delegates project access to `uv --directory __REPO__ run --locked`, which
  establishes the working directory and rejects an outdated lockfile. Use
  absolute paths for the launcher and `uv`; no login-shell setup is needed.
  A mounted volume with a subsequent `Operation not permitted` error from `uv`
  points to macOS file-access permissions, not a missing volume.
- Diagnose startup from the log markers: no `START` means inspect the launchd
  timer and service state; `START` followed by a mount error means the volume
  was absent; `MOUNT_OK` followed by a `uv` error means the volume was present
  but project access or environment setup failed. Application output follows
  `INVOKE`; use `launchctl print` for the final exit status.
- A degraded `publish` candidate and `maintain` finding no folds are **normal**
  early states. The candidate names its cause in `status_reason`; `publish`
  preserves an existing parseable ready document byte-for-byte, or publishes
  and archives the degraded candidate on a cold start so an output still exists.
- launchd `StartCalendarInterval` fires in **local time**; the 6-hourly
  ensemble job uses `StartInterval` (elapsed seconds) precisely so daylight
  saving cannot skip a model cycle.
- `StartCalendarInterval` jobs missed while the machine is **off** are never
  replayed (asleep is fine — launchd coalesces on wake). A machine that
  shuts down overnight therefore silently skips that day's `maintain` run.
  Manually `kickstart` it after such a shutdown;
  the template does not run speculative maintenance at login.
- The station logger upstream (`aw2sqlite serve`) is a hard dependency of
  truth: run it as its own `KeepAlive` LaunchAgent, never from a Terminal
  session — a closed session or reboot otherwise stops truth silently while
  forecasts keep accruing (a week of unusable, label-less snapshots).
- The scheduled `publish` run appends only accepted ready forecasts (and a
  degraded cold-start forecast) to self-verification history. Rejected degraded
  candidates are neither published over a ready document nor appended. Manual
  `predict --no-history` remains the opt-out for experiments.
- `grounded-weather-forecast prune-scores` (preview with `--dry-run`) deletes
  superseded backtest scores files: the newest three per
  (product, source, window), anything referenced by the active release, and
  anything a release promoted in the last 7 days are kept. Files the evaluations
  catalog has never seen are never deleted.
- `maintain` waits for `<dataset dir>/pipeline.lock` and holds it once across
  build → live backtest → report → truth-QC. Interactive mutators wait 60 s and
  exit `75` on contention. This prevents another command from slipping between
  scheduled steps. `predict` and `publish` use the short `dataset.lock` only
  while selecting and generating, so they cannot read a half-published dataset;
  `ingest-ensembles` retains its independent store lock.
- Every degraded candidate is eligible for automatic recovery, not only a
  code-identity mismatch. `publish` derives a signature from the reason plus
  dataset/config/code identities and starts detached `recover` at most once per
  unchanged signature every six hours; a changed signature is immediately
  eligible. `recover` deduplicates concurrent requests, waits behind maintenance,
  rechecks readiness, then runs build → live backtest → report only if still
  degraded. State and output live in `artifacts/auto-restore.json` and
  `artifacts/auto-restore.log`.
- Keep the Synoptic token out of the plist if you prefer: set
  `synoptic_token = "$SYNOPTIC_TOKEN"` in `config.toml` and provide the
  variable via `launchctl setenv SYNOPTIC_TOKEN ...` instead of the
  `EnvironmentVariables` block.
