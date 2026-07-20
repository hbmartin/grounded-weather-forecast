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
| `predict` | **every 10 min** (`StartInterval` 600) | Matches the 10-minute snapshot grid, so each serve sees at most one new snapshot. Every run appends to the self-verification history — the data that later powers the live-MAE demotion gate — and the online expert state advances by consuming only the rows resolved since the last serve, re-validating the processed prefix each time (so a serve costs O(history) in cursor checks, but only O(new rows) in weight updates). Widen to 15–30 min if the machine is battery-constrained; the cost is coarser verification history, not correctness. |
| `maintain` — `build-dataset` → `backtest --source live` → `report` → `truth-qc` | **daily, 02:15 local** | The retrain loop: refreshed truth and ensemble features, refreshed evidence, re-promoted winners in the release ledger, and the neighbor/shield sensor checks. Schedule it after the latest successful ensemble ingest: ensemble rows become model features only when `build-dataset` rematerializes the matrix. Daily is the right floor — truth accrues by the hour but promotion decisions move on days. As the archive and method count grow, backtest runtime grows too; if the nightly run gets slow, pass a curated `--methods` subset nightly and run the full sweep weekly. |

The `backfill` commands are deliberately *not* scheduled: they are one-off
cold-start tools, and re-running them is idempotent but pointless on a cron.
If an ensemble ingest runs after maintenance, rebuild the dataset again before
backtesting or serving methods that consume ensemble features.

## Installing

1. Copy all four templates into `~/Library/LaunchAgents/` and fill their
   `__PLACEHOLDERS__` (repository paths, log directory, coordinates, output
   path, forecast database, and Synoptic token). Keep each label matching its
   filename.

   ```bash
   mkdir -p ~/Library/LaunchAgents ~/.local/state/grounded-weather-forecast
   for job in poll ingest-ensembles predict maintain; do
     cp "docs/launchd/com.grounded-weather-forecast.${job}.plist" ~/Library/LaunchAgents/
     "$EDITOR" "$HOME/Library/LaunchAgents/com.grounded-weather-forecast.${job}.plist"
   done
   ```

2. Load and start every job (modern launchctl syntax):

   ```bash
   for job in poll ingest-ensembles predict maintain; do
     launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.grounded-weather-forecast.${job}.plist"
     launchctl kickstart -k "gui/$(id -u)/com.grounded-weather-forecast.${job}"
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

- The templates invoke `grounded-weather-forecast` from the login shell
  (`sh -lc`), so a `uv tool install grounded-weather-forecast` binary on your
  PATH (`~/.local/bin`) just works. Pin an absolute path if your PATH differs
  under launchd.
- `predict` refusing to serve from stale data and `maintain` finding no folds
  are **normal** early states, not failures — both say so on stdout, and a
  degraded forecast names its cause in `status_reason` (cold start vs a
  fingerprint invalidated by a rebuild).
- launchd `StartCalendarInterval` fires in **local time**; the 6-hourly
  ensemble job uses `StartInterval` (elapsed seconds) precisely so daylight
  saving cannot skip a model cycle.
- The scheduled `predict` run appends to the self-verification history by
  default; the `--no-history`
  ([advanced usage](advanced-usage.md)) flag opts a one-off manual serve out of
  that append, and the cron should not pass it — that history feeds the
  live-MAE demotion gate.
- Keep the Synoptic token out of the plist if you prefer: set
  `synoptic_token = "$SYNOPTIC_TOKEN"` in `config.toml` and provide the
  variable via `launchctl setenv SYNOPTIC_TOKEN ...` instead of the
  `EnvironmentVariables` block.
