# 0.4.0 — the operator dashboard

This release makes the system's built-in honesty visible at a glance:
`report` now writes `reports/dashboard.html`, a fully offline,
self-contained operator console with seven zones (liveness, data trust,
learning readiness, evaluation, model internals, serving, explainability)
and a threshold-alert strip whose every limit is an existing config knob or
module constant. See [Operator Dashboard](dashboard.md).

Two new on-disk signals feed it:

- **`[dataset].dir/runs.parquet`** — an append-only ledger of every command
  whose configuration loads successfully (command, args, timing, exit code or exception name,
  dataset/config fingerprints, code version). Telemetry never breaks a
  command: writes take a 5-second lock timeout and swallow failures. Parser
  and configuration-loading failures cannot be recorded because that
  configuration supplies the ledger destination.
- **`[artifacts].dir/observability/`** — write-only snapshots of each
  fitted blender's compact internals on every `predict` run (grounding
  coefficients, online-expert weights with an ewa/boa trajectory history
  pruned to `[backtest].rolling_window_days`, GBM feature importances,
  anchoring decay timescale, best_provider rankings). Serving output is
  bit-identical whether snapshots land or not; the rehydration store at
  `artifacts/state/` is untouched.

Also new: `reports/alerts.py` (the pure alert evaluator the dashboard
renders), public `OBS_STALENESS` / `LOCATION_TOLERANCE` constants (renamed
from private), an optional `timeout` on `storage.locked_path`, and
`ArtifactStore.load_manifest`.

Both new on-disk signals are bounded: the run ledger is pruned to the last 90
days and 50,000 rows on each append, and observability snapshot trees left
behind by a superseded dataset fingerprint are reclaimed after each write.

Two serving-visible corrections landed alongside the dashboard:

- **Ensemble features are resolved identically in training and serving.**
  `build-dataset` filters `ensembles.parquet` to the configured
  `[ensembles].models`/`variables`; `predict` now applies the same filter. It
  previously loaded every stored row, so EMOS could fit its spread
  coefficient against one predictor and apply it to another — silently
  miscalibrating every interval when a model was retired from config.
- **The minutely nowcast no longer steps between anchoring regimes.** When
  the two bracketing hourly rows came from different method families the path
  fell back to nearest-neighbour, producing a flat line with one large jump
  and flipping the anchoring decision halfway — so the live station
  observation was ignored for the minutes on the anchored side. The regime of
  the row owning lead zero now governs the whole range and the path stays
  interpolated.

No existing artifact schemas changed and no migration is required: the new
files are additive, the dashboard renders honest "not yet" states wherever
history hasn't accumulated, and CI's lizard step now excludes the vendored
Chart.js asset (`-x "*/dashboard/assets/*"`).

## Upgrade

```bash
uv tool install grounded-weather-forecast   # or: uv sync in a checkout
grounded-weather-forecast --version          # expect 0.4.0
grounded-weather-forecast report             # writes reports/dashboard.html
```

Open `reports/dashboard.html` in any browser (it works from `file://`).
On a young deployment most panels are deliberately grey or amber — zero
live folds and no promoted releases are correct behaviour, and the page
says so instead of crying wolf.
