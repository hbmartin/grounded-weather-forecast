# Contributing

Thanks for your interest. This file is the **canonical** description of the
development workflow and the gates CI enforces — `CLAUDE.md` and `AGENTS.md`
carry only agent-specific conventions and point here for everything else.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/hbmartin/grounded-weather-forecast
cd grounded-weather-forecast
uv sync --dev
```

Optional extras:

```bash
uv sync --extra backfill    # dynamical.org Zarr backfill (xarray/zarr)
uv sync --group docs        # zensical + mkdocstrings, for the docs site
```

No sample databases are committed. Every test synthesizes its own fixtures, so
the suite runs with no external data and never calls a live API.

---

## The gates

CI runs these on Python 3.13 **and** 3.14. Run them locally before opening a PR;
the list below is exactly what `.github/workflows/ci.yml` executes.

```bash
# 1. Lockfile integrity
uv lock --check
uv run --no-project python scripts/check_lock_hosts.py

# 2. Lint and format  (CI checks `ruff format --check src tests` — note: src AND tests)
uv run ruff check src --fix
uv run ruff format src tests

# 3. Semgrep guardrails (project-specific invariants, with their own test files)
uvx --from semgrep==1.170.0 semgrep scan --test --config semgrep/provider-qc.yml semgrep/tests/provider_qc_grouping.py
uvx --from semgrep==1.170.0 semgrep scan --metrics=off --error --config semgrep/provider-qc.yml src/grounded_weather_forecast/dataset/matrix.py
uvx --from semgrep==1.170.0 semgrep scan --test --config semgrep/artifact-pointer-paths.yml semgrep/tests/artifact_pointer_paths.py
uvx --from semgrep==1.170.0 semgrep scan --metrics=off --error --config semgrep/artifact-pointer-paths.yml src/grounded_weather_forecast/artifacts.py

# 4. Type checking — both, because their disagreements have been useful
uv run pyrefly check src
uv run ty check src

# 5. Packaging and dependency hygiene
uv run deptry src
uv run pyroma --min 8 .

# 6. Complexity and duplication (ceiling 27; the exclusion skips vendored Chart.js)
uv run lizard -Eduplicate -C 27 -x "*/dashboard/assets/*" src

# 7. Tests and coverage
uv run pytest tests/ --cov=src --cov-report=term-missing
```

| Gate | Threshold |
|---|---|
| Coverage | **`fail_under = 88`** in `pyproject.toml` — that file is authoritative |
| Lizard cyclomatic complexity | **27** — refactor rather than raising it |
| Pyroma | minimum score 8 |

If the docs changed, also:

```bash
uv run zensical build --strict --clean
```

`--strict` validates internal links **and heading anchors**, so a renamed section
breaks the build rather than shipping a dead link.

---

## Conventions

- **Use `uv`, not `python`, to run anything.**
- **Ruff, not Black**, for formatting; keep local and CI aligned.
- **Type hints are first-class.** Prefer explicitness and small functions.
- Use modern Python — assignment expressions, structural pattern matching — where
  it genuinely reads better.
- Use a **parenthesized tuple** of exception classes in `except` clauses.
- When dependencies change, run `uv lock` and verify Deptry still passes.
- Use `httpx2.MockTransport` for provider tests. **Tests must never call live
  weather APIs.**
- Preserve unrelated working-tree changes; inspect existing diffs before editing
  dirty files.
- Add focused regression tests for changed behaviour.

---

## Architectural invariants

These are enforced by tests and will fail CI if broken. They are not style
preferences — each one is a defence against a specific class of bug.

1. **`contracts` and `leads` are the only deep-import targets.** Blenders import
   `contracts` only, never `dataset`.
2. **No column starting `t__` may reach a blender.**
   `ForecastMatrix.__post_init__` raises, so the illegal object cannot be
   constructed. This is leakage defence #4.
3. **The registry stores factories, never instances.** The engine builds a fresh
   blender per fold so a stateful method cannot leak weights across the
   train/test boundary. A test asserts this over the whole registry.
4. **Live and synthetic rows are never pooled.** The provenance is in the matrix
   filename; mixing raises `MixedProvenanceError`.
5. **The dataset build is byte-reproducible.** Stable sorts precede every pivot,
   and the manifest fingerprint gates artifact staleness.
6. **The scores frame is long and dumb.** The engine never declares a winner —
   every leaderboard is a downstream `group_by`, which is what lets `report`
   re-decide under new statistical rules without re-running a fit.
7. **Truth is never imputed or corrected.** QC nulls; alarms report. Nothing
   silently adjusts an observation.

The reasoning behind each is in
[Architecture](https://hbmartin.github.io/grounded-weather-forecast/architecture/)
and [Methods: verification](https://hbmartin.github.io/grounded-weather-forecast/methods/verification/).

---

## Adding a blending method

The bar is deliberately low to *add* and high to *ship*: register it, and the
leaderboard decides. Nothing is promoted because it sounds good.

1. Implement the `Blender` protocol in `src/grounded_weather_forecast/blenders/`.
2. `register("my_method", MyMethod)` — a **factory**, not an instance.
3. Honour the shared conventions: renormalize weights over the availability mask,
   route quantiles through `finalize_quantiles`, and **abstain honestly** (return
   `NaN` with a `fit_status`) rather than fitting on too few rows.
4. Add tests. `tests/test_registry.py` will exercise protocol compliance
   automatically.
5. Document the mathematics in the appropriate `docs/methods/` page, with an
   `Implemented in:` line and the actual constants.

Worked example:
[Advanced usage](https://hbmartin.github.io/grounded-weather-forecast/advanced-usage/#adding-a-blending-method).

---

## Documentation

- **User-facing changes must be documented.** A new CLI flag needs a row in the
  README's commands-at-a-glance table *and* full semantics in
  `docs/reference/cli.md`. A new config key needs an entry in
  `docs/reference/configuration.md`.
- **New methods need mathematics.** `docs/theory.md` owns *why* and *which*;
  `docs/methods/` owns *what exactly*, with constants and guards. **No formula
  should appear in both.**
- **Cite what you implement.** Add the reference to
  `docs/methods/bibliography.md` alongside the module that implements it. If you
  considered an approach and rejected it, record that too — the "declined"
  section is as useful as the rest.
- Docs claims are expected to match the code. If you change a constant, grep the
  docs for it.

---

## Pull requests

- Branch from `main`.
- Keep the diff focused; unrelated cleanups belong in their own PR.
- Run the gates above before pushing.
- Describe *what changed and why*. If the change affects forecast quality, say
  what evidence supports that — this project's whole premise is that claims are
  arbitrated by measurement.

Releases are maintainer-only; the process is in
[docs/releasing.md](docs/releasing.md).
