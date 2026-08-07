"""All dashboard prose in one reviewable place, keyed by panel id.

Static copy explains what a panel shows and why an operator cares; dynamic
threshold *values* are interpolated by the zone builders so config overrides
stay truthful.
"""

from collections.abc import Mapping

from grounded_weather_forecast.dashboard.model import PanelCopy

ZONE_INTROS: Mapping[str, str] = {
    "A": (
        "Is fresh, trustworthy data flowing in, and are we serving from it? "
        "There is no daemon here — predict refuses to serve from stale data "
        "rather than guessing, so freshness is the whole game."
    ),
    "B": (
        "Can the station's own measurements be trusted? Flagged samples are "
        "nulled, never corrected, and thin coverage silently removes truth "
        "rows from training."
    ),
    "C": (
        "Can the system learn yet? A young live archive with zero folds is "
        "correct behaviour, not a fault — this zone says how far away the "
        "first fold is and what the synthetic backfill covers meanwhile."
    ),
    "D": (
        "Which method is winning, and is the win real? Read order matters: "
        "sample size first, then skill, then significance, then bias — a "
        "great MAE with a fat bias is the signature of a broken correction."
    ),
    "E": (
        "The glass box: fitted coefficients, expert weights, feature "
        "importances, and decay timescales, straight from the serve-time "
        "observability snapshots — not a black-box forecast."
    ),
    "F": (
        "Is what we actually served any good? Served-vs-realized error is "
        "the one bug class a backtest can never catch, and every served "
        "value's lineage traces back to the evaluation that justified it."
    ),
    "G": (
        "Why is the 6 a.m. temperature 12 °C? Pick any served value and see "
        "the method that produced it, the release that selected the method, "
        "the reason it was selected, and the provider inputs behind it."
    ),
    "H": (
        "Is the system getting better or worse? Every nightly report "
        "appends compact snapshots to append-only ledgers under "
        "artifacts/history — backtest skill, selection churn, "
        "served-vs-promise gaps, A/B verdicts, and e-process wealth — so "
        "trends survive fingerprint resets instead of living only in "
        "tonight's report."
    ),
}

PANEL_COPY: Mapping[str, PanelCopy] = {
    "a1": PanelCopy(
        what=(
            "The status of the most recently served forecast document: ready "
            "or degraded, its status reason, release ids, and the dataset "
            "fingerprint it served from."
        ),
        why=(
            "'Degraded' means every slice fell back to the equal-weight "
            "floor because no promoted release matched the current dataset "
            "and config — expected on a young archive, alarming after a "
            "promotion has happened."
        ),
        thresholds=(
            "status == 'degraded' or empty release_ids turns this amber; a "
            "refused serve (NoForecastDataError) is the one hard red."
        ),
    ),
    "a2": PanelCopy(
        what="How old the newest station observation is right now.",
        why=(
            "The station's live thermometer is the one asset no provider "
            "has. When observations stop, short-lead anchoring silently "
            "degrades to the unanchored blend."
        ),
        thresholds=(
            "Amber past the 30-minute serve staleness cap "
            "(serve/predict.py::OBS_STALENESS); red once the lag exceeds the "
            "provider freshness cap, because then ingestion itself is down."
        ),
    ),
    "a3": PanelCopy(
        what=(
            "Hours since each provider's forecast was fetched, in the newest "
            "archive snapshot."
        ),
        why=(
            "A provider that ages past the cap silently drops out of the "
            "snapshot — it does not error, it just vanishes. Grey bars make "
            "'gone' visible."
        ),
        thresholds=(
            "The line is config [forecasts].max_forecast_age_hours; anything "
            "past it is excluded from serving and greyed here."
        ),
    ),
    "b1": PanelCopy(
        what=(
            "Per-channel QC outcomes over the raw station samples: clean, "
            "missing, out-of-bounds, spike, and flatline counts."
        ),
        why=(
            "Flagged samples are nulled, never corrected. The chart preserves "
            "lifetime QC counts; the status badge follows the flagged share "
            "and the latest causal flatline state, whichever is worse."
        ),
        thresholds=(
            "Amber past 5% flagged, red past 25% — measured per channel as "
            "well as pooled, so one dead sensor cannot average away, and "
            "against reported samples only, so an uninstalled sensor's nulls "
            "do not count as flags. Bounds and step limits come from "
            "[qc].bounds / max_step; flatline detection from "
            "[qc].flatline_minutes, and a flatline alert clears when the "
            "latest causal state recovers. Flag bits overlap, so "
            "clean+missing+flagged can exceed the sample count."
        ),
    ),
    "b2": PanelCopy(
        what="Daily mean truth coverage per variable, as a calendar heatmap.",
        why=(
            "Hours below the coverage floor are nulled out of the truth "
            "tables entirely — thin coverage is training data quietly "
            "disappearing, and a widening gap means the collector is dying."
        ),
        thresholds=(
            "config [dataset].min_hour_coverage / min_day_coverage (both "
            "0.8 by default) — cells below the floor are the ones being "
            "discarded."
        ),
    ),
    "b3": PanelCopy(
        what="Share of null forecast values per provider in the live matrix.",
        why=(
            "Provider QC nulls implausible values (absolute bounds and "
            "cross-source outliers). One provider being nulled far more than "
            "its peers is a provider going bad — unit error or backend "
            "regression."
        ),
        thresholds=(
            "Cross-source outliers governed by [provider_qc] mad_k and "
            "min_sources; absolute bounds by [provider_qc].bounds."
        ),
    ),
    "b4": PanelCopy(
        what="The live/synthetic provenance wall, asserted.",
        why=(
            "Live and synthetic data must never pool — a synthetic-trained "
            "score would flatter the leaderboard. The wall is a filesystem "
            "property; this badge just makes the guarantee legible."
        ),
        thresholds="Any frame carrying mixed source_kind values is a red.",
    ),
    "b5": PanelCopy(
        what=(
            "SNHT change-point verdict on the daytime, inversion-screened "
            "station-minus-neighbor daily series, with the Craddock cusum "
            "and pairwise attribution (station drift vs regime)."
        ),
        why=(
            "Everything upstream calibrates to the station; a drifted "
            "sensor poisons every fit. When all providers alarm at once the "
            "common factor is the station or the regime — this panel is the "
            "arbiter. Drifts persist, regimes revert, so the verdict only "
            "latches after three consecutive daily exceedances."
        ),
        thresholds=(
            "Amber on an SNHT exceedance (small-sample criticals, 95%); red "
            "only when a station-drift attribution latches for 3 straight "
            "days. [truth_qc].gate_fitting decides whether a latch also "
            "quarantines training labels (ships off)."
        ),
    ),
    "c1": PanelCopy(
        what=(
            "How much live archive exists versus what the first "
            "rolling-origin fold needs, plus snapshots collected per day."
        ),
        why=(
            "Zero folds on a young archive is correct, not broken — but "
            "archive growth stalling is the real emergency, because every "
            "missed snapshot is training data that can never be recovered."
        ),
        thresholds=(
            "A fold needs [backtest].initial_train_days + step_days of "
            "issue-time span. The alert is growth stopping, not the count "
            "being low."
        ),
    ),
    "c2": PanelCopy(
        what="What the synthetic (backfill) archive covers.",
        why=(
            "Synthetic folds let methods be compared before the live "
            "archive matures. Open-Meteo backfill only reaches 24h-multiple "
            "leads — the 0–24 h gap is structural, drawn empty on purpose."
        ),
        thresholds="Synthetic evidence is never promoted to serving.",
    ),
    "c3": PanelCopy(
        what=(
            "Per dual-semantics variable: whether providers publish "
            "instantaneous or hour-mean values, and whether that call is "
            "data-backed or defaulted."
        ),
        why=(
            "On a thin archive every variable silently defaults to "
            "instantaneous — fine until it isn't. A defaulted row is a "
            "decision the data has not confirmed yet."
        ),
        thresholds="Data-backed requires n >= 72 aligned rows per source.",
    ),
    "d1": PanelCopy(
        what=(
            "The per-slice leaderboard: every method's MAE, RMSE, bias, and "
            "probabilistic scores per (product, variable, lead bucket)."
        ),
        why=(
            "Read n first — a method scored on 6 rows beats nothing. Then "
            "skill, then the DM p-value, then bias. A big |bias| next to a "
            "fine MAE is exactly how a grounding bug looks."
        ),
        thresholds=(
            "Bias cells highlight beyond the per-variable consumer "
            "tolerance (reports/leaderboard.py::CONSUMER_TOLERANCES)."
        ),
    ),
    "d2": PanelCopy(
        what="The winning method per slice, after the promotion gate.",
        why=(
            "A challenger only replaces the reference when the gate says "
            "the win is statistically real — otherwise selection bias would "
            "promote lucky methods."
        ),
        thresholds=(
            "Eligibility: coverage >= 0.8, n >= 8, distinct valid times >= "
            "8. Gate: [promotion].rule ('mcs' or 'legacy') at alpha."
        ),
    ),
    "d3": PanelCopy(
        what="The baseline floor: climatology and persistence MAE per bucket.",
        why=(
            "Everything is measured against this floor. If climatology's "
            "MAE balloons (the collinear-harmonic failure) every method "
            "above it is flattered; if it beats the best provider at short "
            "leads, suspect leakage."
        ),
        thresholds="No knob — a structural sanity check on the floor itself.",
    ),
    "d4": PanelCopy(
        what=(
            "Pearson correlation of provider errors, and the effective "
            "ensemble size k_eff it implies."
        ),
        why=(
            "Eight providers that copy the same model are not eight "
            "opinions. k_eff ~= n / (1 + (n-1) * mean_r) — diversity beats "
            "count when choosing what to poll."
        ),
        thresholds="Cells need >= 24 overlapping scored hours to render.",
    ),
    "d5": PanelCopy(
        what=(
            "Probabilistic calibration: CRPS, pinball, interval coverage, "
            "PIT histogram, sharpness, and the PoP reliability diagram."
        ),
        why=(
            "A forecast can have a fine MAE and useless uncertainty bands. "
            "Coverage@80 far from 0.8, a U-shaped PIT, or a reliability "
            "curve off the diagonal all say the quantiles are lying."
        ),
        thresholds=(
            "Needs >= 8 scored rows per slice; PIT chi-square p-value "
            "flags non-uniformity. These scores are computed by the "
            "backtest but surfaced only here."
        ),
    ),
    "e1": PanelCopy(
        what=(
            "Fitted grounding coefficients per provider and lead bucket: "
            "intercept a (the bias correction) and slope b."
        ),
        why=(
            "With the bias-only default, |a| is the provider's measured "
            "bias. Watch affine variants: while a stays large and b sits "
            "near 1, the free slope isn't earning its keep."
        ),
        thresholds=(
            "Grey cells are the IDENTITY fallback (fewer than 24 rows, "
            "degenerate variance, or |slope| > 5) — a correction silently "
            "absent, not zero."
        ),
    ),
    "e2": PanelCopy(
        what=(
            "Online-expert weights per provider and lead bucket, and their "
            "trajectory across serves."
        ),
        why=(
            "Fixed share means no expert can freeze at 0 or 1 — so a "
            "weight collapsing is the online, refit-free signal that a "
            "provider swapped its backend model or degraded."
        ),
        thresholds=(
            "A leading-expert flip within a 3-day window raises the "
            "backend-swap alert. Trajectories come from "
            "artifacts/observability/history.parquet."
        ),
    ),
    "e3": PanelCopy(
        what="GBM stacker feature importances (gain), when the GBM is fit.",
        why=(
            "The importances say what the nonlinear ceiling actually uses — "
            "source columns, lead, spread, or the issue-time observations. "
            "If lightgbm is missing on this host the method is silently "
            "absent from the registry; this panel makes that loud."
        ),
        thresholds="Compact snapshot only — the booster itself is never stored.",
    ),
    "e4": PanelCopy(
        what=(
            "The fitted anchoring decay timescale tau per variable, and the "
            "weight curve exp(-lead/tau) it implies."
        ),
        why=(
            "tau = None means the grid chose 'no anchoring' — honest "
            "degradation, not a fault. On >= 24 h-only synthetic data "
            "anchoring always degrades to the base blend, so a None here "
            "may just mean the data cannot exercise it yet."
        ),
        thresholds=(
            "Grid 0.5–24 h; weights below 0.05 are floored to zero, dead "
            "by ~12 h at typical taus."
        ),
    ),
    "e5": PanelCopy(
        what=(
            "best_provider's per-bucket source rankings, and the fold "
            "origins the backtest actually evaluated."
        ),
        why=(
            "Which single provider wins each horizon is the simplest "
            "glass-box fact there is — and the fold timeline shows the "
            "leakage guard at work (training truth known at origin, not "
            "issued by origin)."
        ),
        thresholds="Rankings are training-MAE order; ties broken by index.",
    ),
    "f1": PanelCopy(
        what=(
            "Served-vs-realized error per slice: live MAE against the "
            "backtest's promise, and the gap between them."
        ),
        why=(
            "A large positive gap means the serving path quietly diverged "
            "from the backtested one — the one bug class a backtest "
            "cannot catch by construction."
        ),
        thresholds=(
            "Needs >= 5 realized rows per release cohort. Red beyond "
            "[promotion].live_gap_factor x backtest MAE. This is an early "
            "warning, not the demotion gate: the gate pools cohorts and also "
            "requires n >= [promotion].min_live_n, so a slice can show red "
            "here while the method keeps serving."
        ),
    ),
    "f2": PanelCopy(
        what="Served slices per day, stacked by selection reason.",
        why=(
            "A persistently 100%-degraded system is expected on a young "
            "live archive; the reasons say whether that is cold start, "
            "invalidated evidence, or gates not yet met."
        ),
        thresholds=(
            "Reasons are the exact selection strings the serve path "
            "records; 'degraded'/'no backtest evidence' count toward the "
            "degraded share."
        ),
    ),
    "f3": PanelCopy(
        what=(
            "Release lineage: dataset fingerprint -> evaluations -> "
            "release -> served documents."
        ),
        why=(
            "Answers 'which evaluation justified this served value?'. "
            "release_id is a content hash (promoted_at excluded), so "
            "identical content promoted twice dedupes to one release and "
            "only genuine churn shows."
        ),
        thresholds=(
            "A release whose dataset fingerprint no longer matches the "
            "manifest is stale evidence — re-run backtest and report."
        ),
    ),
    "g1": PanelCopy(
        what=(
            "The latest served document, explorable: pick a product, time, "
            "and variable to see the value's full provenance."
        ),
        why=(
            "Every emitted value carries its method attribution, selection "
            "reason, release id, and quantiles — this panel just renders "
            "the provenance the document was designed to carry. The same "
            "document is replayable byte-for-byte via predict --now."
        ),
        thresholds=(
            "Hourly and daily provider inputs use the newest provider snapshot "
            "visible at the served issue and match its exact point; each source "
            "shows a value and fetch age. Minutely rows expose served "
            "provenance, but no raw input matrix."
        ),
    ),
    "h1": PanelCopy(
        what=(
            "Best recent-window (14-day) backtest MAE per product.variable, "
            "one point per evaluation, from history/quality.parquet."
        ),
        why=(
            "The nightly leaderboard is a snapshot; this is the film. A slow "
            "MAE rise that never trips a single-night gate is exactly what a "
            "ledger catches."
        ),
        thresholds=(
            "Top 8 series by sample count, last 60 evaluations; amber when "
            "the newest point exceeds 1.15x the median of the prior seven. "
            "Provenance columns keep trends segmentable across fingerprint "
            "resets."
        ),
    ),
    "h2": PanelCopy(
        what=(
            "Share of promoted slices whose serving method changed at each "
            "release transition, plus the newest transition's per-slice diff."
        ),
        why=(
            "Churn is the promotion gate's temperature: healthy selection "
            "settles over time; winners flapping on overnight noise is "
            "multiplicity showing through."
        ),
        thresholds=(
            "Amber when the latest transition changed at least 30% of "
            "compared slices, red past 60%. Needs two promoted releases; "
            "added or removed slices count as changes."
        ),
    ),
    "h3": PanelCopy(
        what=(
            "Realized served MAE beside the backtest MAE it was promoted "
            "on, pooled per product by day, from history/served_quality.parquet."
        ),
        why=(
            "A gap that trends positive means the serving path is drifting "
            "from the backtested one — over time, not just tonight (zone F "
            "holds the single-night view)."
        ),
        thresholds=(
            "Amber when the pooled mae_gap is positive on 4 of the last 7 "
            "days; red when pooled live MAE exceeds [promotion].live_gap_factor "
            "x backtest MAE."
        ),
    ),
    "h4": PanelCopy(
        what=(
            "A/B verdict trends: the share of slices each quantile repair "
            "(raw/pit/cqr) wins, and how often the mcs and seq_mcs promotion "
            "gates agree."
        ),
        why=(
            "One night's A/B is noise; a stable cqr win share is the "
            "evidence that flips [predict].quantile_recalibration, and a "
            "falling agree rate warns before switching [promotion].rule."
        ),
        thresholds=(
            "Wins judged by coverage80 closest to 0.8 (pinball breaks "
            "ties); amber when the latest gate agree rate drops below 0.8."
        ),
    ),
    "h5": PanelCopy(
        what=(
            "Log e-process wealth for the pairs nearest (or past) the "
            "promotion threshold, with log(1/alpha) drawn."
        ),
        why=(
            "Wealth crossing the line is anytime-valid evidence a "
            "challenger genuinely beats its reference; wealth resets when "
            "config or code identity changes, and the ledger keeps eras "
            "segmented."
        ),
        thresholds=(
            "Top 6 pairs by latest wealth, last 60 snapshots; threshold "
            "log(1/[promotion].alpha). Informational — crossing drives "
            "promotion, not alarm."
        ),
    ),
    "h6": PanelCopy(
        what=(
            "14-day interval coverage (nominal 80%) per quantile-emitting "
            "method, with the 0.80 target drawn."
        ),
        why=(
            "The pooled expanding-window coverage is frozen by months of "
            "pre-repair history; only a recent window can show whether a "
            "calibration repair (conformal PID, per-bucket IDR) is working "
            "now rather than eventually."
        ),
        thresholds=(
            "Live rows only, >=20 recent quantile rows per slice; amber "
            "when a method's newest point sits more than 0.10 from the "
            "0.80 target."
        ),
    ),
}
