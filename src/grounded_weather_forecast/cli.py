"""Command-line interface: thin wrappers over library functions."""

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from grounded_weather_forecast import __version__
from grounded_weather_forecast.backtest.outcomes import (
    BacktestDiagnostics,
    ProductBacktestOutcome,
)
from grounded_weather_forecast.config import Config, ConfigError, load_config
from grounded_weather_forecast.contracts import (
    HOURLY_VARIABLES,
    TruthSemantics,
    VariableSpec,
)
from grounded_weather_forecast.serve.schema import Forecast
from grounded_weather_forecast.status import EXIT_NO_FOLDS, CommandOutcome

HOURLY_DEFAULT_VARIABLES = (
    "temp_c,humidity_pct,dew_point_c,wind_speed_ms,wind_gust_ms,"
    "pressure_sea_hpa,precip_mm,pop"
)
DAILY_DEFAULT_VARIABLES = "temp_max_c,temp_min_c,pop,precip_sum_mm"


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _add_backtest_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_source: bool,
) -> None:
    parser.add_argument(
        "--methods",
        default="all",
        help="comma-separated method ids, or 'all' registered (default)",
    )
    parser.add_argument(
        "--hourly-variables",
        default=HOURLY_DEFAULT_VARIABLES,
        help="comma-separated hourly variables",
    )
    parser.add_argument(
        "--daily-variables",
        default=DAILY_DEFAULT_VARIABLES,
        help="comma-separated daily variables",
    )
    parser.add_argument(
        "--products",
        default="hourly,daily,minutely",
        help="comma-separated products to backtest (hourly,daily,minutely;"
        " minutely scores the nowcast path constructions and is live-only)",
    )
    parser.add_argument(
        "--window",
        choices=("expanding", "rolling"),
        default="expanding",
        help="training window mode",
    )
    if include_source:
        parser.add_argument(
            "--source",
            choices=("live", "synthetic"),
            default="live",
            help="which provenance the matrices must carry",
        )
    else:
        parser.set_defaults(source="live")
    parser.add_argument(
        "--semantics",
        choices=("auto", "inst", "mean"),
        default="auto",
        help="hourly truth semantics: inst, mean, or auto (alignment artifact"
        " majority recommendation, falling back to inst); variables without"
        " dual semantics always use inst",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grounded-weather-forecast",
        description="Station-grounded blending of multi-provider weather forecasts.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="path to the TOML configuration (default: config.toml)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("qc", help="summarize station truth quality control")
    subparsers.add_parser(
        "build-dataset",
        help="materialize truth tables and supervised matrices as parquet",
    )
    backtest = subparsers.add_parser(
        "backtest", help="rolling-origin backtest over the supervised matrices"
    )
    _add_backtest_arguments(backtest, include_source=True)
    maintain = subparsers.add_parser(
        "maintain",
        help="run build, live backtest, truth QC, and report under one lock",
    )
    _add_backtest_arguments(maintain, include_source=False)
    maintain.add_argument(
        "--truth-qc-days",
        type=_positive_int,
        default=30,
        help="neighbor history window passed to truth-qc (default: 30)",
    )
    subparsers.add_parser(
        "recover",
        help="deduplicate and run build, live backtest, and report recovery",
    )
    subparsers.add_parser(
        "report", help="render leaderboards and correlation reports from scores"
    )
    subparsers.add_parser(
        "alignment",
        help="study truth semantics per provider; write alignment artifact",
    )
    prune = subparsers.add_parser(
        "prune-scores",
        help="delete superseded scores files (newest per group and"
        " release-referenced evaluations are kept; catalog rows survive)",
    )
    prune.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be deleted without deleting anything",
    )
    backfill = subparsers.add_parser(
        "backfill",
        help="fetch archived forecasts into the synthetic supervised matrix",
    )
    backfill.add_argument(
        "--provider",
        choices=("open_meteo", "dynamical", "open_meteo_ensemble"),
        default="open_meteo",
        help="open_meteo: Previous Runs (24h-multiple leads); dynamical:"
        " dynamical.org full cycles at native steps (sub-24h leads; needs"
        " the 'backfill' optional extra); open_meteo_ensemble: archived"
        " ensemble mean/spread into the ens__ feature store",
    )
    backfill.add_argument(
        "--models",
        default="",
        help="comma-separated models (default: the provider's config list)",
    )
    backfill.add_argument(
        "--start",
        type=date.fromisoformat,
        default=None,
        help="first valid date for open_meteo or first initialization date for"
        " dynamical, YYYY-MM-DD (default: the provider's config start_date)",
    )
    backfill.add_argument(
        "--end",
        type=date.fromisoformat,
        default=None,
        help="last valid date for open_meteo or last initialization date for"
        " dynamical, YYYY-MM-DD (default: yesterday)",
    )
    backfill.add_argument(
        "--chunk-days",
        type=int,
        default=90,
        help="days per request (open_meteo only; default: 90)",
    )
    truth_qc = subparsers.add_parser(
        "truth-qc",
        help="cross-check station truth against lapse-adjusted Synoptic"
        " neighbors and fit the radiation-shield error model",
    )
    truth_qc.add_argument(
        "--days",
        type=int,
        default=30,
        help="neighbor history window in days (default: 30)",
    )
    ingest = subparsers.add_parser(
        "ingest-ensembles",
        help="poll the Open-Meteo Ensemble API and append per-model spread"
        " statistics to the ensembles parquet store",
    )
    ingest.add_argument(
        "--models",
        default="",
        help="comma-separated ensemble models (default: [ensembles].models)",
    )
    predict = subparsers.add_parser(
        "predict", help="emit the current blended forecast as JSON"
    )
    predict.add_argument(
        "--out",
        default="-",
        help="output path, or '-' for stdout (default)",
    )
    predict.add_argument(
        "--method",
        default="auto",
        help="force one method for every slice, or 'auto' to use the leaderboard",
    )
    predict.add_argument(
        "--no-history",
        action="store_true",
        help="do not append this forecast to the self-verification history",
    )
    predict.add_argument(
        "--semantics",
        choices=("auto", "inst", "mean"),
        default="auto",
        help="hourly truth semantics used when fitting (see backtest)",
    )
    predict.add_argument(
        "--now",
        type=datetime.fromisoformat,
        default=None,
        help="issue the forecast as of this UTC instant instead of now; useful"
        " for reproducing a served forecast from an archived snapshot",
    )
    publish = subparsers.add_parser(
        "publish",
        help="publish a forecast with hold-last-good and automatic recovery",
    )
    publish.add_argument(
        "--out",
        required=True,
        help="forecast JSON destination (required)",
    )
    publish.add_argument(
        "--semantics",
        choices=("auto", "inst", "mean"),
        default="auto",
        help="hourly truth semantics used when fitting (see backtest)",
    )
    publish.add_argument(
        "--recovery-methods",
        default="all",
        help="methods to retrain during automatic recovery (default: all)",
    )
    publish.add_argument(
        "--recovery-window",
        choices=("expanding", "rolling"),
        default="expanding",
        help="backtest window used during automatic recovery",
    )
    recover = subparsers.choices["recover"]
    recover.add_argument(
        "--semantics", choices=("auto", "inst", "mean"), default="auto"
    )
    recover.add_argument("--methods", default="all")
    recover.add_argument(
        "--window", choices=("expanding", "rolling"), default="expanding"
    )
    return parser


def _cmd_qc(config: Config) -> int:
    from grounded_weather_forecast.dataset.qc import (  # noqa: PLC0415
        apply_qc,
        qc_summary,
    )
    from grounded_weather_forecast.dataset.station import (  # noqa: PLC0415
        read_observations,
    )
    from grounded_weather_forecast.dataset.truth import (  # noqa: PLC0415
        truth_daily,
        truth_hourly,
        truth_minute,
    )

    channels = sorted(set(config.station.columns.values()))
    observations = read_observations(config.station)
    if observations.is_empty():
        print("no observations found")
        return 1
    flagged = apply_qc(observations, config.qc, channels)
    summary = qc_summary(flagged, channels)
    minute = truth_minute(flagged, config)
    hourly = truth_hourly(minute, config)
    daily = truth_daily(minute, config)

    span = (observations["ts"].min(), observations["ts"].max())
    print(f"observations: {observations.height} samples, {span[0]} .. {span[1]}")
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(summary)
        hourly_nonnull = hourly.select(
            pl.col("valid_hour").len().alias("hours"),
            *(
                pl.col(c).is_not_null().sum().alias(c)
                for c in hourly.columns
                if c.startswith("t__")
            ),
        )
        print("hourly truth (non-null counts):")
        print(hourly_nonnull)
        daily_nonnull = daily.select(
            pl.col("date_local").len().alias("days"),
            *(
                pl.col(c).is_not_null().sum().alias(c)
                for c in daily.columns
                if c.startswith("t__")
            ),
        )
        print("daily truth (non-null counts):")
        print(daily_nonnull)
    return 0


def _cmd_build_dataset(config: Config) -> int:
    from grounded_weather_forecast.dataset.matrix import write_dataset  # noqa: PLC0415

    manifest = write_dataset(config)
    print(f"dataset fingerprint: {manifest.fingerprint}")
    print(f"sources: {', '.join(manifest.sources) or '<none>'}")
    print(f"snapshots: {manifest.snapshots}")
    for name, info in manifest.files.items():
        print(f"  {name}: {info.rows} rows ({info.sha256_16})")
    return 0


def _split_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _alignment_artifact_path(config: Config) -> Path:
    return config.artifacts_dir / "alignment.json"


def _resolve_semantics(
    config: Config, flag: str, variable: str | None = None
) -> TruthSemantics:
    from grounded_weather_forecast.dataset.alignment import (  # noqa: PLC0415
        load_recommended,
    )

    match flag:
        case "inst":
            return TruthSemantics.INSTANTANEOUS
        case "mean":
            return TruthSemantics.INTERVAL_MEAN
        case _:
            recommended = load_recommended(_alignment_artifact_path(config))
            if variable is None or variable not in recommended:
                return TruthSemantics.INSTANTANEOUS
            return TruthSemantics(recommended[variable])


def _semantics_by_variable(
    config: Config, flag: str, variables: Sequence[VariableSpec]
) -> dict[str, TruthSemantics]:
    """Requested truth target per variable name.

    A variable without dual semantics is only ever evaluated against its
    single instantaneous truth column, and backtest evidence records it that
    way. Binding it to an explicit ``--semantics mean`` would demand evidence
    that cannot exist, stranding the variable on the no-evidence fallback.
    """
    return {
        variable.name: (
            _resolve_semantics(config, flag, variable.name)
            if variable.has_dual_semantics
            else TruthSemantics.INSTANTANEOUS
        )
        for variable in variables
    }


def _live_hourly_matrix_path(config: Config) -> Path:
    from grounded_weather_forecast.dataset.matrix import matrix_path  # noqa: PLC0415

    return matrix_path(config.dataset.dir, "hourly", "live")


def _cmd_alignment(config: Config) -> int:
    from grounded_weather_forecast.dataset.alignment import (  # noqa: PLC0415
        alignment_study,
        write_alignment,
    )
    from grounded_weather_forecast.reports.render import (  # noqa: PLC0415
        print_summary,
        write_markdown_report,
    )

    matrix_file = _live_hourly_matrix_path(config)
    if not matrix_file.exists():
        print(f"missing {matrix_file}; run build-dataset first")
        return 1
    matrix = pl.read_parquet(matrix_file)
    study = alignment_study(matrix)
    if study.is_empty():
        print("not enough overlapping data for an alignment study yet")
        return 1
    artifact = write_alignment(study, _alignment_artifact_path(config))
    write_markdown_report(
        config.reports_dir,
        "alignment",
        "Truth-semantics alignment study",
        [("Correlation by semantics", study)],
    )
    print_summary("alignment study", study)
    print(f"recommended: {artifact['recommended']}")
    match artifact.get("data_backed"):
        case dict() as backed:
            defaulted = sorted(
                name for name, is_backed in backed.items() if not is_backed
            )
            if defaulted:
                print(
                    "on the instantaneous DEFAULT (no source reached "
                    f"{artifact.get('min_rows')} overlapping rows): "
                    f"{', '.join(defaulted)}"
                )
            decided = sorted(name for name, is_backed in backed.items() if is_backed)
            if decided:
                print(f"data-backed: {', '.join(decided)}")
    print(f"wrote {_alignment_artifact_path(config)}")
    return 0


def _dynamical_long(
    config: Config, args: argparse.Namespace, end: date
) -> pl.DataFrame:
    from grounded_weather_forecast.dataset.backfill_dynamical import (  # noqa: PLC0415
        backfill_dynamical_long,
    )

    start = args.start or config.backfill.dynamical_start_date
    if start is None:
        msg = "set [backfill.dynamical].start_date or pass --start"
        raise ValueError(msg)
    models = _split_csv(args.models)
    return backfill_dynamical_long(config, start, end, models=models or None)


def _cmd_backfill_ensembles(config: Config, args: argparse.Namespace, end: date) -> int:
    from grounded_weather_forecast.dataset.ensembles import (  # noqa: PLC0415
        EnsembleError,
        append_ensembles,
        backfill_ensembles,
        ensembles_path,
        load_ensembles,
        trim_backfill_to_live,
    )

    start = args.start or config.backfill.start_date
    if start is None:
        msg = "set [backfill.open_meteo].start_date or pass --start"
        print(f"backfill failed: {msg}")
        return 1
    models = tuple(_split_csv(args.models)) or None
    try:
        frame = backfill_ensembles(
            config, start, end, chunk_days=args.chunk_days, models=models
        )
    except (EnsembleError, OSError, ValueError) as exc:
        print(f"backfill failed: {exc}")
        return 1
    store = ensembles_path(config)
    frame = trim_backfill_to_live(frame, load_ensembles(store))
    new_rows, total_rows = append_ensembles(store, frame)
    print(f"backfilled {frame.height} ensemble statistics rows")
    print(f"models: {', '.join(sorted(frame['model'].unique().to_list()))}")
    print(f"ensemble store: +{new_rows} new rows -> {total_rows} total")
    return 0


def _cmd_backfill(config: Config, args: argparse.Namespace) -> int:
    from grounded_weather_forecast.dataset.backfill import (  # noqa: PLC0415
        BackfillError,
        backfill_long,
    )
    from grounded_weather_forecast.dataset.backfill_dynamical import (  # noqa: PLC0415
        DynamicalBackfillError,
    )
    from grounded_weather_forecast.dataset.matrix import (  # noqa: PLC0415
        write_synthetic_matrix,
    )

    end = args.end or (datetime.now(tz=UTC).date() - timedelta(days=1))
    if args.provider == "open_meteo_ensemble":
        return _cmd_backfill_ensembles(config, args, end)
    try:
        match args.provider:
            case "dynamical":
                long_frame = _dynamical_long(config, args, end)
            case _:
                models = _split_csv(args.models)
                long_frame = backfill_long(
                    config,
                    end,
                    models=models or None,
                    chunk_days=args.chunk_days,
                    start=args.start,
                )
    except (BackfillError, DynamicalBackfillError, OSError, ValueError) as exc:
        print(f"backfill failed: {exc}")
        return 1
    path, rows = write_synthetic_matrix(config, long_frame)
    print(f"backfilled {long_frame.height} forecast points")
    print(f"sources: {', '.join(sorted(long_frame['source'].unique().to_list()))}")
    print(f"synthetic matrix: {rows} rows -> {path}")
    return 0


def _drift_verdict_block(
    config: Config, checks: object, hourly_truth: pl.DataFrame
) -> dict[str, object]:
    """Attribution-grade verdict with a latch persisted across cron runs.

    The streak lives in the previous artifact: SNHT is end-sensitive and
    daily differences are autocorrelated, so a single exceedance is noise
    until it survives three consecutive daily evaluations — drifts persist,
    regimes revert. Only a persistent ``station_drift`` attribution latches.
    """
    import json  # noqa: PLC0415

    from grounded_weather_forecast.dataset.neighbors import (  # noqa: PLC0415
        DriftVerdict,
        drift_verdict,
    )

    neighbors = getattr(checks, "neighbors", None)
    if neighbors is None or neighbors.is_empty():
        return {
            "statistic_name": config.truth_qc.drift_statistic,
            "exceeded": None,
            "attribution": None,
            "streak": 0,
            "latched": False,
            "reason": "no neighbor observations",
        }
    verdict: DriftVerdict = drift_verdict(
        hourly_truth,
        neighbors,
        timezone=config.station.timezone,
        lapse_k_per_km=config.truth_qc.lapse_k_per_km,
        site_elevation_m=config.station.elevation_m,
        statistic=config.truth_qc.drift_statistic,
    )
    previous_streak = 0
    previous_date: object = None
    artifact_path = config.artifacts_dir / "truth_qc.json"
    try:
        previous = json.loads(artifact_path.read_text(encoding="utf-8"))
        block = previous.get("drift_verdict", {})
        previous_streak = int(block.get("streak", 0))
        previous_date = block.get("as_of_date")
    except (OSError, ValueError, AttributeError):
        previous_streak = 0
    today = datetime.now(tz=UTC).date().isoformat()
    if not verdict.exceeded:
        streak = 0
    elif previous_date == today:
        # A same-day re-run must not double-count toward the latch.
        streak = max(previous_streak, 1)
    else:
        streak = previous_streak + 1
    latched = streak >= 3 and verdict.attribution == "station_drift"
    return {
        "as_of_date": today,
        "statistic_name": verdict.statistic_name,
        "statistic": verdict.statistic,
        "critical": verdict.critical,
        "exceeded": verdict.exceeded,
        "snht_statistic": verdict.snht_statistic,
        "pettitt_p": verdict.pettitt_p,
        "attribution": verdict.attribution,
        "break_date": verdict.break_date,
        "days_used": verdict.days_used,
        "inversion_days_excluded": verdict.inversion_days_excluded,
        "neighbors_used": verdict.neighbors_used,
        "streak": streak,
        "latched": latched,
        "reason": verdict.reason,
        "series": verdict.series.to_dicts(),
    }


def _cmd_truth_qc(config: Config, args: argparse.Namespace) -> int | CommandOutcome:
    import json  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from grounded_weather_forecast.dataset.matrix import build_truth  # noqa: PLC0415
    from grounded_weather_forecast.dataset.neighbors import (  # noqa: PLC0415
        NeighborError,
        fetch_neighbor_checks,
    )
    from grounded_weather_forecast.dataset.truth_qc import (  # noqa: PLC0415
        fit_shield_error,
    )
    from grounded_weather_forecast.reports.render import (  # noqa: PLC0415
        write_markdown_report,
    )
    from grounded_weather_forecast.solar import toa_irradiance_wm2  # noqa: PLC0415

    if args.days <= 0:
        print("truth-qc failed: --days must be a positive integer")
        return CommandOutcome(1, error_detail="--days must be a positive integer")
    _minute, hourly_truth, _daily = build_truth(config)
    try:
        checks = fetch_neighbor_checks(config, hourly_truth, hours=args.days * 24)
    except (NeighborError, OSError, ValueError) as exc:
        print(f"truth-qc failed: {exc}")
        return CommandOutcome(1, error_detail=str(exc))
    verdict_block = _drift_verdict_block(config, checks, hourly_truth)
    artifact: dict[str, object] = {
        "schema_version": 3,
        "history_days": args.days,
        "n_neighbors": checks.n_neighbors,
        "overlap_hours": checks.overlap_hours,
        "drift_alert": checks.drift_alert,
        "drift_reason": checks.drift_reason,
        "correlation_alert": checks.correlation_alert,
        "correlation_reason": checks.correlation_reason,
        "daily_drift": checks.daily_drift.to_dicts(),
        "drift_verdict": verdict_block,
        "shield_alert": None,
        "shield_reason": "insufficient independent daytime neighbor overlap",
    }
    shield_note = str(artifact["shield_reason"])
    wind_column = "t__wind_speed_ms__inst"
    if wind_column in checks.comparison.columns:
        scored = checks.comparison.drop_nulls(["difference", wind_column])
        if scored.height:
            epoch = (
                scored["valid_hour"]
                .dt.epoch(time_unit="s")
                .to_numpy()
                .astype(np.float64)
            )
            toa = toa_irradiance_wm2(
                epoch, config.station.latitude, config.station.longitude
            )
            residual = scored["difference"].to_numpy().astype(np.float64)
            wind = scored[wind_column].to_numpy().astype(np.float64)
            fit = fit_shield_error(residual, toa, wind)
            if fit is not None:
                artifact["shield_fit"] = {
                    "slope_c_per_unit": fit.slope_c_per_unit,
                    "intercept_c": fit.intercept_c,
                    "slope_se": fit.slope_se,
                    "n_daytime": fit.n_daytime,
                    "significant": fit.significant,
                }
                artifact["shield_alert"] = fit.significant
                artifact["shield_reason"] = (
                    f"slope {fit.slope_c_per_unit:+.2f} C per unit load "
                    f"(se {fit.slope_se:.2f}, n={fit.n_daytime})"
                )
                shield_note = (
                    f"slope {fit.slope_c_per_unit:+.2f} degC per unit load "
                    f"(se {fit.slope_se:.2f}, n={fit.n_daytime}); a growing "
                    "positive slope is the failing-shield signature"
                )
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact["generated_at"] = datetime.now(tz=UTC).isoformat()
    (config.artifacts_dir / "truth_qc.json").write_text(
        json.dumps(artifact, indent=2, default=str), encoding="utf-8"
    )
    write_markdown_report(
        config.reports_dir,
        "truth_qc",
        "Station truth cross-checks",
        [
            ("Daily station-minus-consensus (degC)", checks.daily_drift),
            ("Rolling 72h correlation", checks.rolling_correlation.tail(24)),
        ],
    )
    print(f"neighbors: {checks.n_neighbors}")
    print(f"overlap: {checks.overlap_hours} hours")
    print(
        f"drift alert: {checks.drift_alert}  correlation alert: {checks.correlation_alert}"
    )
    print(
        f"drift verdict: exceeded={verdict_block['exceeded']} "
        f"attribution={verdict_block['attribution']} "
        f"streak={verdict_block['streak']} latched={verdict_block['latched']} "
        f"({verdict_block['reason']})"
    )
    print(f"shield: {shield_note}")
    print(f"wrote {config.artifacts_dir / 'truth_qc.json'}")
    # A cold start has no neighbour overlap yet, which is a normal state, not a
    # failure: the shipped `maintain` cron chains truth-qc, so a non-zero exit
    # would report the whole nightly job as broken for the first week — and it
    # is now recorded in the run ledger, where it would read as a real fault.
    # The verdicts and their reasons are in the artifact either way.
    if not any(
        (
            checks.drift_alert is not None,
            checks.correlation_alert is not None,
            artifact["shield_alert"] is not None,
        )
    ):
        print("no check is evaluable yet; see the reasons in truth_qc.json")
    return 0


def _cmd_ingest_ensembles(config: Config, args: argparse.Namespace) -> int:
    from dataclasses import replace  # noqa: PLC0415

    from grounded_weather_forecast.dataset.ensembles import (  # noqa: PLC0415
        EnsembleError,
        append_ensembles,
        ensembles_path,
        ingest_ensembles,
    )

    models = _split_csv(args.models)
    if models:
        config = replace(config, ensembles=replace(config.ensembles, models=models))
    try:
        fresh = ingest_ensembles(config)
        path = ensembles_path(config)
        new_rows, total_rows = append_ensembles(path, fresh)
    except (EnsembleError, OSError, ValueError) as exc:
        print(f"ensemble ingest failed: {exc}")
        return 1
    models_seen = ", ".join(sorted(fresh["model"].unique().to_list()))
    print(f"ingested {fresh.height} statistic rows from {models_seen}")
    print(f"ensembles store: +{new_rows} new rows, {total_rows} total -> {path}")
    return 0


def _forecast_document(
    config: Config,
    *,
    method: str,
    semantics_flag: str,
    now: datetime | None,
    lock_timeout: float | None = None,
) -> Forecast:
    from grounded_weather_forecast.serve.dataset_snapshot import (  # noqa: PLC0415
        load_live_snapshot,
    )
    from grounded_weather_forecast.serve.predict import predict  # noqa: PLC0415
    from grounded_weather_forecast.serve.selection import (  # noqa: PLC0415
        load_approved_selections,
    )

    forced = None if method == "auto" else method
    if now is not None and now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    semantics = _semantics_by_variable(config, semantics_flag, HOURLY_VARIABLES)
    dataset = load_live_snapshot(config, lock_timeout=lock_timeout)
    selections = (
        load_approved_selections(
            config,
            as_of=now,
            semantics=semantics,
            dataset_id=dataset.fingerprint,
            matrices={"hourly": dataset.hourly, "daily": dataset.daily},
        )
        if not forced
        else {}
    )
    return predict(
        config,
        selections,
        now=now,
        semantics=semantics,
        force_method=forced,
        dataset_snapshot=dataset,
        approved_release_loaded=bool(selections),
    )


def _cmd_predict(config: Config, args: argparse.Namespace) -> int:
    from filelock import Timeout  # noqa: PLC0415

    from grounded_weather_forecast.evaluation import (  # noqa: PLC0415
        activate_served_document,
    )
    from grounded_weather_forecast.serve.history import append_history  # noqa: PLC0415
    from grounded_weather_forecast.serve.predict import (  # noqa: PLC0415
        NoForecastDataError,
        UnsupportedMethodError,
    )
    from grounded_weather_forecast.storage import (  # noqa: PLC0415
        atomic_write_text,
        output_target,
    )

    try:
        document = _forecast_document(
            config,
            method=args.method,
            semantics_flag=args.semantics,
            now=args.now,
        )
    except (NoForecastDataError, UnsupportedMethodError) as exc:
        print(f"cannot predict: {exc}")
        return 1
    except Timeout:
        print(
            "dataset publication is in progress; try prediction again", file=sys.stderr
        )
        return EXIT_CONTENTION

    if document.status == "degraded" and document.status_reason:
        print(f"degraded: {document.status_reason}", file=sys.stderr)
    payload = document.to_json()
    if args.out == "-":
        print(payload)
    else:
        out = Path(args.out)
        target = output_target(out)
        atomic_write_text(payload, target)
        activate_served_document(config.artifacts_dir, document.release_ids, target)
        print(f"wrote {out}")
    if not args.no_history:
        added = append_history(document, config.predict.history_path)
        print(
            f"appended {added} rows to {config.predict.history_path}",
            file=sys.stderr if args.out == "-" else sys.stdout,
        )
    return 0


def _cmd_publish(config: Config, args: argparse.Namespace) -> int:
    import polars as pl  # noqa: PLC0415
    from filelock import Timeout  # noqa: PLC0415

    from grounded_weather_forecast.evaluation import (  # noqa: PLC0415
        activate_served_document,
    )
    from grounded_weather_forecast.serve.dataset_snapshot import (  # noqa: PLC0415
        DatasetIntegrityError,
    )
    from grounded_weather_forecast.serve.predict import (  # noqa: PLC0415
        NoForecastDataError,
        UnsupportedMethodError,
    )
    from grounded_weather_forecast.serve.publish import (  # noqa: PLC0415
        publish_document,
        record_publish_attempt,
        schedule_recovery,
    )
    from grounded_weather_forecast.storage import output_target  # noqa: PLC0415

    try:
        document = _forecast_document(
            config,
            method="auto",
            semantics_flag=args.semantics,
            now=None,
        )
    except Timeout:
        print(
            "dataset publication is in progress; kept existing output", file=sys.stderr
        )
        return EXIT_CONTENTION
    except (DatasetIntegrityError, pl.exceptions.PolarsError) as exc:
        print(
            f"cannot generate candidate: live dataset unreadable: {exc}",
            file=sys.stderr,
        )
        try:
            schedule_recovery(
                config,
                Path(args.config),
                None,
                reason="live dataset unreadable",
                semantics=args.semantics,
                methods=args.recovery_methods,
                window=args.recovery_window,
            )
        except (OSError, ValueError) as recovery_exc:
            print(f"could not schedule recovery: {recovery_exc}", file=sys.stderr)
        return 1
    except (NoForecastDataError, UnsupportedMethodError, OSError, ValueError) as exc:
        print(f"cannot publish: {exc}", file=sys.stderr)
        return 1

    destination = Path(args.out)
    try:
        outcome = publish_document(
            document,
            destination,
            config.predict.history_path,
        )
    except (OSError, ValueError) as exc:
        print(f"cannot publish: {exc}", file=sys.stderr)
        return 1

    problems: list[str] = []
    if outcome.history_error:
        problems.append(
            f"document published but history failed: {outcome.history_error}"
        )
    for operation in (
        lambda: activate_served_document(
            config.artifacts_dir,
            outcome.served_document.release_ids,
            output_target(destination),
        ),
        lambda: record_publish_attempt(config, document, outcome, destination),
    ):
        try:
            operation()
        except (OSError, ValueError) as exc:
            problems.append(f"publication metadata failed: {exc}")
    scheduled = False
    try:
        scheduled = schedule_recovery(
            config,
            Path(args.config),
            document,
            semantics=args.semantics,
            methods=args.recovery_methods,
            window=args.recovery_window,
        )
    except (OSError, ValueError) as exc:
        problems.append(f"document published but recovery scheduling failed: {exc}")
    print(f"publish: {outcome.action} -> {args.out}")
    if outcome.history_rows:
        print(f"appended {outcome.history_rows} rows to {config.predict.history_path}")
    if scheduled:
        print("scheduled detached recovery", file=sys.stderr)
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


def _run_minutely_backtest(
    config: Config, args: argparse.Namespace
) -> ProductBacktestOutcome:
    """The minutely product: path constructions over the HOURLY live matrix.

    Missing or unusable inputs are failures, rather than insufficient history.
    """
    from grounded_weather_forecast.backtest.minutely import (  # noqa: PLC0415
        MINUTELY_SCORE_VARIABLES,
        MinutelyRequest,
        iter_minutely_scores,
    )
    from grounded_weather_forecast.backtest.scores import (  # noqa: PLC0415
        write_score_batches,
    )
    from grounded_weather_forecast.dataset.matrix import matrix_path  # noqa: PLC0415
    from grounded_weather_forecast.dataset.truth import (  # noqa: PLC0415
        truth_minute_grid,
    )

    if args.source != "live":
        # No synthetic minute truth exists; scoring the nowcast against a
        # synthetic archive would be provenance theater.
        print("minutely: backtest is live-only, skipping")
        return ProductBacktestOutcome("minutely", "not_applicable")
    matrix_file = matrix_path(config.dataset.dir, "hourly", "live")
    truth_file = config.dataset.dir / "truth_minute.parquet"
    for required, hint in (
        (matrix_file, "build-dataset"),
        (truth_file, "build-dataset"),
    ):
        if not required.exists():
            print(f"missing {required}; run {hint} first")
            return ProductBacktestOutcome("minutely", "failed")
    matrix = pl.read_parquet(matrix_file)
    grid = truth_minute_grid(pl.read_parquet(truth_file), MINUTELY_SCORE_VARIABLES)
    if (
        not _valid_product_matrix(matrix, daily=False)
        or matrix["source_kind"].unique().to_list() != ["live"]
        or grid.is_empty()
        or not _has_finite_truth(grid, MINUTELY_SCORE_VARIABLES)
    ):
        print("minutely: no evaluable inputs or truth")
        return ProductBacktestOutcome("minutely", "failed")
    request = MinutelyRequest(window=args.window)
    diagnostics = _backtest_diagnostics(config, "hourly", "live", truth_minute=True)
    result = write_score_batches(
        iter_minutely_scores(matrix, grid, request, config, diagnostics=diagnostics),
        config.dataset.dir / "scores",
        "minutely",
        "live",
        args.window,
    )
    print(f"minutely: {result.row_count} score rows -> {result.path}")
    return _product_outcome("minutely", result.row_count, diagnostics)


def _has_finite_truth(frame: pl.DataFrame, columns: Sequence[str]) -> bool:
    return any(
        column in frame.columns and frame[column].is_finite().any()
        for column in columns
    )


def _product_outcome(
    product: str, rows: int, diagnostics: BacktestDiagnostics
) -> ProductBacktestOutcome:
    outcome = ProductBacktestOutcome.from_scores(product, rows, diagnostics)
    if outcome.status == "pending":
        print(f"{product}: evaluation pending: {outcome.pending_reason}")
    elif outcome.status == "failed":
        print(
            f"{product}: no evaluable scores; expected pending work was not established"
        )
        if diagnostics.snapshot_at is None:
            print(
                "  snapshot metadata unavailable; rebuild the dataset/backfill to establish truth maturity"
            )
    return outcome


def _backtest_diagnostics(
    config: Config, product: str, source: str, *, truth_minute: bool = False
) -> BacktestDiagnostics:
    from grounded_weather_forecast.dataset.evaluation_snapshot import (  # noqa: PLC0415
        evaluation_snapshot,
    )

    return BacktestDiagnostics(
        snapshot_at=evaluation_snapshot(
            config.dataset.dir, product, source, truth_minute=truth_minute
        )
    )


def _applicable_variables(
    args: argparse.Namespace, product: str, methods: tuple[str, ...]
) -> tuple[VariableSpec, ...]:
    from grounded_weather_forecast.backtest.engine import (  # noqa: PLC0415
        variables_from_names,
    )
    from grounded_weather_forecast.blenders.registry import (  # noqa: PLC0415
        supports_product,
    )
    from grounded_weather_forecast.contracts import (  # noqa: PLC0415
        DAILY_VARIABLES,
        Product,
    )

    daily = product == "daily"
    names = _split_csv(args.daily_variables if daily else args.hourly_variables)
    variables = variables_from_names(
        names, DAILY_VARIABLES if daily else HOURLY_VARIABLES
    )
    kind = Product.DAILY if daily else Product.HOURLY
    return tuple(
        variable
        for variable in variables
        if any(supports_product(method, kind, variable) for method in methods)
    )


def _valid_product_matrix(matrix: pl.DataFrame, *, daily: bool) -> bool:
    required = {"issue_time", "source_kind", "lead_bucket"}
    required.update(
        ("forecast_date", "lead_days") if daily else ("valid_time", "lead_hours")
    )
    if matrix.is_empty() or not required <= set(matrix.columns):
        return False
    lead = matrix["lead_days" if daily else "lead_hours"]
    forecasts = [
        column for column in matrix.columns if column.startswith(("fx__", "fxd__"))
    ]
    return (
        all(matrix[column].null_count() == 0 for column in required)
        and isinstance(matrix.schema["issue_time"], pl.Datetime)
        and lead.dtype.is_numeric()
        and lead.is_finite().all()
        and bool(forecasts)
        and all(matrix.schema[column].is_numeric() for column in forecasts)
    )


def _run_matrix_backtest(
    config: Config,
    args: argparse.Namespace,
    product: str,
    methods: tuple[str, ...],
) -> ProductBacktestOutcome:
    """Own one product's matrices and generator until its score file is closed."""
    from grounded_weather_forecast.backtest.engine import (  # noqa: PLC0415
        BacktestRequest,
        iter_backtest_scores,
    )
    from grounded_weather_forecast.backtest.scores import (  # noqa: PLC0415
        write_score_batches,
    )
    from grounded_weather_forecast.dataset.matrix import (  # noqa: PLC0415
        matrix_path,
        truth_column_for,
    )

    daily = product == "daily"
    variables = _applicable_variables(args, product, methods)
    if not variables:
        print(f"{product}: no selected method applies, skipping")
        return ProductBacktestOutcome(product, "not_applicable")
    matrix_file = matrix_path(config.dataset.dir, product, args.source)
    if not matrix_file.exists():
        source_hint = "backfill" if args.source == "synthetic" else "build-dataset"
        print(f"missing {matrix_file}; run {source_hint} first")
        return ProductBacktestOutcome(product, "failed")
    matrix = pl.read_parquet(matrix_file)
    if not _valid_product_matrix(matrix, daily=daily):
        print(f"{product}: matrix is empty or has invalid forecast schema")
        return ProductBacktestOutcome(product, "failed")
    kinds = matrix["source_kind"].unique().to_list()
    if kinds != [args.source]:
        print(f"{product}: matrix carries {kinds}, expected [{args.source}]")
        return ProductBacktestOutcome(product, "failed")
    semantics = _semantics_by_variable(config, args.semantics, variables)
    truth_columns = tuple(
        truth_column_for(variable, semantics[variable.name]) for variable in variables
    )
    if any(column not in matrix.columns for column in truth_columns):
        print(f"{product}: missing truth columns")
        return ProductBacktestOutcome(product, "failed")
    if not _has_finite_truth(matrix, truth_columns):
        print(f"{product}: no finite truth available")
        return ProductBacktestOutcome(product, "failed")
    request = BacktestRequest(
        variables=variables,
        methods=methods,
        window=args.window,
        daily=daily,
        semantics=semantics,
    )
    diagnostics = _backtest_diagnostics(config, product, args.source)
    result = write_score_batches(
        iter_backtest_scores(matrix, request, config, diagnostics=diagnostics),
        config.dataset.dir / "scores",
        product,
        args.source,
        args.window,
    )
    print(f"{product}: {result.row_count} score rows -> {result.path}")
    if not result.row_count and diagnostics.history_too_short:
        _explain_no_folds(config, matrix, product)
    return _product_outcome(product, result.row_count, diagnostics)


def _cmd_backtest(config: Config, args: argparse.Namespace) -> CommandOutcome:
    from grounded_weather_forecast.backtest.score_storage import (  # noqa: PLC0415
        ScoreCoordinationProbeError,
        ScoreCoordinationReleaseError,
        ScoreCoordinationTimeout,
    )

    try:
        return _backtest_products(config, args)
    except (ScoreCoordinationProbeError,) as error:  # noqa: B013
        detail = "; ".join((str(error), *getattr(error, "__notes__", ())))
        print(detail, file=sys.stderr)
        return CommandOutcome(1, "score_coordination_probe", detail)
    except (ScoreCoordinationTimeout,) as error:  # noqa: B013
        print(f"score directory coordination timed out: {error}", file=sys.stderr)
        return CommandOutcome(EXIT_CONTENTION, error_detail=str(error))
    except (ScoreCoordinationReleaseError,) as error:  # noqa: B013
        if error.partial_result is not None:
            partial = error.partial_result
            _print_score_temporaries(partial.deleted, partial.freed_mb)
            for note in partial.skipped:
                print(f"skipped {note}")
        detail = "; ".join((str(error), *getattr(error, "__notes__", ())))
        print(detail, file=sys.stderr)
        return CommandOutcome(1, "score_coordination_release", detail)


def _backtest_products(config: Config, args: argparse.Namespace) -> CommandOutcome:
    from grounded_weather_forecast.backtest.score_storage import (  # noqa: PLC0415
        cleanup_score_temporaries,
    )
    from grounded_weather_forecast.blenders import available_methods  # noqa: PLC0415

    temporary = cleanup_score_temporaries(config.dataset.dir / "scores")
    _print_score_temporaries(temporary.deleted, temporary.freed_mb)
    for note in temporary.skipped:
        print(f"skipped {note}")
    methods = available_methods() if args.methods == "all" else _split_csv(args.methods)
    products = _split_csv(args.products)
    if (
        not methods
        or set(methods) - set(available_methods())
        or not products
        or set(products) - {"hourly", "daily", "minutely"}
    ):
        print("invalid backtest methods or products", file=sys.stderr)
        return CommandOutcome(2)
    try:
        for product in products:
            if product != "minutely":
                _applicable_variables(args, product, methods)
    except ValueError as exc:
        print(f"invalid backtest selection: {exc}", file=sys.stderr)
        return CommandOutcome(2, error_detail=str(exc))
    promotion_semantics = _semantics_by_variable(
        config, args.semantics, HOURLY_VARIABLES
    )
    outcomes: list[ProductBacktestOutcome] = []
    for product in products:
        outcome = _evaluate_product(config, args, product, methods)
        outcomes.append(outcome)
    total = sum(outcome.row_count for outcome in outcomes)
    if total and args.source == "live":
        from grounded_weather_forecast.serve.selection import (  # noqa: PLC0415
            promote_methods,
        )

        promoted = promote_methods(
            config,
            config.dataset.dir / "scores",
            semantics=promotion_semantics,
            products=frozenset(
                outcome.product for outcome in outcomes if outcome.status == "evaluated"
            ),
        )
        if promoted.created:
            print(f"promoted model release: {promoted.release_id}")
        elif promoted.approval_changed:
            print(f"reapproved model release: {promoted.release_id}")
        elif promoted.selections:
            print("retained approved selections; no new model release")
        else:
            print("no eligible model release promoted")
    if failed := next(
        (outcome for outcome in outcomes if outcome.status == "failed"), None
    ):
        return CommandOutcome(failed.exit_code, error_detail=failed.error_detail)
    if any(outcome.status == "pending" for outcome in outcomes):
        return CommandOutcome(EXIT_NO_FOLDS, "no_folds")
    if not total:
        print(
            "no requested product is applicable to this source and method selection",
            file=sys.stderr,
        )
        return CommandOutcome(2)
    return CommandOutcome()


def _evaluate_product(
    config: Config, args: argparse.Namespace, product: str, methods: tuple[str, ...]
) -> ProductBacktestOutcome:
    from grounded_weather_forecast.backtest.score_storage import (  # noqa: PLC0415
        ScoreCoordinationProbeError,
        ScoreCoordinationReleaseError,
        ScoreCoordinationTimeout,
        check_score_coordination,
    )

    try:
        return (
            _run_minutely_backtest(config, args)
            if product == "minutely"
            else _run_matrix_backtest(config, args, product, methods)
        )
    except (ScoreCoordinationReleaseError, ScoreCoordinationTimeout):
        raise  # a failed shared directory guard is not product-local
    except Exception as exc:
        try:
            check_score_coordination(config.dataset.dir / "scores")
        except (ScoreCoordinationReleaseError,) as coordination:  # noqa: B013
            coordination.add_note(
                f"primary {product} failure: {type(exc).__name__}: {exc}"
            )
            for note in getattr(exc, "__notes__", ()):
                coordination.add_note(note)
            raise coordination from exc
        except (OSError,) as probe:  # noqa: B013 -- unknown ownership fails closed
            raise ScoreCoordinationProbeError(
                config.dataset.dir / "scores", exc, probe
            ) from probe
        print(f"{product}: evaluation failed: {exc!r}", file=sys.stderr)
        return ProductBacktestOutcome(
            product, "failed", error_detail=f"{type(exc).__name__}: {exc}"
        )


def _explain_no_folds(config: Config, matrix: pl.DataFrame, product: str) -> None:
    """A young archive produces no folds; say so instead of a bare zero."""
    issues = matrix["issue_time"]
    first, last = issues.min(), issues.max()
    epoch_us = issues.cast(pl.Int64).to_numpy()
    span_days = float(epoch_us.max() - epoch_us.min()) / 86_400_000_000.0
    needed = config.backtest.initial_train_days
    print(
        f"  {product}: no rolling-origin folds. The archive spans {span_days:.1f} "
        f"days of issue times ({first} .. {last}) but the first test origin needs "
        f"more than initial_train_days = {needed}. Keep polling, or backtest "
        f"--source synthetic against an Open-Meteo backfill."
    )


def _cmd_report(config: Config) -> int:
    from grounded_weather_forecast.backtest.scores import load_scores  # noqa: PLC0415
    from grounded_weather_forecast.contracts import (  # noqa: PLC0415
        MixedProvenanceError,
        hourly_variable,
    )
    from grounded_weather_forecast.dashboard import write_dashboard  # noqa: PLC0415
    from grounded_weather_forecast.dashboard.score_summary import (  # noqa: PLC0415
        ScoreSummary,
        summarize_scores,
    )
    from grounded_weather_forecast.dataset.matrix import build_truth  # noqa: PLC0415
    from grounded_weather_forecast.reports import (  # noqa: PLC0415
        evidence,
        operations,
    )
    from grounded_weather_forecast.reports.correlation import (  # noqa: PLC0415
        error_correlation,
    )
    from grounded_weather_forecast.reports.eprocess import (  # noqa: PLC0415
        EProcessStore,
        product_eprocess_store,
        promotion_comparison,
    )
    from grounded_weather_forecast.reports.leaderboard import (  # noqa: PLC0415
        aggregate_leaderboard,
        bh_adjusted,
        blocked_promotions,
        ebh_adjusted,
        leaderboard,
        slice_winners,
        union_references,
    )
    from grounded_weather_forecast.reports.recalibration import (  # noqa: PLC0415
        recalibration_report,
    )
    from grounded_weather_forecast.reports.render import (  # noqa: PLC0415
        print_summary,
        write_markdown_report,
    )
    from grounded_weather_forecast.reports.verification import (  # noqa: PLC0415
        compare_to_backtest,
        verify_history,
    )
    from grounded_weather_forecast.reports.winner_curse import (  # noqa: PLC0415
        winner_curse_adjusted,
    )

    scores_dir = config.dataset.dir / "scores"
    score_files = sorted(scores_dir.glob("scores_*.parquet"))
    summaries: dict[str, ScoreSummary] | None = (
        None if config.promotion.rule == "seq_mcs" else {}
    )
    if not score_files:
        print(f"no scores found in {scores_dir}; run backtest first")
        _report_pipeline_health(config, [], [])
        print(f"wrote {write_dashboard(config, score_summaries=summaries)}")
        return 1
    written: list[Path] = []
    score_read_failures: list[str] = []
    catalog_rows: list[dict[str, object]] = []
    gap_threshold = config.promotion.report_gap_threshold
    references = union_references(config.promotion)
    eprocess_stores: dict[str, EProcessStore] = {}
    for path in score_files:
        try:
            scores = load_scores(path, compact=True)
        except (FileNotFoundError,):  # noqa: B013 -- project convention
            # The pipeline lock keeps prune out of a running report; this
            # guard covers deletion by anything outside that lock's scope.
            print(f"skipping {path.name}: deleted since listing", file=sys.stderr)
            continue
        except (OSError, MixedProvenanceError, pl.exceptions.PolarsError) as error:
            print(f"skipping {path.name}: {error}", file=sys.stderr)
            score_read_failures.append(path.name)
            continue
        catalog_rows.append(operations.evaluation_catalog_row(path, scores))
        raw_board = leaderboard(scores, references=references)
        board = bh_adjusted(raw_board)
        evidence.record_quality(config, scores, board)
        # Serving runs on the live provider set, so its realized skill may only
        # be compared with a leaderboard built from the same provenance — a
        # synthetic board describes different sources entirely. The provenance
        # comes from the filename, since an empty scores frame (a young archive
        # with no folds yet) carries no rows to read it from.
        product = path.stem.split("_")[1]
        filename_kind = path.stem.split("_")[2]
        is_live = (
            filename_kind == "live"
            if scores.is_empty()
            else set(scores["source_kind"].unique()) == {"live"}
        )
        eprocess_store: EProcessStore | None = None
        if is_live:
            eprocess_store = product_eprocess_store(config, product, eprocess_stores)
        live_sections: list[tuple[str, pl.DataFrame]] = []
        if is_live and eprocess_store is not None:
            # Both promotion rules run every report so the operator can watch
            # them disagree before switching [promotion].rule; this is also
            # the single writer that persists the e-process wealth.
            comparison = promotion_comparison(
                board,
                scores,
                alpha=config.promotion.alpha,
                promotion=config.promotion,
                store=eprocess_store,
            )
            eprocess_store.save()
            evidence.record_eprocess_wealth(config, product, eprocess_store)
            # e-BH reads the wealth just banked: the mutually exclusive
            # multiplicity correction beside the DM p-value BH family.
            board = ebh_adjusted(board, eprocess_store, alpha=config.promotion.alpha)
            # Offline A/B of the two mutually exclusive post-hoc quantile
            # repairs; [predict.quantile_recalibration] serves the winner.
            recalib = recalibration_report(scores)
            evidence.record_verdicts(
                config,
                product,
                scores,
                recalib,
                comparison,
                board=board,
                alpha=config.promotion.alpha,
            )
            live_sections = [
                ("Promotion rule comparison (mcs vs seq_mcs)", comparison),
                (
                    "Quantile recalibration (offline holdout: raw vs pit vs cqr)",
                    recalib,
                ),
            ]
        winners = slice_winners(
            board,
            scores=scores,
            rule=config.promotion.rule,
            alpha=config.promotion.alpha,
            promotion=config.promotion,
            eprocess_store=eprocess_store,
        )
        # Two curse-corrected estimates beside every argmin-selected winner
        # mae; report-layer only, same convention as bh/ebh_adjusted.
        winners = winner_curse_adjusted(winners, scores, promotion=config.promotion)
        if summaries is not None:
            summaries[path.stem] = summarize_scores(
                config,
                path.stem,
                scores,
                eprocess_stores,
                board=raw_board,
                winners=winners,
            )
        if is_live:
            evidence.record_winner_curse(config, product, scores, winners)
            evidence.record_nbm_benchmark(config, product, scores, board)
        sections = [
            ("Per-slice leaderboard", board),
            ("Aggregate (n-weighted MAE)", aggregate_leaderboard(board)),
            ("Per-slice winners", winners),
            (
                f"Blocked promotions (served MAE above slice best "
                f"by more than {gap_threshold:.0%})",
                blocked_promotions(winners, gap_threshold),
            ),
            *live_sections,
        ]
        if is_live and config.predict.history_path.exists():
            try:
                minute_truth, hourly_truth, daily_truth = build_truth(config)
                live = verify_history(
                    config.predict.history_path,
                    hourly_truth,
                    minute_truth,
                    daily_truth,
                )
                # Only this product's served rows may face this product's
                # board: the whole-history frame once poisoned the ledger —
                # the first live file in the loop wrote every product's rows
                # with null gaps and first-write-wins dedupe kept them.
                live = live.filter(pl.col("product") == product)
                served = compare_to_backtest(live, board)
                sections.append(("Self-verification (served vs realized)", served))
                evidence.record_served_quality(config, served)
            # PolarsError derives straight from Exception, so a corrupt or
            # schema-incompatible history parquet escaped this guard and took
            # the whole report -- and the dashboard with it -- down. Skipping
            # self-verification is the documented contract here.
            except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
                print(f"self-verification skipped: {exc}")
        report_name = path.stem.replace("scores_", "leaderboard_")
        written.append(
            write_markdown_report(
                config.reports_dir, report_name, report_name, sections
            )
        )
        print_summary(f"winners ({path.stem})", winners)
        blocked = blocked_promotions(winners, gap_threshold)
        if not blocked.is_empty():
            print_summary(f"blocked promotions ({path.stem})", blocked)
        del scores
    matrix_file = _live_hourly_matrix_path(config)
    if matrix_file.exists():
        matrix = pl.read_parquet(matrix_file)
        if not matrix.is_empty():
            from grounded_weather_forecast.reports.drift import (  # noqa: PLC0415
                drift_report,
                write_drift_artifact,
            )

            truth_qc_payload: dict[str, object] | None = None
            truth_qc_path = config.artifacts_dir / "truth_qc.json"
            if truth_qc_path.exists():
                import json  # noqa: PLC0415

                try:
                    truth_qc_payload = json.loads(
                        truth_qc_path.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError):
                    truth_qc_payload = None
            alarms = drift_report(matrix, HOURLY_VARIABLES, truth_qc_payload)
            write_drift_artifact(alarms, config.artifacts_dir / "drift.json")
            if alarms.is_empty():
                print("drift: no alarms")
            else:
                print_summary("drift alarms", alarms)
            written.append(
                write_markdown_report(
                    config.reports_dir,
                    "drift",
                    "Provider drift alarms (consensus + residual tiers)",
                    [("Alarms", alarms)],
                )
            )
            correlation = error_correlation(
                matrix, hourly_variable("temp_c"), TruthSemantics.INSTANTANEOUS
            )
            written.append(
                write_markdown_report(
                    config.reports_dir,
                    "correlation_temp_c",
                    "Provider error correlation (temp_c)",
                    [("Pearson correlation of forecast errors", correlation)],
                )
            )
    _report_evidence_tail(config, written)
    _report_pipeline_health(config, written, catalog_rows)
    written.append(
        write_dashboard(
            config,
            score_summaries=summaries,
            score_read_failures=tuple(score_read_failures),
        )
    )
    for path in written:
        print(f"wrote {path}")
    return 1 if score_read_failures or not catalog_rows else 0


def _report_evidence_tail(config: Config, written: list[Path]) -> None:
    """Release-level history: the churn markdown and the quality delta line."""
    from grounded_weather_forecast.reports import evidence  # noqa: PLC0415
    from grounded_weather_forecast.reports.render import (  # noqa: PLC0415
        print_summary,
        write_markdown_report,
    )

    churn = evidence.record_selection_churn(config)
    if churn.is_empty():
        print("selection churn: fewer than two releases, nothing to compare")
    else:
        changed = churn.filter(pl.col("changed"))
        written.append(
            write_markdown_report(
                config.reports_dir,
                "selection_churn",
                "Selection churn (vs previous release)",
                [("Changed slices", changed), ("All slices", churn)],
            )
        )
        print_summary("selection churn (vs previous release)", changed)
    if (line := evidence.quality_delta_line(config)) is not None:
        print(line)
    if (line := evidence.nbm_benchmark_line(config)) is not None:
        print(line)


def _report_pipeline_health(
    config: Config, written: list[Path], catalog_rows: list[dict[str, object]]
) -> None:
    """Operational evidence: append the edge ledgers and print the alarms."""
    from grounded_weather_forecast import runs  # noqa: PLC0415
    from grounded_weather_forecast.reports import operations  # noqa: PLC0415
    from grounded_weather_forecast.reports.render import (  # noqa: PLC0415
        write_markdown_report,
    )

    health = operations.record_operations(
        config,
        runs_frame=runs.load_runs(runs.runs_path(config)),
        catalog_rows=catalog_rows,
    )
    if health.alarms:
        print(f"PIPELINE ALARMS: {'; '.join(health.alarms)}")
    else:
        print("pipeline health: OK")
    for note in health.contractions:
        print(f"provider contraction: {note}")
    if not health.changes.is_empty():
        for row in health.changes.iter_rows(named=True):
            detail = f" ({row['detail']})" if row["detail"] else ""
            print(
                f"identity change: {row['kind']} "
                f"{row['from_value']} -> {row['to_value']}{detail}"
            )
    written.append(
        write_markdown_report(
            config.reports_dir,
            "pipeline_health",
            "Pipeline health (freshness, providers, build funnel, changes)",
            [
                ("End-to-end freshness (24h window)", health.freshness),
                ("Provider health (24h window)", health.provider_health),
                (
                    "Build funnel (collector -> long -> matrix, trailing 14d;"
                    " lead units: hours for hourly rows, days for daily)",
                    health.funnel,
                ),
                ("Identity changes since last report", health.changes),
                ("Evaluations cataloged this report", health.catalog),
            ],
        )
    )


def _print_score_temporaries(
    paths: tuple[Path, ...], freed_mb: float, *, dry_run: bool = False
) -> None:
    verb = "would delete" if dry_run else "deleted"
    for path in paths:
        print(f"{verb} abandoned temporary {path.name}")
    if paths:
        print(
            f"{verb} {len(paths)} abandoned score temporary files ({freed_mb:.1f} MB)"
        )


def _cmd_prune_scores(config: Config, args: argparse.Namespace) -> int | CommandOutcome:
    from grounded_weather_forecast.reports.operations import (  # noqa: PLC0415
        prune_scores_files,
    )

    result = prune_scores_files(config, dry_run=args.dry_run)
    verb = "would delete" if args.dry_run else "deleted"
    for path in result.deleted:
        print(f"{verb} {path.name}")
    for note in result.skipped:
        print(f"skipped {note}")
    print(
        f"{verb} {len(result.deleted)} scores files ({result.freed_mb:.1f} MB), "
        f"kept {len(result.kept)}"
    )
    _print_score_temporaries(
        result.temporary_deleted, result.temporary_freed_mb, dry_run=args.dry_run
    )
    if result.coordination_release_failure is not None:
        detail = result.coordination_release_failure.diagnostic()
        print(detail, file=sys.stderr)
        return CommandOutcome(1, "score_coordination_release", detail)
    return 0


def _run_pipeline_steps(
    config: Config,
    args: argparse.Namespace,
    steps: tuple[tuple[str, Callable[[], int | CommandOutcome]], ...],
    *,
    backtest_args: argparse.Namespace | None = None,
    continue_on_failure: frozenset[str] = frozenset(),
    stage_args: Mapping[str, argparse.Namespace] | None = None,
    tolerate_pending: bool = False,
) -> CommandOutcome:
    first_failure: CommandOutcome | None = None
    pending = False
    for name, step in steps:
        fallback = (
            backtest_args
            if name.startswith("backtest") and backtest_args is not None
            else args
        )
        record_args = (stage_args or {}).get(name, fallback)
        print(f"pipeline step: {name}")
        started_at = datetime.now(tz=UTC)
        result: CommandOutcome | None = None
        try:
            result = CommandOutcome.normalize(step())
        finally:
            stage = argparse.Namespace(
                **{
                    key: value
                    for key, value in vars(record_args).items()
                    if not key.startswith("_")
                }
            )
            stage.command = name.split()[0]
            _record_run(
                config,
                stage,
                started_at=started_at,
                exit_code=result.exit_code if result is not None else None,
                record_kind="stage",
                parent_run_id=_parent_run_id(args),
                failure_kind=result.failure_kind if result is not None else None,
                error_detail=result.error_detail if result is not None else None,
            )
        if (
            tolerate_pending
            and name.startswith("backtest")
            and result.exit_code == EXIT_NO_FOLDS
            and result.failure_kind == "no_folds"
        ):
            pending = True
            print(f"pipeline pending: {name}; continuing diagnostics")
            continue
        if result.exit_code:
            first_failure = first_failure or result
            if name in continue_on_failure:
                print(
                    f"pipeline continuing: {name} exited {result.exit_code}; remaining outputs will run, but the pipeline will return failure",
                    file=sys.stderr,
                )
                continue
            print(
                f"pipeline stopped: {name} exited {result.exit_code}", file=sys.stderr
            )
            return first_failure
    return first_failure or CommandOutcome(0, "no_folds" if pending else None)


def _cmd_maintain(config: Config, args: argparse.Namespace) -> CommandOutcome:
    truth_args = argparse.Namespace(days=args.truth_qc_days)
    return _run_pipeline_steps(
        config,
        args,
        (
            ("build-dataset", lambda: _cmd_build_dataset(config)),
            ("backtest --source live", lambda: _cmd_backtest(config, args)),
            ("truth-qc", lambda: _cmd_truth_qc(config, truth_args)),
            ("report", lambda: _cmd_report(config)),
        ),
        continue_on_failure=frozenset({"truth-qc"}),
        stage_args={"truth-qc": truth_args},
        tolerate_pending=True,
    )


def _recovery_backtest_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        methods=args.methods,
        hourly_variables=HOURLY_DEFAULT_VARIABLES,
        daily_variables=DAILY_DEFAULT_VARIABLES,
        products="hourly,daily,minutely",
        window=args.window,
        source="live",
        semantics=args.semantics,
    )


def _recovery_needed(config: Config, semantics: str = "auto") -> bool:
    """Re-run serving; valid selections can still yield no served release id."""
    from grounded_weather_forecast.serve.dataset_snapshot import (  # noqa: PLC0415
        DatasetIntegrityError,
    )
    from grounded_weather_forecast.serve.predict import (  # noqa: PLC0415
        NoForecastDataError,
        UnsupportedMethodError,
    )

    try:
        document = _forecast_document(
            config,
            method="auto",
            semantics_flag=semantics,
            now=None,
            lock_timeout=-1,
        )
    except (
        NoForecastDataError,
        UnsupportedMethodError,
        DatasetIntegrityError,
        pl.exceptions.PolarsError,
        OSError,
        ValueError,
    ):
        return True
    return document.status == "degraded"


def _cmd_recover(config: Config, args: argparse.Namespace) -> int | CommandOutcome:
    from filelock import FileLock, Timeout  # noqa: PLC0415

    from grounded_weather_forecast.storage import pipeline_lock  # noqa: PLC0415

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    try:
        with (
            FileLock(config.artifacts_dir / "auto-restore.lock", timeout=0),
            pipeline_lock(config.dataset.dir, timeout=-1),
        ):
            if not _recovery_needed(config, args.semantics):
                print("recovery no longer needed")
                return 0
            backtest_args = _recovery_backtest_args(args)
            return _run_pipeline_steps(
                config,
                args,
                (
                    ("build-dataset", lambda: _cmd_build_dataset(config)),
                    (
                        "backtest --source live",
                        lambda: _cmd_backtest(config, backtest_args),
                    ),
                    ("report", lambda: _cmd_report(config)),
                ),
                backtest_args=backtest_args,
            )
    except Timeout:
        print("another automatic recovery is already running", file=sys.stderr)
        return EXIT_CONTENTION


# Commands that mutate or bulk-read the scores directory and its artifacts.
# `predict` is excluded: serving must never block behind an hour-long report.
# `ingest-ensembles` is excluded: its store has its own sidecar lock and
# atomic replace, and the coarse lock would make its 6-hourly launchd fire
# lose an API snapshot whenever the maintain chain is running.
_LOCKED_COMMANDS = frozenset(
    {
        "build-dataset",
        "backtest",
        "maintain",
        "report",
        "alignment",
        "backfill",
        "truth-qc",
        "prune-scores",
    }
)
# EX_TEMPFAIL: lets the runs ledger and scheduler scripts tell lock
# contention apart from a genuine command failure (1) or config error (2).
EXIT_CONTENTION = 75


def _dispatch(
    config: Config, args: argparse.Namespace, parser: argparse.ArgumentParser
) -> int | CommandOutcome:
    if args.command not in _LOCKED_COMMANDS:
        return _run_command(config, args, parser)
    from filelock import Timeout  # noqa: PLC0415

    from grounded_weather_forecast.storage import pipeline_lock  # noqa: PLC0415

    try:
        timeout = -1 if args.command == "maintain" else None
        with pipeline_lock(config.dataset.dir, timeout=timeout):
            return _run_command(config, args, parser)
    except Timeout:
        print(
            "another pipeline command is running "
            f"(lock: {config.dataset.dir / 'pipeline.lock'}); "
            "try again once it finishes",
            file=sys.stderr,
        )
        return EXIT_CONTENTION


def _run_command(
    config: Config, args: argparse.Namespace, parser: argparse.ArgumentParser
) -> int | CommandOutcome:
    match args.command:
        case "qc":
            return _cmd_qc(config)
        case "build-dataset":
            return _cmd_build_dataset(config)
        case "backtest":
            return _cmd_backtest(config, args)
        case "maintain":
            return _cmd_maintain(config, args)
        case "report":
            return _cmd_report(config)
        case "alignment":
            return _cmd_alignment(config)
        case "backfill":
            return _cmd_backfill(config, args)
        case "ingest-ensembles":
            return _cmd_ingest_ensembles(config, args)
        case "truth-qc":
            return _cmd_truth_qc(config, args)
        case "predict":
            return _cmd_predict(config, args)
        case "publish":
            return _cmd_publish(config, args)
        case "recover":
            return _cmd_recover(config, args)
        case "prune-scores":
            return _cmd_prune_scores(config, args)
        case _:  # pragma: no cover - argparse enforces the choices
            parser.print_help()
            return 2


def _args_summary(args: argparse.Namespace) -> str:
    import json  # noqa: PLC0415

    summary = {
        key: str(value) if isinstance(value, (Path, date, datetime)) else value
        for key, value in vars(args).items()
        if key not in {"command", "config"}
        and not key.startswith("_")
        and value is not None
    }
    return json.dumps(summary, sort_keys=True)


def _parent_run_id(args: argparse.Namespace) -> str | None:
    started_at = getattr(args, "_run_started_at", None)
    if started_at is None:
        return None
    from grounded_weather_forecast.runs import run_id_for  # noqa: PLC0415

    return run_id_for(args.command, started_at)


def _record_run(
    config: Config,
    args: argparse.Namespace,
    *,
    started_at: datetime,
    exit_code: int | None,
    record_kind: str = "invocation",
    parent_run_id: str | None = None,
    failure_kind: str | None = None,
    error_detail: str | None = None,
) -> None:
    """Append this invocation to the run ledger; never raises."""
    try:
        # Inside the try: this runs from a `finally`, so an import failure
        # escaping here would replace the real command's result or exception.
        from grounded_weather_forecast import runs  # noqa: PLC0415
        from grounded_weather_forecast.evaluation import (  # noqa: PLC0415
            config_fingerprint,
            dataset_fingerprint,
        )

        error = sys.exc_info()[1]
        try:
            dataset_print = dataset_fingerprint(config)
        except (OSError, ValueError):
            dataset_print = "unknown"
        try:
            config_print = config_fingerprint(config)
        except (OSError, ValueError):
            config_print = "unknown"
        record = runs.RunRecord(
            run_id=runs.run_id_for(args.command, started_at),
            command=args.command,
            args_json=_args_summary(args),
            started_at=started_at,
            ended_at=datetime.now(tz=UTC),
            exit_code=exit_code,
            error=type(error).__name__ if error is not None else None,
            dataset_fingerprint=dataset_print,
            config_fingerprint=config_print,
            code_version=__version__,
            record_kind=record_kind,
            parent_run_id=parent_run_id,
            failure_kind=failure_kind,
            error_detail=(str(error) if error is not None else error_detail),
        )
        runs.append_run(record, runs.runs_path(config))
    except (Exception,):  # noqa: B013 - project style requires tuple clauses
        return


def main(argv: Sequence[str] | None = None) -> int:
    started_at = datetime.now(tz=UTC)
    parser = build_parser()
    args = parser.parse_args(argv)
    vars(args)["_run_started_at"] = started_at
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}")
        return 2
    outcome: CommandOutcome | None = None
    try:
        outcome = CommandOutcome.normalize(_dispatch(config, args, parser))
        return outcome.exit_code
    finally:
        _record_run(
            config,
            args,
            started_at=started_at,
            exit_code=outcome.exit_code if outcome is not None else None,
            failure_kind=outcome.failure_kind if outcome is not None else None,
            error_detail=outcome.error_detail if outcome is not None else None,
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
