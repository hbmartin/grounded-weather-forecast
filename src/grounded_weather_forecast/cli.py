"""Command-line interface: thin wrappers over library functions."""

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from grounded_weather_forecast import __version__
from grounded_weather_forecast.config import Config, ConfigError, load_config
from grounded_weather_forecast.contracts import HOURLY_VARIABLES, TruthSemantics

HOURLY_DEFAULT_VARIABLES = (
    "temp_c,humidity_pct,dew_point_c,wind_speed_ms,wind_gust_ms,"
    "pressure_sea_hpa,precip_mm,pop"
)
DAILY_DEFAULT_VARIABLES = "temp_max_c,temp_min_c,pop,precip_sum_mm"


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
    backtest.add_argument(
        "--methods",
        default="all",
        help="comma-separated method ids, or 'all' registered (default)",
    )
    backtest.add_argument(
        "--hourly-variables",
        default=HOURLY_DEFAULT_VARIABLES,
        help="comma-separated hourly variables",
    )
    backtest.add_argument(
        "--daily-variables",
        default=DAILY_DEFAULT_VARIABLES,
        help="comma-separated daily variables",
    )
    backtest.add_argument(
        "--products",
        default="hourly,daily",
        help="comma-separated products to backtest (hourly,daily)",
    )
    backtest.add_argument(
        "--window",
        choices=("expanding", "rolling"),
        default="expanding",
        help="training window mode",
    )
    backtest.add_argument(
        "--source",
        choices=("live", "synthetic"),
        default="live",
        help="which provenance the matrices must carry",
    )
    backtest.add_argument(
        "--semantics",
        choices=("auto", "inst", "mean"),
        default="auto",
        help="hourly truth semantics: inst, mean, or auto (alignment artifact"
        " majority recommendation, falling back to inst)",
    )
    subparsers.add_parser(
        "report", help="render leaderboards and correlation reports from scores"
    )
    subparsers.add_parser(
        "alignment",
        help="study truth semantics per provider; write alignment artifact",
    )
    backfill = subparsers.add_parser(
        "backfill",
        help="fetch archived forecasts into the synthetic supervised matrix",
    )
    backfill.add_argument(
        "--provider",
        choices=("open_meteo", "dynamical"),
        default="open_meteo",
        help="open_meteo: Previous Runs (24h-multiple leads); dynamical:"
        " dynamical.org full cycles at native steps (sub-24h leads; needs"
        " the 'backfill' optional extra)",
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
    config: Config, flag: str, variables: Sequence[str]
) -> dict[str, TruthSemantics]:
    return {
        variable: _resolve_semantics(config, flag, variable) for variable in variables
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


def _cmd_truth_qc(config: Config, args: argparse.Namespace) -> int:
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
        return 1
    _minute, hourly_truth, _daily = build_truth(config)
    try:
        checks = fetch_neighbor_checks(config, hourly_truth, hours=args.days * 24)
    except (NeighborError, OSError, ValueError) as exc:
        print(f"truth-qc failed: {exc}")
        return 1
    artifact: dict[str, object] = {
        "schema_version": 2,
        "history_days": args.days,
        "n_neighbors": checks.n_neighbors,
        "overlap_hours": checks.overlap_hours,
        "drift_alert": checks.drift_alert,
        "drift_reason": checks.drift_reason,
        "correlation_alert": checks.correlation_alert,
        "correlation_reason": checks.correlation_reason,
        "daily_drift": checks.daily_drift.to_dicts(),
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
    print(f"shield: {shield_note}")
    print(f"wrote {config.artifacts_dir / 'truth_qc.json'}")
    evaluable = (
        checks.drift_alert is not None,
        checks.correlation_alert is not None,
        artifact["shield_alert"] is not None,
    )
    return 0 if any(evaluable) else 2


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
    except (EnsembleError, OSError, ValueError) as exc:
        print(f"ensemble ingest failed: {exc}")
        return 1
    path = ensembles_path(config)
    new_rows, total_rows = append_ensembles(path, fresh)
    models_seen = ", ".join(sorted(fresh["model"].unique().to_list()))
    print(f"ingested {fresh.height} statistic rows from {models_seen}")
    print(f"ensembles store: +{new_rows} new rows, {total_rows} total -> {path}")
    return 0


def _cmd_predict(config: Config, args: argparse.Namespace) -> int:
    from grounded_weather_forecast.serve.history import append_history  # noqa: PLC0415
    from grounded_weather_forecast.serve.predict import (  # noqa: PLC0415
        NoForecastDataError,
        UnsupportedMethodError,
        predict,
    )
    from grounded_weather_forecast.serve.selection import (  # noqa: PLC0415
        select_methods,
    )

    forced = None if args.method == "auto" else args.method
    now = args.now
    if now is not None and now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    selections = (
        select_methods(config, config.dataset.dir / "scores", as_of=now)
        if not forced
        else {}
    )
    try:
        document = predict(
            config,
            selections,
            now=now,
            semantics=_semantics_by_variable(
                config,
                args.semantics,
                tuple(variable.name for variable in HOURLY_VARIABLES),
            ),
            force_method=forced,
        )
    except (NoForecastDataError, UnsupportedMethodError) as exc:
        print(f"cannot predict: {exc}")
        return 1

    if document.status == "degraded" and document.status_reason:
        print(f"degraded: {document.status_reason}", file=sys.stderr)
    payload = document.to_json()
    if args.out == "-":
        print(payload)
    else:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        print(f"wrote {out}")
    if not args.no_history:
        added = append_history(document, config.predict.history_path)
        print(
            f"appended {added} rows to {config.predict.history_path}",
            file=sys.stderr if args.out == "-" else sys.stdout,
        )
    return 0


def _cmd_backtest(config: Config, args: argparse.Namespace) -> int:
    from grounded_weather_forecast.backtest.engine import (  # noqa: PLC0415
        BacktestRequest,
        run_backtest,
        variables_from_names,
    )
    from grounded_weather_forecast.backtest.scores import (  # noqa: PLC0415
        scores_path,
        write_scores,
    )
    from grounded_weather_forecast.blenders import available_methods  # noqa: PLC0415
    from grounded_weather_forecast.contracts import (  # noqa: PLC0415
        DAILY_VARIABLES,
        HOURLY_VARIABLES,
    )
    from grounded_weather_forecast.dataset.matrix import matrix_path  # noqa: PLC0415

    methods = available_methods() if args.methods == "all" else _split_csv(args.methods)
    products = _split_csv(args.products)
    total = 0
    for product in products:
        daily = product == "daily"
        matrix_file = matrix_path(config.dataset.dir, product, args.source)
        if not matrix_file.exists():
            source_hint = "backfill" if args.source == "synthetic" else "build-dataset"
            print(f"missing {matrix_file}; run {source_hint} first")
            return 1
        matrix = pl.read_parquet(matrix_file)
        if matrix.is_empty():
            print(f"{product}: matrix is empty, skipping")
            continue
        kinds = matrix["source_kind"].unique().to_list()
        if kinds != [args.source]:
            print(f"{product}: matrix carries {kinds}, expected [{args.source}]")
            return 1
        names = _split_csv(args.daily_variables if daily else args.hourly_variables)
        semantics = _semantics_by_variable(config, args.semantics, names)
        request = BacktestRequest(
            variables=variables_from_names(
                names, DAILY_VARIABLES if daily else HOURLY_VARIABLES
            ),
            methods=methods,
            window=args.window,
            daily=daily,
            semantics=semantics,
        )
        scores = run_backtest(matrix, request, config)
        evaluation_id = (
            str(scores["evaluation_id"][0]) if not scores.is_empty() else None
        )
        path = scores_path(
            config.dataset.dir / "scores",
            product,
            args.source,
            args.window,
            evaluation_id,
        )
        write_scores(scores, path)
        print(f"{product}: {scores.height} score rows -> {path}")
        if scores.is_empty():
            _explain_no_folds(config, matrix, product)
        total += scores.height
    if total and args.source == "live":
        from grounded_weather_forecast.serve.selection import (  # noqa: PLC0415
            select_methods,
        )

        promoted = select_methods(config, config.dataset.dir / "scores")
        release_ids = sorted(
            {
                choice.release_id
                for choice in promoted.values()
                if choice.release_id is not None
            }
        )
        if release_ids:
            print(f"promoted model release: {', '.join(release_ids)}")
    return 0 if total else 1


def _explain_no_folds(config: Config, matrix: pl.DataFrame, product: str) -> None:
    """A young archive produces no folds; say so instead of a bare zero."""
    issues = matrix["issue_time"]
    first, last = issues.min(), issues.max()
    epoch_us = issues.cast(pl.Int64).to_numpy()
    span_days = float(epoch_us.max() - epoch_us.min()) / 86_400_000_000.0
    needed = config.backtest.initial_train_days + config.backtest.step_days
    print(
        f"  {product}: no rolling-origin folds. The archive spans {span_days:.1f} "
        f"days of issue times ({first} .. {last}) but a fold needs "
        f"initial_train_days + step_days = {needed}. Keep polling, or backtest "
        f"--source synthetic against an Open-Meteo backfill."
    )


def _cmd_report(config: Config) -> int:
    from grounded_weather_forecast.backtest.scores import load_scores  # noqa: PLC0415
    from grounded_weather_forecast.contracts import hourly_variable  # noqa: PLC0415
    from grounded_weather_forecast.dataset.matrix import build_truth  # noqa: PLC0415
    from grounded_weather_forecast.reports.correlation import (  # noqa: PLC0415
        error_correlation,
    )
    from grounded_weather_forecast.reports.leaderboard import (  # noqa: PLC0415
        aggregate_leaderboard,
        leaderboard,
        slice_winners,
    )
    from grounded_weather_forecast.reports.render import (  # noqa: PLC0415
        print_summary,
        write_markdown_report,
    )
    from grounded_weather_forecast.reports.verification import (  # noqa: PLC0415
        compare_to_backtest,
        verify_history,
    )

    scores_dir = config.dataset.dir / "scores"
    score_files = sorted(scores_dir.glob("scores_*.parquet"))
    if not score_files:
        print(f"no scores found in {scores_dir}; run backtest first")
        return 1
    written: list[Path] = []
    for path in score_files:
        scores = load_scores(path)
        board = leaderboard(scores)
        sections = [
            ("Per-slice leaderboard", board),
            ("Aggregate (n-weighted MAE)", aggregate_leaderboard(board)),
            (
                "Per-slice winners",
                slice_winners(
                    board,
                    scores=scores,
                    rule=config.promotion.rule,
                    alpha=config.promotion.alpha,
                ),
            ),
        ]
        # Serving runs on the live provider set, so its realized skill may only
        # be compared with a leaderboard built from the same provenance — a
        # synthetic board describes different sources entirely. The provenance
        # comes from the filename, since an empty scores frame (a young archive
        # with no folds yet) carries no rows to read it from.
        filename_kind = path.stem.split("_")[2]
        is_live = (
            filename_kind == "live"
            if scores.is_empty()
            else set(scores["source_kind"].unique()) == {"live"}
        )
        if is_live and config.predict.history_path.exists():
            minute_truth, hourly_truth, daily_truth = build_truth(config)
            live = verify_history(
                config.predict.history_path,
                hourly_truth,
                minute_truth,
                daily_truth,
            )
            sections.append(
                (
                    "Self-verification (served vs realized)",
                    compare_to_backtest(live, board),
                )
            )
        report_name = path.stem.replace("scores_", "leaderboard_")
        written.append(
            write_markdown_report(
                config.reports_dir, report_name, report_name, sections
            )
        )
        print_summary(
            f"winners ({path.stem})",
            slice_winners(
                board,
                scores=scores,
                rule=config.promotion.rule,
                alpha=config.promotion.alpha,
            ),
        )
    matrix_file = _live_hourly_matrix_path(config)
    if matrix_file.exists():
        matrix = pl.read_parquet(matrix_file)
        if not matrix.is_empty():
            from grounded_weather_forecast.reports.drift import (  # noqa: PLC0415
                drift_report,
                write_drift_artifact,
            )

            alarms = drift_report(matrix, HOURLY_VARIABLES)
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
    for path in written:
        print(f"wrote {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}")
        return 2
    match args.command:
        case "qc":
            return _cmd_qc(config)
        case "build-dataset":
            return _cmd_build_dataset(config)
        case "backtest":
            return _cmd_backtest(config, args)
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
        case _:  # pragma: no cover - argparse enforces the choices
            parser.print_help()
            return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
