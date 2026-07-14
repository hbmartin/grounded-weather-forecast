"""Command-line interface: thin wrappers over library functions."""

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from omni_forecast import __version__
from omni_forecast.config import Config, ConfigError, load_config
from omni_forecast.contracts import HOURLY_VARIABLES, TruthSemantics

HOURLY_DEFAULT_VARIABLES = (
    "temp_c,humidity_pct,dew_point_c,wind_speed_ms,wind_gust_ms,"
    "pressure_sea_hpa,precip_mm,pop"
)
DAILY_DEFAULT_VARIABLES = "temp_max_c,temp_min_c,pop,precip_sum_mm"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omni-forecast",
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
        help="fetch Open-Meteo Previous Runs into a synthetic supervised matrix",
    )
    backfill.add_argument(
        "--models",
        default="",
        help="comma-separated Open-Meteo models (default: [backfill.open_meteo].models)",
    )
    backfill.add_argument(
        "--end",
        type=date.fromisoformat,
        default=None,
        help="last valid date to fetch, YYYY-MM-DD (default: yesterday)",
    )
    backfill.add_argument(
        "--chunk-days",
        type=int,
        default=90,
        help="days per request (default: 90)",
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
    from omni_forecast.dataset.qc import apply_qc, qc_summary  # noqa: PLC0415
    from omni_forecast.dataset.station import read_observations  # noqa: PLC0415
    from omni_forecast.dataset.truth import (  # noqa: PLC0415
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
    from omni_forecast.dataset.matrix import write_dataset  # noqa: PLC0415

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
    from omni_forecast.dataset.alignment import load_recommended  # noqa: PLC0415

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
    from omni_forecast.dataset.matrix import matrix_path  # noqa: PLC0415

    return matrix_path(config.dataset.dir, "hourly", "live")


def _cmd_alignment(config: Config) -> int:
    from omni_forecast.dataset.alignment import (  # noqa: PLC0415
        alignment_study,
        write_alignment,
    )
    from omni_forecast.reports.render import (  # noqa: PLC0415
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
    print(f"wrote {_alignment_artifact_path(config)}")
    return 0


def _cmd_backfill(config: Config, args: argparse.Namespace) -> int:
    from omni_forecast.dataset.backfill import (  # noqa: PLC0415
        BackfillError,
        backfill_long,
    )
    from omni_forecast.dataset.matrix import write_synthetic_matrix  # noqa: PLC0415

    end = args.end or (datetime.now(tz=UTC).date() - timedelta(days=1))
    models = _split_csv(args.models)
    try:
        long_frame = backfill_long(
            config, end, models=models or None, chunk_days=args.chunk_days
        )
    except (BackfillError, OSError) as exc:
        print(f"backfill failed: {exc}")
        return 1
    path, rows = write_synthetic_matrix(config, long_frame)
    print(f"backfilled {long_frame.height} forecast points")
    print(f"sources: {', '.join(sorted(long_frame['source'].unique().to_list()))}")
    print(f"synthetic matrix: {rows} rows -> {path}")
    return 0


def _cmd_predict(config: Config, args: argparse.Namespace) -> int:
    from omni_forecast.serve.history import append_history  # noqa: PLC0415
    from omni_forecast.serve.predict import (  # noqa: PLC0415
        NoForecastDataError,
        predict,
    )
    from omni_forecast.serve.selection import select_methods  # noqa: PLC0415

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
    except NoForecastDataError as exc:
        print(f"cannot predict: {exc}")
        return 1

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
    from omni_forecast.backtest.engine import (  # noqa: PLC0415
        BacktestRequest,
        run_backtest,
        variables_from_names,
    )
    from omni_forecast.backtest.scores import scores_path, write_scores  # noqa: PLC0415
    from omni_forecast.blenders import available_methods  # noqa: PLC0415
    from omni_forecast.contracts import (  # noqa: PLC0415
        DAILY_VARIABLES,
        HOURLY_VARIABLES,
    )
    from omni_forecast.dataset.matrix import matrix_path  # noqa: PLC0415

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
        from omni_forecast.serve.selection import select_methods  # noqa: PLC0415

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
    from omni_forecast.backtest.scores import load_scores  # noqa: PLC0415
    from omni_forecast.contracts import hourly_variable  # noqa: PLC0415
    from omni_forecast.dataset.matrix import build_truth  # noqa: PLC0415
    from omni_forecast.reports.correlation import error_correlation  # noqa: PLC0415
    from omni_forecast.reports.leaderboard import (  # noqa: PLC0415
        aggregate_leaderboard,
        leaderboard,
        slice_winners,
    )
    from omni_forecast.reports.render import (  # noqa: PLC0415
        print_summary,
        write_markdown_report,
    )
    from omni_forecast.reports.verification import (  # noqa: PLC0415
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
            ("Per-slice winners", slice_winners(board)),
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
        print_summary(f"winners ({path.stem})", slice_winners(board))
    matrix_file = _live_hourly_matrix_path(config)
    if matrix_file.exists():
        matrix = pl.read_parquet(matrix_file)
        if not matrix.is_empty():
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
        case "predict":
            return _cmd_predict(config, args)
        case _:  # pragma: no cover - argparse enforces the choices
            parser.print_help()
            return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
