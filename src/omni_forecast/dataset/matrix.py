"""Materialize the wide training matrices and convert slices to contracts.

The hourly matrix has one row per (issue snapshot, valid hour) with per-source
forecast columns (``fx__{source}__{var}``), per-source ages, issue-time station
observations (leakage-safe past data), calendar features, and BOTH truth
semantics. The daily matrix is keyed by (issue snapshot, target local date)
with native daily forecasts plus deterministic raw-equal-weight hourly
aggregates (``ewagg__*``).
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from omni_forecast.config import Config
from omni_forecast.contracts import (
    ForecastMatrix,
    MixedProvenanceError,
    Product,
    SourceKind,
    SupervisedSlice,
    TruthSemantics,
    VariableSpec,
    age_col,
    fx_col,
    fxd_col,
    obs_col,
    parse_fx_col,
    truth_col,
)
from omni_forecast.dataset.provider_qc import apply_provider_qc
from omni_forecast.dataset.providers import (
    DAILY_COLUMN_MAP,
    HOURLY_COLUMN_MAP,
    read_forecast_archive,
)
from omni_forecast.dataset.qc import apply_causal_qc, apply_qc
from omni_forecast.dataset.snapshots import snapshot_long, snapshot_times
from omni_forecast.dataset.station import read_observations
from omni_forecast.dataset.truth import (
    STATE_VARIABLES,
    truth_daily,
    truth_hourly,
    truth_minute,
)
from omni_forecast.leads import daily_bucket_expr, hourly_bucket_expr
from omni_forecast.timeutil import local_date_expr, local_day_minutes

_SECONDS_PER_HOUR = 3600.0
_OBS_TOLERANCE = timedelta(minutes=30)
HOURLY_MATRIX_VARIABLES: tuple[str, ...] = tuple(HOURLY_COLUMN_MAP.values())
DAILY_MATRIX_VARIABLES: tuple[str, ...] = tuple(
    dict.fromkeys(DAILY_COLUMN_MAP.values())
)
OBS_VARIABLES: tuple[str, ...] = (*STATE_VARIABLES, "wind_gust_ms")


def assert_single_kind(frame: pl.DataFrame, *, allow_mixed: bool = False) -> str:
    """Enforce the live/synthetic provenance wall; returns the single kind."""
    kinds = frame["source_kind"].unique().to_list() if frame.height else []
    match kinds:
        case []:
            return SourceKind.LIVE.value
        case [kind]:
            return str(kind)
        case _ if allow_mixed:
            return "mixed"
        case _:
            msg = f"frame mixes source kinds {sorted(map(str, kinds))}; pass allow_mixed=True only if deliberate"
            raise MixedProvenanceError(msg)


def _pivot(
    snap_long: pl.DataFrame,
    index: list[str],
    variable: str,
    column_builder: Callable[[str, str], str],
) -> pl.DataFrame:
    wide = snap_long.pivot(
        on="source", index=index, values=variable, aggregate_function="last"
    )
    return wide.rename(
        {
            source: column_builder(source, variable)
            for source in wide.columns
            if source not in index
        }
    )


def build_hourly_matrix(
    hourly_long: pl.DataFrame,
    snapshots: pl.DataFrame,
    truth_hourly_frame: pl.DataFrame,
    truth_minute_frame: pl.DataFrame,
    config: Config,
) -> pl.DataFrame:
    """One row per (issue snapshot, valid hour), sources wide, truth joined."""
    kind = assert_single_kind(hourly_long)
    # Provider QC runs AFTER the as-of snapshot join, grouped per snapshot, so the
    # cross-source outlier test never compares different historical vintages of the
    # same valid time (which would flag a genuine, freshly-forecast weather shift).
    snap = apply_provider_qc(
        snapshot_long(hourly_long, snapshots, config.forecasts.max_forecast_age_hours),
        config,
        value_columns=HOURLY_MATRIX_VARIABLES,
        group_key=["issue_time", "valid_time"],
    ).with_columns(
        (
            (pl.col("valid_time") - pl.col("issue_time")).dt.total_seconds()
            / _SECONDS_PER_HOUR
        ).alias("lead_hours")
    )
    # A stable sort fixes pivot column order, keeping parquet bytes and the
    # dataset fingerprint deterministic across rebuilds.
    snap = snap.filter(pl.col("lead_hours") >= 0.0).sort(
        "source", "issue_time", "valid_time"
    )
    if snap.is_empty():
        return _empty_hourly_matrix()

    index = ["issue_time", "valid_time"]
    wide = snap.select(index).unique(maintain_order=True).sort(index)
    for variable in HOURLY_MATRIX_VARIABLES:
        wide = wide.join(_pivot(snap, index, variable, fx_col), on=index, how="left")
    ages = (
        snap.select(
            "issue_time",
            "source",
            (
                (pl.col("issue_time") - pl.col("fetched_at")).dt.total_seconds()
                / _SECONDS_PER_HOUR
            ).alias("age_hours"),
        )
        .unique(maintain_order=True)
        .pivot(on="source", index=["issue_time"], values="age_hours")
    )
    ages = ages.rename({c: age_col(c) for c in ages.columns if c != "issue_time"})
    observations = truth_minute_frame.sort("ts").select(
        "ts", *(pl.col(v).alias(obs_col(v)) for v in OBS_VARIABLES)
    )
    timezone_name = config.station.timezone
    return (
        wide.join(ages, on="issue_time", how="left")
        .sort("issue_time")
        .join_asof(
            observations,
            left_on="issue_time",
            right_on="ts",
            strategy="backward",
            tolerance=_OBS_TOLERANCE,
        )
        .drop("ts")
        .join(
            truth_hourly_frame.rename({"valid_hour": "valid_time"}),
            on="valid_time",
            how="left",
        )
        .with_columns(
            (
                (pl.col("valid_time") - pl.col("issue_time")).dt.total_seconds()
                / _SECONDS_PER_HOUR
            ).alias("lead_hours"),
            pl.lit(kind).alias("source_kind"),
            pl.col("valid_time")
            .dt.convert_time_zone(timezone_name)
            .dt.hour()
            .cast(pl.Int8)
            .alias("valid_hour_local"),
            pl.col("valid_time")
            .dt.convert_time_zone(timezone_name)
            .dt.month()
            .cast(pl.Int8)
            .alias("valid_month"),
        )
        .with_columns(hourly_bucket_expr(pl.col("lead_hours")).alias("lead_bucket"))
        .sort("issue_time", "valid_time")
    )


def _empty_hourly_matrix() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "issue_time": pl.Datetime("us", "UTC"),
            "valid_time": pl.Datetime("us", "UTC"),
            "lead_hours": pl.Float64(),
            "lead_bucket": pl.String(),
            "source_kind": pl.String(),
        }
    )


def _equal_weight_daily_aggregates(
    hourly_matrix: pl.DataFrame, config: Config
) -> pl.DataFrame:
    """Deterministic raw-equal-weight aggregates of the hourly path per local day.

    No fitting is involved, so materializing these is leakage-free; fitted-blend
    aggregates are recomputed inside backtest folds instead.
    """
    if hourly_matrix.is_empty():
        return pl.DataFrame(
            schema={
                "issue_time": pl.Datetime("us", "UTC"),
                "forecast_date": pl.Date(),
            }
        )
    timezone_name = config.station.timezone

    def fx_cols(suffix: str) -> list[str]:
        return [
            c
            for c in hourly_matrix.columns
            if c.startswith("fx__") and c.endswith(f"__{suffix}")
        ]

    temp_cols = fx_cols("temp_c")
    pop_cols = fx_cols("pop")
    precip_cols = fx_cols("precip_mm")
    ew = hourly_matrix.select(
        "issue_time",
        local_date_expr(pl.col("valid_time"), timezone_name).alias("forecast_date"),
        pl.mean_horizontal([pl.col(c) for c in temp_cols]).alias("ew_temp"),
        pl.mean_horizontal([pl.col(c) for c in pop_cols]).alias("ew_pop"),
        pl.mean_horizontal([pl.col(c) for c in precip_cols]).alias("ew_precip"),
    )
    dates = ew["forecast_date"].unique().sort()
    day_lengths = pl.DataFrame(
        {
            "forecast_date": dates,
            "expected_hours": [
                local_day_minutes(day, timezone_name) / 60.0 for day in dates
            ],
        }
    )
    return (
        ew.group_by("issue_time", "forecast_date")
        .agg(
            pl.col("ew_temp").max().alias("ewagg__temp_max_c"),
            pl.col("ew_temp").min().alias("ewagg__temp_min_c"),
            pl.col("ew_pop").max().alias("ewagg__pop"),
            pl.col("ew_precip").sum().alias("ewagg__precip_sum_mm"),
            pl.col("ew_temp").is_not_null().sum().alias("covered_hours"),
        )
        .join(day_lengths, on="forecast_date", how="left")
        .with_columns(
            (pl.col("covered_hours") / pl.col("expected_hours")).alias(
                "ewagg__coverage_frac"
            )
        )
        .drop("covered_hours", "expected_hours")
        .sort("issue_time", "forecast_date")
    )


def build_daily_matrix(
    daily_long: pl.DataFrame,
    snapshots: pl.DataFrame,
    hourly_matrix: pl.DataFrame,
    truth_daily_frame: pl.DataFrame,
    config: Config,
) -> pl.DataFrame:
    """One row per (issue snapshot, target local date)."""
    kind = assert_single_kind(daily_long)
    snap = apply_provider_qc(
        snapshot_long(daily_long, snapshots, config.forecasts.max_forecast_age_hours),
        config,
        value_columns=DAILY_MATRIX_VARIABLES,
        group_key=["issue_time", "forecast_date"],
    )
    if snap.is_empty():
        return pl.DataFrame(
            schema={
                "issue_time": pl.Datetime("us", "UTC"),
                "forecast_date": pl.Date(),
                "lead_days": pl.Int16(),
                "lead_bucket": pl.String(),
                "source_kind": pl.String(),
            }
        )
    snap = snap.sort("source", "issue_time", "forecast_date")
    timezone_name = config.station.timezone
    index = ["issue_time", "forecast_date"]
    wide = snap.select(index).unique(maintain_order=True).sort(index)
    for variable in DAILY_MATRIX_VARIABLES:
        wide = wide.join(_pivot(snap, index, variable, fxd_col), on=index, how="left")
    return (
        wide.with_columns(
            (
                pl.col("forecast_date")
                - local_date_expr(pl.col("issue_time"), timezone_name)
            )
            .dt.total_days()
            .cast(pl.Int16)
            .alias("lead_days"),
            pl.lit(kind).alias("source_kind"),
        )
        .filter(pl.col("lead_days") >= 0)
        .with_columns(
            daily_bucket_expr(pl.col("lead_days").cast(pl.Float64)).alias("lead_bucket")
        )
        .join(
            _equal_weight_daily_aggregates(hourly_matrix, config),
            on=index,
            how="left",
        )
        .join(
            truth_daily_frame.rename({"date_local": "forecast_date"}),
            on="forecast_date",
            how="left",
        )
        .sort("issue_time", "forecast_date")
    )


def matrix_path(directory: Path, product: str, source_kind: str) -> Path:
    """Matrices are keyed by provenance on disk, so the two can never collide."""
    return directory / f"{product}_matrix_{source_kind}.parquet"


@dataclass(frozen=True, slots=True)
class DatasetPaths:
    truth_minute: Path
    truth_hourly: Path
    truth_daily: Path
    forecasts_long: Path
    daily_long: Path
    minutely_long: Path
    hourly_matrix: Path
    daily_matrix: Path
    manifest: Path

    @classmethod
    def in_dir(cls, directory: Path) -> "DatasetPaths":
        live = SourceKind.LIVE.value
        return cls(
            truth_minute=directory / "truth_minute.parquet",
            truth_hourly=directory / "truth_hourly.parquet",
            truth_daily=directory / "truth_daily.parquet",
            forecasts_long=directory / "forecasts_long.parquet",
            daily_long=directory / "daily_long.parquet",
            minutely_long=directory / "minutely_long.parquet",
            hourly_matrix=matrix_path(directory, "hourly", live),
            daily_matrix=matrix_path(directory, "daily", live),
            manifest=directory / "manifest.json",
        )


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class FileInfo:
    rows: int
    sha256_16: str


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    created_at: str
    sources: tuple[str, ...]
    snapshots: int
    files: dict[str, FileInfo]
    fingerprint: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "created_at": self.created_at,
                "sources": list(self.sources),
                "snapshots": self.snapshots,
                "files": {
                    name: {"rows": info.rows, "sha256_16": info.sha256_16}
                    for name, info in self.files.items()
                },
                "fingerprint": self.fingerprint,
            },
            indent=2,
        )


def write_dataset(config: Config) -> DatasetManifest:
    """Run the full dataset build and persist parquet artifacts + manifest."""
    paths = DatasetPaths.in_dir(config.dataset.dir)
    config.dataset.dir.mkdir(parents=True, exist_ok=True)

    channels = sorted(set(config.station.columns.values()))
    observations = read_observations(config.station)
    flagged = apply_qc(observations, config.qc, channels)
    causal_flagged = apply_causal_qc(observations, config.qc, channels)
    minute = truth_minute(flagged, config)
    causal_minute = truth_minute(causal_flagged, config)
    hourly_truth = truth_hourly(minute, config)
    daily_truth = truth_daily(minute, config)

    archive = read_forecast_archive(config.forecasts)
    hourly_long = archive.hourly
    daily_long_frame = archive.daily
    minutely_long = archive.minutely
    snapshots = snapshot_times(archive.completions)

    hourly_matrix = build_hourly_matrix(
        hourly_long, snapshots, hourly_truth, causal_minute, config
    )
    daily_matrix = build_daily_matrix(
        daily_long_frame, snapshots, hourly_matrix, daily_truth, config
    )

    frames: dict[str, tuple[Path, pl.DataFrame]] = {
        "truth_minute": (paths.truth_minute, minute),
        "truth_hourly": (paths.truth_hourly, hourly_truth),
        "truth_daily": (paths.truth_daily, daily_truth),
        "forecasts_long": (paths.forecasts_long, hourly_long),
        "daily_long": (paths.daily_long, daily_long_frame),
        "minutely_long": (paths.minutely_long, minutely_long),
        "hourly_matrix": (paths.hourly_matrix, hourly_matrix),
        "daily_matrix": (paths.daily_matrix, daily_matrix),
    }
    files: dict[str, FileInfo] = {}
    for name, (path, frame) in frames.items():
        frame.write_parquet(path)
        files[name] = FileInfo(rows=frame.height, sha256_16=_file_digest(path))
    fingerprint = hashlib.sha256(
        json.dumps(
            {name: [info.rows, info.sha256_16] for name, info in files.items()},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    manifest = DatasetManifest(
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        sources=tuple(sorted(hourly_long["source"].unique().to_list()))
        if hourly_long.height
        else (),
        snapshots=snapshots.height,
        files=files,
        fingerprint=fingerprint,
    )
    paths.manifest.write_text(manifest.to_json(), encoding="utf-8")
    return manifest


def build_truth(config: Config) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """QC'd minute truth plus its hourly and daily aggregations."""
    channels = sorted(set(config.station.columns.values()))
    flagged = apply_qc(read_observations(config.station), config.qc, channels)
    minute = truth_minute(flagged, config)
    return minute, truth_hourly(minute, config), truth_daily(minute, config)


def build_observation_features(config: Config) -> pl.DataFrame:
    """Canonical issue-time observations using only past-dependent QC rules."""
    channels = sorted(set(config.station.columns.values()))
    observations = read_observations(config.station)
    return truth_minute(apply_causal_qc(observations, config.qc, channels), config)


def _synthetic_daily_matrix(
    hourly_matrix: pl.DataFrame,
    truth_daily_frame: pl.DataFrame,
    config: Config,
) -> pl.DataFrame:
    """Derive daily provider targets from each synthetic hourly snapshot."""
    if hourly_matrix.is_empty():
        return pl.DataFrame()
    timezone_name = config.station.timezone
    keys = ["issue_time", "forecast_date"]
    with_date = hourly_matrix.with_columns(
        local_date_expr(pl.col("valid_time"), timezone_name).alias("forecast_date")
    )
    daily = with_date.select(keys).unique().sort(keys)
    for source in matrix_sources(hourly_matrix):
        expressions: list[pl.Expr] = []
        for source_variable, daily_variable, method in (
            ("temp_c", "temp_max_c", "max"),
            ("temp_c", "temp_min_c", "min"),
            ("pop", "pop", "max"),
            ("precip_mm", "precip_sum_mm", "sum"),
        ):
            column = fx_col(source, source_variable)
            if column not in with_date.columns:
                continue
            expr = getattr(pl.col(column), method)().alias(
                fxd_col(source, daily_variable)
            )
            expressions.append(expr)
        if expressions:
            aggregated = with_date.group_by(keys).agg(*expressions)
            daily = daily.join(aggregated, on=keys, how="left")
    return (
        daily.with_columns(
            (
                pl.col("forecast_date")
                - local_date_expr(pl.col("issue_time"), timezone_name)
            )
            .dt.total_days()
            .cast(pl.Int16)
            .alias("lead_days"),
            pl.lit(SourceKind.SYNTHETIC.value).alias("source_kind"),
        )
        .filter(pl.col("lead_days") >= 0)
        .with_columns(
            daily_bucket_expr(pl.col("lead_days").cast(pl.Float64)).alias("lead_bucket")
        )
        .join(
            _equal_weight_daily_aggregates(hourly_matrix, config), on=keys, how="left"
        )
        .join(
            truth_daily_frame.rename({"date_local": "forecast_date"}),
            on="forecast_date",
            how="left",
        )
        .sort(keys)
    )


def write_synthetic_matrix(
    config: Config, synthetic_long: pl.DataFrame
) -> tuple[Path, int]:
    """Build and persist the synthetic hourly matrix from a backfill long frame.

    Snapshot times come from the backfill's own synthetic fetch times, since a
    backfilled archive has no ``forecast_runs`` rows to anchor to.
    """
    _, hourly_truth, daily_truth = build_truth(config)
    causal_minute = build_observation_features(config)
    snapshots = (
        synthetic_long.select(pl.col("fetched_at").alias("issue_time"))
        .unique()
        .sort("issue_time")
    )
    matrix = build_hourly_matrix(
        synthetic_long, snapshots, hourly_truth, causal_minute, config
    )
    config.dataset.dir.mkdir(parents=True, exist_ok=True)
    path = matrix_path(config.dataset.dir, "hourly", SourceKind.SYNTHETIC.value)
    matrix.write_parquet(path)
    daily_matrix = _synthetic_daily_matrix(matrix, daily_truth, config)
    daily_matrix.write_parquet(
        matrix_path(config.dataset.dir, "daily", SourceKind.SYNTHETIC.value)
    )
    synthetic_long.write_parquet(
        config.dataset.dir / "forecasts_long_synthetic.parquet"
    )
    return path, matrix.height


def matrix_sources(frame: pl.DataFrame) -> tuple[str, ...]:
    """Sources present in a wide matrix, from its fx column names."""
    sources = {
        parse_fx_col(column)[0]
        for column in frame.columns
        if column.startswith(("fx__", "fxd__"))
    }
    return tuple(sorted(sources))


def truth_column_for(variable: VariableSpec, semantics: TruthSemantics) -> str:
    if variable.has_dual_semantics:
        return truth_col(variable.name, semantics)
    return truth_col(variable.name)


def to_forecast_matrix(
    frame: pl.DataFrame,
    variable: VariableSpec,
    *,
    daily: bool = False,
    sources: tuple[str, ...] | None = None,
) -> ForecastMatrix:
    """Wide matrix rows -> a ForecastMatrix, with no truth requirement.

    ``sources`` pins the column order and set. Serving must pass the *training*
    source list so a provider that is merely missing right now becomes an
    unavailable column rather than shifting every other blender's indices.
    """
    chosen = sources if sources is not None else matrix_sources(frame)
    if not chosen:
        msg = "matrix has no forecast columns"
        raise ValueError(msg)
    column_builder = fxd_col if daily else fx_col
    lead_expr = (
        (pl.col("lead_days") * 24.0).cast(pl.Float64)
        if daily
        else pl.col("lead_hours").cast(pl.Float64)
    )
    usable = frame.with_columns(lead_expr.alias("__lead"))
    values = np.column_stack(
        [
            usable[column_builder(source, variable.name)].to_numpy()
            if column_builder(source, variable.name) in usable.columns
            else np.full(usable.height, np.nan)
            for source in chosen
        ]
    ).astype(np.float64)
    feature_columns = [
        c
        for c in usable.columns
        if c in ("issue_time", "lead_bucket", "valid_hour_local", "valid_month")
        or c.startswith(("age__", "obs__", "ewagg__"))
    ]
    return ForecastMatrix.build(
        sources=chosen,
        values=values if usable.height else np.empty((0, len(chosen))),
        lead_hours=usable["__lead"].to_numpy().astype(np.float64),
        features=usable.select(feature_columns),
        product=Product.DAILY if daily else Product.HOURLY,
    )


def to_supervised_slice(
    frame: pl.DataFrame,
    variable: VariableSpec,
    *,
    daily: bool = False,
    semantics: TruthSemantics = TruthSemantics.INSTANTANEOUS,
) -> SupervisedSlice:
    """Wide matrix rows -> contract types, dropping null-truth rows."""
    kind = assert_single_kind(frame)
    truth_column = truth_column_for(variable, semantics)
    sources = matrix_sources(frame)
    if not sources:
        msg = "matrix has no forecast columns"
        raise ValueError(msg)
    column_builder = fxd_col if daily else fx_col
    lead_expr = (
        (pl.col("lead_days") * 24.0).cast(pl.Float64)
        if daily
        else pl.col("lead_hours").cast(pl.Float64)
    )
    usable = frame.filter(pl.col(truth_column).is_not_null()).with_columns(
        lead_expr.alias("__lead")
    )
    values = np.column_stack(
        [
            usable[column_builder(source, variable.name)].to_numpy()
            if column_builder(source, variable.name) in usable.columns
            else np.full(usable.height, np.nan)
            for source in sources
        ]
    ).astype(np.float64)
    feature_columns = [
        c
        for c in usable.columns
        if c in ("issue_time", "lead_bucket", "valid_hour_local", "valid_month")
        or c.startswith(("age__", "obs__", "ewagg__"))
    ]
    x = ForecastMatrix.build(
        sources=sources,
        values=values if usable.height else np.empty((0, len(sources))),
        lead_hours=usable["__lead"].to_numpy().astype(np.float64),
        features=usable.select(feature_columns),
        product=Product.DAILY if daily else Product.HOURLY,
    )
    y = usable[truth_column].to_numpy().astype(np.float64)
    return SupervisedSlice(x=x, y=y, variable=variable, source_kind=SourceKind(kind))
