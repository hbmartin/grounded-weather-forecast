"""TOML configuration loaded into frozen dataclasses with explicit validation.

The config carries everything location-specific (DB paths, coordinates,
station column/unit mappings) so the codebase itself stays station-agnostic.
"""

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class ConfigError(ValueError):
    """The TOML file is missing required keys or has ill-typed values."""


DEFAULT_STATION_COLUMNS: Mapping[str, str] = MappingProxyType(
    {
        "outTemp": "temp",
        "outHumi": "humidity",
        "avgwind": "wind_speed",
        "gustspeed": "wind_gust",
        "eventrain": "rain_counter",
        "AbsPress": "pressure_station",
    }
)

DEFAULT_STATION_UNITS: Mapping[str, str] = MappingProxyType(
    {
        "temp": "degF",
        "humidity": "pct",
        "wind_speed": "mph",
        "wind_gust": "mph",
        "rain_counter": "inch",
        "pressure_station": "inHg",
    }
)

DEFAULT_QC_BOUNDS: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        "temp": (-40.0, 55.0),
        "humidity": (0.0, 100.0),
        "wind_speed": (0.0, 60.0),
        "wind_gust": (0.0, 90.0),
        "rain_counter": (0.0, 1000.0),
        "pressure_station": (600.0, 1100.0),
    }
)

DEFAULT_QC_MAX_STEP: Mapping[str, float] = MappingProxyType(
    {"temp": 5.0, "humidity": 25.0, "pressure_station": 3.0}
)

DEFAULT_QC_FLATLINE_MINUTES: Mapping[str, int] = MappingProxyType(
    {"temp": 180, "pressure_station": 360}
)

# Absolute physical plausibility bounds for provider (forecast) values, keyed by
# canonical variable. These catch gross unit/garbage errors (e.g. a snow depth
# written into a liquid field, a pressure in the wrong unit) before grounding.
DEFAULT_PROVIDER_QC_BOUNDS: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        "temp_c": (-90.0, 60.0),
        "temp_max_c": (-90.0, 60.0),
        "temp_min_c": (-90.0, 60.0),
        "dew_point_c": (-90.0, 45.0),
        "humidity_pct": (0.0, 100.0),
        "wind_speed_ms": (0.0, 120.0),
        "wind_gust_ms": (0.0, 150.0),
        "pressure_sea_hpa": (850.0, 1090.0),
        "precip_mm": (0.0, 500.0),
        "precip_sum_mm": (0.0, 2000.0),
        "pop": (0.0, 1.0),
    }
)

# Variables where a single provider disagreeing wildly with the others is far more
# likely an error than genuine diversity (roughly Gaussian, not zero-inflated).
# Skewed/zero-inflated fields (precip, pop, gusts) are deliberately excluded.
DEFAULT_PROVIDER_QC_CROSS_SOURCE: tuple[str, ...] = (
    "temp_c",
    "temp_max_c",
    "temp_min_c",
    "dew_point_c",
    "humidity_pct",
    "pressure_sea_hpa",
)

# Minimum absolute deviation from the cross-source median before a value can be
# flagged, so tightly-agreeing providers cannot make a merely-different value an
# outlier. A value is nulled only when it exceeds BOTH k*scaled-MAD and this floor,
# which keeps the pass conservative and preserves genuine provider diversity.
DEFAULT_PROVIDER_QC_MIN_DEVIATION: Mapping[str, float] = MappingProxyType(
    {
        "temp_c": 8.0,
        "temp_max_c": 8.0,
        "temp_min_c": 8.0,
        "dew_point_c": 8.0,
        "humidity_pct": 40.0,
        "pressure_sea_hpa": 20.0,
    }
)


@dataclass(frozen=True, slots=True)
class StationConfig:
    db_path: Path
    timezone: str
    latitude: float
    longitude: float
    elevation_m: float
    immutable: bool
    columns: Mapping[str, str]
    units: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ForecastsConfig:
    db_path: Path
    sources: tuple[str, ...]
    max_forecast_age_hours: float
    immutable: bool
    latitude: float
    longitude: float
    # Per-source lead cap in hours: rows beyond a source's cap are dropped at
    # matrix build, trimming horizon-edge garbage without excluding the source.
    max_lead_hours: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    # (source, variable) pairs whose values are nulled at matrix build — for
    # providers whose signal is genuinely bad for one variable only.
    exclude: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    dir: Path
    min_hour_coverage: float
    min_day_coverage: float
    pop_threshold_mm: float
    precip_reset_fraction: float


@dataclass(frozen=True, slots=True)
class ProviderQcConfig:
    """Plausibility QC applied to provider (forecast) values before grounding."""

    enabled: bool
    bounds: Mapping[str, tuple[float, float]]
    cross_source_variables: tuple[str, ...]
    mad_k: float
    min_sources: int
    min_deviation: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class QcConfig:
    bounds: Mapping[str, tuple[float, float]]
    max_step: Mapping[str, float]
    flatline_minutes: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class BackfillConfig:
    models: tuple[str, ...]
    start_date: date | None
    dynamical_models: tuple[str, ...] = ()
    dynamical_start_date: date | None = None
    dynamical_publication_lag_hours: float = 6.0
    dynamical_max_lead_hours: float = 48.0


# Canonical variables the Ensemble API ingest reduces by default.
DEFAULT_ENSEMBLE_VARIABLES: tuple[str, ...] = (
    "temp_c",
    "dew_point_c",
    "wind_speed_ms",
    "wind_gust_ms",
    "pressure_sea_hpa",
    "precip_mm",
)


@dataclass(frozen=True, slots=True)
class EnsemblesConfig:
    """Open-Meteo Ensemble API ingestion (spread features, not sources)."""

    models: tuple[str, ...]
    variables: tuple[str, ...]

    @property
    def enabled(self) -> bool:
        return bool(self.models)


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    initial_train_days: int
    step_days: int
    rolling_window_days: int


@dataclass(frozen=True, slots=True)
class TruthQcConfig:
    """Neighbor-station cross-checks (Synoptic free tier)."""

    synoptic_token: str = ""
    radius_km: float = 25.0
    elevation_band_m: float = 300.0
    lapse_k_per_km: float = 6.5


@dataclass(frozen=True, slots=True)
class PromotionConfig:
    """How slice winners are promoted to serving releases."""

    rule: str = "mcs"
    alpha: float = 0.1
    live_gap_factor: float = 1.5
    min_live_n: int = 24
    # Relative served-vs-board-minimum MAE gap above which the leaderboard
    # report flags a slice as a blocked promotion.
    report_gap_threshold: float = 0.15


@dataclass(frozen=True, slots=True)
class PredictConfig:
    selection: str
    history_path: Path
    methods: Mapping[str, str]
    minutely_tau_hours: float
    # Post-hoc transform applied to natively-emitted quantiles at serve time;
    # the offline report section arbitrates which mode earns this switch.
    quantile_recalibration: str


@dataclass(frozen=True, slots=True)
class Config:
    station: StationConfig
    forecasts: ForecastsConfig
    dataset: DatasetConfig
    qc: QcConfig
    provider_qc: ProviderQcConfig
    backfill: BackfillConfig
    ensembles: EnsemblesConfig
    backtest: BacktestConfig
    predict: PredictConfig
    promotion: PromotionConfig
    truth_qc: TruthQcConfig
    reports_dir: Path
    artifacts_dir: Path


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    match raw.get(name, {}):
        case dict() as section:
            return section
        case _:
            msg = f"[{name}] must be a table"
            raise ConfigError(msg)


def _require(section: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in section:
        msg = f"missing required key {key!r} in [{context}]"
        raise ConfigError(msg)
    return section[key]


def _number(value: Any, key: str, context: str) -> float:
    match value:
        case bool():
            pass
        case int() | float():
            return float(value)
    msg = f"{key!r} in [{context}] must be a number, got {type(value).__name__}"
    raise ConfigError(msg)


def _finite_number(value: Any, key: str, context: str) -> float:
    number = _number(value, key, context)
    if not math.isfinite(number):
        msg = f"{key!r} in [{context}] must be finite"
        raise ConfigError(msg)
    return number


def _positive_number(value: Any, key: str, context: str) -> float:
    number = _finite_number(value, key, context)
    if number <= 0.0:
        msg = f"{key!r} in [{context}] must be > 0"
        raise ConfigError(msg)
    return number


def _choice(value: Any, allowed: tuple[str, ...], key: str, context: str) -> str:
    text = str(value)
    if text not in allowed:
        options = ", ".join(repr(option) for option in allowed)
        msg = f"{key!r} in [{context}] must be one of {options}, got {text!r}"
        raise ConfigError(msg)
    return text


def _positive_int(value: Any, key: str, context: str) -> int:
    number = _finite_number(value, key, context)
    if not number.is_integer() or number <= 0.0:
        msg = f"{key!r} in [{context}] must be a positive integer"
        raise ConfigError(msg)
    return int(number)


def _fraction(value: Any, key: str, context: str) -> float:
    number = _finite_number(value, key, context)
    if not 0.0 <= number <= 1.0:
        msg = f"{key!r} in [{context}] must be between 0 and 1"
        raise ConfigError(msg)
    return number


def _str_map(value: Any, key: str, context: str) -> dict[str, str]:
    match value:
        case dict() as mapping if all(
            isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()
        ):
            return dict(mapping)
        case _:
            msg = f"{key!r} in [{context}] must be a table of strings"
            raise ConfigError(msg)


def _station(raw: Mapping[str, Any]) -> StationConfig:
    section = _section(raw, "station")
    columns = dict(DEFAULT_STATION_COLUMNS)
    columns |= _str_map(section.get("columns", {}), "columns", "station")
    duplicate_targets = sorted(
        channel
        for channel in set(columns.values())
        if list(columns.values()).count(channel) > 1
    )
    if duplicate_targets:
        msg = f"[station.columns] maps multiple database columns to {duplicate_targets}"
        raise ConfigError(msg)
    units = dict(DEFAULT_STATION_UNITS)
    units |= _str_map(section.get("units", {}), "units", "station")
    timezone = str(section.get("timezone", "UTC"))
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError,) as exc:  # noqa: B013 - project exception style
        msg = f"unknown [station].timezone {timezone!r}"
        raise ConfigError(msg) from exc
    latitude = _finite_number(
        _require(section, "latitude", "station"), "latitude", "station"
    )
    longitude = _finite_number(
        _require(section, "longitude", "station"), "longitude", "station"
    )
    if not -90.0 <= latitude <= 90.0:
        raise ConfigError("'latitude' in [station] must be between -90 and 90")
    if not -180.0 <= longitude <= 180.0:
        raise ConfigError("'longitude' in [station] must be between -180 and 180")
    return StationConfig(
        db_path=Path(str(_require(section, "db_path", "station"))),
        timezone=timezone,
        latitude=latitude,
        longitude=longitude,
        elevation_m=_finite_number(
            _require(section, "elevation_m", "station"), "elevation_m", "station"
        ),
        immutable=bool(section.get("immutable", False)),
        columns=MappingProxyType(columns),
        units=MappingProxyType(units),
    )


def _forecast_exclusions(section: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    raw = section.get("exclude", [])
    if not isinstance(raw, list) or not all(isinstance(e, str) for e in raw):
        msg = "'exclude' in [forecasts] must be a list of 'source:variable' strings"
        raise ConfigError(msg)
    pairs: list[tuple[str, str]] = []
    for entry in raw:
        source, _, variable = entry.partition(":")
        if not source or not variable:
            msg = (
                f"[forecasts].exclude entries must be 'source:variable', got {entry!r}"
            )
            raise ConfigError(msg)
        pairs.append((source, variable))
    return tuple(pairs)


def _forecast_lead_caps(section: Mapping[str, Any]) -> Mapping[str, float]:
    raw = section.get("max_lead_hours", {})
    if not isinstance(raw, Mapping):
        msg = "[forecasts].max_lead_hours must be a table of source = hours"
        raise ConfigError(msg)
    caps = {
        str(source): _positive_number(hours, f"max_lead_hours.{source}", "forecasts")
        for source, hours in sorted(raw.items())
    }
    return MappingProxyType(caps)


def _forecasts(raw: Mapping[str, Any], station: StationConfig) -> ForecastsConfig:
    section = _section(raw, "forecasts")
    sources = section.get("sources", [])
    if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
        msg = "'sources' in [forecasts] must be a list of strings"
        raise ConfigError(msg)
    return ForecastsConfig(
        db_path=Path(str(_require(section, "db_path", "forecasts"))),
        sources=tuple(sources),
        max_forecast_age_hours=_positive_number(
            section.get("max_forecast_age_hours", 12.0),
            "max_forecast_age_hours",
            "forecasts",
        ),
        immutable=bool(section.get("immutable", False)),
        latitude=station.latitude,
        longitude=station.longitude,
        max_lead_hours=_forecast_lead_caps(section),
        exclude=_forecast_exclusions(section),
    )


def _dataset(raw: Mapping[str, Any]) -> DatasetConfig:
    section = _section(raw, "dataset")
    return DatasetConfig(
        dir=Path(str(section.get("dir", "data"))),
        min_hour_coverage=_fraction(
            section.get("min_hour_coverage", 0.8), "min_hour_coverage", "dataset"
        ),
        min_day_coverage=_fraction(
            section.get("min_day_coverage", 0.8), "min_day_coverage", "dataset"
        ),
        pop_threshold_mm=_positive_number(
            section.get("pop_threshold_mm", 0.254), "pop_threshold_mm", "dataset"
        ),
        precip_reset_fraction=_fraction(
            section.get("precip_reset_fraction", 0.5),
            "precip_reset_fraction",
            "dataset",
        ),
    )


def _bounds_map(value: Any, section: str = "qc") -> dict[str, tuple[float, float]]:
    match value:
        case dict() as mapping:
            result: dict[str, tuple[float, float]] = {}
            for key, pair in mapping.items():
                match pair:
                    case [lo, hi] if isinstance(lo, (int, float)) and isinstance(
                        hi, (int, float)
                    ):
                        low = float(lo)
                        high = float(hi)
                        if (
                            not math.isfinite(low)
                            or not math.isfinite(high)
                            or low > high
                        ):
                            msg = f"bounds for {key!r} must be finite and ordered"
                            raise ConfigError(msg)
                        result[str(key)] = (low, high)
                    case _:
                        msg = f"bounds for {key!r} must be [low, high]"
                        raise ConfigError(msg)
            return result
        case _:
            msg = f"'bounds' in [{section}] must be a table"
            raise ConfigError(msg)


def _qc(raw: Mapping[str, Any]) -> QcConfig:
    section = _section(raw, "qc")
    bounds = dict(DEFAULT_QC_BOUNDS) | _bounds_map(section.get("bounds", {}))
    max_step = dict(DEFAULT_QC_MAX_STEP)
    max_step_section = _section(section, "max_step") if "max_step" in section else {}
    for key, value in max_step_section.items():
        max_step[str(key)] = _positive_number(value, str(key), "qc.max_step")
    flatline = dict(DEFAULT_QC_FLATLINE_MINUTES)
    flatline_section = (
        _section(section, "flatline_minutes") if "flatline_minutes" in section else {}
    )
    for key, value in flatline_section.items():
        flatline[str(key)] = _positive_int(value, str(key), "qc.flatline_minutes")
    return QcConfig(
        bounds=MappingProxyType(bounds),
        max_step=MappingProxyType(max_step),
        flatline_minutes=MappingProxyType(flatline),
    )


def _deviation_map(value: Any) -> dict[str, float]:
    match value:
        case dict() as mapping:
            return {
                str(key): _positive_number(v, str(key), "provider_qc.min_deviation")
                for key, v in mapping.items()
            }
        case _:
            msg = "'min_deviation' in [provider_qc] must be a table"
            raise ConfigError(msg)


def _provider_qc(raw: Mapping[str, Any]) -> ProviderQcConfig:
    section = _section(raw, "provider_qc") if "provider_qc" in raw else {}
    bounds = dict(DEFAULT_PROVIDER_QC_BOUNDS) | _bounds_map(
        section.get("bounds", {}), "provider_qc"
    )
    min_deviation = dict(DEFAULT_PROVIDER_QC_MIN_DEVIATION) | _deviation_map(
        section.get("min_deviation", {})
    )
    cross_source = section.get(
        "cross_source_variables", list(DEFAULT_PROVIDER_QC_CROSS_SOURCE)
    )
    if not isinstance(cross_source, list) or not all(
        isinstance(v, str) for v in cross_source
    ):
        msg = "'cross_source_variables' in [provider_qc] must be a list of strings"
        raise ConfigError(msg)
    return ProviderQcConfig(
        enabled=bool(section.get("enabled", True)),
        bounds=MappingProxyType(bounds),
        cross_source_variables=tuple(cross_source),
        mad_k=_positive_number(section.get("mad_k", 5.0), "mad_k", "provider_qc"),
        min_sources=_positive_int(
            section.get("min_sources", 4), "min_sources", "provider_qc"
        ),
        min_deviation=MappingProxyType(min_deviation),
    )


def _optional_date(value: Any, context: str) -> date | None:
    match value:
        case None:
            return None
        case datetime():
            msg = f"'start_date' in [{context}] must be a date, not datetime"
            raise ConfigError(msg)
        case date():
            return value
        case str():
            try:
                return date.fromisoformat(value)
            except (ValueError,) as exc:  # noqa: B013 - project exception style
                msg = f"'start_date' in [{context}] must be YYYY-MM-DD"
                raise ConfigError(msg) from exc
        case _:
            msg = f"'start_date' in [{context}] must be a date"
            raise ConfigError(msg)


def _backfill(raw: Mapping[str, Any]) -> BackfillConfig:
    section = _section(raw, "backfill")
    open_meteo = _section(section, "open_meteo") if "open_meteo" in section else {}
    models = open_meteo.get("models", [])
    if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
        msg = "'models' in [backfill.open_meteo] must be a list of strings"
        raise ConfigError(msg)
    dynamical = _section(section, "dynamical") if "dynamical" in section else {}
    return BackfillConfig(
        models=tuple(models),
        start_date=_optional_date(open_meteo.get("start_date"), "backfill.open_meteo"),
        dynamical_models=_string_tuple(
            dynamical.get("models", []), "models", "backfill.dynamical"
        ),
        dynamical_start_date=_optional_date(
            dynamical.get("start_date"), "backfill.dynamical"
        ),
        dynamical_publication_lag_hours=_positive_number(
            dynamical.get("publication_lag_hours", 6.0),
            "publication_lag_hours",
            "backfill.dynamical",
        ),
        dynamical_max_lead_hours=_positive_number(
            dynamical.get("max_lead_hours", 48.0),
            "max_lead_hours",
            "backfill.dynamical",
        ),
    )


_ENSEMBLE_ALLOWED_VARIABLES: tuple[str, ...] = (
    *DEFAULT_ENSEMBLE_VARIABLES,
    "humidity_pct",
)


def _string_tuple(value: Any, key: str, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        msg = f"{key!r} in [{context}] must be a list of strings"
        raise ConfigError(msg)
    return tuple(value)


def _ensembles(raw: Mapping[str, Any]) -> EnsemblesConfig:
    section = _section(raw, "ensembles") if "ensembles" in raw else {}
    models = _string_tuple(section.get("models", []), "models", "ensembles")
    variables = _string_tuple(
        section.get("variables", list(DEFAULT_ENSEMBLE_VARIABLES)),
        "variables",
        "ensembles",
    )
    if unknown := sorted(set(variables) - set(_ENSEMBLE_ALLOWED_VARIABLES)):
        msg = (
            f"unknown [ensembles].variables {unknown}; "
            f"allowed: {sorted(_ENSEMBLE_ALLOWED_VARIABLES)}"
        )
        raise ConfigError(msg)
    return EnsemblesConfig(models=models, variables=variables)


def _backtest(raw: Mapping[str, Any]) -> BacktestConfig:
    section = _section(raw, "backtest")
    return BacktestConfig(
        initial_train_days=_positive_int(
            section.get("initial_train_days", 90), "initial_train_days", "backtest"
        ),
        step_days=_positive_int(section.get("step_days", 7), "step_days", "backtest"),
        rolling_window_days=_positive_int(
            section.get("rolling_window_days", 180),
            "rolling_window_days",
            "backtest",
        ),
    )


def _truth_qc(raw: Mapping[str, Any]) -> TruthQcConfig:
    section = _section(raw, "truth_qc") if "truth_qc" in raw else {}
    return TruthQcConfig(
        synoptic_token=str(section.get("synoptic_token", "")),
        radius_km=_positive_number(
            section.get("radius_km", 25.0), "radius_km", "truth_qc"
        ),
        elevation_band_m=_positive_number(
            section.get("elevation_band_m", 300.0), "elevation_band_m", "truth_qc"
        ),
        lapse_k_per_km=_positive_number(
            section.get("lapse_k_per_km", 6.5), "lapse_k_per_km", "truth_qc"
        ),
    )


def _promotion(raw: Mapping[str, Any]) -> PromotionConfig:
    section = _section(raw, "promotion") if "promotion" in raw else {}
    rule = str(section.get("rule", "mcs"))
    if rule not in ("mcs", "legacy"):
        msg = f"[promotion].rule must be 'mcs' or 'legacy', got {rule!r}"
        raise ConfigError(msg)
    return PromotionConfig(
        rule=rule,
        alpha=_fraction(section.get("alpha", 0.1), "alpha", "promotion"),
        live_gap_factor=_positive_number(
            section.get("live_gap_factor", 1.5), "live_gap_factor", "promotion"
        ),
        min_live_n=_positive_int(
            section.get("min_live_n", 24), "min_live_n", "promotion"
        ),
        report_gap_threshold=_positive_number(
            section.get("report_gap_threshold", 0.15),
            "report_gap_threshold",
            "promotion",
        ),
    )


def _predict(raw: Mapping[str, Any], dataset_dir: Path) -> PredictConfig:
    section = _section(raw, "predict")
    return PredictConfig(
        selection=str(section.get("selection", "skill_per_slice")),
        history_path=Path(
            str(section.get("history_path", dataset_dir / "predict_history.parquet"))
        ),
        methods=MappingProxyType(
            _str_map(section.get("methods", {}), "methods", "predict")
        ),
        minutely_tau_hours=_positive_number(
            section.get("minutely_tau_hours", 3.0),
            "minutely_tau_hours",
            "predict",
        ),
        quantile_recalibration=_choice(
            section.get("quantile_recalibration", "none"),
            ("none", "pit", "cqr"),
            "quantile_recalibration",
            "predict",
        ),
    )


def load_config(path: Path) -> Config:
    """Load and validate a config file; raises :class:`ConfigError` on problems."""
    try:
        raw: Mapping[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        msg = f"cannot load config {path}: {exc}"
        raise ConfigError(msg) from exc
    dataset = _dataset(raw)
    station = _station(raw)
    return Config(
        station=station,
        forecasts=_forecasts(raw, station),
        dataset=dataset,
        qc=_qc(raw),
        provider_qc=_provider_qc(raw),
        backfill=_backfill(raw),
        ensembles=_ensembles(raw),
        backtest=_backtest(raw),
        predict=_predict(raw, dataset.dir),
        promotion=_promotion(raw),
        truth_qc=_truth_qc(raw),
        reports_dir=Path(str(_section(raw, "reports").get("dir", "reports"))),
        artifacts_dir=Path(str(_section(raw, "artifacts").get("dir", "artifacts"))),
    )
