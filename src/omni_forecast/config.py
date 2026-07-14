"""TOML configuration loaded into frozen dataclasses with explicit validation.

The config carries everything location-specific (DB paths, coordinates,
station column/unit mappings) so the codebase itself stays station-agnostic.
"""

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    dir: Path
    min_hour_coverage: float
    min_day_coverage: float
    pop_threshold_mm: float


@dataclass(frozen=True, slots=True)
class QcConfig:
    bounds: Mapping[str, tuple[float, float]]
    max_step: Mapping[str, float]
    flatline_minutes: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class BackfillConfig:
    models: tuple[str, ...]
    start_date: date | None


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    initial_train_days: int
    step_days: int
    rolling_window_days: int


@dataclass(frozen=True, slots=True)
class PredictConfig:
    selection: str
    history_path: Path
    methods: Mapping[str, str]
    minutely_tau_hours: float


@dataclass(frozen=True, slots=True)
class Config:
    station: StationConfig
    forecasts: ForecastsConfig
    dataset: DatasetConfig
    qc: QcConfig
    backfill: BackfillConfig
    backtest: BacktestConfig
    predict: PredictConfig
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
            section.get("elevation_m", 0.0), "elevation_m", "station"
        ),
        immutable=bool(section.get("immutable", False)),
        columns=MappingProxyType(columns),
        units=MappingProxyType(units),
    )


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
    )


def _bounds_map(value: Any) -> dict[str, tuple[float, float]]:
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
            msg = "'bounds' in [qc] must be a table"
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


def _backfill(raw: Mapping[str, Any]) -> BackfillConfig:
    section = _section(raw, "backfill")
    open_meteo = _section(section, "open_meteo") if "open_meteo" in section else {}
    models = open_meteo.get("models", [])
    if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
        msg = "'models' in [backfill.open_meteo] must be a list of strings"
        raise ConfigError(msg)
    raw_start = open_meteo.get("start_date")
    match raw_start:
        case None:
            start = None
        case datetime():
            msg = "'start_date' in [backfill.open_meteo] must be a date, not datetime"
            raise ConfigError(msg)
        case date():
            start = raw_start
        case str():
            try:
                start = date.fromisoformat(raw_start)
            except (ValueError,) as exc:  # noqa: B013 - project exception style
                msg = "'start_date' in [backfill.open_meteo] must be YYYY-MM-DD"
                raise ConfigError(msg) from exc
        case _:
            msg = "'start_date' in [backfill.open_meteo] must be a date"
            raise ConfigError(msg)
    return BackfillConfig(models=tuple(models), start_date=start)


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
        backfill=_backfill(raw),
        backtest=_backtest(raw),
        predict=_predict(raw, dataset.dir),
        reports_dir=Path(str(_section(raw, "reports").get("dir", "reports"))),
        artifacts_dir=Path(str(_section(raw, "artifacts").get("dir", "artifacts"))),
    )
