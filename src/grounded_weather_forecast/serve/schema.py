"""The emitted forecast document: typed, versioned, JSON-serializable."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime

SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class MinutelyPoint:
    valid_time: str
    minutes_ahead: int
    temp_c: float | None = None
    humidity_pct: float | None = None
    dew_point_c: float | None = None
    wind_speed_ms: float | None = None
    precip_intensity_mmh: float | None = None
    pop: float | None = None
    methods: dict[str, str] = field(default_factory=dict)
    quantiles: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HourlyPoint:
    valid_time: str
    lead_hours: float
    lead_bucket: str | None = None
    values: dict[str, float | None] = field(default_factory=dict)
    methods: dict[str, str] = field(default_factory=dict)
    quantiles: dict[str, dict[str, float]] = field(default_factory=dict)
    selection_reasons: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DailyPoint:
    date_local: str
    lead_days: int
    values: dict[str, float | None] = field(default_factory=dict)
    methods: dict[str, str] = field(default_factory=dict)
    quantiles: dict[str, dict[str, float]] = field(default_factory=dict)
    selection_reasons: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Forecast:
    schema_version: int
    issued_at: str
    latitude: float
    longitude: float
    dataset_fingerprint: str
    sources: list[str]
    observation_at: str | None
    minutely: list[MinutelyPoint]
    hourly: list[HourlyPoint]
    daily: list[DailyPoint]
    timezone: str = "UTC"
    status: str = "ready"
    release_ids: list[str] = field(default_factory=list)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(asdict(self), indent=indent, default=_encode, allow_nan=False)

    @classmethod
    def from_json(cls, payload: str) -> "Forecast":
        """Load the versioned document, accepting schema-1 optional omissions."""
        raw = json.loads(payload)
        return cls(
            schema_version=int(raw["schema_version"]),
            issued_at=str(raw["issued_at"]),
            latitude=float(raw["latitude"]),
            longitude=float(raw["longitude"]),
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            sources=[str(source) for source in raw["sources"]],
            observation_at=raw.get("observation_at"),
            minutely=[MinutelyPoint(**point) for point in raw.get("minutely", [])],
            hourly=[HourlyPoint(**point) for point in raw.get("hourly", [])],
            daily=[DailyPoint(**point) for point in raw.get("daily", [])],
            timezone=str(raw.get("timezone", "UTC")),
            status=str(raw.get("status", "ready")),
            release_ids=[str(value) for value in raw.get("release_ids", [])],
        )


def _encode(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    msg = f"cannot serialize {type(value).__name__}"
    raise TypeError(msg)
