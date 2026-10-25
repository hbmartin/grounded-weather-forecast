"""Pure UTC validation, without filesystem or approval dependencies."""

from datetime import UTC, datetime


def aware_utc(value: object) -> datetime | None:
    """Accept aware datetimes/ISO strings; never interpret a local wall clock."""
    try:
        if isinstance(value, datetime):
            moment = value
        elif isinstance(value, str):
            moment = datetime.fromisoformat(value)
        else:
            return None
        if moment.tzinfo is None or moment.utcoffset() is None:
            return None
        return moment.astimezone(UTC)
    except (ValueError, TypeError, OverflowError):
        return None
