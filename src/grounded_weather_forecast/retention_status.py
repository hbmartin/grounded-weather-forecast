"""Read-only interpretation of pruning verification recorded in the run ledger."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import polars as pl

from grounded_weather_forecast.status import ComponentDiagnostic


@dataclass(frozen=True, slots=True)
class RetentionVerification:
    state: Literal["clear", "suspended", "not_recorded"] = "not_recorded"
    recorded_at: datetime | None = None
    details: tuple[str, ...] = ()

    def diagnostics(self) -> tuple[ComponentDiagnostic, ...]:
        if self.state == "not_recorded":
            return ()
        return tuple(
            ComponentDiagnostic(
                "retention_verification", 0, f"retention_{self.state}", detail
            )
            for detail in (self.details or ("published-score protection verified",))
        )

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "verification": [self.state],
                "recorded_at": [
                    self.recorded_at.isoformat() if self.recorded_at else None
                ],
                "detail": ["; ".join(self.details) or None],
            }
        )


def _moment(value: object) -> datetime | None:
    try:
        moment = (
            value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        )
        return moment.astimezone(UTC) if moment.tzinfo is not None else None
    except (ValueError, TypeError, OverflowError):
        return None


def latest_retention_verification(
    runs: pl.DataFrame, *, as_of: datetime
) -> RetentionVerification:
    if runs.is_empty() or not {"command", "ended_at"}.issubset(runs.columns):
        return RetentionVerification()
    rows = [
        (moment, row)
        for row in runs.filter(pl.col("command") == "prune-scores").iter_rows(
            named=True
        )
        if (moment := _moment(row.get("ended_at"))) is not None and moment <= as_of
    ]
    if not rows:
        return RetentionVerification()
    moment, row = max(rows, key=lambda pair: pair[0])
    try:
        raw = json.loads(row.get("failure_details_json") or "null")
        if not isinstance(raw, list):
            return RetentionVerification(recorded_at=moment)
        items = [
            item
            for item in raw
            if isinstance(item, dict)
            and item.get("stage") == "retention_verification"
            and item.get("exit_code") == 0
        ]
        kinds = {item.get("failure_kind") for item in items}
        state = (
            "suspended"
            if "retention_suspended" in kinds
            else "clear"
            if kinds == {"retention_clear"}
            else "not_recorded"
        )
        details = tuple(
            str(item["detail"])[:400] for item in items if item.get("detail")
        )
        return RetentionVerification(state, moment, details)
    except (ValueError, TypeError):
        return RetentionVerification(recorded_at=moment)
