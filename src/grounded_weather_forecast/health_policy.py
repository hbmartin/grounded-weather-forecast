"""Read-only freshness policy shared by reporting and evidence collectors."""

from collections.abc import Mapping

FRESHNESS_THRESHOLDS: Mapping[str, float] = {
    "truth_age_minutes": 120.0,
    "collector_age_minutes": 180.0,
    "served_history_age_minutes": 180.0,
    "forecast_document_age_minutes": 180.0,
    "truth_qc_age_minutes": 2880.0,
}
