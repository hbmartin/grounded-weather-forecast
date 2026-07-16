def invalid_hourly_grouping(frame, config):
    # ruleid: provider-qc-matrix-grouping-must-include-snapshot
    return apply_provider_qc(
        frame,
        config,
        value_columns=("temp_c",),
        group_key="valid_time",
    )


def invalid_daily_grouping(frame, config):
    # ruleid: provider-qc-matrix-grouping-must-include-snapshot
    return apply_provider_qc(
        frame,
        config,
        value_columns=("temp_max_c",),
        group_key="forecast_date",
    )


def valid_hourly_grouping(frame, config):
    # ok: provider-qc-matrix-grouping-must-include-snapshot
    return apply_provider_qc(
        frame,
        config,
        value_columns=("temp_c",),
        group_key=["issue_time", "valid_time"],
    )


def valid_daily_grouping(frame, config):
    # ok: provider-qc-matrix-grouping-must-include-snapshot
    return apply_provider_qc(
        frame,
        config,
        value_columns=("temp_max_c",),
        group_key=["issue_time", "forecast_date"],
    )
