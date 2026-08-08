import polars as pl
from conftest import utc, write_config

from grounded_weather_forecast.dataset.provider_qc import apply_provider_qc

VALID = utc(2026, 7, 13, 12, 0)


def _frame(**columns):
    n = len(next(iter(columns.values())))
    return pl.DataFrame({"valid_time": [VALID] * n, **columns})


class TestAbsoluteBounds:
    def test_out_of_range_nulled(self, tmp_path):
        config = write_config(tmp_path)
        frame = _frame(pressure_sea_hpa=[1013.0, 5000.0])
        out = apply_provider_qc(
            frame, config, value_columns=["pressure_sea_hpa"], group_key="valid_time"
        )
        assert out["pressure_sea_hpa"].to_list() == [1013.0, None]

    def test_in_range_kept(self, tmp_path):
        config = write_config(tmp_path)
        frame = _frame(temp_c=[-20.0, 0.0, 42.0])
        out = apply_provider_qc(
            frame, config, value_columns=["temp_c"], group_key="valid_time"
        )
        assert out["temp_c"].to_list() == [-20.0, 0.0, 42.0]


class TestCrossSourceOutliers:
    def test_gross_pressure_outlier_nulled(self, tmp_path):
        # One provider reports 1074 hPa (in absolute bounds but a robust outlier
        # among peers) — the empirical weatherbit case.
        config = write_config(tmp_path)
        pressures = [1010.0, 1012.0, 1013.0, 1011.0, 1074.0]
        frame = _frame(pressure_sea_hpa=pressures)
        out = apply_provider_qc(
            frame, config, value_columns=["pressure_sea_hpa"], group_key="valid_time"
        )
        assert out["pressure_sea_hpa"].to_list() == [*pressures[:4], None]

    def test_plausible_diversity_preserved(self, tmp_path):
        # A wide-but-plausible spread must NOT be flagged: the blend needs diversity.
        config = write_config(tmp_path)
        temps = [8.0, 10.0, 12.0, 14.0, 16.0]
        frame = _frame(temp_c=temps)
        out = apply_provider_qc(
            frame, config, value_columns=["temp_c"], group_key="valid_time"
        )
        assert out["temp_c"].to_list() == temps

    def test_below_min_sources_not_flagged(self, tmp_path):
        # With fewer than min_sources providers, no cross-source flagging happens
        # (absolute bounds still apply, and 40 C is within them).
        config = write_config(tmp_path)
        frame = _frame(temp_c=[10.0, 11.0, 40.0])
        out = apply_provider_qc(
            frame, config, value_columns=["temp_c"], group_key="valid_time"
        )
        assert out["temp_c"].to_list() == [10.0, 11.0, 40.0]


class TestToggleAndEmpty:
    def test_disabled_passthrough(self, tmp_path):
        config = write_config(tmp_path, extra_toml="[provider_qc]\nenabled = false")
        frame = _frame(pressure_sea_hpa=[1013.0, 9999.0])
        out = apply_provider_qc(
            frame, config, value_columns=["pressure_sea_hpa"], group_key="valid_time"
        )
        assert out["pressure_sea_hpa"].to_list() == [1013.0, 9999.0]

    def test_empty_frame_passthrough(self, tmp_path):
        config = write_config(tmp_path)
        frame = pl.DataFrame(
            {"valid_time": [], "temp_c": []},
            schema={"valid_time": pl.Datetime("us", "UTC"), "temp_c": pl.Float64},
        )
        out = apply_provider_qc(
            frame, config, value_columns=["temp_c"], group_key="valid_time"
        )
        assert out.height == 0


class TestPrecipitationCrossSource:
    def test_lone_cloudburst_on_a_dry_day_nulled(self, tmp_path):
        # The empirical tomorrow_io case (2026-08-06): 138 mm stored for a
        # bone-dry day every other provider forecast at ~0. Median and MAD are
        # zero, so the min_deviation floor (25 mm) carries the whole test.
        config = write_config(tmp_path)
        sums = [0.0, 0.0, 0.1, 0.0, 138.0]
        frame = _frame(precip_sum_mm=sums)
        out = apply_provider_qc(
            frame, config, value_columns=["precip_sum_mm"], group_key="valid_time"
        )
        assert out["precip_sum_mm"].to_list() == [*sums[:4], None]

    def test_drizzle_disagreement_preserved(self, tmp_path):
        # 2.4 mm against a dry median is genuine diversity (the visual_crossing
        # monsoon signal), far under the 25 mm floor.
        config = write_config(tmp_path)
        sums = [0.0, 0.0, 0.0, 0.5, 2.4]
        frame = _frame(precip_sum_mm=sums)
        out = apply_provider_qc(
            frame, config, value_columns=["precip_sum_mm"], group_key="valid_time"
        )
        assert out["precip_sum_mm"].to_list() == sums

    def test_storm_consensus_survives(self, tmp_path):
        # In a genuine atmospheric river every provider forecasts heavy rain;
        # the median rises and the wettest forecast is no longer an outlier.
        config = write_config(tmp_path)
        sums = [35.0, 42.0, 50.0, 55.0, 58.0]
        frame = _frame(precip_sum_mm=sums)
        out = apply_provider_qc(
            frame, config, value_columns=["precip_sum_mm"], group_key="valid_time"
        )
        assert out["precip_sum_mm"].to_list() == sums

    def test_far_lead_pairs_never_flagged(self, tmp_path):
        # Past day six only ~2 providers publish dailies — under min_sources,
        # so even a large disagreement passes untouched.
        config = write_config(tmp_path)
        sums = [0.5, 30.0]
        frame = _frame(precip_sum_mm=sums)
        out = apply_provider_qc(
            frame, config, value_columns=["precip_sum_mm"], group_key="valid_time"
        )
        assert out["precip_sum_mm"].to_list() == sums

    def test_hourly_intensity_uses_its_own_floor(self, tmp_path):
        config = write_config(tmp_path)
        rates = [0.0, 0.0, 0.0, 0.2, 15.0]
        frame = _frame(precip_mm=rates)
        out = apply_provider_qc(
            frame, config, value_columns=["precip_mm"], group_key="valid_time"
        )
        assert out["precip_mm"].to_list() == [*rates[:4], None]
