import numpy as np
import polars as pl
import pytest

from grounded_weather_forecast.contracts import (
    BlendResult,
    ContractViolationError,
    ForecastMatrix,
    Product,
    SourceKind,
    SupervisedSlice,
    TruthSemantics,
    age_col,
    daily_variable,
    fx_col,
    fxd_col,
    hourly_variable,
    is_truth_col,
    obs_col,
    parse_fx_col,
    provider_age_hours,
    provider_age_is_fresh,
    truth_col,
)


def make_matrix(n=4, sources=("open_meteo", "nws")):
    values = np.arange(n * len(sources), dtype=np.float64).reshape(n, len(sources))
    values[0, 1] = np.nan
    return ForecastMatrix.build(
        sources=tuple(sources),
        values=values,
        lead_hours=np.linspace(1.0, 24.0, n),
        features=pl.DataFrame({"valid_hour_local": list(range(n))}),
    )


class TestColumnBuilders:
    def test_fx_round_trip(self):
        assert fx_col("open_meteo", "temp_c") == "fx__open_meteo__temp_c"
        assert parse_fx_col("fx__open_meteo__temp_c") == ("open_meteo", "temp_c")

    def test_fxd_round_trip(self):
        assert parse_fx_col(fxd_col("nws", "temp_max_c")) == ("nws", "temp_max_c")

    def test_reserved_separator_round_trip(self):
        source = "provider__model%experimental"
        variable = "temp__c"
        encoded = fx_col(source, variable)
        assert encoded == "fx__provider%5F%5Fmodel%25experimental__temp%5F%5Fc"
        assert parse_fx_col(encoded) == (source, variable)

    def test_parse_rejects_non_forecast(self):
        for bad in ("t__temp_c__inst", "obs__temp_c", "fx__only", "fx____", "plain"):
            with pytest.raises(ValueError, match="not a forecast column"):
                parse_fx_col(bad)

    def test_truth_col_semantics(self):
        assert truth_col("temp_c") == "t__temp_c"
        assert truth_col("temp_c", TruthSemantics.INSTANTANEOUS) == "t__temp_c__inst"
        assert truth_col("temp_c", TruthSemantics.INTERVAL_MEAN) == "t__temp_c__mean"

    def test_is_truth_col(self):
        assert is_truth_col("t__temp_c__inst")
        assert not is_truth_col("fx__nws__temp_c")
        assert not is_truth_col(obs_col("temp_c"))
        assert not is_truth_col(age_col("nws"))


class TestVariableLookup:
    def test_hourly_lookup(self):
        assert hourly_variable("temp_c").unit == "°C"
        assert hourly_variable("pop").kind.value == "probability"

    def test_daily_lookup(self):
        assert daily_variable("temp_max_c").name == "temp_max_c"

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            hourly_variable("nope")
        with pytest.raises(KeyError):
            daily_variable("temp_c" + "x")


class TestProviderAge:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, 0.0),
            (1.5, 1.5),
            (True, None),
            (None, None),
            (float("nan"), None),
            (float("inf"), None),
            (float("-inf"), None),
        ],
    )
    def test_normalizes_only_finite_numbers(self, value, expected):
        assert provider_age_hours(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (11.999, True),
            (12.0, False),
            (12.001, False),
            (None, False),
            (True, False),
            (float("nan"), False),
            (float("inf"), False),
        ],
    )
    def test_freshness_is_strict_and_finite(self, value, expected):
        assert provider_age_is_fresh(value, 12.0) is expected


class TestForecastMatrix:
    def test_build_derives_availability(self):
        m = make_matrix()
        assert not m.availability[0, 1]
        assert m.availability.sum() == m.values.size - 1
        assert m.n_rows == 4
        assert m.product is Product.HOURLY

    def test_source_count_mismatch(self):
        with pytest.raises(ContractViolationError, match="columns for"):
            ForecastMatrix.build(
                sources=("a",),
                values=np.zeros((2, 2)),
                lead_hours=np.zeros(2),
                features=pl.DataFrame({"f": [0, 1]}),
            )

    def test_lead_shape_mismatch(self):
        with pytest.raises(ContractViolationError, match="lead_hours"):
            ForecastMatrix.build(
                sources=("a", "b"),
                values=np.zeros((2, 2)),
                lead_hours=np.zeros(3),
                features=pl.DataFrame({"f": [0, 1]}),
            )

    def test_features_row_mismatch(self):
        with pytest.raises(ContractViolationError, match="features"):
            ForecastMatrix.build(
                sources=("a", "b"),
                values=np.zeros((2, 2)),
                lead_hours=np.zeros(2),
                features=pl.DataFrame({"f": [0]}),
            )

    def test_truth_leak_rejected(self):
        with pytest.raises(ContractViolationError, match="leaked"):
            ForecastMatrix.build(
                sources=("a", "b"),
                values=np.zeros((2, 2)),
                lead_hours=np.zeros(2),
                features=pl.DataFrame({"t__temp_c__inst": [0.0, 1.0]}),
            )


class TestSupervisedSlice:
    def test_valid(self):
        m = make_matrix()
        s = SupervisedSlice(
            x=m,
            y=np.ones(4),
            variable=hourly_variable("temp_c"),
            source_kind=SourceKind.LIVE,
        )
        assert s.source_kind is SourceKind.LIVE

    def test_nan_truth_rejected(self):
        m = make_matrix()
        y = np.ones(4)
        y[2] = np.nan
        with pytest.raises(ContractViolationError, match="NaN"):
            SupervisedSlice(
                x=m,
                y=y,
                variable=hourly_variable("temp_c"),
                source_kind=SourceKind.LIVE,
            )

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ContractViolationError, match="y shape"):
            SupervisedSlice(
                x=make_matrix(),
                y=np.ones(3),
                variable=hourly_variable("temp_c"),
                source_kind=SourceKind.SYNTHETIC,
            )


class TestBlendResult:
    def test_point_only(self):
        BlendResult(point=np.zeros(3))

    def test_quantiles_with_levels(self):
        BlendResult(
            point=np.zeros(3),
            quantiles=np.zeros((3, 2)),
            quantile_levels=(0.1, 0.9),
        )

    def test_quantiles_without_levels_rejected(self):
        with pytest.raises(ContractViolationError, match="together"):
            BlendResult(point=np.zeros(3), quantiles=np.zeros((3, 2)))

    def test_levels_without_quantiles_rejected(self):
        with pytest.raises(ContractViolationError, match="together"):
            BlendResult(point=np.zeros(3), quantile_levels=(0.5,))

    def test_quantile_shape_mismatch_rejected(self):
        with pytest.raises(ContractViolationError, match="quantiles shape"):
            BlendResult(
                point=np.zeros(3),
                quantiles=np.zeros((3, 3)),
                quantile_levels=(0.1, 0.9),
            )

    def test_duplicate_quantile_levels_rejected(self):
        with pytest.raises(ContractViolationError, match="strictly increasing"):
            BlendResult(
                point=np.zeros(3),
                quantiles=np.zeros((3, 2)),
                quantile_levels=(0.1, 0.1),
            )


class TestFiniteNumber:
    """The single numeric guard for operator-facing evidence."""

    @pytest.mark.parametrize(
        "value", [float("nan"), float("inf"), float("-inf"), None, "1.0", True, False]
    )
    def test_unusable_values_become_none(self, value):
        from grounded_weather_forecast.contracts import finite_number

        assert finite_number(value) is None

    @pytest.mark.parametrize(("value", "expected"), [(1, 1.0), (0, 0.0), (-2.5, -2.5)])
    def test_real_numbers_pass_through(self, value, expected):
        from grounded_weather_forecast.contracts import finite_number

        assert finite_number(value) == expected
