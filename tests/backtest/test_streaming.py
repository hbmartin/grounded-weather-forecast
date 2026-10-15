"""Bounded score construction, compatible files, and atomic failure handling."""

import os
import stat
import weakref
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest
from conftest import synthetic_hourly_matrix, write_config

from grounded_weather_forecast import cli
from grounded_weather_forecast.backtest import engine, minutely
from grounded_weather_forecast.backtest import scores as score_module
from grounded_weather_forecast.backtest.engine import (
    BacktestRequest,
    iter_backtest_scores,
    run_backtest,
)
from grounded_weather_forecast.backtest.score_storage import (
    SCORE_WRITE_LOCK,
    cleanup_score_temporaries,
)
from grounded_weather_forecast.backtest.scores import (
    COMPACT_SCORES_SCHEMA,
    SCORE_BATCH_ROWS,
    SCORES_SCHEMA,
    load_scores,
    scores_path,
    stamped_score_batches,
    stamped_scores,
    write_score_batches,
)
from grounded_weather_forecast.contracts import (
    BlendResult,
    daily_variable,
    hourly_variable,
)
from grounded_weather_forecast.dataset import truth
from grounded_weather_forecast.evaluation import EvaluationRun
from grounded_weather_forecast.serve import selection


@pytest.fixture
def evaluation():
    return EvaluationRun(
        evaluation_id="fixed",
        created_at="2026-10-14T00:00:00+00:00",
        dataset_fingerprint="dataset",
        source_kind="live",
        source_set=("alpha",),
        feature_set=("x" * 8449,),
        product="hourly",
        window="expanding",
        semantics={"temp_c": "inst"},
        methods=("equal_weight",),
        code_version="fixed-code",
        config_fingerprint="fixed-config",
    )


def predictions(count, evaluation, **overrides):
    moment = datetime(2026, 10, 14, tzinfo=UTC)
    keys = pl.DataFrame(
        {
            "issue_time": pl.repeat(moment, count, eager=True),
            "valid_time": pl.repeat(moment, count, eager=True),
            "lead_hours": np.arange(count, dtype=float),
            "lead_bucket": ["0-24h"] * count,
        }
    )
    arguments = {
        "method_id": "equal_weight",
        "variable": "temp_c",
        "product": "hourly",
        "source_kind": "live",
        "semantics": "inst",
        "window": "expanding",
        "fold_origin": moment,
        "y_pred": np.arange(count, dtype=float),
        "y_true": np.arange(count, dtype=float),
    }
    arguments.update(overrides)
    return keys, evaluation, arguments


def compact_batch(count, evaluation, **overrides):
    keys, evaluation, arguments = predictions(count, evaluation, **overrides)
    return stamped_scores(keys, evaluation, compact=True, **arguments)


def score_outputs(directory):
    return sorted(path for path in directory.iterdir() if path.name != SCORE_WRITE_LOCK)


@pytest.mark.parametrize("count", [0, 1, SCORE_BATCH_ROWS, SCORE_BATCH_ROWS + 1])
def test_boundaries_roundtrip_and_row_group_limit(tmp_path, evaluation, count):
    keys, evaluation, arguments = predictions(count, evaluation)
    observed = []

    def batches():
        for batch in stamped_score_batches(keys, evaluation, **arguments):
            observed.append(batch.height)
            assert batch.schema == COMPACT_SCORES_SCHEMA
            yield batch

    result = write_score_batches(batches(), tmp_path, "hourly", "live", "expanding")
    actual = load_scores(result.path)
    expected = stamped_scores(keys, evaluation, **arguments)
    assert actual.schema == SCORES_SCHEMA
    assert actual.equals(expected)
    assert result.row_count == count
    assert result.evaluation_id == ("fixed" if count else None)
    assert result.path == scores_path(
        tmp_path, "hourly", "live", "expanding", result.evaluation_id
    )
    assert sum(observed) == count
    assert all(size <= SCORE_BATCH_ROWS for size in observed)
    metadata = pq.ParquetFile(result.path).metadata
    assert all(
        metadata.row_group(i).num_rows <= SCORE_BATCH_ROWS
        for i in range(metadata.num_row_groups)
    )
    assert b"ARROW:schema" not in (metadata.metadata or {})


def test_new_categories_quantiles_and_buffering(tmp_path, evaluation):
    first = compact_batch(2, evaluation)
    second = compact_batch(
        2,
        evaluation,
        method_id="new_method",
        variable="new_variable",
        y_pred=np.array([np.nan, 4.0]),
        quantile_levels=(0.1, 0.5, 0.9),
        quantiles=np.array([[np.nan, np.inf, -np.inf], [1.0, 2.0, 3.0]]),
    )
    result = write_score_batches(
        iter([first, second]), tmp_path, "hourly", "live", "expanding"
    )
    actual = load_scores(result.path)
    expected = pl.concat([first, second]).cast(SCORES_SCHEMA)
    assert actual.equals(expected)
    assert actual["quantiles_json"].to_list() == [
        None,
        None,
        "[null, null, null]",
        "[1.0, 2.0, 3.0]",
    ]
    assert actual["y_pred"].null_count() == 1
    assert pq.ParquetFile(result.path).metadata.num_row_groups == 1
    assert load_scores(result.path, compact=True).schema == COMPACT_SCORES_SCHEMA


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("evaluation_id", "other"),
        ("product", "daily"),
        ("source_kind", "synthetic"),
        ("window", "rolling"),
    ],
)
def test_identity_mismatch_aborts_publication(tmp_path, evaluation, column, value):
    first = compact_batch(1, evaluation)
    wrong = first.with_columns(pl.lit(value).cast(pl.Categorical).alias(column))
    with pytest.raises(ValueError, match="identity"):
        write_score_batches(
            iter([first, wrong]), tmp_path, "hourly", "live", "expanding"
        )
    assert not score_outputs(tmp_path)


@pytest.mark.parametrize("bad", ["schema", "mixed", "null_id", "empty_id"])
def test_invalid_first_batch_is_rejected_and_producer_closed(tmp_path, evaluation, bad):
    batch = compact_batch(2, evaluation)
    if bad == "schema":
        batch = batch.cast(SCORES_SCHEMA)
    elif bad == "mixed":
        batch = batch.with_columns(
            pl.Series("evaluation_id", ["one", "two"]).cast(pl.Categorical)
        )
    else:
        batch = batch.with_columns(
            pl.lit(None if bad == "null_id" else "")
            .cast(pl.Categorical)
            .alias("evaluation_id")
        )
    closed = []

    def producer():
        try:
            yield batch
        finally:
            closed.append(True)

    with pytest.raises(ValueError, match="score batch"):
        write_score_batches(producer(), tmp_path, "hourly", "live", "expanding")
    assert closed == [True]
    assert not score_outputs(tmp_path)


def test_empty_compact_batches_are_valid(tmp_path):
    result = write_score_batches(
        iter([pl.DataFrame(schema=COMPACT_SCORES_SCHEMA)]),
        tmp_path,
        "hourly",
        "live",
        "rolling",
    )
    assert result.row_count == 0
    assert load_scores(result.path).schema == SCORES_SCHEMA


@pytest.mark.parametrize(
    "phase", ["iterate", "construct", "write", "close", "replace", "interrupt"]
)
def test_failures_preserve_destination_and_close_producer(
    tmp_path, evaluation, monkeypatch, phase
):
    monkeypatch.setattr(score_module, "SCORE_BATCH_ROWS", 2)
    batch = compact_batch(2, evaluation)
    destination = scores_path(tmp_path, "hourly", "live", "expanding", "fixed")
    destination.write_bytes(b"original")
    closed = []
    writer_class = pq.ParquetWriter

    class Writer:
        def __init__(self, *args, **kwargs):
            if phase == "construct":
                raise OSError("construct failure")
            self.writer = writer_class(*args, **kwargs)

        def write_table(self, *args, **kwargs):
            assert destination.read_bytes() == b"original"
            if phase == "write":
                raise OSError("write failure")
            self.writer.write_table(*args, **kwargs)

        def close(self):
            self.writer.close()
            if phase == "close":
                raise OSError("close failure")

    monkeypatch.setattr(pq, "ParquetWriter", Writer)
    real_replace = Path.replace

    def replace_file(path, target):
        if phase == "replace":
            raise OSError("replace failure")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace_file)

    def producer():
        try:
            yield batch
            if phase == "iterate":
                raise OSError("iterate failure")
            if phase == "interrupt":
                raise KeyboardInterrupt
        finally:
            closed.append(True)

    expected = KeyboardInterrupt if phase == "interrupt" else OSError
    with pytest.raises(expected):
        write_score_batches(producer(), tmp_path, "hourly", "live", "expanding")
    assert closed == [True]
    assert destination.read_bytes() == b"original"
    assert score_outputs(tmp_path) == [destination]


def test_producer_close_failure_prevents_publication(tmp_path, evaluation):
    class Producer:
        def __init__(self):
            self.frames = iter([compact_batch(1, evaluation)])

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.frames)

        def close(self):
            raise OSError("producer close failure")

    with pytest.raises(OSError, match="producer close"):
        write_score_batches(Producer(), tmp_path, "hourly", "live", "expanding")
    assert not score_outputs(tmp_path)


def test_secondary_close_error_does_not_mask_primary(tmp_path, evaluation, monkeypatch):
    class Writer:
        def __init__(self, *_args, **_kwargs):
            pass

        def write_table(self, *_args, **_kwargs):
            raise ValueError("primary write error")

        def close(self):
            raise OSError("secondary close error")

    monkeypatch.setattr(pq, "ParquetWriter", Writer)
    with pytest.raises(ValueError, match="primary write") as caught:
        write_score_batches(
            iter([compact_batch(1, evaluation)]),
            tmp_path,
            "hourly",
            "live",
            "expanding",
        )
    assert "secondary close" in caught.value.__notes__[0]
    assert not score_outputs(tmp_path)


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode semantics")
def test_streaming_modes(tmp_path, evaluation):
    previous = os.umask(0o022)
    try:
        result = write_score_batches(
            iter([compact_batch(1, evaluation)]),
            tmp_path,
            "hourly",
            "live",
            "expanding",
        )
    finally:
        os.umask(previous)
    assert stat.S_IMODE(result.path.stat().st_mode) == 0o644
    result.path.chmod(0o640)
    write_score_batches(
        iter([compact_batch(2, evaluation)]), tmp_path, "hourly", "live", "expanding"
    )
    assert stat.S_IMODE(result.path.stat().st_mode) == 0o640
    assert load_scores(result.path).height == 2


@pytest.mark.parametrize("daily", [False, True])
@pytest.mark.parametrize("window", ["expanding", "rolling"])
def test_engine_stream_matches_materialized_api(tmp_path, monkeypatch, daily, window):
    config = write_config(
        tmp_path,
        extra_toml="\n[backtest]\ninitial_train_days = 10\nstep_days = 5\nrolling_window_days = 12\n",
    )
    matrix = synthetic_hourly_matrix(days=25, max_lead=6)
    variable = hourly_variable("temp_c")
    if daily:
        variable = daily_variable("temp_max_c")
        matrix = matrix.rename(
            {
                "fx__alpha__temp_c": "fxd__alpha__temp_max_c",
                "fx__beta__temp_c": "fxd__beta__temp_max_c",
                "t__temp_c__inst": "t__temp_max_c",
            }
        ).with_columns(
            pl.col("valid_time").dt.date().alias("forecast_date"),
            pl.lit(1).alias("lead_days"),
        )
    real_create = EvaluationRun.create

    def fixed_create(_cls, *args, **kwargs):
        return replace(
            real_create(*args, **kwargs),
            evaluation_id="fixed",
            created_at="2026-10-14T00:00:00+00:00",
        )

    monkeypatch.setattr(EvaluationRun, "create", classmethod(fixed_create))
    request = BacktestRequest(
        variables=(variable,),
        methods=("equal_weight", "climatology", "emos"),
        daily=daily,
        window=window,
    )
    expected = run_backtest(matrix, request, config)
    stream = iter_backtest_scores(matrix, request, config)
    product = "daily" if daily else "hourly"
    result = write_score_batches(stream, tmp_path / "scores", product, "live", window)
    assert expected.height > 0
    assert load_scores(result.path).equals(expected)
    assert stream.gi_frame is None


def test_cli_releases_all_products_before_next_product_and_promotion(
    tmp_path, evaluation, monkeypatch
):

    config = write_config(tmp_path)
    config.dataset.dir.mkdir()
    for product in ("hourly", "daily"):
        pl.DataFrame({"source_kind": ["live"]}).write_parquet(
            config.dataset.dir / f"{product}_matrix_live.parquet"
        )
    pl.DataFrame({"minute": [1]}).write_parquet(
        config.dataset.dir / "truth_minute.parquet"
    )
    references = []
    events = []
    read_parquet = pl.read_parquet

    def assert_released():
        assert all(reference() is None for reference in references)

    def read(path, *args, **kwargs):
        if Path(path).name.endswith("matrix_live.parquet"):
            assert_released()
        frame = read_parquet(path, *args, **kwargs)
        references.append(weakref.ref(frame))
        return frame

    def produce(_matrix, request, _config):
        product = "daily" if request.daily else "hourly"
        return stream_for(product)

    def stream_for(product):
        def generate():
            try:
                batch = compact_batch(
                    1, replace(evaluation, evaluation_id=product), product=product
                )
                references.append(weakref.ref(batch))
                yield batch
            finally:
                events.append(product)

        stream = generate()
        references.append(weakref.ref(stream))
        return stream

    def promote(*_args, **_kwargs):
        assert_released()
        assert events == ["hourly", "daily", "minutely"]
        return {}

    monkeypatch.setattr(pl, "read_parquet", read)
    monkeypatch.setattr(engine, "iter_backtest_scores", produce)
    monkeypatch.setattr(
        minutely, "iter_minutely_scores", lambda *_args: stream_for("minutely")
    )
    monkeypatch.setattr(truth, "truth_minute_grid", lambda frame, _variables: frame)
    monkeypatch.setattr(selection, "select_methods", promote)
    args = cli.build_parser().parse_args(["backtest", "--methods", "equal_weight"])
    assert cli._cmd_backtest(config, args) == 0
    assert_released()


def test_cli_does_not_promote_failed_stream(tmp_path, monkeypatch):

    config = write_config(tmp_path)
    config.dataset.dir.mkdir()
    pl.DataFrame({"source_kind": ["live"]}).write_parquet(
        config.dataset.dir / "hourly_matrix_live.parquet"
    )
    promoted = []

    def fail(*_args):
        raise OSError("stream failed")
        yield  # pragma: no cover - make this a generator

    monkeypatch.setattr(engine, "iter_backtest_scores", fail)
    monkeypatch.setattr(
        selection, "select_methods", lambda *_args, **_kwargs: promoted.append(True)
    )
    args = cli.build_parser().parse_args(
        ["backtest", "--products", "hourly", "--methods", "equal_weight"]
    )
    with pytest.raises(OSError, match="stream failed"):
        cli._cmd_backtest(config, args)
    assert not promoted


@pytest.mark.parametrize("count", [0, 1, SCORE_BATCH_ROWS, SCORE_BATCH_ROWS + 1])
@pytest.mark.parametrize("column", ["y_pred", "y_true", "quantiles"])
@pytest.mark.parametrize("difference", [-1, 1])
def test_shape_mismatch_fails_before_stamping_or_first_yield(
    tmp_path, evaluation, count, column, difference
):
    keys, evaluation, arguments = predictions(count, evaluation)
    wrong_count = max(0, count + difference) if count else 1
    arguments[column] = (
        np.zeros((wrong_count, 3)) if column == "quantiles" else np.zeros(wrong_count)
    )
    if column == "quantiles":
        arguments["quantile_levels"] = (0.1, 0.5, 0.9)
    with pytest.raises(pl.exceptions.ShapeError, match=f"{column} shape .*expected"):
        stamped_scores(keys, evaluation, **arguments)
    stream = stamped_score_batches(keys, evaluation, **arguments)
    with pytest.raises(pl.exceptions.ShapeError, match=f"{column} shape .*expected"):
        write_score_batches(stream, tmp_path, "hourly", "live", "expanding")
    assert not score_outputs(tmp_path)


@pytest.mark.parametrize("column", ["y_pred", "y_true", "quantiles"])
@pytest.mark.parametrize("shape", [(), (2, 1, 1)])
def test_prediction_dimensions_are_validated(evaluation, column, shape):
    keys, evaluation, arguments = predictions(2, evaluation)
    arguments[column] = np.zeros(shape)
    with pytest.raises(pl.exceptions.ShapeError, match=column):
        next(stamped_score_batches(keys, evaluation, **arguments))


def test_compact_stamping_normalizes_convertible_key_types(tmp_path, evaluation):
    keys, evaluation, arguments = predictions(3, evaluation)
    keys = keys.with_columns(
        pl.col("issue_time").cast(pl.Datetime("ns", "UTC")),
        pl.col("valid_time").cast(pl.Datetime("ms", "UTC")),
        pl.col("lead_hours").cast(pl.Float32),
    )
    expected = stamped_scores(keys, evaluation, **arguments).cast(SCORES_SCHEMA)
    result = write_score_batches(
        stamped_score_batches(keys, evaluation, **arguments),
        tmp_path,
        "hourly",
        "live",
        "expanding",
    )
    assert load_scores(result.path).equals(expected)


@pytest.mark.parametrize("lead_dtype", [pl.Int64, pl.Float32])
def test_engine_stream_accepts_convertible_key_types(tmp_path, monkeypatch, lead_dtype):
    config = write_config(
        tmp_path, extra_toml="\n[backtest]\ninitial_train_days = 10\nstep_days = 5\n"
    )
    matrix = synthetic_hourly_matrix(days=25, max_lead=6).with_columns(
        pl.col("lead_hours").cast(lead_dtype),
        pl.col("valid_time").cast(pl.Datetime("ns", "UTC")),
    )
    real_create = EvaluationRun.create

    def create(*args, **kwargs):
        return replace(
            real_create(*args, **kwargs),
            evaluation_id="fixed",
            created_at="2026-10-14T00:00:00+00:00",
        )

    monkeypatch.setattr(EvaluationRun, "create", create)
    request = BacktestRequest(
        variables=(hourly_variable("temp_c"),), methods=("equal_weight",)
    )
    expected = run_backtest(matrix, request, config)
    result = write_score_batches(
        iter_backtest_scores(matrix, request, config),
        tmp_path / "scores",
        "hourly",
        "live",
        "expanding",
    )
    assert expected.height > 0
    assert load_scores(result.path).equals(expected)


@pytest.mark.parametrize("phase", ["writer", "producer"])
@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_cleanup_control_flow_takes_precedence(
    tmp_path, evaluation, monkeypatch, phase, interrupt
):
    closed = []
    real_writer = pq.ParquetWriter

    class Writer:
        def __init__(self, *args, **kwargs):
            self.writer = real_writer(*args, **kwargs)

        def write_table(self, *_args, **_kwargs):
            raise ValueError("primary write failure")

        def close(self):
            self.writer.close()
            if phase == "writer":
                raise interrupt("cleanup interrupt")

    class Producer:
        def __iter__(self):
            return self

        def __next__(self):
            if closed:
                raise StopIteration
            closed.append("yielded")
            return compact_batch(1, evaluation)

        def close(self):
            closed.append("closed")
            if phase == "producer":
                raise interrupt("cleanup interrupt")

    monkeypatch.setattr(pq, "ParquetWriter", Writer)
    destination = scores_path(tmp_path, "hourly", "live", "expanding", "fixed")
    destination.write_bytes(b"existing")
    with pytest.raises(interrupt, match="cleanup interrupt") as caught:
        write_score_batches(Producer(), tmp_path, "hourly", "live", "expanding")
    assert isinstance(caught.value.__context__, ValueError)
    assert closed == ["yielded", "closed"]
    assert score_outputs(tmp_path) == [destination]
    assert destination.read_bytes() == b"existing"


def test_primary_interrupt_survives_ordinary_cleanup_failure():
    def close():
        raise OSError("cleanup failure")

    with (
        pytest.raises(KeyboardInterrupt) as caught,
        score_module._close_callback(close),
    ):
        raise KeyboardInterrupt
    assert "cleanup failure" in caught.value.__notes__[0]


def test_publication_lock_spans_writer_and_producer_close_and_replace(
    tmp_path, evaluation, monkeypatch
):
    events = []

    def assert_owned(event):
        assert cleanup_score_temporaries(tmp_path).skipped
        events.append(event)

    real_writer = pq.ParquetWriter

    class Writer:
        def __init__(self, *args, **kwargs):
            self.writer = real_writer(*args, **kwargs)

        def write_table(self, *args, **kwargs):
            self.writer.write_table(*args, **kwargs)

        def close(self):
            assert_owned("writer")
            self.writer.close()

    class Producer:
        def __init__(self):
            self.frames = iter([compact_batch(1, evaluation)])

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.frames)

        def close(self):
            assert_owned("producer")

    real_replace = Path.replace

    def replace(path, target):
        assert_owned("replace")
        return real_replace(path, target)

    monkeypatch.setattr(pq, "ParquetWriter", Writer)
    monkeypatch.setattr(Path, "replace", replace)
    write_score_batches(Producer(), tmp_path, "hourly", "live", "expanding")
    assert events == ["writer", "producer", "replace"]
    assert not cleanup_score_temporaries(tmp_path).skipped


@pytest.mark.parametrize("count", [0, 1, SCORE_BATCH_ROWS, SCORE_BATCH_ROWS + 1])
@pytest.mark.parametrize(
    ("levels", "width", "error"),
    [
        (None, 1, ValueError),
        ((), 0, ValueError),
        ((0.5,), 2, pl.exceptions.ShapeError),
        ((0.5, 0.5), 2, ValueError),
        ((0.9, 0.1), 2, ValueError),
        ((0.0,), 1, ValueError),
        ((1.0,), 1, ValueError),
        ((float("nan"),), 1, ValueError),
        ((float("inf"),), 1, ValueError),
    ],
)
def test_quantile_metadata_validated_before_first_yield(
    tmp_path, evaluation, count, levels, width, error
):
    keys, evaluation, arguments = predictions(
        count, evaluation, quantile_levels=levels, quantiles=np.zeros((count, width))
    )
    with pytest.raises(error):
        stamped_scores(keys, evaluation, **arguments)
    with pytest.raises(error):
        write_score_batches(
            stamped_score_batches(keys, evaluation, **arguments),
            tmp_path,
            "hourly",
            "live",
            "expanding",
        )
    assert not score_outputs(tmp_path)


@pytest.mark.parametrize("levels", [None, (), (0.5,)])
def test_point_level_metadata_preserves_null_and_empty_json(evaluation, levels):
    keys, evaluation, arguments = predictions(1, evaluation, quantile_levels=levels)
    if levels:
        with pytest.raises(ValueError, match="require quantiles"):
            stamped_scores(keys, evaluation, **arguments)
        with pytest.raises(ValueError, match="require quantiles"):
            next(stamped_score_batches(keys, evaluation, **arguments))
    else:
        assert stamped_scores(keys, evaluation, **arguments)["quantile_levels_json"][
            0
        ] == ("null" if levels is None else "[]")


@pytest.mark.parametrize("zone", [None, "America/Los_Angeles", "UTC"])
def test_timestamp_timezone_normalization_is_explicit(tmp_path, evaluation, zone):
    keys, evaluation, arguments = predictions(1, evaluation)
    if zone is None:
        keys = keys.with_columns(
            pl.col("issue_time", "valid_time").dt.replace_time_zone(None)
        )
        arguments["fold_origin"] = arguments["fold_origin"].replace(tzinfo=None)
        evaluation = replace(evaluation, created_at="2026-10-14T00:00:00")
    else:
        keys = keys.with_columns(
            pl.col("issue_time", "valid_time").dt.convert_time_zone(zone)
        )
    expected = stamped_scores(keys, evaluation, **arguments).cast(SCORES_SCHEMA)
    result = write_score_batches(
        stamped_score_batches(keys, evaluation, **arguments),
        tmp_path,
        "hourly",
        "live",
        "expanding",
    )
    actual = load_scores(result.path)
    assert actual.equals(expected)
    assert actual["issue_time"][0] == datetime(2026, 10, 14, tzinfo=UTC)


@pytest.mark.parametrize("daily", [False, True])
@pytest.mark.parametrize("all_invalid", [False, True])
def test_engine_filters_nonfinite_truth_before_keys_and_fitting(
    tmp_path, daily, all_invalid
):
    config = write_config(
        tmp_path, extra_toml="\n[backtest]\ninitial_train_days = 10\nstep_days = 5\n"
    )
    matrix = synthetic_hourly_matrix(days=25, max_lead=6)
    column = "t__temp_c__inst"
    variable = hourly_variable("temp_c")
    if daily:
        matrix = matrix.rename(
            {
                "fx__alpha__temp_c": "fxd__alpha__temp_max_c",
                "fx__beta__temp_c": "fxd__beta__temp_max_c",
                column: "t__temp_max_c",
            }
        ).with_columns(
            pl.col("valid_time").dt.date().alias("forecast_date"),
            pl.lit(1).alias("lead_days"),
        )
        column = "t__temp_max_c"
        variable = daily_variable("temp_max_c")
    values = matrix[column].to_numpy().copy()
    for index, invalid in enumerate([np.nan, np.inf, -np.inf]):
        values[index :: (3 if all_invalid else 7)] = invalid
    matrix = matrix.with_columns(pl.Series(column, values))

    class FiniteBlender:
        def fit(self, training):
            assert np.isfinite(training.y).all()
            assert training.y.size
            return self

        def predict(self, x):
            return BlendResult(point=np.zeros(x.values.shape[0]))

    # The truth values themselves identify their matching keys.
    request = BacktestRequest(
        variables=(variable,), methods=("equal_weight",), daily=daily
    )
    frames = list(
        iter_backtest_scores(
            matrix, request, config, factories={"equal_weight": FiniteBlender}
        )
    )
    if all_invalid:
        assert not frames
    else:
        actual = pl.concat(frames).select("issue_time", "valid_time", "y_true")
        finite = matrix.filter(pl.col(column).is_finite())
        expected = (
            engine._lead_and_valid(
                finite, daily=daily, timezone_name=config.station.timezone
            )
            .with_columns(finite[column].alias("y_true"))
            .select("issue_time", "valid_time", "y_true")
        )
        assert actual.height > 0
        assert actual.join(
            expected, on=["issue_time", "valid_time", "y_true"], how="anti"
        ).is_empty()
        assert actual["y_true"].is_finite().all()
