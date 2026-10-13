import json
import math
from datetime import timedelta

import pytest
from conftest import make_forecast_db, make_station_db, minute_series, utc, write_config

from grounded_weather_forecast.cli import main
from grounded_weather_forecast.contracts import TruthSemantics
from grounded_weather_forecast.dataset.matrix import write_dataset
from grounded_weather_forecast.serve.predict import (
    NoForecastDataError,
    UnsupportedMethodError,
    build_snapshot,
    predict,
)
from grounded_weather_forecast.serve.schema import SCHEMA_VERSION
from grounded_weather_forecast.serve.selection import Selection, select_methods

NOW = utc(2026, 3, 22, 17, 0)
FETCH = "2026-03-22T16:30:00+00:00"


@pytest.mark.parametrize("cause", ["cold", "synthetic", "dataset", "config", "code"])
@pytest.mark.parametrize("promoted", [False, True])
def test_rainfall_does_not_mask_evidence_diagnosis(
    config, cause, promoted, capsys, monkeypatch
):
    import polars as pl
    from score_fixtures import scores
    from grounded_weather_forecast.evaluation import (
        code_identity,
        config_fingerprint,
        dataset_fingerprint,
    )
    from grounded_weather_forecast.serve.selection import no_evidence_reason

    directory = config.dataset.dir / "scores"
    if cause != "cold":
        directory.mkdir(exist_ok=True)
        frame = scores().with_columns(
            pl.lit("synthetic" if cause == "synthetic" else "live").alias(
                "source_kind"
            ),
            pl.lit("old" if cause == "dataset" else dataset_fingerprint(config)).alias(
                "dataset_fingerprint"
            ),
            pl.lit("old" if cause == "config" else config_fingerprint(config)).alias(
                "config_fingerprint"
            ),
            pl.lit("old" if cause == "code" else code_identity()).alias("code_version"),
        )
        frame.write_parquet(
            directory
            / f"scores_hourly_{'synthetic' if cause == 'synthetic' else 'live'}_expanding.parquet"
        )
    expected = no_evidence_reason(config, directory)
    selections = (
        {
            ("hourly", "precip_mm", "0-1h"): Selection(
                "equal_weight", "promoted", release_id="rain", truth_semantics="inst"
            )
        }
        if promoted
        else {}
    )
    document = predict(config, selections, now=NOW)
    assert document.status == "degraded"
    assert document.status_reason == expected
    assert any(
        "rainfall fallback:" in p.selection_reasons.get("precip_mm", "")
        for p in document.hourly
    )
    assert any(
        p.methods.get("precip_sum_mm") == "rainfall_accumulation_floor"
        for p in document.daily
    )
    assert (
        main(
            [
                "--config",
                str(config.dataset.dir.parent / "config.toml"),
                "predict",
                "--now",
                NOW.isoformat(),
                "--no-history",
                "--out",
                "-",
            ]
        )
        == 0
    )
    assert expected in capsys.readouterr().err
    if cause == "code":
        _assert_diagnosis_channels(config, document, monkeypatch)


def _assert_diagnosis_channels(config, document, monkeypatch):
    import importlib.util
    import subprocess
    from dataclasses import replace
    from pathlib import Path
    from types import SimpleNamespace
    from grounded_weather_forecast.dashboard.context import DashboardContext
    from grounded_weather_forecast.dashboard.derive import Derived
    from grounded_weather_forecast.dashboard.writer import _alert_inputs
    from grounded_weather_forecast.dashboard.zones.liveness import _serving_status
    from grounded_weather_forecast.reports.alerts import evaluate_alerts
    from grounded_weather_forecast.serve.publish import (
        _recovery_signature,
        publish_document,
        record_publish_attempt,
        schedule_recovery,
    )

    target = config.dataset.dir.parent / "forecast.json"
    target.write_text(replace(document, status="ready", status_reason=None).to_json())
    outcome = publish_document(document, target, config.predict.history_path, now=NOW)
    assert outcome.action == "held_last_good"
    record_publish_attempt(config, document, outcome, target, now=NOW)
    attempt = json.loads(
        (config.artifacts_dir / "latest_publish_attempt.json").read_text()
    )
    assert attempt["status_reason"] == document.status_reason
    ctx = DashboardContext(
        config=config, now=NOW, latest_forecast=document, latest_publish_attempt=attempt
    )
    assert _serving_status(ctx).intro == f"degraded: {document.status_reason}"
    alerts = evaluate_alerts(_alert_inputs(ctx, Derived()))
    assert any(
        document.status_reason in alert.message and "held" in alert.message
        for alert in alerts
    )
    calls = []
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda command, **kwargs: calls.append(command) or SimpleNamespace(pid=123),
    )
    assert schedule_recovery(
        config, config.dataset.dir.parent / "config.toml", document, now=NOW
    )
    state = json.loads((config.artifacts_dir / "auto-restore.json").read_text())
    assert state["signature"] == _recovery_signature(
        config, document.status_reason, "auto", "all", "expanding"
    )
    assert state["reason"] == document.status_reason
    target.write_text(document.to_json())
    script = Path(__file__).parents[2] / "docs/launchd/hourly-predict-publish.py"
    spec = importlib.util.spec_from_file_location("rainfall_restore_regression", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "OUT", str(target))
    monkeypatch.setattr(module, "RESTORE_LOG", target.with_suffix(".log"))
    module.maybe_spawn_restore()
    implementation_changed = "implementation changed" in document.status_reason
    assert len(calls) == 1 + int(implementation_changed)
    if implementation_changed:
        assert calls[-1][-1] == "--restore"


@pytest.mark.parametrize("daily", [False, True])
@pytest.mark.parametrize(
    "removed", [(), (None, None, None), ("", None, ""), ("removed", None, "removed")]
)
def test_only_finite_guarded_rows_record_attribution_loss(daily, removed):
    from dataclasses import replace
    import numpy as np
    from grounded_weather_forecast.contracts import daily_variable, hourly_variable
    from grounded_weather_forecast.serve.predict import (
        VariableBlend,
        _point_fields,
        _record_rainfall_guards,
    )
    from grounded_weather_forecast.serve.rainfall import RainfallEffects
    from grounded_weather_forecast.serve.schema import DailyPoint, HourlyPoint

    blend = VariableBlend(
        point=np.array([np.nan, 1.0, 1.0]),
        methods=["equal_weight"] * 3,
        reasons=["guarded"] * 3,
        release_ids=[None] * 3,
        quantiles=[{}] * 3,
        guarded_release_ids=removed,
    )
    name = "precip_sum_mm" if daily else "precip_mm"
    spec = daily_variable(name) if daily else hourly_variable(name)
    unrelated = "temp_max_c" if daily else "temp_c"
    unrelated_spec = daily_variable(unrelated) if daily else hourly_variable(unrelated)
    blended = {name: blend, unrelated: replace(blend, guarded_release_ids=())}
    variables = {name: spec, unrelated: unrelated_spec}
    points = [
        DailyPoint(str(NOW.date()), 0, **_point_fields(blended, variables, i))
        if daily
        else HourlyPoint(NOW.isoformat(), 0.0, **_point_fields(blended, variables, i))
        for i in range(3)
    ]
    effects = RainfallEffects()
    _record_rainfall_guards(blended, points, None)
    _record_rainfall_guards(blended, points[:2], effects)
    assert not effects.guarded_release_removed
    _record_rainfall_guards(blended, points, effects)
    assert effects.guarded_release_removed == bool(removed and removed[2] is not None)


@pytest.mark.parametrize("daily", [False, True])
@pytest.mark.parametrize("source", [None, "dressed_bucket", "recalibrated_pit_bucket"])
@pytest.mark.parametrize(
    "raw", [None, float("nan"), float("inf"), -float("inf"), 0.0, 1.25]
)
def test_release_attribution_requires_finite_emitted_values(daily, source, raw):
    import numpy as np
    from grounded_weather_forecast.contracts import daily_variable, hourly_variable
    from grounded_weather_forecast.serve.predict import (
        VariableBlend,
        _point_fields,
        _served_release_ids,
    )
    from grounded_weather_forecast.serve.schema import DailyPoint, HourlyPoint

    name = "precip_sum_mm" if daily else "precip_mm"
    spec = daily_variable(name) if daily else hourly_variable(name)
    blend = VariableBlend(
        point=np.array([raw], dtype=float),
        methods=["equal_weight"],
        reasons=["promoted"],
        release_ids=["candidate"],
        quantiles=[{"0.1": -1.0, "0.9": 3.0}],
        quantile_sources=(source,),
    )
    fields = _point_fields({name: blend}, {name: spec}, 0)
    point = (
        DailyPoint(str(NOW.date()), 0, **fields)
        if daily
        else HourlyPoint(NOW.isoformat(), 0.0, **fields)
    )
    expected = ["candidate"] if raw is not None and math.isfinite(raw) else []
    assert point.release_ids == ({name: "candidate"} if expected else {})
    assert (point.values[name] is not None) == bool(expected)
    assert point.quantiles == ({name: {"0.1": 0.0, "0.9": 3.0}} if expected else {})
    assert point.quantiles_source == ({name: source} if expected and source else {})
    assert (
        _served_release_ids([] if daily else [point], [point] if daily else [])
        == expected
    )


@pytest.mark.parametrize(
    "raw_quantiles", [{}, {"0.1": float("nan"), "0.9": float("inf")}]
)
def test_uncertainty_source_requires_usable_quantiles(raw_quantiles):
    import numpy as np
    from grounded_weather_forecast.contracts import hourly_variable
    from grounded_weather_forecast.serve.predict import VariableBlend, _point_fields

    blend = VariableBlend(
        point=np.array([1.0]),
        methods=["equal_weight"],
        reasons=["promoted"],
        release_ids=["release"],
        quantiles=[raw_quantiles],
        quantile_sources=("recalibrated_pit_bucket",),
    )
    fields = _point_fields(
        {"precip_mm": blend}, {"precip_mm": hourly_variable("precip_mm")}, 0
    )
    assert fields["values"] == {"precip_mm": 1.0}
    assert fields["release_ids"] == {"precip_mm": "release"}
    assert fields["quantiles"] == fields["quantiles_source"] == {}


def compatible_promotion(config, product, name, method, *, dataset_id=None):
    import polars as pl
    from score_fixtures import scores
    from grounded_weather_forecast.evaluation import (
        ModelRelease,
        code_identity,
        config_fingerprint,
        dataset_fingerprint,
    )

    dataset_id = dataset_id or dataset_fingerprint(config)
    configuration = config_fingerprint(config)
    bucket = "D1" if product == "daily" else "0-1h"
    release = ModelRelease.create(
        dataset=dataset_id,
        configuration=configuration,
        evaluation_ids=("eval",),
        evaluation_contexts=(),
        training_cutoff=NOW,
        selections={
            f"{product}.{name}.{bucket}": {
                "method_id": method,
                "reason": "promoted",
                "truth_semantics": "inst",
            }
        },
    )
    release.write(config.artifacts_dir / "releases")
    directory = config.dataset.dir / "scores"
    directory.mkdir(exist_ok=True)
    scores(methods=(method,)).with_columns(
        pl.lit(product).alias("product"),
        pl.lit(name).alias("variable"),
        pl.lit(bucket).alias("lead_bucket"),
        pl.lit(dataset_id).alias("dataset_fingerprint"),
        pl.lit(configuration).alias("config_fingerprint"),
        pl.lit(code_identity()).alias("code_version"),
    ).write_parquet(directory / f"scores_{product}_live_expanding.parquet")
    return {
        (product, name, bucket): Selection(
            method,
            "promoted",
            release_id=release.release_id,
            truth_semantics="inst",
        )
    }


@pytest.mark.parametrize("daily", [False, True])
@pytest.mark.parametrize("rain", [False, True])
@pytest.mark.parametrize("method", ["equal_weight", "best_provider"])
def test_compatible_promotion_without_training_names_fitting_failure(
    config_no_provider_qc, monkeypatch, capsys, daily, rain, method
):
    import grounded_weather_forecast.serve.predict as module
    from grounded_weather_forecast.contracts import daily_variable, hourly_variable

    config = config_no_provider_qc
    product = "daily" if daily else "hourly"
    name = (
        ("precip_sum_mm" if daily else "precip_mm")
        if rain
        else ("temp_max_c" if daily else "temp_c")
    )
    selections = compatible_promotion(config, product, name, method)
    training = module._training_matrix(config, product, NOW)
    assert training.is_empty()
    snapshot = build_snapshot(config, NOW)
    assert (
        module._fit_methods(
            training,
            snapshot.daily if daily else snapshot.hourly,
            daily_variable(name) if daily else hourly_variable(name),
            {method},
            daily=daily,
            semantics=TruthSemantics.INSTANTANEOUS,
        )
        is None
    )
    document = predict(config, selections, now=NOW)
    assert document.status == "degraded"
    assert document.release_ids == []
    expected_cause = (
        "no finite training truth; wait for valid truth and rebuild once it is incorporated"
        if rain
        else (
            "training truth on already-issued rows is not yet available at the requested issuance; "
            "keep polling as live issuance advances; repeating a fixed historical --now "
            "cannot admit future evidence"
        )
    )
    assert document.status_reason == (
        "some compatible promoted selections could not be fitted at issuance: "
        f"{product}.{name} (inst: {expected_cause})"
    )
    if rain:
        point = document.daily[0] if daily else document.hourly[0]
        assert "0 wet local days" in point.selection_reasons[name]
    elif method == "best_provider":
        assert name not in (document.daily[0] if daily else document.hourly[0]).values
    monkeypatch.setattr(module, "predict", lambda *_args, **_kwargs: document)
    assert (
        main(
            [
                "--config",
                str(config.dataset.dir.parent / "config.toml"),
                "predict",
                "--now",
                NOW.isoformat(),
                "--no-history",
                "--out",
                "-",
            ]
        )
        == 0
    )
    assert document.status_reason in capsys.readouterr().err
    if rain and not daily and method == "equal_weight":
        _assert_diagnosis_channels(config, document, monkeypatch)


def scoreable_rain_training(*, daily=False, wet_days=0):
    import polars as pl
    from grounded_weather_forecast.contracts import DAILY_VARIABLES, HOURLY_VARIABLES

    dates = [NOW - timedelta(days=i) for i in range(1, 7)]
    data = {
        "issue_time": [day - timedelta(hours=1) for day in dates],
        "source_kind": ["live"] * len(dates),
    }
    if daily:
        data.update(
            forecast_date=[day.date() for day in dates], lead_days=[0] * len(dates)
        )
    else:
        data.update(
            valid_time=dates,
            lead_hours=[0.5] * len(dates),
            lead_bucket=["0-1h"] * len(dates),
        )
    for variable in DAILY_VARIABLES if daily else HOURLY_VARIABLES:
        values = (
            [float(i < wet_days) for i in range(len(dates))]
            if variable.name == "pop"
            else [0.0] * len(dates)
        )
        semantics = ("inst", "mean") if variable.has_dual_semantics else (None,)
        for meaning in semantics:
            name = f"t__{variable.name}" + (f"__{meaning}" if meaning else "")
            data[name] = values
        for source in ("nws", "open_meteo"):
            data[f"{'fxd' if daily else 'fx'}__{source}__{variable.name}"] = values
    return pl.DataFrame(data)


@pytest.mark.parametrize("invalid", ["method", "semantics"])
def test_unavailable_fit_diagnostic_uses_validated_selection(config, invalid):
    import polars as pl
    import grounded_weather_forecast.serve.predict as module
    from grounded_weather_forecast.contracts import hourly_variable

    diagnostics = module.FitDiagnostics()
    selection = Selection(
        "retired_method" if invalid == "method" else "equal_weight",
        "promoted",
        release_id="ineligible",
        truth_semantics="mean" if invalid == "semantics" else "inst",
    )
    result = module._blend_variable(
        pl.DataFrame(),
        build_snapshot(config, NOW).hourly,
        hourly_variable("temp_c"),
        {("hourly", "temp_c", "0-1h"): selection},
        config,
        daily=False,
        semantics=TruthSemantics.INSTANTANEOUS,
        fit_diagnostics=diagnostics,
    )
    assert result is not None
    assert not diagnostics.promoted_fit_unavailable
    assert not any(result.release_ids)


@pytest.mark.parametrize("daily", [False, True])
def test_causal_fit_becomes_available_without_rebuilding(config_no_provider_qc, daily):
    import polars as pl
    import grounded_weather_forecast.serve.predict as module
    from grounded_weather_forecast.backtest.splits import (
        daily_truth_known_at,
        hourly_truth_known_at,
    )
    from grounded_weather_forecast.contracts import daily_variable, hourly_variable
    from grounded_weather_forecast.serve.diagnostics import (
        FitDiagnostics,
        FitUnavailableCause,
    )

    config = config_no_provider_qc
    product = "daily" if daily else "hourly"
    name = "temp_max_c" if daily else "temp_c"
    variable = daily_variable(name) if daily else hourly_variable(name)
    raw = (
        scoreable_rain_training(daily=daily)
        .head(1)
        .with_columns(
            pl.lit(NOW - timedelta(hours=1)).alias("issue_time"),
            pl.lit(NOW.date()).alias("forecast_date")
            if daily
            else pl.lit(NOW).alias("valid_time"),
        )
    )
    known = (
        daily_truth_known_at(raw, config.station.timezone)
        if daily
        else hourly_truth_known_at(raw)
    ).item()
    original = raw.serialize()
    selections = compatible_promotion(config, product, name, "equal_weight")
    snapshot = build_snapshot(config, NOW)
    future = snapshot.daily if daily else snapshot.hourly
    first, later = FitDiagnostics(), FitDiagnostics()
    excluded = module._training_matrix(
        config,
        product,
        known - timedelta(microseconds=1),
        frame=raw,
        fit_diagnostics=first,
    )
    eligible = module._training_matrix(
        config, product, known, frame=raw, fit_diagnostics=later
    )
    assert excluded.is_empty()
    assert eligible.height == 1
    early_blend = module._blend_variable(
        excluded,
        future,
        variable,
        selections,
        config,
        daily=daily,
        semantics=TruthSemantics.INSTANTANEOUS,
        fit_diagnostics=first,
    )
    later_blend = module._blend_variable(
        eligible,
        future,
        variable,
        selections,
        config,
        daily=daily,
        semantics=TruthSemantics.INSTANTANEOUS,
        fit_diagnostics=later,
    )
    assert early_blend is not None and all(
        value is None for value in early_blend.release_ids
    )
    assert later_blend is not None and any(
        value is not None for value in later_blend.release_ids
    )
    first.resolve_training(product, raw, known - timedelta(microseconds=1))
    later.resolve_training(product, raw, known)
    assert first.failures[0].cause == FitUnavailableCause.CAUSAL_EXCLUSION
    assert not later.failures
    assert raw.serialize() == original


@pytest.mark.parametrize(
    "cause", ["empty_matrix", "no_forecast_columns", "no_finite_truth"]
)
def test_unavailable_fit_classifies_observed_training_facts(
    config_no_provider_qc, cause
):
    import polars as pl
    import grounded_weather_forecast.serve.predict as module
    from grounded_weather_forecast.contracts import hourly_variable
    from grounded_weather_forecast.serve.diagnostics import FitDiagnostics

    config = config_no_provider_qc
    raw = scoreable_rain_training()
    if cause == "empty_matrix":
        raw = raw.head(0)
    elif cause == "no_forecast_columns":
        raw = raw.drop(name for name in raw.columns if name.startswith("fx__"))
    else:
        raw = raw.with_columns(
            pl.Series(
                "t__temp_c__inst",
                [None, float("nan"), float("inf"), -float("inf"), None, None],
            )
        )
    diagnostics = FitDiagnostics()
    training = module._training_matrix(
        config, "hourly", NOW, frame=raw, fit_diagnostics=diagnostics
    )
    selections = compatible_promotion(config, "hourly", "temp_c", "equal_weight")
    module._blend_variable(
        training,
        build_snapshot(config, NOW).hourly,
        hourly_variable("temp_c"),
        selections,
        config,
        daily=False,
        semantics=TruthSemantics.INSTANTANEOUS,
        fit_diagnostics=diagnostics,
    )
    diagnostics.resolve_training("hourly", raw, NOW)
    assert [
        (
            failure.product,
            failure.variable,
            failure.truth_semantics,
            failure.cause.value,
        )
        for failure in diagnostics.failures
    ] == [("hourly", "temp_c", "inst", cause)]


@pytest.mark.parametrize("malformed", ["missing_column", "wrong_type"])
def test_fit_diagnostics_preserve_malformed_training_exceptions(config, malformed):
    import polars as pl
    import grounded_weather_forecast.serve.predict as module
    from grounded_weather_forecast.contracts import hourly_variable
    from grounded_weather_forecast.serve.diagnostics import FitDiagnostics

    raw = scoreable_rain_training()
    raw = (
        raw.drop("t__temp_c__inst")
        if malformed == "missing_column"
        else raw.with_columns(pl.lit("bad").alias("t__temp_c__inst"))
    )
    diagnostics = FitDiagnostics()
    training = module._training_matrix(
        config, "hourly", NOW, frame=raw, fit_diagnostics=diagnostics
    )
    with pytest.raises(pl.exceptions.PolarsError):
        module._blend_variable(
            training,
            build_snapshot(config, NOW).hourly,
            hourly_variable("temp_c"),
            {},
            config,
            daily=False,
            semantics=TruthSemantics.INSTANTANEOUS,
            fit_diagnostics=diagnostics,
        )
    assert not diagnostics.failures


def test_training_diagnostics_load_once_and_release_frames(config, monkeypatch):
    import gc
    import weakref
    import polars as pl
    import grounded_weather_forecast.serve.predict as module
    from grounded_weather_forecast.dataset.matrix import matrix_path
    from grounded_weather_forecast.serve.diagnostics import FitDiagnostics

    path = matrix_path(config.dataset.dir, "hourly", "live")
    scoreable_rain_training().write_parquet(path)
    real_read = pl.read_parquet
    reads, references = [], []

    def read_once(path):
        frame = real_read(path)
        reads.append(path)
        references.append(weakref.ref(frame))
        return frame

    monkeypatch.setattr(module.pl, "read_parquet", read_once)
    diagnostics = FitDiagnostics()
    training = module._training_matrix(
        config, "hourly", NOW, fit_diagnostics=diagnostics
    )
    references.append(weakref.ref(training))
    del training
    gc.collect()
    assert reads == [path]
    assert all(reference() is None for reference in references)
    assert diagnostics.training["hourly"].raw_rows == 6


def test_missing_training_matrix_records_absence_without_reading(config, monkeypatch):
    import grounded_weather_forecast.serve.predict as module
    from grounded_weather_forecast.dataset.matrix import matrix_path
    from grounded_weather_forecast.serve.diagnostics import FitDiagnostics

    matrix_path(config.dataset.dir, "hourly", "live").unlink()
    monkeypatch.setattr(
        module.pl, "read_parquet", lambda *_args: pytest.fail("missing matrix was read")
    )
    diagnostics = FitDiagnostics()
    assert module._training_matrix(
        config, "hourly", NOW, fit_diagnostics=diagnostics
    ).is_empty()
    assert diagnostics.training["hourly"].raw_rows == 0


def test_training_cause_is_specific_to_truth_semantics(config_no_provider_qc):
    import polars as pl
    import grounded_weather_forecast.serve.predict as module
    from grounded_weather_forecast.contracts import hourly_variable
    from grounded_weather_forecast.serve.diagnostics import (
        FitDiagnostics,
        FitUnavailableCause,
    )

    config = config_no_provider_qc
    raw = (
        scoreable_rain_training()
        .head(2)
        .with_columns(
            pl.Series("valid_time", [NOW - timedelta(days=1), NOW]),
            pl.Series("t__temp_c__inst", [None, 10.0]),
            pl.Series("t__temp_c__mean", [5.0, None]),
        )
    )
    diagnostics = FitDiagnostics()
    training = module._training_matrix(
        config, "hourly", NOW, frame=raw, fit_diagnostics=diagnostics
    )
    snapshot = build_snapshot(config, NOW)
    for semantics in (TruthSemantics.INSTANTANEOUS, TruthSemantics.INTERVAL_MEAN):
        selections = {
            ("hourly", "temp_c", "0-1h"): Selection(
                "equal_weight",
                "promoted",
                release_id="semantic-release",
                truth_semantics=semantics.value,
            )
        }
        module._blend_variable(
            training,
            snapshot.hourly,
            hourly_variable("temp_c"),
            selections,
            config,
            daily=False,
            semantics=semantics,
            fit_diagnostics=diagnostics,
        )
    diagnostics.resolve_training("hourly", raw, NOW)
    assert [
        (failure.truth_semantics, failure.cause) for failure in diagnostics.failures
    ] == [("inst", FitUnavailableCause.CAUSAL_EXCLUSION)]


def test_boolean_fit_diagnostic_compatibility_uses_conditional_advice(config):
    from grounded_weather_forecast.serve.selection import no_evidence_reason

    compatible_promotion(config, "hourly", "temp_c", "equal_weight")
    reason = no_evidence_reason(
        config, config.dataset.dir / "scores", promoted_fit_unavailable=True
    )
    assert reason.startswith("at least one compatible promoted selection")
    assert "rebuild only when" in reason


@pytest.mark.parametrize("daily", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_training_cutoffs_distinguish_later_issuance_and_pending_truth(
    config_no_provider_qc, daily, mixed
):
    import polars as pl
    import grounded_weather_forecast.serve.predict as module
    from grounded_weather_forecast.contracts import daily_variable, hourly_variable
    from grounded_weather_forecast.serve.diagnostics import (
        FitDiagnostics,
        FitUnavailableCause,
        describe_fit_failures,
    )

    config = config_no_provider_qc
    product = "daily" if daily else "hourly"
    name = "temp_max_c" if daily else "temp_c"
    raw = (
        scoreable_rain_training(daily=daily)
        .head(2 if mixed else 1)
        .with_columns(
            pl.Series(
                "issue_time",
                [NOW + timedelta(hours=1), NOW - timedelta(hours=1)]
                if mixed
                else [NOW + timedelta(hours=1)],
            ),
            pl.lit(NOW.date()).alias("forecast_date")
            if daily
            else pl.lit(NOW).alias("valid_time"),
        )
    )
    diagnostics = FitDiagnostics()
    train = module._training_matrix(
        config, product, NOW, frame=raw, fit_diagnostics=diagnostics
    )
    assert train.is_empty()
    snapshot = build_snapshot(config, NOW)
    module._blend_variable(
        train,
        snapshot.daily if daily else snapshot.hourly,
        daily_variable(name) if daily else hourly_variable(name),
        compatible_promotion(config, product, name, "equal_weight"),
        config,
        daily=daily,
        semantics=TruthSemantics.INSTANTANEOUS,
        fit_diagnostics=diagnostics,
    )
    diagnostics.resolve_training(product, raw, NOW)
    assert diagnostics.failures[0].cause == (
        FitUnavailableCause.CAUSAL_EXCLUSION
        if mixed
        else FitUnavailableCause.LATER_ISSUANCE
    )
    reason = describe_fit_failures(diagnostics.failures)
    assert "fixed historical --now" in reason
    assert ("later-issued archive rows" in reason) == (not mixed)
    assert module._training_matrix(config, product, NOW, frame=raw).is_empty()


def test_unavailable_fit_without_authoritative_branch_is_unknown(config, monkeypatch):
    import polars as pl
    import grounded_weather_forecast.serve.predict as module
    from grounded_weather_forecast.contracts import hourly_variable
    from grounded_weather_forecast.serve.diagnostics import (
        FitDiagnostics,
        FitUnavailableCause,
    )

    diagnostics = FitDiagnostics()
    train = module._training_matrix(
        config,
        "hourly",
        NOW,
        frame=scoreable_rain_training(),
        fit_diagnostics=diagnostics,
    )
    monkeypatch.setattr(module, "_fit_methods", lambda *_args, **_kwargs: None)
    module._blend_variable(
        train,
        build_snapshot(config, NOW).hourly,
        hourly_variable("temp_c"),
        compatible_promotion(config, "hourly", "temp_c", "equal_weight"),
        config,
        daily=False,
        semantics=TruthSemantics.INSTANTANEOUS,
        fit_diagnostics=diagnostics,
    )
    assert diagnostics.failures[0].cause == FitUnavailableCause.UNKNOWN_HISTORY


def test_successful_fits_do_not_aggregate_diagnostic_truth(
    config_no_provider_qc, monkeypatch
):
    import grounded_weather_forecast.serve.diagnostics as diagnostics_module
    import grounded_weather_forecast.serve.predict as module
    from grounded_weather_forecast.contracts import hourly_variable

    config = config_no_provider_qc
    raw = scoreable_rain_training()
    diagnostics = diagnostics_module.FitDiagnostics()
    monkeypatch.setattr(
        diagnostics_module,
        "_truth_availability",
        lambda *_args: pytest.fail("successful fits requested diagnostic scans"),
    )
    train = module._training_matrix(
        config, "hourly", NOW, frame=raw, fit_diagnostics=diagnostics
    )
    result = module._blend_variable(
        train,
        build_snapshot(config, NOW).hourly,
        hourly_variable("temp_c"),
        compatible_promotion(config, "hourly", "temp_c", "equal_weight"),
        config,
        daily=False,
        semantics=TruthSemantics.INSTANTANEOUS,
        fit_diagnostics=diagnostics,
    )
    assert result is not None and any(result.release_ids)
    diagnostics.resolve_training("hourly", raw, NOW)
    assert not diagnostics.failures


def test_failed_fits_scan_only_required_columns_once(
    config_no_provider_qc, monkeypatch
):
    import polars as pl
    import grounded_weather_forecast.serve.diagnostics as diagnostics_module
    import grounded_weather_forecast.serve.predict as module
    from grounded_weather_forecast.contracts import hourly_variable

    config = config_no_provider_qc
    raw = scoreable_rain_training().with_columns(pl.lit(NOW).alias("valid_time"))
    diagnostics = diagnostics_module.FitDiagnostics()
    real = diagnostics_module._truth_availability
    calls = []

    def count(raw, columns, issue):
        calls.append(tuple(columns))
        return real(raw, columns, issue)

    monkeypatch.setattr(diagnostics_module, "_truth_availability", count)
    train = module._training_matrix(
        config, "hourly", NOW, frame=raw, fit_diagnostics=diagnostics
    )
    assert not calls
    snapshot = build_snapshot(config, NOW)
    for name in ("temp_c", "wind_speed_ms", "temp_c"):
        module._blend_variable(
            train,
            snapshot.hourly,
            hourly_variable(name),
            compatible_promotion(config, "hourly", name, "equal_weight"),
            config,
            daily=False,
            semantics=TruthSemantics.INSTANTANEOUS,
            fit_diagnostics=diagnostics,
        )
    assert not calls
    diagnostics.resolve_training("hourly", raw, NOW)
    diagnostics.resolve_training("hourly", raw, NOW)
    assert calls == [("t__temp_c__inst", "t__wind_speed_ms__inst")]
    assert len(diagnostics.failures) == 2
    assert all(
        f.cause == diagnostics_module.FitUnavailableCause.CAUSAL_EXCLUSION
        for f in diagnostics.failures
    )


def test_prediction_releases_raw_hourly_before_loading_daily(
    config_no_provider_qc, monkeypatch
):
    import gc
    import weakref
    import polars as pl
    import grounded_weather_forecast.serve.predict as module
    from grounded_weather_forecast.dataset.matrix import matrix_path

    config = config_no_provider_qc
    hourly_path = matrix_path(config.dataset.dir, "hourly", "live")
    daily_path = matrix_path(config.dataset.dir, "daily", "live")
    real = pl.read_parquet
    loaded = []
    refs = []

    def read(path, *args, **kwargs):
        if path == daily_path:
            gc.collect()
            assert all(reference() is None for reference in refs)
        frame = real(path, *args, **kwargs)
        if path in (hourly_path, daily_path):
            loaded.append(path)
        if path == hourly_path:
            refs.append(weakref.ref(frame))
        return frame

    monkeypatch.setattr(module.pl, "read_parquet", read)
    module.predict(config, {}, now=NOW)
    assert loaded == [hourly_path, daily_path]


def test_null_issue_times_do_not_default_to_missing_truth(config_no_provider_qc):
    import polars as pl
    import grounded_weather_forecast.serve.predict as module
    from grounded_weather_forecast.contracts import hourly_variable
    from grounded_weather_forecast.serve.diagnostics import (
        FitDiagnostics,
        FitUnavailableCause,
    )

    config = config_no_provider_qc
    raw = scoreable_rain_training().with_columns(
        pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("issue_time")
    )
    diagnostics = FitDiagnostics()
    train = module._training_matrix(
        config, "hourly", NOW, frame=raw, fit_diagnostics=diagnostics
    )
    module._blend_variable(
        train,
        build_snapshot(config, NOW).hourly,
        hourly_variable("temp_c"),
        compatible_promotion(config, "hourly", "temp_c", "equal_weight"),
        config,
        daily=False,
        semantics=TruthSemantics.INSTANTANEOUS,
        fit_diagnostics=diagnostics,
    )
    diagnostics.resolve_training("hourly", raw, NOW)
    assert diagnostics.failures[0].cause == FitUnavailableCause.UNKNOWN_HISTORY


@pytest.mark.parametrize("provider_available", [False, True])
def test_mixed_fitting_outcomes_name_only_unavailable_variable(
    config_no_provider_qc, provider_available
):
    from dataclasses import replace
    import polars as pl
    import grounded_weather_forecast.serve.predict as module
    from grounded_weather_forecast.contracts import hourly_variable
    from grounded_weather_forecast.serve.dataset_snapshot import LiveDatasetSnapshot

    config = config_no_provider_qc
    train = scoreable_rain_training().with_columns(
        pl.lit(None, dtype=pl.Float64).alias("t__wind_speed_ms__inst")
    )
    if not provider_available:
        train = train.rename(
            {
                name: name.replace("__nws__", "__old_nws__").replace(
                    "__open_meteo__", "__old_open_meteo__"
                )
                for name in train.columns
                if name.startswith("fx__")
            }
        )
    selections = compatible_promotion(
        config, "hourly", "temp_c", "equal_weight", dataset_id="fixture"
    )
    selections.update(
        compatible_promotion(
            config, "hourly", "wind_speed_ms", "equal_weight", dataset_id="fixture"
        )
    )
    # Daily fitting has a distinct unavailable promoted slice as well.
    selections.update(
        compatible_promotion(
            config, "daily", "temp_max_c", "equal_weight", dataset_id="fixture"
        )
    )
    snapshot = build_snapshot(config, NOW)
    fitted = module._fit_methods(
        train,
        snapshot.hourly,
        hourly_variable("temp_c"),
        {"equal_weight"},
        daily=False,
        semantics=TruthSemantics.INSTANTANEOUS,
    )
    assert fitted is not None
    dataset = LiveDatasetSnapshot("fixture", train, pl.DataFrame(), pl.DataFrame())
    document = predict(config, selections, now=NOW, dataset_snapshot=dataset)
    if provider_available:
        assert document.status == "ready"
        assert document.status_reason is None
    else:
        assert document.status == "degraded"
        assert document.status_reason.startswith("some compatible promoted selections")
        assert "hourly.wind_speed_ms" in document.status_reason
        assert "daily.temp_max_c" in document.status_reason
        assert "hourly.temp_c" not in document.status_reason
        assert document.status_reason.index(
            "daily.temp_max_c"
        ) < document.status_reason.index("hourly.wind_speed_ms")


@pytest.mark.parametrize("daily", [False, True])
@pytest.mark.parametrize("compatible_evidence", [False, True])
def test_fitted_null_rain_raised_by_native_floor_retains_evidence_diagnosis(
    config, daily, compatible_evidence
):
    import numpy as np
    import polars as pl
    import grounded_weather_forecast.serve.predict as module
    from grounded_weather_forecast.contracts import daily_variable, hourly_variable
    from grounded_weather_forecast.serve.dataset_snapshot import LiveDatasetSnapshot
    from grounded_weather_forecast.serve.selection import no_evidence_reason

    name = "precip_sum_mm" if daily else "precip_mm"
    snapshot = build_snapshot(config, NOW)
    train = scoreable_rain_training(daily=daily)
    frame = snapshot.daily if daily else snapshot.hourly
    fitted = module._fit_methods(
        train,
        frame,
        daily_variable(name) if daily else hourly_variable(name),
        {"equal_weight"},
        daily=daily,
        semantics=TruthSemantics.INSTANTANEOUS,
    )
    assert fitted is not None
    assert np.isnan(fitted[0]["equal_weight"].point).all()
    dataset = LiveDatasetSnapshot(
        "fixture",
        pl.DataFrame() if daily else train,
        train if daily else pl.DataFrame(),
        pl.DataFrame(),
    )
    selection = Selection(
        "equal_weight",
        "promoted",
        release_id="null-rain",
        truth_semantics="inst",
        pinned=True,
    )
    selections = {
        ("daily" if daily else "hourly", name, "D1" if daily else "0-1h"): selection
    }
    if compatible_evidence:
        from dataclasses import replace

        selections = {
            key: replace(chosen, pinned=True)
            for key, chosen in compatible_promotion(
                config,
                "daily" if daily else "hourly",
                name,
                "equal_weight",
                dataset_id="fixture",
            ).items()
        }
    document = predict(config, selections, now=NOW, dataset_snapshot=dataset)
    point = document.daily[0] if daily else document.hourly[0]
    assert point.values[name] == pytest.approx(0.25)
    assert name not in point.release_ids
    assert document.release_ids == []
    assert document.status == "degraded"
    assert document.status_reason == no_evidence_reason(
        config, config.dataset.dir / "scores", dataset_id="fixture"
    )
    if compatible_evidence:
        assert document.status_reason == (
            "live evidence exists but no promoted release produced a finite hourly "
            "or daily forecast; keep polling"
        )


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("other_release", [False, True])
def test_real_fitted_rain_guard_records_loss(
    config_no_provider_qc, monkeypatch, native, other_release
):
    from dataclasses import replace
    import polars as pl
    import grounded_weather_forecast.serve.predict as module
    from grounded_weather_forecast.contracts import hourly_variable
    from grounded_weather_forecast.serve.dataset_snapshot import LiveDatasetSnapshot
    from grounded_weather_forecast.serve.rainfall import RainfallEffects

    config = config_no_provider_qc
    snapshot = build_snapshot(config, NOW)
    snapshot = replace(snapshot, daily=snapshot.daily.head(1))
    if not native:
        snapshot = replace(snapshot, minutely=snapshot.minutely.head(0))
    train = scoreable_rain_training()
    train = train.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("t__wind_speed_ms__inst")
    )
    fitted = module._fit_methods(
        train,
        snapshot.hourly,
        hourly_variable("precip_mm"),
        {"equal_weight"},
        daily=False,
        semantics=TruthSemantics.INSTANTANEOUS,
    )
    assert fitted is not None
    assert math.isfinite(fitted[0]["equal_weight"].point[0])
    monkeypatch.setattr(module, "build_snapshot", lambda *_args: snapshot)
    selections = {
        ("hourly", "precip_mm", "0-1h"): Selection(
            "equal_weight", "promoted", release_id="rain-only", truth_semantics="inst"
        )
    }
    selections[("hourly", "wind_speed_ms", "0-1h")] = Selection(
        "equal_weight",
        "promoted",
        release_id="unavailable-wind",
        truth_semantics="inst",
    )
    if native:
        selections[("daily", "precip_sum_mm", "D1")] = Selection(
            "equal_weight",
            "promoted",
            release_id="daily-rain",
            truth_semantics="inst",
            pinned=True,
        )
    if other_release:
        selections[("hourly", "temp_c", "0-1h")] = Selection(
            "equal_weight", "promoted", release_id="temperature", truth_semantics="inst"
        )
    effects = RainfallEffects()
    monkeypatch.setattr(module, "RainfallEffects", lambda: effects)
    dataset = LiveDatasetSnapshot(
        "fixture", train, scoreable_rain_training(daily=True), pl.DataFrame()
    )
    document = predict(config, selections, now=NOW, dataset_snapshot=dataset)
    assert effects.guarded_release_removed
    assert effects.reconciled_release_removed == native
    assert "rain-only" not in document.release_ids
    assert "0 wet local days" in document.hourly[0].selection_reasons["precip_mm"]
    if other_release:
        assert document.status == "ready"
        assert document.status_reason is None
        assert document.release_ids == ["temperature"]
    else:
        assert document.status == "degraded"
        assert "rainfall safeguard" in document.status_reason


@pytest.mark.parametrize(
    ("rain_release", "other_release"), [(True, False), (False, False), (True, True)]
)
def test_document_reason_requires_reconciliation_attribution_loss(
    config, monkeypatch, rain_release, other_release
):
    import grounded_weather_forecast.serve.predict as module
    from grounded_weather_forecast.serve.schema import HourlyPoint, MinutelyPoint

    point = HourlyPoint(NOW.isoformat(), 0.0, values={"precip_mm": 1.0})
    if rain_release:
        point.release_ids["precip_mm"] = "rain-release"
    if other_release:
        point.values["temp_c"] = 20.0
        point.release_ids["temp_c"] = "temperature-release"
    monkeypatch.setattr(
        module, "hourly_product", lambda *_args, **_kwargs: ([point], {})
    )
    monkeypatch.setattr(module, "daily_product", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        module,
        "minutely_product",
        lambda *_args, **_kwargs: [
            MinutelyPoint(
                (NOW + timedelta(minutes=i)).isoformat(), i, precip_intensity_mmh=2.0
            )
            for i in range(1, 61)
        ],
    )
    document = predict(config, {}, now=NOW)
    assert point.values["precip_mm"] == pytest.approx(2.0)
    if other_release:
        assert document.status == "ready"
        assert document.status_reason is None
    elif rain_release:
        assert "rainfall reconciliation" in document.status_reason
    else:
        assert "cold start" in document.status_reason


def build_fixture(tmp_path, *, obs_temp_f=68.0, extra_toml=""):
    """A station with recent obs plus two providers with 48h of hourly data.

    The temperature wobbles: a dead-flat series would (correctly) be caught by
    the flatline QC filter and never become truth.
    """
    samples = minute_series(NOW - timedelta(hours=6), 6 * 12, 300)
    station_rows = [
        (
            ts.strftime("%Y-%m-%d %H:%M:%S"),
            {
                "outTemp": obs_temp_f + 0.1 * (index % 5),
                "outHumi": 40.0 + 0.1 * (index % 3),
            },
        )
        for index, ts in enumerate(samples)
    ]
    make_station_db(tmp_path / "station.db", station_rows)

    results = []
    for provider, offset in (("nws", 0.0), ("open_meteo", 2.0)):
        hourly = [
            (
                NOW + timedelta(hours=lead),
                {
                    "temperature": 15.0 + offset,
                    "humidity": 140.0,
                    "wind_speed": -2.0,
                    "precipitation": -5.0,
                    "precipitation_probability": 2.0,
                },
            )
            for lead in range(49)
        ]
        daily = [
            (
                (NOW + timedelta(days=d)).strftime("%Y-%m-%d"),
                {
                    "temperature_max": 22.0 + offset,
                    "temperature_min": 8.0 + offset,
                    "precipitation_sum": -3.0,
                    "precipitation_probability_max": 2.0,
                },
            )
            for d in range(10)
        ]
        results.append(
            {
                "provider": provider,
                "fetched_at": FETCH,
                "hourly": hourly,
                "daily": daily,
                "minutely": [
                    (NOW + timedelta(minutes=m), 0.5, 0.4) for m in range(1, 31)
                ],
            }
        )
    results.append(
        {
            "provider": "stale_minutely",
            "fetched_at": (NOW - timedelta(hours=13)).isoformat(),
            "minutely": [(NOW + timedelta(minutes=1), 99.0, 1.0)],
        }
    )
    make_forecast_db(
        tmp_path / "fx.sqlite",
        [{"completed_at": FETCH, "results": results}],
    )
    config = write_config(
        tmp_path,
        min_hour_coverage=0.05,
        min_day_coverage=0.02,
        extra_toml=extra_toml,
    )
    write_dataset(config)
    return config


@pytest.fixture
def config(tmp_path):
    return build_fixture(tmp_path)


@pytest.fixture
def config_no_provider_qc(tmp_path):
    # The provider values in the fixture are deliberately out of physical bounds
    # (humidity 140, wind -2, ...) to exercise the serve-layer clamp; disable the
    # provider QC that would otherwise (correctly) null them before serving.
    return build_fixture(tmp_path, extra_toml="[provider_qc]\nenabled = false")


class TestSnapshot:
    def test_as_of_view(self, config):
        snapshot = build_snapshot(config, NOW)
        assert snapshot.issue_time == NOW
        assert not snapshot.hourly.is_empty()
        assert not snapshot.daily.is_empty()
        assert snapshot.observation_at is not None
        assert snapshot.observation["temp_c"] == pytest.approx(20.0, abs=0.1)
        assert (snapshot.hourly["lead_hours"] >= 0).all()
        assert "stale_minutely" not in snapshot.minutely["source"].to_list()

    def test_stale_archive_refuses(self, config):
        with pytest.raises(NoForecastDataError, match="no provider forecast"):
            build_snapshot(config, NOW + timedelta(days=30))


@pytest.mark.parametrize(
    ("native", "wet"), [(False, True), (True, True), (True, False)]
)
def test_wet_guard_skips_diagnostics_only_for_guarded_rows(
    tmp_path, monkeypatch, native, wet
):
    import numpy as np
    import polars as pl
    import grounded_weather_forecast.serve.predict as module
    from grounded_weather_forecast.contracts import BlendResult, hourly_variable

    config = write_config(tmp_path)
    train = pl.DataFrame(
        {
            "t__pop": [1.0 if wet else 0.0] * 5,
            "lead_hours": [25.0] * 5,
            "valid_time": [NOW - timedelta(days=i) for i in range(1, 6)],
        }
    )
    future = pl.DataFrame({"lead_hours": [2.0, 25.0]})
    result = BlendResult(
        np.array([0.2, 0.2]),
        quantiles=np.array([[0.1, 0.3]] * 2) if native else None,
        quantile_levels=(0.1, 0.9) if native else (),
    )
    monkeypatch.setattr(
        module,
        "_fit_methods",
        lambda *_args, **_kwargs: ({"equal_weight": result}, None),
    )
    pools, fits = [], []

    class Dresser:
        def errors_for(self, _variable, _semantics, _methods, _method, bucket):
            pools.append(bucket)
            return np.arange(40.0) / 100.0

    class Recalibrator:
        mode = "cqr"

        def fitted_for(self, _variable, _semantics, _methods, _method, bucket):
            fits.append(bucket)
            return None

    selections = {
        ("hourly", "pop", bucket): Selection(
            "equal_weight",
            "promoted",
            release_id="rain-release",
            truth_semantics="inst",
        )
        for bucket in ("1-3h", "24-48h")
    }
    blend = module._blend_variable(
        train,
        future,
        hourly_variable("pop"),
        selections,
        config,
        daily=False,
        semantics=TruthSemantics.INSTANTANEOUS,
        dresser=Dresser(),
        recalibrator=Recalibrator(),
    )
    assert blend.quantiles[0] == {}
    assert blend.release_ids[0] is None
    assert pools == (["24-48h"] if wet and not native else [])
    assert fits == (["24-48h"] if wet and native else [])
    assert bool(blend.quantiles[1]) == wet
    assert blend.release_ids[1] == ("rain-release" if wet else None)
    if native and wet:
        assert blend.quantiles[1] == {"0.1": 0.1, "0.9": 0.3}


class TestPredict:
    def test_three_products(self, config):
        document = predict(config, {}, now=NOW)
        assert document.schema_version == SCHEMA_VERSION
        assert len(document.minutely) == 60
        assert len(document.hourly) == 49
        assert len(document.daily) == 10
        assert set(document.sources) == {"nws", "open_meteo"}

    def test_json_round_trip(self, config):
        payload = json.loads(predict(config, {}, now=NOW).to_json())
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["hourly"][0]["lead_bucket"] == "0-1h"
        assert "temp_c" in payload["hourly"][0]["values"]
        assert "temp_c" in payload["hourly"][0]["methods"]

    def test_minutely_is_anchored_to_the_observation(self, config):
        # providers say 15-17C; the station says 20C. The nowcast must start
        # near the observation and relax toward the blend.
        document = predict(config, {}, now=NOW)
        first = document.minutely[0].temp_c
        last = document.minutely[-1].temp_c
        blend_now = document.hourly[0].values["temp_c"]
        assert first is not None
        assert last is not None
        assert blend_now is not None
        assert abs(first - 20.0) < abs(first - blend_now)  # anchored to obs
        assert abs(last - blend_now) < abs(first - blend_now)  # relaxing back

    def test_minutely_precip_from_native_points(self, config):
        document = predict(config, {}, now=NOW)
        assert document.minutely[0].precip_intensity_mmh == pytest.approx(0.5)
        assert document.minutely[0].pop == pytest.approx(0.4)
        # providers only published 30 minutes of it
        assert document.minutely[-1].precip_intensity_mmh is None

    def test_daily_hi_lo(self, config):
        document = predict(config, {}, now=NOW)
        values = document.daily[0].values
        assert values["temp_max_c"] is not None
        assert values["temp_min_c"] is not None
        assert values["temp_max_c"] > values["temp_min_c"]

    def test_emitted_values_obey_physical_bounds(self, config_no_provider_qc):
        document = predict(config_no_provider_qc, {}, now=NOW)
        for point in document.hourly:
            assert point.values["humidity_pct"] == 100.0
            assert point.values["wind_speed_ms"] == 0.0
            assert point.values["precip_mm"] >= 0.0
            assert point.values["pop"] == 1.0
        for point in document.daily:
            assert point.values["precip_sum_mm"] >= 0.0
            assert point.values["pop"] == 1.0

    def test_force_method(self, config):
        document = predict(config, {}, now=NOW, force_method="equal_weight")
        methods = {m for p in document.hourly for m in p.methods.values()}
        assert methods == {"equal_weight"}
        assert document.status == "ready"
        assert document.status_reason is None

    @pytest.mark.parametrize("daily", [False, True])
    @pytest.mark.parametrize("fixture_name", ["config", "config_no_provider_qc"])
    def test_rain_candidate_does_not_mask_cold_start(
        self, request, fixture_name, daily
    ):
        import grounded_weather_forecast.serve.predict as module
        from grounded_weather_forecast.contracts import daily_variable, hourly_variable
        from grounded_weather_forecast.serve.selection import no_evidence_reason

        config = request.getfixturevalue(fixture_name)
        product = "daily" if daily else "hourly"
        name = "precip_sum_mm" if daily else "precip_mm"
        selections = {
            (product, name, "D1" if daily else "0-1h"): Selection(
                "equal_weight",
                "promoted",
                release_id="rain-only",
                truth_semantics="inst",
            )
        }
        snapshot = build_snapshot(config, NOW)
        train = module._training_matrix(config, product, NOW)
        assert train.is_empty()
        blend = module._blend_variable(
            train,
            snapshot.daily if daily else snapshot.hourly,
            daily_variable(name) if daily else hourly_variable(name),
            selections,
            config,
            daily=daily,
            semantics=TruthSemantics.INSTANTANEOUS,
        )
        assert blend is not None
        assert not blend.guarded_release_ids
        document = predict(config, selections, now=NOW)
        assert document.status == "degraded"
        assert "rain-only" not in document.release_ids
        assert document.status_reason == no_evidence_reason(
            config, config.dataset.dir / "scores"
        )
        point = document.daily[0] if daily else document.hourly[0]
        assert "0 wet local days" in point.selection_reasons[name]
        assert document.hourly[0].values["precip_mm"] == pytest.approx(0.25)
        assert document.daily[0].values["precip_sum_mm"] == pytest.approx(0.25)

    def test_archived_attribution_replays_without_rewriting(self, config, monkeypatch):
        from dataclasses import replace
        import grounded_weather_forecast.serve.predict as module
        from grounded_weather_forecast.serve.history import append_history

        document = predict(config, {}, now=NOW)
        document.hourly[0].values["precip_mm"] = None
        document.hourly[0].release_ids["precip_mm"] = "historical-release"
        document = replace(
            document,
            status="ready",
            status_reason=None,
            release_ids=["historical-release"],
        )
        append_history(document, config.predict.history_path)
        expected = document.to_json()
        monkeypatch.setattr(
            module,
            "build_snapshot",
            lambda *_args: pytest.fail("archived forecasts must not be rebuilt"),
        )
        assert predict(config, {}, now=NOW).to_json() == expected

    def test_unsupported_forced_method_fails_the_whole_prediction(self, config):
        with pytest.raises(UnsupportedMethodError, match="hourly.precip_mm"):
            predict(config, {}, now=NOW, force_method="persistence")

    def test_unsupported_stale_selection_degrades_without_provenance(self, config):
        selections = {
            ("hourly", "precip_mm", "0-1h"): Selection(
                "persistence",
                reason="old release",
                release_id="stale-release",
                evaluation_id="old-evaluation",
                dataset_fingerprint="old-data",
            )
        }
        document = predict(config, selections, now=NOW)
        first = document.hourly[0]
        assert first.methods["precip_mm"] == "minutely_rainfall_floor"
        assert "degraded stale selection" in first.selection_reasons["precip_mm"]
        assert "stale-release" not in document.release_ids

    def test_unknown_stale_selection_degrades_without_provenance(self, config):
        selections = {
            ("hourly", "temp_c", "0-1h"): Selection(
                "retired_method",
                reason="old release",
                release_id="retired-release",
                evaluation_id="old-evaluation",
                dataset_fingerprint="old-data",
            )
        }
        document = predict(config, selections, now=NOW)
        first = document.hourly[0]
        assert first.methods["temp_c"] == "equal_weight"
        assert "unknown method retired_method" in first.selection_reasons["temp_c"]
        assert "retired-release" not in document.release_ids

    def test_semantics_mismatch_degrades_without_release_provenance(self, config):
        selections = {
            ("hourly", "temp_c", "0-1h"): Selection(
                "gbm",
                reason="instantaneous winner",
                release_id="instantaneous-release",
                truth_semantics="inst",
            )
        }

        document = predict(
            config,
            selections,
            now=NOW,
            semantics={"temp_c": TruthSemantics.INTERVAL_MEAN},
        )

        first = document.hourly[0]
        assert first.methods["temp_c"] == "equal_weight"
        assert first.truth_semantics["temp_c"] == "mean"
        assert "requested mean" in first.selection_reasons["temp_c"]
        assert "instantaneous-release" not in document.release_ids

    def test_fallback_method_when_no_scores(self, config):
        document = predict(config, {}, now=NOW)
        methods = {m for p in document.hourly for m in p.methods.values()}
        assert methods == {"equal_weight", "minutely_rainfall_floor"}
        assert document.status == "degraded"

    def test_incompatible_historical_release_degrades_without_provenance(self, config):
        from grounded_weather_forecast.evaluation import (
            config_fingerprint,
            dataset_fingerprint,
        )

        release = {
            "release_id": "old-code-release",
            "promoted_at": (NOW - timedelta(days=1)).isoformat(),
            "dataset_fingerprint": dataset_fingerprint(config),
            "config_fingerprint": config_fingerprint(config),
            "evaluation_ids": ["old-evaluation"],
            "evaluation_contexts": [],
            "training_cutoff": None,
            "selections": {
                "hourly.temp_c.0-1h": {
                    "method_id": "gbm",
                    "reason": "historical winner",
                    "evaluation_id": "old-evaluation",
                    "code_version": "0.4.0+retired-implementation",
                    "n": 100,
                    "mae": 1.0,
                }
            },
        }
        release_dir = config.artifacts_dir / "releases"
        release_dir.mkdir(parents=True)
        (release_dir / "old-code-release.json").write_text(
            json.dumps(release), encoding="utf-8"
        )

        selections = select_methods(config, config.dataset.dir / "scores", as_of=NOW)
        document = predict(config, selections, now=NOW)

        assert selections == {}
        assert document.status == "degraded"
        assert document.release_ids == []


class TestPredictCli:
    def test_writes_json_and_history(self, tmp_path, capsys):
        config = build_fixture(tmp_path)
        out = tmp_path / "forecast.json"
        code = main(
            [
                "--config",
                str(tmp_path / "config.toml"),
                "predict",
                "--now",
                NOW.isoformat(),
                "--out",
                str(out),
            ]
        )
        assert code == 0
        printed = capsys.readouterr().out
        assert "wrote" in printed
        assert "appended" in printed
        assert out.exists()
        assert config.predict.history_path.exists()

    def test_stale_archive_exits_nonzero(self, tmp_path, capsys):
        build_fixture(tmp_path)
        code = main(
            [
                "--config",
                str(tmp_path / "config.toml"),
                "predict",
                "--now",
                (NOW + timedelta(days=30)).isoformat(),
                "--no-history",
            ]
        )
        assert code == 1
        assert "cannot predict" in capsys.readouterr().out

    def test_stdout_is_json_only(self, tmp_path, capsys):
        build_fixture(tmp_path)
        code = main(
            [
                "--config",
                str(tmp_path / "config.toml"),
                "predict",
                "--now",
                NOW.isoformat(),
            ]
        )
        assert code == 0
        captured = capsys.readouterr()
        assert json.loads(captured.out)["issued_at"] == NOW.isoformat()
        assert "appended" in captured.err

    def test_unsupported_forced_method_is_an_actionable_cli_error(
        self, tmp_path, capsys
    ):
        build_fixture(tmp_path)
        code = main(
            [
                "--config",
                str(tmp_path / "config.toml"),
                "predict",
                "--now",
                NOW.isoformat(),
                "--method",
                "persistence",
                "--no-history",
            ]
        )
        assert code == 1
        output = capsys.readouterr().out
        assert "cannot predict" in output
        assert "does not support hourly.precip_mm" in output

    def test_unknown_forced_method_is_an_actionable_cli_error(self, tmp_path, capsys):
        build_fixture(tmp_path)
        code = main(
            [
                "--config",
                str(tmp_path / "config.toml"),
                "predict",
                "--now",
                NOW.isoformat(),
                "--method",
                "retired_method",
                "--no-history",
            ]
        )
        assert code == 1
        output = capsys.readouterr().out
        assert "cannot predict" in output
        assert "unknown method 'retired_method'" in output


class TestMinutelySinglePass:
    """Anchoring is applied exactly once, by whichever stage was promoted."""

    def make_snapshot(self):
        import numpy as np
        import polars as pl

        from grounded_weather_forecast.serve.predict import Snapshot
        from grounded_weather_forecast.timeutil import utc

        issue = utc(2026, 3, 22, 12, 0)
        hourly = pl.DataFrame(
            {
                "valid_time": [issue + timedelta(hours=1), issue + timedelta(hours=2)],
                "lead_hours": [1.0, 2.0],
            },
            schema_overrides={"valid_time": pl.Datetime("us", "UTC")},
        )
        return Snapshot(
            issue_time=issue,
            hourly=hourly,
            daily=pl.DataFrame(),
            minutely=pl.DataFrame(),
            observation={"temp_c": 10.0},
            observation_at=issue,
        ), np.array([20.0, 21.0])

    def blend(self, path, method_id):
        from grounded_weather_forecast.serve.predict import VariableBlend

        return VariableBlend(
            point=path,
            methods=[method_id, method_id],
            reasons=["", ""],
            release_ids=[None, None],
            quantiles=[{}, {}],
        )

    def test_anchored_selection_is_not_re_anchored(self, la_config):
        from grounded_weather_forecast.serve.predict import minutely_product

        snapshot, path = self.make_snapshot()
        points = minutely_product(
            snapshot,
            {"temp_c": self.blend(path, "anchored_fitted_grounded")},
            la_config,
        )
        # pure interpolation on the lead-zero-extended hourly path; the
        # 10-degree observation is deliberately ignored (already anchored)
        assert points[0].temp_c == pytest.approx(19.0 + 1.0 / 60.0, abs=0.001)

    def test_unanchored_selection_converges_to_the_observation(self, la_config):
        from grounded_weather_forecast.serve.predict import minutely_product

        snapshot, path = self.make_snapshot()
        points = minutely_product(
            snapshot,
            {"temp_c": self.blend(path, "grounded_equal_weight")},
            la_config,
        )
        # Both the base path and residual use the 19.0 lead-zero extrapolation,
        # so the first minute stays continuous with the observation.
        expected_first = 19.0 + 1.0 / 60.0 + math.exp(-(1 / 60) / 3.0) * -9.0
        assert points[0].temp_c == pytest.approx(expected_first, abs=0.02)
        # minute 60: w = exp(-1/tau) with the 3h config default
        expected = 20.0 + math.exp(-1.0 / 3.0) * -9.0
        assert points[-1].temp_c == pytest.approx(expected, abs=0.1)
        assert points[-1].temp_c > points[0].temp_c  # decaying toward the path

    def test_minutely_dew_point_is_not_above_temperature(self, la_config):
        from grounded_weather_forecast.serve.predict import minutely_product

        snapshot, path = self.make_snapshot()
        points = minutely_product(
            snapshot,
            {
                "temp_c": self.blend(path, "anchored_fitted_grounded"),
                "dew_point_c": self.blend(path + 5.0, "anchored_fitted_grounded"),
            },
            la_config,
        )
        assert all(
            point.dew_point_c <= point.temp_c
            for point in points
            if point.dew_point_c is not None and point.temp_c is not None
        )

    def test_mixed_anchor_regimes_follow_the_lead_zero_regime(self, la_config):
        """The row owning lead zero decides anchoring for the whole range.

        Switching regime partway would both step the path and flip the
        anchoring decision mid-range. Here lead zero is anchored, so the
        config decay must not be re-applied and the path stays continuous.
        """
        from grounded_weather_forecast.serve.predict import (
            VariableBlend,
            minutely_product,
        )

        snapshot, path = self.make_snapshot()
        blend = VariableBlend(
            point=path,
            methods=["anchored_fitted_grounded", "grounded_equal_weight"],
            reasons=["", ""],
            release_ids=[None, None],
            quantiles=[{}, {}],
        )

        points = minutely_product(snapshot, {"temp_c": blend}, la_config)
        values = [point.temp_c for point in points]

        # Already anchored: the 10.0 C observation is not pulled in again.
        assert min(values) > 15.0
        # Continuous, and arriving at the hourly value at lead 1 h.
        assert values[-1] == pytest.approx(path[0])
        assert max(abs(b - a) for a, b in zip(values, values[1:])) < 0.1

    def test_mixed_anchor_regimes_do_not_step_inside_the_horizon(self, la_config):
        """Regression: the bracketing midpoint falling inside the 60-min horizon.

        With sub-hourly first leads (the normal case for a non-o'clock issue
        time) the old nearest-neighbour fallback produced a flat line with one
        large jump, and flipped `already_anchored` halfway, so the live
        observation was ignored for the minutes on the anchored side.
        """
        import numpy as np
        import polars as pl

        from grounded_weather_forecast.serve.predict import (
            Snapshot,
            VariableBlend,
            minutely_product,
        )
        from grounded_weather_forecast.timeutil import utc

        issue = utc(2026, 3, 22, 12, 37)
        hourly = pl.DataFrame(
            {
                "valid_time": [utc(2026, 3, 22, 13), utc(2026, 3, 22, 14)],
                "lead_hours": [23.0 / 60.0, 83.0 / 60.0],
            },
            schema_overrides={"valid_time": pl.Datetime("us", "UTC")},
        )
        snapshot = Snapshot(
            issue_time=issue,
            hourly=hourly,
            daily=pl.DataFrame(),
            minutely=pl.DataFrame(),
            observation={"temp_c": 21.0},
            observation_at=issue,
        )
        blend = VariableBlend(
            point=np.array([20.0, 23.0]),
            methods=["grounded_equal_weight", "anchored_fitted_grounded"],
            reasons=["", ""],
            release_ids=[None, None],
            quantiles=[{}, {}],
        )

        values = [
            point.temp_c
            for point in minutely_product(snapshot, {"temp_c": blend}, la_config)
        ]

        jumps = [abs(b - a) for a, b in zip(values, values[1:])]
        assert max(jumps) < 0.2, "minutely path must not step between regimes"
        # Lead zero is un-anchored, so the live observation is honoured.
        assert values[0] == pytest.approx(21.0, abs=0.5)

    @pytest.mark.parametrize("observation", [21.0, 22.4])
    def test_mixed_anchor_regimes_stay_inside_the_hourly_bracket(
        self, la_config, observation
    ):
        """Regression: interpolating across a regime boundary anchored twice.

        The far hourly row is ``anchored_*``, so its point already carries a
        fitted correction. Drawing the minutely path through it while
        reporting the segment un-anchored added the live-observation residual
        on top, pushing the nowcast outside the bracketing hourly values --
        which a monotone interpolation can never legitimately do.
        """
        import numpy as np
        import polars as pl

        from grounded_weather_forecast.serve.predict import (
            Snapshot,
            VariableBlend,
            minutely_product,
        )
        from grounded_weather_forecast.timeutil import utc

        low, high = 20.0, 23.0
        issue = utc(2026, 3, 22, 12, 37)
        hourly = pl.DataFrame(
            {
                "valid_time": [utc(2026, 3, 22, 13), utc(2026, 3, 22, 14)],
                "lead_hours": [23.0 / 60.0, 83.0 / 60.0],
            },
            schema_overrides={"valid_time": pl.Datetime("us", "UTC")},
        )
        snapshot = Snapshot(
            issue_time=issue,
            hourly=hourly,
            daily=pl.DataFrame(),
            minutely=pl.DataFrame(),
            observation={"temp_c": observation},
            observation_at=issue,
        )
        blend = VariableBlend(
            point=np.array([low, high]),
            methods=["grounded_equal_weight", "anchored_fitted_grounded"],
            reasons=["", ""],
            release_ids=[None, None],
            quantiles=[{}, {}],
        )

        values = [
            point.temp_c
            for point in minutely_product(snapshot, {"temp_c": blend}, la_config)
            if point.temp_c is not None
        ]

        assert values, "the nowcast must emit temperatures"
        assert max(values) <= high + 1e-9, (
            f"emitted {max(values):.3f} above the hourly bracket max {high}"
        )
        assert min(values) >= min(low, observation) - 1e-9


class TestPhysicalCoherence:
    def test_hourly_points_and_differing_quantile_grids_are_coherent(self):
        import numpy as np

        from grounded_weather_forecast.serve.predict import _cohere_hourly
        from grounded_weather_forecast.serve.schema import HourlyPoint

        point = HourlyPoint(
            valid_time=NOW.isoformat(),
            lead_hours=1.0,
            values={
                "temp_c": 1.0,
                "dew_point_c": 5.0,
                "wind_speed_ms": 3.0,
                "wind_gust_ms": 1.0,
            },
            quantiles={
                "temp_c": {"0.1": 0.0, "0.9": 2.0},
                "dew_point_c": {"0.25": 3.0, "0.75": 4.0},
                "wind_speed_ms": {"0.2": 2.0, "0.8": 4.0},
                "wind_gust_ms": {"0.1": 0.0, "0.9": 1.0},
            },
        )
        _cohere_hourly([point])

        assert point.values["dew_point_c"] <= point.values["temp_c"]
        assert point.values["wind_gust_ms"] >= point.values["wind_speed_ms"]
        levels = np.array([0.1, 0.2, 0.25, 0.75, 0.8, 0.9])

        def curve(name):
            raw = point.quantiles[name]
            x = np.array([float(level) for level in raw])
            y = np.array(list(raw.values()))
            order = np.argsort(x)
            return np.interp(levels, x[order], y[order])

        assert (curve("dew_point_c") <= curve("temp_c") + 1e-12).all()
        assert (curve("wind_gust_ms") >= curve("wind_speed_ms") - 1e-12).all()
        for name, value in point.values.items():
            quantiles = point.quantiles[name].values()
            assert min(quantiles) <= value <= max(quantiles)

    def test_daily_low_high_points_and_quantiles_are_coherent(self):
        import numpy as np

        from grounded_weather_forecast.serve.predict import _cohere_daily
        from grounded_weather_forecast.serve.schema import DailyPoint

        point = DailyPoint(
            date_local="2026-03-22",
            lead_days=1,
            values={"temp_min_c": 12.0, "temp_max_c": 8.0},
            quantiles={
                "temp_min_c": {"0.25": 10.0, "0.75": 14.0},
                "temp_max_c": {"0.1": 5.0, "0.9": 9.0},
            },
        )
        _cohere_daily([point])
        assert point.values["temp_min_c"] <= point.values["temp_max_c"]
        levels = np.array([0.1, 0.25, 0.75, 0.9])

        def curve(name):
            raw = point.quantiles[name]
            x = np.array([float(level) for level in raw])
            y = np.array(list(raw.values()))
            order = np.argsort(x)
            return np.interp(levels, x[order], y[order])

        assert (curve("temp_min_c") <= curve("temp_max_c") + 1e-12).all()

    def test_point_clipping_cannot_reintroduce_dew_point_violation(self):
        from grounded_weather_forecast.serve.predict import _cohere_hourly
        from grounded_weather_forecast.serve.schema import HourlyPoint

        point = HourlyPoint(
            valid_time=NOW.isoformat(),
            lead_hours=1.0,
            values={"temp_c": 10.0, "dew_point_c": 12.0},
            quantiles={"temp_c": {"0.1": 0.0, "0.9": 5.0}},
        )

        _cohere_hourly([point])

        assert point.values["temp_c"] == 5.0
        assert point.values["dew_point_c"] == 5.0


class TestCoherenceLeavesValidDistributionsAlone:
    """Coherence enforcement must be a no-op on input that is already coherent."""

    LEVELS = (0.05, 0.1, 0.25, 0.75, 0.9, 0.95)

    def _grid(self, mu, sd=2.0):
        from scipy.stats import norm

        return {str(level): float(norm.ppf(level, mu, sd)) for level in self.LEVELS}

    @pytest.mark.parametrize("depression", [0.5, 1.0, 2.0, 3.0])
    def test_shared_grid_coherent_dew_point_is_untouched(self, depression):
        """Regression: the lower q75 was being bounded by the upper q25.

        Both curves live on the same conformal grid and are coherent at every
        level, so nothing needs adjusting. Enforcing knot-to-neighbour bounds
        anyway shifted dew-point q75 down by up to 2.2 K -- worst exactly when
        the dew-point depression is small, i.e. fog and humid nights.
        """
        from grounded_weather_forecast.serve.predict import _cohere_pair

        temperature = 21.0
        dew_point = temperature - depression
        dew_q, temp_q = self._grid(dew_point), self._grid(temperature)
        for level in self.LEVELS:
            assert dew_q[str(level)] <= temp_q[str(level)], "fixture is incoherent"
        before = dict(dew_q)

        _cohere_pair(
            {"dew_point_c": dew_point, "temp_c": temperature},
            {"dew_point_c": dew_q, "temp_c": temp_q},
            "dew_point_c",
            "temp_c",
            adjust="lower",
        )

        for level in self.LEVELS:
            assert dew_q[str(level)] == pytest.approx(before[str(level)], abs=1e-9), (
                f"level {level} moved {dew_q[str(level)] - before[str(level)]:+.3f} K "
                "on an already-coherent shared grid"
            )

    def test_differing_grids_are_still_made_coherent(self):
        """The unequal-grid path -- which needs the neighbour bound -- survives."""
        import numpy as np

        from grounded_weather_forecast.serve.predict import _cohere_pair

        dew_q = {"0.25": 12.0, "0.75": 18.0}
        temp_q = {"0.1": 9.0, "0.5": 11.0, "0.9": 13.0}
        _cohere_pair(
            {"dew_point_c": 15.0, "temp_c": 11.0},
            {"dew_point_c": dew_q, "temp_c": temp_q},
            "dew_point_c",
            "temp_c",
            adjust="lower",
        )
        grid = np.linspace(0.25, 0.75, 11)
        dew = np.interp(
            grid, [float(k) for k in sorted(dew_q, key=float)], sorted(dew_q.values())
        )
        temp = np.interp(
            grid, [float(k) for k in sorted(temp_q, key=float)], sorted(temp_q.values())
        )
        assert np.all(dew <= temp + 1e-9), "dew point must not exceed temperature"

    @pytest.mark.parametrize("depression", [0.5, 1.0])
    def test_coherent_conformal_and_emos_grids_are_untouched(self, depression):
        """Real serving grids must not enter the lossy neighbour projection."""
        from scipy.stats import norm

        from grounded_weather_forecast.serve.predict import _cohere_pair

        conformal = (0.05, 0.1, 0.25, 0.75, 0.9, 0.95)
        emos = tuple(index / 20 for index in range(1, 20))
        temperature = 21.0
        dew_point = temperature - depression
        dew_q = {
            str(level): float(norm.ppf(level, dew_point, 2.0)) for level in conformal
        }
        temp_q = {
            str(level): float(norm.ppf(level, temperature, 2.0)) for level in emos
        }
        before = dict(dew_q)

        _cohere_pair(
            {"dew_point_c": dew_point, "temp_c": temperature},
            {"dew_point_c": dew_q, "temp_c": temp_q},
            "dew_point_c",
            "temp_c",
            adjust="lower",
        )

        assert dew_q == pytest.approx(before)


class TestServingPathFitting:
    """`_fit_methods` past its early returns: the warm-start path serving uses.

    Every other test returns at the `no truth sources` / `n_rows == 0` guards,
    so the artifact round trip, the stateful advance, and the observability
    snapshot all shipped unexercised through this entry point.
    """

    def _slice_frames(self):
        import polars as pl
        from conftest import synthetic_hourly_matrix

        matrix = synthetic_hourly_matrix(days=12, max_lead=12, seed=77)
        return matrix, matrix.filter(pl.col("lead_hours") <= 6.0)

    def _fit(self, config, method_ids, issue_time=NOW):
        import numpy as np  # noqa: F401

        from grounded_weather_forecast.contracts import TruthSemantics, hourly_variable
        from grounded_weather_forecast.serve.predict import _fit_methods

        train, predict_frame = self._slice_frames()
        return _fit_methods(
            train,
            predict_frame,
            hourly_variable("temp_c"),
            method_ids,
            daily=False,
            semantics=TruthSemantics.INSTANTANEOUS,
            config=config,
            issue_time=issue_time,
        )

    def test_fits_and_predicts_each_requested_method(self, tmp_path):
        config = write_config(tmp_path)
        fitted = self._fit(config, {"equal_weight", "grounded_equal_weight"})

        assert fitted is not None
        results, x = fitted
        assert set(results) == {"equal_weight", "grounded_equal_weight"}
        for result in results.values():
            assert result.point.shape[0] == x.n_rows

    def test_a_stateful_method_persists_and_warm_starts(self, tmp_path, monkeypatch):
        """Second serve must load the artifact and advance it, not refit.

        Output equality alone cannot prove this: a from-scratch refit over the
        same replayed evidence lands in the same place. So spy on the two
        mutually exclusive entry points -- ``from_state`` (rehydrate) versus
        ``fit`` (full refit) -- and require the second serve to take the former.
        """
        config = write_config(tmp_path)

        import numpy as np

        from grounded_weather_forecast.blenders import experts as experts_mod

        first = self._fit(config, {"ewa"})
        assert first is not None
        state_dir = config.artifacts_dir / "state"
        assert list(state_dir.rglob("*.json")), "the serve path must persist state"

        calls = {"from_state": 0, "fit": 0}
        real_from_state = experts_mod.OnlineExperts.from_state.__func__
        real_fit = experts_mod.OnlineExperts.fit

        def spy_from_state(cls, state, method_id):
            calls["from_state"] += 1
            return real_from_state(cls, state, method_id)

        def spy_fit(self, train):
            calls["fit"] += 1
            return real_fit(self, train)

        monkeypatch.setattr(
            experts_mod.OnlineExperts, "from_state", classmethod(spy_from_state)
        )
        monkeypatch.setattr(experts_mod.OnlineExperts, "fit", spy_fit)

        second = self._fit(config, {"ewa"}, issue_time=NOW + timedelta(minutes=10))
        assert second is not None
        assert calls["from_state"] == 1, "warm start must rehydrate the artifact"
        assert calls["fit"] == 0, "a warm start must not refit from scratch"
        # Same evidence replayed: the warm start must still land where the first did.
        np.testing.assert_allclose(
            second[0]["ewa"].point, first[0]["ewa"].point, atol=1e-9
        )

    def test_a_corrupt_artifact_falls_back_to_a_full_refit(self, tmp_path):
        """The digest guard's whole point: never silently combine histories."""
        import numpy as np

        config = write_config(tmp_path)
        assert self._fit(config, {"ewa"}) is not None
        corrupted = list((config.artifacts_dir / "state").rglob("*.json"))
        assert corrupted, "the first serve must have persisted state to corrupt"
        for path in corrupted:
            path.write_text('{"schema_version": 1}', encoding="utf-8")

        refitted = self._fit(config, {"ewa"})
        assert refitted is not None

        # An independent clean cold start over the same evidence, in a config
        # with no persisted state at all. Discarding the corrupt artifact and
        # replaying from scratch must reproduce it exactly; a silent warm start
        # from the corrupt file -- or any other partial state -- would diverge.
        clean_dir = tmp_path / "clean"
        clean_dir.mkdir()
        clean = write_config(clean_dir)
        baseline = self._fit(clean, {"ewa"})
        assert baseline is not None
        np.testing.assert_allclose(
            refitted[0]["ewa"].point, baseline[0]["ewa"].point, atol=1e-9
        )

    def test_a_non_object_pointer_does_not_abort_stateful_serving(self, tmp_path):
        config = write_config(tmp_path)
        state_dir = config.artifacts_dir / "state"
        state_dir.mkdir(parents=True)
        (state_dir / "latest.json").write_text("[]", encoding="utf-8")

        fitted = self._fit(config, {"ewa"})

        assert fitted is not None
        assert fitted[0]["ewa"].point.size > 0
        assert (state_dir / "latest.json").read_text(encoding="utf-8") == "[]"

    def test_observability_is_snapshotted_through_the_serving_path(self, tmp_path):
        config = write_config(tmp_path)
        assert self._fit(config, {"ewa"}) is not None
        assert list((config.artifacts_dir / "observability").rglob("*.json")), (
            "the dashboard's only serving-path hook must actually fire"
        )


class TestMinutelyPromotedPlan:
    """A promoted minutely selection replaces the config-tau construction."""

    def setup_case(self):
        import numpy as np

        helper = TestMinutelySinglePass()
        snapshot, path = helper.make_snapshot()
        blend = helper.blend(path, "equal_weight")  # un-anchored path
        return snapshot, blend, np.asarray(path)

    def plans(self, method_id, buckets=("0-5m", "5-15m", "15-30m", "30-45m", "45-60m")):
        from grounded_weather_forecast.serve.predict import _plan_from_selection

        plan = _plan_from_selection(method_id, None)
        assert plan is not None
        from dataclasses import replace

        plan = replace(
            plan,
            selection=Selection(
                method_id,
                "promoted minutely",
                n=30,
                mae=1.0,
                release_id="minute-release",
                truth_semantics="inst",
            ),
        )
        return {"temp_c": dict.fromkeys(buckets, plan)}

    def test_promoted_tau_replaces_the_config_decay(self, la_config):
        import math

        from grounded_weather_forecast.serve.predict import minutely_product

        snapshot, blend, _path = self.setup_case()
        points = minutely_product(
            snapshot,
            {"temp_c": blend},
            la_config,
            plans=self.plans("minutely_anchor_tau_0.25h"),
        )
        # lead-zero extrapolation = 19.0; residual = 10 - 19 = -9; minute 1
        # under tau 0.25h: weight exp(-(1/60)/0.25)
        expected = (19.0 + 1.0 / 60.0) + math.exp(-(1.0 / 60.0) / 0.25) * -9.0
        assert points[0].temp_c == pytest.approx(expected, abs=0.001)
        assert points[0].methods["temp_c"] == "minutely_anchor_tau_0.25h"

    def test_persistence_plan_serves_the_flat_observation(self, la_config):
        from grounded_weather_forecast.serve.predict import minutely_product

        snapshot, blend, _path = self.setup_case()
        points = minutely_product(
            snapshot,
            {"temp_c": blend},
            la_config,
            plans=self.plans("minutely_persistence"),
        )
        assert points[0].temp_c == pytest.approx(10.0)
        assert points[59].temp_c == pytest.approx(10.0)
        assert points[30].methods["temp_c"] == "minutely_persistence"
        assert points[30].release_ids == {"temp_c": "minute-release"}
        assert points[30].selection_reasons == {"temp_c": "promoted minutely"}
        assert points[30].truth_semantics == {"temp_c": "inst"}

    def test_already_anchored_path_ignores_the_plan(self, la_config):
        from grounded_weather_forecast.serve.predict import minutely_product

        helper = TestMinutelySinglePass()
        snapshot, path = helper.make_snapshot()
        points = minutely_product(
            snapshot,
            {"temp_c": helper.blend(path, "anchored_fitted_grounded")},
            la_config,
            plans=self.plans("minutely_anchor_full"),
        )
        # identical to the no-plan anchored case: pure interpolation
        assert points[0].temp_c == pytest.approx(19.0 + 1.0 / 60.0, abs=0.001)
        assert points[0].methods["temp_c"] == "anchored_hourly_blend"
        assert not points[0].release_ids
        assert not points[0].selection_reasons

    def test_missing_observation_degrades_to_labeled_interp(self, la_config):
        from grounded_weather_forecast.serve.predict import minutely_product

        snapshot, blend, _path = self.setup_case()
        snapshot = replace_snapshot_observation(snapshot, {})
        points = minutely_product(
            snapshot,
            {"temp_c": blend},
            la_config,
            plans=self.plans("minutely_anchor_full"),
        )
        assert points[0].temp_c == pytest.approx(19.0 + 1.0 / 60.0, abs=0.001)
        assert points[0].methods["temp_c"] == "minutely_interp"
        assert not points[0].release_ids

    def test_bucket_scoped_plan_leaves_other_buckets_on_the_fallback(self, la_config):
        from grounded_weather_forecast.serve.predict import minutely_product

        snapshot, blend, _path = self.setup_case()
        points = minutely_product(
            snapshot,
            {"temp_c": blend},
            la_config,
            plans=self.plans("minutely_persistence", buckets=("0-5m",)),
        )
        assert points[0].methods["temp_c"] == "minutely_persistence"
        # minute 10 has no plan for its bucket: config-tau fallback label
        assert points[9].methods["temp_c"] == "anchored_hourly_blend"
        assert points[0].release_ids == {"temp_c": "minute-release"}
        assert not points[9].release_ids

    def test_failed_fit_has_no_selected_release(self, la_config, monkeypatch):
        import polars as pl
        import grounded_weather_forecast.serve.predict as module

        snapshot, blend, _ = self.setup_case()
        monkeypatch.setattr(module, "_fitted_minutely_predictor", lambda *_: None)
        selections = {
            ("minutely", "temp_c", "0-5m"): Selection(
                "minutely_ramp", "promoted", release_id="minute-release"
            )
        }
        plans = module._minutely_plans(la_config, selections, pl.DataFrame())
        assert plans is None
        point = module.minutely_product(
            snapshot, {"temp_c": blend}, la_config, plans=plans
        )[0]
        assert point.methods["temp_c"] == "anchored_hourly_blend"
        assert not point.release_ids

    def test_served_minutely_history_drives_guarded_replacement(
        self, la_config, tmp_path, monkeypatch
    ):
        from dataclasses import replace
        from datetime import datetime
        import polars as pl
        from grounded_weather_forecast.backtest.scores import (
            SCORES_SCHEMA,
            write_scores,
        )
        from grounded_weather_forecast.evaluation import (
            code_identity,
            config_fingerprint,
        )
        from grounded_weather_forecast.serve.predict import (
            minutely_product,
            _minutely_plans,
        )
        from grounded_weather_forecast.serve.schema import Forecast
        from grounded_weather_forecast.serve.history import append_history
        from grounded_weather_forecast.reports.verification import verify_history
        import grounded_weather_forecast.serve.selection as selection_module

        snapshot, blend, _ = self.setup_case()
        key = ("minutely", "temp_c", "0-5m")
        rows = [
            {
                "method_id": method,
                "variable": "temp_c",
                "product": "minutely",
                "source_kind": "live",
                "evaluation_id": "minute-eval",
                "evaluation_created_at": snapshot.issue_time,
                "dataset_fingerprint": "dataset",
                "config_fingerprint": config_fingerprint(la_config),
                "code_version": code_identity(),
                "source_set_json": '["nws"]',
                "feature_set_json": "[]",
                "semantics": "inst",
                "window": "expanding",
                "issue_time": snapshot.issue_time - timedelta(hours=step + 1),
                "valid_time": snapshot.issue_time
                - timedelta(hours=step + 1)
                + timedelta(minutes=1),
                "lead_hours": 1 / 60,
                "lead_bucket": "0-5m",
                "y_pred": error,
                "y_true": 0.0,
            }
            for method, error in (
                ("minutely_persistence", 1.0),
                ("minutely_interp", 1.2),
            )
            for step in range(30)
        ]
        scores_dir = tmp_path / "scores"
        write_scores(
            pl.DataFrame(rows, schema=SCORES_SCHEMA),
            scores_dir / "scores_minute.parquet",
        )
        monkeypatch.setattr(
            selection_module, "_live_verification", lambda *_: pl.DataFrame()
        )
        selections = select_methods(la_config, scores_dir, dataset_id="dataset")
        selected = selections[key]
        assert selected.method_id == "minutely_persistence"
        plans = _minutely_plans(la_config, selections, pl.DataFrame())
        truth_rows = []
        path = tmp_path / "history.parquet"
        for hour in range(6):
            current = replace(
                snapshot, issue_time=snapshot.issue_time + timedelta(hours=hour)
            )
            points = minutely_product(
                current,
                {"temp_c": blend},
                la_config,
                plans=plans,
            )
            assert points[0].release_ids == {"temp_c": selected.release_id}
            forecast = Forecast(
                SCHEMA_VERSION,
                current.issue_time.isoformat(),
                0.0,
                0.0,
                "dataset",
                [],
                None,
                points,
                [],
                [],
            )
            append_history(forecast, path)
            truth_rows.extend(
                {
                    "ts": datetime.fromisoformat(point.valid_time),
                    "temp_c": point.temp_c - 3.0,
                }
                for point in points
            )
        live = verify_history(
            path, pl.DataFrame(), truth_minute=pl.DataFrame(truth_rows)
        )
        monkeypatch.setattr(selection_module, "_live_verification", lambda *_: live)
        result = select_methods(la_config, scores_dir, dataset_id="dataset")[key]
        assert result.method_id == "minutely_interp"
        assert result.release_id != selected.release_id
        assert "demoted minutely_persistence" in result.reason


def replace_snapshot_observation(snapshot, observation):
    from dataclasses import replace

    return replace(snapshot, observation=observation)


def test_minutely_release_list_preserves_hourly_daily_readiness(tmp_path):
    config = build_fixture(tmp_path)
    selections = {
        ("minutely", "temp_c", bucket): Selection(
            "minutely_interp",
            "promoted",
            release_id="minute-release",
            truth_semantics="inst",
        )
        for bucket in ("0-5m", "5-15m", "15-30m", "30-45m", "45-60m")
    }
    selections[("minutely", "pop", "0-5m")] = Selection(
        "minutely_anchor_full", "selected", release_id="unused-native-release"
    )
    forecast = predict(config, selections, now=NOW)
    assert forecast.release_ids == ["minute-release"]
    assert forecast.status == "degraded"
    assert all("pop" not in point.release_ids for point in forecast.minutely)
    raw = json.loads(forecast.to_json())
    raw["schema_version"] = 5
    for point in raw["minutely"]:
        for key in ("release_ids", "selection_reasons", "truth_semantics"):
            point.pop(key)
    from grounded_weather_forecast.serve.schema import Forecast

    old = Forecast.from_json(json.dumps(raw))
    assert all(
        not point.release_ids
        and not point.selection_reasons
        and not point.truth_semantics
        for point in old.minutely
    )
