"""The rolling-origin backtest loop.

Per fold and variable, a FRESH blender instance is constructed from its
registry factory (stateful instances can never leak across folds), fitted on
rows whose truth was knowable at the fold origin, and evaluated on the
snapshots issued in the following step. Output is the scores frame.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from omni_forecast.backtest.scores import SCORES_SCHEMA, empty_scores
from omni_forecast.backtest.splits import (
    WindowMode,
    daily_truth_known_at,
    fold_plans,
    hourly_truth_known_at,
)
from omni_forecast.blenders.registry import (
    BlenderFactory,
    get_factory,
    supports_product,
)
from omni_forecast.config import Config
from omni_forecast.contracts import Product, TruthSemantics, VariableSpec
from omni_forecast.dataset.matrix import (
    assert_single_kind,
    matrix_sources,
    to_supervised_slice,
    truth_column_for,
)
from omni_forecast.evaluation import EvaluationRun
from omni_forecast.timeutil import local_day_start_utc


@dataclass(frozen=True, slots=True)
class BacktestRequest:
    variables: tuple[VariableSpec, ...]
    methods: tuple[str, ...]
    window: WindowMode = "expanding"
    daily: bool = False
    semantics: TruthSemantics | dict[str, TruthSemantics] = TruthSemantics.INSTANTANEOUS

    def semantics_for(self, variable: VariableSpec) -> TruthSemantics:
        if not variable.has_dual_semantics:
            return TruthSemantics.INSTANTANEOUS
        if isinstance(self.semantics, dict):
            return self.semantics.get(variable.name, TruthSemantics.INSTANTANEOUS)
        return self.semantics


def _lead_and_valid(
    frame: pl.DataFrame, *, daily: bool, timezone_name: str
) -> pl.DataFrame:
    if daily:
        valid_times = [
            local_day_start_utc(day, timezone_name)
            for day in frame["forecast_date"].to_list()
        ]
        return pl.DataFrame(
            {
                "issue_time": frame["issue_time"],
                "valid_time": valid_times,
                "lead_hours": frame["lead_days"].cast(pl.Float64) * 24.0,
                "lead_bucket": frame["lead_bucket"],
            },
            schema_overrides={"valid_time": pl.Datetime("us", "UTC")},
        )
    return frame.select("issue_time", "valid_time", "lead_hours", "lead_bucket")


def run_backtest(
    matrix: pl.DataFrame,
    request: BacktestRequest,
    config: Config,
    *,
    factories: dict[str, BlenderFactory] | None = None,
) -> pl.DataFrame:
    """Run all folds x variables x methods; returns the scores frame.

    ``factories`` overrides registry lookup (used by leakage tests to observe
    instance lifecycles).
    """
    if matrix.is_empty():
        return empty_scores()
    kind = assert_single_kind(matrix)
    product = "daily" if request.daily else "hourly"
    product_kind = Product.DAILY if request.daily else Product.HOURLY
    semantics_by_variable = {
        variable.name: request.semantics_for(variable) for variable in request.variables
    }
    evaluation = EvaluationRun.create(
        config,
        source_kind=kind,
        source_set=matrix_sources(matrix),
        product=product,
        window=request.window,
        semantics=semantics_by_variable,
        methods=request.methods,
    )
    truth_known = (
        daily_truth_known_at(matrix, config.station.timezone)
        if request.daily
        else hourly_truth_known_at(matrix)
    )
    folds = fold_plans(
        matrix["issue_time"], truth_known, config.backtest, request.window
    )
    results: list[pl.DataFrame] = []
    for fold in folds:
        train_frame = matrix[fold.train_rows]
        test_frame = matrix[fold.test_rows]
        for variable in request.variables:
            semantics = semantics_by_variable[variable.name]
            truth_column = truth_column_for(variable, semantics)
            if truth_column not in matrix.columns:
                continue
            train_scored = train_frame.filter(pl.col(truth_column).is_not_null())
            test_scored = test_frame.filter(pl.col(truth_column).is_not_null())
            if train_scored.is_empty() or test_scored.is_empty():
                continue
            train_slice = to_supervised_slice(
                train_scored, variable, daily=request.daily, semantics=semantics
            )
            test_slice = to_supervised_slice(
                test_scored, variable, daily=request.daily, semantics=semantics
            )
            keys = _lead_and_valid(
                test_scored,
                daily=request.daily,
                timezone_name=config.station.timezone,
            )
            for method_id in request.methods:
                if not supports_product(method_id, product_kind, variable):
                    continue
                factory = (
                    factories[method_id]
                    if factories is not None
                    else get_factory(method_id)
                )
                blender = factory().fit(train_slice)
                prediction = blender.predict(test_slice.x)
                results.append(
                    keys.with_columns(
                        pl.lit(method_id).alias("method_id"),
                        pl.lit(variable.name).alias("variable"),
                        pl.lit(product).alias("product"),
                        pl.lit(kind).alias("source_kind"),
                        pl.lit(evaluation.evaluation_id).alias("evaluation_id"),
                        pl.lit(datetime.fromisoformat(evaluation.created_at)).alias(
                            "evaluation_created_at"
                        ),
                        pl.lit(evaluation.dataset_fingerprint).alias(
                            "dataset_fingerprint"
                        ),
                        pl.lit(json.dumps(evaluation.source_set)).alias(
                            "source_set_json"
                        ),
                        pl.lit(semantics.value).alias("semantics"),
                        pl.lit(evaluation.code_version).alias("code_version"),
                        pl.lit(evaluation.config_fingerprint).alias(
                            "config_fingerprint"
                        ),
                        pl.lit(request.window).alias("window"),
                        pl.lit(fold.origin).alias("fold_origin"),
                        pl.Series("y_pred", prediction.point, dtype=pl.Float64)
                        .fill_nan(None)
                        .alias("y_pred"),
                        pl.Series("y_true", test_slice.y, dtype=pl.Float64).alias(
                            "y_true"
                        ),
                        pl.lit(json.dumps(prediction.quantile_levels)).alias(
                            "quantile_levels_json"
                        ),
                        pl.Series(
                            "quantiles_json",
                            [json.dumps(row.tolist()) for row in prediction.quantiles]
                            if prediction.quantiles is not None
                            else [None] * test_slice.x.n_rows,
                            dtype=pl.String,
                        ),
                    ).select(list(SCORES_SCHEMA))
                )
    if not results:
        return empty_scores()
    scores = pl.concat(results)
    return scores.cast(SCORES_SCHEMA)


def variables_from_names(
    names: Sequence[str], lookup: Sequence[VariableSpec]
) -> tuple[VariableSpec, ...]:
    by_name = {spec.name: spec for spec in lookup}
    missing = [name for name in names if name not in by_name]
    if missing:
        msg = f"unknown variables: {missing}; known: {sorted(by_name)}"
        raise ValueError(msg)
    return tuple(by_name[name] for name in names)
