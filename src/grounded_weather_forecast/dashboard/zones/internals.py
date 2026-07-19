"""Zone E: model internals — the glass box, read from observability snapshots."""

import json
import math
from collections.abc import Mapping

import polars as pl

from grounded_weather_forecast.blenders.gbm import HAVE_LIGHTGBM
from grounded_weather_forecast.dashboard.charts import bar_chart, line_chart
from grounded_weather_forecast.dashboard.context import DashboardContext
from grounded_weather_forecast.dashboard.copy import PANEL_COPY, ZONE_INTROS
from grounded_weather_forecast.dashboard.derive import Derived
from grounded_weather_forecast.dashboard.model import Panel, Stat, TableSpec, Zone
from grounded_weather_forecast.dashboard.zones.common import empty_panel, fmt
from grounded_weather_forecast.serve.observability import ObservabilitySnapshot

_IDENTITY = (0.0, 1.0)
_WEIGHT_FLOOR = 0.05  # mirrors blenders/anchoring._WEIGHT_FLOOR


def _grounding_state(
    snapshots: tuple[ObservabilitySnapshot, ...],
) -> tuple[ObservabilitySnapshot, Mapping[str, object]] | None:
    for snapshot in snapshots:
        grounding = snapshot.state.get("grounding")
        if isinstance(grounding, Mapping) and grounding:
            return snapshot, grounding
    return None


def _grounding_panel(ctx: DashboardContext) -> Panel:
    found = _grounding_state(ctx.observability_states)
    if found is None:
        return empty_panel(
            "e1",
            "e1",
            "Grounding coefficients",
            "info",
            "no grounding snapshot persisted yet — internals land in "
            "artifacts/observability/ on each `predict` run",
        )
    snapshot, grounding = found
    buckets: list[str] = []
    for per_source in grounding.values():
        if isinstance(per_source, Mapping):
            per_bucket = per_source.get("buckets")
            if isinstance(per_bucket, Mapping):
                buckets.extend(str(label) for label in per_bucket)
    bucket_order = sorted(set(buckets))
    columns = ("source", "global", *bucket_order)
    rows: list[tuple[str, ...]] = []
    classes: list[tuple[str, ...]] = []
    identity_cells = 0
    for source in sorted(grounding):
        per_source = grounding[source]
        if not isinstance(per_source, Mapping):
            continue
        cells = [str(source)]
        row_classes = [""]
        entries: list[tuple[str, object]] = [("global", per_source.get("global"))]
        per_bucket = per_source.get("buckets")
        bucket_map = per_bucket if isinstance(per_bucket, Mapping) else {}
        entries.extend((label, bucket_map.get(label)) for label in bucket_order)
        for _label, coefficients in entries:
            if (
                isinstance(coefficients, (list, tuple))
                and len(coefficients) == 2
                and isinstance(coefficients[0], (int, float))
                and isinstance(coefficients[1], (int, float))
            ):
                a, b = float(coefficients[0]), float(coefficients[1])
                identity = math.isclose(a, _IDENTITY[0]) and math.isclose(
                    b, _IDENTITY[1]
                )
                identity_cells += int(identity)
                cells.append(
                    f"{a:+.2f}" if math.isclose(b, 1.0) else f"{a:+.2f}·{b:.2f}"
                )
                row_classes.append("cell-muted" if identity else "")
            else:
                cells.append("")
                row_classes.append("heat-none")
        rows.append(tuple(cells))
        classes.append(tuple(row_classes))
    return Panel(
        panel_id="e1",
        title=f"Grounding coefficients ({snapshot.method_id})",
        status="ok",
        copy=PANEL_COPY["e1"],
        stats=(
            Stat("slice", f"{snapshot.product}.{snapshot.variable}"),
            Stat("identity fallbacks", str(identity_cells)),
            Stat("as of", snapshot.issue_time or snapshot.created_at),
        ),
        intro=(
            "Cell = fitted intercept a (bias-only default keeps slope b at 1; "
            "a·b shown when the slope is free). Muted cells fell back to "
            "IDENTITY — no correction is applied there."
        ),
        table=TableSpec(columns=columns, rows=tuple(rows), cell_classes=tuple(classes)),
    )


def _expert_snapshot(
    snapshots: tuple[ObservabilitySnapshot, ...],
) -> ObservabilitySnapshot | None:
    for snapshot in snapshots:
        if snapshot.method_id in ("ewa", "boa"):
            return snapshot
    return None


def _weights_trajectory(history: pl.DataFrame) -> Panel | None:
    if history.is_empty():
        return None
    group = (
        history.sort("issue_time")
        .group_by(["method_id", "product", "variable"], maintain_order=True)
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    if group.is_empty() or group.row(0, named=True)["n"] < 2:
        return None
    key = group.row(0, named=True)
    rows = history.filter(
        (pl.col("method_id") == key["method_id"])
        & (pl.col("product") == key["product"])
        & (pl.col("variable") == key["variable"])
    ).sort("issue_time")
    labels: list[str] = []
    per_source: dict[str, list[float | None]] = {}
    bucket_label: str | None = None
    for row in rows.iter_rows(named=True):
        try:
            state = json.loads(row["state_json"])
        except (TypeError, ValueError):
            continue
        sources = state.get("sources")
        bucket_map = state.get("buckets")
        if not isinstance(sources, list) or not isinstance(bucket_map, Mapping):
            continue
        if bucket_label is None:
            bucket_label = max(
                (
                    (label, bucket.get("steps", 0))
                    for label, bucket in bucket_map.items()
                    if isinstance(bucket, Mapping)
                ),
                key=lambda item: item[1],
                default=(None, 0),
            )[0]
        if bucket_label is None:
            continue
        bucket = bucket_map.get(bucket_label)
        weights = bucket.get("weights") if isinstance(bucket, Mapping) else None
        if not isinstance(weights, list) or len(weights) != len(sources):
            continue
        labels.append(str(row["issue_time"])[:16])
        for values in per_source.values():
            values.append(None)
        for source, weight in zip(sources, weights, strict=True):
            series = per_source.setdefault(str(source), [None] * len(labels))
            series[-1] = float(weight) if isinstance(weight, (int, float)) else None
    if len(labels) < 2 or not per_source:
        return None
    return Panel(
        panel_id="e2",
        title=(
            f"Expert weight trajectory — {key['method_id']} "
            f"{key['product']}.{key['variable']} [{bucket_label}]"
        ),
        status="ok",
        copy=PANEL_COPY["e2"],
        chart=line_chart(
            labels,
            sorted(per_source.items())[:8],
            y_label="weight",
        ),
    )


def _expert_weights_panel(ctx: DashboardContext) -> Panel:
    trajectory = _weights_trajectory(ctx.observability_history)
    if trajectory is not None:
        return trajectory
    snapshot = _expert_snapshot(ctx.observability_states)
    if snapshot is None:
        return empty_panel(
            "e2",
            "e2",
            "Expert weights",
            "info",
            "no ewa/boa snapshot yet — expert weights appear once an online "
            "method is fit during a `predict` run",
        )
    sources = snapshot.state.get("sources")
    bucket_map = snapshot.state.get("buckets")
    if not isinstance(sources, list) or not isinstance(bucket_map, Mapping):
        return empty_panel(
            "e2", "e2", "Expert weights", "info", "expert state is unreadable"
        )
    labels = sorted(str(label) for label in bucket_map)
    series = []
    for index, source in enumerate(sources[:8]):
        values: list[float | None] = []
        for label in labels:
            bucket = bucket_map.get(label)
            weights = bucket.get("weights") if isinstance(bucket, Mapping) else None
            values.append(
                float(weights[index])
                if isinstance(weights, list) and index < len(weights)
                else None
            )
        series.append((str(source), values))
    return Panel(
        panel_id="e2",
        title=f"Expert weights — {snapshot.method_id} "
        f"{snapshot.product}.{snapshot.variable}",
        status="ok",
        copy=PANEL_COPY["e2"],
        intro=(
            "Only the current weights exist so far; a trajectory line chart "
            "replaces this bar view once two or more serve snapshots span "
            "the history window."
        ),
        chart=bar_chart(labels, series, y_label="weight"),
    )


def _gbm_panel(ctx: DashboardContext) -> Panel:
    if not HAVE_LIGHTGBM:
        return empty_panel(
            "e3",
            "e3",
            "GBM feature importances",
            "red",
            "lightgbm is not importable on this host, so the `gbm` method is "
            "silently absent from the registry — install the wheel to "
            "restore the nonlinear ceiling",
        )
    snapshot = next((s for s in ctx.observability_states if s.method_id == "gbm"), None)
    if snapshot is None:
        return empty_panel(
            "e3",
            "e3",
            "GBM feature importances",
            "info",
            "no gbm snapshot yet — importances land once the GBM is "
            "selected and fit during a `predict` run",
        )
    gain = snapshot.state.get("importance_gain")
    if not isinstance(gain, Mapping) or not gain:
        return empty_panel(
            "e3", "e3", "GBM feature importances", "info", "gbm state unreadable"
        )
    ranked = sorted(
        ((str(name), float(value)) for name, value in gain.items()),  # type: ignore[arg-type]
        key=lambda item: item[1],
        reverse=True,
    )[:15]
    return Panel(
        panel_id="e3",
        title=f"GBM importances — {snapshot.product}.{snapshot.variable}",
        status="ok",
        copy=PANEL_COPY["e3"],
        stats=(Stat("trees", fmt(snapshot.state.get("num_trees"))),),
        chart=bar_chart(
            [name for name, _ in ranked],
            [("gain", [value for _, value in ranked])],
            y_label="gain",
            horizontal=True,
        ),
    )


def _anchoring_panel(ctx: DashboardContext) -> Panel:
    taus: list[tuple[str, float | None]] = []
    for snapshot in ctx.observability_states:
        if "tau_hours" in snapshot.state:
            tau = snapshot.state.get("tau_hours")
            taus.append(
                (
                    f"{snapshot.method_id} {snapshot.product}.{snapshot.variable}",
                    float(tau) if isinstance(tau, (int, float)) else None,
                )
            )
    if not taus:
        return empty_panel(
            "e4",
            "e4",
            "Anchoring decay",
            "info",
            "no anchored-method snapshot yet — tau appears once an anchored "
            "method is fit during a `predict` run",
        )
    fitted = next((tau for _name, tau in taus if tau is not None), None)
    chart = None
    if fitted is not None:
        leads = [lead / 2.0 for lead in range(25)]
        weights = [
            weight if (weight := math.exp(-lead / fitted)) >= _WEIGHT_FLOOR else 0.0
            for lead in leads
        ]
        chart = line_chart(
            [f"{lead:g}h" for lead in leads],
            [(f"exp(-lead/{fitted:g}h), floored", weights)],
            y_label="anchor weight",
        )
    rows = tuple(
        (name, "none (grid chose no anchoring)" if tau is None else f"{tau:g} h")
        for name, tau in taus
    )
    return Panel(
        panel_id="e4",
        title="Anchoring decay timescale",
        status="ok",
        copy=PANEL_COPY["e4"],
        table=TableSpec(columns=("method / slice", "tau"), rows=rows),
        chart=chart,
    )


def _rankings_panel(ctx: DashboardContext) -> Panel:
    snapshot = next(
        (s for s in ctx.observability_states if s.method_id == "best_provider"),
        None,
    )
    fold_origins: set[object] = set()
    for frame in ctx.score_frames.values():
        if "fold_origin" in frame.columns and not frame.is_empty():
            fold_origins.update(frame["fold_origin"].unique().to_list())
    stats = (
        (Stat("backtested fold origins", str(len(fold_origins))),)
        if fold_origins
        else ()
    )
    if snapshot is None:
        return empty_panel(
            "e5",
            "e5",
            "Provider rankings",
            "info",
            "no best_provider snapshot yet — rankings appear once the "
            "method is fit during a `predict` run",
        )
    buckets = snapshot.state.get("buckets")
    bucket_map = buckets if isinstance(buckets, Mapping) else {}
    global_ranking = snapshot.state.get("global")
    rows = [
        (
            "global",
            " > ".join(str(name) for name in global_ranking)
            if isinstance(global_ranking, list)
            else "—",
        )
    ]
    rows.extend(
        (str(label), " > ".join(str(name) for name in ranking))
        for label, ranking in sorted(bucket_map.items())
        if isinstance(ranking, list)
    )
    return Panel(
        panel_id="e5",
        title=f"Provider rankings — {snapshot.product}.{snapshot.variable}",
        status="ok",
        copy=PANEL_COPY["e5"],
        stats=stats,
        table=TableSpec(
            columns=("lead bucket", "ranking (best first)"), rows=tuple(rows)
        ),
    )


def build(ctx: DashboardContext, derived: Derived) -> Zone:  # noqa: ARG001
    return Zone(
        zone_id="E",
        title="Model internals",
        intro=ZONE_INTROS["E"],
        panels=(
            _grounding_panel(ctx),
            _expert_weights_panel(ctx),
            _gbm_panel(ctx),
            _anchoring_panel(ctx),
            _rankings_panel(ctx),
        ),
    )
