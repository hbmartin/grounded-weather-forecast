"""Choose complete evaluations before materializing their score rows."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import cast

import polars as pl

from grounded_weather_forecast.backtest.scores import scan_score_file
from grounded_weather_forecast.config import PromotionConfig
from grounded_weather_forecast.contracts import MixedProvenanceError, TruthSemantics
from grounded_weather_forecast.reports.leaderboard import gate_references
from grounded_weather_forecast.serve.evidence import (
    CONTEXT_COLUMNS,
    ContextAccumulator,
    EvaluationContext,
)

type SliceKey = tuple[str, str, str]
_SLICE_COLUMNS = ("product", "variable", "lead_bucket")
_FINGERPRINT_COLUMNS = ("dataset_fingerprint", "config_fingerprint", "code_version")
MAX_SCORE_SCAN_ATTEMPTS = 3
SCORE_BATCH_ROWS = 65_536
_METADATA_COLUMNS = (
    "evaluation_id",
    *CONTEXT_COLUMNS,
    "dataset_fingerprint",
    "evaluation_created_at",
    "product",
    "variable",
    "semantics",
    "lead_bucket",
)
_SCORE_COLUMNS = (
    "method_id",
    *_METADATA_COLUMNS,
    "issue_time",
    "valid_time",
    "lead_hours",
    "y_pred",
    "y_true",
)


class EvidenceChangedError(RuntimeError):
    """A file changed between its metadata and score reads."""

    def __init__(self, message: str, evaluations: frozenset[str] = frozenset()) -> None:
        super().__init__(message)
        self.evaluations = evaluations


@dataclass(frozen=True, slots=True)
class ScoreIdentity:
    dataset: str
    configuration: str
    code: str
    as_of: datetime | None = None
    semantics: Mapping[str, TruthSemantics] | None = None

    def fingerprint_predicate(self) -> pl.Expr:
        return (
            (pl.col("dataset_fingerprint") == self.dataset)
            & (pl.col("config_fingerprint") == self.configuration)
            & (pl.col("code_version") == self.code)
        )

    def predicate(self) -> pl.Expr:
        predicate = self.fingerprint_predicate()
        if self.as_of is not None:
            predicate &= (pl.col("evaluation_created_at") <= self.as_of) & (
                pl.col("valid_time") <= self.as_of
            )
        return predicate

    def semantics_predicate(self) -> pl.Expr:
        predicate = pl.lit(True)  # noqa: FBT003 -- expression literal, not a flag
        for variable, semantic in (self.semantics or {}).items():
            predicate &= (
                (pl.col("product") != "hourly")
                | (pl.col("variable") != variable)
                | (pl.col("semantics") == semantic.value)
            )
        return predicate

    def accepts(self, product: str, variable: str, semantic: object) -> bool:
        expected = (self.semantics or {}).get(variable)
        return (
            product != "hourly"
            or expected is None
            or (semantic or "inst") == expected.value
        )


@dataclass(frozen=True, slots=True)
class ScoreScanResult:
    slices: Mapping[SliceKey, pl.DataFrame] = field(default_factory=dict)
    contexts: Mapping[str, EvaluationContext] = field(default_factory=dict)
    cutoffs: Mapping[SliceKey, datetime | None] = field(default_factory=dict)


@dataclass(slots=True)
class ScoreFragment:
    rows: int = 0
    methods: set[str] = field(default_factory=set)


@dataclass(slots=True)
class SliceEvidence:
    evaluation_id: str
    created_at: datetime
    key: SliceKey
    methods: set[str] = field(default_factory=set)
    fragments: dict[Path, ScoreFragment] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationCatalog:
    contexts: ContextAccumulator = field(default_factory=ContextAccumulator)
    identities: dict[str, tuple[str, ...]] = field(default_factory=dict)
    slices: dict[tuple[str, SliceKey], SliceEvidence] = field(default_factory=dict)
    path_evaluations: dict[Path, frozenset[str]] = field(default_factory=dict)

    def add_identities(self, path: Path, probe: pl.DataFrame) -> None:
        self.path_evaluations[path] = frozenset(str(e) for e in probe["evaluation_id"])
        for row in probe.iter_rows(named=True):
            evaluation = str(row["evaluation_id"])
            identity = tuple(str(row[name]) for name in _FINGERPRINT_COLUMNS)
            previous = self.identities.get(evaluation)
            if previous is not None and previous != identity:
                self.contexts.reject(evaluation)
            self.identities[evaluation] = identity

    def add_summary(
        self, path: Path, summary: pl.DataFrame, identity: ScoreIdentity
    ) -> None:
        self.contexts.add(summary)
        for row in summary.iter_rows(named=True):
            created = row["evaluation_created_at"]
            if not isinstance(created, datetime) or row["lead_bucket"] is None:
                continue
            key = cast(SliceKey, tuple(str(row[column]) for column in _SLICE_COLUMNS))
            if not identity.accepts(key[0], key[1], row["semantics"]):
                continue
            evaluation = str(row["evaluation_id"])
            marker = (evaluation, key)
            evidence = self.slices.get(marker)
            if evidence is None:
                evidence = SliceEvidence(evaluation, created, key)
                self.slices[marker] = evidence
            evidence.methods.update(str(method) for method in row["methods"])
            fragment = evidence.fragments.get(path)
            if fragment is None:
                fragment = ScoreFragment()
                evidence.fragments[path] = fragment
            fragment.rows += int(row["n"])
            fragment.methods.update(str(method) for method in row["methods"])

    def newest(self, promotion: PromotionConfig) -> dict[SliceKey, SliceEvidence]:
        selected: dict[SliceKey, SliceEvidence] = {}
        for evidence in self.slices.values():
            if evidence.evaluation_id in self.contexts.invalid:
                continue
            references = gate_references(
                evidence.key[1], promotion, product=evidence.key[0]
            )
            if not set(references) <= evidence.methods:
                continue
            previous = selected.get(evidence.key)
            marker = (evidence.created_at, evidence.evaluation_id)
            if previous is None or marker > (
                previous.created_at,
                previous.evaluation_id,
            ):
                selected[evidence.key] = evidence
        return {key: selected[key] for key in sorted(selected)}


def scan_scores(path: Path) -> pl.LazyFrame:
    """One seam for lock-free reads and deterministic prune-race tests."""
    return scan_score_file(path)


def _live_scan(path: Path) -> pl.LazyFrame | None:
    scan = scan_scores(path)
    kinds = set(
        scan.select("source_kind")
        .unique()
        .collect(engine="streaming")["source_kind"]
        .to_list()
    )
    if len(kinds) > 1:
        raise MixedProvenanceError(
            f"scores at {path} mix source kinds {sorted(map(str, kinds))}"
        )
    if kinds != {"live"}:
        return None
    if "semantics" in scan.collect_schema():
        scan = scan.with_columns(pl.col("semantics").fill_null("inst"))
    return scan


def _catalog_file(
    catalog: EvaluationCatalog, path: Path, identity: ScoreIdentity
) -> None:
    scan = _live_scan(path)
    if scan is None or not set(_SCORE_COLUMNS) <= set(scan.collect_schema().names()):
        return
    probe = (
        scan.select("evaluation_id", *_FINGERPRINT_COLUMNS)
        .unique()
        .collect(engine="streaming")
    )
    catalog.add_identities(path, probe)
    matches = probe.filter(identity.fingerprint_predicate())
    if matches.is_empty():
        return
    # Aggregate inside the lazy scan: only small metadata/method summaries
    # cross into Python, rather than every numeric and string-view buffer.
    summary = (
        scan.filter(identity.predicate())
        .group_by(*_METADATA_COLUMNS, maintain_order=True)
        .agg(pl.col("method_id").unique().alias("methods"), pl.len().alias("n"))
        .collect(engine="streaming")
    )
    catalog.add_summary(path, summary, identity)


def _slice_predicate(evidence: SliceEvidence) -> pl.Expr:
    expression = pl.col("evaluation_id") == evidence.evaluation_id
    for column, value in zip(_SLICE_COLUMNS, evidence.key, strict=True):
        expression &= pl.col(column) == value
    return expression


def _fragment_valid(
    frame: pl.DataFrame, evidence: SliceEvidence, path: Path, catalog: EvaluationCatalog
) -> bool:
    expected_fragment = evidence.fragments[path]
    if (
        frame.height != expected_fragment.rows
        or set(frame["method_id"].unique()) != expected_fragment.methods
    ):
        return False
    observed = ContextAccumulator()
    observed.add(frame)
    context = observed.contexts.get(evidence.evaluation_id)
    expected = catalog.contexts.contexts[evidence.evaluation_id]
    return (
        not observed.invalid
        and context is not None
        and context.metadata == expected.metadata
        and context.identity == expected.identity
        and context.semantics.get(evidence.key[1])
        == expected.semantics.get(evidence.key[1])
    )


def _materialize(
    catalog: EvaluationCatalog,
    selected: Mapping[SliceKey, SliceEvidence],
    identity: ScoreIdentity,
) -> dict[SliceKey, pl.DataFrame]:
    by_path: dict[Path, list[SliceEvidence]] = {}
    for evidence in selected.values():
        for path in evidence.fragments:
            by_path.setdefault(path, []).append(evidence)
    fragments: dict[SliceKey, list[pl.DataFrame]] = {key: [] for key in selected}
    for path, evidence_rows in by_path.items():
        predicate = pl.any_horizontal(
            *(_slice_predicate(evidence) for evidence in evidence_rows)
        )
        try:
            scan = _live_scan(path)
            if scan is None:
                raise EvidenceChangedError(f"source provenance changed: {path}")
            batches = (
                scan.filter(
                    identity.predicate() & identity.semantics_predicate() & predicate
                )
                .select(*_SCORE_COLUMNS)
                .collect_batches(chunk_size=SCORE_BATCH_ROWS, engine="streaming")
            )
            path_fragments: dict[SliceKey, list[pl.DataFrame]] = {
                evidence.key: [] for evidence in evidence_rows
            }
            for batch in batches:
                partitions = batch.partition_by(
                    "evaluation_id", *_SLICE_COLUMNS, as_dict=True
                )
                for evidence in evidence_rows:
                    if (
                        part := partitions.get((evidence.evaluation_id, *evidence.key))
                    ) is not None:
                        path_fragments[evidence.key].append(part)
            for evidence in evidence_rows:
                pieces = path_fragments[evidence.key]
                fragment = (
                    pl.concat(pieces)
                    if pieces
                    else pl.DataFrame(schema=scan.collect_schema()).select(
                        *_SCORE_COLUMNS
                    )
                )
                if not _fragment_valid(fragment, evidence, path, catalog):
                    raise EvidenceChangedError(f"evaluation fragment changed: {path}")
                fragments[evidence.key].append(fragment)
        except (FileNotFoundError, EvidenceChangedError) as error:
            raise EvidenceChangedError(
                str(error), catalog.path_evaluations[path]
            ) from error
    return {
        key: pl.concat(frames, how="diagonal_relaxed")
        for key, frames in fragments.items()
    }


def _scan_once(
    scores_dir: Path,
    identity: ScoreIdentity,
    promotion: PromotionConfig,
    *,
    excluded: frozenset[str] = frozenset(),
) -> ScoreScanResult:
    catalog = EvaluationCatalog()
    catalog.contexts.invalid.update(excluded)
    for path in sorted(scores_dir.glob("scores_*.parquet")):
        try:
            _catalog_file(catalog, path, identity)
        except (FileNotFoundError,) as error:  # noqa: B013 -- project convention
            raise EvidenceChangedError(
                str(error), catalog.path_evaluations.get(path, frozenset())
            ) from error
    while selected := catalog.newest(promotion):
        try:
            frames = _materialize(catalog, selected, identity)
        except (EvidenceChangedError,) as error:  # noqa: B013 -- project convention
            newly_invalid = error.evaluations - catalog.contexts.invalid
            if not newly_invalid:
                raise EvidenceChangedError(
                    str(error), frozenset(catalog.contexts.invalid) | error.evaluations
                ) from error
            catalog.contexts.invalid.update(newly_invalid)
            continue
        cutoffs = {
            key: value
            if isinstance(value := frame["valid_time"].max(), datetime)
            else None
            for key, frame in frames.items()
        }
        evaluations = {evidence.evaluation_id for evidence in selected.values()}
        return ScoreScanResult(
            slices=frames,
            contexts={
                key: catalog.contexts.contexts[key] for key in sorted(evaluations)
            },
            cutoffs=cutoffs,
        )
    return ScoreScanResult()


def compatible_slices(
    scores_dir: Path, identity: ScoreIdentity, promotion: PromotionConfig
) -> ScoreScanResult:
    excluded: frozenset[str] = frozenset()
    for _ in range(MAX_SCORE_SCAN_ATTEMPTS):
        try:
            return _scan_once(scores_dir, identity, promotion, excluded=excluded)
        except (EvidenceChangedError,) as error:  # noqa: B013 -- project convention
            excluded |= error.evaluations
    return ScoreScanResult()


def live_identity_sets(
    scores_dir: Path, *, skip_missing: bool
) -> tuple[bool, set[str], set[str], set[str]]:
    paths = sorted(scores_dir.glob("scores_*.parquet"))
    identities: dict[str, set[str]] = {name: set() for name in _FINGERPRINT_COLUMNS}
    for path in paths:
        try:
            scan = _live_scan(path)
            if scan is None:
                continue
            columns = [
                name for name in _FINGERPRINT_COLUMNS if name in scan.collect_schema()
            ]
            probe = (
                scan.select(columns).unique().collect(engine="streaming")
                if columns
                else pl.DataFrame()
            )
            for name in columns:
                identities[name].update(str(value) for value in probe[name])
        except (FileNotFoundError,):  # noqa: B013 -- project convention
            if not skip_missing:
                raise
    return (
        bool(paths),
        identities["dataset_fingerprint"],
        identities["config_fingerprint"],
        identities["code_version"],
    )
