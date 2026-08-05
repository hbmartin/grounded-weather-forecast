"""Anytime-valid sequential promotion evidence via betting e-processes.

The bootstrap MCS gate is re-run from scratch after every nightly backtest
with no multiplicity control across those re-runs — winners flap in and out
on overnight noise. An e-process is the sequential fix: per (slice x
candidate x reference) pair, a nonnegative supermartingale accumulates
evidence that the candidate beats the reference, one *new* collapsed loss
differential at a time. By Ville's inequality, ``sup_t e_t >= 1/alpha`` has
probability at most ``alpha`` under the null (reference at least as good),
no matter how many times or when the gate is consulted — the anytime
validity the fixed-alpha re-run lacks.

The bet is the ONS (online Newton step) betting martingale on bounded
scaled differentials — no likelihood model, tune-free, near-log-optimal
(Waudby-Smith & Ramdas; sequential-MCS framing per arXiv:2404.18678):

    z_t   = clip((loss_ref - loss_cand) / scale_{t-1}, -1, 1)
    e_t   = e_{t-1} * (1 + lam_t * z_t),  lam_t in [0, 1/2]
    lam   updated by ONS on the log-wealth gradient

``scale`` is the running max |differential| over *prior* observations —
predictable, so each factor has conditional mean <= 1 under the null.

State is persisted per (product, source_kind) under
``artifacts/eprocess/`` and keyed by slice, candidate, and reference. A
``cursor_valid_time`` per entry makes updates idempotent: the expanding
backtest re-scores overlapping folds, and a bet, once made, is never
revised. The header pins config fingerprint and code version — either
changing resets the store, because a different implementation makes
"reference weakly better" a different null. Deliberately NOT keyed to the
dataset fingerprint, which rotates on every ingested observation.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
import polars as pl

from grounded_weather_forecast.contracts import FloatArray
from grounded_weather_forecast.storage import atomic_write_text, locked_path

if TYPE_CHECKING:
    from grounded_weather_forecast.config import PromotionConfig

_ONS_RATE = 2.0 / (2.0 - math.log(3.0))
_LAMBDA_INIT = 0.1
_LAMBDA_MAX = 0.5
_SCHEMA_VERSION = 1


@dataclass
class EProcessEntry:
    """One pair's accumulated wealth and betting state."""

    log_e: float = 0.0
    t: int = 0
    lam: float = _LAMBDA_INIT
    ons_a: float = 1.0
    scale: float = 0.0
    cursor_valid_time: str | None = None


def pair_key(
    product: str,
    variable: str,
    semantics: str,
    lead_bucket: str,
    candidate: str,
    reference: str,
) -> str:
    return f"{product}|{variable}|{semantics}|{lead_bucket}|{candidate}|{reference}"


@dataclass
class EProcessStore:
    """Locked, atomically-written JSON persistence for e-process entries."""

    path: Path
    config_fingerprint: str
    code_version: str
    entries: dict[str, EProcessEntry] = field(default_factory=dict)
    resets: int = 0

    @classmethod
    def load(cls, path: Path, config_fingerprint: str, code_version: str) -> Self:
        store = cls(
            path=path,
            config_fingerprint=config_fingerprint,
            code_version=code_version,
        )
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return store
        if (
            raw.get("config_fingerprint") != config_fingerprint
            or raw.get("code_version") != code_version
        ):
            # A changed implementation or configuration changes the null;
            # wealth earned under the old one does not transfer.
            store.resets = int(raw.get("resets", 0)) + 1
            return store
        store.resets = int(raw.get("resets", 0))
        for key, entry in raw.get("entries", {}).items():
            store.entries[str(key)] = EProcessEntry(
                log_e=float(entry["log_e"]),
                t=int(entry["t"]),
                lam=float(entry["lam"]),
                ons_a=float(entry["ons_a"]),
                scale=float(entry["scale"]),
                cursor_valid_time=entry.get("cursor_valid_time"),
            )
        return store

    def save(self) -> None:
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "config_fingerprint": self.config_fingerprint,
            "code_version": self.code_version,
            "resets": self.resets,
            "entries": {
                key: asdict(entry) for key, entry in sorted(self.entries.items())
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with locked_path(self.path):
            atomic_write_text(json.dumps(payload, indent=2), self.path)

    def update_pair(
        self,
        key: str,
        candidate_losses: FloatArray,
        reference_losses: FloatArray,
        valid_times: list[datetime],
    ) -> EProcessEntry:
        """Bet on every differential newer than the cursor, in time order."""
        entry = self.entries.setdefault(key, EProcessEntry())
        cursor = (
            datetime.fromisoformat(entry.cursor_valid_time)
            if entry.cursor_valid_time is not None
            else None
        )
        order = np.argsort(np.asarray([t.timestamp() for t in valid_times]))
        for index in order:
            valid_time = valid_times[int(index)]
            if cursor is not None and valid_time <= cursor:
                continue
            differential = float(
                reference_losses[int(index)] - candidate_losses[int(index)]
            )
            if not math.isfinite(differential):
                continue
            if entry.scale > 0.0:
                z = max(-1.0, min(1.0, differential / entry.scale))
            else:
                # First bet: unit scale from the sign alone.
                z = float(np.sign(differential))
            entry.log_e += math.log1p(entry.lam * z)
            gradient = z / (1.0 + entry.lam * z)
            entry.ons_a += gradient * gradient
            entry.lam = max(
                0.0,
                min(_LAMBDA_MAX, entry.lam + _ONS_RATE * gradient / entry.ons_a),
            )
            entry.scale = max(entry.scale, abs(differential))
            entry.t += 1
            cursor = valid_time
        entry.cursor_valid_time = cursor.isoformat() if cursor is not None else None
        return entry

    def log_e(self, key: str) -> float:
        entry = self.entries.get(key)
        return entry.log_e if entry is not None else 0.0


def promotion_comparison(
    board: pl.DataFrame,
    scores: pl.DataFrame,
    *,
    alpha: float,
    promotion: "PromotionConfig | None",
    store: EProcessStore,
) -> pl.DataFrame:
    """Side-by-side decisions of the bootstrap-MCS and seq-MCS gates.

    Both rules are computed every report regardless of which one serves
    (``[promotion].rule``), so the operator can watch them disagree before
    switching — the compare-and-contrast requirement of the Aug-2026 round.
    """
    from grounded_weather_forecast.reports.leaderboard import (  # noqa: PLC0415
        slice_winners,
    )

    mcs = slice_winners(
        board, scores=scores, rule="mcs", alpha=alpha, promotion=promotion
    )
    seq = slice_winners(
        board,
        scores=scores,
        rule="seq_mcs",
        alpha=alpha,
        promotion=promotion,
        eprocess_store=store,
    )
    keys = [
        key
        for key in ("product", "variable", "truth_semantics", "lead_bucket")
        if key in mcs.columns
    ]
    if not keys or (mcs.is_empty() and seq.is_empty()):
        return pl.DataFrame(
            schema={
                **dict.fromkeys(keys or ["product"], pl.String),
                "mcs_choice": pl.String,
                "mcs_gate": pl.String,
                "seq_choice": pl.String,
                "seq_gate": pl.String,
                "agree": pl.Boolean,
            }
        )
    joined = mcs.select(
        *keys,
        pl.col("method_id").alias("mcs_choice"),
        pl.col("gate").alias("mcs_gate"),
    ).join(
        seq.select(
            *keys,
            pl.col("method_id").alias("seq_choice"),
            pl.col("gate").alias("seq_gate"),
        ),
        on=keys,
        how="full",
        coalesce=True,
    )
    return joined.with_columns(
        (pl.col("mcs_choice") == pl.col("seq_choice")).alias("agree")
    ).sort(keys)
