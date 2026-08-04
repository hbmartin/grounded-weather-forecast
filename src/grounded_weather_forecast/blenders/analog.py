"""Analog ensemble: nonparametric distributions from the forecast archive.

AnEn (Delle Monache et al. 2011, 2013): for each new forecast, find the k
training rows whose provider forecast vectors look most alike, and serve the
distribution of the observations that actually verified under them. The
point is the analog median; the quantiles are the analog empirical
quantiles. Identity-free by construction — similarity is measured on
forecast values, never provider names — and valid for every variable
including precipitation, because empirical quantiles of realized
observations can never leave the physical support.

Distances use Delle Monache's normalized Euclidean metric over the sources
finite in both rows (per-source sigma from the training archive), with
candidates drawn from the target's lead bucket when it holds enough rows
and from the full archive otherwise.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np

from grounded_weather_forecast.blenders.protocol import (
    finalize_point,
    finalize_quantiles,
)
from grounded_weather_forecast.blenders.registry import register
from grounded_weather_forecast.contracts import (
    Blender,
    BlendResult,
    FloatArray,
    ForecastMatrix,
    SupervisedSlice,
    TargetKind,
    VariableSpec,
)
from grounded_weather_forecast.leads import LeadBucket, buckets_for_product

QUANTILE_LEVELS: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))
_MIN_FIT_ROWS = 100
_MIN_SIGMA = 1e-6
_CHUNK_ROWS = 128


def _column_sigma(values: FloatArray) -> FloatArray:
    """Per-source standard deviation over finite entries; NaN below 2 rows."""
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    filled = np.where(finite, values, 0.0)
    means = filled.sum(axis=0) / np.maximum(counts, 1)
    squared = np.where(finite, (values - means) ** 2, 0.0).sum(axis=0)
    sigma = np.sqrt(squared / np.maximum(counts, 1))
    return np.where(counts >= 2, np.maximum(sigma, _MIN_SIGMA), np.nan)


@dataclass
class AnalogEnsemble:
    method_id: str = "analog_ensemble"
    n_analogs: int = 25
    min_pool_factor: int = 2
    _kind: TargetKind = TargetKind.CONTINUOUS
    _variable: VariableSpec | None = None
    _train_values: FloatArray | None = None

    def fit(self, train: SupervisedSlice) -> Self:
        self._kind = train.variable.kind
        self._variable = train.variable
        self._buckets = buckets_for_product(train.x.product)
        if train.x.n_rows < _MIN_FIT_ROWS:
            self._train_values = None
            return self
        self._train_values = train.x.values
        self._train_y = train.y
        self._train_lead = train.x.lead_hours
        self._sigma = _column_sigma(train.x.values)
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        if (train_values := self._train_values) is None:
            point = np.full(x.n_rows, np.nan)
            return BlendResult(point=finalize_point(point, self._kind, self._variable))
        point = np.full(x.n_rows, np.nan)
        quantiles = np.full((x.n_rows, len(QUANTILE_LEVELS)), np.nan)
        all_rows = np.arange(x.n_rows)
        assigned = np.zeros(x.n_rows, dtype=bool)
        for bucket in self._buckets:
            mask = (x.lead_hours >= bucket.lo) & (x.lead_hours < bucket.hi)
            if mask.any():
                self._predict_rows(
                    x,
                    train_values,
                    all_rows[mask],
                    self._pool_for(bucket),
                    point,
                    quantiles,
                )
                assigned |= mask
        if (~assigned).any():
            full_pool = np.arange(train_values.shape[0])
            self._predict_rows(
                x, train_values, all_rows[~assigned], full_pool, point, quantiles
            )
        quantiles = finalize_quantiles(quantiles, self._kind, self._variable)
        quantiles[~np.isfinite(point)] = np.nan
        return BlendResult(
            point=finalize_point(point, self._kind, self._variable),
            quantiles=quantiles,
            quantile_levels=QUANTILE_LEVELS,
        )

    def _pool_for(self, bucket: LeadBucket) -> np.ndarray:
        lead = self._train_lead
        rows = np.nonzero((lead >= bucket.lo) & (lead < bucket.hi))[0]
        if rows.shape[0] >= self.n_analogs * self.min_pool_factor:
            return rows
        return np.arange(lead.shape[0])

    def _predict_rows(
        self,
        x: ForecastMatrix,
        train_values: FloatArray,
        rows: np.ndarray,
        pool: np.ndarray,
        point: FloatArray,
        quantiles: FloatArray,
    ) -> None:
        train_z = train_values[pool] / self._sigma
        y_pool = self._train_y[pool]
        levels = np.asarray(QUANTILE_LEVELS)
        n_sources = train_z.shape[1]
        for start in range(0, rows.shape[0], _CHUNK_ROWS):
            chunk = rows[start : start + _CHUNK_ROWS]
            target_z = x.values[chunk] / self._sigma
            squared = np.zeros((chunk.shape[0], pool.shape[0]))
            counts = np.zeros((chunk.shape[0], pool.shape[0]), dtype=np.int64)
            for source in range(n_sources):
                delta = target_z[:, source, np.newaxis] - train_z[np.newaxis, :, source]
                shared = np.isfinite(delta)
                squared += np.where(shared, delta * delta, 0.0)
                counts += shared
            with np.errstate(invalid="ignore", divide="ignore"):
                distance = np.sqrt(squared / counts)
            distance = np.where(counts > 0, distance, np.inf)
            k = min(self.n_analogs, pool.shape[0])
            nearest = np.argpartition(distance, k - 1, axis=1)[:, :k]
            for offset, (row, analogs) in enumerate(zip(chunk, nearest, strict=True)):
                usable = analogs[np.isfinite(distance[offset, analogs])]
                if usable.shape[0] == 0:
                    continue
                outcomes = y_pool[usable]
                point[row] = float(np.median(outcomes))
                quantiles[row] = np.quantile(outcomes, levels)

    def to_state(self) -> dict[str, object]:
        """Compact observability state: pool geometry, never the archive."""
        if self._train_values is None:
            return {"method_id": self.method_id, "fit_status": "insufficient_rows"}
        return {
            "method_id": self.method_id,
            "fit_status": "fit",
            "n_analogs": self.n_analogs,
            "training_rows": int(self._train_values.shape[0]),
            "bucket_pool_rows": {
                bucket.label: int(
                    (
                        (self._train_lead >= bucket.lo) & (self._train_lead < bucket.hi)
                    ).sum()
                )
                for bucket in self._buckets
            },
        }


register("analog_ensemble", AnalogEnsemble)

_PROTOCOL_CHECK: tuple[type[Blender], ...] = (AnalogEnsemble,)
