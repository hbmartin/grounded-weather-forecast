"""Provider error-correlation matrix.

Most consumer APIs repackage the same handful of global models, so their
errors are highly correlated — this report quantifies how many *effective*
sources exist, which caps what any weighting scheme can add over grounding.
"""

import numpy as np
import polars as pl

from grounded_weather_forecast.contracts import TruthSemantics, VariableSpec, fx_col
from grounded_weather_forecast.dataset.matrix import matrix_sources, truth_column_for

_MIN_OVERLAP = 24


def error_correlation(
    matrix: pl.DataFrame,
    variable: VariableSpec,
    semantics: TruthSemantics = TruthSemantics.INSTANTANEOUS,
) -> pl.DataFrame:
    """Pairwise Pearson correlation of per-source errors against truth."""
    truth_column = truth_column_for(variable, semantics)
    sources = [
        source
        for source in matrix_sources(matrix)
        if fx_col(source, variable.name) in matrix.columns
    ]
    if truth_column not in matrix.columns or not sources:
        return pl.DataFrame()
    usable = matrix.filter(pl.col(truth_column).is_not_null())
    errors = {
        source: usable[fx_col(source, variable.name)].to_numpy()
        - usable[truth_column].to_numpy()
        for source in sources
    }
    rows: list[dict[str, object]] = []
    for source_a in sources:
        row: dict[str, object] = {"source": source_a}
        for source_b in sources:
            a, b = errors[source_a], errors[source_b]
            overlap = ~(np.isnan(a) | np.isnan(b))
            if overlap.sum() < _MIN_OVERLAP:
                row[source_b] = None
            else:
                with np.errstate(invalid="ignore"):
                    row[source_b] = float(np.corrcoef(a[overlap], b[overlap])[0, 1])
        rows.append(row)
    return pl.DataFrame(rows)
