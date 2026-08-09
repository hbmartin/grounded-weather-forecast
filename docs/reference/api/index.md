# API reference

Generated from source docstrings for the **public surface** — the modules
intended for import from outside the package. Internals (`dataset`, `backtest`,
`reports`, `dashboard`, individual blender implementations) are deliberately not
published here: they change without notice, and depending on them is depending on
an implementation detail.

<div class="grid cards" markdown>

- **[Contracts](contracts.md)** — the cross-layer types every other layer speaks:
  `ForecastMatrix`, `SupervisedSlice`, `BlendResult`, the `Blender` protocol, and
  the column-naming helpers.
- **[Configuration](config.md)** — the frozen dataclasses `load_config` produces.
- **[Blender protocol and registry](blenders.md)** — what you implement and how you
  register it.
- **[Metrics](metrics.md)** — deterministic scores, probabilistic scores, and the
  Diebold–Mariano test.
- **[Lead buckets](leads.md)** — the stratification grid.
- **[Forecast schema](schema.md)** — the emitted document.

</div>

---

## Stability

`contracts` and `leads` are the only modules other layers may deep-import — a
rule the package enforces by convention and reviews. Blenders import `contracts`
only, never `dataset`.

Two invariants worth knowing before you build on any of this:

- **`ForecastMatrix.__post_init__` raises** if any feature column starts with
  `t__`. The leakage guard is structural: the illegal object cannot be
  constructed.
- **The registry stores factories, never instances.** The backtest engine builds
  a fresh blender per fold so a stateful method cannot carry weights across the
  train/test boundary. `register()` takes a callable returning a `Blender`.

## A minimal blender

```python
from dataclasses import dataclass
from typing import Self

import numpy as np

from grounded_weather_forecast.blenders.protocol import finalize_point
from grounded_weather_forecast.blenders.registry import register
from grounded_weather_forecast.contracts import (
    BlendResult, Blender, ForecastMatrix, SupervisedSlice, VariableSpec,
)


@dataclass
class MyBlend:
    def fit(self, train: SupervisedSlice, spec: VariableSpec) -> Self:
        return self

    def predict(self, x: ForecastMatrix) -> BlendResult:
        point = np.nanmedian(x.values, axis=1)
        return BlendResult(point=finalize_point(point, self._spec))


register("my_blend", MyBlend)   # a factory, not MyBlend()
```

Full walkthrough, including how to get it onto the leaderboard, in
[Advanced usage](../../advanced-usage.md#adding-a-blending-method). The
mathematical conventions your method is expected to honour — availability
renormalization, monotone quantiles, honest abstention — are in
[Methods: notation](../../methods/notation.md).
