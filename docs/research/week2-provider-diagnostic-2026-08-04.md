# Week-2 provider diagnostic (2026-08-04)

Per-source error profile vs station truth in 24 h lead bins from
`data/hourly_matrix_live.parquet` (live archive through 2026-08-04), run to
decide `[forecasts].max_lead_hours` trim values before building on the
168-240h bucket. Script inline below; MAE/bias in variable units.

## Horizon census

Only three sources reach the 168-240h bucket at all: visual_crossing
(max lead 359 h), stormglass_sg (239 h), met_norway (231 h, ragged — n per
week-2 bin is only ~100-170). nws/open_meteo/weatherapi cliff at 154-167 h,
inside 96-168h; tomorrow_io at ~119 h.

## temp_c — MAE (bias) by bin

| source | 96h | 120h | 144h | 168h | 192h | 216h | 240h | 288h | 336h |
|---|---|---|---|---|---|---|---|---|---|
| met_norway | 1.09 (−0.81) | 0.99 (−0.68) | 0.76 (−0.57) | 1.47 (−0.06) | 1.64 (+0.64) | 2.04 (+0.23) | | | |
| visual_crossing | 2.09 (+0.67) | 2.16 (+1.28) | 2.28 (+1.74) | 4.07 (+3.82) | 6.78 (+6.78) | 6.50 (+6.50) | 6.49 (+6.49) | 4.45 (+4.36) | 3.63 (+3.26) |
| stormglass_sg | 3.39 (+1.45) | 5.70 (+5.68) | 6.57 (+6.57) | 7.19 (+7.19) | 6.79 (+6.79) | 6.68 (+6.68) | | | |

## humidity_pct — MAE (bias) by bin

| source | 96h | 144h | 168h | 192h | 216h | 240h | 288h |
|---|---|---|---|---|---|---|---|
| met_norway | 6.65 (−4.7) | 5.84 (−4.6) | 11.94 (−9.8) | 13.88 (−11.8) | 13.17 (−11.4) | | |
| visual_crossing | 5.99 (−2.6) | 5.50 (−1.1) | 10.82 (−9.2) | 18.06 (−17.7) | 17.17 (−16.9) | 15.23 (−14.0) | 9.05 (−2.3) |
| stormglass_sg | 9.07 (−8.2) | 9.00 (−7.7) | 14.72 (−14.7) | 18.71 (−18.7) | 15.92 (−15.9) | | |

## Findings

1. **The horizon-cliff hypothesis is wrong.** Week-2 error is not "last
   hours before each provider's horizon are noisy." It is **systematic
   bias**: for visual_crossing and stormglass, week-2 MAE ≈ |bias| in both
   variables (+6.5 °C warm, −17 % dry at 192-240 h). The residual after
   removing the offset is small — the anomaly signal out there is actually
   decent; the *level* is broken. This is exactly the −17.9 % humidity bias
   the served (ungrounded) `equal_weight` blend inherits at 168-240h.
2. **stormglass_sg is the one true trim case.** Its warm bias switches on
   at 120 h (+5.7) and stays 6-7 °C through its horizon. The provider is
   also payment-lapsed in the collector (402 since 2026-08-03), so its
   archive tail only pollutes ungrounded blends going forward.
   → `[forecasts.max_lead_hours] stormglass_sg = 120`.
3. **visual_crossing should NOT be trimmed at 168 h** despite the bias
   hump (168-288 h, recovering by 312 h): capping it would leave the
   168-240h bucket with only met_norway's ~150 rows per bin — coverage
   would collapse below eligibility and the 240h+ product would vanish.
   The bias is grounding-correctable signal, and the bias-vulnerable
   serving choice (raw equal_weight) is now challenged by
   `damped_grounded_equal_weight` (bias-free climatology component) and
   `analog_ensemble`. Revisit after one promotion cycle.
4. **met_norway is the only well-calibrated week-2 source** (MAE 1.5-2.0,
   |bias| < 0.7 on temp) but too thin to carry the bucket alone.
5. Why grounding hasn't already fixed this: the bias is an **episode**
   (visual_crossing's hump spans 168-288 h and shrinks again by 312 h —
   consistent with a summer regime the provider's long-lead product
   handles badly), while expanding-window grounding is dominated by the
   longer low-bias history. Drift-tracking methods (ewma/harmonic
   grounding) and the damped blend are the right responses, and the
   promotion-transparency report now shows what the gate is holding back.

## Actions taken

- `config.toml`: `stormglass_sg = 120` lead cap; `exclude =
  ["weatherapi:dew_point_c", "weatherapi:pop"]` (dew point anti-correlates
  with truth, −0.42; pop throws Page-Hinkley residual excursions 166-259
  in every bucket).
- No visual_crossing cap (finding 3).

## Script

```python
frame = pl.read_parquet("data/hourly_matrix_live.parquet")
frame = frame.filter(pl.col("lead_hours") >= 96.0)
# per (variable, source): bin = (lead_hours // 24) * 24;
# err = fx__{source}__{variable} - t__{variable}__inst
# aggregate MAE = mean |err|, bias = mean err, n per bin
```
