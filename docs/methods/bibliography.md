# Bibliography

The canonical reference list. [Theory §7](../theory.md#7-selected-reading) holds a
short orienting subset; everything is here, matched to the module that implements
it.

Three sections, kept deliberately separate:

1. **[Implemented](#1-implemented)** — papers whose method is in the codebase.
2. **[Declined](#2-declined)** — papers considered and *not* implemented, with the
   recorded reason. A reading list is not a claim of coverage, and knowing what a
   system chose not to do is often more informative than knowing what it did.
3. **[Operational analogues](#3-operational-analogues)** — institutional systems
   that informed the design but are not papers.

---

## 1. Implemented

### Forecast combination

| Reference | Implemented in |
|---|---|
| Bates, J. M. & Granger, C. W. J. (1969). The combination of forecasts. *Operational Research Quarterly* 20(4), 451–468. | `blenders/combine.py::InverseErrorWeights` |
| Stock, J. H. & Watson, M. W. (2004). Combination forecasts of output growth in a seven-country data set. *Journal of Forecasting* 23(6), 405–430. | Framing — the combination puzzle; `equal_weight` as benchmark |
| Timmermann, A. (2006). Forecast combinations. *Handbook of Economic Forecasting* 1, 135–196. | Motivation for `inverse_*`; `invcov` is registered as its counterfactual |
| Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis* 88(2), 365–411. | `blenders/invcov.py::ledoit_wolf_covariance` |

### Statistical post-processing

| Reference | Implemented in |
|---|---|
| Vannitsem, S. et al. (2021). Statistical postprocessing for weather forecasts: review, challenges, and avenues. *BAMS* 102(3), E681–E699. | Framing for the grounding stage |
| Gneiting, T., Raftery, A. E., Westveld, A. H. & Goldman, T. (2005). Calibrated probabilistic forecasting using ensemble model output statistics and minimum CRPS estimation. *MWR* 133(5), 1098–1118. | `blenders/emos.py::Emos` |
| Scheuerer, M. & Hamill, T. M. (2015). Statistical postprocessing of ensemble precipitation forecasts by fitting censored, shifted gamma distributions. *MWR* 143(11), 4578–4596. | `blenders/csgd.py::CsgdEmos` |
| Henzi, A., Ziegel, J. F. & Gneiting, T. (2021). Isotonic distributional regression. *JRSS-B* 83(5), 963–993. | `blenders/idr.py` (incl. the subagging scheme) |
| Delle Monache, L. et al. (2011). Kalman filter and analog schemes to postprocess numerical weather predictions. *MWR* 139(11); and (2013) Probabilistic weather prediction with an analog ensemble. *MWR* 141(10). | `blenders/analog.py::AnalogEnsemble` |
| Schuhen, N., Thorarinsdottir, T. L. & Lenkoski, A. (2020). Rapid adjustment and post-processing of temperature forecast trajectories. | `blenders/raft.py::RaftGrounded` |
| Dabernig, M. & Atencia, A. (2024). Seamless multimodel postprocessing. | `blenders/seamless.py::SeamlessRegression` |
| Monhart, S. et al. (2018). Skill of subseasonal forecasts in Europe: effect of bias correction and downscaling using surface observations. | `blenders/damped.py::DampedBlend` |
| Meng, X. & Taylor, J. W. (2022). Comparing probabilistic forecasts of the daily minimum and maximum temperature. | `blenders/daily_heads.py::DailyMarginalEmos` |
| Schefzik, R. Ensemble copula coupling. | `blenders/daily_heads.py::DailyPathExtreme` (collapsed to the scalar extreme) |
| Alduchov, O. A. & Eskridge, R. E. (1996). Improved Magnus form approximation of saturation vapor pressure. *J. Applied Meteorology* 35(4), 601–609. | `units.py` — dew point constants |

### Probability calibration

| Reference | Implemented in |
|---|---|
| Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. | `blenders/pop_calibration.py` — `pop_platt` |
| Kull, M., Silva Filho, T. & Flach, P. (2017). Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers. *AISTATS*. | `blenders/pop_calibration.py` — `pop_beta` |
| Gupta, C. & Ramdas, A. Calibeating / online calibration. | `pop_platt` framing |

### Online learning

| Reference | Implemented in |
|---|---|
| Vovk, V.; Littlestone, N. & Warmuth, M. K. (1994). The weighted majority algorithm. | `blenders/experts.py` — `ewa` |
| Wintenberger, O. (2017). Optimal learning with Bernstein online aggregation. *Machine Learning* 106, 119–141. | `blenders/experts.py` — `boa` |
| Herbster, M. & Warmuth, M. K. (1998). Tracking the best expert. *Machine Learning* 32, 151–178. | `blenders/experts.py::_step` — fixed share |
| Blum, A. Empirical support for Winnow and weighted-majority; the sleeping-experts reduction. | `blenders/experts.py` — awake-set handling |

### Uncertainty quantification

| Reference | Implemented in |
|---|---|
| Angelopoulos, A., Candès, E. & Tibshirani, R. (2023). Conformal PID control for time series prediction. *NeurIPS*. | `blenders/conformal.py::_CellState.update` / `effective_radius` |
| Chernozhukov, V., Wüthrich, K. & Zhu, Y. (2021). Distributional conformal prediction. *PNAS* 118(48). | `blenders/idr.py::dcp_adjusted_levels`, `IdrBucketDcp` |
| Kuleshov, V., Fenner, N. & Ermon, S. (2018). Accurate uncertainties for deep learning using calibrated regression. *ICML*. | `reports/recalibration.py::fit_pit_levels` |
| Romano, Y., Patterson, E. & Candès, E. (2019). Conformalized quantile regression. *NeurIPS*. | `reports/recalibration.py::fit_cqr_margins` — **a variant**, see [calibration §6](calibration.md#6-post-hoc-quantile-recalibration) |

### Verification, testing, and selection

| Reference | Implemented in |
|---|---|
| Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA* 102(477), 359–378. | `metrics/probabilistic.py` |
| Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. *JBES* 13(3), 253–263. | `metrics/dm.py::diebold_mariano` |
| Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *IJF* 13(2), 281–291. | `metrics/dm.py::_hln_factor` |
| Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. *IJF* 16(4), 437–450. | `backtest/splits.py` |
| Hansen, P. R., Lunde, A. & Nason, J. M. (2011). The model confidence set. *Econometrica* 79(2), 453–497. | `reports/mcs.py::model_confidence_set` |
| Künsch, H. R. (1989); Politis, D. N. & Romano, J. P. (1992). Moving/circular block bootstrap. | `reports/mcs.py::_block_indices` |
| Ville, J. (1939). *Étude critique de la notion de collectif.* | `reports/eprocess.py`; `leaderboard._seq_gate` |
| Waudby-Smith, I. & Ramdas, A. Estimating means of bounded random variables by betting. *JRSS-B*. | `reports/eprocess.py::EProcessStore.update_pair` |
| Cutkosky, A. & Orabona, F. — online Newton step lineage for the bet sizing. | `reports/eprocess.py` — `_ONS_RATE` |
| Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B* 57(1), 289–300. | `reports/leaderboard.py::_benjamini_hochberg` |
| Wang, R. & Ramdas, A. (2022). False discovery rate control with e-values. *JRSS-B* ([arXiv:2009.02824](https://arxiv.org/abs/2009.02824)). | `reports/leaderboard.py::ebh_adjusted` |
| Andrews, I., Kitagawa, T. & McCloskey, A. (2024). Inference on winners. *QJE* 139(1). | `reports/winner_curse.py` — the hybrid estimator |
| Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap.* | `reports/winner_curse.py::_slice_correction` |
| Fang, Z. & Santos, A. — first-order validity of the bootstrap at directional-differentiability failures. | `reports/winner_curse.py` — cited as the near-tie caveat |
| Sequential model confidence sets ([arXiv:2404.18678](https://arxiv.org/abs/2404.18678)). | `reports/eprocess.py` framing |
| Anytime-valid inference read at fixed calendar times ([arXiv:2502.08539](https://arxiv.org/abs/2502.08539)). | `reports/leaderboard.py::ebh_adjusted` |
| Météo-France expert aggregation ([arXiv:2506.15217](https://arxiv.org/pdf/2506.15217)). | Closest published analogue to the expert layer |

### Homogenization and station QC

| Reference | Implemented in |
|---|---|
| Alexandersson, H. (1986). A homogeneity test applied to precipitation data. *J. Climatology* 6(6), 661–675. | `dataset/drift_stats.py::snht` |
| Khaliq, M. N. & Ouarda, T. B. M. J. (2007). On the critical values of the standard normal homogeneity test. *Int. J. Climatology* 27(5), 681–687. | `drift_stats._SNHT_CRITICAL_95` — scale check at $n = 30$ |
| Pettitt, A. N. (1979). A non-parametric approach to the change-point problem. *Applied Statistics* 28(2), 126–135. | `dataset/drift_stats.py::pettitt` |
| Craddock, J. M. (1979). Methods of comparing annual rainfall records for climatic purposes. *Weather* 34, 332–346. | `dataset/drift_stats.py::craddock_cusum` |
| Menne, M. J. & Williams, C. N. (2009). Homogenization of temperature series via pairwise comparisons. *J. Climate* 22(7), 1700–1717. | `dataset/drift_stats.py::attribute_break` |
| Page, E. S. (1954); Hinkley, D. V. (1971). Cumulative-sum change detection. | `reports/drift.py::page_hinkley` |
| Nakamura, R. & Mahrt, L. (2005). Air temperature measurement errors in naturally ventilated radiation shields. *JAOT* 22(7), 1046–1058. | `dataset/truth_qc.py::solar_load`, `fit_shield_error` |
| NOAA GMD solar position algorithm. | `solar.py` |

---

## 2. Declined

Papers considered and deliberately not implemented. The reason matters as much as
the decision.

| Reference | Why not |
|---|---|
| Raftery, A. E., Gneiting, T., Balabdaoui, F. & Polakowski, M. (2005). Using Bayesian model averaging to calibrate forecast ensembles. *MWR* 133(5). | Adjacent and reasonable; no BMA blender exists. EMOS and IDR occupy the same niche with less machinery. Not ruled out — simply not built. |
| Athanasopoulos, G., Hyndman, R. J., Kourentzes, N. & Petropoulos, F. (2017). Forecasting with temporal hierarchies. *EJOR* 262(1); Wickramasuriya, S. L., Athanasopoulos, G. & Hyndman, R. J. (2019). Optimal forecast reconciliation (MinT). *JASA* 114(526). | Reconciliation constrains **linear** aggregates. The daily targets here are max, min, max gust, max PoP — nonlinear, with no summing matrix, outside MinT by construction. Nonlinearly-constrained reconciliation exists as a 2025 preprint, carries no error-reduction guarantee, and max/min kinks are its worst case. |
| Denton, F. T. (1971); Chow, G. C. & Lin, A. (1971). Temporal disaggregation. | A coarse forecast contains no fine-scale shape information; disaggregation can only impose an assumed shape. The live observation *does* carry minute-scale information, and that is what anchoring uses. [ADR 0002](../adr/0002-anchored-nowcast-instead-of-temporal-disaggregation.md). |
| Kalman, R. E. (1960), and state-space post-processing generally. | Anchoring extracts the useful "correct toward the observation" step without the covariance tuning, and — decisively — without losing the ability to A/B test the pieces separately. |
| Kochendorfer, J. et al. (2018). WMO-SPICE gauge catch efficiency transfer functions. *HESS* 22(2). | Coefficients require careful transcription and the archive has essentially no precipitation to validate against. Deferred, not rejected. |
| Prewhitening for autocorrelated change-point series. | Rejected in favour of a persistence guard: drifts persist, regimes revert. Prewhitening would also remove part of the slow trend a failing sensor produces — the signal. |

---

## 3. Operational analogues

Institutional systems cited in the source as design references. These are not
papers and are kept separate so the bibliography above stays a literature claim.

| System | Informs |
|---|---|
| **NOAA National Blend of Models (NBM)** | The whole architecture: bias-correct, weight by lead time, anchor to observations at short lead. `blenders/combine.py`, `cluster.py`, `ewma_grounding.py` |
| **NCEP decaying-average bias correction** (operational since 2006) | `blenders/ewma_grounding.py` |
| **LAMP** (Localized Aviation MOS Program) | `blenders/anchoring.py` |
| **INCA** persist-then-ramp nowcasting | `blenders/anchoring.py` |
| **SAMOS** (standardized anomaly MOS) | `blenders/harmonic_grounding.py` |
| **EasyUQ / EUPPBench** | `blenders/idr.py` |
| **CrowdQC+** (m4 module) | `dataset/neighbors.py` |
| **PRISM** elevation-layer logic | `dataset/neighbors.py` |
| **MADIS / GHCN / ECMWF station blacklisting** | `dataset/truth_qc.py` |
| **ForecastAdvisor / ForecastWatch** | The consumer analogue this system strictly dominates for one location — it scores per hour, to 10 days, with proper scoring rules, against your actual backyard. |
