# Inactive temperature shadow candidate

Implemented in: `scripts/shadow_candidate_v1.py` and
`scripts/temperature_shadow.py`. These standalone scripts register no new
production method. They preserve the preselected October 26 experiment's
`simplex_decay21` candidate and `decay21` reference. The definitions below are
self-contained local experimental choices, not a reproduction of an operational
NOAA implementation or a production skill claim.

For each source, estimate forecast-minus-truth bias using weights
`2 ** (-age_days / 21)`. Train only on labels whose valid hour plus two hours
is at or before issuance, with archive issue time at or before issuance.
Retain the latest issue per valid hour and lead group. Groups are [0,6),
[6,24), [24,72), [72,168), [168,360) hours. Divide each global residual's
weight by that valid hour's lead-group multiplicity. Shrink each group bias
toward the global bias by `n / (n + 7)`, with `n` the count of UTC dates with
finite source values and positive weight. This is a count, not a decayed
effective sample size. Subtract the correction before combining providers.

Fit one nonnegative sum-to-one vector on corrected training rows. Minimize
weighted squared error divided by training target variance (standard-deviation
floor 0.1), plus `2 * sum((w - equal) ** 2)`. SLSQP uses bounds [0.0001,1],
100 iterations and tolerance 1e-8. Use equal weights with fewer than 20 usable
rows or a failed solve. Final weights are 50% fitted and 50% equal. Renormalize
over finite providers at both fit and prediction. The reference uses the same
corrections with equal weights throughout. Neither method silently alters truth.

Evaluation covers only the served 0–96 hours. It averages repeated issuance
losses per valid hour, then hours per UTC date. Circular calendar blocks of
3 and 7 days preserve missing dates. The proposal uses 10,000 resamples, seed
20261031, and two-sided Bonferroni limits for twelve predeclared comparisons
(four horizons times production, decayed equal weight and best provider).
Quality requires 28 matched dates, 300 matched valid hours, four weeks with
data, 80% common-row coverage and 80% of expected hourly issuances. Candidate
review requires at least 0.1°C and 5% daily-MAE improvement over production,
adjusted interval upper bound below zero against production and at most
0.1°C against the two main references under both block lengths. All six
methods must be finite on common cases. Raw pairwise availability and
attrition must also be inspected. Decisions occur once after six weeks plus
the last 96-hour outcomes and two-hour truth lag. No automatic promotion.
