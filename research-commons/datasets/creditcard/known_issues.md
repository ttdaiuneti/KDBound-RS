# Known issues — Credit card fraud (OpenML 1597)

- Extreme class imbalance: accuracy is uninformative; report balanced metrics if used for accuracy.
- Extreme outliers: in 2 feature(s) the central 99% of values occupies < 5% of the min-max range after v1 scaling (e.g. V7, V28); distances on these features are dominated by a few rows.
- (auto-flag checked 2026-09-29: the 7 values 999 are genuine `Amount` values, not missing-value codes.)
- OpenML 1597 has no `Time` column (29 features: V1–V28, Amount).
- Class imbalance: smallest class = 492 of 284807 (0.17%).
- 9144 duplicated feature rows.
