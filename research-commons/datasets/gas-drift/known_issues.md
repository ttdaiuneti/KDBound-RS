# Known issues — Gas sensor array drift (OpenML 1476)

- Collected in 10 batches over 36 months with sensor drift; random CV mixes batches (optimistic).
- Extreme outliers: in 16 feature(s) the central 99% of values occupies < 5% of the min-max range after v1 scaling (e.g. V2, V10, V21, V29, V32); distances on these features are dominated by a few rows.
