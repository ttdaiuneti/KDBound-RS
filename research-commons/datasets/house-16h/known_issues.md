# Known issues — house_16H, US census, binarised target (OpenML 821)

- Regression target (median house price) binarised by OpenML; features are census ratios.
- Extreme outliers: in 1 feature(s) the central 99% of values occupies < 5% of the min-max range after v1 scaling (e.g. P1); distances on these features are dominated by a few rows.
- Sentinel value 999 appears in 3 row(s) (likely missing-value codes, kept in v1).
- 1 duplicated feature rows.
