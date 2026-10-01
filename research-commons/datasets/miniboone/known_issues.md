# Known issues — MiniBooNE particle identification (OpenML 41150)

- Extreme outliers: in 41 feature(s) the central 99% of values occupies < 5% of the min-max range after v1 scaling (e.g. ParticleID_0, ParticleID_1, ParticleID_3, ParticleID_4, ParticleID_5); distances on these features are dominated by a few rows.
- Sentinel value -999 appears in 468 row(s) (likely missing-value codes, kept in v1).
- 467 duplicated feature rows.
