# Known issues — Internet firewall log actions (OpenML 46936)

- Features are integer counts / port numbers (>=100 distinct each, so pass the rule), not physically continuous; port numbers are nominal in meaning.
- Extreme outliers: in 6 feature(s) the central 99% of values occupies < 5% of the min-max range after v1 scaling (e.g. Bytes, Bytes_Sent, Bytes_Received, Packets, pkts_sent); distances on these features are dominated by a few rows.
- Values 999 / 9999 were auto-flagged as possible sentinels. Checked on 2026-09-29 which columns hold them: 999 appears in Bytes_Sent (2), Bytes_Received (3) and pkts_sent (1); 9999 appears in Destination_Port (2), NAT_Destination_Port (2) and Bytes (2). These are isolated values in count and port columns, not a column-wide fill, so they are more consistent with genuine values than with missing-value codes. The dataset documentation was NOT checked, so this is not confirmed. Kept in v1.
- Class imbalance: smallest class = 54 of 65532 (0.08%).
- 8371 duplicated feature rows.
