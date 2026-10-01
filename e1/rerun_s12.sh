#!/bin/bash
# e1_prereg.md section 12: re-run the four pair-engine +B3 methods (NRS delta = 0.10, lean grid) on the NRS datasets.
cd "$(dirname "$0")"
S="mammography htru2 magic-telescope house-16h gas-drift internet-firewall jannis run-or-walk vehicle-sensit miniboone creditcard hls4ml-jets"
for m in HASH+B3 CABIN+B3 BUCKET+B3 BF-PA+B3; do
  PYTHONWARNINGS=ignore python3 run_e1.py rerun "$m" $S
done
echo "=== section 12 re-run done $(date +%H:%M:%S)"
