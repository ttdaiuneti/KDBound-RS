"""Pre-registered degeneracy rule applied to all E1 datasets BEFORE locking the grid (same rule as the eeg/numerai
v2 check): a dataset is kept iff the exact NRS greedy gives |R| >= 2 at delta = 0.1 on the (capped) full table."""
import time
from data_e1 import DATASETS, load
from engines_e1 import run_nrs
for s in DATASETS:
    X, y = load(s); t = time.perf_counter(); R, ev, st = run_nrs("KD-PA+B3", X, y, 0.1)
    print(f"{s:26s} n={len(y):6d} d={X.shape[1]:3d} |R|={len(R):3d} {'KEEP' if len(R) >= 2 else 'DROP'} t={time.perf_counter()-t:.1f}s", flush=True)
