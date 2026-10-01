"""Per-criterion degeneracy rule (user 2026-09-29): FRS kept iff |R| >= 2 at sigma = 0.25 on the capped full table.
Appends to degeneracy_screen_frs.log; datasets already in the log are skipped."""
import os, sys, time
from data_e1 import DATASETS, load
from frs import run_frs
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "degeneracy_screen_frs.log")
done = {ln.split()[0] for ln in open(LOG)} if os.path.exists(LOG) else set()
for s in DATASETS:
    if s in done:
        continue
    X, y = load(s); t = time.perf_counter(); R, ev, st = run_frs("KD-PA+B3", X, y, 0.25)
    line = f"{s:26s} n={len(y):6d} d={X.shape[1]:3d} |R|={len(R):3d} {'KEEP' if len(R) >= 2 else 'DROP'} t={time.perf_counter()-t:.1f}s"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)
