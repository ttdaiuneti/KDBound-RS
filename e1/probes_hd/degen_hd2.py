"""Pre-registered degeneracy rule (E1 sections 1 and 7) on a candidate: greedy on the whole table, keep a criterion iff
|R| >= 2 (NRS delta = 0.10, FRS sigma = 0.25), min-max normalized like commons v1. Screening only."""
import sys, time, signal
import numpy as np
from sklearn.datasets import fetch_openml
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from engines_e1 import run_nrs
from frs import run_frs
did = int(sys.argv[1]); crit = sys.argv[2]
d = fetch_openml(data_id=did, as_frame=True, parser="auto", data_home='cache')
X = d.data.to_numpy(float); y = np.unique(d.target.astype(str), return_inverse=True)[1]
if len(y) > 100_000:                                   # E1 cap: one stratified sample of 1e5 (seed 0)
    from sklearn.model_selection import train_test_split
    i = np.sort(train_test_split(np.arange(len(y)), train_size=100_000, stratify=y, random_state=0)[0]); X, y = X[i], y[i]
rng = X.max(0) - X.min(0); X = (X - X.min(0)) / np.where(rng > 0, rng, 1)
def _alarm(*_):
    raise TimeoutError
signal.signal(signal.SIGALRM, _alarm); signal.alarm(900)   # protocol limit per unit
t = time.perf_counter()
try:
    R, ev, steps = run_nrs("KD-PA+B3", X, y, 0.10) if crit == "NRS" else run_frs("KD-PA+B3", X, y, 0.25)
    print(f"{d.details['name']} {crit} n={len(y)} m={X.shape[1]} |R|={len(R)} evals={ev} first-step gain={steps[0].get('gain')} "
      f"{'KEEP' if len(R) >= 2 else 'DROP'} t={time.perf_counter()-t:.1f}s", flush=True)

except TimeoutError:
    print(f"{d.details['name']} {crit} n={len(y)} m={X.shape[1]} TIMEOUT >900 s (KD-PA+B3, whole table)", flush=True)
