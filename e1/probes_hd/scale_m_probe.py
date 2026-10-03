"""Design probe for the attribute-scaling experiment on dilbert (numbers are NOT reported; they only choose the grid).
Nested random attribute subsets (permutation seed 0), fold-1 training part, NRS delta 0.10 / FRS sigma 0.25."""
import json, signal, sys, time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, '/Users/tranthanhdai/Documents/2026/NCS-Tram/KDBound-RS/e1')
from engines_e1 import run_nrs
from frs import run_frs
CAP = int(sys.argv[1]); GRID = [int(v) for v in sys.argv[2].split(',')]
d = fetch_openml(data_id=41163, as_frame=True, parser="auto", data_home='cache')
X = d.data.to_numpy(float); y = np.unique(d.target.astype(str), return_inverse=True)[1]
r = X.max(0) - X.min(0); X = (X - X.min(0)) / np.where(r > 0, r, 1)
perm = np.random.default_rng(0).permutation(X.shape[1])
tr = next(iter(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y)))[0]
dead = set()
def alarm(*_): raise TimeoutError
signal.signal(signal.SIGALRM, alarm)
for m in GRID:
    Xm = X[tr][:, perm[:m]]; ym = y[tr]
    for crit in ('NRS', 'FRS'):
        for meth in ('KD-PA+B3', 'KD-PA'):
            rec = dict(m=m, n_train=len(ym), crit=crit, method=meth)
            if (crit, meth) in dead:
                rec['status'] = 'skipped-after-timeout'
            else:
                signal.alarm(CAP); t0 = time.perf_counter()
                try:
                    R, ev, steps = run_nrs(meth, Xm, ym, 0.10) if crit == 'NRS' else run_frs(meth, Xm, ym, 0.25)
                    rec.update(status='ok', R=len(R), evals=int(ev), first_gain=steps[0].get('gain') if steps else None)
                except TimeoutError:
                    rec['status'] = 'timeout'; dead.add((crit, meth))
                signal.alarm(0); rec['time'] = round(time.perf_counter() - t0, 1)
            print(json.dumps(rec, default=float), flush=True)
