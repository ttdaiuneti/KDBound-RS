"""R1.20: nearest-enemy ties on internet-firewall (FRS, sigma = 0.25, fold 1 as in x86_check/run_check.py), after the
first six attributes of the reduct. Explains why KD-PA+B3 evaluated 25 candidates on the x86 machine and 26 on the M4:
distances and gains are exact on both machines, so only the choice among tied nearest enemies (the witness) can differ.

python witness_ties.py   (locked engines and data of e1/; seed 1 sample of 1000 objects)
"""
import os, sys
import numpy as np
from sklearn.model_selection import StratifiedKFold

E1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "e1")
sys.path.insert(0, E1)
import frs  # noqa: E402
from data_e1 import load  # noqa: E402

X, y = load("internet-firewall")
tr = next(iter(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(y)), y)))[0]
X, y = X[tr], y[tr]
Z = frs.scale(X, 0.25)
Rt = [3, 2, 0, 1, 8, 6]            # first six attributes of the reduct on both machines
rest = (4, 5, 7, 9, 10)            # attributes not yet selected that the bound is computed for
S = np.random.default_rng(1).choice(len(y), 1000, replace=False)
near = multi = diff = 0
for q in S:
    E = np.where(y != y[q])[0]
    dist = np.abs(Z[E][:, Rt] - Z[q, Rt]).max(1)
    m = dist.min()
    if m >= 1:
        continue
    near += 1
    W = E[dist == m]
    if len(W) > 1:
        multi += 1
        diff += any(len(np.unique(np.abs(Z[W, a] - Z[q, a]))) > 1 for a in rest)
print(f"sample 1000: ne<1 for {near}; several nearest enemies for {multi}; "
      f"tied witnesses differ on a remaining attribute for {diff}")
