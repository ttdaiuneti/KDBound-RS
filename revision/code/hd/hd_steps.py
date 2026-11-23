"""Revision run for R1.18: per-step records of KD-PA+B3 (NRS, delta = 0.10) on the two probed high-dimensional tables
of Appendix A.1, so that they can be placed with the 14 datasets on the same step-level quantities (share of candidates
skipped, share of bounds equal to zero, share of attributes with a first-step gain of at least half the best).

Fixed before the run (2026-11-23), as in the probes (e1/probes_hd): dilbert (OpenML 41163), nested random attribute
subsets of m in {50, 100, 200, 400} (permutation seed 0), training part of fold 1 (StratifiedKFold(5, shuffle, 0));
SantanderCustomerSatisfaction (OpenML 42395, the 200 numeric attributes), one stratified sample of 1e5 objects
(seed 0), fold-1 training part. Min-max scaling on the table, as in commons v1. 900 s per unit; one unit at a time.
Driver: code/steps/steps_run.py nrs() (KD-PA+B3 with step records).

python hd_steps.py -> hd_steps.jsonl
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "steps"))
OUT = os.path.join(HERE, "hd_steps.jsonl"); CACHE = os.path.join(HERE, "oml_cache")


def table(did):
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=did, as_frame=True, parser="auto", data_home=CACHE)
    Xf = d.data.select_dtypes("number")
    X = Xf.to_numpy(float); y = np.unique(d.target.astype(str), return_inverse=True)[1]
    if len(y) > 100_000:
        from sklearn.model_selection import train_test_split
        i = np.sort(train_test_split(np.arange(len(y)), train_size=100_000, stratify=y, random_state=0)[0])
        X, y = X[i], y[i]
    r = X.max(0) - X.min(0); X = (X - X.min(0)) / np.where(r > 0, r, 1)
    from sklearn.model_selection import StratifiedKFold
    tr = next(iter(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y)))[0]
    return X[tr], y[tr]


def main():
    import steps_run as S
    sys.path.insert(0, S.E1); sys.path.insert(0, S.NUM)
    S.CAP = 900
    have = {(r["dataset"], r["m"]) for r in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()
    units = [("dilbert", 41163, m) for m in (50, 100, 200, 400)] + [("santander", 42395, 200)]
    cache = {}
    for name, did, m in units:
        if (name, m) in have:
            continue
        if did not in cache:
            cache[did] = table(did)
        X, y = cache[did]
        if name == "dilbert":
            perm = np.random.default_rng(0).permutation(X.shape[1]); Xm = X[:, perm[:m]]
        else:
            assert X.shape[1] == 200, X.shape; Xm = X
        T0 = time.perf_counter()
        R, steps, partial = S.nrs(np.ascontiguousarray(Xm), y, 0.10, T0)
        rec = dict(dataset=name, m=m, n_train=int(len(y)), criterion="NRS", param="0.10", R=[int(a) for a in R],
                   partial=partial, time=time.perf_counter() - T0, steps=steps)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        print(name, m, len(R), partial, round(rec["time"], 1), flush=True)


if __name__ == "__main__":
    main()
