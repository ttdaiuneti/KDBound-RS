"""Cost of one significance evaluation with three exact engines as the subset grows (revision, R1.7).

For gas-drift (m = 128) and miniboone (m = 50), NRS, delta = 0.10, fold 1: follow the e1 greedy reduct R; at subset
sizes s in SIZES (s <= |R|), with B = R[:s] and the active set of R[:s-1], time one evaluation blocked(B, act) with
  KD    per-class cKDTree, k = 1 query within delta (revised KD engine, the paper's engine);
  BALL  per-class sklearn BallTree, k = 1 query, hits within delta confirmed with the common d2 test;
  BF    brute force in blocks of 512 rows (e1 BF-PA engine).
All three return the same mask (checked). KD_build: time to build the per-class trees alone (R1.8). Median of REPS repeats, one process, nothing else running.
Output: index_bench.jsonl.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "e1")); NUM = os.path.join(HERE, "..", "numerics")
sys.path.insert(0, E1); sys.path.insert(0, NUM)
from data_e1 import load      # noqa: E402
from run_e1 import folds      # noqa: E402
import engines_rev as REV     # noqa: E402

SIZES = (1, 2, 4, 8, 16, 32, 64, 96)
REPS = 3
DELTA = 0.10


def ball_blocked(X, y, delta, B, act):
    from sklearn.neighbors import BallTree
    Z = np.ascontiguousarray(X[:, B]); out = np.zeros(len(act), bool); ya = y[act]; d2 = delta * delta
    for k in np.unique(y):
        idx = np.flatnonzero(y == k); qs = np.flatnonzero((ya != k) & ~out)
        if not len(qs):
            continue
        d, j = BallTree(Z[idx]).query(Z[act[qs]], k=1)
        d, j = d[:, 0], j[:, 0]
        f = d <= delta * (1 + 1e-9)
        ok = REV.d2seq(X, act[qs[f]], idx[j[f]], B) <= d2
        out[qs[f][ok]] = True
    return out


def main():
    ref = {}
    for ln in open(os.path.join(E1, "e1_raw.jsonl")):
        r = json.loads(ln)
        if r["criterion"] == "NRS" and r["param"] == "0.10" and r["fold"] == 1 and r.get("status") == "ok" and "R" in r:
            ref.setdefault(r["dataset"], r["R"])
    out = open(os.path.join(HERE, "index_bench.jsonl"), "a")
    for slug in ("miniboone", "gas-drift"):
        X, y = load(slug); tr = folds(y)[0][0]; X, y = X[tr], y[tr]; R = ref[slug]
        kd = REV.KD(X, y, DELTA); bf = REV.E.make("BF-PA", X, y, DELTA)
        for s in [s for s in SIZES if s <= len(R)] + [len(R)]:
            act = np.arange(len(y)) if s == 1 else None
            if act is None:
                act = np.arange(len(y)); act = act[kd.blocked(R[:s - 1], act)]
            B = R[:s]; t = {}; masks = {}
            for name, fn in (("KD", lambda: kd.blocked(B, act)), ("BALL", lambda: ball_blocked(X, y, DELTA, B, act)),
                             ("BF", lambda: bf.blocked(B, act))):
                ts = []
                for _ in range(REPS):
                    t0 = time.perf_counter(); masks[name] = fn(); ts.append(time.perf_counter() - t0)
                t[name] = float(np.median(ts))
            # share of tree construction in the KD evaluation (R1.8): build the per-class trees alone
            from scipy.spatial import cKDTree
            Z = np.ascontiguousarray(X[:, B]); tb = []
            for _ in range(REPS):
                t0 = time.perf_counter()
                for k in np.unique(y):
                    cKDTree(Z[y == k])
                tb.append(time.perf_counter() - t0)
            t["KD_build"] = float(np.median(tb))
            same = all(np.array_equal(masks["KD"], masks[k]) for k in masks)
            rec = dict(dataset=slug, size=s, active=int(len(act)), n=int(len(y)), time=t, same_mask=same)
            out.write(json.dumps(rec) + "\n"); out.flush(); print(rec, flush=True)


if __name__ == "__main__":
    main()
