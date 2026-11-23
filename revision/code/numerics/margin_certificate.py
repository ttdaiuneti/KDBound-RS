"""Margin certificate for the NRS runs (revision, R2.2 / R1.9).

For every (dataset, delta, fold) of the NRS grid with a finished exact run, follow the greedy path R_0 = {}, R_1, ...
of that run and, at every step t and for every remaining candidate a (all candidates, a superset of what any method
evaluates), compute for every active object x the squared distance to a nearest enemy on B = R_t + [a] and its
relative gap to delta^2:  g(x, B) = |ne2_B(x) - delta^2| / delta^2.  Objects with no enemy within delta (1 + 1e-6) get
g >= 2e-6 (recorded as 2e-6). The rounding error of any engine's computed d2 is at most about (|B| + 3) u relative
(u = 2^-53), i.e. below 1e-13 for |B| <= 128. If min g exceeds this, every engine of the e1 code and of the revised
code decides every pair as exact arithmetic does; the FNC first-step window (WIN = 1e-9) is covered when min g over
one-attribute sets exceeds 2 WIN / delta + 1e-13 (at most 4.0e-8).

python margin_certificate.py            -> all units, 4 at a time (resumable; margin_certificate.jsonl)
python margin_certificate.py reverse [W] -> same, from the end of the list, W workers
python margin_certificate.py one SLUG DELTA FOLD -> one unit
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "e1"))
OUT = os.path.join(HERE, "margin_certificate.jsonl")
CAP = 3600; WORKERS = 4; CAPG = 2e-6
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONWARNINGS="ignore")


def paths():
    P = {}
    for ln in open(os.path.join(E1, "e1_raw.jsonl")):
        r = json.loads(ln)
        if r["criterion"] == "NRS" and r.get("status") == "ok" and "R" in r and r["method"] != "FARA":
            P.setdefault((r["dataset"], r["param"], r["fold"]), (r["R"], r["method"]))
    return P


def one(slug, param, fold):
    sys.path.insert(0, E1); sys.path.insert(0, HERE)
    from data_e1 import load
    from run_e1 import folds
    import engines_rev as REV
    R, src = paths()[(slug, param, fold)]
    X, y = load(slug); tr = folds(y)[fold - 1][0]; X, y = X[tr], y[tr]
    delta = float(param); d2 = delta * delta; n, m = X.shape; classes = np.unique(y)
    eng = REV.KD(X, y, delta)
    t0 = time.perf_counter(); act = np.arange(n); gmin, arg, n_pairs, n12, n9, gmin1 = np.inf, None, 0, 0, 0, np.inf
    steps_done = 0; complete = True
    for t in range(len(R) + 1):
        if not len(act):
            break
        Rt = R[:t]; rem = [a for a in range(m) if a not in Rt]
        for a in rem:
            B = Rt + [a]; Z = np.ascontiguousarray(X[:, B]); ya = y[act]
            best = np.full(len(act), np.inf)
            for k in classes:
                idx = np.flatnonzero(y == k); qs = np.flatnonzero(ya != k)
                if not len(qs) or not len(idx):
                    continue
                dd, j = cKDTree(Z[idx]).query(Z[act[qs]], k=1, distance_upper_bound=delta * (1 + 1e-6))
                f = np.isfinite(dd)
                if f.any():
                    s = REV.d2seq(X, act[qs[f]], idx[j[f]], B)
                    best[qs[f]] = np.minimum(best[qs[f]], s)
            g = np.where(np.isfinite(best), np.abs(best - d2) / d2, CAPG); g = np.minimum(g, CAPG)
            n_pairs += len(g); n12 += int((g < 1e-12).sum()); n9 += int((g < 1e-9).sum())
            i = int(g.argmin())
            if g[i] < gmin:
                gmin, arg = float(g[i]), dict(step=t, attr=a, B_size=len(B), obj=int(act[i]))
            if len(B) == 1:
                gmin1 = min(gmin1, float(g.min()))
        steps_done += 1
        if t < len(R):
            act = act[eng.blocked(R[:t + 1], act)]
        if time.perf_counter() - t0 > CAP:
            complete = t >= len(R); break
    print(json.dumps(dict(dataset=slug, param=param, fold=fold, path_from=src, R=R, steps_checked=steps_done,
                          steps_total=len(R) + 1, complete=complete, checks=n_pairs, min_gap=gmin, min_gap_1attr=gmin1,
                          argmin=arg, n_gap_lt_1e12=n12, n_gap_lt_1e9=n9, time=time.perf_counter() - t0)))


def main(reverse=False, workers=WORKERS):
    P = paths(); have = set()
    if os.path.exists(OUT):
        for ln in open(OUT):
            r = json.loads(ln); have.add((r["dataset"], r["param"], r["fold"]))
    todo = sorted((k for k in P if k not in have), reverse=reverse)

    def work(k):
        p = subprocess.run([sys.executable, __file__, "one", k[0], k[1], str(k[2])], capture_output=True, text=True,
                           env=ENV, timeout=CAP + 1800)
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
        rec = json.loads(lines[-1]) if lines else dict(dataset=k[0], param=k[1], fold=k[2], status="error",
                                                       err=p.stderr[-400:])
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(k, rec.get("min_gap"), rec.get("complete"), round(rec.get("time", -1), 1), flush=True)
    print(len(todo), "units", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, todo))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        one(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif len(sys.argv) > 1 and sys.argv[1] == "reverse":   # second process from the other end (duplicates removed in analysis)
        main(reverse=True, workers=int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    else:
        main()
