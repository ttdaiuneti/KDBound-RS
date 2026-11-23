"""Witness choice and pruning (revision, R1.3). Theorem 1 holds for any enemy as witness; this run measures how the
choice changes the number of evaluations of the pruned greedy search (machine-independent; times not used).

Witnesses, recomputed at every step on the current subset R:
  NE      a nearest enemy on R (the paper's choice);
  K5      the 5th nearest enemy on R (any valid but farther enemy);
  RAND    an enemy drawn uniformly at random from all enemies (seed 0), whatever its distance.
NRS: revised KD engine, bound = count of objects with d2_R(x, w) + (x_a - w_a)^2 > delta^2 - 1e-12.
FRS (minimum t-norm): e1/frs.py KD engine, bound = fsum over x of min(1, max(rho_R(x, w), |Z_xa - Z_wa|)) - mu_R(x).
Greedy otherwise identical to the e1 drivers; the reduct is checked against e1. Fold 1 of every configuration with a
finished exact e1 run; 900 s cap per unit.

python witness_run.py            -> run (resumable; witness_run.jsonl)
python witness_run.py one CRIT SLUG PARAM WITNESS
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "e1"))
NUM = os.path.abspath(os.path.join(HERE, "..", "numerics"))
OUT = os.path.join(HERE, "witness_run.jsonl")
CAP = 900; K = 5; WITNESSES = ("NE", "K5", "RAND")
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONWARNINGS="ignore")


def e1_reducts():
    P = {}
    for ln in open(os.path.join(E1, "e1_raw.jsonl")):
        r = json.loads(ln)
        if r.get("status") == "ok" and "R" in r and r["method"] != "FARA" and r["fold"] == 1:
            P.setdefault((r["criterion"], r["dataset"], r["param"]), r["R"])
    return P


class Witness:
    def __init__(self, Z, y, kind, p):
        self.Z, self.y, self.kind, self.p = Z, y, kind, p
        self.classes = np.unique(y); self.rng = np.random.default_rng(0)
        self.enemies = {k: np.flatnonzero(y != k) for k in self.classes}

    def __call__(self, R, act):
        Z, y = self.Z, self.y; ya = y[act]; w = np.empty(len(act), np.int64)
        if self.kind == "RAND":
            for k in self.classes:
                qs = np.flatnonzero(ya == k); en = self.enemies[k]
                w[qs] = en[self.rng.integers(len(en), size=len(qs))]
            return w
        ZR = np.ascontiguousarray(Z[:, R]); kk = 1 if self.kind == "NE" else K
        D = np.full((len(act), kk * (len(self.classes) - 1) + kk), np.inf); J = np.zeros(D.shape, np.int64); col = 0
        for c in self.classes:
            idx = np.flatnonzero(y == c); qs = np.flatnonzero(ya != c)
            kc = min(kk, len(idx))
            if not len(qs) or not kc:
                continue
            d, j = cKDTree(ZR[idx]).query(ZR[act[qs]], k=kc, p=self.p)
            d = d.reshape(len(qs), kc); j = j.reshape(len(qs), kc)
            D[qs, col:col + kc] = d; J[qs, col:col + kc] = idx[j]; col += kc
        D, J = D[:, :col], J[:, :col]
        o = np.argsort(D, axis=1, kind="stable")
        r = np.minimum(kk, np.isfinite(D).sum(1)) - 1                    # kk-th nearest (or the farthest available)
        return J[np.arange(len(act)), o[np.arange(len(act)), r]]


def nrs(X, y, delta, kind, T0):
    import engines_rev as REV
    E = REV.E; eng = REV.KD(X, y, delta); n, m = X.shape; wit = Witness(X, y, kind, 2)
    act = np.arange(n); R, rem, evals, per_step = [], list(range(m)), 0, []
    while rem and len(act):
        ub = None
        if R:
            ub = E.bound_ub(X, act, wit(R, act), R, rem, eng.d2); order = sorted(rem, key=lambda a: (-ub[a], a))
        else:
            order = rem
        best_gain, best, best_mask, ne_ = 0, None, None, 0
        for a in order:
            if ub is not None and ub[a] < max(best_gain, 1):
                break
            mask = eng.blocked(R + [a], act); evals += 1; ne_ += 1
            g = int(len(act) - mask.sum())
            if g > best_gain or (g == best_gain and g > 0 and a < best):
                best_gain, best, best_mask = g, a, mask
        per_step.append([len(rem), ne_])
        if best_gain == 0:
            break
        R.append(best); rem.remove(best); act = act[best_mask]
        if time.perf_counter() - T0 > CAP:
            return R, evals, per_step, True
    return R, evals, per_step, False


def frs(X, y, sigma, kind, T0):
    import frs as F
    Z = F.scale(X, sigma); eng = F.KD(Z, y); n, m = Z.shape; wit = Witness(Z, y, kind, np.inf); TOL = F.TOL
    act = np.arange(n); R, rem, evals, per_step = [], list(range(m)), 0, []
    mu_act = np.zeros(n)
    while rem and len(act):
        ub = None
        if R:
            w = wit(R, act); rho = np.abs(Z[act][:, R] - Z[w][:, R]).max(1)
            ub = {a: math.fsum(np.minimum(1.0, np.maximum(rho, np.abs(Z[act, a] - Z[w, a]))) - mu_act) for a in rem}
            order = sorted(rem, key=lambda a: (-ub[a], a))
        else:
            order = rem
        best_gain, best, best_mu, ne_ = 0.0, None, None, 0
        for a in order:
            if ub is not None and (ub[a] < best_gain or ub[a] <= TOL):
                break
            ne_a, _ = eng.ne(R + [a], act); evals += 1; ne_ += 1
            mu = np.minimum(1.0, ne_a); g = math.fsum(mu - mu_act)
            if g > best_gain or (g == best_gain and best is not None and a < best):
                best_gain, best, best_mu = g, a, mu
        per_step.append([len(rem), ne_])
        if best is None or best_gain <= TOL:
            break
        R.append(best); rem.remove(best)
        keep = best_mu < 1.0; act, mu_act = act[keep], best_mu[keep]
        if time.perf_counter() - T0 > CAP:
            return R, evals, per_step, True
    return R, evals, per_step, False


def one(crit, slug, param, kind):
    sys.path.insert(0, E1); sys.path.insert(0, NUM)
    from data_e1 import load
    from run_e1 import folds
    X, y = load(slug); tr = folds(y)[0][0]; X, y = X[tr], y[tr]
    T0 = time.perf_counter()
    if crit == "NRS":
        R, evals, per_step, partial = nrs(X, y, float(param), kind, T0)
    else:
        R, evals, per_step, partial = frs(X, y, param if param == "std" else float(param), kind, T0)
    ref = e1_reducts().get((crit, slug, param))
    print(json.dumps(dict(criterion=crit, dataset=slug, param=param, fold=1, witness=kind, m=int(X.shape[1]), R=R,
                          evals=evals, per_step=per_step, partial=partial, same_as_e1=None if partial else R == ref,
                          time=time.perf_counter() - T0)))


def main():
    P = e1_reducts(); have = set()
    if os.path.exists(OUT):
        for ln in open(OUT):
            r = json.loads(ln); have.add((r["criterion"], r["dataset"], r["param"], r["witness"]))
    todo = [k + (w,) for k in sorted(P) for w in WITNESSES if k + (w,) not in have]
    print(len(todo), "units", flush=True)
    for k in todo:
        try:
            p = subprocess.run([sys.executable, __file__, "one", *k], capture_output=True, text=True, env=ENV,
                               timeout=CAP + 1200)
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            rec = json.loads(lines[-1]) if lines else dict(criterion=k[0], dataset=k[1], param=k[2], witness=k[3],
                                                           status="error", err=p.stderr[-400:])
        except subprocess.TimeoutExpired:
            rec = dict(criterion=k[0], dataset=k[1], param=k[2], witness=k[3], status="timeout")
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(k, rec.get("evals"), rec.get("same_as_e1"), rec.get("partial"), round(rec.get("time", -1), 1), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        one(*sys.argv[2:6])
    else:
        main()
