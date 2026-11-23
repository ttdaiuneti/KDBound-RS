"""Per-step instrumented runs of KD-PA+B3 (revision: R1.6 first-step cost, R1.10 FRS threshold, R1.11 radius vs
active objects vs bound tightness vs pruning).

NRS: revised KD engine (numerics/engines_rev.py), greedy identical to e1 greedy_nrs (selection, tie-break, stop);
     per step: |Act|, candidates, every evaluated candidate (attribute, bound, gain), the bounds of all candidates in
     scan order (ub_all), witness time, step time.
FRS (minimum t-norm, Kleene-Dienes; e1/frs.py engine KD + positive approximation + bound) with tau = 0: greedy is
     identical to frs.greedy except the threshold. Since tau only decides when the search stops (the selected
     attribute at each step is the argmax whatever tau is, and a candidate is skipped only if its bound is below the
     best gain or not above tau), the reduct for any tau > 0 is the prefix of this path before the first step whose
     best gain is <= tau.
Every unit checks its reduct against the e1 reduct of the same configuration (NRS) or the tau = 1e-9 prefix (FRS).
Units: all NRS (dataset, delta, fold) and FRS (dataset, sigma, fold) with a finished exact e1 run; 900 s cap inside
the unit (partial steps kept); one unit at a time.

python steps_run.py [nrs|frs]       -> run (resumable; steps_run.jsonl)
python steps_run.py one CRIT SLUG PARAM FOLD
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "e1"))
NUM = os.path.abspath(os.path.join(HERE, "..", "numerics"))
OUT = os.path.join(HERE, "steps_run.jsonl")
CAP = 900
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONWARNINGS="ignore")


def e1_reducts():
    P = {}
    for ln in open(os.path.join(E1, "e1_raw.jsonl")):
        r = json.loads(ln)
        if r.get("status") == "ok" and "R" in r and r["method"] != "FARA":
            P.setdefault((r["criterion"], r["dataset"], r["param"], r["fold"]), r["R"])
    return P


def nrs(X, y, delta, T0):
    import engines_rev as REV
    E = REV.E
    eng = REV.KD(X, y, delta); n, m = X.shape
    act = np.arange(n); R, rem, steps, partial = [], list(range(m)), [], False
    while rem and len(act):
        t0 = time.perf_counter(); ub = None
        if R:
            ub = E.bound_ub(X, act, eng.witness(R, act), R, rem, eng.d2)
            order = sorted(rem, key=lambda a: (-ub[a], a))
        else:
            order = rem
        tw = time.perf_counter() - t0
        best_gain, best, best_mask, ev = 0, None, None, []
        for a in order:
            if ub is not None and ub[a] < max(best_gain, 1):
                break
            mask = eng.blocked(R + [a], act)
            g = int(len(act) - mask.sum()); ev.append([a, None if ub is None else ub[a], g])
            if g > best_gain or (g == best_gain and g > 0 and a < best):
                best_gain, best, best_mask = g, a, mask
        steps.append(dict(active=int(len(act)), candidates=len(rem), evaluated=len(ev), ev=ev, selected=best,
                          gain=best_gain, ub_all=None if ub is None else [ub[a] for a in order],
                          t_witness=tw, time=time.perf_counter() - t0))
        if best_gain == 0:
            break
        R.append(best); rem.remove(best); act = act[best_mask]
        if time.perf_counter() - T0 > CAP:
            partial = True; break
    return R, steps, partial


def frs(X, y, sigma, T0):
    import frs as F
    Z = F.scale(X, sigma); eng = F.KD(Z, y); n, m = Z.shape; TOL = 0.0
    act = np.arange(n); R, rem, steps, partial = [], list(range(m)), [], False
    mu_act = np.zeros(n)
    while rem and len(act):
        t0 = time.perf_counter(); ub = None
        if R:
            ne, ys = eng.ne(R, act); ys = np.where(ys < 0, act, ys)
            m0 = np.minimum(1.0, ne)
            ub = {a: math.fsum(np.minimum(1.0, np.maximum(ne, np.abs(Z[act, a] - Z[ys, a]))) - m0) for a in rem}
            order = sorted(rem, key=lambda a: (-ub[a], a))
        else:
            order = rem
        tw = time.perf_counter() - t0
        best_gain, best, best_mu, ev = 0.0, None, None, []
        for a in order:
            if ub is not None and (ub[a] < best_gain or ub[a] <= TOL):
                break
            ne_a, _ = eng.ne(R + [a], act)
            mu = np.minimum(1.0, ne_a); g = math.fsum(mu - mu_act); ev.append([a, None if ub is None else ub[a], g])
            if g > best_gain or (g == best_gain and best is not None and a < best):
                best_gain, best, best_mu = g, a, mu
        steps.append(dict(active=int(len(act)), candidates=len(rem), evaluated=len(ev), ev=ev, selected=best,
                          gain=best_gain, ub_all=None if ub is None else [ub[a] for a in order],
                          t_witness=tw, time=time.perf_counter() - t0))
        if best is None or best_gain <= TOL:
            break
        R.append(best); rem.remove(best)
        keep = best_mu < 1.0; act, mu_act = act[keep], best_mu[keep]
        if time.perf_counter() - T0 > CAP:
            partial = True; break
    return R, steps, partial


def prefix(steps, tau):
    out = []
    for s in steps:
        if s["selected"] is None or s["gain"] <= tau:
            return out
        out.append(s["selected"])
    return None          # path ended (all attributes or partial) before the gain fell to tau


def one(crit, slug, param, fold):
    sys.path.insert(0, E1); sys.path.insert(0, NUM)
    from data_e1 import load
    from run_e1 import folds
    X, y = load(slug); tr = folds(y)[fold - 1][0]; X, y = X[tr], y[tr]
    T0 = time.perf_counter()
    if crit == "NRS":
        R, steps, partial = nrs(X, y, float(param), T0)
    else:
        R, steps, partial = frs(X, y, param if param == "std" else float(param), T0)
    ref = e1_reducts().get((crit, slug, param, fold))
    rec = dict(criterion=crit, dataset=slug, param=param, fold=fold, n_train=int(len(y)), m=int(X.shape[1]), R=R,
               partial=partial, time=time.perf_counter() - T0, steps=steps, e1_R=ref)
    if crit == "NRS":
        rec["same_as_e1"] = None if partial else (R == ref)
    else:
        p = prefix(steps, 1e-9) if not partial else None
        if p is None and not partial:
            p = R
        rec["same_as_e1"] = None if p is None else (p == ref)
    print(json.dumps(rec, default=float))


def main(which):
    P = e1_reducts(); have = set()
    if os.path.exists(OUT):
        for ln in open(OUT):
            r = json.loads(ln); have.add((r["criterion"], r["dataset"], r["param"], r["fold"]))
    todo = sorted(k for k in P if k[0] in which and k not in have)
    print(len(todo), "units", flush=True)
    for k in todo:
        p = subprocess.run([sys.executable, __file__, "one", k[0], k[1], k[2], str(k[3])], capture_output=True,
                           text=True, env=ENV, timeout=CAP + 1200)
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
        rec = json.loads(lines[-1]) if lines else dict(criterion=k[0], dataset=k[1], param=k[2], fold=k[3],
                                                       status="error", err=p.stderr[-400:])
        with open(OUT, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        print(k, rec.get("same_as_e1"), rec.get("partial"), round(rec.get("time", -1), 1), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        one(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    else:
        main(tuple(a.upper() for a in sys.argv[1:]) or ("NRS", "FRS"))
