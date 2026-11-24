"""Supplementary run for the revision (R1.2): Gaussian kernel FRS (Hu et al. 2010) with and without the witness bound.

Fixed before the run (2026-11-21): datasets htru2 (m=8), magic-telescope (10), house-16h (16), creditcard (29),
miniboone (50), gas-drift (128); kernel parameter kappa in {0.1, 0.2}; eps = 1e-4 (tau = eps * n); methods KD, KD+B3;
the five folds of the main runs (training part); 900 s per unit, one unit at a time; after a timeout on fold 1,
folds 2-5 of that (dataset, kappa, method) are recorded as skipped. Exactness of BF/KD +- B3 was checked beforehand on
stratified samples of 1000 objects (gauss_check.py / gauss_check.log).
Amendment 2026-11-21, before any eps=1e-4 unit: eps = 0.01 stopped greedy at the first step on magic-telescope
(first-step gamma gain 3e-3 to 7e-3 on full folds; probe with tau = 0 on fold 1 of four tables gave first-step gains
1e-3 to 9e-2); those rows are kept in gauss_run_eps0.01_degenerate.jsonl and not used. eps set to 1e-4.

Amendment 2026-11-24, written before any unit of the datasets below (the first run covered six datasets only): the
same design (kappa, eps, methods, folds, 900 s, skip rule, one unit at a time) on the
other eight datasets of Table 3: mammography, electrical-grid-stability, fried, internet-firewall, run-or-walk,
hls4ml-jets, jannis, vehicle-sensit (run with "python gauss_run.py ext"; rows go to the same gauss_run.jsonl).

python gauss_run.py            -> run all units in sequence (resumable; append-only gauss_run.jsonl)
python gauss_run.py one SLUG KAPPA FOLD METHOD -> one unit (internal)
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.abspath(os.path.join(HERE, "..", "..", "e1")); sys.path.insert(0, E1)
OUT = os.path.join(HERE, "gauss_run.jsonl")
SLUGS = ("htru2", "magic-telescope", "house-16h", "creditcard", "miniboone", "gas-drift")
SLUGS_EXT = ("mammography", "electrical-grid-stability", "fried", "internet-firewall", "run-or-walk", "hls4ml-jets",
             "jannis", "vehicle-sensit")
KAPPAS = ("0.1", "0.2")
EPS = 1e-4
METHODS = ("KD+B3", "KD")
LIMIT = 900


def one(slug, kappa, fold, method):
    from data_e1 import load
    from run_e1 import folds
    import frs_gauss as G
    X, y = load(slug); tr = folds(y)[fold - 1][0]; X, y = X[tr], y[tr]
    t0 = time.perf_counter()
    R, evals, steps = G.run(method, X, y, float(kappa), EPS)
    rec = dict(dataset=slug, kappa=kappa, eps=EPS, fold=fold, method=method, n_train=int(len(y)), m=int(X.shape[1]),
               status="ok", R=[int(a) for a in R], evals=evals, time=time.perf_counter() - t0, steps=steps)
    print(json.dumps(rec, default=float))


def done():
    if not os.path.exists(OUT):
        return {}
    d = {}
    for ln in open(OUT):
        r = json.loads(ln); d[(r["dataset"], r["kappa"], r["fold"], r["method"])] = r["status"]
    return d


def main(slugs=SLUGS):
    have = done()
    for slug in slugs:
        for kappa in KAPPAS:
            for method in METHODS:
                for fold in range(1, 6):
                    k = (slug, kappa, fold, method)
                    if k in have:
                        continue
                    if fold > 1 and have.get((slug, kappa, 1, method)) in ("timeout", "skipped"):
                        rec = dict(dataset=slug, kappa=kappa, eps=EPS, fold=fold, method=method, status="skipped")
                    else:
                        try:
                            p = subprocess.run([sys.executable, __file__, "one", slug, kappa, str(fold), method],
                                               capture_output=True, text=True, timeout=LIMIT)
                            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
                            rec = json.loads(lines[-1]) if lines else dict(
                                dataset=slug, kappa=kappa, eps=EPS, fold=fold, method=method, status="error",
                                err=p.stderr[-500:])
                        except subprocess.TimeoutExpired:
                            rec = dict(dataset=slug, kappa=kappa, eps=EPS, fold=fold, method=method, status="timeout")
                    with open(OUT, "a") as f:
                        f.write(json.dumps(rec, default=float) + "\n")
                    have[k] = rec["status"]
                    print(slug, kappa, fold, method, rec["status"], round(rec.get("time", -1), 1), rec.get("evals"),
                          flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        one(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    elif len(sys.argv) > 1 and sys.argv[1] == "ext":
        main(SLUGS_EXT)
    else:
        main()
