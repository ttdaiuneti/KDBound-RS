"""Revision run for R1.14: the three tables capped at 1e5 objects, at their full size.

Fixed before the run (2026-11-23): creditcard (284 807 objects), miniboone (130 064), hls4ml-jets (830 000), commons
standardized/v1 without the cap; training part of fold 1 (StratifiedKFold(5, shuffle=True, random_state=0) on the
full table); NRS delta = 0.10 and FRS sigma = 0.25 (the settings of the scaling run, Appendix A.4); methods KD-PA+B3
and KD-PA; 3 h per unit, 32 GB memory cap (run_capped of e1); one unit at a time, nothing else running.
Existing exact methods are not run: at 1e5 objects (8e4 training) they already exceeded 900 s or the memory limit on
these tables, or were far slower (Table 11, Appendix A.4).
NRS engine: numerics/engines_rev.py (released code); FRS: e1/frs.py.

python bign_run.py                       -> all units (resumable; append-only bign_run.jsonl)
python bign_run.py one SLUG CRIT PARAM METHOD
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "e1")); NUM = os.path.join(HERE, "..", "numerics")
sys.path.insert(0, E1); sys.path.insert(0, NUM)
OUT = os.path.join(HERE, "bign_run.jsonl"); LIMIT = 3 * 3600
SLUGS = ("creditcard", "miniboone", "hls4ml-jets")
GRID = [("NRS", "0.10", "KD-PA+B3"), ("NRS", "0.10", "KD-PA"), ("FRS", "0.25", "KD-PA+B3"), ("FRS", "0.25", "KD-PA")]
# Amendment 2026-11-23 (after the 12 units above, before any of these): on creditcard the fastest existing exact
# method at 1e5 objects was FNC (NRS, delta = 0.10, Table 5), so FNC and FNC+B3 are run there at full size (same caps).
# On miniboone and hls4ml-jets the existing methods had exceeded the limits at <= 1e5 objects or took 662 s (HASH,
# hls4ml-jets NRS), and are not run.
EXTRA = [("creditcard", "NRS", "0.10", "FNC"), ("creditcard", "NRS", "0.10", "FNC+B3")]


def load_full(slug):
    import pandas as pd
    df = pd.read_csv(os.path.expanduser(f"~/Documents/research-commons/datasets/{slug}/standardized/v1/data.csv"))
    return df.drop(columns="label").to_numpy(float), df["label"].to_numpy()


def one(slug, crit, param, method):
    from run_e1 import folds
    X, y = load_full(slug); n = len(y); tr = folds(y)[0][0]; X, y = X[tr], y[tr]
    t0 = time.perf_counter()
    if crit == "NRS":
        from engines_rev import run_nrs
        R, evals, _ = run_nrs(method, X, y, float(param))
    else:
        from frs import run_frs
        R, evals, _ = run_frs(method, X, y, float(param))
    rec = dict(dataset=slug, criterion=crit, param=param, method=method, n=int(n), n_train=int(len(y)), m=int(X.shape[1]), status="ok",
               R=[int(a) for a in R], evals=int(evals), time=time.perf_counter() - t0)
    print(json.dumps(rec), flush=True)


def main():
    import run_e1
    run_e1.LIMIT = LIMIT                                    # hard kill in run_capped at LIMIT + 120 s
    have = {(r["dataset"], r["criterion"], r["param"], r["method"]) for r in map(json.loads, open(OUT))} \
        if os.path.exists(OUT) else set()
    for slug, crit, param, m in [(s, c, p, m) for s in SLUGS for c, p, m in GRID] + list(EXTRA):
        if True:
            if (slug, crit, param, m) in have:
                continue
            out, err, note, wall, peak = run_e1.run_capped([sys.executable, os.path.abspath(__file__), "one", slug,
                                                            crit, param, m])
            lines = [ln for ln in out.splitlines() if ln.startswith("{")]
            if lines and not note:
                r = json.loads(lines[-1])
            else:
                st = "timeout" if note == "hard kill" else ("memout" if note else "error")
                r = dict(dataset=slug, criterion=crit, param=param, method=m, status=st, time=wall,
                         note=note or err[-300:])
            r["peak_gb"] = round(peak, 2)
            with open(OUT, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(slug, crit, m, r["status"], round(r.get("time", -1), 1), r.get("evals"), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        one(*sys.argv[2:6])
    else:
        main()
