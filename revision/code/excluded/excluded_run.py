"""The two tables dropped at screening (revision, R1.13): eeg-eye-state and numerai28-6 (research-commons
standardized/v1, both below the 1e5 cap). They were dropped before the per-criterion rule of Section 5.1 was adopted,
because greedy NRS selected fewer than two attributes; greedy FRS selects 7-8 (eeg) and 21 (numerai) attributes on the
whole table (screen 2026-11-23). This runs the grid of Section 5 on them: NRS (delta in 0.05..0.20; KD-PA, KD-PA+B3,
FNC, FNC+B3, HASH) and FRS (sigma 0.25 and std; KD-PA, KD-PA+B3, FA-FPR, FA-FPR+B3); same folds, 900 s per unit,
skip after a fold-1 timeout; one unit at a time (timing).
NRS engines: numerics/engines_rev.py (same decisions as e1 on the e1 grid, margin certificate); FRS: e1/frs.py.

python excluded_run.py            -> run (resumable; excluded_run.jsonl)
python excluded_run.py one SLUG CRIT PARAM FOLD METHOD
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "e1")); NUM = os.path.join(HERE, "..", "numerics")
OUT = os.path.join(HERE, "excluded_run.jsonl"); LIMIT = 900
SLUGS = ("eeg-eye-state", "numerai28-6")
GRID = [("NRS", d, m) for d in ("0.05", "0.10", "0.15", "0.20") for m in ("KD-PA+B3", "KD-PA", "FNC+B3", "FNC", "HASH")] + \
       [("FRS", s, m) for s in ("0.25", "std") for m in ("KD-PA+B3", "KD-PA", "FA-FPR+B3", "FA-FPR")]
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONWARNINGS="ignore")


def load(slug):
    import pandas as pd
    df = pd.read_csv(os.path.expanduser(f"~/Documents/research-commons/datasets/{slug}/standardized/v1/data.csv"))
    return df.drop(columns="label").to_numpy(float), df["label"].to_numpy()


def one(slug, crit, param, fold, method):
    sys.path.insert(0, E1); sys.path.insert(0, NUM)
    from run_e1 import folds
    X, y = load(slug); tr = folds(y)[fold - 1][0]; X, y = X[tr], y[tr]
    t0 = time.perf_counter()
    if crit == "NRS":
        import engines_rev as REV
        R, evals, _ = REV.run_nrs(method, X, y, float(param))
    else:
        import frs as F
        R, evals, _ = F.run_frs(method, X, y, param if param == "std" else float(param))
    print(json.dumps(dict(dataset=slug, criterion=crit, param=param, fold=fold, method=method, n_train=int(len(y)),
                          d=int(X.shape[1]), status="ok", R=[int(a) for a in R], evals=evals,
                          time=time.perf_counter() - t0)))


def main():
    have = {}
    if os.path.exists(OUT):
        for ln in open(OUT):
            r = json.loads(ln); have[(r["dataset"], r["criterion"], r["param"], r["fold"], r["method"])] = r["status"]
    for slug in SLUGS:
        for crit, param, method in GRID:
            for fold in range(1, 6):
                k = (slug, crit, param, fold, method)
                if k in have:
                    continue
                if fold > 1 and have.get((slug, crit, param, 1, method)) in ("timeout", "skipped", "error"):
                    rec = dict(zip(("dataset", "criterion", "param", "fold", "method"), k), status="skipped")
                else:
                    try:
                        p = subprocess.run([sys.executable, __file__, "one", slug, crit, param, str(fold), method],
                                           capture_output=True, text=True, env=ENV, timeout=LIMIT)
                        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
                        rec = json.loads(lines[-1]) if lines else dict(
                            zip(("dataset", "criterion", "param", "fold", "method"), k), status="error",
                            err=p.stderr[-400:])
                    except subprocess.TimeoutExpired:
                        rec = dict(zip(("dataset", "criterion", "param", "fold", "method"), k), status="timeout")
                with open(OUT, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                have[k] = rec["status"]
                print(k, rec["status"], round(rec.get("time", -1), 1), rec.get("evals"), len(rec.get("R", [])), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        one(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6])
    else:
        main()
