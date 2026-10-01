"""Supplementary scaling run (e1_prereg.md section 13): reduction time against the number of objects n.

python scale_n.py                          -> run every unit in sequence (resumable; append-only scale_n.jsonl)
python scale_n.py one SLUG N CRIT PARAM M  -> one unit in a fresh process (used internally)

n < 1e5: stratified subsample (seed 0) of the capped E1 table; n = 1e5: the table itself (fold 1 then equals E1).
Reduction on the training part of fold 1 (StratifiedKFold(5, shuffle=True, random_state=0)). 900 s / 32 GB caps as
in E1 (run_capped). Skip rule: after a timeout or memout at some n, larger n are recorded as skipped-larger-n.
"""
import json
import os
import signal
import sys
import time

import numpy as np

os.environ["E1_NO_REDUCT"] = "1"
from run_e1 import run_capped, folds, code_version, HERE, LIMIT  # noqa: E402

OUT = os.path.join(HERE, "scale_n.jsonl")
NS = (12_500, 25_000, 50_000, 100_000)
SLUGS = ("hls4ml-jets", "miniboone")
METHODS = {("NRS", "0.10"): ("KD-PA+B3", "KD-PA", "FNC", "HASH"), ("FRS", "0.25"): ("KD-PA+B3", "KD-PA", "FA-FPR")}


def subsample(slug, n):
    from data_e1 import load
    from sklearn.model_selection import train_test_split
    X, y = load(slug)
    if n < len(y):
        i = np.sort(train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=0)[0])
        X, y = X[i], y[i]
    return X, y


def one(slug, n, crit, param, method):
    X, y = subsample(slug, n); tr = folds(y)[0][0]; X, y = X[tr], y[tr]

    def alarm(*_):
        raise TimeoutError
    signal.signal(signal.SIGALRM, alarm); signal.alarm(LIMIT)
    rec = dict(dataset=slug, n=n, n_train=int(len(y)), criterion=crit, param=param, method=method, code=code_version())
    t0 = time.perf_counter()
    try:
        if crit == "NRS":
            from engines_e1 import run_nrs
            R, evals, _ = run_nrs(method, X, y, float(param))
        else:
            from frs import run_frs
            R, evals, _ = run_frs(method, X, y, float(param))
        rec.update(status="ok", R=[int(a) for a in R], evals=int(evals))
    except TimeoutError:
        rec.update(status="timeout")
    except Exception as e:  # noqa: BLE001
        rec.update(status="memout" if type(e).__name__ == "MemOut" else "error", note=f"{type(e).__name__}: {e}")
    signal.alarm(0)
    rec["time"] = time.perf_counter() - t0
    print(json.dumps(rec), flush=True)


def main():
    done = [json.loads(l) for l in open(OUT)] if os.path.exists(OUT) else []
    key = {(r["dataset"], r["n"], r["criterion"], r["param"], r["method"]) for r in done}
    failed = {(r["dataset"], r["criterion"], r["param"], r["method"]) for r in done if r["status"] != "ok"}
    for slug in SLUGS:
        for (crit, param), ms in METHODS.items():
            for m in ms:
                for n in NS:
                    k = (slug, n, crit, param, m)
                    if k in key:
                        continue
                    if (slug, crit, param, m) in failed:
                        r = dict(dataset=slug, n=n, criterion=crit, param=param, method=m, status="skipped-larger-n")
                    else:
                        out, err, note, wall, peak = run_capped([sys.executable, os.path.abspath(__file__), "one", slug,
                                                                 str(n), crit, param, m])
                        lines = [ln for ln in out.splitlines() if ln.startswith("{")]
                        r = json.loads(lines[-1]) if lines and not note else dict(
                            dataset=slug, n=n, criterion=crit, param=param, method=m, time=wall, note=note or err[-300:],
                            status="memout" if note and note != "hard kill" else ("timeout" if note else "error"))
                        r["peak_footprint_gb"] = round(peak, 2)
                        if r["status"] != "ok":
                            failed.add((slug, crit, param, m))
                    with open(OUT, "a") as f:
                        f.write(json.dumps(r) + "\n")
                    print(f"{slug:12s} n={n:6d} {crit} {param} {m:9s} {r['status']:16s} t={r.get('time', float('nan')):.1f}",
                          flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        one(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])
    else:
        main()
