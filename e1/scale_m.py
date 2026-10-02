"""Supplementary run (e1_prereg.md section 14): reduction time against the number of attributes m on one table.

python scale_m.py                            -> run every unit in sequence (resumable; append-only scale_m.jsonl)
python scale_m.py one M SEED CRIT PARAM METH -> one unit in a fresh process (used internally)

gas-drift, training part of fold 1; for m < 128 the first m attributes of a random permutation (seed SEED), nested in m;
m = 128 is the whole table (run once, seed recorded as -1). 900 s / 32 GB caps as in E1. Skip rule: after a timeout or
memout at some m, larger m of the same (criterion, method, seed) are recorded as skipped-larger-m.
"""
import json
import os
import signal
import sys
import time

import numpy as np

os.environ["E1_NO_REDUCT"] = "1"
from run_e1 import run_capped, folds, code_version, HERE, LIMIT  # noqa: E402

OUT = os.path.join(HERE, "scale_m.jsonl")
SLUG, MS, SEEDS = "gas-drift", (16, 32, 64), (0, 1, 2)
GRID = (("NRS", "0.10"), ("FRS", "0.25"))
METHODS = ("KD-PA+B3", "KD-PA")


def one(m, seed, crit, param, method):
    from data_e1 import load
    X, y = load(SLUG); tr = folds(y)[0][0]; X, y = X[tr], y[tr]
    if seed >= 0:
        X = X[:, np.random.default_rng(seed).permutation(X.shape[1])[:m]]

    def alarm(*_):
        raise TimeoutError
    signal.signal(signal.SIGALRM, alarm); signal.alarm(LIMIT)
    rec = dict(dataset=SLUG, m=m, seed=seed, n_train=int(len(y)), criterion=crit, param=param, method=method,
               code=code_version())
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


def units():
    for crit, param in GRID:
        for meth in METHODS:
            for seed in SEEDS:
                for m in MS:
                    yield m, seed, crit, param, meth
            yield 128, -1, crit, param, meth


def main():
    done = [json.loads(l) for l in open(OUT)] if os.path.exists(OUT) else []
    key = {(r["m"], r["seed"], r["criterion"], r["method"]) for r in done}
    failed = {(r["seed"], r["criterion"], r["method"]) for r in done if r["status"] != "ok"}
    for m, seed, crit, param, meth in units():
        if (m, seed, crit, meth) in key:
            continue
        if (seed, crit, meth) in failed:
            r = dict(dataset=SLUG, m=m, seed=seed, criterion=crit, param=param, method=meth, status="skipped-larger-m")
        else:
            out, err, note, wall, peak = run_capped([sys.executable, os.path.abspath(__file__), "one", str(m), str(seed),
                                                     crit, param, meth])
            lines = [ln for ln in out.splitlines() if ln.startswith("{")]
            r = json.loads(lines[-1]) if lines and not note else dict(
                dataset=SLUG, m=m, seed=seed, criterion=crit, param=param, method=meth, time=wall,
                note=note or err[-300:], status="memout" if note and note != "hard kill" else ("timeout" if note else "error"))
            r["peak_footprint_gb"] = round(peak, 2)
            if r["status"] != "ok":
                failed.add((seed, crit, meth))
        with open(OUT, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"m={m:4d} seed={seed:2d} {crit} {meth:9s} {r['status']:16s} t={r.get('time', float('nan')):.1f} "
              f"|R|={len(r.get('R', []))} evals={r.get('evals')}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        one(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])
    else:
        main()
