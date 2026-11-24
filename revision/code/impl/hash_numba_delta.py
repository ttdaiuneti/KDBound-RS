"""Revision run for R2.3, second part (design fixed 2026-11-24, before any of these units was timed).

Question: does the margin of KD-PA+B3 over the compiled HASH (HASH-NB) on the tables with many attributes hold at the
other values of delta, not only at delta = 0.10 (hash_numba.py)?

Tables: the five NRS tables with m >= 29 (creditcard 29, miniboone 50, jannis 54, vehicle-sensit 100, gas-drift 128),
the split used in Section 5.4 and in the Gaussian run. delta in {0.05, 0.15, 0.20}. Fold 1 of the main runs. Methods:
KD-PA+B3 first, then HASH-NB (same kernel and class as hash_numba.py). One unit at a time in its own process, one
thread, 900 s cap. Rule fixed in advance: if KD-PA+B3 exceeds 900 s, HASH-NB is not run for that (table, delta),
since no ratio can be formed. The NumPy HASH is not re-run: in the main runs it exceeded 900 s on fold 1 in 12 of
these 15 settings and took 12-501 s in the other 3 (e1_raw.jsonl). Reducts are compared with the KD-PA+B3 reduct of
e1_raw.jsonl (or e1/remeasure_idle.jsonl), and are recorded even when no reference exists.
Output: hash_numba_delta.jsonl (append; done units skipped).

python hash_numba_delta.py run
python hash_numba_delta.py one SLUG DELTA METHOD
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hash_numba as H  # noqa: E402  (sets the e1/g1 paths)

OUT = os.path.join(HERE, "hash_numba_delta.jsonl")
LIMIT = 900
SLUGS = ("creditcard", "miniboone", "jannis", "vehicle-sensit", "gas-drift")
DELTAS = ("0.05", "0.15", "0.20")
METHODS = ("KD-PA+B3", "HASH-NB")


def ref_reduct(slug, param):
    for f in ("e1_raw.jsonl", "remeasure_idle.jsonl"):
        for ln in open(os.path.join(H.E1, f)):
            r = json.loads(ln)
            if (r["dataset"], r["criterion"], r["param"], r["fold"], r["method"], r["status"]) == \
                    (slug, "NRS", param, 1, "KD-PA+B3", "ok"):
                return r["R"]
    return None


def one(slug, param, method):
    delta = float(param)
    X, y = H.data(slug)
    rec = dict(dataset=slug, criterion="NRS", param=param, fold=1, method=method, n_train=int(len(y)),
               d=int(X.shape[1]))
    if method == "HASH-NB":
        E, HashNB = H.hash_nb_class()
        t0 = time.perf_counter()
        i = np.sort(np.random.default_rng(0).choice(len(y), 200, replace=False))
        HashNB(X[i], y[i], delta).blocked([0], np.arange(200))
        rec["compile_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        R, evals, _ = E.greedy_nrs(HashNB(X, y, delta), False)
    else:
        import engines_e1 as E
        t0 = time.perf_counter()
        R, evals, _ = E.run_nrs(method, X, y, delta)
    rec.update(status="ok", time=time.perf_counter() - t0, R=[int(a) for a in R], evals=int(evals))
    ref = ref_reduct(slug, param)
    rec["same_as_e1"] = None if ref is None else rec["R"] == ref
    print(json.dumps(rec), flush=True)


def run():
    done = {}
    if os.path.exists(OUT):
        for r in map(json.loads, open(OUT)):
            done[(r["dataset"], r["param"], r["method"])] = r["status"]
    for slug in SLUGS:
        for param in DELTAS:
            for m in METHODS:
                if (slug, param, m) in done:
                    continue
                if m == "HASH-NB" and done.get((slug, param, "KD-PA+B3")) != "ok":
                    r = dict(dataset=slug, criterion="NRS", param=param, fold=1, method=m, status="not-run",
                             note="KD-PA+B3 did not finish (rule fixed in advance)")
                else:
                    t0 = time.time()
                    try:
                        p = subprocess.run([sys.executable, __file__, "one", slug, param, m], env=H.ENV,
                                           capture_output=True, text=True, timeout=LIMIT + 60)
                        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
                        r = json.loads(lines[-1]) if lines else dict(dataset=slug, param=param, method=m,
                                                                     status="error", note=p.stderr[-400:],
                                                                     time=time.time() - t0)
                        if r.get("time", 0) > LIMIT:
                            r["status"] = "timeout"
                    except subprocess.TimeoutExpired:
                        r = dict(dataset=slug, criterion="NRS", param=param, fold=1, method=m, status="timeout",
                                 time=time.time() - t0)
                with open(OUT, "a") as f:
                    f.write(json.dumps(r) + "\n")
                done[(slug, param, m)] = r["status"]
                print(f"{slug:16s} {param} {m:9s} {r['status']:8s} {r.get('time', 0):8.1f} s "
                      f"same={r.get('same_as_e1')}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    {"run": run, "one": lambda: one(a[1], a[2], a[3])}[a[0]]()
