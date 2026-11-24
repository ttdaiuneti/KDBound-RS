"""Revision run for R2.3: a compiled HASH (fixed 2026-11-24, before any timed run).

Question: how much of the speed-up of KD-PA and KD-PA+B3 over HASH comes from HASH being written in NumPy?
HASH-NB compiles the evaluation of HASH (the blocked mask of g1.Hash, as in the main runs: attribute superiority Gamma,
K = Kmax of Eq. (C8), cell codes of the K best attributes, 3^K neighbouring cells) with Numba (single thread). For each
active object it scans the enemies in the neighbouring cells and tests each pair with the same float64 rule as every
other engine (squared differences added in the order of B, early rejection once the sum exceeds delta^2), and stops at
the first enemy within delta. The NumPy HASH tests all pairs of a cell at once and drops blocked objects only between
cells, so HASH-NB examines at most as many pairs: the comparison favours the baseline. Cell codes and offsets are
computed exactly as in g1.Hash (NumPy, included in the timed evaluation). Before any timing the mask of HASH-NB is
compared with the mask of g1.Hash at every step of the greedy path (check mode).

Units: NRS, delta = 0.10, fold 1 of the main runs, the 12 NRS datasets; methods KD-PA, KD-PA+B3, HASH (NumPy, as in the
main runs) and HASH-NB (full exact greedy reduction, no bound); one unit at a time in its own process, one thread, 900 s
cap. The Numba kernel is compiled on a 200-object sample in the same process before the timer starts (compile time is
recorded separately). Output: hash_numba.jsonl (append; done units skipped).

python hash_numba.py check SLUG   -> masks of HASH-NB vs g1.Hash along the e1 greedy path of SLUG (not timed)
python hash_numba.py run          -> all units
python hash_numba.py one SLUG METHOD
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
E1 = os.path.join(ROOT, "e1"); sys.path.insert(0, E1); sys.path.insert(0, os.path.join(ROOT, "g1"))
OUT = os.path.join(HERE, "hash_numba.jsonl")
DELTA = 0.10; LIMIT = 900
SLUGS = ("mammography", "htru2", "magic-telescope", "house-16h", "creditcard", "internet-firewall", "run-or-walk",
         "hls4ml-jets", "gas-drift", "jannis", "miniboone", "vehicle-sensit")
METHODS = ("KD-PA", "KD-PA+B3", "HASH-NB", "HASH")
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
           NUMBA_NUM_THREADS="1", PYTHONWARNINGS="ignore")


def kernel():
    from numba import njit

    @njit(cache=False)
    def blocked_nb(X, cls, ncls, act, B, code, off, d2):
        out = np.zeros(len(act), np.bool_)
        n = X.shape[0]
        for k in range(ncls):
            # enemies of class k sorted by cell code (stable, as in g1.Hash)
            ne = 0
            for i in range(n):
                if cls[i] != k:
                    ne += 1
            E = np.empty(ne, np.int64); j = 0
            for i in range(n):
                if cls[i] != k:
                    E[j] = i; j += 1
            E = E[np.argsort(code[E], kind="mergesort")]
            cE = code[E]
            for t in range(len(act)):
                q = act[t]
                if cls[q] != k:
                    continue
                cq = code[q]; found = False
                for o in range(len(off)):
                    tgt = cq + off[o]
                    lo = np.searchsorted(cE, tgt, side="left")
                    hi = np.searchsorted(cE, tgt, side="right")
                    for p in range(lo, hi):
                        e = E[p]; s = 0.0; ok = True
                        for c in B:
                            diff = X[q, c] - X[e, c]
                            s += diff * diff
                            if s > d2:
                                ok = False
                                break
                        if ok:
                            found = True
                            break
                    if found:
                        break
                out[t] = found
        return out
    return blocked_nb


def hash_nb_class():
    import engines_e1 as E
    import methods as g1
    blocked_nb = kernel()

    class HashNB(g1.Hash):
        name = "HASH-NB"

        def __init__(self, X, y, delta):
            super().__init__(np.ascontiguousarray(X), y, delta)
            self.cls = np.searchsorted(self.classes, y).astype(np.int64)

        def blocked(self, B, act):
            K = min(self.Kmax, len(B))
            attrs = sorted(B, key=lambda j: (self.sup[j], j))[:K]
            w = self.base ** np.arange(K, dtype=np.int64)
            code = ((self.cells[:, attrs] + 1) * w).sum(1)
            offs = np.array(np.meshgrid(*[[-1, 0, 1]] * K, indexing="ij")).reshape(K, -1).T
            off_codes = (offs * w).sum(1).astype(np.int64)
            return blocked_nb(self.X, self.cls, len(self.classes), np.asarray(act, np.int64),
                              np.asarray(B, np.int64), code.astype(np.int64), off_codes, self.d2)
    return E, HashNB


def data(slug):
    from data_e1 import load
    from run_e1 import folds
    X, y = load(slug); tr = folds(y)[0][0]
    return X[tr], y[tr]


def e1_reduct(slug):
    for ln in open(os.path.join(E1, "e1_raw.jsonl")):
        r = json.loads(ln)
        if (r["dataset"], r["criterion"], r["param"], r["fold"], r["method"], r["status"]) == \
                (slug, "NRS", "0.10", 1, "KD-PA+B3", "ok"):
            R = r["R"]
    return R


def check(slug):
    """Mask of HASH-NB == mask of g1.Hash on (R[:t] + [a], act_t) for every step t and every candidate a."""
    import methods as g1
    E, HashNB = hash_nb_class()
    X, y = data(slug)
    R = e1_reduct(slug)
    h, nb = g1.Hash(X, y, DELTA), HashNB(X, y, DELTA)
    act = np.arange(len(y)); bad = tot = 0
    for t in range(len(R)):
        for a in [a for a in range(X.shape[1]) if a not in R[:t]]:
            m1 = h.blocked(R[:t] + [a], act); m2 = nb.blocked(R[:t] + [a], act)
            tot += 1; bad += int(not np.array_equal(m1, m2))
        act = act[h.blocked(R[:t + 1], act)]
    print(json.dumps(dict(dataset=slug, masks=tot, different=bad, reduct_len=len(R))), flush=True)


def one(slug, method):
    X, y = data(slug)
    rec = dict(dataset=slug, criterion="NRS", param="0.10", fold=1, method=method, n_train=int(len(y)),
               d=int(X.shape[1]))
    if method == "HASH-NB":
        E, HashNB = hash_nb_class()
        t0 = time.perf_counter()
        i = np.sort(np.random.default_rng(0).choice(len(y), 200, replace=False))
        HashNB(X[i], y[i], DELTA).blocked([0], np.arange(200))
        rec["compile_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        R, evals, _ = E.greedy_nrs(HashNB(X, y, DELTA), False)
    else:
        import engines_e1 as E
        t0 = time.perf_counter()
        R, evals, _ = E.run_nrs(method, X, y, DELTA)
    rec.update(status="ok", time=time.perf_counter() - t0, R=[int(a) for a in R], evals=int(evals))
    rec["same_as_e1"] = rec["R"] == e1_reduct(slug)
    print(json.dumps(rec), flush=True)


# Not re-run (decided 2026-11-24 during the run, to save time): units that already exceeded 900 s one at a time --
# HASH on these four tables (main runs and hash_audit greedy) and KD-PA on vehicle-sensit (e1/remeasure_idle.jsonl).
KNOWN_TIMEOUT = {("gas-drift", "HASH"), ("jannis", "HASH"), ("miniboone", "HASH"), ("vehicle-sensit", "HASH"),
                 ("vehicle-sensit", "KD-PA")}


def run():
    done = {(r["dataset"], r["method"]) for r in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()
    for slug in SLUGS:
        for m in METHODS:
            if (slug, m) in done:
                continue
            if (slug, m) in KNOWN_TIMEOUT:
                r = dict(dataset=slug, criterion="NRS", param="0.10", fold=1, method=m, status="timeout",
                         note="not re-run: exceeded 900 s in earlier runs (see KNOWN_TIMEOUT)")
                with open(OUT, "a") as f:
                    f.write(json.dumps(r) + "\n")
                continue
            t0 = time.time()
            try:
                p = subprocess.run([sys.executable, __file__, "one", slug, m], env=ENV, capture_output=True,
                                   text=True, timeout=LIMIT + 60)
                lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
                r = json.loads(lines[-1]) if lines else dict(dataset=slug, method=m, status="error",
                                                             note=p.stderr[-400:], time=time.time() - t0)
                if r.get("time", 0) > LIMIT:
                    r["status"] = "timeout"
            except subprocess.TimeoutExpired:
                r = dict(dataset=slug, criterion="NRS", param="0.10", fold=1, method=m, status="timeout",
                         time=time.time() - t0)
            with open(OUT, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"{slug:18s} {m:9s} {r['status']:8s} {r.get('time', 0):8.1f} s same={r.get('same_as_e1')}",
                  flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    {"check": lambda: check(a[1]), "run": run, "one": lambda: one(a[1], a[2])}[a[0]]()
