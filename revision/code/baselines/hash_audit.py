"""Revision run for R2.3 / R1.15: checks of the reimplemented NRS accelerators (fixed 2026-11-23, before any run).

(1) path  -- pair-count audit along the exact greedy path. NRS, delta = 0.10, fold 1 of the main runs, the 11 datasets
    with a finished KD-PA+B3 reduct (vehicle-sensit has none). For step t = 1..|R| of the e1 reduct R, B_t = R[:t] and
    act_t = objects not yet in the positive region after R[:t-1] (computed with the released k-d tree engine, not
    timed). Each engine evaluates the blocked mask on (B_t, act_t) once. Recorded per step: pairs passed to the exact
    distance test (any_within input), the naive count sum_k |act_t of class k| * |objects of other classes|, time,
    and agreement of the mask with the k-d tree mask. Engines: HASH-G (attribute superiority = Gamma_j, Eq. B4, as in
    the submitted runs), HASH-B7 (S_j = max_i n_ij / N, Eq. B7, the rule the paper adopts), CABIN, BUCKET, BF-PA,
    KD-PA. One subprocess per (dataset, engine), 900 s cap.
(2) greedy -- full exact greedy reduction with HASH-B7 (no bound), same 12 NRS datasets, delta = 0.10, fold 1,
    900 s cap; compared with HASH-G of the main runs (e1_raw.jsonl, fold 1) and with the e1 reduct.
One unit at a time, nothing else running.

python hash_audit.py path | greedy
python hash_audit.py one-path SLUG ENGINE ; python hash_audit.py one-greedy SLUG
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
E1 = os.path.join(ROOT, "e1"); NUM = os.path.join(HERE, "..", "numerics")
sys.path.insert(0, E1); sys.path.insert(0, NUM)
DELTA = 0.10; LIMIT = 900
SLUGS = ("mammography", "htru2", "magic-telescope", "house-16h", "gas-drift", "internet-firewall", "jannis",
         "run-or-walk", "miniboone", "creditcard", "hls4ml-jets", "vehicle-sensit")
ENGINES = ("KD-PA", "HASH-G", "HASH-B7", "CABIN", "BUCKET", "BF-PA")
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
           PYTHONWARNINGS="ignore")


def e1_reduct(slug):
    for ln in open(os.path.join(E1, "e1_raw.jsonl")):
        r = json.loads(ln)
        if (r["dataset"], r["criterion"], r["param"], r["fold"], r["method"]) == (slug, "NRS", "0.10", 1, "KD-PA+B3") \
                and r["status"] == "ok":
            return r["R"]
    return None


def data(slug):
    from data_e1 import load
    from run_e1 import folds
    X, y = load(slug); tr = folds(y)[0][0]
    return X[tr], y[tr]


def counting(cls):
    class C(cls):
        pairs = 0

        def any_within(self, B, q, e):
            self.pairs += len(q)
            return super().any_within(B, q, e)
    return C


def build(name, X, y):
    import engines_e1 as E
    import engines_rev as V
    if name == "KD-PA":
        return V.KD(X, y, DELTA)
    if name in ("HASH-G", "HASH-B7"):
        eng = counting(E.Hash)(X, y, DELTA)
        if name == "HASH-B7":                       # Eq. (B7): S_j = max_i n_ij / N, smaller is better
            n = len(y)
            eng.sup = np.array([np.bincount(eng.cells[:, j]).max() / n for j in range(X.shape[1])])
        return eng
    cls = {"CABIN": (E.Cabin, 2), "BUCKET": (E.Cabin, 1), "BF-PA": (E.BF, None)}[name]
    return counting(cls[0])(X, y, DELTA, cls[1]) if cls[1] else counting(cls[0])(X, y, DELTA)


def one_path(slug, name):
    import engines_rev as V
    X, y = data(slug); R = e1_reduct(slug)
    ref = V.KD(X, y, DELTA); act = np.arange(len(y)); path = []
    for t in range(len(R)):                                   # (B_t, act_t, reference mask), not timed
        m = ref.blocked(R[:t + 1], act); path.append((R[:t + 1], act, m)); act = act[m]
    eng = build(name, X, y)
    for t, (B, act, mref) in enumerate(path, 1):
        p0 = getattr(eng, "pairs", 0); t0 = time.perf_counter()
        m = eng.blocked(B, act); dt = time.perf_counter() - t0
        naive = int(sum((y[act] == k).sum() * (y != k).sum() for k in np.unique(y)))
        rec = dict(dataset=slug, engine=name, step=t, B=len(B), active=int(len(act)), time=dt,
                   pairs=None if name == "KD-PA" else int(eng.pairs - p0), naive=naive, same=bool((m == mref).all()))
        if name.startswith("HASH"):
            K = min(eng.Kmax, len(B)); rec["K"] = K
            rec["hash_attrs"] = [int(a) for a in sorted(B, key=lambda j: (eng.sup[j], j))[:K]]
        print(json.dumps(rec), flush=True)


def one_greedy(slug):
    import engines_e1 as E
    X, y = data(slug)
    eng = build("HASH-B7", X, y)
    t0 = time.perf_counter()
    R, evals, _ = E.greedy_nrs(eng, False)
    print(json.dumps(dict(dataset=slug, method="HASH-B7", n_train=int(len(y)), status="ok", R=[int(a) for a in R],
                          evals=int(evals), pairs=int(eng.pairs), time=time.perf_counter() - t0,
                          same_as_e1=[int(a) for a in R] == e1_reduct(slug))), flush=True)


def run(cmd, out, base):
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, __file__] + cmd, capture_output=True, text=True, timeout=LIMIT, env=ENV)
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
        if p.returncode != 0:
            lines.append(json.dumps(dict(base, status="error", err=p.stderr[-400:])))
    except subprocess.TimeoutExpired as ex:
        so = ex.stdout.decode() if isinstance(ex.stdout, bytes) else (ex.stdout or "")
        lines = [ln for ln in so.splitlines() if ln.startswith("{")]
        lines.append(json.dumps(dict(base, status="timeout", time=time.perf_counter() - t0)))
    with open(out, "a") as f:
        for ln in lines:
            f.write(ln + "\n")
    print(base, lines[-1][:160] if lines else None, flush=True)


def main(mode):
    out = os.path.join(HERE, f"hash_audit_{mode}.jsonl")
    seen = set()
    if os.path.exists(out):
        for ln in open(out):
            r = json.loads(ln); seen.add((r["dataset"], r.get("engine", r.get("method"))))
    for slug in SLUGS:
        if mode == "path":
            if e1_reduct(slug) is None:
                continue
            for name in ENGINES:
                if (slug, name) not in seen:
                    run(["one-path", slug, name], out, dict(dataset=slug, engine=name))
        elif (slug, "HASH-B7") not in seen:
            run(["one-greedy", slug], out, dict(dataset=slug, method="HASH-B7"))


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "one-path":
        one_path(sys.argv[2], sys.argv[3])
    elif a == "one-greedy":
        one_greedy(sys.argv[2])
    else:
        main(a)
