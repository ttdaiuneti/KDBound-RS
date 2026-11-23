"""Revision run for R2.3: how much of the speed-up over the existing accelerators is work, and how much is the cost of
one unit of work (implementation). Fixed 2026-11-23 before any run.

Unit of work: object pairs examined in one evaluation of the criterion, i.e. pairs for which the engine computes a
distance or applies a test individually.
  KD-PA   objects in the visited leaves of the per-class k-d trees, plus the confirming test of each hit. The tree is
          rebuilt here with SciPy's default rule (median split on the dimension of largest spread, leaves of at most 16
          objects). A best-first search with radius r visits exactly the nodes whose region lies within r of the query
          (r = distance to the nearest object of that class, capped by the search bound). Two counts: regions = cells
          bounded by the split planes ("cell", an upper count) and regions = bounding boxes of the node's objects
          ("box", SciPy's compact nodes). The nearest distance is taken from SciPy's own query.
  HASH, BUCKET  pairs passed to the exact test (= pairs in the searched cells / windows; from the pair-count audit).
  CABIN   pairs in the window of the first control sample (each is tested against the second control).
  BF-PA, FA-FPR  |act| x n (every pair).
  FNC     stored pairs updated by one evaluation (one coordinate per pair); step 1 uses sorted windows (0 pairs).
Time per evaluation: NRS engines from the pair-count audit (code/baselines/hash_audit_path.jsonl), except FNC (timed
here, on the code of e1/engines_e1.py::fnc_e1, copied, not modified). FRS: KD-PA timed here; FA-FPR timed on the
first ROWS_FA active objects and scaled by |act| / ROWS_FA (its cost is linear in the rows: blocks of 256 rows).
Path: the e1 reduct of fold 1 (NRS delta = 0.10; FRS sigma = 0.25), B_t = R[:t], act_t as in the greedy with positive
approximation; one evaluation of B_t per step and engine.
One subprocess per (dataset, unit), 900 s cap, one at a time, nothing else running.

python work_count.py all ; python work_count.py one SLUG UNIT
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
E1 = os.path.join(ROOT, "e1"); NUM = os.path.join(HERE, "..", "numerics"); BAS = os.path.join(HERE, "..", "baselines")
sys.path.insert(0, E1); sys.path.insert(0, NUM); sys.path.insert(0, BAS); sys.path.insert(0, os.path.join(ROOT, "g1"))
DELTA = 0.10; SIGMA = 0.25; LIMIT = 900; LEAF = 16; ROWS_FA = 2048
OUT = os.path.join(HERE, "work_count.jsonl")
NRS_SLUGS = ("mammography", "htru2", "magic-telescope", "house-16h", "gas-drift", "internet-firewall", "jannis",
             "run-or-walk", "miniboone", "creditcard", "hls4ml-jets")
UNITS_NRS = ("KD-count", "CABIN-win", "FNC")
UNITS_FRS = ("FRS-KD", "FRS-FA")
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
           PYTHONWARNINGS="ignore")


# ----------------------------------------------------------------------------------------------- counting k-d tree
class Tree:
    def __init__(self, Z):
        n, d = Z.shape
        perm = np.arange(n)
        lo, hi, blo, bhi, left, right, start, end = [], [], [], [], [], [], [], []

        def new(l, h, s, e):
            P = Z[perm[s:e]]
            lo.append(l); hi.append(h); blo.append(P.min(0)); bhi.append(P.max(0))
            left.append(-1); right.append(-1); start.append(s); end.append(e)
            return len(start) - 1

        stack = [new(Z.min(0), Z.max(0), 0, n)]
        while stack:
            i = stack.pop(); s, e = start[i], end[i]
            if e - s <= LEAF:
                continue
            spread = bhi[i] - blo[i]; sd = int(np.argmax(spread))
            if spread[sd] == 0:
                continue
            m = (e - s) // 2
            seg = perm[s:e]; o = np.argpartition(Z[seg, sd], m); perm[s:e] = seg[o]
            split = Z[perm[s + m], sd]
            lh = hi[i].copy(); lh[sd] = split; rl = lo[i].copy(); rl[sd] = split
            a = new(lo[i], lh, s, s + m); b = new(rl, hi[i], s + m, e)
            left[i], right[i] = a, b; stack += [a, b]
        self.lo, self.hi, self.blo, self.bhi = map(np.array, (lo, hi, blo, bhi))
        self.left, self.right = np.array(left), np.array(right)
        self.size = np.array(end) - np.array(start); self.start = np.array(start); self.perm = perm

    def count(self, Zq, r, p, box):
        """Objects in visited leaves and visited nodes, per query, for radius r (p = 2 or inf)."""
        L, H = (self.blo, self.bhi) if box else (self.lo, self.hi)
        nq, d = Zq.shape; pts = np.zeros(nq, np.int64); nod = np.zeros(nq, np.int64)
        rr = (r * r if p == 2 else r) * (1 + 1e-12) + 1e-300
        C = max(64, int(4_000_000 / max(d, 1) / 64))
        for s in range(0, nq, C):
            qi = np.arange(s, min(nq, s + C)); nd = np.zeros(len(qi), np.int64)
            while len(qi):
                x = Zq[qi]; side = np.maximum(0.0, np.maximum(L[nd] - x, x - H[nd]))
                md = (side * side).sum(1) if p == 2 else side.max(1)
                k = md <= rr[qi]; qi, nd = qi[k], nd[k]
                nod += np.bincount(qi, minlength=nq)
                leaf = self.left[nd] < 0
                pts += np.bincount(qi[leaf], weights=self.size[nd[leaf]], minlength=nq).astype(np.int64)
                qi, nd = qi[~leaf], nd[~leaf]
                qi = np.concatenate([qi, qi]); nd = np.concatenate([self.left[nd], self.right[nd]])
        return pts, nod


# ----------------------------------------------------------------------------------------------- data and paths
def e1_reduct(slug, crit):
    key = (slug, crit, "0.10" if crit == "NRS" else "0.25", 1, "KD-PA+B3")
    for ln in open(os.path.join(E1, "e1_raw.jsonl")):
        r = json.loads(ln)
        if (r["dataset"], r["criterion"], r["param"], r["fold"], r["method"]) == key and r["status"] == "ok":
            return r["R"]
    return None


def data(slug):
    from data_e1 import load
    from run_e1 import folds
    X, y = load(slug); tr = folds(y)[0][0]
    return X[tr], y[tr]


def nrs_path(X, y, R):
    import engines_rev as V
    ref = V.KD(X, y, DELTA); act = np.arange(len(y)); path = []
    for t in range(len(R)):
        m = ref.blocked(R[:t + 1], act); path.append((R[:t + 1], act)); act = act[m]
    return path


def frs_path(Z, y, R):
    import frs
    eng = frs.KD(Z, y); act = np.arange(len(y)); path = []
    for t in range(len(R)):
        path.append((R[:t + 1], act))
        ne, _ = eng.ne(R[:t + 1], act); act = act[np.minimum(1.0, ne) < 1.0]
    return path


def emit(**k):
    print(json.dumps(k), flush=True)


# ----------------------------------------------------------------------------------------------- units
def kd_count(slug):
    X, y = data(slug); R = e1_reduct(slug, "NRS"); bound = DELTA * (1 + 1e-9)
    for t, (B, act) in enumerate(nrs_path(X, y, R), 1):
        Z = np.ascontiguousarray(X[:, B]); ya = y[act]; out = np.zeros(len(act), bool)
        rec = dict(cell=0, box=0, nodes_cell=0, nodes_box=0, confirm=0)
        for k in np.unique(y):                                   # mirrors engines_rev.KD.blocked
            idx = np.flatnonzero(y == k); qs = np.flatnonzero((ya != k) & ~out)
            if not len(qs) or not len(idx):
                continue
            d, j = cKDTree(Z[idx]).query(Z[act[qs]], k=1, distance_upper_bound=bound)
            r = np.minimum(d, bound); T = Tree(Z[idx])
            for box in (False, True):
                p_, n_ = T.count(Z[act[qs]], r, 2, box)
                rec["box" if box else "cell"] += int(p_.sum()); rec["nodes_box" if box else "nodes_cell"] += int(n_.sum())
            f = np.isfinite(d); rec["confirm"] += int(f.sum())
            if f.any():
                q, e = act[qs[f]], idx[j[f]]
                ok = ((X[q][:, B] - X[e][:, B]) ** 2).sum(1) <= DELTA * DELTA
                out[qs[f][ok]] = True
        emit(dataset=slug, unit="KD-count", crit="NRS", step=t, B=len(B), active=int(len(act)), n=int(len(y)), **rec)


def cabin_win(slug):
    """Pairs in the window of the first control sample (= BUCKET's window), per class, as g1.Cabin.blocked."""
    X, y = data(slug); R = e1_reduct(slug, "NRS"); cmin = X.min(0)
    for t, (B, act) in enumerate(nrs_path(X, y, R), 1):
        D1 = np.sqrt(((X[:, B] - cmin[B]) ** 2).sum(1)); tot = 0
        for k in np.unique(y):
            Q = act[y[act] == k]
            if not len(Q):
                continue
            E = np.flatnonzero(y != k); v = np.sort(D1[E])
            lo = np.searchsorted(v, D1[Q] - DELTA - 1e-9, "left"); hi = np.searchsorted(v, D1[Q] + DELTA + 1e-9, "right")
            tot += int((hi - lo).sum())
        emit(dataset=slug, unit="CABIN-win", crit="NRS", step=t, B=len(B), active=int(len(act)), window=tot)


def fnc(slug):
    """fnc_e1 (no bound) along the path: per step, one evaluation of B_t = R[:t], timed; pairs updated."""
    import methods as g1
    X, y = data(slug); R = e1_reduct(slug, "NRS"); n, d = X.shape; d2 = DELTA * DELTA; classes = np.unique(y)
    onehot = (y[:, None] == classes[None, :]).astype(np.int64); a1 = R[0]
    t0 = time.perf_counter()
    o = np.argsort(X[:, a1], kind="stable"); v = X[o, a1]
    P = np.vstack([np.zeros((1, len(classes)), np.int64), np.cumsum(onehot[o], 0)])
    lo = np.searchsorted(v, v - DELTA - g1.WIN, "left"); hi = np.searchsorted(v, v + DELTA + g1.WIN, "right")
    own = np.searchsorted(classes, y[o]); enemies = (hi - lo) - (P[hi, own] - P[lo, own])
    pos1 = np.zeros(n, bool); pos1[o] = enemies == 0
    emit(dataset=slug, unit="FNC", crit="NRS", step=1, B=1, active=n, pairs=0, time=time.perf_counter() - t0)
    act = np.flatnonzero(~pos1); Q, E, D2 = [], [], []; tot = 0; ts = time.perf_counter()
    for k in classes:                                           # pair construction on a1 (once per run)
        q = act[y[act] == k]; e = np.flatnonzero(y != k)
        e = e[np.argsort(X[e, a1], kind="stable")]; ve = X[e, a1]
        lo = np.searchsorted(ve, X[q, a1] - DELTA - g1.WIN, "left"); hi = np.searchsorted(ve, X[q, a1] + DELTA + g1.WIN, "right")
        tot += int((hi - lo).sum())
        if tot > g1.PAIR_LIMIT:
            emit(dataset=slug, unit="FNC", crit="NRS", status="memout", pairs_a1=tot); return
        for i, j in g1.chunked(hi - lo):
            r, p = g1.expand(lo[i:j], hi[i:j])
            qq, ee = q[i:j][r], e[p]; dd = (X[qq, a1] - X[ee, a1]) ** 2; m = dd <= d2
            Q.append(qq[m]); E.append(ee[m]); D2.append(dd[m])
    Q, E, D2 = np.concatenate(Q), np.concatenate(E), np.concatenate(D2)
    emit(dataset=slug, unit="FNC", crit="NRS", step=0, setup_pairs=tot, setup_time=time.perf_counter() - ts)
    for t in range(2, len(R) + 1):
        a = R[t - 1]; nact = len(np.unique(Q))
        t0 = time.perf_counter()
        bad = D2 + (X[Q, a] - X[E, a]) ** 2 <= d2
        g = int(nact - len(np.unique(Q[bad])))
        dt = time.perf_counter() - t0
        emit(dataset=slug, unit="FNC", crit="NRS", step=t, B=t, active=int(nact), pairs=int(len(Q)), time=dt, gain=g)
        D2 = D2 + (X[Q, a] - X[E, a]) ** 2; m = D2 <= d2; Q, E, D2 = Q[m], E[m], D2[m]


def frs_unit(slug, unit):
    import frs
    X, y = data(slug); R = e1_reduct(slug, "FRS")
    if R is None:
        emit(dataset=slug, unit=unit, crit="FRS", status="no-reduct"); return
    Z = frs.scale(X, SIGMA); n = len(y)
    for t, (B, act) in enumerate(frs_path(Z, y, R), 1):
        if unit == "FRS-KD":
            eng = frs.KD(Z, y); t0 = time.perf_counter(); eng.ne(B, act); dt = time.perf_counter() - t0
            ZB = np.ascontiguousarray(Z[:, B]); ya = y[act]; rec = dict(cell=0, box=0, nodes_cell=0, nodes_box=0, confirm=0)
            for c in np.unique(y):                                  # mirrors frs.KD.ne
                idx = np.flatnonzero(y == c); qs = np.flatnonzero(ya != c)
                if not len(qs) or not len(idx):
                    continue
                d, _ = cKDTree(ZB[idx]).query(ZB[act[qs]], k=1, p=np.inf, distance_upper_bound=1.0 + 1e-9)
                r = np.minimum(d, 1.0 + 1e-9); T = Tree(ZB[idx]); rec["confirm"] += int(np.isfinite(d).sum())
                for box in (False, True):
                    p_, n_ = T.count(ZB[act[qs]], r, np.inf, box)
                    rec["box" if box else "cell"] += int(p_.sum()); rec["nodes_box" if box else "nodes_cell"] += int(n_.sum())
            emit(dataset=slug, unit=unit, crit="FRS", step=t, B=len(B), active=int(len(act)), n=n, time=dt, **rec)
        else:
            eng = frs.BF(Z, y); S = min(len(act), ROWS_FA)
            t0 = time.perf_counter(); eng.ne(B, act[:S]); dt = time.perf_counter() - t0
            emit(dataset=slug, unit=unit, crit="FRS", step=t, B=len(B), active=int(len(act)), n=n, rows_timed=S,
                 time=dt * len(act) / S, pairs=int(len(act)) * n)


# ----------------------------------------------------------------------------------------------- driver
def run(cmd, base):
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
    with open(OUT, "a") as f:
        for ln in lines:
            f.write(ln + "\n")
    print(base, lines[-1][:160] if lines else None, flush=True)


def main():
    seen = set()
    if os.path.exists(OUT):
        for ln in open(OUT):
            r = json.loads(ln); seen.add((r["dataset"], r["unit"]))
    for slug in NRS_SLUGS:
        for u in UNITS_NRS:
            if (slug, u) not in seen:
                run(["one", slug, u], dict(dataset=slug, unit=u))
    for slug in FRS_SLUGS():
        for u in UNITS_FRS:
            if (slug, u) not in seen:
                run(["one", slug, u], dict(dataset=slug, unit=u))


def FRS_SLUGS():
    out = []
    for ln in open(os.path.join(E1, "e1_raw.jsonl")):
        r = json.loads(ln)
        if r["criterion"] == "FRS" and r["param"] == "0.25" and r["fold"] == 1 and r["method"] == "KD-PA+B3" \
                and r["status"] == "ok" and r["dataset"] not in out:
            out.append(r["dataset"])
    return out


if __name__ == "__main__":
    if sys.argv[1] == "one":
        slug, u = sys.argv[2], sys.argv[3]
        {"KD-count": kd_count, "CABIN-win": cabin_win, "FNC": fnc}.get(u, lambda s: frs_unit(s, u))(slug)
    else:
        main()
