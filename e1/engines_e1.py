"""E1 NRS engines with the B3 bound on every engine and per-step instrumentation (e1_prereg.md).

The G1 engines (../g1/methods.py, locked) are reused unchanged for the neighbourhood tests. What is added here:
- witness(B, act): for every object of act (all of them have an enemy within delta on B, since act = not POS_B),
  one enemy within delta found by the engine itself. The B3 bound only needs *some* enemy y within delta:
  x can enter POS_{B+a} only if d_{B+a}(x, y) > delta, i.e. d_B(x,y)^2 + (x_a - y_a)^2 > delta^2 (theory/framework.md,
  valid for any enemy within delta; the nearest one gives the tightest test). Engines that enumerate all pairs
  within delta (BF, CABIN, BUCKET, FNC) return the nearest; HASH stops at the first hit per object; KD returns the
  nearest (k = 1 query).
- greedy_nrs(eng, bound): the G1 greedy, with per-step records.
- fnc_e1(X, y, delta, bound): the G1 FNC, with the bound (witness = stored pair of minimal D2) and per-step records.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "g1"))
import methods as g1  # noqa: E402

EPS = 1e-12


class Collect:
    """Mixin: record the (query, enemy, d2) pairs that survive the full delta test in any_within."""
    _col = None

    def any_within(self, B, q, e):
        d2 = np.zeros(len(q))
        for c in B:
            col = self.X[:, c]
            d2 += (col[q] - col[e]) ** 2
            m = d2 <= self.d2
            q, e, d2 = q[m], e[m], d2[m]
            if not len(q):
                break
        if self._col is not None and len(q):
            self._col.append((q, e, d2))
        return np.unique(q)

    def witness(self, B, act):
        self._col = []
        try:
            self.blocked(B, act)
            cols = self._col
        finally:
            self._col = None
        q = np.concatenate([c[0] for c in cols]); e = np.concatenate([c[1] for c in cols])
        d2 = np.concatenate([c[2] for c in cols])
        # O(P) (e1_prereg.md section 12; was lexsort, O(P log P)): minimal d2 per object, ties -> smallest enemy index
        n = len(self.X); best = np.full(n, np.inf); np.minimum.at(best, q, d2)
        hit = d2 == best[q]; w = np.full(n, n, dtype=np.int64); np.minimum.at(w, q[hit], e[hit])
        return w[act]


class BF(Collect, g1.BF):
    pass


class Cabin(Collect, g1.Cabin):
    pass


class Hash(Collect, g1.Hash):
    pass


class KD(g1.KD):
    def witness(self, B, act):
        return self.nearest_enemy(B, act)[1]


def make(name, X, y, delta):
    return {"BF-PA": lambda: BF(X, y, delta), "BUCKET": lambda: Cabin(X, y, delta, 1),
            "CABIN": lambda: Cabin(X, y, delta, 2), "HASH": lambda: Hash(X, y, delta),
            "KD-PA": lambda: KD(X, y, delta)}[name]()


def bound_ub(X, act, w, R, rem, d2):
    ne2 = ((X[act][:, R] - X[w][:, R]) ** 2).sum(1)
    return {a: int((ne2 + (X[act, a] - X[w, a]) ** 2 > d2 - EPS).sum()) for a in rem}


def greedy_nrs(eng, bound):
    X, n, d = eng.X, eng.X.shape[0], eng.X.shape[1]
    act = np.arange(n); R, rem, evals, steps = [], list(range(d)), 0, []
    while rem and len(act):
        t0 = time.perf_counter(); ub = None
        if bound and R:
            ub = bound_ub(X, act, eng.witness(R, act), R, rem, eng.d2)
            order = sorted(rem, key=lambda a: (-ub[a], a))
        else:
            order = rem
        best_gain, best, best_mask, ne_ = 0, None, None, 0
        for a in order:
            if ub is not None and ub[a] < max(best_gain, 1):
                break
            mask = eng.blocked(R + [a], act); evals += 1; ne_ += 1
            g = int(len(act) - mask.sum())
            if g > best_gain or (g == best_gain and g > 0 and a < best):
                best_gain, best, best_mask = g, a, mask
        steps.append(dict(active=int(len(act)), candidates=len(rem), evaluated=ne_, pruned=len(rem) - ne_,
                          selected=best, gain=best_gain, ub_selected=None if ub is None or best is None else ub[best],
                          time=time.perf_counter() - t0))
        if best_gain == 0:
            break
        R.append(best); rem.remove(best); act = act[best_mask]
    return R, evals, steps


def fnc_e1(X, y, delta, bound):
    """G1 fnc() + optional B3 bound + per-step records. Selection, tie-break (lowest index) and stop unchanged."""
    n, d = X.shape; d2 = delta * delta; classes = np.unique(y); evals = 0; steps = []
    onehot = (y[:, None] == classes[None, :]).astype(np.int64)
    t0 = time.perf_counter(); best_gain, a1 = 0, None
    for a in range(d):
        o = np.argsort(X[:, a], kind="stable"); v = X[o, a]
        P = np.vstack([np.zeros((1, len(classes)), np.int64), np.cumsum(onehot[o], 0)])
        lo = np.searchsorted(v, v - delta - g1.WIN, "left"); hi = np.searchsorted(v, v + delta + g1.WIN, "right")
        own = np.searchsorted(classes, y[o])
        enemies = (hi - lo) - (P[hi, own] - P[lo, own])
        pos = np.zeros(n, bool); pos[o] = enemies == 0
        g = int(pos.sum()); evals += 1
        if g > best_gain:
            best_gain, a1, pos1 = g, a, pos
    steps.append(dict(active=n, candidates=d, evaluated=d, pruned=0, selected=a1, gain=best_gain, ub_selected=None,
                      time=time.perf_counter() - t0))
    if a1 is None:
        return [], evals, steps
    R = [a1]; rem = [a for a in range(d) if a != a1]
    act = np.flatnonzero(~pos1)
    Q, E, D2 = [], [], []; tot = 0
    for k in classes:
        q = act[y[act] == k]; e = np.flatnonzero(y != k)
        e = e[np.argsort(X[e, a1], kind="stable")]; ve = X[e, a1]
        lo = np.searchsorted(ve, X[q, a1] - delta - g1.WIN, "left")
        hi = np.searchsorted(ve, X[q, a1] + delta + g1.WIN, "right")
        tot += int((hi - lo).sum())
        if tot > g1.PAIR_LIMIT:
            raise g1.MemOut(f"{tot} pairs")
        for i, j in g1.chunked(hi - lo):
            r, p = g1.expand(lo[i:j], hi[i:j])
            qq, ee = q[i:j][r], e[p]; dd = (X[qq, a1] - X[ee, a1]) ** 2
            m = dd <= d2
            Q.append(qq[m]); E.append(ee[m]); D2.append(dd[m])
    Q, E, D2 = np.concatenate(Q), np.concatenate(E), np.concatenate(D2)
    act = np.unique(Q)
    while rem and len(act):
        t0 = time.perf_counter(); ub = None
        if bound:
            # nearest stored pair per object in O(P) (fix 2026-09-29: the earlier lexsort was O(P log P) per step)
            best = np.full(n, np.inf); np.minimum.at(best, Q, D2)
            hit = np.flatnonzero(D2 == best[Q])
            qa, fi = np.unique(Q[hit], return_index=True)                            # qa == act (sorted unique)
            wa, nd = E[hit[fi]], D2[hit[fi]]
            ub = {a: int((nd + (X[qa, a] - X[wa, a]) ** 2 > d2 - EPS).sum()) for a in rem}
            order = sorted(rem, key=lambda a: (-ub[a], a))
        else:
            order = rem
        best_gain, best, ne_ = 0, None, 0
        for a in order:
            if ub is not None and ub[a] < max(best_gain, 1):
                break
            bad = D2 + (X[Q, a] - X[E, a]) ** 2 <= d2; evals += 1; ne_ += 1
            g = int(len(act) - len(np.unique(Q[bad])))
            if g > best_gain or (g == best_gain and g > 0 and a < best):
                best_gain, best = g, a
        steps.append(dict(active=int(len(act)), candidates=len(rem), evaluated=ne_, pruned=len(rem) - ne_,
                          selected=best, gain=best_gain, ub_selected=None if ub is None or best is None else ub[best],
                          time=time.perf_counter() - t0))
        if best_gain == 0:
            break
        R.append(best); rem.remove(best)
        D2 = D2 + (X[Q, best] - X[E, best]) ** 2
        m = D2 <= d2
        Q, E, D2 = Q[m], E[m], D2[m]
        act = np.unique(Q)
    return R, evals, steps


NRS_METHODS = ("KD-PA", "KD-PA+B3", "FNC", "FNC+B3", "HASH", "HASH+B3", "CABIN", "CABIN+B3",
               "BUCKET", "BUCKET+B3", "BF-PA", "BF-PA+B3", "FARA")


def run_nrs(method, X, y, delta):
    base, _, b3 = method.partition("+")
    if base == "FNC":
        return fnc_e1(X, y, delta, b3 == "B3")
    if base == "FARA":
        from fara import fara
        R, evals, log = fara(X, y, delta)
        return R, evals, [dict(fara_log=log)]
    return greedy_nrs(make(base, X, y, delta), b3 == "B3")
