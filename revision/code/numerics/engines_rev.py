"""Revised NRS engines with uniform boundary handling (revision, R2.2 / R1.9). The locked e1/g1 code is not modified;
the classes below subclass it and change only how a pair is decided.

Rule: a pair (x, y) is within delta on B iff d2seq(x, y; B) <= delta * delta, where d2seq adds the squared coordinate
differences sequentially in the order of the list B. Every engine receives the same list B (R + [a]), so every engine
performs the same floating-point operations on every pair it decides, and the engines agree by construction.
In the e1 code three places differed from this rule:
  (1) KD confirmed a tree hit with numpy .sum(1), which uses pairwise summation when |B| >= 8;
  (2) KD computed the nearest-enemy distance for the bound with .sum(1) (only used inside the slack of the bound);
  (3) the first step of FNC counted enemies in one-dimensional windows widened by WIN = 1e-9 with no explicit test.
The other engines (BF-PA, CABIN, BUCKET, HASH, FNC after step 1) already followed the rule (any_within / stored D2).
"""
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

E1 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "e1"))
sys.path.insert(0, E1); sys.path.insert(0, os.path.join(E1, "..", "g1"))
import engines_e1 as E   # noqa: E402
import methods as g1     # noqa: E402


def d2seq(X, q, e, B):
    out = np.zeros(len(q))
    for c in B:
        diff = X[q, c] - X[e, c]
        out += diff * diff
    return out


class KD(E.KD):
    def blocked(self, B, act):
        Z = np.ascontiguousarray(self.X[:, B]); out = np.zeros(len(act), bool); ya = self.y[act]
        for k in self.classes:
            idx = np.flatnonzero(self.y == k)
            qs = np.flatnonzero((ya != k) & ~out)
            if not len(qs) or not len(idx):
                continue
            d, j = cKDTree(Z[idx]).query(Z[act[qs]], k=1, distance_upper_bound=self.delta * (1 + 1e-9))
            f = np.isfinite(d)
            if f.any():
                q, e = act[qs[f]], idx[j[f]]
                ok = d2seq(self.X, q, e, B) <= self.d2
                out[qs[f][ok]] = True
        return out

    def nearest_enemy(self, B, act):
        ne2, arg = super().nearest_enemy(B, act)
        return d2seq(self.X, act, arg, B), arg


def fnc_step1_positive(X, y, delta, a):
    """Objects with no enemy within delta on attribute a: windows widened by WIN, then shrunk to the pairs that pass the
    explicit test (v_j - v_i)^2 <= delta^2. In float, fl(v_j - v_i) is monotone in v_j, so the passing set is a
    contiguous run inside the widened window and two monotone scans give it exactly."""
    n = len(y); d2 = delta * delta; classes = np.unique(y)
    onehot = (y[:, None] == classes[None, :]).astype(np.int64)
    o = np.argsort(X[:, a], kind="stable"); v = X[o, a]; i = np.arange(n)
    P = np.vstack([np.zeros((1, len(classes)), np.int64), np.cumsum(onehot[o], 0)])
    lo = np.searchsorted(v, v - delta - g1.WIN, "left"); hi = np.searchsorted(v, v + delta + g1.WIN, "right")
    while True:
        m = (lo < i) & ((v[np.minimum(lo, n - 1)] - v) ** 2 > d2)
        if not m.any():
            break
        lo[m] += 1
    while True:
        m = (hi > i + 1) & ((v[np.maximum(hi - 1, 0)] - v) ** 2 > d2)
        if not m.any():
            break
        hi[m] -= 1
    own = np.searchsorted(classes, y[o])
    enemies = (hi - lo) - (P[hi, own] - P[lo, own])
    pos = np.zeros(n, bool); pos[o] = enemies == 0
    return pos


def fnc_rev(X, y, delta, bound):
    """e1 fnc_e1 with step 1 replaced by fnc_step1_positive (explicit test). Steps >= 2 are those of fnc_e1:
    stored pairs with D2 accumulated sequentially in the order R + [a], i.e. d2seq."""
    n, d = X.shape; d2 = delta * delta; classes = np.unique(y); evals = 0; steps = []
    t0 = time.perf_counter(); best_gain, a1 = 0, None
    for a in range(d):
        pos = fnc_step1_positive(X, y, delta, a); g = int(pos.sum()); evals += 1
        if g > best_gain:
            best_gain, a1, pos1 = g, a, pos
    steps.append(dict(active=n, candidates=d, evaluated=d, pruned=0, selected=a1, gain=best_gain, ub_selected=None,
                      time=time.perf_counter() - t0))
    if a1 is None:
        return [], evals, steps
    # continue exactly as fnc_e1 from the stored pairs of attribute a1 (code shared through a small shim)
    R, ev2, st2 = _fnc_from(X, y, delta, bound, a1, pos1)
    return R, evals + ev2, steps + st2


def _fnc_from(X, y, delta, bound, a1, pos1):
    n, d = X.shape; d2 = delta * delta; classes = np.unique(y); evals = 0; steps = []
    R = [a1]; rem = [a for a in range(d) if a != a1]
    act = np.flatnonzero(~pos1)
    Q, E_, D2 = [], [], []; tot = 0
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
            Q.append(qq[m]); E_.append(ee[m]); D2.append(dd[m])
    Q, E_, D2 = np.concatenate(Q), np.concatenate(E_), np.concatenate(D2)
    act = np.unique(Q)
    while rem and len(act):
        t0 = time.perf_counter(); ub = None
        if bound:
            best = np.full(n, np.inf); np.minimum.at(best, Q, D2)
            hit = np.flatnonzero(D2 == best[Q])
            qa, fi = np.unique(Q[hit], return_index=True)
            wa, nd = E_[hit[fi]], D2[hit[fi]]
            ub = {a: int((nd + (X[qa, a] - X[wa, a]) ** 2 > d2 - E.EPS).sum()) for a in rem}
            order = sorted(rem, key=lambda a: (-ub[a], a))
        else:
            order = rem
        best_gain, best, ne_ = 0, None, 0
        for a in order:
            if ub is not None and ub[a] < max(best_gain, 1):
                break
            bad = D2 + (X[Q, a] - X[E_, a]) ** 2 <= d2; evals += 1; ne_ += 1
            g = int(len(act) - len(np.unique(Q[bad])))
            if g > best_gain or (g == best_gain and g > 0 and a < best):
                best_gain, best = g, a
        steps.append(dict(active=int(len(act)), candidates=len(rem), evaluated=ne_, pruned=len(rem) - ne_,
                          selected=best, gain=best_gain, ub_selected=None if ub is None or best is None else ub[best],
                          time=time.perf_counter() - t0))
        if best_gain == 0:
            break
        R.append(best); rem.remove(best)
        D2 = D2 + (X[Q, best] - X[E_, best]) ** 2
        m = D2 <= d2
        Q, E_, D2 = Q[m], E_[m], D2[m]
        act = np.unique(Q)
    return R, evals, steps


def make(name, X, y, delta):
    if name == "KD-PA":
        return KD(X, y, delta)
    return E.make(name, X, y, delta)


def run_nrs(method, X, y, delta):
    base, _, b3 = method.partition("+")
    if base == "FNC":
        return fnc_rev(X, y, delta, b3 == "B3")
    return E.greedy_nrs(make(base, X, y, delta), b3 == "B3")
