"""Gaussian kernel fuzzy rough sets (Hu et al. 2010, IJAR 51:453-471) as an NE-determined criterion.

Model (Hu 2010, Sec. 3.2 and Algorithm 1 DGKA, verified against the full text 2026-11-21):
    R_B(x, y) = exp(-||x - y||_B^2 / (2 d^2))           (Euclidean distance on the attributes of B)
    lower approximation of the own class: inf over enemies y of sqrt(1 - R_B(x, y)^2)
    = f(NE_B(x)),  f(t) = sqrt(1 - exp(-t^2 / d^2))   (f increasing, f(+inf) = 1, never saturates at finite t)
    Q(B) = sum_x f(NE_B(x));  gamma_B = Q(B) / |U|.
Greedy (Hu 2010, Algorithm 2 FS-GKA): add argmax gain (ties -> smallest index), stop when gain <= eps * |U|
(eps is not given in Hu 2010; set here).

Engines: BF (brute force) and KD (per-class cKDTree, p = 2). Both return, for each query object, the squared distance
to a nearest enemy recomputed by d2(), which adds the squared coordinate differences in increasing attribute order,
so both engines produce the same values. Gains are summed with math.fsum.
Bound (Theorem 1 with p = 2): UB(a) = sum_x [f(d2_R(x, w_x) + (x_a - w_a)^2) - f(d2_R(x))] + SLACK * |U|,
with w_x a nearest enemy on R; SLACK absorbs the rounding difference between the two sums of squares.
"""
import math
import time

import numpy as np
from scipy.spatial import cKDTree

ROWS = 512
SLACK = 1e-12


def d2(X, q, e, B):
    """Squared Euclidean distance between rows q and e on the attributes of B, summed in increasing attribute order."""
    out = np.zeros(len(q))
    for a in sorted(B):
        diff = X[q, a] - X[e, a]
        out += diff * diff
    return out


def f_of(dsq, d):
    return np.sqrt(-np.expm1(-dsq / (d * d)))


class BF:
    def __init__(self, X, y):
        self.X, self.y = X, y

    def ne(self, B, q):
        """(squared NE distance, index of a nearest enemy) for the objects q on the attribute set B."""
        X, y = self.X, self.y; arg = np.empty(len(q), int); Bs = sorted(B)
        for s in range(0, len(q), ROWS):
            qq = q[s:s + ROWS]
            D = np.zeros((len(qq), len(y)))
            for a in Bs:
                diff = X[qq, a][:, None] - X[None, :, a]
                D += diff * diff
            D[y[qq][:, None] == y[None, :]] = np.inf
            arg[s:s + ROWS] = D.argmin(1)
        return d2(X, q, arg, B), arg


class KD(BF):
    def ne(self, B, q):
        X, y = self.X, self.y; best = np.full(len(q), np.inf); arg = np.full(len(q), -1); yq = y[q]
        XB = np.ascontiguousarray(X[:, sorted(B)])
        for c in np.unique(y):
            idx = np.flatnonzero(y == c); qs = np.flatnonzero(yq != c)
            if not len(qs) or not len(idx):
                continue
            _, j = cKDTree(XB[idx]).query(XB[q[qs]], k=1, p=2)
            e = idx[j]
            dd = d2(X, q[qs], e, B)                                  # same arithmetic as BF
            b = dd < best[qs]
            best[qs[b]] = dd[b]; arg[qs[b]] = e[b]
        return best, arg


def greedy(eng, d, eps, bound):
    X = eng.X; n, m = X.shape
    act = np.arange(n); R, rem, evals, steps = [], list(range(m)), 0, []
    tau = eps * n
    q_act = np.zeros(n)                                              # f(NE_R(x)); R empty -> NE = 0 -> f = 0
    while rem:
        t0 = time.perf_counter(); ub = None
        if bound and R:
            dsq, w = eng.ne(R, act)
            base = f_of(dsq, d)
            ub = {a: math.fsum(f_of(dsq + (X[act, a] - X[w, a]) ** 2, d) - base) + SLACK * n for a in rem}
            order = sorted(rem, key=lambda a: (-ub[a], a))
        else:
            order = rem
        t_wit = time.perf_counter() - t0
        best_gain, best, best_q, n_eval = 0.0, None, None, 0
        for a in order:
            if ub is not None and (ub[a] < best_gain or ub[a] <= tau):
                break
            dsq_a, _ = eng.ne(R + [a], act); evals += 1; n_eval += 1
            qa = f_of(dsq_a, d)
            g = math.fsum(qa - q_act)
            if g > best_gain or (g == best_gain and best is not None and a < best):
                best_gain, best, best_q = g, a, qa
        steps.append(dict(candidates=len(rem), evaluated=n_eval, selected=best, gain=best_gain,
                          ub_selected=None if ub is None or best is None else ub[best],
                          t_witness=t_wit, time=time.perf_counter() - t0))
        if best is None or best_gain <= tau:
            break
        R.append(best); rem.remove(best); q_act = best_q
    return R, evals, steps


def run(method, X, y, d, eps):
    base, _, b3 = method.partition("+")
    eng = KD(X, y) if base == "KD" else BF(X, y)
    return greedy(eng, d, eps, b3 == "B3")


METHODS = ("BF", "BF+B3", "KD", "KD+B3")
