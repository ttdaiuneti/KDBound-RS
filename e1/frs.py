"""E1 FRS-min engines (e1_prereg_DRAFT.md, section 2).

Criterion (Dubois-Prade, Kleene-Dienes implicator, t-norm min, R_a(x,z) = max(0, 1 - |x_a - z_a|/sigma_a), crisp
decision; Qian et al. 2015 FSS = L83, Jensen & Shen 2009 = L85):
    mu_B(x) = min(1, ne_B(x)),  ne_B(x) = min over enemies z of max_{a in B} |Z_xa - Z_za|,  Z = X / sigma,
    Q(B) = sum_x mu_B(x); greedy adds argmax gain (ties -> lowest index), stops when best gain <= TOL.

Exactness across engines by construction: every engine computes distances on the same Z with subtraction and max
only (no rounding differences), and every gain / bound is summed with math.fsum (correctly rounded, so independent
of the order and of which saturated objects were dropped).

Methods (grid {BF, FA-FPR, KD} x {no bound, B3}):
    BF-DAR       brute force, all objects queried every evaluation
    FA-FPR       brute force + positive approximation (drop objects with mu = 1; Qian 2015 Thm 6)
    KD-PA        per-class cKDTree (p = inf, distance_upper_bound 1) + positive approximation
    <E>+B3       the same engine with the nearest-enemy bound
                 UB(a) = sum_x [min(1, max(ne_B(x), |Z_xa - Z_y*a|)) - min(1, ne_B(x))]   (theory/framework.md)
"""
import math
import time

import numpy as np
from scipy.spatial import cKDTree

TOL = 1e-9
ROWS = 256


def scale(X, sigma):
    """sigma = float (same for every attribute) or 'std' (per-attribute std of X; constant attribute -> weight 0)."""
    if sigma == "std":
        s = X.std(0)
        return np.where(s > 0, X / np.where(s > 0, s, 1.0), 0.0)
    return X / float(sigma)


class BF:
    def __init__(self, Z, y):
        self.Z, self.y = Z, y

    def ne(self, B, q):
        """(ne_B(q), index of a nearest enemy); ne capped at 1 is enough for mu, but kept exact for the bound."""
        Z, y = self.Z, self.y; out = np.empty(len(q)); arg = np.empty(len(q), int)
        for s in range(0, len(q), ROWS):
            qq = q[s:s + ROWS]
            D = np.zeros((len(qq), len(y)))
            for a in B:
                np.maximum(D, np.abs(Z[qq, a][:, None] - Z[None, :, a]), out=D)
            D[y[qq][:, None] == y[None, :]] = np.inf
            j = D.argmin(1); arg[s:s + ROWS] = j; out[s:s + ROWS] = D[np.arange(len(qq)), j]
        return out, arg


class KD(BF):
    def ne(self, B, q):
        Z, y = self.Z, self.y; out = np.full(len(q), np.inf); arg = np.full(len(q), -1); yq = y[q]
        ZB = np.ascontiguousarray(Z[:, B])
        for c in np.unique(y):
            idx = np.flatnonzero(y == c); qs = np.flatnonzero(yq != c)
            if not len(qs) or not len(idx):
                continue
            d, j = cKDTree(ZB[idx]).query(ZB[q[qs]], k=1, p=np.inf, distance_upper_bound=1.0 + 1e-9)
            f = np.isfinite(d)
            e = idx[j[f]]
            dd = np.abs(ZB[q[qs[f]]] - ZB[e]).max(1)          # same subtraction + max as BF
            b = dd < out[qs[f]]
            out[qs[f][b]] = dd[b]; arg[qs[f][b]] = e[b]
        far = arg < 0                                          # no enemy within 1: mu = 1, exact ne not needed
        out[far] = 1.0
        return out, arg


def greedy(eng, pa, bound):
    Z = eng.Z; n, d = Z.shape
    act = np.arange(n); R, rem, evals, steps = [], list(range(d)), 0, []
    mu_act = np.zeros(n)                                       # mu_R of the objects in act (R empty -> 0)
    while rem and len(act):
        t0 = time.perf_counter(); ub = None
        if bound and R:
            ne, ys = eng.ne(R, act)
            if isinstance(eng, KD):                            # far objects: mu already 1, contribute 0
                ys = np.where(ys < 0, act, ys)
            m0 = np.minimum(1.0, ne)
            ub = {a: math.fsum(np.minimum(1.0, np.maximum(ne, np.abs(Z[act, a] - Z[ys, a]))) - m0) for a in rem}
            order = sorted(rem, key=lambda a: (-ub[a], a))
        else:
            order = rem
        best_gain, best, best_mu, n_eval = 0.0, None, None, 0
        for a in order:
            if ub is not None and (ub[a] < best_gain or ub[a] <= TOL):
                break
            ne_a, _ = eng.ne(R + [a], act); evals += 1; n_eval += 1
            mu = np.minimum(1.0, ne_a)
            g = math.fsum(mu - mu_act)
            if g > best_gain or (g == best_gain and best is not None and a < best):
                best_gain, best, best_mu = g, a, mu
        steps.append(dict(active=int(len(act)), candidates=len(rem), evaluated=n_eval, pruned=len(rem) - n_eval,
                          selected=best, gain=best_gain,
                          ub_selected=None if ub is None or best is None else ub[best], time=time.perf_counter() - t0))
        if best is None or best_gain <= TOL:
            break
        R.append(best); rem.remove(best)
        if pa:
            keep = best_mu < 1.0
            act, mu_act = act[keep], best_mu[keep]
        else:
            mu_act = best_mu
    return R, evals, steps


def run_frs(method, X, y, sigma):
    Z = scale(X, sigma)
    base, _, b3 = method.partition("+")
    eng = KD(Z, y) if base == "KD-PA" else BF(Z, y)
    pa = base in ("FA-FPR", "KD-PA")
    return greedy(eng, pa, b3 == "B3")


METHODS = ("BF-DAR", "BF-DAR+B3", "FA-FPR", "FA-FPR+B3", "KD-PA", "KD-PA+B3")
