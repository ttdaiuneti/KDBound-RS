"""G1: exact NRS greedy reduction with positive approximation, seven neighbourhood engines
(pre-registration: g1_prereg.md).

Common criterion: delta-neighbourhood d_B(x,z)^2 <= delta^2 (Euclidean, float64); x is in POS_B iff no enemy
lies in its neighbourhood. Greedy: at each step add the attribute that moves most remaining objects into POS
(ties -> lowest index); stop when the best gain is 0.
"""
import math

import numpy as np
from scipy.spatial import cKDTree

PAIR_LIMIT = 300_000_000    # FNC stores its neighbour pairs; above this -> MemOut
CHUNK = 4_000_000           # pairs processed at once by the window-based engines
WIN = 1e-9                  # widening of filter windows (filters must be supersets)


class MemOut(Exception):
    pass


def expand(lo, hi):
    """Pairs (row, position) for windows [lo_i, hi_i)."""
    cnt = hi - lo
    rows = np.repeat(np.arange(len(lo)), cnt)
    start = np.repeat(lo - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt)
    return rows, np.arange(int(cnt.sum())) + start


def chunked(cnt, limit=CHUNK):
    """Split row indices so that each chunk holds at most `limit` pairs (at least one row)."""
    cs = np.concatenate(([0], np.cumsum(cnt)))
    i, n = 0, len(cnt)
    while i < n:
        j = max(i + 1, int(np.searchsorted(cs, cs[i] + limit, "right")) - 1)
        yield i, j
        i = j


class Engine:
    name = "?"

    def __init__(self, X, y, delta):
        self.X, self.y, self.delta, self.d2 = X, y, delta, delta * delta
        self.classes = np.unique(y)

    def any_within(self, B, q, e):
        """Query objects q[i] that have e[i] within delta on B (column-wise with early rejection)."""
        d2 = np.zeros(len(q))
        for c in B:
            col = self.X[:, c]
            d2 += (col[q] - col[e]) ** 2
            m = d2 <= self.d2
            q, e, d2 = q[m], e[m], d2[m]
            if not len(q):
                break
        return np.unique(q)

    def blocked(self, B, act):          # -> boolean over act: True = has an enemy within delta
        raise NotImplementedError

    # hooks used by the greedy driver (stateless engines ignore them)
    def select(self, B, a, act, blocked_mask):
        pass


# --------------------------------------------------------------------------- brute force
class BF(Engine):
    name = "BF-PA"

    def blocked(self, B, act):
        Z = np.ascontiguousarray(self.X[:, B]); sq = (Z ** 2).sum(1); out = np.zeros(len(act), bool)
        for s in range(0, len(act), 512):
            qa = act[s:s + 512]
            D2 = sq[qa][:, None] + sq[None, :] - 2.0 * Z[qa] @ Z.T
            cand = (D2 <= self.d2 + 1e-9) & (self.y[qa][:, None] != self.y[None, :])
            r, e = np.nonzero(cand)
            hit = self.any_within(B, qa[r], e)
            out[s:s + 512] = np.isin(qa, hit)
        return out


# --------------------------------------------------------------------------- control-sample filters
class Cabin(Engine):
    """Granular cabin (two control samples, two-sided triangle filter); n_ctrl=1 gives BUCKET."""

    def __init__(self, X, y, delta, n_ctrl=2):
        super().__init__(X, y, delta)
        self.n_ctrl = n_ctrl
        self.name = "CABIN" if n_ctrl == 2 else "BUCKET"
        self.cmin, self.cmax = X.min(0), X.max(0)

    def blocked(self, B, act):
        Z = self.X[:, B]
        D1 = np.sqrt(((Z - self.cmin[B]) ** 2).sum(1))
        D2 = np.sqrt(((Z - self.cmax[B]) ** 2).sum(1)) if self.n_ctrl == 2 else None
        hit_all = []
        for k in self.classes:
            Q = act[self.y[act] == k]
            if not len(Q):
                continue
            E = np.flatnonzero(self.y != k)
            E = E[np.argsort(D1[E], kind="stable")]; D1E = D1[E]
            lo = np.searchsorted(D1E, D1[Q] - self.delta - WIN, "left")
            hi = np.searchsorted(D1E, D1[Q] + self.delta + WIN, "right")
            for i, j in chunked(hi - lo):
                r, p = expand(lo[i:j], hi[i:j])
                q, e = Q[i:j][r], E[p]
                if D2 is not None:
                    m = np.abs(D2[e] - D2[q]) <= self.delta + WIN
                    q, e = q[m], e[m]
                hit_all.append(self.any_within(B, q, e))
        hit = np.concatenate(hit_all) if hit_all else np.empty(0, int)
        return np.isin(act, hit)


# --------------------------------------------------------------------------- hash hypercube
class Hash(Engine):
    name = "HASH"

    def __init__(self, X, y, delta):
        super().__init__(X, y, delta)
        n, M = X.shape
        self.cells = np.ceil(X / delta).astype(np.int64)            # cell index per attribute, 0..ceil(1/delta)
        self.base = int(math.ceil(1 / delta)) + 3
        sup = []
        for j in range(M):                                          # attribute superiority (App. B): smaller = better
            cnt = np.bincount(self.cells[:, j], minlength=self.base)
            pad = np.concatenate(([0], cnt, [0]))
            sup.append(float((cnt * (pad[:-2] + pad[1:-1] + pad[2:])).sum()))
        self.sup = np.array(sup)
        self.Kmax = min(int(round(abs(math.log(M * n) / math.log(0.5 * (delta + 0.1))))), M)   # Eq. (C8)

    def blocked(self, B, act):
        K = min(self.Kmax, len(B))
        attrs = sorted(B, key=lambda j: (self.sup[j], j))[:K]
        w = self.base ** np.arange(K, dtype=np.int64)
        code = ((self.cells[:, attrs] + 1) * w).sum(1)
        offs = np.array(np.meshgrid(*[[-1, 0, 1]] * K, indexing="ij")).reshape(K, -1).T
        off_codes = (offs * w).sum(1)
        hit_all = []
        for k in self.classes:
            Q = act[self.y[act] == k]
            E = np.flatnonzero(self.y != k)
            E = E[np.argsort(code[E], kind="stable")]
            uq, start, count = np.unique(code[E], return_index=True, return_counts=True)
            for oc in off_codes:
                if not len(Q):
                    break
                tgt = code[Q] + oc
                ix = np.searchsorted(uq, tgt)
                ok = ix < len(uq)
                ok[ok] = uq[ix[ok]] == tgt[ok]
                if not ok.any():
                    continue
                Qo = Q[ok]; lo = start[ix[ok]]; hi = lo + count[ix[ok]]
                hits = []
                for i, j in chunked(hi - lo):
                    r, p = expand(lo[i:j], hi[i:j])
                    hits.append(self.any_within(B, Qo[i:j][r], E[p]))
                h = np.concatenate(hits)
                hit_all.append(h)
                Q = Q[~np.isin(Q, h)]                                # blocked queries need no further offsets
        hit = np.concatenate(hit_all) if hit_all else np.empty(0, int)
        return np.isin(act, hit)


# --------------------------------------------------------------------------- KD tree (ours)
class KD(Engine):
    name = "KD-PA"

    def blocked(self, B, act):
        Z = np.ascontiguousarray(self.X[:, B]); out = np.zeros(len(act), bool); ya = self.y[act]
        for k in self.classes:
            m = self.y == k; idx = np.flatnonzero(m)
            qs = np.flatnonzero((ya != k) & ~out)
            if not len(qs) or not len(idx):
                continue
            d, j = cKDTree(Z[idx]).query(Z[act[qs]], k=1, distance_upper_bound=self.delta * (1 + 1e-9))
            f = np.isfinite(d)
            if f.any():   # confirm with the common float64 test (nearest within the widened bound decides)
                q, e = act[qs[f]], idx[j[f]]
                ok = ((self.X[q][:, B] - self.X[e][:, B]) ** 2).sum(1) <= self.d2
                out[qs[f][ok]] = True
        return out

    def nearest_enemy(self, B, act):
        """(ne^2 from coordinates, index of a nearest enemy) for act on B."""
        Z = np.ascontiguousarray(self.X[:, B]); ya = self.y[act]
        best = np.full(len(act), np.inf); arg = np.full(len(act), -1)
        for k in self.classes:
            idx = np.flatnonzero(self.y == k); qs = np.flatnonzero(ya != k)
            if not len(qs):
                continue
            d, j = cKDTree(Z[idx]).query(Z[act[qs]], k=1)
            b = d < best[qs]
            best[qs[b]] = d[b]; arg[qs[b]] = idx[j[b]]
        ne2 = ((self.X[act][:, B] - self.X[arg][:, B]) ** 2).sum(1)
        return ne2, arg


# --------------------------------------------------------------------------- greedy drivers
def greedy(eng, bound=False):
    """Positive-approximation greedy; bound=True uses the B3 nearest-enemy upper bound (KD engine)."""
    X, n, d = eng.X, eng.X.shape[0], eng.X.shape[1]
    act = np.arange(n); R, rem, evals = [], list(range(d)), 0
    while rem and len(act):
        if bound and R:
            ne2, ys = eng.nearest_enemy(R, act)
            ub = {a: int((ne2 + (X[act, a] - X[ys, a]) ** 2 > eng.d2 - 1e-12).sum()) for a in rem}
            order = sorted(rem, key=lambda a: (-ub[a], a))
        else:
            ub, order = None, rem
        best_gain, best, best_mask = 0, None, None
        for a in order:
            if ub is not None and ub[a] < max(best_gain, 1):
                break
            mask = eng.blocked(R + [a], act); evals += 1
            g = int(len(act) - mask.sum())
            if g > best_gain or (g == best_gain and g > 0 and a < best):
                best_gain, best, best_mask = g, a, mask
        if best_gain == 0:
            break
        R.append(best); rem.remove(best); act = act[best_mask]
    return R, evals


def fnc(X, y, delta):
    """FNC (Peng et al. 2022): sorted sliding window for the first attribute, then stored neighbour pairs
    filtered as attributes are added (enemy pairs only)."""
    n, d = X.shape; d2 = delta * delta; classes = np.unique(y); evals = 0
    onehot = (y[:, None] == classes[None, :]).astype(np.int64)
    # step 1: one sliding window per attribute
    best_gain, a1 = 0, None
    for a in range(d):
        o = np.argsort(X[:, a], kind="stable"); v = X[o, a]
        P = np.vstack([np.zeros((1, len(classes)), np.int64), np.cumsum(onehot[o], 0)])
        lo = np.searchsorted(v, v - delta - WIN, "left"); hi = np.searchsorted(v, v + delta + WIN, "right")
        own = np.searchsorted(classes, y[o])
        enemies = (hi - lo) - (P[hi, own] - P[lo, own])
        pos = np.zeros(n, bool); pos[o] = enemies == 0
        g = int(pos.sum()); evals += 1
        if g > best_gain:
            best_gain, a1, pos1 = g, a, pos
    if a1 is None:
        return [], evals
    R = [a1]; rem = [a for a in range(d) if a != a1]
    act = np.flatnonzero(~pos1)
    # store enemy pairs of the remaining objects on attribute a1
    Q, E, D2 = [], [], []
    tot = 0
    for k in classes:
        q = act[y[act] == k]; e = np.flatnonzero(y != k)
        e = e[np.argsort(X[e, a1], kind="stable")]; ve = X[e, a1]
        lo = np.searchsorted(ve, X[q, a1] - delta - WIN, "left"); hi = np.searchsorted(ve, X[q, a1] + delta + WIN, "right")
        tot += int((hi - lo).sum())
        if tot > PAIR_LIMIT:
            raise MemOut(f"{tot} pairs")
        for i, j in chunked(hi - lo):
            r, p = expand(lo[i:j], hi[i:j])
            qq, ee = q[i:j][r], e[p]; dd = (X[qq, a1] - X[ee, a1]) ** 2
            m = dd <= d2
            Q.append(qq[m]); E.append(ee[m]); D2.append(dd[m])
    Q, E, D2 = np.concatenate(Q), np.concatenate(E), np.concatenate(D2)
    act = np.unique(Q)
    while rem and len(act):
        best_gain, best = 0, None
        for a in rem:
            bad = D2 + (X[Q, a] - X[E, a]) ** 2 <= d2; evals += 1
            g = int(len(act) - len(np.unique(Q[bad])))
            if g > best_gain:
                best_gain, best = g, a
        if best_gain == 0:
            break
        R.append(best); rem.remove(best)
        D2 = D2 + (X[Q, best] - X[E, best]) ** 2
        m = D2 <= d2
        Q, E, D2 = Q[m], E[m], D2[m]
        act = np.unique(Q)
    return R, evals


def run(method, X, y, delta):
    if method == "FNC":
        return fnc(X, y, delta)
    eng = {"BF-PA": lambda: BF(X, y, delta), "BUCKET": lambda: Cabin(X, y, delta, 1),
           "CABIN": lambda: Cabin(X, y, delta, 2), "HASH": lambda: Hash(X, y, delta),
           "KD-PA": lambda: KD(X, y, delta), "KD-B3": lambda: KD(X, y, delta)}[method]()
    return greedy(eng, bound=(method == "KD-B3"))
