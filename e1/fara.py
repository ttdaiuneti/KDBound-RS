"""FARA (Chen, Yang, Li et al. 2022, Information Sciences 587:354-370; paper store L71), Algorithm 2, for the
fixed-delta NRS criterion, on the same KD neighbourhood engine as KD-PA/KD-B3 (equal engine, so only the
search strategy differs).

Reading of Algorithm 2 (line numbers of the paper) and the three points it leaves open (fixed before E1 runs):
- rho(x, A) = 1 if x in POS_A else 0; rho(U', A) = |U' ∩ POS_A|; granularity G_a = |{(x,y): |x_a-y_a| <= delta}|/|U|^2
  (Def. 1 with the delta-neighbourhood relation).
- line 11: U' = {x : rho(x, A) != rho(x, AT)} = POS_AT \\ POS_A (positive approximation).
- line 14: T = {a in AT-A : G_a <= G_b}, b = attribute selected in the previous round.
- line 19: after selecting b, sort AT-A by rho(U', {a} ∪ {b}) (literally the pair {a, b}) and add the first t.
- lines 22-27: backward deletion while the constraint stays satisfied.
OPEN 1: t is not reported in the paper -> t = 1 (least aggressive).
OPEN 2: k is never incremented in the pseudocode (literal reading = always branch k = 1) -> k is incremented
        after every round, as the text of Sec. 3.4 describes.
OPEN 3: C_rho = "approximation quality >= 95% of AT" in the paper's experiments -> here gamma(A) >= gamma(AT),
        so that FARA targets the same quality as the exact greedy it is compared with.
Guards (not in the paper, logged): T empty -> T = AT-A; best rho gain 0 in a round -> stop ("stall").
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "g1"))
from methods import KD  # noqa: E402


def granularity(X, delta):
    n = X.shape[0]; G = np.empty(X.shape[1])
    for a in range(X.shape[1]):
        v = np.sort(X[:, a])
        G[a] = float((np.searchsorted(v, v + delta + 1e-9, "right") - np.searchsorted(v, v - delta - 1e-9, "left")).sum())
    return G / (n * n)


def fara(X, y, delta, t=1):
    eng = KD(X, y, delta); n, d = X.shape; allU = np.arange(n); evals = 0; log = {"stall": False, "t_empty": 0}
    G = granularity(X, delta)
    pos_AT = ~eng.blocked(list(range(d)), allU); gamma_AT = int(pos_AT.sum())
    A, k, b = [], 1, None
    pos_A = np.zeros(n, bool)
    while int(pos_A.sum()) < gamma_AT:
        rest = [a for a in range(d) if a not in A]
        if not rest:
            break
        Uq = np.flatnonzero(pos_AT & ~pos_A)                        # U' (line 11)
        if k == 1:
            cand = rest
        else:
            cand = [a for a in rest if G[a] <= G[b]]                 # line 14
            if not cand:
                cand = rest; log["t_empty"] += 1
        best, bg = None, 0
        for a in cand:                                               # lines 8 / 15
            g = int((~eng.blocked(A + [a], Uq)).sum()); evals += 1
            if g > bg:
                best, bg = a, g
        if best is None:
            log["stall"] = True; break
        b = best; A.append(b); k += 1
        rest = [a for a in range(d) if a not in A]
        if rest and t > 0:                                           # line 19 (attribute accelerator)
            Uq = np.flatnonzero(pos_AT & ~(~eng.blocked(A, allU)))
            sc = []
            for a in rest:
                sc.append((-int((~eng.blocked([a, b], Uq)).sum()), a)); evals += 1
            A += [a for _, a in sorted(sc)[:t]]
        pos_A = ~eng.blocked(A, allU)                                # line 20
    changed = True                                                   # lines 22-27 (backward deletion)
    while changed and len(A) > 1:
        changed = False
        for c in list(A):
            B = [a for a in A if a != c]
            if int((~eng.blocked(B, allU)).sum()) >= gamma_AT:
                A = B; changed = True; evals += 1
                break
            evals += 1
    return A, evals, log
