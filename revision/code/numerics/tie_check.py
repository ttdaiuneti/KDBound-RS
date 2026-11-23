"""Boundary objects found by the margin certificate (relative gap < 1e-9; 11 units, 8 of them on house-16h): for each
such (unit, step, candidate), call every engine of the e1 code and of the revised code on these objects and compare
the decisions "has an enemy within delta" with each other and with exact arithmetic on the stored values (fractions,
over all enemies within delta (1 + 1e-6)). One-attribute subsets also get the FNC first-step decision (e1 window,
revised explicit test). Output: tie_check.jsonl, one line per (unit, step, candidate) with boundary objects."""
import json
import os
import sys
from fractions import Fraction

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engines_rev as REV            # noqa: E402
from boundary_test import fnc_step1_positive_e1   # noqa: E402
E = REV.E
sys.path.insert(0, REV.E1)
from data_e1 import load             # noqa: E402
from run_e1 import folds             # noqa: E402

UNITS = []
for ln in open(os.path.join(HERE, "margin_certificate.jsonl")):
    r = json.loads(ln)
    if r.get("n_gap_lt_1e9", 0) > 0 and (r["dataset"], r["param"], r["fold"]) not in [u[:3] for u in UNITS]:
        UNITS.append((r["dataset"], r["param"], r["fold"], r["R"]))


def main():
    out = open(os.path.join(HERE, "tie_check.jsonl"), "w"); tot = dis_e1 = dis_rev = vs_exact = 0
    for slug, param, fold, R in UNITS:
        X, y = load(slug); tr = folds(y)[fold - 1][0]; X, y = X[tr], y[tr]
        delta = float(param); d2 = delta * delta; n, m = X.shape; classes = np.unique(y)
        kd = REV.KD(X, y, delta); act = np.arange(n)
        eng = {f"{t}:{nm}": mod.make(nm, X, y, delta) for t, mod in (("e1", E), ("rev", REV))
               for nm in ("KD-PA", "BF-PA", "CABIN", "BUCKET", "HASH")}
        for t in range(len(R) + 1):
            Rt = R[:t]
            for a in [a for a in range(m) if a not in Rt]:
                B = Rt + [a]; Z = np.ascontiguousarray(X[:, B]); ya = y[act]; best = np.full(len(act), np.inf)
                for k in classes:
                    idx = np.flatnonzero(y == k); qs = np.flatnonzero(ya != k)
                    dd, j = cKDTree(Z[idx]).query(Z[act[qs]], k=1, distance_upper_bound=delta * (1 + 1e-6))
                    f = np.isfinite(dd)
                    best[qs[f]] = np.minimum(best[qs[f]], REV.d2seq(X, act[qs[f]], idx[j[f]], B))
                g = np.abs(best - d2) / d2
                bnd = act[np.isfinite(g) & (g < 1e-9)]
                if not len(bnd):
                    continue
                dec = {k_: eng[k_].blocked(B, bnd).tolist() for k_ in eng}
                if len(B) == 1:
                    dec["e1:FNC"] = (~fnc_step1_positive_e1(X, y, delta, a)[bnd]).tolist()
                    dec["rev:FNC"] = (~REV.fnc_step1_positive(X, y, delta, a)[bnd]).tolist()
                ex = []
                for x in bnd:
                    en = np.flatnonzero(y != y[x])
                    near = en[np.flatnonzero(REV.d2seq(X, np.full(len(en), x), en, B) <= d2 * (1 + 1e-5))]
                    ex.append(any(sum((Fraction(float(X[x, c])) - Fraction(float(X[e, c]))) ** 2 for c in B)
                                  <= Fraction(d2) for e in near))
                e1v = [v for k_, v in dec.items() if k_.startswith("e1")]; rv = [v for k_, v in dec.items() if k_.startswith("rev")]
                a_e1 = all(v == e1v[0] for v in e1v); a_rev = all(v == rv[0] for v in rv)
                tot += 1; dis_e1 += not a_e1; dis_rev += not a_rev; vs_exact += rv[0] != ex
                out.write(json.dumps(dict(dataset=slug, param=param, fold=fold, step=t, attr=a, B_size=len(B),
                                          objects=[int(v) for v in bnd], exact=ex, decisions=dec,
                                          e1_agree=a_e1, rev_agree=a_rev, rev_equals_exact=rv[0] == ex,
                                          e1_equals_rev=e1v[0] == rv[0])) + "\n")
            if t < len(R):
                act = act[kd.blocked(R[:t + 1], act)]
        print(slug, param, fold, "cases so far", tot, flush=True)
    print(f"(unit, step, candidate) with boundary objects: {tot}; e1 engines disagree: {dis_e1}; revised disagree: {dis_rev}; "
          f"revised differs from exact arithmetic: {vs_exact}")


if __name__ == "__main__":
    main()
