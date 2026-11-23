"""Targeted boundary tests for the NRS engines (revision, R2.2 / R1.9).

Each case is a table with one object x of class 0 and one object y of class 1 (plus 30 far fillers of class 0) whose squared distance lies within a few ulps of delta^2,
or, for one attribute, within 1e-9 of delta (the FNC window). For every case we record the decision "the pair is
within delta" of
  - exact arithmetic on the stored values: sum (x_a - y_a)^2 <= fl(delta*delta), computed with fractions;
  - each e1 engine (KD, BF-PA, CABIN, BUCKET, HASH, and FNC step 1 for one attribute);
  - each revised engine (engines_rev: uniform sequential sum, explicit test in FNC step 1).
Output: boundary_test.jsonl (one line per case) and a summary printed to boundary_test.log.
"""
import json
import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engines_rev as REV   # noqa: E402
E = REV.E; g1 = REV.g1

HERE = os.path.dirname(os.path.abspath(__file__))
DELTAS = (0.05, 0.10, 0.15, 0.20)
DIMS = (1, 2, 3, 4, 8, 12, 16)
PER = 150            # cases per (delta, dim)
rng = np.random.default_rng(0)


def exact_within(x, y, delta):
    s = sum((Fraction(float(a)) - Fraction(float(b))) ** 2 for a, b in zip(x, y))
    return s <= Fraction(delta * delta)


def fnc_step1_positive_e1(X, y, delta, a):
    """Verbatim copy of the first-step window count of e1 fnc_e1 (engines_e1.py), returning the positive objects."""
    n = len(y); classes = np.unique(y)
    onehot = (y[:, None] == classes[None, :]).astype(np.int64)
    o = np.argsort(X[:, a], kind="stable"); v = X[o, a]
    P = np.vstack([np.zeros((1, len(classes)), np.int64), np.cumsum(onehot[o], 0)])
    lo = np.searchsorted(v, v - delta - g1.WIN, "left"); hi = np.searchsorted(v, v + delta + g1.WIN, "right")
    own = np.searchsorted(classes, y[o])
    enemies = (hi - lo) - (P[hi, own] - P[lo, own])
    pos = np.zeros(n, bool); pos[o] = enemies == 0
    return pos


def engines(X, yv, delta):
    out = {}
    for tag, mod in (("e1", E), ("rev", REV)):
        for name in ("KD-PA", "BF-PA", "CABIN", "BUCKET", "HASH"):
            eng = mod.make(name, X, yv, delta)
            out[f"{tag}:{name}"] = bool(eng.blocked(list(range(X.shape[1])), np.arange(len(yv)))[0])
    if X.shape[1] == 1:
        out["e1:FNC"] = not bool(fnc_step1_positive_e1(X, yv, delta, 0)[0])
        out["rev:FNC"] = not bool(REV.fnc_step1_positive(X, yv, delta, 0)[0])
    return out


def cases():
    for delta in DELTAS:
        for dim in DIMS:
            made = 0
            while made < PER:
                x = 0.3 + 0.2 * rng.random(dim)
                u = rng.normal(size=dim); u /= np.linalg.norm(u)
                y = x + delta * u
                # walk one coordinate by ulps until the sequential sum is within 4 ulps of delta^2
                c = int(rng.integers(dim)); d2 = delta * delta
                for _ in range(200):
                    s = float(REV.d2seq(np.vstack([x, y]), np.array([0]), np.array([1]), list(range(dim)))[0])
                    gap = (s - d2) / np.spacing(d2)
                    if abs(gap) <= 4:
                        break
                    step = np.sign(y[c] - x[c]) * (-1 if gap > 0 else 1)
                    y[c] = np.nextafter(y[c], y[c] + step * np.inf)
                else:
                    continue
                kind = "sphere"
                made += 1
                yield delta, dim, kind, x, y
            if dim == 1:   # the FNC window: |x - y| just above delta, up to WIN = 1e-9
                for off in (1e-12, 1e-11, 1e-10, 3e-10, 6e-10, 9e-10, 9.9e-10):
                    x = np.array([0.4 + 0.1 * rng.random()]); y = x + delta + off
                    yield delta, dim, "window", x, y


def main():
    rows = []
    with open(os.path.join(HERE, "boundary_test.jsonl"), "w") as f:
        for delta, dim, kind, x, y in cases():
            # 30 own-class fillers of x near the corner (1, ..., 1), farther than 0.2 from x and y, so that the
            # engines see a table of ordinary size (HASH sizes its hypercube from n and m)
            F = 0.95 + 0.05 * rng.random((30, dim))
            X = np.vstack([x, y, F]); yv = np.array([0, 1] + [0] * 30)
            r = dict(delta=delta, dim=dim, kind=kind, exact=bool(exact_within(x, y, delta)), **engines(X, yv, delta))
            rows.append(r); f.write(json.dumps(r) + "\n")
    names = sorted(k for k in rows[0] if ":" in k) + ["e1:FNC", "rev:FNC"]
    names = list(dict.fromkeys(names))
    print(f"cases: {len(rows)} (sphere {sum(r['kind'] == 'sphere' for r in rows)}, "
          f"window {sum(r['kind'] == 'window' for r in rows)})")
    print("cases with exact decision 'within':", sum(r["exact"] for r in rows))
    for tag in ("e1", "rev"):
        cols = [n for n in names if n.startswith(tag)]
        dis = [r for r in rows if len({r[n] for n in cols if n in r}) > 1]
        print(f"\n[{tag}] cases where the engines disagree among themselves: {len(dis)}")
        for kind in ("sphere", "window"):
            for dim in DIMS:
                sub = [r for r in dis if r["kind"] == kind and r["dim"] == dim]
                if sub:
                    print(f"   {kind:6s} dim={dim:2d}: {len(sub)}  e.g. " +
                          ", ".join(f"{n.split(':')[1]}={sub[0][n]}" for n in cols if n in sub[0]))
        for n in cols:
            have = [r for r in rows if n in r]
            print(f"   {n:12s} differs from exact arithmetic in {sum(r[n] != r['exact'] for r in have)} of {len(have)}")


if __name__ == "__main__":
    main()
