"""R2.1 / R1.17: Friedman + Nemenyi with one block per (dataset, criterion) instead of one per configuration.
Block value per method = geometric mean over the usable configurations of that dataset of the 5-fold mean time
(unfinished -> LIMIT). Sensitivity for unfinished runs: LIMIT x {1, 2, 10}. Reads e1/e1_raw.jsonl only."""
import json, os, sys, statistics as st
from collections import defaultdict
import numpy as np
from scipy import stats
E1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "e1"))
sys.path.insert(0, E1)
from run_e1 import superseded, is_superseded, LIMIT
from stats_e1 import MAIN, Q05

sup = superseded(); T = defaultdict(lambda: defaultdict(dict)); zero = set()
for r in map(json.loads, open(os.path.join(E1, "e1_raw.jsonl"))):
    if is_superseded(r, sup) or r["status"] != "ok":
        continue
    k = (r["dataset"], r["criterion"], r["param"]); T[k][r["method"]][r["fold"]] = r["time"]
    if r["method"] != "FARA" and len(r["R"]) == 0:
        zero.add(k)
blocks = sorted(k for k in T if len(T[k].get("KD-PA+B3", {})) == 5 and k not in zero)
full = lambda k, m: st.mean(T[k][m].values()) if len(T[k].get(m, {})) == 5 else None
for mult in (1, 2, 10):
    print(f"=== unfinished = {mult} x LIMIT ===")
    for crit, meths in MAIN.items():
        ds = sorted({k[0] for k in blocks if k[1] == crit})
        M = []
        for d in ds:
            ks = [k for k in blocks if k[1] == crit and k[0] == d]
            M.append([np.exp(np.mean([np.log(full(k, m) if full(k, m) is not None else mult * LIMIT) for k in ks]))
                      for m in meths])
        M = np.array(M); N = len(ds); k_ = len(meths)
        chi, p = stats.friedmanchisquare(*M.T)
        R = np.mean([stats.rankdata(row) for row in M], axis=0)
        cd = Q05[k_] * np.sqrt(k_ * (k_ + 1) / (6 * N))
        best = meths[int(np.argmin(R))]
        print(f"{crit}: {N} datasets, chi2 = {chi:.2f}, p = {p:.2e}, CD = {cd:.2f}")
        for m, rk in sorted(zip(meths, R), key=lambda x: x[1]):
            gap = rk - R[meths.index("KD-PA+B3")]
            print(f"   {m:10s} rank {rk:.2f}  gap vs KD-PA+B3 {gap:+.2f}{' SIG' if gap > cd else ''}")
        # Wilcoxon per dataset KD-PA+B3 vs each other (Holm), as an alternative to Nemenyi
        ps = []
        for m in meths:
            if m == "KD-PA+B3": continue
            w = stats.wilcoxon(M[:, meths.index(m)], M[:, meths.index("KD-PA+B3")], alternative="greater")
            ps.append((m, w.pvalue))
        ps.sort(key=lambda x: x[1]); kk = len(ps)
        print("   Wilcoxon one-sided KD-PA+B3 faster, Holm:", ", ".join(
            f"{m} p={pv:.4f} adj={min(1, pv*(kk-i)):.4f}" for i, (m, pv) in enumerate(ps)))
