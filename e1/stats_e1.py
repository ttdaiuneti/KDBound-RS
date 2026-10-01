"""E1 statistics (e1_prereg.md section 11.3), from the raw file only (superseded rows excluded).

Blocks = usable configurations (dataset, criterion, param): KD-PA+B3 finished all 5 folds and no exact fold has |R| = 0.
Per method and block: mean time over the 5 folds; a method that did not finish all 5 folds (timeout / memout / skipped)
gets LIMIT (900 s), i.e. ranked last with ties.
- Friedman + Nemenyi (critical difference, alpha = 0.05) on the main-grid exact methods, per criterion.
- One-sided Wilcoxon signed-rank (E+B3 faster than E) on blocks where both E and E+B3 finished all 5 folds.
Output: printed; stats_e1.txt.
"""
import json
import os
import statistics as st
from collections import defaultdict

import numpy as np
from scipy import stats

from run_e1 import superseded, is_superseded, LIMIT

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = {"NRS": ("KD-PA+B3", "KD-PA", "FNC", "FNC+B3", "HASH"),
        "FRS": ("KD-PA+B3", "KD-PA", "FA-FPR", "FA-FPR+B3")}
PAIRS = [("KD-PA", "KD-PA+B3"), ("FNC", "FNC+B3"), ("HASH", "HASH+B3"), ("CABIN", "CABIN+B3"),
         ("BUCKET", "BUCKET+B3"), ("BF-PA", "BF-PA+B3"), ("FA-FPR", "FA-FPR+B3")]
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}  # Demsar 2006, Nemenyi q_alpha


def main():
    sup = superseded()
    T = defaultdict(lambda: defaultdict(dict)); zero = set()
    for r in map(json.loads, open(os.path.join(HERE, "e1_raw.jsonl"))):
        if is_superseded(r, sup) or r["status"] != "ok":
            continue
        k = (r["dataset"], r["criterion"], r["param"])
        T[k][r["method"]][r["fold"]] = r["time"]
        if r["method"] != "FARA" and len(r["R"]) == 0:
            zero.add(k)
    blocks = sorted(k for k in T if len(T[k].get("KD-PA+B3", {})) == 5 and k not in zero)
    full = lambda k, m: st.mean(T[k][m].values()) if len(T[k].get(m, {})) == 5 else None
    lines = []
    for crit, meths in MAIN.items():
        B = [k for k in blocks if k[1] == crit]
        M = np.array([[full(k, m) if full(k, m) is not None else LIMIT for m in meths] for k in B])
        chi, p = stats.friedmanchisquare(*M.T)
        ranks = np.mean([stats.rankdata(row) for row in M], axis=0)
        cd = Q05[len(meths)] * np.sqrt(len(meths) * (len(meths) + 1) / (6 * len(B)))
        lines.append(f"{crit}: {len(B)} blocks, Friedman chi2 = {chi:.1f}, p = {p:.2e}, Nemenyi CD = {cd:.2f}")
        for m, rk in sorted(zip(meths, ranks), key=lambda x: x[1]):
            lines.append(f"   {m:10s} mean rank {rk:.2f}")
        best = min(zip(ranks, meths))[1]
        lines.append(f"   rank gap of the others to {best}: " + ", ".join(
            f"{m} {rk - ranks[meths.index(best)]:+.2f}{' (sig)' if rk - ranks[meths.index(best)] > cd else ''}"
            for m, rk in zip(meths, ranks) if m != best))
    lines.append("Wilcoxon one-sided (E+B3 faster), blocks where both finished all 5 folds:")
    for e, eb in PAIRS:
        ks = [k for k in blocks if full(k, e) is not None and full(k, eb) is not None]
        if len(ks) < 5:
            lines.append(f"   {e:7s} vs {eb:10s}: n = {len(ks)} (too few)"); continue
        a = np.array([full(k, e) for k in ks]); b = np.array([full(k, eb) for k in ks])
        w = stats.wilcoxon(a, b, alternative="greater")
        ratio = a / b
        lines.append(f"   {e:7s} vs {eb:10s}: n = {len(ks):2d}, W = {w.statistic:.0f}, p = {w.pvalue:.2e}, "
                     f"median speedup {np.median(ratio):.2f}x, B3 slower in {int((ratio < 1).sum())}/{len(ks)}")
    txt = "\n".join(lines); print(txt)
    open(os.path.join(HERE, "stats_e1.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
