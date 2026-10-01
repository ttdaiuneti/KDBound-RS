"""E1 summary from the raw files only (e1_raw.jsonl, e1_acc.jsonl; superseded rows excluded).

Per (dataset, criterion, param):
- time mean/std over completed folds for KD-PA+B3 (ours), KD-PA (bound ablation) and every rival;
- best EXISTING exact rival = lowest mean time among rivals that completed all 5 folds
  (NRS: FNC, HASH, CABIN, BUCKET, BF-PA; FRS: FA-FPR). If none completed, the speedup is a lower bound 900/ours;
- speedup vs best existing and vs KD-PA;
- |R| of the exact reduct and of FARA; 3-NN / SVM accuracy (mean, std over folds) of the EXACT and FARA groups;
- flags: degenerate (exact |R| = 0 in some fold), ours incomplete.
Output: e1_summary.csv (+ printed table). Numbers for the paper must come from this file or the raw files.
"""
import csv
import json
import os
import statistics as st
from collections import defaultdict

from run_e1 import superseded, is_superseded, LIMIT

HERE = os.path.dirname(os.path.abspath(__file__))
EXISTING = {"NRS": ("FNC", "HASH", "CABIN", "BUCKET", "BF-PA"), "FRS": ("FA-FPR",)}


def ms(v):
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0) if v else (None, None)


def main():
    sup = superseded()
    T = defaultdict(dict); R = defaultdict(dict); ST = defaultdict(dict)
    for r in map(json.loads, open(os.path.join(HERE, "e1_raw.jsonl"))):
        if is_superseded(r, sup):
            continue
        k = (r["dataset"], r["criterion"], r["param"])
        ST[k].setdefault(r["method"], {})[r["fold"]] = r["status"]
        if r["status"] == "ok":
            T[k].setdefault(r["method"], {})[r["fold"]] = r["time"]
            R[k].setdefault(r["method"], {})[r["fold"]] = len(r["R"])
    A = defaultdict(lambda: defaultdict(list))
    for r in map(json.loads, open(os.path.join(HERE, "e1_acc.jsonl"))):
        if r["status"] == "ok":
            A[(r["dataset"], r["criterion"], r["param"], r["group"])]["3nn"].append(r["acc_3nn"])
            A[(r["dataset"], r["criterion"], r["param"], r["group"])]["svm"].append(r["acc_svm_rbf"])
    out = []
    for k in sorted(T.keys() | ST.keys()):
        ds, crit, param = k
        t = T.get(k, {})
        ours = list(t.get("KD-PA+B3", {}).values())
        full = {m: list(v.values()) for m, v in t.items() if len(v) == 5}
        riv = {m: st.mean(full[m]) for m in EXISTING[crit] if m in full}
        best = min(riv, key=riv.get) if riv else None
        om, osd = ms(ours)
        exact_R = [n for m, v in R.get(k, {}).items() if m != "FARA" for n in v.values()]
        row = dict(dataset=ds, criterion=crit, param=param, ours_folds=len(ours),
                   ours_mean=om, ours_std=osd,
                   kdpa_mean=ms(list(t.get("KD-PA", {}).values()))[0], kdpa_folds=len(t.get("KD-PA", {})),
                   best_existing=best or "none<900s", best_mean=riv.get(best),
                   speedup_vs_best=(riv[best] / om if best and om else (LIMIT / om if om else None)),
                   speedup_is_lower_bound=int(best is None and om is not None),
                   speedup_vs_kdpa=(st.mean(full["KD-PA"]) / om if "KD-PA" in full and om else None),
                   fnc_b3_mean=ms(list(t.get("FNC+B3", {}).values()))[0],
                   exact_R_mean=st.mean(exact_R) if exact_R else None,
                   fara_R_mean=ms(list(R.get(k, {}).get("FARA", {}).values()))[0],
                   degenerate=int(any(n == 0 for n in exact_R)))
        for g in ("EXACT", "FARA"):
            for c in ("3nn", "svm"):
                row[f"{g}_{c}_mean"], row[f"{g}_{c}_std"] = ms(A[(ds, crit, param, g)][c])
        out.append(row)
    with open(os.path.join(HERE, "e1_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    f2 = lambda x, p=1: "-" if x is None else f"{x:.{p}f}"
    print(f"{'dataset':26s} crit param  ours(s)        best existing       x_best   x_KDPA  |R|ex |R|FARA  "
          f"SVM ex / FARA")
    for r in out:
        sp = ("-" if r["speedup_vs_best"] is None else
              (">" if r["speedup_is_lower_bound"] else "") + f"{r['speedup_vs_best']:.1f}")
        print(f"{r['dataset']:26s} {r['criterion']} {r['param']:5s} "
              f"{f2(r['ours_mean']):>7s}±{f2(r['ours_std']):5s}({r['ours_folds']}) "
              f"{r['best_existing']:8s}{f2(r['best_mean']):>8s} {sp:>8s} {f2(r['speedup_vs_kdpa']):>7s} "
              f"{f2(r['exact_R_mean']):>5s} {f2(r['fara_R_mean']):>6s}  "
              f"{f2(r['EXACT_svm_mean'], 3)} / {f2(r['FARA_svm_mean'], 3)}{'  DEGENERATE' if r['degenerate'] else ''}")


if __name__ == "__main__":
    main()
