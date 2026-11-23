"""R1.12: fold-wise vs whole-table min-max normalization. Compares, per (dataset, criterion, param, fold), the reduct
of code/foldnorm/foldnorm_run.jsonl with the EXACT reduct of e1/e1_acc.jsonl, and the 3-NN / SVM-RBF accuracies."""
import json, os, statistics as st
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
fn = [json.loads(l) for l in open(os.path.join(HERE, "..", "code", "foldnorm", "foldnorm_run.jsonl"))]
e1 = [json.loads(l) for l in open(os.path.join(HERE, "..", "..", "e1", "e1_acc.jsonl"))]
E = {(r["dataset"], r["criterion"], r["param"], r["fold"]): r for r in e1 if r.get("group") == "EXACT" and r.get("status") == "ok"}
F = {(r["dataset"], r["criterion"], r["param"], r["fold"]): r for r in fn if "fold" in r and r.get("group", "EXACT") == "EXACT"}
FA = {(r["dataset"], r["criterion"], r["param"], r["fold"]): r for r in fn if "fold" in r and r.get("group") == "FARA"}
EF = {(r["dataset"], r["criterion"], r["param"], r["fold"]): r for r in e1 if r.get("group") == "FARA" and r.get("status") == "ok"}
keys = sorted(k for k in F if k in E)
same = sum(F[k]["selected_features"] == E[k]["selected_features"] for k in keys)
sameset = sum(set(F[k]["selected_features"].split("|")) == set(E[k]["selected_features"].split("|")) for k in keys)
print(f"folds compared: {len(keys)}; identical ordered reduct: {same}; identical set: {sameset}")
cfg = defaultdict(list)
for k in keys:
    cfg[k[:3]].append(k)
for crit in ("NRS", "FRS"):
    rows = [c for c in cfg if c[1] == crit and len(cfg[c]) == 5]
    for clf in ("acc_3nn", "acc_svm_rbf"):
        d = [100 * (st.mean(F[k][clf] for k in cfg[c]) - st.mean(E[k][clf] for k in cfg[c])) for c in rows]
        if d:
            print(f"{crit} {clf}: configs {len(d)}, mean diff {st.mean(d):+.3f} pts, max |diff| {max(map(abs, d)):.3f} pts, "
                  f"configs with |diff| > 0.5 pts: {sum(abs(x) > 0.5 for x in d)}")
    nd = [c for c in rows if any(F[k]["selected_features"] != E[k]["selected_features"] for k in cfg[c])]
    print(f"{crit}: configs with a different reduct in some fold: {len(nd)} of {len(rows)}",
          sorted(set(c[0] for c in nd)))
    for c in nd:
        sz = [(len(E[k]["selected_features"].split("|")), len(F[k]["selected_features"].split("|"))) for k in cfg[c]]
        print("   ", c, "sizes e1/fold:", sz)

# dataset level (one value per dataset: mean over its configurations and folds), Wilcoxon signed-rank two-sided
from scipy.stats import wilcoxon
for crit in ("NRS", "FRS"):
    for clf in ("acc_3nn", "acc_svm_rbf"):
        ds = sorted({c[0] for c in cfg if c[1] == crit and len(cfg[c]) == 5})
        d = []
        for name in ds:
            ks = [k for c in cfg if c[0] == name and c[1] == crit and len(cfg[c]) == 5 for k in cfg[c]]
            d.append(100 * (st.mean(F[k][clf] for k in ks) - st.mean(E[k][clf] for k in ks)))
        p = wilcoxon(d).pvalue if any(abs(x) > 0 for x in d) else 1.0
        print(f"dataset level {crit} {clf}: {len(d)} datasets, fold-wise minus whole-table mean {st.mean(d):+.3f} pts, "
              f"higher in {sum(x > 0 for x in d)}, lower in {sum(x < 0 for x in d)}, Wilcoxon p = {p:.3f}")

# exact vs FARA under both normalizations (NRS), as in Section 5.6: equal within 0.05 points
def wdl(A, B, keys, clf):
    cf = defaultdict(list)
    for k in keys:
        cf[k[:3]].append(k)
    w = d_ = l = 0; ds = defaultdict(list)
    for c, ks in cf.items():
        if len(ks) < 5:
            continue
        diff = 100 * (st.mean(B[k][clf] for k in ks) - st.mean(A[k][clf] for k in ks))   # FARA - exact
        ds[c[0]].append(diff)
        if diff > 0.05: w += 1
        elif diff < -0.05: l += 1
        else: d_ += 1
    v = [st.mean(x) for x in ds.values()]
    p = wilcoxon(v).pvalue if any(abs(x) > 0 for x in v) else 1.0
    return w, l, d_, len(v), p
for name, (A, B) in {"whole-table": (E, EF), "fold-wise": (F, FA)}.items():
    keys = [k for k in A if k in B and k[1] == "NRS"]
    for clf in ("acc_svm_rbf", "acc_3nn"):
        if keys:
            w, l, d_, nd, p = wdl(A, B, keys, clf)
            print(f"FARA vs exact, {name}, {clf}: FARA higher {w}, lower {l}, equal {d_}; {nd} datasets, Wilcoxon p = {p:.2f}")
