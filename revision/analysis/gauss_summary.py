"""Summary of the Gaussian kernel FRS run (code/gauss_run.jsonl): exactness KD vs KD+B3 per fold, evaluations, time."""
import json, os, statistics as st
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "..", "code", "gauss_run.jsonl"))]
by = defaultdict(dict)
for r in rows:
    by[(r["dataset"], r["kappa"])][(r["fold"], r["method"])] = r
print(f"{'dataset':16s} kappa  n_train  m  |R|  same  evB3  evKD  full  saved%  tB3(s)       tKD(s)      speedup  gainsNonDim")
for (ds, k), d in by.items():
    folds = sorted({f for f, _ in d})
    same, eb, ek, tb, tk, R, nondim, st_ = [], [], [], [], [], [], 0, set()
    for f in folds:
        b, kk = d.get((f, "KD+B3")), d.get((f, "KD"))
        for x in (b, kk):
            if x: st_.add(x["status"])
        if b and b["status"] == "ok":
            eb.append(b["evals"]); tb.append(b["time"]); R.append(len(b["R"]))
            g = [s["gain"] for s in b["steps"] if s["selected"] is not None]
            nondim += any(g[i + 1] > g[i] for i in range(len(g) - 1))
        if kk and kk["status"] == "ok":
            ek.append(kk["evals"]); tk.append(kk["time"])
        if b and kk and b["status"] == kk["status"] == "ok":
            same.append(b["R"] == kk["R"])
    m = next(x["m"] for x in d.values() if "m" in x); n = next(x["n_train"] for x in d.values() if "n_train" in x)
    full = m * (m + 1) // 2
    f = lambda v: f"{st.mean(v):7.1f}±{st.stdev(v):5.1f}" if len(v) > 1 else "   -   "
    sp = f"{st.mean(tk)/st.mean(tb):5.2f}" if tb and tk else "  -  "
    sv = f"{100*(1-st.mean(eb)/st.mean(ek)):5.1f}" if eb and ek else "  -  "
    print(f"{ds:16s} {k:5s} {n:7d} {m:3d} {st.mean(R) if R else float('nan'):4.1f} {sum(same)}/{len(same)} "
          f"{st.mean(eb) if eb else float('nan'):5.1f} {st.mean(ek) if ek else float('nan'):5.1f} {full:5d} {sv} "
          f"{f(tb)} {f(tk)} {sp}  {nondim}/{len(eb)}  status={sorted(st_)}")

print("\nPer-step facts (KD+B3, ok folds): first-step gain/n; step index of max gain (1-based); last step evaluated 0?")
for (ds, k), d in by.items():
    fs, pk, z = [], [], 0
    for f in range(1, 6):
        b = d.get((f, "KD+B3"))
        if not b or b["status"] != "ok":
            continue
        g = [s["gain"] / b["n_train"] for s in b["steps"]]
        fs.append(g[0]); pk.append(1 + max(range(len(g)), key=g.__getitem__)); z += b["steps"][-1]["evaluated"] == 0
    if fs:
        print(f"{ds:16s} {k}  first gain/n {min(fs):.1e}..{max(fs):.1e}  (<=0.01: {sum(x <= 0.01 for x in fs)}/{len(fs)})"
              f"  peak step {min(pk)}..{max(pk)}  last-step-zero-eval {z}/{len(fs)}")
