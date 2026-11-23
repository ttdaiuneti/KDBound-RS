"""R1.18: which properties of a table decide how many candidates the bound skips.

Sources: code/steps/steps_run.jsonl (KD-PA+B3 with step records, all NRS and FRS configurations of the 14 datasets) and
code/hd/hd_steps.jsonl (NRS delta = 0.10 on dilbert subsets m = 50..400 and Santander m = 200).
Per configuration (FRS: steps up to the tau = 1e-9 prefix), over the steps after the first:
  skip   = 1 - sum(evaluated) / sum(candidates)
  zero   = share of candidates whose bound is 0 (they cannot move any active object; skipped by any best gain >= 1)
  low    = share of candidates whose bound is below the best gain of the step (= skip up to ties, Prop. 5)
from the first step (no bound involved):
  half   = share of attributes whose first-step gain is at least half the best first-step gain
Dataset level = median over the configurations of a (dataset, criterion); Spearman correlations across them."""
import json, os, statistics as st
from collections import defaultdict
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "..", "code")
rows = [json.loads(l) for l in open(os.path.join(C, "steps", "steps_run.jsonl"))]
rows = [r for r in rows if "steps" in r and not r["partial"]]
hd = [json.loads(l) for l in open(os.path.join(C, "hd", "hd_steps.jsonl"))] if os.path.exists(
    os.path.join(C, "hd", "hd_steps.jsonl")) else []


def nsteps(r):
    if r["criterion"] == "NRS":
        return len(r["steps"])
    for i, s in enumerate(r["steps"]):
        if s["selected"] is None or s["gain"] <= 1e-9:
            return i + 1
    return len(r["steps"])


def measures(r):
    S = r["steps"][:nsteps(r)]
    T = [s for s in S[1:] if s["ub_all"]]
    ca = sum(s["candidates"] for s in T)
    if len(S) < 2 or not ca:
        return None
    g1 = [e[2] for e in S[0]["ev"]]; b = max(g1)
    return dict(m=len(g1), skip=1 - sum(s["evaluated"] for s in T) / ca,
                zero=sum(sum(u == 0 for u in s["ub_all"]) for s in T) / ca,
                half=sum(x >= 0.5 * b for x in g1) / len(g1) if b > 0 else float("nan"),
                rlen=sum(1 for s in S if s["selected"] is not None and s["gain"] > 1e-9))


agg = defaultdict(list)
for r in rows:
    v = measures(r)
    if v:
        agg[(r["criterion"], r["dataset"])].append(v)
L = [dict(k=k, **{f: st.median(x[f] for x in v) for f in ("m", "skip", "zero", "half", "rlen")}) for k, v in agg.items()]
print(f"{len(L)} dataset-criterion pairs (14 datasets)")
for f in ("m", "zero", "half"):
    rho, p = spearmanr([x[f] for x in L], [x["skip"] for x in L]); print(f"  Spearman skip ~ {f}: {rho:.2f} (p = {p:.1e})")
for x in sorted(L, key=lambda x: x["skip"]):
    print(f"  {x['k']}: m {x['m']:.0f} skip {x['skip']:.2f} zero {x['zero']:.2f} half {x['half']:.2f} |R| {x['rlen']:.0f}")
print("probed high-dimensional tables (NRS, delta = 0.10, fold 1):")
for r in hd:
    v = measures(r)
    if v:
        print(f"  {r['dataset']} m={r['m']}: skip {v['skip']:.2f} zero {v['zero']:.2f} half {v['half']:.2f} |R| {v['rlen']}"
              f" partial {r['partial']} time {r['time']:.1f}")

# A property measured before the run: share of attributes with interquartile range below delta (capped e1 tables,
# min-max scaled as in the runs), against the skipped share of NRS delta = 0.10, fold 1.
import sys
import numpy as np
sys.path.insert(0, os.path.join(HERE, "..", "..", "e1"))
from data_e1 import load
sk = {}
for r in rows:
    if r["criterion"] == "NRS" and r["param"] == "0.10" and r["fold"] == 1:
        v = measures(r)
        if v:
            sk[r["dataset"]] = v["skip"]
xs, ys = [], []
for d in sorted(sk):
    X, _ = load(d); q = np.percentile(X, [25, 75], axis=0)
    xs.append(float(np.mean(q[1] - q[0] < 0.10))); ys.append(sk[d])
rho, p = spearmanr(xs, ys)
print(f"NRS delta=0.10 fold 1, {len(xs)} tables: Spearman skip ~ share of attributes with IQR < delta: {rho:.2f} (p = {p:.2f})")
