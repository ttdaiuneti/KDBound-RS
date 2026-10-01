"""Numbers of manuscript Section 5.2 (RQ1), from e1_raw.jsonl only (superseded rows excluded).
Re-run after any re-run of E1 rows and update sec_exp.tex (marked RECHECK)."""
import collections
import json
import statistics as st

from run_e1 import superseded, is_superseded

sup = superseded(); g = collections.defaultdict(dict); fara = {}
for r in map(json.loads, open("e1_raw.jsonl")):
    if is_superseded(r, sup) or r["status"] != "ok":
        continue
    k = (r["dataset"], r["criterion"], r["param"], r["fold"])
    if r["method"] == "FARA":
        fara[k] = r["R"]
    else:
        g[k][r["method"]] = tuple(r["R"])
multi = [k for k, m in g.items() if len(m) >= 2]
nm = [len(g[k]) for k in multi]
print("groups", len(g), "| >=2 exact methods", len(multi), "| identical ordered",
      sum(len(set(g[k].values())) == 1 for k in multi), "| methods per group min/median/max",
      min(nm), st.median(nm), max(nm), "| runs", sum(nm))
print("single-method groups", collections.Counter(" ".join(k[:3]) for k, m in g.items() if len(m) == 1))
cmp = [(k, fara[k], next(iter(g[k].values()))) for k in fara if k in g and g[k]]
cmp = [c for c in cmp if len(c[2]) > 0]
d = [len(a) - len(b) for _, a, b in cmp]
print("FARA folds", len(cmp), "| same set", sum(set(a) == set(b) for _, a, b in cmp), "| larger", sum(x > 0 for x in d),
      "| smaller", sum(x < 0 for x in d), "| mean diff %.2f max %d" % (st.mean(d), max(d)))
print("differing folds per dataset", dict(collections.Counter(k[0] for k, a, b in cmp if set(a) != set(b))))
