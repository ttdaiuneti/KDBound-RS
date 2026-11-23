"""R1.3: evaluations of the pruned greedy search with three witness families (code/witness/witness_run.jsonl).
Skipped share = 1 - evaluations / candidates offered (sum over steps of the remaining candidates)."""
import json, os, statistics as st
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "..", "code", "witness", "witness_run.jsonl"))]
by = defaultdict(dict)
for r in rows:
    by[(r["criterion"], r["dataset"], r["param"])][r["witness"]] = r
bad = [r for r in rows if r.get("status") in ("timeout", "error") or r.get("partial")]
print("units:", len(rows), " not finished:", len(bad), sorted({(r['criterion'], r['dataset'], r['witness']) for r in bad}))
print("reduct differs from e1:", [(r["criterion"], r["dataset"], r["param"], r["witness"]) for r in rows if r.get("same_as_e1") is False])
for crit in ("NRS", "FRS"):
    sk = defaultdict(list); ratio = defaultdict(list); worse = defaultdict(int); n = 0
    for k, d in by.items():
        if k[0] != crit or not all(w in d and d[w].get("status") is None and not d[w].get("partial") for w in ("NE", "K5", "RAND")):
            continue
        n += 1
        offered = sum(c for c, _ in d["NE"]["per_step"])
        for w in ("NE", "K5", "RAND"):
            sk[w].append(1 - d[w]["evals"] / offered)
            if w != "NE":
                ratio[w].append(d[w]["evals"] / d["NE"]["evals"]); worse[w] += d[w]["evals"] < d["NE"]["evals"]
    if n:
        print(f"\n{crit}: {n} configurations (fold 1) where all three finished")
        for w in ("NE", "K5", "RAND"):
            print(f"   {w:4s} skipped share: median {st.median(sk[w]):.3f}, range {min(sk[w]):.3f}-{max(sk[w]):.3f}"
                  + (f"; evals vs NE: median x{st.median(ratio[w]):.2f}, max x{max(ratio[w]):.2f}; fewer than NE in {worse[w]}"
                     if w != "NE" else ""))
