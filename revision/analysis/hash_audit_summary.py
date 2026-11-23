"""R2.3 / R1.15: summary of code/baselines/hash_audit_{path,greedy}.jsonl.

path: per dataset and engine, totals over the steps of the greedy path (one evaluation per step on (B_t, act_t)):
      pairs passed to the exact distance test, time, agreement with the k-d tree mask. Reference counts: naive
      (every active object against every enemy) and BF-PA (its BLAS prefilter passes only the enemy pairs within
      delta, up to 1e-9, so its count is the number of enemy pairs within delta of the active objects).
greedy: full HASH-B7 reduction vs the e1 reduct and the e1 HASH (Gamma) fold-1 time (e1 times were measured with four
      units running concurrently; idle/concurrent median 0.90, Section 5.1)."""
import json, os, statistics as st
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "..", "code", "baselines")
P = [json.loads(l) for l in open(os.path.join(C, "hash_audit_path.jsonl"))]
G = [json.loads(l) for l in open(os.path.join(C, "hash_audit_greedy.jsonl"))]
E1 = {}
for l in open(os.path.join(HERE, "..", "..", "e1", "e1_raw.jsonl")):
    r = json.loads(l)
    if r["criterion"] == "NRS" and r["param"] == "0.10" and r["fold"] == 1 and r["method"] == "HASH":
        E1[r["dataset"]] = r

tot = defaultdict(lambda: dict(pairs=0, time=0.0, naive=0, steps=0, same=True, status="ok", diff_attrs=0))
for r in P:
    k = (r["dataset"], r["engine"]); t = tot[k]
    if r.get("status") in ("timeout", "error"):
        t["status"] = r["status"]; continue
    t["pairs"] += r["pairs"] or 0; t["time"] += r["time"]; t["naive"] += r["naive"]; t["steps"] += 1
    t["same"] &= r["same"]
H = defaultdict(dict)
for r in P:
    if r.get("engine", "").startswith("HASH") and "hash_attrs" in r:
        H[(r["dataset"], r["step"])][r["engine"]] = r["hash_attrs"]
ndiff = defaultdict(int)
for (d, s), v in H.items():
    if len(v) == 2 and sorted(v["HASH-G"]) != sorted(v["HASH-B7"]):
        ndiff[d] += 1

ds = sorted({k[0] for k in tot}, key=lambda d: tot[(d, "BF-PA")]["naive"])
print("dataset: steps | pairs / pairs-within-delta (BF-PA) for HASH-G, HASH-B7, CABIN, BUCKET | pairs / naive | "
      "time per path (s) KD, HASH-G, HASH-B7, CABIN, BUCKET, BF | masks agree | steps with different hash attributes")
allsame = True; ratios = defaultdict(list)
for d in ds:
    ref = tot[(d, "BF-PA")]["pairs"]; nv = tot[(d, "BF-PA")]["naive"]
    row = [f"{d}: {tot[(d, 'KD-PA')]['steps']}"]
    for e in ("HASH-G", "HASH-B7", "CABIN", "BUCKET"):
        t = tot[(d, e)]
        if t["status"] != "ok":
            row.append(f"{e} {t['status']}"); continue
        row.append(f"{e} {t['pairs'] / max(ref, 1):.1f}x ({t['pairs'] / nv:.3f} of naive)")
        ratios[e].append(t["pairs"] / nv)
    row.append("times " + " ".join(f"{tot[(d, e)]['time']:.2f}" if tot[(d, e)]["status"] == "ok" else "T"
                                  for e in ("KD-PA", "HASH-G", "HASH-B7", "CABIN", "BUCKET", "BF-PA")))
    same = all(tot[(d, e)]["same"] for e in ("HASH-G", "HASH-B7", "CABIN", "BUCKET", "BF-PA")
               if tot[(d, e)]["status"] == "ok")
    allsame &= same
    row.append(f"agree {same}; hash attrs differ in {ndiff[d]} steps")
    print(" | ".join(row))
print("all masks agree:", allsame)
for e, v in ratios.items():
    print(f"{e}: pairs / naive median {st.median(v):.4f} range {min(v):.4f}-{max(v):.4f} over {len(v)} datasets")
bg = [(d, tot[(d, "HASH-B7")]["time"] / tot[(d, "HASH-G")]["time"], tot[(d, "HASH-B7")]["pairs"] / max(tot[(d, "HASH-G")]["pairs"], 1))
      for d in ds if tot[(d, "HASH-G")]["status"] == tot[(d, "HASH-B7")]["status"] == "ok" and tot[(d, "HASH-G")]["time"] > 0.05]
print("HASH-B7 / HASH-G along the path (time, pairs):", [(d, round(a, 2), round(b, 2)) for d, a, b in bg])
print("  time ratio median", round(st.median(a for _, a, _ in bg), 2), "range", round(min(a for _, a, _ in bg), 2),
      round(max(a for _, a, _ in bg), 2))
print("\nfull greedy HASH-B7 (fold 1, delta 0.10, 900 s):")
for r in G:
    e = E1.get(r["dataset"], {})
    print(f"  {r['dataset']}: B7 {r['status']} {r.get('time', 0):.1f} s same_as_e1 {r.get('same_as_e1')} | e1 HASH(Gamma) "
          f"{e.get('status')} {e.get('time', 0):.1f} s")

# Table for Appendix A.4 (manuscript/tab_audit.tex)
def pct(x):
    return f"{100 * x:.1f}"


L = [r"\begin{table}[pos=!htbp]", r"\centering",
     r"\caption{Checks of the NRS engines along the greedy path ($\delta=0.10$, fold~1, one evaluation per step on the"
     r" subset and active objects of that step). Left: object pairs passed to the exact distance test, in \% of all"
     r" pairs of an active object and an enemy; ``within $\delta$'' is the share of pairs that are within $\delta$."
     r" Right: total time of the evaluations (seconds). T: exceeded $900$\,s.}",
     r"\label{tab:audit}", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lr rrrr rrrrr}", r"\toprule",
     r" & & \multicolumn{4}{c}{Pairs tested (\%)} & \multicolumn{5}{c}{Time (s)} \\",
     r"\cmidrule(lr){3-6}\cmidrule(lr){7-11}",
     r"Dataset & Steps & HASH & CABIN & BUCKET & within $\delta$ & KD-PA & HASH & CABIN & BUCKET & BF-PA \\",
     r"\midrule"]
for d in ds:
    nv = tot[(d, "BF-PA")]["naive"]
    cells = []
    for e in ("HASH-G", "CABIN", "BUCKET", "BF-PA"):
        t = tot[(d, e)]; cells.append(pct(t["pairs"] / nv) if t["status"] == "ok" else "T")
    times = []
    for e in ("KD-PA", "HASH-G", "CABIN", "BUCKET", "BF-PA"):
        t = tot[(d, e)]; times.append(f"{t['time']:.2f}" if t["status"] == "ok" else "T")
    L.append(f"{d} & {tot[(d, 'KD-PA')]['steps']} & " + " & ".join(cells) + " & " + " & ".join(times) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(HERE, "..", "manuscript", "tab_audit.tex"), "w").write(
    "% generated by revision_R1/analysis/hash_audit_summary.py from code/baselines/hash_audit_path.jsonl\n" + "\n".join(L) + "\n")
print("wrote tab_audit.tex")
fac = {d: tot[(d, "HASH-G")]["time"] / tot[(d, "KD-PA")]["time"] for d in ds}
print("HASH-G / KD-PA time along the path:", {d: round(v, 1) for d, v in fac.items()}, "range", round(min(fac.values()), 1),
      round(max(fac.values()), 1))
print("tables where hash attributes differ in some step:", sum(ndiff[d] > 0 for d in ds), "of", len(ds))
print("CABIN / BUCKET pairs on magic-telescope:", round(tot[("magic-telescope", "CABIN")]["pairs"] /
                                                        tot[("magic-telescope", "BUCKET")]["pairs"], 3))
print("CABIN < BUCKET pairs on every table where both finished:",
      all(tot[(d, "CABIN")]["pairs"] < tot[(d, "BUCKET")]["pairs"] for d in ds if tot[(d, "CABIN")]["status"] == "ok"))
