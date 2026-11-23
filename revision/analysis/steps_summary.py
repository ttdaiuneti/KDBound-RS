"""R1.6, R1.10, R1.11 from code/steps/steps_run.jsonl (KD-PA+B3 with per-step records).

R1.6  first step: share of evaluations (m / total) and of reduction time (step-1 time / total).
R1.10 FRS threshold: from the tau = 0 path, the reduct for tau in TAUS is the prefix before the first step whose best
      gain is <= tau; report how many folds change relative to tau = 1e-9 and by how many attributes.
R1.11 NRS radius: per delta, active share |Act_t| / n over steps, skipped share, and bound tightness = gain / bound of
      the evaluated candidates (steps >= 2).
For FRS, R1.6 and R1.11 use the prefix for tau = 1e-9 (the setting of the paper)."""
import json, os, statistics as st
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "..", "code", "steps", "steps_run.jsonl"))]
rows = [r for r in rows if "steps" in r]
TAUS = (0.0, 1e-12, 1e-9, 1e-6, 1e-4)
print("units:", len(rows), " partial:", sum(r["partial"] for r in rows),
      " reduct differs from e1:", [(r["criterion"], r["dataset"], r["param"], r["fold"]) for r in rows if r["same_as_e1"] is False])


def cut(steps, tau):
    out = []
    for i, s in enumerate(steps):
        if s["selected"] is None or s["gain"] <= tau:
            return i + 1, out          # number of steps executed, reduct
        out.append(s["selected"])
    return len(steps), out


# R1.6
for crit in ("NRS", "FRS"):
    ev_share, t_share = [], []
    for r in rows:
        if r["criterion"] != crit or r["partial"]:
            continue
        ns = cut(r["steps"], 1e-9)[0] if crit == "FRS" else len(r["steps"])
        S = r["steps"][:ns]
        tot_ev = sum(s["evaluated"] for s in S); tot_t = sum(s["time"] for s in S)
        if tot_t > 0.5:                                      # ignore runs below 0.5 s
            ev_share.append(S[0]["evaluated"] / tot_ev); t_share.append(S[0]["time"] / tot_t)
    if t_share:
        print(f"\nR1.6 {crit}: {len(t_share)} folds (> 0.5 s); first-step share of evaluations median {st.median(ev_share):.2f} "
              f"(range {min(ev_share):.2f}-{max(ev_share):.2f}); of time median {st.median(t_share):.2f} "
              f"(range {min(t_share):.2f}-{max(t_share):.2f})")

# R1.10
fr = [r for r in rows if r["criterion"] == "FRS"]
if fr:
    print(f"\nR1.10 FRS: {len(fr)} folds")
    base = {id(r): cut(r["steps"], 1e-9)[1] for r in fr}
    for tau in TAUS:
        ch = [(len(cut(r["steps"], tau)[1]) - len(base[id(r)])) for r in fr if not r["partial"]]
        pre = all(cut(r["steps"], tau)[1][:len(base[id(r)])] == base[id(r)] or len(cut(r["steps"], tau)[1]) < len(base[id(r)])
                  for r in fr if not r["partial"])
        print(f"   tau={tau:g}: folds with a different reduct than tau=1e-9: {sum(c != 0 for c in ch)}; "
              f"size change range {min(ch)}..{max(ch)}; prefix-consistent: {pre}")
    small = [s["gain"] for r in fr for s in r["steps"] if s["selected"] is not None and 0 < s["gain"] <= 1e-6]
    print(f"   selected steps with 0 < gain <= 1e-6: {len(small)}")

# R1.11
nr = [r for r in rows if r["criterion"] == "NRS" and not r["partial"]]
if nr:
    print("\nR1.11 NRS by delta (steps >= 2):")
    for d in sorted({r["param"] for r in nr}):
        act, skip, tight = [], [], []
        for r in nr:
            if r["param"] != d:
                continue
            n = r["n_train"]
            for s in r["steps"][1:]:
                act.append(s["active"] / n); skip.append(1 - s["evaluated"] / s["candidates"])
                tight += [g / ub for _, ub, g in s["ev"] if ub]
        print(f"   delta={d}: active share median {st.median(act):.3f}; skipped share median {st.median(skip):.3f} "
              f"(mean {st.mean(skip):.3f}); gain/bound of evaluated candidates median {st.median(tight):.3f}")
