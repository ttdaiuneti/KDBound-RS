"""Generate manuscript/tab_gauss.tex from code/gauss_run.jsonl (Gaussian kernel FRS, KD vs KD+B3)."""
import json, os, statistics as st
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "..", "code", "gauss_run.jsonl"))]
ORDER = ("mammography", "run-or-walk", "htru2", "magic-telescope", "fried", "internet-firewall",
         "electrical-grid-stability", "house-16h", "hls4ml-jets", "creditcard", "miniboone", "jannis", "vehicle-sensit",
         "gas-drift")  # by number of attributes (2026-11-24: 8 tables added, docstring amendment of gauss_run.py)
LIMIT = 900
by = defaultdict(dict)
for r in rows:
    by[(r["dataset"], r["kappa"])][(r["fold"], r["method"])] = r
meta = {r["dataset"]: (r["n_train"], r["m"]) for r in rows if "m" in r}
for ds in ORDER:  # tables where no unit finished carry no m in their records: read it from the data file
    if ds not in meta:
        import sys; sys.path.insert(0, os.path.join(HERE, "..", "..", "e1"))
        from data_e1 import load
        meta[ds] = (None, load(ds)[0].shape[1])

def ok(d, meth):
    return [d[(f, meth)] for f in range(1, 6) if (f, meth) in d and d[(f, meth)]["status"] == "ok"]

def tcell(v):
    if len(v) < 5:
        return f"$>{LIMIT}$" if not v else f"{len(v)}/5 folds"
    return f"${st.mean(v):.1f}\\pm{st.stdev(v):.1f}$"

lines = []
for ds in ORDER:
    for k in ("0.1", "0.2"):
        d = by.get((ds, k))
        if not d:
            continue
        n, m = meta.get(ds, (None, None))
        b, kk = ok(d, "KD+B3"), ok(d, "KD")
        R = f"${st.mean(len(x['R']) for x in b):.1f}$" if b else "--"
        eb = f"${st.mean(x['evals'] for x in b):.1f}$" if b else "--"
        ek = f"${st.mean(x['evals'] for x in kk):.1f}$" if len(kk) == 5 else "--"
        ratio = (f"${st.mean(x['time'] for x in kk) / st.mean(x['time'] for x in b):.2f}$"
                 if len(b) == 5 and len(kk) == 5 else "--")
        name = ds if k == "0.1" else ""
        mm = f"${m}$" if k == "0.1" else ""
        lines.append(f"{name} & {mm} & ${k}$ & {R} & ${m*(m+1)//2}$ & {ek} & {eb} & {tcell([x['time'] for x in kk])} & "
                     f"{tcell([x['time'] for x in b])} & {ratio} \\\\")
out = r"""\begin{table}[pos=htbp]
\centering
\caption{Gaussian kernel FRS: reduct size, evaluations per fold and reduction time (s, mean $\pm$ std over five
folds) of KD-PA without and with the bound. Full: $m(m+1)/2$ evaluations. Ratio: time of KD-PA over time of
KD-PA+B3.}\label{tab:gauss}
\small
\begin{tabular}{llrrrrrrrr}
\toprule
 & & & & \multicolumn{3}{c}{Evaluations} & \multicolumn{2}{c}{Time (s)} & \\
\cmidrule(lr){5-7}\cmidrule(lr){8-9}
Dataset & $m$ & $\kappa$ & $|R|$ & Full & KD-PA & +B3 & KD-PA & KD-PA+B3 & Ratio \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(HERE, "..", "manuscript", "tab_gauss.tex"), "w").write(out)
print(out)
