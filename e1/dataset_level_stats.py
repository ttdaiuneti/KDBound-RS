"""Dataset-level statistics for the bound (reviewer point, 2026-10-02): configurations of one dataset are not
independent, so the paired test of KD-PA vs KD-PA+B3 is done on one value per dataset (mean log time ratio over its
usable configurations). Configurations where KD-PA did not finish enter with a rigorous lower bound: folds that ran
count with their time, folds that timed out with 900 s; if only fold 1 ran (skip rule), the fold-1 ratio is used.
Prints the numbers used in Section 5.4-5.5. Source: e1_raw.jsonl (non-superseded), e1_summary.csv."""
import csv, glob, json, math, os, statistics as st, sys
from collections import defaultdict
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_e1 import superseded, is_superseded, LIMIT

HERE = os.path.dirname(os.path.abspath(__file__))
M = {os.path.basename(f)[:-4]: np.load(f)['X'].shape[1] for f in glob.glob(os.path.join(HERE, 'data', '*.npz'))}
sup = superseded(); R = {}
for r in map(json.loads, open(os.path.join(HERE, 'e1_raw.jsonl'))):
    if not is_superseded(r, sup):
        R[(r['dataset'], r['criterion'], r['param'], r['method'], r['fold'])] = r


def ratio(d, c, p, a, b):
    """(log ratio time(a)/time(b), censored?) or None; b must finish all folds."""
    fb = [R.get((d, c, p, b, f)) for f in range(1, 6)]
    if not all(x and x['status'] == 'ok' for x in fb):
        return None
    fa = [R.get((d, c, p, a, f)) for f in range(1, 6)]
    if all(x and x['status'] == 'ok' for x in fa):
        return math.log(st.mean(x['time'] for x in fa) / st.mean(x['time'] for x in fb)), False
    ran = [f for f in range(5) if fa[f] and fa[f]['status'] in ('ok', 'timeout')]
    if not ran or any(fa[f]['status'] not in ('ok', 'timeout') for f in ran):
        return None
    ta = st.mean(fa[f]['time'] if fa[f]['status'] == 'ok' else LIMIT for f in ran)
    return math.log(ta / st.mean(fb[f]['time'] for f in ran)), True


usable = [r for r in csv.DictReader(open(os.path.join(HERE, 'e1_summary.csv'))) if r['degenerate'] in ('', 'False', '0')]
for a, b, crits in (('KD-PA', 'KD-PA+B3', ('NRS', 'FRS')), ('FNC', 'FNC+B3', ('NRS',))):
    per = defaultdict(list); cens = []
    for r in usable:
        if r['criterion'] not in crits:
            continue
        x = ratio(r['dataset'], r['criterion'], r['param'], a, b)
        if x:
            per[r['dataset']].append(x[0])
            if x[1]:
                cens.append((r['dataset'], r['criterion'], r['param'], round(math.exp(x[0]), 2)))
    v = {d: st.mean(l) for d, l in per.items()}
    w = stats.wilcoxon(list(v.values()), alternative='greater')
    sp = stats.spearmanr([M[d] for d in v], list(v.values()))
    print(f"{b} vs {a}: {len(v)} datasets, {sum(map(len, per.values()))} configurations ({len(cens)} censored {cens}); "
          f"one-sided Wilcoxon p={w.pvalue:.3f}; Spearman(m) rho={sp.correlation:.2f} p={sp.pvalue:.4f}")
    lo = [x for d, x in v.items() if M[d] <= 16]; hi = [x for d, x in v.items() if M[d] >= 29]
    print(f"   m<=16: {len(lo)} datasets, geometric mean {math.exp(st.mean(lo)):.2f}; "
          f"m>=29: {len(hi)} datasets, geometric mean {math.exp(st.mean(hi)):.2f} "
          f"{sorted((d, round(math.exp(x), 2)) for d, x in v.items() if M[d] >= 29)}" if hi else "")
    for crit in crits if len(crits) > 1 else ():
        pc = defaultdict(list)
        for r in usable:
            if r['criterion'] == crit:
                x = ratio(r['dataset'], crit, r['param'], a, b)
                if x:
                    pc[r['dataset']].append(x[0])
        vc = {d: st.mean(l) for d, l in pc.items()}
        print(f"   {crit}: {len(vc)} datasets, Wilcoxon p={stats.wilcoxon(list(vc.values()), alternative='greater').pvalue:.3f}, "
              f"Spearman rho={stats.spearmanr([M[d] for d in vc], list(vc.values())).correlation:.2f} "
              f"p={stats.spearmanr([M[d] for d in vc], list(vc.values())).pvalue:.4f}")
