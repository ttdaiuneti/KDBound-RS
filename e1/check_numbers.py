"""Recompute the numbers stated in the prose of Section 5 that no table/figure script prints (number-consistency
check before submission). Each line: claim id (as in the review sweep), the manuscript value, the recomputed value.
Sources: e1_raw.jsonl (non-superseded), e1_summary.csv, e1_acc.jsonl, remeasure_idle.jsonl, results/reducts."""
import csv, glob, json, math, os, statistics as st, sys
from collections import defaultdict
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_e1 import superseded, is_superseded, LIMIT  # noqa: E402

M = {os.path.basename(f)[:-4]: np.load(f)['X'].shape[1] for f in glob.glob(os.path.join(HERE, 'data', '*.npz'))}
sup = superseded(); R = {}
for r in map(json.loads, open(os.path.join(HERE, 'e1_raw.jsonl'))):
    if not is_superseded(r, sup):
        R[(r['dataset'], r['criterion'], r['param'], r['method'], r['fold'])] = r
S = list(csv.DictReader(open(os.path.join(HERE, 'e1_summary.csv'))))
U = [r for r in S if r['degenerate'] in ('', 'False', '0')]


def show(cid, text, val):
    print(f"[{cid:>3}] {text:62s} -> {val}")


def full(d, c, p, m):
    f = [R.get((d, c, p, m, k)) for k in range(1, 6)]
    return f if all(x and x['status'] == 'ok' for x in f) else None


def mean5(d, c, p, m):
    f = full(d, c, p, m); return st.mean(x['time'] for x in f) if f else None


# [35] idle re-measurement
idle = [json.loads(l) for l in open(os.path.join(HERE, 'remeasure_idle.jsonl'))]
rat = []
for r in idle:
    e = R.get((r['dataset'], r['criterion'], r['param'], r['method'], 1))
    if r['status'] == 'ok' and e and e['status'] == 'ok' and e['time'] > 1 and r['dataset'] != 'vehicle-sensit':
        rat.append(r['time'] / e['time'])
show(35, 'idle/concurrent: 20 units, 0.85-0.98, median 0.90',
     f"{len(rat)} units, {min(rat):.2f}-{max(rat):.2f}, median {st.median(rat):.2f}")
vs = {(r['param'], r['method']): (r['status'], round(r.get('time', 0))) for r in idle if r['dataset'] == 'vehicle-sensit'}
show(96, 'vehicle-sensit alone KD-PA+B3: 443 s (0.10), 735 s (0.15), not 0.20',
     {k: v for k, v in vs.items() if k[1] == 'KD-PA+B3'})

# [44]-[45] speed-ups below two
sp = sorted((float(r['speedup_vs_best']), r['dataset'], r['criterion'], r['param'], r['best_existing'],
             r['speedup_is_lower_bound']) for r in U if r['speedup_vs_best'] not in ('', 'nan'))
show(44, 'below 2x: creditcard 0.05 (0.78), creditcard 0.10 (1.3), htru2 0.05 (1.6), gas FRS std (1.98)',
     [(round(s, 2), d, c, p) for s, d, c, p, *_ in sp if s < 2])
show(44, 'creditcard NRS 0.05: ours 2.2 s, FNC 1.7 s, FNC+B3 1.0 s',
     [round(mean5('creditcard', 'NRS', '0.05', m), 2) for m in ('KD-PA+B3', 'FNC', 'FNC+B3')])
show(45, 'htru2 NRS 0.05 reduction takes 0.2 s', round(mean5('htru2', 'NRS', '0.05', 'KD-PA+B3'), 2))

# [54] engine part
eng = []
for r in U:
    k = mean5(r['dataset'], r['criterion'], r['param'], 'KD-PA')
    if k and r['best_mean'] not in ('', 'nan') and r['speedup_is_lower_bound'] in ('0', 'False', ''):
        eng.append((float(r['best_mean']) / k, r['dataset'], r['criterion'], r['param']))
show(54, 'engine part median 18.3 over 48 configs', f"median {st.median(e for e, *_ in eng):.1f} over {len(eng)}")
show(54, 'exceptions <1: creditcard d<=0.15 0.24-0.82, gas 0.10/0.15 0.90/0.64',
     sorted((round(e, 2), d, c, p) for e, d, c, p in eng if e < 1))

# [61], [63], [92], [93] per-configuration bound part
bp = []
for r in U:
    a, b = full(r['dataset'], r['criterion'], r['param'], 'KD-PA'), full(r['dataset'], r['criterion'], r['param'], 'KD-PA+B3')
    if a and b:
        bp.append((st.mean(x['time'] for x in a) / st.mean(x['time'] for x in b),
                   sum(x['evals'] for x in b) / sum(x['evals'] for x in a), M[r['dataset']], r['dataset'], r['param']))
lo = [x for x in bp if x[2] <= 16]; hi = [x for x in bp if x[2] >= 29]
show(53, '63 configurations where KD-PA finished all folds', len(bp))
show(61, 'm<=16: median 0.94, 0.67-1.99, <1 in 25 of 44',
     f"median {st.median(x[0] for x in lo):.2f}, {min(x[0] for x in lo):.2f}-{max(x[0] for x in lo):.2f}, "
     f"<1 in {sum(x[0] < 1 for x in lo)} of {len(lo)}")
show(61, 'm>=29: median 3.31, 1.29-13.22, 19 configs',
     f"median {st.median(x[0] for x in hi):.2f}, {min(x[0] for x in hi):.2f}-{max(x[0] for x in hi):.2f}, {len(hi)}")
show(92, 'worst ratio 0.67 on hls4ml-jets 0.10', min((round(x[0], 2), x[3], x[4]) for x in bp))
show(93, 'm>=29 skips 32-92% of evaluations',
     f"{100 * (1 - max(x[1] for x in hi)):.0f}-{100 * (1 - min(x[1] for x in hi)):.0f}%")
show(93, 'm>=29 median skipped fraction per configuration (abstract)',
     f"{100 * (1 - st.median(x[1] for x in hi)):.0f}% over {len(hi)} configs")
show(63, 'Spearman(m, frac) -0.72; frac 1.00 run-or-walk .. 0.08 gas; Spearman(ratio,frac) -0.96',
     f"{stats.spearmanr([x[2] for x in bp], [x[1] for x in bp]).correlation:.2f}; "
     f"max {max((round(x[1], 2), x[3]) for x in bp)}, min {min((round(x[1], 2), x[3]) for x in bp)}; "
     f"{stats.spearmanr([x[0] for x in bp], [x[1] for x in bp]).correlation:.2f}")

# [75], [76] evaluation counts across engines (NRS 0.10)
same = tot = 0; diff = []
for d in M:
    for k in range(1, 6):
        ev = {m: R[(d, 'NRS', '0.10', m, k)]['evals'] for m in ('KD-PA+B3', 'FNC+B3', 'CABIN+B3', 'BUCKET+B3', 'BF-PA+B3')
              if (d, 'NRS', '0.10', m, k) in R and R[(d, 'NRS', '0.10', m, k)]['status'] == 'ok'}
        if len(ev) >= 2:
            tot += 1; same += len(set(ev.values())) == 1
            if len(set(ev.values())) > 1:
                diff.append((d, k, ev))
show(75, 'nearest-witness engines: same counts in 125 of 126 folds', f"{same} of {tot} {diff}")
show(76, 'gas-drift fold 1 HASH+B3 2951 vs KD-PA+B3 640',
     [R[('gas-drift', 'NRS', '0.10', m, 1)].get('evals') for m in ('HASH+B3', 'KD-PA+B3')])

# [80], [81] pair-enumerating engines
pr = []
for d in M:
    for e in ('HASH', 'CABIN', 'BUCKET', 'BF-PA'):
        a, b = mean5(d, 'NRS', '0.10', e), mean5(d, 'NRS', '0.10', e + '+B3')
        if a and b:
            pr.append((a / b, d, e))
show(80, '20 combos, median 1.03, 0.89-3.88, slower in 7 (all m<=16)',
     f"{len(pr)}, median {st.median(x[0] for x in pr):.2f}, {min(x[0] for x in pr):.2f}-{max(x[0] for x in pr):.2f}, "
     f"slower {sum(x[0] < 1 for x in pr)} (max m {max([M[x[1]] for x in pr if x[0] < 1] or [0])})")
show(81, 'gas-drift pair engines +B3 finished in 347-495 s',
     [round(mean5('gas-drift', 'NRS', '0.10', e + '+B3') or -1) for e in ('HASH', 'CABIN', 'BUCKET', 'BF-PA')])

# [83]-[85] memory
mem = [(d, e) for d in M for e in ('HASH', 'CABIN', 'BUCKET', 'BF-PA')
       if R.get((d, 'NRS', '0.10', e + '+B3', 1), {}).get('status') == 'memout']
fin = [(d, e) for d, e in mem if mean5(d, 'NRS', '0.10', e)]
show(83, '12 memout combos; 7 where engine w/o bound finished (fw, row, hls4ml)',
     f"{len(mem)}; {len(fin)} {sorted({d for d, _ in fin})}")

# [87]-[91] accuracy and FARA
A = [r for r in U if r['criterion'] == 'NRS' and r['EXACT_svm_mean'] not in ('', 'nan') and r['FARA_svm_mean'] not in ('', 'nan')]
for key, cid, txt in (('svm', 87, 'SVM: FARA higher 5, lower 6, equal 30, p=0.64'), ('3nn', 87, '3NN: higher 6, lower 1, equal 34, p=0.10')):
    dd = [float(r[f'FARA_{key}_mean']) - float(r[f'EXACT_{key}_mean']) for r in A]
    hi_, lo_ = sum(x > 0.0005 for x in dd), sum(x < -0.0005 for x in dd)
    nz = [x for x in dd if abs(x) > 1e-12]
    show(cid, txt, f"n={len(dd)} higher {hi_} lower {lo_} equal {len(dd) - hi_ - lo_} p={stats.wilcoxon(nz).pvalue:.2f}")
    per = defaultdict(list)
    for r, x in zip(A, dd):
        per[r['dataset']].append(x)
    v = [st.mean(l) for l in per.values()]
    show(cid, f'  dataset level ({key})', f"{len(v)} datasets, higher {sum(x > 0.0005 for x in v)}, lower "
         f"{sum(x < -0.0005 for x in v)}, p={stats.wilcoxon([x for x in v if abs(x) > 1e-12]).pvalue:.2f}")
j = [r for r in A if r['dataset'] == 'jannis']
show(89, 'jannis FARA +3.7-7.6 SVM, +2.7-6.0 3NN',
     [(r['param'], round(100 * (float(r['FARA_svm_mean']) - float(r['EXACT_svm_mean'])), 1),
       round(100 * (float(r['FARA_3nn_mean']) - float(r['EXACT_3nn_mean'])), 1)) for r in j])
g = [r for r in A if r['dataset'] == 'gas-drift']
show(88, 'gas-drift FARA SVM lower by up to 2.2', [round(100 * (float(r['FARA_svm_mean']) - float(r['EXACT_svm_mean'])), 1) for r in g])
ft = []
for r in U:
    if r['criterion'] != 'NRS':
        continue
    a, b = mean5(r['dataset'], 'NRS', r['param'], 'FARA'), mean5(r['dataset'], 'NRS', r['param'], 'KD-PA+B3')
    if a and b:
        ft.append((a / b, r['dataset'], r['param']))
show(90, 'FARA faster in 10 of 41; 1.7x on hls4ml 0.10; median ratio 1.45',
     f"{sum(x[0] < 1 for x in ft)} of {len(ft)}; hls4ml 0.10 {1 / [x[0] for x in ft if x[1] == 'hls4ml-jets' and x[2] == '0.10'][0]:.2f}; "
     f"median {st.median(x[0] for x in ft):.2f}")
show(91, 'FARA 207/19.9 gas, 174/27.5 mini, 50/33 jannis (0.10)',
     [(d, round(mean5(d, 'NRS', '0.10', 'FARA') or -1), round(mean5(d, 'NRS', '0.10', 'KD-PA+B3'), 1))
      for d in ('gas-drift', 'miniboone', 'jannis')])

# [5.3] against the standard greedy procedure with positive approximation
for c, p, base in (('NRS', '0.10', 'BF-PA'), ('FRS', '0.25', 'FA-FPR')):
    v = []; to = 0
    for d in M:
        o = mean5(d, c, p, 'KD-PA+B3')
        if o is None or (d, c, p, base, 1) not in R:
            continue
        b = mean5(d, c, p, base)
        if b:
            v.append(b / o)
        elif R[(d, c, p, base, 1)]['status'] in ('timeout', 'memout'):
            v.append(LIMIT / o); to += 1
    show('5.3', f'vs {base}: median 58 (NRS, 11) / 64 (FRS, 12); over 900 s on 6 / 5',
         f"{len(v)} datasets, median {st.median(v):.1f}, over limit {to}")

# [A1] probes on tables not used (Appendix A.1; e1/probes_hd/)
pr = [json.loads(l) for l in open(os.path.join(HERE, 'probes_hd', 'scale_m_probe.jsonl'))]
nrs = {(r['m'], r['method']): r for r in pr if r['crit'] == 'NRS' and r['status'] == 'ok'}
ms = sorted({m for m, _ in nrs})
show('A1', 'dilbert NRS: |R| 11-13, evaluated 59-63%, speed-up 1.16-1.42',
     f"|R| {min(nrs[(m, 'KD-PA')]['R'] for m in ms)}-{max(nrs[(m, 'KD-PA')]['R'] for m in ms)}, "
     f"evaluated {min(nrs[(m, 'KD-PA+B3')]['evals'] / nrs[(m, 'KD-PA')]['evals'] for m in ms):.2f}-"
     f"{max(nrs[(m, 'KD-PA+B3')]['evals'] / nrs[(m, 'KD-PA')]['evals'] for m in ms):.2f}, speed-up "
     f"{min(nrs[(m, 'KD-PA')]['time'] / nrs[(m, 'KD-PA+B3')]['time'] for m in ms):.2f}-"
     f"{max(nrs[(m, 'KD-PA')]['time'] / nrs[(m, 'KD-PA+B3')]['time'] for m in ms):.2f}")
import re  # noqa: E402
sa = {}
for f in ('degen_hd2.out', 'probe_sant_kdpa.out'):
    for ln in open(os.path.join(HERE, 'probes_hd', f)):
        mm = re.search(r'Santander\S* NRS (?:(KD-PA) )?n=\d+ m=200 \|R\|=(\d+) evals=(\d+).* t=([\d.]+)s', ln)
        if mm:
            sa[mm.group(1) or 'KD-PA+B3'] = (int(mm.group(2)), int(mm.group(3)), float(mm.group(4)))
show('A1', 'Santander NRS: |R| 11, evaluated 74%, 1.15x (210 s vs 240 s)',
     f"|R| {sa['KD-PA+B3'][0]}, evaluated {sa['KD-PA+B3'][1] / sa['KD-PA'][1]:.2f}, "
     f"{sa['KD-PA'][2] / sa['KD-PA+B3'][2]:.2f}x ({sa['KD-PA+B3'][2]:.0f} s vs {sa['KD-PA'][2]:.0f} s)")
