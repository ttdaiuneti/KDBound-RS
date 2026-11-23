"""Fold-wise min-max normalization (revision, R1.12).

The e1 data were min-max normalized to [0, 1] on the whole source table (research-commons standardized/v1). Here each
fold is rescaled with the minimum and maximum of its training part only, and the test part is mapped with the same
affine map (values may leave [0, 1]). Since v1 is itself an affine map of the raw values per attribute, rescaling v1
in this way gives the fold-wise min-max of the raw data (up to rounding). The reduct is computed by KD-PA+B3 on the
rescaled training part (NRS: numerics/engines_rev.py; FRS: e1/frs.py), and accuracy by the same research-commons
protocol (evaluate_reduct: 3-NN and SVM-RBF, defaults, same StratifiedKFold(5, shuffle, seed 0)).
Output: foldnorm_run.jsonl, one line per (dataset, criterion, param, fold).

python foldnorm_run.py [GROUP]  -> all configurations with a GROUP (EXACT or FARA) row in e1_acc.jsonl, 4 at a time
python foldnorm_run.py one SLUG CRIT PARAM [GROUP]
FARA (added for the exact-vs-FARA comparison): e1/fara.py on the rescaled training part. Rows without "group" are EXACT.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "e1"))
NUM = os.path.abspath(os.path.join(HERE, "..", "numerics"))
OUT = os.path.join(HERE, "foldnorm_run.jsonl")
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONWARNINGS="ignore")


def one(slug, crit, param, group="EXACT"):
    sys.path.insert(0, E1); sys.path.insert(0, NUM)
    sys.path.insert(0, os.path.expanduser("~/Documents/research-commons/eval_harness"))
    from protocol import evaluate_reduct
    from data_e1 import load
    X, y = load(slug)

    def reduct_fn(Xtr, ytr):
        lo, hi = Xtr.min(0), Xtr.max(0); r = np.where(hi > lo, hi - lo, 1.0)
        Z = (Xtr - lo) / r
        if group == "FARA":
            from fara import fara
            R = fara(Z, ytr, float(param))[0]
        elif crit == "NRS":
            import engines_rev as REV
            R = REV.run_nrs("KD-PA+B3", Z, ytr, float(param))[0]
        else:
            import frs as F
            R = F.run_frs("KD-PA+B3", Z, ytr, param if param == "std" else float(param))[0]
        if not R:
            R = []
        return Z[:, R], (lambda Xt, R=R: ((Xt - lo) / r)[:, R]), R

    for row in evaluate_reduct(X, y, reduct_fn, slug, f"FOLDNORM-{group}|{crit}|{param}", seed=0,
                               git_commit_of_method="KDBound-RS/revision_R1"):
        print(json.dumps(dict(row, dataset=slug, criterion=crit, param=param, group=group), default=float), flush=True)


def main(group="EXACT"):
    todo = sorted({(r["dataset"], r["criterion"], r["param"]) for r in map(json.loads, open(os.path.join(E1, "e1_acc.jsonl")))
                   if r.get("group") == group and r.get("status") == "ok"})
    have = set()
    if os.path.exists(OUT):
        have = {(r["dataset"], r["criterion"], r["param"]) for r in map(json.loads, open(OUT))
                if r.get("group", "EXACT") == group}
    todo = [k for k in todo if k not in have]
    print(len(todo), "configurations", flush=True)

    def work(k):
        p = subprocess.run([sys.executable, __file__, "one", *k, group], capture_output=True, text=True, env=ENV)
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
        with open(OUT, "a") as f:
            if len(lines) == 5:
                f.write("\n".join(lines) + "\n")
            else:
                f.write(json.dumps(dict(dataset=k[0], criterion=k[1], param=k[2], group=group, status="error",
                                        err=p.stderr[-400:])) + "\n")
        print(k, len(lines), flush=True)
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(work, todo))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        one(*sys.argv[2:6])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "EXACT")
