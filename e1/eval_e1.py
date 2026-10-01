"""E1 accuracy (e1_prereg.md, section 7): research-commons eval_harness/protocol.py::evaluate_reduct, default 3-NN
and SVM-RBF, the same StratifiedKFold(5, shuffle=True, random_state=0) split as the timing runs.

For every locked dataset and every (criterion, param):
- group EXACT: the reduct of the exact methods for each fold (K1: all finished exact methods must agree; a
  disagreement is flagged in e1_k1.jsonl and that group is not evaluated);
- group FARA: FARA's own reduct for each fold.
A group is evaluated only when all 5 folds have a non-empty reduct; otherwise one 'not-evaluated' row with the reason.
Output: e1_acc.jsonl (append-only; groups already present are skipped). time_reduct_sec in these rows is a lookup
time and is NOT used; reduction times come from e1_raw.jsonl.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser("~/Documents/research-commons/eval_harness"))
from protocol import evaluate_reduct  # noqa: E402

from data_e1 import load  # noqa: E402

RAW, ACC, K1 = (os.path.join(HERE, f) for f in ("e1_raw.jsonl", "e1_acc.jsonl", "e1_k1.jsonl"))


def main(only=None):
    locked = [ln.split("\t")[0] for ln in open(os.path.join(HERE, "e1_locked.txt"))] \
        if os.path.exists(os.path.join(HERE, "e1_locked.txt")) else []
    if only:                                                          # parallel runs: python eval_e1.py slug [slug ...]
        locked = [s for s in locked if s in only]
    from run_e1 import superseded, is_superseded
    sup = superseded()
    rows = [r for r in (json.loads(ln) for ln in open(RAW)) if not is_superseded(r, sup)] if os.path.exists(RAW) else []
    have = {(r["dataset"], r["criterion"], r["param"], r["group"]) for r in map(json.loads, open(ACC))} \
        if os.path.exists(ACC) else set()
    for slug in locked:
        X, y = load(slug)
        for crit, param in sorted({(r["criterion"], r["param"]) for r in rows if r["dataset"] == slug}):
            sel = [r for r in rows if r["dataset"] == slug and r["criterion"] == crit and r["param"] == param
                   and r["status"] == "ok"]
            groups = {"EXACT": {}, "FARA": {}}
            k1_bad = []
            for k in range(1, 6):
                ex = {r["method"]: tuple(r["R"]) for r in sel if r["fold"] == k and r["method"] != "FARA"}
                if len(set(ex.values())) > 1:
                    k1_bad.append(dict(fold=k, reducts={m: list(v) for m, v in ex.items()}))
                if ex:
                    groups["EXACT"][k] = list(next(iter(ex.values())))
                fa = [r["R"] for r in sel if r["fold"] == k and r["method"] == "FARA"]
                if fa:
                    groups["FARA"][k] = fa[0]
            if k1_bad:
                with open(K1, "a") as f:
                    f.write(json.dumps(dict(dataset=slug, criterion=crit, param=param, mismatches=k1_bad)) + "\n")
            if os.environ.get("EVAL_PARAMS") and param not in os.environ["EVAL_PARAMS"].split(","):
                continue                                              # split one dataset across processes
            for g, red in groups.items():
                have = {(r["dataset"], r["criterion"], r["param"], r["group"]) for r in map(json.loads, open(ACC))} \
                    if os.path.exists(ACC) else set()                 # re-read: parallel processes share ACC
                if (slug, crit, param, g) in have or (g == "FARA" and crit == "FRS"):
                    continue
                base = dict(dataset=slug, criterion=crit, param=param, group=g)
                reason = "K1 mismatch" if g == "EXACT" and k1_bad else \
                    "missing folds" if len(red) < 5 else "empty reduct" if any(not red[k] for k in red) else None
                with open(ACC, "a") as f:
                    if reason:
                        f.write(json.dumps(dict(base, status="not-evaluated", reason=reason)) + "\n")
                        continue
                    it = iter(range(1, 6))

                    def reduct_fn(Xtr, ytr):
                        R = red[next(it)]
                        return Xtr[:, R], (lambda Xt, R=R: Xt[:, R]), R
                    for row in evaluate_reduct(X, y, reduct_fn, slug, f"{g}|{crit}|{param}", seed=0,
                                               git_commit_of_method="KDBound-RS/e1"):
                        f.write(json.dumps(dict(base, status="ok", **row)) + "\n")
                print(f"{slug} {crit} {param} {g}: {reason or 'evaluated'}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
