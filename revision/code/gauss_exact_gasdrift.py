"""Exactness-only check (no time limit) for the Gaussian kernel FRS on gas-drift, kappa = 0.1, fold 1: KD without the
bound, to compare its ordered reduct with the KD+B3 row of gauss_run.jsonl (KD timed out at 900 s there).
Time is recorded but not used as a timing result. Output appended to gauss_exact_gasdrift.jsonl."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "e1")))
from data_e1 import load
from run_e1 import folds
import frs_gauss as G
from gauss_run import EPS
X, y = load("gas-drift"); tr = folds(y)[0][0]; X, y = X[tr], y[tr]
t0 = time.perf_counter(); R, evals, steps = G.run("KD", X, y, 0.1, EPS)
rec = dict(dataset="gas-drift", kappa="0.1", eps=EPS, fold=1, method="KD", purpose="exactness only", R=[int(a) for a in R],
           evals=evals, time=time.perf_counter() - t0)
ref = [json.loads(l) for l in open(os.path.join(HERE, "gauss_run.jsonl"))]
ref = next(r for r in ref if r["dataset"] == "gas-drift" and r["kappa"] == "0.1" and r["fold"] == 1 and r["method"] == "KD+B3")
rec["same_as_KD+B3"] = rec["R"] == ref["R"]
open(os.path.join(HERE, "gauss_exact_gasdrift.jsonl"), "a").write(json.dumps(rec) + "\n")
print(json.dumps({k: rec[k] for k in ("evals", "time", "same_as_KD+B3")}))
