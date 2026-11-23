"""Pre-check for the Gaussian-kernel FRS run: (1) f(NE) equals Hu's formula; (2) four methods give the same ordered
reduct on stratified samples of 2000 objects (training part of fold 1); (3) reduct length for eps/d choices."""
import sys, os, time
import numpy as np
from sklearn.model_selection import train_test_split
E1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "e1")); sys.path.insert(0, E1)
from data_e1 import load
from run_e1 import folds
import frs_gauss as G

# (1) formula check on a tiny random table
rng = np.random.default_rng(0); X = rng.random((60, 4)); y = rng.integers(0, 3, 60); d = 0.2; B = [0, 2, 3]
hu = [min(np.sqrt(1 - np.exp(-np.sum((X[i, B] - X[j, B]) ** 2) / (2 * d * d)) ** 2) for j in range(60) if y[j] != y[i]) for i in range(60)]
dsq, _ = G.BF(X, y).ne(B, np.arange(60))
print("formula max abs diff:", np.max(np.abs(np.array(hu) - G.f_of(dsq, d))))

for slug in sys.argv[1:]:
    Xf, yf = load(slug); tr = folds(yf)[0][0]; Xf, yf = Xf[tr], yf[tr]
    print(slug, "range", Xf.min().round(3), Xf.max().round(3), "m", Xf.shape[1])
    i = train_test_split(np.arange(len(yf)), train_size=1000, stratify=yf, random_state=0)[0]
    X, y = Xf[i], yf[i]
    for d in (0.1, 0.2):
        sys.stdout.flush()
        for eps in (1e-3, 1e-2):
            res = {}
            for meth in G.METHODS:
                t = time.perf_counter(); R, ev, st = G.run(meth, X, y, d, eps); res[meth] = (R, ev, time.perf_counter() - t)
            same = len({tuple(r[0]) for r in res.values()}) == 1
            print(f"  d={d} eps={eps}: |R|={len(res['KD'][0])} same={same} evals KD={res['KD'][1]} KD+B3={res['KD+B3'][1]} "
                  f"t KD={res['KD'][2]:.1f}s KD+B3={res['KD+B3'][2]:.1f}s")
