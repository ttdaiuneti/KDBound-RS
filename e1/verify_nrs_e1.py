"""Exactness check for engines_e1.py: the 12 exact NRS methods (6 engines x {no bound, B3}) must return the identical
attribute sequence; FARA reported separately. 2000-row stratified samples, delta in {0.05, 0.2}. Output: verify_nrs_e1.log"""
import os, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engines_e1 import NRS_METHODS, run_nrs  # noqa: E402
C = os.path.expanduser("~/Documents/research-commons/datasets"); ok_all = True
for ds in ("magic-telescope", "jannis", "htru2", "electrical-grid-stability", "creditcard", "gas-drift"):
    df = pd.read_csv(os.path.join(C, ds, "standardized", "v1", "data.csv"))
    X, y = df.drop(columns="label").to_numpy(float), df["label"].to_numpy()
    i = train_test_split(np.arange(len(y)), train_size=2000, stratify=y, random_state=0)[0]; X, y = X[i], y[i]
    for delta in (0.05, 0.2):
        res = {}
        for m in NRS_METHODS:
            t = time.perf_counter(); R, ev, st = run_nrs(m, X, y, delta); res[m] = (R, ev, time.perf_counter() - t)
        ref = res["KD-PA"][0]; same = all(r[0] == ref for m, r in res.items() if m != "FARA"); ok_all &= same
        print(f"{ds:26s} delta={delta:.2f} |R|={len(ref):3d} exact-same={same} FARA|R|={len(res['FARA'][0])} " +
              " ".join(f"{m}:{r[1]}" for m, r in res.items()), flush=True)
        if not same:
            for m, r in res.items(): print(f"    {m:10s} {r[0]}", flush=True)
print("ALL EXACT IDENTICAL" if ok_all else "MISMATCH", flush=True)
