"""Exactness check for frs.py: all six methods must return the identical attribute sequence (K1 for FRS),
on stratified 2000-row samples of five commons datasets, sigma in {0.25, 'std'}. Output: verify_frs.log"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frs import METHODS, run_frs  # noqa: E402

C = os.path.expanduser("~/Documents/research-commons/datasets")
ok_all = True
for ds in ("magic-telescope", "jannis", "htru2", "electrical-grid-stability", "creditcard"):
    df = pd.read_csv(os.path.join(C, ds, "standardized", "v1", "data.csv"))
    X, y = df.drop(columns="label").to_numpy(float), df["label"].to_numpy()
    idx = train_test_split(np.arange(len(y)), train_size=2000, stratify=y, random_state=0)[0]
    X, y = X[idx], y[idx]
    for sigma in (0.25, "std"):
        res = {}
        for m in METHODS:
            t = time.perf_counter(); R, ev, pr = run_frs(m, X, y, sigma)
            res[m] = (R, ev, time.perf_counter() - t)
        ref = res["BF-DAR"][0]; same = all(r[0] == ref for r in res.values()); ok_all &= same
        print(f"{ds:26s} sigma={sigma!s:4s} |R|={len(ref):3d} same={same} " +
              " ".join(f"{m}:{r[1]}ev/{r[2]:.1f}s" for m, r in res.items()), flush=True)
        if not same:
            for m, r in res.items():
                print(f"    {m:10s} {r[0]}", flush=True)
print("ALL IDENTICAL" if ok_all else "MISMATCH", flush=True)
