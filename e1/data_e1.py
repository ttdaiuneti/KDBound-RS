"""E1 inputs: commons standardized/v1, capped at N_CAP rows by one stratified sample (seed 0), saved once to
data/<slug>.npz and never rebuilt (e1_prereg.md). Values are kept as in v1 (no re-scaling after sampling)."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
COMMONS = os.path.expanduser("~/Documents/research-commons/datasets")
N_CAP = 100_000
DATASETS = ("electrical-grid-stability", "mammography", "gas-drift", "htru2", "magic-telescope", "house-16h",
            "fried", "internet-firewall", "jannis", "run-or-walk", "vehicle-sensit", "miniboone", "creditcard",
            "hls4ml-jets")


def load(slug):
    p = os.path.join(DATA, f"{slug}.npz")
    if not os.path.exists(p):
        os.makedirs(DATA, exist_ok=True)
        df = pd.read_csv(os.path.join(COMMONS, slug, "standardized", "v1", "data.csv"))
        X, y = df.drop(columns="label").to_numpy(float), df["label"].to_numpy()
        if len(y) > N_CAP:
            i = np.sort(train_test_split(np.arange(len(y)), train_size=N_CAP, stratify=y, random_state=0)[0])
            X, y = X[i], y[i]
        np.savez(p, X=X, y=y)
    z = np.load(p)
    return z["X"], z["y"]
