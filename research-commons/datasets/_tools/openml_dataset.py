"""Shared helpers for datasets/<slug>/fetch.py and standardize.py (see datasets/README.md).

raw/            data.csv.gz (original values as returned by the source; target in the last column),
                SHA256SUMS, source.json (source id, version, name, target, url, download date)
standardized/v1 data.csv: rows with any missing value dropped; every feature min-max scaled to [0, 1]
                over the whole table (constant column -> 0); target label-encoded as 0..c-1 in sorted
                order of its string values; header = original feature names + "label".
                Written with %.17g so values round-trip exactly to float64.
"""
import datetime
import hashlib
import json
import os

import numpy as np
import pandas as pd

FMT = "%.17g"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _write_raw(root, df, source):
    raw = os.path.join(root, "raw")
    if os.path.exists(os.path.join(raw, "data.csv.gz")):
        raise FileExistsError(f"{raw}/data.csv.gz exists — raw/ is immutable")
    os.makedirs(raw)
    p = os.path.join(raw, "data.csv.gz")
    df.to_csv(p, index=False, float_format=FMT, compression="gzip")
    source["downloaded"] = datetime.date.today().isoformat()
    json.dump(source, open(os.path.join(raw, "source.json"), "w"), indent=2)
    with open(os.path.join(raw, "SHA256SUMS"), "w") as f:
        f.write(f"{_sha256(p)}  data.csv.gz\n")


def fetch_openml(root, data_id):
    from sklearn.datasets import fetch_openml as fo
    cache = os.environ.get("OPENML_CACHE")          # optional: reuse an existing sklearn download cache
    try:
        d = fo(data_id=data_id, as_frame=True, parser="auto", n_retries=5, data_home=cache)
        df = d.frame.copy()
        tgt = d.target_names[0] if d.target_names else d.target.name
    except ValueError as e:                          # sparse ARFF (e.g. vehicle_sensIT) cannot load as a frame
        if "Sparse" not in str(e):
            raise
        d = fo(data_id=data_id, as_frame=False, parser="auto", n_retries=5, data_home=cache)
        tgt = d.target_names[0] if d.target_names else "class"
        df = pd.DataFrame(d.data.toarray(), columns=d.feature_names)
        df[tgt] = d.target
    df = df[[c for c in df.columns if c != tgt] + [tgt]]
    _write_raw(root, df, {"source": "OpenML", "data_id": data_id, "name": d.details.get("name"),
                          "version": d.details.get("version"), "target": tgt,
                          "url": f"https://www.openml.org/d/{data_id}",
                          "licence": d.details.get("licence")})


def fetch_sklearn_covtype(root):
    from sklearn.datasets import fetch_covtype
    d = fetch_covtype(as_frame=True)
    df = d.frame.copy()
    _write_raw(root, df, {"source": "sklearn.datasets.fetch_covtype (UCI Covertype)", "target": "Cover_Type",
                          "url": "https://archive.ics.uci.edu/dataset/31/covertype"})


def check_raw(root):
    raw = os.path.join(root, "raw"); want = open(os.path.join(raw, "SHA256SUMS")).read().split()[0]
    if _sha256(os.path.join(raw, "data.csv.gz")) != want:
        raise ValueError(f"checksum mismatch in {raw}")


def standardize_v1(root):
    check_raw(root)
    out = os.path.join(root, "standardized", "v1")
    if os.path.exists(os.path.join(out, "data.csv")):
        raise FileExistsError(f"{out}/data.csv exists — v1 is immutable")
    df = pd.read_csv(os.path.join(root, "raw", "data.csv.gz"))
    tgt = df.columns[-1]
    feats = df.columns[:-1]
    nonnum = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])]
    if nonnum:
        raise ValueError(f"non-numeric features {nonnum}: v1 of this tool handles numeric tables only")
    n0 = len(df); df = df.dropna()
    X = df[feats].to_numpy(float); lo, hi = X.min(0), X.max(0)
    Xs = (X - lo) / np.where(hi > lo, hi - lo, 1.0)
    y = np.unique(df[tgt].astype(str).to_numpy(), return_inverse=True)[1]
    os.makedirs(out)
    std = pd.DataFrame(Xs, columns=feats); std["label"] = y
    std.to_csv(os.path.join(out, "data.csv"), index=False, float_format=FMT)
    classes = sorted(df[tgt].astype(str).unique())
    with open(os.path.join(out, "normalize_config.md"), "w") as f:
        f.write(f"# standardized v1\n\n- Rows: {n0} raw, {n0 - len(df)} dropped for missing values, {len(df)} kept.\n"
                f"- Features: {len(feats)}, all numeric; min-max to [0, 1] over the full table "
                f"(constant columns -> 0; {int((hi == lo).sum())} constant).\n"
                f"- Target `{tgt}` label-encoded in sorted string order: "
                + ", ".join(f"{c} -> {i}" for i, c in enumerate(classes)) + ".\n"
                f"- Written with `{FMT}` (exact float64 round trip).\n"
                f"- Why min-max: the eval_harness (3-NN / SVM-RBF) and the rough-set criteria in the commons use "
                f"distances on [0, 1]-scaled attributes.\n")
    return std, df, feats, tgt, classes


def write_meta(root, title, std, df, feats, tgt, classes, source_note, issues):
    counts = np.bincount(std["label"].to_numpy())
    nunq = df[feats].nunique()
    with open(os.path.join(root, "meta.md"), "w") as f:
        f.write(f"# {title}\n\n| | |\n|---|---|\n| Instances (v1) | {len(std)} |\n| Features | {len(feats)} "
                f"(all numeric) |\n| Classes | {len(classes)} ({', '.join(classes)}) |\n"
                f"| Class counts (v1 label order) | {', '.join(map(str, counts))} |\n"
                f"| Min / median distinct values per feature | {int(nunq.min())} / {int(nunq.median())} |\n"
                f"| Real-valued rule (every feature >= 100 distinct values) | "
                f"{'yes' if nunq.min() >= 100 else 'no'} |\n| Target column (raw) | `{tgt}` |\n"
                f"| Source | {source_note} |\n")
    with open(os.path.join(root, "known_issues.md"), "w") as f:
        f.write(f"# Known issues — {title}\n\n" + "\n".join(f"- {s}" for s in issues) + "\n")

V2_Q = (0.005, 0.995)


def standardize_v2(root):
    """v2 = v1 rows/labels, each feature clipped to its [0.5%, 99.5%] quantiles over the full table, then min-max.
    Added 2026-09-29 (KDBound-RS A3): v1 min-max is squashed by extreme outliers (eeg-eye-state, numerai28-6)."""
    v1 = pd.read_csv(os.path.join(root, "standardized", "v1", "data.csv"))
    out = os.path.join(root, "standardized", "v2")
    if os.path.exists(os.path.join(out, "data.csv")):
        raise FileExistsError(f"{out}/data.csv exists — v2 is immutable")
    feats = v1.columns[:-1]; X = v1[feats].to_numpy(float)
    ql, qh = np.quantile(X, V2_Q[0], axis=0), np.quantile(X, V2_Q[1], axis=0)
    Xc = np.clip(X, ql, qh)
    Xs = (Xc - ql) / np.where(qh > ql, qh - ql, 1.0)
    clipped = int((X != Xc).any(1).sum()); flat = int((qh == ql).sum())
    os.makedirs(out)
    std = pd.DataFrame(Xs, columns=feats); std["label"] = v1["label"]
    std.to_csv(os.path.join(out, "data.csv"), index=False, float_format=FMT)
    with open(os.path.join(out, "normalize_config.md"), "w") as f:
        f.write(f"# standardized v2\n\n- Built from v1 (same rows, order and labels).\n"
                f"- Each feature clipped to its [{V2_Q[0]:.1%}, {V2_Q[1]:.1%}] quantiles over the full table, then "
                f"min-max to [0, 1]; unsupervised, label-free (same leakage status as v1).\n"
                f"- Rows with at least one clipped value: {clipped} of {len(v1)}. Features with equal quantiles "
                f"(-> 0): {flat}.\n- Written with `{FMT}`.\n"
                f"- Why: v1 min-max is dominated by extreme outliers on some tables, which squashes distances.\n")
    return std
