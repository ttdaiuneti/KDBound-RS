"""E1 dataset screening (rule fixed in e1_prereg_DRAFT.md: all features numeric, each >=100 distinct values, N>=1e4).
Candidate list fixed 2026-09-29 BEFORE downloading (OpenML metadata filter; synthetic BNG/RandomRBF/Hyperplane and
near-duplicate families FOREX/timing-attack/padding-attack excluded). Output: screen_datasets.log"""
import pandas as pd
from sklearn.datasets import fetch_openml

CANDS = [(46298, "electrical-grid-stability"), (32, "pendigits"), (310, "mammography"), (734, "ailerons"),
         (1040, "sylva_prior"), (722, "pol"), (846, "elevators"), (45558, "htru2"), (821, "house_16H"),
         (43979, "california"), (42477, "default-credit-card"), (43945, "electricity"), (901, "fried"),
         (41166, "volkert"), (357, "vehicle_sensIT"), (45548, "otto"), (1503, "spoken-arabic-digit"),
         (351, "codrna"), (42769, "higgs"), (45549, "atlas-higgs"), (42468, "hls4ml-jets")]
for did, nm in CANDS:
    try:
        d = fetch_openml(data_id=did, as_frame=True, parser="auto", n_retries=5,
                         data_home="oml_cache")
    except Exception as e:  # noqa: BLE001
        print(f"{did}\t{nm}\tDOWNLOAD-FAIL\t{e}", flush=True); continue
    df = d.data
    cat = sum(not pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
    nun = df.nunique(); mind = int(nun.min())
    ok = cat == 0 and mind >= 100 and len(df) >= 10000 and df.isna().sum().sum() == 0
    print(f"{did}\t{nm}\tN={len(df)}\td={df.shape[1]}\tnonnum={cat}\tmin_distinct={mind}"
          f"\tlow={list(nun[nun < 100].index)[:4]}\t{'PASS' if ok else 'fail'}", flush=True)
