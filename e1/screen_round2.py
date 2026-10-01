import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.datasets import fetch_openml
C=[(42395,"santander")]
for did,nm in C:
    try:
        d=fetch_openml(data_id=did,as_frame="auto",parser="auto",n_retries=5,data_home="oml_cache")
        X=d.data
        X=pd.DataFrame(X.toarray()) if sp.issparse(X) else pd.DataFrame(X)
        cat=sum(not pd.api.types.is_numeric_dtype(X[c]) for c in X.columns); nun=X.nunique(); mind=int(nun.min()); nan=int(X.isna().sum().sum())
        ok=cat==0 and mind>=100 and len(X)>=10000 and nan==0
        print(f"{did}\t{nm}\tN={len(X)}\td={X.shape[1]}\tnonnum={cat}\tmin_distinct={mind}\tnan={nan}\tlow={list(nun[nun<100].index)[:4]}\t{'PASS' if ok else 'fail'}",flush=True)
    except Exception as e: print(f"{did}\t{nm}\tDOWNLOAD-FAIL\t{str(e)[:120]}",flush=True)
