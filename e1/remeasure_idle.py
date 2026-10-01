"""Idle re-measure (e1_prereg.md section 11.2): one unit at a time, fold 1, same 900 s / 32 GB caps. Sensitivity
analysis only; e1_raw.jsonl is not touched. Output: remeasure_idle.jsonl (append; done units skipped)."""
import json
import os
import sys

os.environ["E1_NO_REDUCT"] = "1"                                   # set before run_e1 builds ENV
from run_e1 import run_capped, HERE, LIMIT

OUT = os.path.join(HERE, "remeasure_idle.jsonl")
UNITS = [(s, "NRS", "0.10", m) for s in ("gas-drift", "creditcard", "house-16h", "magic-telescope", "jannis", "miniboone")
         for m in ("KD-PA+B3", "KD-PA", "FNC", "FNC+B3")] + \
        [(s, "NRS", "0.10", "HASH") for s in ("house-16h", "magic-telescope")] + \
        [("vehicle-sensit", "NRS", p, m) for p in ("0.10", "0.15", "0.20") for m in ("KD-PA+B3", "KD-PA")]

done = {(r["dataset"], r["criterion"], r["param"], r["method"]) for r in map(json.loads, open(OUT))} \
    if os.path.exists(OUT) else set()
for slug, crit, param, m in UNITS:
    if (slug, crit, param, m) in done:
        continue
    out, err, note, wall, peak = run_capped([sys.executable, os.path.join(HERE, "run_e1.py"), "one", slug, crit, param,
                                             "1", m])
    lines = [ln for ln in out.splitlines() if ln.startswith("{")]
    r = json.loads(lines[-1]) if lines and not note else dict(dataset=slug, criterion=crit, param=param, fold=1,
                                                              method=m, status="memout" if note and note != "hard kill"
                                                              else "timeout", note=note, time=wall)
    r.update(peak_footprint_gb=round(peak, 2), run="idle")
    with open(OUT, "a") as f:
        f.write(json.dumps(r) + "\n")
    print(f"{slug} {param} {m:9s} {r['status']:8s} t={r.get('time', float('nan')):.1f}", flush=True)
