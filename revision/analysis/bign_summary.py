"""R1.14: numbers for Appendix A.4 (full-size tables) from code/bign/bign_run.jsonl, with the idle n = 1e5 times of the
subsampling run (e1/scale_n.jsonl) and the fold-1 e1 times of FNC on creditcard (concurrent; for orientation only)."""
import json, math, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
B = {(r["dataset"], r["criterion"], r["method"]): r for r in map(json.loads, open(os.path.join(HERE, "..", "code", "bign",
                                                                                                  "bign_run.jsonl")))}
S = {(r["dataset"], r["criterion"], r["method"]): r for r in map(json.loads, open(os.path.join(ROOT, "e1", "scale_n.jsonl")))
     if r.get("n") == 100_000 and r["status"] == "ok"}
for k, b in sorted(B.items()):
    line = f"{k}: n_train {b.get('n_train')} status {b['status']} time {b.get('time', 0):.1f} s evals {b.get('evals')} " \
           f"|R| {len(b.get('R') or [])} peak {b['peak_gb']} GB"
    s = S.get(k)
    if s and b["status"] == "ok":
        line += f" | idle 1e5: {s['time']:.1f} s, ratio {b['time'] / s['time']:.1f}, exponent " \
                f"{math.log(b['time'] / s['time']) / math.log(b['n_train'] / s['n_train']):.2f}"
    print(line)
for d in ("creditcard", "miniboone", "hls4ml-jets"):
    for c in ("NRS", "FRS"):
        a, b = B[(d, c, "KD-PA")], B[(d, c, "KD-PA+B3")]
        print(f"{d} {c}: KD-PA / KD-PA+B3 = {a['time'] / b['time']:.2f}; same R {a['R'] == b['R']}")
ok = [r for r in B.values() if r["status"] == "ok" and r["method"].startswith("KD")]
print("KD units:", len(ok), "max time", round(max(r["time"] for r in ok), 1), "max peak GB", max(r["peak_gb"] for r in ok))
f = [B.get(("creditcard", "NRS", m)) for m in ("FNC", "FNC+B3")]
if all(f):
    R0 = B[("creditcard", "NRS", "KD-PA+B3")]["R"]
    for m, r in zip(("FNC", "FNC+B3"), f):
        print(m, r["status"], round(r.get("time", 0), 1), "same R as KD-PA+B3:", r.get("R") == R0, "peak", r["peak_gb"])
