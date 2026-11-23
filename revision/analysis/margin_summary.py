"""R2.2 / R1.9: margin certificate summary (code/numerics/margin_certificate.jsonl; duplicate units from the two
processes are identical computations, the first record is kept) and boundary test summary (boundary_test.log)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
rs = {}
for l in open(os.path.join(HERE, "..", "code", "numerics", "margin_certificate.jsonl")):
    r = json.loads(l); rs.setdefault((r["dataset"], r["param"], r["fold"]), r)
v = [r for r in rs.values() if "min_gap" in r]
U = 2.0 ** -53
mb = max(len(r["R"]) + 1 for r in v)
print(f"units: {len(v)} (errors {len(rs) - len(v)}), complete {sum(r['complete'] for r in v)}, datasets {len({r['dataset'] for r in v})}, "
      f"configurations {len({(r['dataset'], r['param']) for r in v})}")
print(f"checks (object, subset) pairs: {sum(r['checks'] for r in v):.3e}")
g = min(v, key=lambda r: r["min_gap"])
print(f"smallest relative gap |ne2 - delta^2| / delta^2: {g['min_gap']:.2e} ({g['dataset']}, delta {g['param']}, fold {g['fold']}, {g['argmin']})")
print(f"smallest gap on one attribute: {min(r['min_gap_1attr'] for r in v):.2e}  (FNC window needs > {2e-9 / 0.05 + 1e-13:.1e})")
print(f"pairs with gap < 1e-9: {sum(r['n_gap_lt_1e9'] for r in v)}, < 1e-12: {sum(r['n_gap_lt_1e12'] for r in v)}")
print(f"rounding bound (|B|+3) u for |B| <= {mb}: {(mb + 3) * U:.1e}")
per = {}
for r in v:
    per[r["dataset"]] = min(per.get(r["dataset"], 9), r["min_gap"])
print("per dataset:", {k: f"{x:.1e}" for k, x in sorted(per.items(), key=lambda t: t[1])})
