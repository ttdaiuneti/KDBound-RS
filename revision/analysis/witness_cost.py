"""Simulated round-2 review, Soundness (a): cost of the (unbounded) witness query versus one evaluation.
Source: code/steps/steps_run.jsonl (KD-PA+B3 with step records). For each step after the first with at least one
evaluation: t_witness against (step time - t_witness) / evaluated. The second quantity also contains the bound
computations, so it overestimates one evaluation and the share printed is a lower bound."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "..", "code", "steps", "steps_run.jsonl"))]
tot = worse = 0
for r in rows:
    for s in r.get("steps", [])[1:]:
        tw, ev = s.get("t_witness"), s["evaluated"]
        if tw is None or ev == 0:
            continue
        tot += 1; worse += tw > (s["time"] - tw) / ev
print(f"steps {tot}; witness query longer than the average evaluation (upper estimate): {worse} ({worse / tot:.1%})")
