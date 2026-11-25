# Second-machine check (Appendix A.6, Table 17)

A fixed subset of 34 timing units (fold 1) was run with `run_check.py` on the Apple M4 Pro of the main runs
(`reference/m4_reference.*`) and on an x86 laptop (Intel Core i7-1165G7, 16 GB RAM, Windows 11, Python 3.12.8;
`results/`). Part A: NRS at delta = 0.10 (KD-PA, KD-PA+B3 on six tables, FNC, FNC+B3 on four, HASH on two). Part B:
FRS at sigma = 0.25 (KD-PA, KD-PA+B3 on five tables, FA-FPR on two). One unit at a time, one BLAS thread, 1800 s limit.

The bundle that was copied to the second machine also held `code/e1/engines_e1.py`, `code/e1/frs.py`,
`code/g1/methods.py` and `data/<slug>.npz`. They are not duplicated here: they are `e1/engines_e1.py`, `e1/frs.py`,
`g1/methods.py` and the cache `e1/data/<slug>.npz` written by `e1/data_e1.py`, with the SHA-256 listed in
`reference/SHA256SUMS.txt`. To re-run, copy them into `code/` and `data/` and run `python run_check.py check`, then
`python run_check.py run all` and `python run_check.py summary` (Windows: the three `.bat` files).

`revision/analysis/x86_summary.py` computes the numbers and Table 17 from `results/x86_results.jsonl` and
`reference/m4_reference.jsonl`; `revision/analysis/witness_ties.py` counts the tied nearest enemies that explain the
one unit whose number of evaluations differed. The peak memory in `results/` was not measured on Windows (it reads
0.0) and is not used.
