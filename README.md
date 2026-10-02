# KDBound-RS

Code, protocol and raw results for the manuscript

> T.D. Tran, *Nearest-enemy bounds for exact and fast greedy attribute reduction in neighborhood and fuzzy rough
> sets* (submitted to Information Sciences).

The method selects attributes with the standard greedy forward procedure for neighborhood rough sets (NRS) and fuzzy
rough sets with the minimum t-norm (FRS). A per-class k-d tree answers nearest-enemy queries, and a witness bound skips
candidate attributes that cannot be selected. The output is exactly the ordered reduct of the standard greedy procedure.

## Layout

| Path | Content |
|---|---|
| `e1/engines_e1.py` | NRS engines (KD-PA, FNC/HNAR, HASH, CABIN, BUCKET, BF-PA), each with and without the bound (`+B3`), and the greedy driver |
| `e1/frs.py` | FRS engines (KD-PA, FA-FPR, brute force), each with and without the bound |
| `e1/fara.py` | FARA (heuristic accelerator) on the same k-d tree engine |
| `g1/methods.py` | shared engine code used by `e1/engines_e1.py` |
| `e1/run_e1.py` | experiment runner (resumable, append-only; time and memory limits) |
| `e1/eval_e1.py` | classification accuracy of the reducts (3-NN, SVM-RBF; `research-commons/eval_harness/protocol.py`) |
| `e1/verify_*.py`, `e1/verify_*.log` | exactness checks: all exact methods return identical reducts |
| `e1/PROTOCOL_EN.md` | English summary of the protocol and its dated amendments (sections 1-13) |
| `e1/e1_prereg.md` | the protocol as recorded during the study, with every amendment (sections 8-13; in Vietnamese) |
| `e1/e1_raw.jsonl` | raw results, one line per (dataset, criterion, parameter, fold, method) |
| `e1/e1_superseded.jsonl` | rows replaced by a re-run (kept in the raw file, excluded from all analyses) |
| `e1/e1_acc.jsonl` | per-fold classification accuracy |
| `e1/results/reducts/` | the reduct of every run, with per-step records (candidates, evaluations, gains) |
| `e1/summarize_e1.py`, `stats_e1.py`, `rq1_stats.py`, `make_*.py` (incl. `make_fig_mech.py`, the illustrative step of Fig. 1) | every number, table and figure of the manuscript is computed by these scripts from the raw files |
| `e1/remeasure_idle.*` | re-measurement of a fixed subset of runs on an idle machine |
| `e1/scale_n.py`, `e1/scale_n.jsonl` | supplementary run: reduction time against the number of objects (protocol section 13; Appendix A.3) |
| `e1/scale_m.py`, `e1/scale_m.jsonl` | supplementary run: reduction time against the number of attributes on gas-drift (protocol section 14; Fig. 6) |
| `research-commons/` | data preparation scripts (OpenML download and min-max normalization) and the evaluation harness |

## Reproducing

Requirements: Python 3.9, see `requirements.txt`.

1. Copy (or symlink) `research-commons/` to `~/Documents/research-commons/`; the scripts read the data and the
   harness from that location.
2. For each dataset, run `python fetch.py` and then `python standardize.py` in
   `~/Documents/research-commons/datasets/<dataset>/` (data are downloaded from OpenML; identifiers in `meta.md`).
3. `cd e1 && python run_e1.py` runs the timing grid (datasets already listed in `e1_locked.txt` are skipped; remove
   that file and `e1_raw.jsonl` to start from scratch). `python eval_e1.py` computes the accuracies.
4. `python summarize_e1.py && python stats_e1.py && python rq1_stats.py` reproduce the statistics; the `make_*.py`
   scripts write the LaTeX tables and figures to `../paper/manuscript/` (create that folder first).

The `code` field of each raw row is a hash of the source files at the time of the run. Times depend on the machine;
the reported runs used an Apple M4 Pro with 48 GB of memory.

## License

The code is released under the MIT License (see `LICENSE`). The datasets are not redistributed here; they are
downloaded from OpenML and remain under the licenses stated by their providers there.
