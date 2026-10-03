# E1 experimental protocol — English summary

This file summarizes, in English, the protocol `e1_prereg.md` (written in Vietnamese) and its dated amendments. The
Vietnamese file is the record that was kept during the study; section numbers below match it. Every amendment was
written before the units it affects were run or re-run. Rows replaced by a re-run are kept in `e1_raw.jsonl` and
listed in `e1_superseded.jsonl`; all analyses exclude them.

## Locked before the first dataset was run (2026-09-29)

**§1 Datasets.**
- Selection rules: every condition attribute is numeric with at least 100 distinct values, and N ≥ 10⁴.
- Candidates came from a list fixed before download (OpenML via a shared repository) and were screened in two rounds
  (`screen_datasets.log`).
- 14 tables are usable. eeg-eye-state and numerai28.6 passed the rules but were dropped because greedy NRS selected
  fewer than two attributes on them (`v2_degeneracy_check.log`).
- Excluded by the rules: letter and covertype (integer or binary values), and the tables listed in `screen_datasets.log`.
  higgs and santander failed the OpenML checksum; microsoft-ltr was abandoned before its data were inspected.
- Properties to report: fried is synthetic, internet-firewall holds counts and port numbers, and run-or-walk is a time
  series (random folds give optimistic accuracies).

**§2 Grid.**
- NRS: Euclidean distance, δ ∈ {0.05, 0.10, 0.15, 0.20}, τ = 0.
- FRS: Dubois–Prade model, Kleene–Dienes implicator, minimum t-norm, R_a = max(0, 1 − |Δ_a|/σ_a);
  σ_a = 0.25 (setting of Qian et al. 2015) or σ_a = std of a on the training part (Jensen & Shen 2009); τ = 1e-9.
- Objects capped at 10⁵ by one stratified sample (seed 0), drawn before the folds.
- NRS engines {BF, FNC/HNAR, CABIN, HASH, BUCKET, KD} × {without, with the bound}; FRS engines {BF, FA-FPR, KD} ×
  {without, with}; FARA as the heuristic accelerator (t = 1, k increasing per round, stop when γ(A) ≥ γ(AT); fixed
  before the runs).
- Folds: StratifiedKFold(5, shuffle=True, random_state=0). Limit: 900 s per unit; FNC raises MemOut above 3e8 pairs.

**§3–§4 Recording.** Per fold: time, |R|, 3-NN and SVM-RBF accuracy (evaluation harness defaults). Each run stores its
ordered reduct and per-step records (candidates, evaluations, gains, bound of the selected attribute, time).

**§5 Planned analysis.**
- Friedman + Nemenyi on times; Wilcoxon for each pair E vs E+B3; time ratio E / (E+B3) per engine.
- A scaling curve in n was planned for n ∈ {10⁴, 3·10⁴, 10⁵, 3·10⁵}. With the cap at 10⁵ the largest size was not
  available; the scaling run was done after the main runs with the grid of §13.
- Limits to report: extreme minority classes (creditcard, where FNC wins) and tables with |R| ≈ m (no pruning).

**§7 Run procedure.**
- Per-criterion degeneracy rule: a dataset is kept for a criterion if greedy reduction on the whole capped table
  selects at least two attributes (NRS δ = 0.10, FRS σ = 0.25). NRS keeps 12 datasets (electrical-grid-stability and
  fried excluded).
- One unit = (dataset, criterion, parameter, fold, method), run on the training part of the fold in a fresh process
  with one BLAS thread; four units concurrently; at most one FNC unit at a time.
- Skip rule: a timeout or memory-out on fold 1 skips folds 2–5 of that configuration.
- Exactness check: all exact methods that finish must return the same ordered reduct; a mismatch is flagged, never
  silently re-run. Checked before the main runs on 2000-object samples (NRS 12/12, FRS 10/10).

## Amendments

| § | Date | Change | Reason |
|---|---|---|---|
| 8 | 2026-09-29 16:36 | Lean grid: engine ablation (HASH+B3, CABIN±B3, BUCKET±B3, BF-PA±B3) at δ = 0.10 only; brute-force FRS dropped (FA-FPR is brute force + positive approximation). Rows already run outside the lean grid are kept but not analysed. | Full grid estimated at 2–3 days, mostly units exceeding the limit. No NRS dataset completed yet. |
| 9 | 2026-09-29 19:15 | FNC+B3 witness step: sort O(P log P) replaced by a linear minimum O(P); exactness re-checked; FNC+B3 re-run on completed datasets. | The slow step made FNC+B3 2–4× slower than FNC on house-16h; competitors must be equally optimized. |
| 10 | 2026-09-30 | Memory rule: pair-enumerating +B3 units (like FNC) run one at a time; a unit is stopped and recorded as memout above 32 GB physical footprint (including compressed memory) or at critical system memory pressure. A first version monitored RSS and was replaced after a few minutes. run-or-walk re-measured entirely under the rule; miniboone re-measured from the start (an unrelated job had run on the machine). | Two machine crashes traced to memory exhaustion by concurrent pair-enumerating +B3 units. |
| 11 | 2026-10-01 | Degenerate configurations (empty exact reduct in some fold: hls4ml-jets NRS δ ∈ {0.15, 0.20}) reported but excluded from speed-up statistics and accuracy. Idle re-measurement of a fixed subset (sensitivity only, does not replace values). Statistics: Friedman + Nemenyi with timeouts ranked as 900 s; one-sided Wilcoxon per engine pair. | Decided after the runs, before the analysis. |
| 12 | 2026-10-01 | Witness step of HASH+B3, CABIN+B3, BUCKET+B3, BF-PA+B3: same O(P) fix as §9; exactness re-checked (evaluation counts unchanged); these four methods re-run on all datasets under the §10 rule. | Same slow step as §9, found while writing the paper. |
| 13 | 2026-10-01 | Supplementary scaling run: hls4ml-jets and miniboone, n ∈ {12 500, 25 000, 50 000, 100 000}, fold 1, NRS δ = 0.10 and FRS σ = 0.25, one unit at a time, same limits. | Scaling curve of §5 with sizes available under the 10⁵ cap. |
| 14 | 2026-10-02 | Supplementary run on the number of attributes: gas-drift, fold 1, m ∈ {16, 32, 64} (first m attributes of three random permutations, nested) and m = 128; NRS δ = 0.10 and FRS σ = 0.25; KD-PA and KD-PA+B3; same limits. Probes on dilbert (m = 2000) and Santander (m = 200) were used only to decide this design and are not reported. | Isolate the effect of m from other properties of the tables. |
| 15 | 2026-10-03 | (a) The probes of item 14 (dilbert, SantanderCustomerSatisfaction) are reported in Appendix A.1; scripts and logs in `e1/probes_hd/`. (b) The Wilcoxon test of E against E+B3 uses one value per dataset (mean log ratio over its configurations) instead of one per configuration. (c) The correlations with m, the partial correlations, the sign test and the accuracy tests are exploratory (chosen after the runs); p-values are unadjusted. | (a) The probes show a smaller gain of the bound at large m; leaving them out would select results. (b) Configurations of one dataset are not independent. |

## Files

`e1_prereg.md` (original record, Vietnamese), `e1_report.md` (run report), `e1_raw.jsonl`, `e1_superseded.jsonl`,
`e1_acc.jsonl`, `results/reducts/`, `remeasure_idle.jsonl`, `scale_n.jsonl`, and the scripts that compute every number
of the manuscript (`summarize_e1.py`, `stats_e1.py`, `rq1_stats.py`, `make_*.py`).
