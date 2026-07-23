# Verification Log

**Paper ID**: arxiv_2607_06340
**Comparison run timestamp**: 2026-07-13T00:00:00Z

---

## Inputs Used

- `docs/sir.json` — version 1 (overall confidence 0.82)
- `docs/architecture_plan.json` — version 1
- User results (pasted notebook output), Experiment 6.1 + 6.8_low_nu + 6.8_high_nu:
  ```
  Classes: ['Heston', 'OU', 'rB0.1', 'rB0.3']
  Train shape: (12000, 31), Test shape: (3200, 31)
  Train accuracy: 0.9423
  Test accuracy:  0.7856
          Heston      OU   rB0.1   rB0.3
  Heston  88.750  10.625   0.000   0.625
  OU      29.875  53.750   0.625  15.750
  rB0.1    0.000   0.125  92.750   7.125
  rB0.3    0.500  12.750   7.750  79.000
  Top built-in feature: sig_27 (0.191)
  Top permutation feature: sig_21 (0.158)
  Low nu:  Heston misclassified as OU: 1.6%
  High nu: Heston misclassified as OU: 6.6%
  ```

---

## Metric Matching

- Paper metrics available in `sir.json -> evaluation_protocol.reported_results`
  plus prose-described Figure values: ~23 total across all 9 experiments.
- User-reported metrics this round: 8 (overall test/train accuracy, 4
  per-class accuracies for Exp 6.1, 2 misclassification rates for the 6.8
  variants) plus 2 qualitative feature-identity checks (not scored
  numerically).
- Matched: 8/8 of what the user reported found a paper counterpart.
  8/23 of the paper's total tracked metrics addressed this round.

---

## Computation Trace

```
test_accuracy_overall:     paper=0.9863  user=0.7856  dev=0.2007  pct=20.35%  -> Significant
train_accuracy_overall:    paper=1.0000  user=0.9423  dev=0.0577  pct=5.77%   -> Moderate
Heston_per_class_acc:      paper=96.60   user=88.75   dev=7.85    pct=8.13%   -> Moderate
OU_per_class_acc:          paper=100.00  user=53.75   dev=46.25   pct=46.25%  -> Critical
rB0.1_per_class_acc:       paper=98.90   user=92.75   dev=6.15    pct=6.22%   -> Moderate
rB0.3_per_class_acc:       paper=99.00   user=79.00   dev=20.00   pct=20.20%  -> Significant
low_nu_misclass:           paper=69.80   user=1.60    dev=68.20   pct=97.71%  -> Critical (known issue)
high_nu_misclass:          paper=9.10    user=6.60    dev=2.50    pct=27.47%  -> Significant

per_metric_score(pct) = 1 - min(pct/50, 1.0):
  0.5930, 0.8846, 0.8374, 0.0750, 0.8756, 0.5960, 0.0000, 0.4506

base_score = mean(above) = 4.3122 / 8 = 0.5390

sir_confidence_penalty = (1 - 0.82) * 0.15 = 0.0270
unmatched_penalty = (15 / 23) * 0.2 = 0.1304
  (15 = 23 total tracked paper metrics - 8 matched this round)

reproducibility_score = max(0, 0.5390 - 0.0270 - 0.1304) = 0.3816 ~= 0.38

score_confidence = "low"
  (rationale: 8/23 paper metrics matched; user ran at 83x-smaller scale than
  the paper's protocol; one matched metric -- low-nu -- is a known,
  pre-existing open issue rather than fresh evidence of a new problem.)
```

---

## SIR / Architecture Plan Versions Used

- SIR version: 1 (unchanged)
- Architecture plan version: 1 (unchanged)
- Two code fixes were applied to `src/sig_vol_id/data/experiment_builder.py`
  and `src/sig_vol_id/models/xgb_classifier.py` during Stage 4 testing,
  **before** this comparison round -- not modified as part of this specific
  comparison. See `hallucination_report.md` for the audit trail of both fixes.

---

## Manual Review Flag

**Requires manual review: Yes.**

Reason: the OU per-class accuracy collapse (100% -> 53.75%, Critical
severity) is a new finding not previously flagged or explained, and
disproportionate to the milder degradation in the other 3 classes at the
same demo scale -- worth confirming via a larger-scale rerun before
concluding whether this is a scale artifact (most likely) or a deeper
issue. The low-nu deviation is separately flagged but is a known,
pre-existing open item, not new information from this run.

---

## Files Produced This Run

- `benchmark_comparison.md`
- `reproducibility_score.json`
- `hallucination_report.md`
- `verification_log.md` (this file)
- `docs/pipeline_state.json` and `docs/metadata.json` updated:
  `has_comparison_report: true`, stage 6 added to `stages_completed`
