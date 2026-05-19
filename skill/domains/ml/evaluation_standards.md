# Domain: ML — Evaluation Standards (Stage 6 Enrichment)

Load alongside `agents/06_results_comparator.md` when domain is **ML**.

---

## Deviation thresholds for ML metrics

| Metric | Excellent | Good | Moderate | Notes |
|--------|-----------|------|----------|-------|
| Accuracy | ≤ 0.5% | ≤ 1.5% | ≤ 3% | Average over seeds |
| AUC-ROC | ≤ 0.005 | ≤ 0.01 | ≤ 0.03 | Absolute |
| F1 | ≤ 0.5% | ≤ 1.5% | ≤ 3% | |
| RMSE | ≤ 1% | ≤ 3% | ≤ 8% | Relative |
| Log-loss | ≤ 1% | ≤ 3% | ≤ 7% | |
| ELBO | ≤ 2% | ≤ 5% | ≤ 12% | High variance |

**ML results are noisier than DL results.** Always check whether the paper reports mean ± std.
A deviation within one standard deviation of the paper's reported std is Excellent even if it
appears numerically large.

## Root causes specific to ML

1. **Different CV fold assignment** — if random state differs, different samples go into each fold.
   This alone can explain 0.5–2% accuracy differences.
2. **Library version differences** — sklearn changed default hyperparameters across major versions.
   Always pin the exact version.
3. **Hyperparameter search not reproduced** — if paper tuned hyperparameters and doesn't report
   the exact values used, the reported metric is not reproducible without the same search.
4. **Missing feature preprocessing step** — normalisation, imputation, encoding. These are
   frequently underspecified.
