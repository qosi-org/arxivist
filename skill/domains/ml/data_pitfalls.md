# Domain: ML — Data Pitfalls (Stage 1 + Stage 4 Enrichment)

---

## Critical data-level reproducibility traps in ML papers

**Target leakage** — features computed using information from the target variable (or future
time steps) that wouldn't be available at prediction time. Common in tabular ML papers.
Flag any feature that is temporally derived or computed from the full dataset.

**Preprocessing fitted on full dataset** — normalisation, PCA, or imputation fitted on train+test
before splitting produces optimistic results. Verify the paper specifies fit-on-train-only.

**Class imbalance handling** — SMOTE, oversampling, and class weighting change results
significantly. If the paper mentions imbalanced data without specifying the handling strategy,
flag as ambiguous.

**Benchmark dataset versions** — UCI, OpenML, and Kaggle datasets are updated over time.
Different download dates can produce different data. Always record the dataset version or
hash in `data/README_data.md`.

**Feature selection done inside CV** — feature selection must be inside the CV loop, not
outside. If the paper is unclear about this, flag it as a potential source of optimism.

**Missing value imputation strategy** — mean, median, mode, or model-based imputation
produce different results. Almost never specified. Add to `implementation_assumptions`.
